"""S3-based shard coordinator for dynamic multi-machine work distribution.

Each machine independently claims shards from S3, enabling:
- Dynamic add/remove of machines at any time
- Auto-recovery when machines die (stale heartbeat -> reclaimable)
- No fixed shard assignment required upfront

Architecture:
    S3 (bucket/prefix/)
      manifest.json              <- shard definitions (article ranges)
      claims/000.json            <- "in_progress" by machine-1
      claims/001.json            <- "completed" by machine-2
      claims/002.json            <- unclaimed (no file = available)
      output/shard_000/          <- screenshots + checkpoint
"""

import json
import logging
import os
import socket
import time

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3ShardCoordinator:
    """Coordinates shard distribution across machines via S3 claim files.

    Each machine runs a claim loop:
        while True:
            shard = coordinator.claim_next()
            if shard is None: break
            pipeline.run(start=shard.start, end=shard.end)
            coordinator.mark_done(shard.id)
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "kiwix",
        machine_id: str | None = None,
        heartbeat_interval: int = 60,
        stale_timeout: int = 1800,
    ):
        """Initialize coordinator.

        Args:
            bucket: S3 bucket name (without s3:// prefix).
            prefix: Key prefix within bucket for manifest/claims/output.
            machine_id: Unique ID for this machine. Auto-generated if None.
            heartbeat_interval: Seconds between heartbeat updates.
            stale_timeout: Seconds before an in-progress shard is considered stale
                          and reclaimable (default: 30 min).
        """
        self.s3 = boto3.client("s3")
        self.bucket = bucket
        self.prefix = prefix
        self.machine_id = machine_id or f"{socket.gethostname()}-{os.getpid()}"
        self.hostname = socket.gethostname()
        self.heartbeat_interval = heartbeat_interval
        self.stale_timeout = stale_timeout
        self._manifest: dict | None = None
        self._claimed_at: dict[int, float] = {}  # shard_id → claim timestamp

    def load_manifest(self) -> dict:
        """Load shard manifest from S3.

        Returns:
            Manifest dict with keys: total, num_shards, shards.
        """
        key = f"{self.prefix}/manifest.json"
        logger.info("Loading manifest from s3://%s/%s", self.bucket, key)
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        self._manifest = json.loads(obj["Body"].read())
        logger.info(
            "Loaded manifest: %d shards, %d total articles",
            self._manifest["num_shards"],
            self._manifest["total"],
        )
        return self._manifest

    def claim_next(self) -> dict | None:
        """Claim the next available shard.

        Iterates through all shards and claims the first one that is either
        unclaimed or stale (heartbeat older than stale_timeout).

        Uses S3 conditional writes (IfNoneMatch / IfMatch) to prevent race
        conditions where two machines claim the same shard simultaneously.

        Returns:
            Shard dict with keys {id, start, end, count}, or None if all done.
        """
        if not self._manifest:
            self.load_manifest()

        for shard in self._manifest["shards"]:
            key = f"{self.prefix}/claims/{shard['id']:03d}.json"

            # Check if already claimed
            etag = None  # Track ETag for conditional reclaim
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=key)
                claim = json.loads(obj["Body"].read())
                if claim["status"] == "completed":
                    continue
                if claim["status"] == "partial":
                    # Partial shard — only reclaim on the SAME host that
                    # originally ran it (that host still has the local data
                    # on NVMe, so it can resume efficiently).
                    claim_host = claim.get("hostname", "")
                    if claim_host and claim_host != self.hostname:
                        continue  # Leave it for the original host
                    etag = obj.get("ETag")
                    logger.info(
                        "Reclaiming partial shard %d (completed=%d, host=%s)",
                        shard["id"],
                        claim.get("completed", 0),
                        claim_host or claim.get("machine", "?"),
                    )
                elif claim["status"] == "in_progress":
                    age = time.time() - claim.get("heartbeat", 0)
                    if age < self.stale_timeout:
                        continue  # Still active
                    # Stale — prefer same host (local NVMe data).
                    # Only allow cross-host reclaim after 2x stale_timeout
                    # (original host is probably permanently down).
                    claim_host = claim.get("hostname", "")
                    if claim_host and claim_host != self.hostname:
                        if age < self.stale_timeout * 2:
                            continue  # Give the original host more time
                    etag = obj.get("ETag")
                    logger.info(
                        "Reclaiming stale shard %d (last heartbeat %.0fs ago, host=%s)",
                        shard["id"],
                        age,
                        claim_host or claim.get("machine", "?"),
                    )
            except ClientError as e:
                if e.response["Error"]["Code"] != "NoSuchKey":
                    raise
                # Key doesn't exist -> unclaimed, will use IfNoneMatch

            # Try to claim with conditional write to prevent races
            claim_data = {
                "machine": self.machine_id,
                "hostname": self.hostname,
                "status": "in_progress",
                "claimed_at": time.time(),
                "heartbeat": time.time(),
                "completed": 0,
                "failed": 0,
                "skipped": 0,
            }
            try:
                put_kwargs = {
                    "Bucket": self.bucket,
                    "Key": key,
                    "Body": json.dumps(claim_data),
                }
                if etag is not None:
                    # Reclaiming stale shard: only succeed if nobody else
                    # reclaimed it since our GET (ETag still matches).
                    put_kwargs["IfMatch"] = etag
                else:
                    # New claim: only succeed if key doesn't exist yet.
                    put_kwargs["IfNoneMatch"] = "*"

                self.s3.put_object(**put_kwargs)
                self._claimed_at[shard["id"]] = claim_data["claimed_at"]
                logger.info(
                    "Claimed shard %d (articles %d-%d)",
                    shard["id"],
                    shard["start"],
                    shard["end"],
                )
                return shard
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if code in (
                    "PreconditionFailed",
                    "ConditionalCheckFailed",
                    "ConditionalRequestConflict",
                ):
                    # Another machine claimed it first — try next shard
                    logger.debug(
                        "Lost claim race for shard %d, trying next",
                        shard["id"],
                    )
                    continue
                raise
            except Exception:
                continue

        return None  # All shards claimed or completed

    def heartbeat(
        self,
        shard_id: int,
        completed: int = 0,
        failed: int = 0,
        skipped: int = 0,
        tiles: int = 0,
        **extra,
    ) -> None:
        """Update claim with current progress.

        Args:
            shard_id: Shard ID to update.
            completed: Number of completed articles.
            failed: Number of failed articles.
            skipped: Number of skipped articles.
            tiles: Number of tile images produced.
            **extra: Additional fields merged into the claim JSON. Known fields:
                disk_free_gb (float) — free disk space on the worker's output volume.
                s3_sync (bool) — whether this worker syncs output to S3.
                in_flight (list[str]) — article IDs currently being processed.
                recent_errors (list[str]) — last N error messages.
                fail_rate (float) — failed / total articles ratio.
        """
        key = f"{self.prefix}/claims/{shard_id:03d}.json"
        claim_data = {
            "machine": self.machine_id,
            "hostname": self.hostname,
            "status": "in_progress",
            "claimed_at": self._claimed_at.get(shard_id, time.time()),
            "heartbeat": time.time(),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "tiles": tiles,
            **extra,
        }
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=json.dumps(claim_data))

    def mark_done(
        self,
        shard_id: int,
        completed: int = 0,
        failed: int = 0,
        skipped: int = 0,
        tiles: int = 0,
        expected: int = 0,
    ) -> None:
        """Mark shard as completed or partial.

        Compares actual output (completed + skipped) against *expected* to
        decide the final status.  If expected > 0 and actual < 90% of
        expected, the shard is marked ``partial`` so it can be reclaimed
        and resumed later.

        Args:
            shard_id: Shard ID to mark done.
            completed: Final number of completed articles.
            failed: Final number of failed articles.
            skipped: Final number of skipped articles.
            tiles: Final number of tile images produced.
            expected: Expected number of non-redirect articles in this shard.
                      Pass 0 to skip the completeness check (always "completed").
        """
        actual = completed + skipped
        if expected > 0 and actual < expected * 0.9:
            status = "partial"
        else:
            status = "completed"

        key = f"{self.prefix}/claims/{shard_id:03d}.json"
        claim_data = {
            "machine": self.machine_id,
            "hostname": self.hostname,
            "status": status,
            "claimed_at": self._claimed_at.pop(shard_id, time.time()),
            "heartbeat": time.time(),
            "completed_at": time.time(),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "tiles": tiles,
            "expected": expected,
        }
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=json.dumps(claim_data))
        logger.info(
            "Shard %d marked %s (completed=%d, failed=%d, skipped=%d, tiles=%d, expected=%d)",
            shard_id,
            status,
            completed,
            failed,
            skipped,
            tiles,
            expected,
        )

    def mark_partial(
        self,
        shard_id: int,
        completed: int = 0,
        failed: int = 0,
        skipped: int = 0,
        tiles: int = 0,
        error: str = "",
    ) -> None:
        """Explicitly mark a shard as partial after an error.

        Unlike mark_done, this always sets status to ``partial`` regardless
        of counts.  The shard stays reclaimable so another worker (or the
        same worker on restart) can resume from where it left off.
        """
        key = f"{self.prefix}/claims/{shard_id:03d}.json"
        claim_data = {
            "machine": self.machine_id,
            "hostname": self.hostname,
            "status": "partial",
            "claimed_at": self._claimed_at.pop(shard_id, time.time()),
            "heartbeat": time.time(),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "tiles": tiles,
            "error": error,
        }
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=json.dumps(claim_data))
        logger.info(
            "Shard %d marked partial (completed=%d, tiles=%d, error=%s)",
            shard_id,
            completed,
            tiles,
            error[:120],
        )

    def get_all_claims(self) -> list[dict]:
        """Read all claim files from S3.

        Returns:
            List of claim dicts, each augmented with 'shard_id' parsed from the key.
        """
        paginator = self.s3.get_paginator("list_objects_v2")
        claims = []
        for page in paginator.paginate(
            Bucket=self.bucket, Prefix=f"{self.prefix}/claims/"
        ):
            for obj in page.get("Contents", []):
                try:
                    data = self.s3.get_object(Bucket=self.bucket, Key=obj["Key"])
                    claim = json.loads(data["Body"].read())
                    # Parse shard ID from key: "kiwix/claims/042.json" -> 42
                    fname = obj["Key"].rsplit("/", 1)[-1]
                    claim["shard_id"] = int(fname.replace(".json", ""))
                    claims.append(claim)
                except Exception as e:
                    logger.warning("Failed to read claim %s: %s", obj["Key"], e)
        return claims

    def get_status(self) -> dict:
        """Read all claims from S3 and return global status.

        Returns:
            Dict with keys: total_shards, completed, in_progress, stale,
            unclaimed, articles_done, machines, claims.
        """
        claims = self.get_all_claims()
        now = time.time()

        completed = sum(1 for c in claims if c["status"] == "completed")
        in_progress = sum(1 for c in claims if c["status"] == "in_progress")
        stale = sum(
            1
            for c in claims
            if c["status"] == "in_progress"
            and now - c.get("heartbeat", 0) > self.stale_timeout
        )
        total = self._manifest["num_shards"] if self._manifest else "?"
        unclaimed = (total - len(claims)) if isinstance(total, int) else "?"

        return {
            "total_shards": total,
            "completed": completed,
            "in_progress": in_progress - stale,
            "stale": stale,
            "unclaimed": unclaimed,
            "articles_done": sum(
                c.get("completed", 0) + c.get("failed", 0) + c.get("skipped", 0)
                for c in claims
            ),
            "machines": list(
                set(c["machine"] for c in claims if c["status"] == "in_progress")
            ),
            "claims": claims,
        }
