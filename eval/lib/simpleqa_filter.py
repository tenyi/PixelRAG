"""Dataset filtering utilities for SimpleQA."""

import ast
import json
import logging

from .simpleqa_data import load_simpleqa_data, load_simpleqa_verified_data

logger = logging.getLogger(__name__)


def _get_urls_from_metadata(example: dict) -> list[str]:
    """Extract all URLs from example metadata.

    Supports both SimpleQA format (metadata as string) and SimpleQA Verified format (metadata as dict or string).
    """
    meta = example.get("metadata")

    # If metadata is already a dict (SimpleQA Verified format)
    if isinstance(meta, dict):
        urls = meta.get("urls", [])
        if isinstance(urls, list):
            return urls
        return []

    # If metadata is a string (SimpleQA format or SimpleQA Verified string format)
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            try:
                meta = ast.literal_eval(meta)
            except (ValueError, SyntaxError):
                return []

    if isinstance(meta, dict) and "urls" in meta:
        urls = meta["urls"]
        if isinstance(urls, list):
            return urls
    return []


def load_simpleqa_wikipedia(
    num_examples: int | None = None,
    verified: bool = False,
    no_wiki_filter: bool = False,
) -> list[dict]:
    """Load SimpleQA examples that have Wikipedia URLs.

    Args:
        num_examples: Number of Wikipedia examples to return.
        verified: If True, use SimpleQA Verified dataset instead of original SimpleQA.

    Returns:
        List of examples where at least one URL contains 'wikipedia'.
        Maintains the original CSV file order.
    """
    # Load all data first (maintains original CSV order)
    if verified:
        all_data = load_simpleqa_verified_data(num_examples=None)
    else:
        all_data = load_simpleqa_data(num_examples=None)

    if no_wiki_filter:
        wikipedia_examples = list(all_data)
        logger.info(
            f"Skipping Wikipedia URL filter: returning all {len(wikipedia_examples)} examples"
        )
    else:
        # Filter for Wikipedia URLs, preserving original order
        # Exclude non-English Wikipedia and Category pages (e.g. de.wikipedia.org, Category:...)
        wikipedia_examples = []
        for example in all_data:
            urls = _get_urls_from_metadata(example)
            if any(
                "en.wikipedia.org/wiki/" in url and "/Category:" not in url
                for url in urls
            ):
                wikipedia_examples.append(example)

        logger.info(f"Found {len(wikipedia_examples)} examples with Wikipedia URLs")

    if num_examples:
        wikipedia_examples = wikipedia_examples[:num_examples]
        logger.info(f"Limiting to first {num_examples} Wikipedia examples")

    return wikipedia_examples


def load_simpleqa_by_domain(domain: str, num_examples: int | None = None) -> list[dict]:
    """Load SimpleQA examples filtered by URL domain.

    Args:
        domain: Domain to filter by (e.g., 'wikipedia', 'arxiv', 'github').
        num_examples: Number of examples to return.

    Returns:
        List of examples where at least one URL contains the domain.
    """
    all_data = load_simpleqa_data(num_examples=None)

    filtered = []
    for example in all_data:
        urls = _get_urls_from_metadata(example)
        if any(domain.lower() in url.lower() for url in urls):
            filtered.append(example)

    logger.info(f"Found {len(filtered)} examples with '{domain}' URLs")

    if num_examples:
        filtered = filtered[:num_examples]
        logger.info(f"Limiting to first {num_examples} examples")

    return filtered
