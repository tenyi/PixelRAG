# Chromium Build on Centralia Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a patched Chromium (v150.0.7844.0) on Centralia (SSH: `CentraliaB200`, user `yichuan_wang`) with our two custom CDP features: `rawFilePath` parameter for `Page.captureScreenshot` and `directClip` parameter for parallel tile capture.

**Architecture:** All work happens under `/work/yichuan_wang/chromium-build/` on Centralia (NFS-mounted /work has 12TB free). depot_tools is cloned alongside the chromium checkout. The patch is generated locally from `~/chromium/src` (HEAD~1 diff) and transferred via scp. Build uses a release/official/no-debug/no-PGO args.gn for a fast, deployable binary.

**Tech Stack:** Chromium source (~30GB no-history), depot_tools, gn, autoninja (ninja), Python 3.12 (already on Centralia), Ubuntu 24.04, 224 cores for parallel build.

---

## Environment Facts (verified pre-plan)

- **Centralia SSH alias:** `CentraliaB200` (user `yichuan_wang`)
- **Workspace:** `/work/yichuan_wang/chromium-build/` (NFS, ~12TB free — plenty of room)
- **Local disk on Centralia:** `/dev/md0` 209GB free — avoid storing large files there
- **Local patch source:** `~/chromium/src` on local machine, `HEAD~1` diff covers 6 files, 166 insertions
- **Chromium version:** 150.0.7844.0 (MAJOR=150, BUILD=7844)
- **OS on Centralia:** Ubuntu 24.04.4 LTS (Noble)
- **Python:** `/usr/bin/python3` (3.12.3) — already present, no install needed
- **ninja/autoninja:** NOT present on Centralia — comes from depot_tools, added to PATH

---

## File Map

| Location | Purpose |
|---|---|
| `/work/yichuan_wang/chromium-build/depot_tools/` | Google's build tools (gn, fetch, autoninja, gclient) |
| `/work/yichuan_wang/chromium-build/chromium/` | gclient checkout root (contains `.gclient`) |
| `/work/yichuan_wang/chromium-build/chromium/src/` | Chromium source tree |
| `/work/yichuan_wang/chromium-build/chromium/src/out/Release/` | Build output dir |
| `/work/yichuan_wang/chromium-build/chromium/src/out/Release/args.gn` | Build configuration |
| `/work/yichuan_wang/chromium-build/chromium_patches.diff` | Our custom patch (transferred from local) |
| `/work/yichuan_wang/chromium-build/build.log` | autoninja build log (stream with `tail -f`) |

---

## Task 1: Verify workspace and install depot_tools

**Files:**
- Create: `/work/yichuan_wang/chromium-build/` (directory)
- Create: `/work/yichuan_wang/chromium-build/depot_tools/` (git clone)

- [ ] **Step 1.1: Create workspace directory on Centralia**

```bash
ssh CentraliaB200 "mkdir -p /work/yichuan_wang/chromium-build && echo 'workspace ready'"
```

Expected output: `workspace ready`

- [ ] **Step 1.2: Clone depot_tools into workspace**

Note: the correct URL is `chromium/tools/depot_tools` (not `chromium/depot_tools`).

```bash
ssh CentraliaB200 "git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git /work/yichuan_wang/chromium-build/depot_tools"
```

Expected: clone completes, last line something like `Resolving deltas: 100%`.

- [ ] **Step 1.3: Verify depot_tools tools exist**

```bash
ssh CentraliaB200 "ls /work/yichuan_wang/chromium-build/depot_tools/fetch /work/yichuan_wang/chromium-build/depot_tools/gclient /work/yichuan_wang/chromium-build/depot_tools/autoninja"
```

Expected: three file paths printed (no errors).

---

## Task 2: Fetch Chromium source (no history)

This is the longest step — `fetch --no-history chromium` downloads ~30GB and runs `gclient sync`. With a fast connection it takes 30–90 minutes.

**Files:**
- Create: `/work/yichuan_wang/chromium-build/chromium/` (gclient root)
- Create: `/work/yichuan_wang/chromium-build/chromium/src/` (source tree, ~30GB)

- [ ] **Step 2.1: Create the chromium checkout directory**

```bash
ssh CentraliaB200 "mkdir -p /work/yichuan_wang/chromium-build/chromium"
```

- [ ] **Step 2.2: Start the fetch in a detached screen session**

`fetch` must run from the chromium checkout root dir. We use `screen` so SSH disconnection doesn't kill it. The output is redirected to a log file for monitoring.

IMPORTANT: The home dir (`/home/eecs/yichuan_wang`) is full (10GB NFS, 0 bytes free). Set XDG dirs to /work to prevent depot_tools from failing on `~/.config/depot_tools`. Also put both `depot_tools/.cipd_bin` and `depot_tools` in PATH — vpython3 needs cipd in PATH.

```bash
ssh CentraliaB200 "screen -dmS chromium_fetch bash -c '
  export XDG_CONFIG_HOME=/work/yichuan_wang/chromium-build/xdg/config
  export XDG_CACHE_HOME=/work/yichuan_wang/chromium-build/xdg/cache
  export XDG_DATA_HOME=/work/yichuan_wang/chromium-build/xdg/data
  export XDG_STATE_HOME=/work/yichuan_wang/chromium-build/xdg/state
  export PATH=/work/yichuan_wang/chromium-build/depot_tools/.cipd_bin:/work/yichuan_wang/chromium-build/depot_tools:\$PATH
  export DEPOT_TOOLS_DIR=/work/yichuan_wang/chromium-build/depot_tools
  cd /work/yichuan_wang/chromium-build/chromium
  echo \"FETCH_START \$(date)\" > /work/yichuan_wang/chromium-build/fetch.log
  fetch --no-history chromium >> /work/yichuan_wang/chromium-build/fetch.log 2>&1
  echo \"FETCH_DONE exit=\$? at \$(date)\" >> /work/yichuan_wang/chromium-build/fetch.log
'"
```

- [ ] **Step 2.3: Verify screen session started**

```bash
ssh CentraliaB200 "screen -ls | grep chromium_fetch"
```

Expected: a line like `12345.chromium_fetch  (Detached)`.

- [ ] **Step 2.4: Monitor fetch progress (check periodically)**

```bash
ssh CentraliaB200 "tail -20 /work/yichuan_wang/chromium-build/fetch.log"
```

Re-run this command to watch progress. Fetch is done when the log contains `FETCH_DONE exit=0`.

- [ ] **Step 2.5: Verify source tree exists after fetch completes**

```bash
ssh CentraliaB200 "ls /work/yichuan_wang/chromium-build/chromium/src/chrome/VERSION"
```

Expected: file path printed (no error). If the file doesn't exist, fetch failed — check `fetch.log` for errors.

- [ ] **Step 2.6: Verify Chromium version matches local**

```bash
ssh CentraliaB200 "cat /work/yichuan_wang/chromium-build/chromium/src/chrome/VERSION"
```

Expected:
```
MAJOR=150
MINOR=0
BUILD=7844
PATCH=0
```

If the version differs, the patch may not apply cleanly. Record the actual version for the patch step.

---

## Task 3: Transfer and apply our patches

Our patch adds two CDP features to `Page.captureScreenshot`:
- `rawFilePath`: write screenshot directly to a file path (bypassing base64 encoding)
- `directClip`: clip parameter for parallel tile capture

**Files:**
- Modify: `content/browser/devtools/protocol/page_handler.cc` (primary patch target)
- Modify: `content/browser/devtools/protocol/page_handler.h`
- Modify: `content/renderer/render_widget_host/render_widget_host_impl.cc`
- Modify: `content/renderer/render_widget_host/render_widget_host_impl.h`
- Modify: `third_party/blink/public/devtools_protocol/domains/Page.pdl`
- Modify: `third_party/blink/renderer/platform/widget/widget_base.cc`
- Transfer: `/work/yichuan_wang/chromium-build/chromium_patches.diff`

- [ ] **Step 3.1: Generate the patch from local machine**

Run this on the LOCAL machine:

```bash
git -C ~/chromium/src diff HEAD~1 > /tmp/chromium_patches.diff
wc -l /tmp/chromium_patches.diff
```

Expected: file is non-empty (~300+ lines).

- [ ] **Step 3.2: Transfer patch to Centralia**

Run this on the LOCAL machine:

```bash
scp /tmp/chromium_patches.diff CentraliaB200:/work/yichuan_wang/chromium-build/chromium_patches.diff
```

- [ ] **Step 3.3: Verify patch arrived on Centralia**

```bash
ssh CentraliaB200 "wc -l /work/yichuan_wang/chromium-build/chromium_patches.diff"
```

Expected: same line count as local.

- [ ] **Step 3.4: Apply the patch**

```bash
ssh CentraliaB200 "cd /work/yichuan_wang/chromium-build/chromium/src && git apply /work/yichuan_wang/chromium-build/chromium_patches.diff"
```

Expected: no output (silent success). If you see errors like "patch does not apply", see Step 3.5.

- [ ] **Step 3.5: (If patch fails) Try with --3way or check fuzz**

If Step 3.4 fails with "patch does not apply":

```bash
ssh CentraliaB200 "cd /work/yichuan_wang/chromium-build/chromium/src && git apply --3way /work/yichuan_wang/chromium-build/chromium_patches.diff"
```

If that also fails, the Chromium version on Centralia differs from the local checkout. Check `cat /work/yichuan_wang/chromium-build/chromium/src/chrome/VERSION` and compare to local (`cat ~/chromium/src/chrome/VERSION`). If versions differ significantly, you may need to regenerate the patch from the correct base commit — fetch the local HEAD's commit hash with `git -C ~/chromium/src rev-parse HEAD~1` and use that.

- [ ] **Step 3.6: Verify patch was applied**

```bash
ssh CentraliaB200 "cd /work/yichuan_wang/chromium-build/chromium/src && git diff --stat"
```

Expected output:
```
 content/browser/devtools/protocol/page_handler.cc  | 129 +++++...
 content/browser/devtools/protocol/page_handler.h   |  14 +++
 content/renderer/render_widget_host/render_widget_host_impl.cc  |   9 ++
 content/renderer/render_widget_host/render_widget_host_impl.h   |   5 +
 third_party/blink/public/devtools_protocol/domains/Page.pdl     |   6 +
 third_party/blink/renderer/platform/widget/widget_base.cc       |  12 +-
 6 files changed, 166 insertions(+), 9 deletions(-)
```

---

## Task 4: Configure the build

**Files:**
- Create: `/work/yichuan_wang/chromium-build/chromium/src/out/Release/` (directory)
- Create: `/work/yichuan_wang/chromium-build/chromium/src/out/Release/args.gn`

- [ ] **Step 4.1: Create the build output directory**

```bash
ssh CentraliaB200 "mkdir -p /work/yichuan_wang/chromium-build/chromium/src/out/Release"
```

- [ ] **Step 4.2: Write args.gn**

```bash
ssh CentraliaB200 "cat > /work/yichuan_wang/chromium-build/chromium/src/out/Release/args.gn << 'EOF'
is_debug = false
is_official_build = true
is_component_build = false
symbol_level = 0
blink_symbol_level = 0
chrome_pgo_phase = 0
EOF"
```

- [ ] **Step 4.3: Verify args.gn content**

```bash
ssh CentraliaB200 "cat /work/yichuan_wang/chromium-build/chromium/src/out/Release/args.gn"
```

Expected exact output:
```
is_debug = false
is_official_build = true
is_component_build = false
symbol_level = 0
blink_symbol_level = 0
chrome_pgo_phase = 0
```

---

## Task 5: Run gn gen

`gn gen` reads `args.gn` and generates all the ninja build files. This takes 2–5 minutes on 224 cores.

**Files:**
- Create: `/work/yichuan_wang/chromium-build/chromium/src/out/Release/build.ninja` (generated)
- Create: `/work/yichuan_wang/chromium-build/gn_gen.log`

- [ ] **Step 5.1: Run gn gen in a screen session**

```bash
ssh CentraliaB200 "screen -dmS chromium_gn bash -c '
  export PATH=/work/yichuan_wang/chromium-build/depot_tools:\$PATH
  cd /work/yichuan_wang/chromium-build/chromium/src
  gn gen out/Release > /work/yichuan_wang/chromium-build/gn_gen.log 2>&1
  echo \"GN_DONE exit=\$?\" >> /work/yichuan_wang/chromium-build/gn_gen.log
'"
```

- [ ] **Step 5.2: Wait for gn gen to finish**

```bash
ssh CentraliaB200 "tail -5 /work/yichuan_wang/chromium-build/gn_gen.log"
```

Re-run until you see `GN_DONE exit=0`. If exit is non-zero, check the full log:

```bash
ssh CentraliaB200 "cat /work/yichuan_wang/chromium-build/gn_gen.log"
```

Common gn errors and fixes:
- `Python not found`: verify `which python3` works on Centralia (it does per our check)
- `No targets match`: args.gn typo — re-check Step 4.2

- [ ] **Step 5.3: Verify build.ninja was generated**

```bash
ssh CentraliaB200 "ls -lh /work/yichuan_wang/chromium-build/chromium/src/out/Release/build.ninja"
```

Expected: file exists, non-zero size.

---

## Task 6: Build chrome with autoninja

This is the main build step. With 224 cores and no debug symbols, expect 60–120 minutes for a full build. The output is the `chrome` binary.

**Files:**
- Create: `/work/yichuan_wang/chromium-build/chromium/src/out/Release/chrome` (built binary)
- Create: `/work/yichuan_wang/chromium-build/build.log`

- [ ] **Step 6.1: Start autoninja build in screen session**

`autoninja` automatically sets `-j` based on CPU count (will use ~224 jobs). It reads the `NINJA_SUMMARIZE_BUILD` env var to show progress.

```bash
ssh CentraliaB200 "screen -dmS chromium_build bash -c '
  export PATH=/work/yichuan_wang/chromium-build/depot_tools:\$PATH
  cd /work/yichuan_wang/chromium-build/chromium/src
  autoninja -C out/Release chrome > /work/yichuan_wang/chromium-build/build.log 2>&1
  echo \"BUILD_DONE exit=\$?\" >> /work/yichuan_wang/chromium-build/build.log
'"
```

- [ ] **Step 6.2: Verify screen session started**

```bash
ssh CentraliaB200 "screen -ls | grep chromium_build"
```

Expected: a line like `12345.chromium_build  (Detached)`.

- [ ] **Step 6.3: Monitor build progress**

```bash
ssh CentraliaB200 "tail -5 /work/yichuan_wang/chromium-build/build.log"
```

You'll see ninja progress lines like `[1234/89000] CXX obj/content/...`. Re-run every few minutes to watch progress. Build is done when you see `BUILD_DONE exit=0`.

To watch CPU utilization:
```bash
ssh CentraliaB200 "uptime"
```

If the build is running, load average should be ~200+.

- [ ] **Step 6.4: Check for build errors (if BUILD_DONE shows non-zero exit)**

```bash
ssh CentraliaB200 "grep -i 'error:' /work/yichuan_wang/chromium-build/build.log | tail -20"
```

Common errors:
- `undefined reference`: usually means a `.h` change wasn't matched with a `.cc` change in the patch. Check that the patch applied fully (Task 3).
- `ninja: build stopped`: check the lines above for the actual C++ error.
- Disk full: run `df -h /work` — if /work is at 100%, free space or use a different path.

- [ ] **Step 6.5: Verify chrome binary exists and is executable**

```bash
ssh CentraliaB200 "ls -lh /work/yichuan_wang/chromium-build/chromium/src/out/Release/chrome"
```

Expected: file ~200–300MB, executable bit set (permissions like `-rwxr-xr-x`).

- [ ] **Step 6.6: Smoke-test the binary**

```bash
ssh CentraliaB200 "/work/yichuan_wang/chromium-build/chromium/src/out/Release/chrome --version"
```

Expected: `Chromium 150.0.7844.0` (or similar version line).

Note: Chrome may print warnings about display/GPU on a headless server — that's normal. We care only that the binary runs and prints its version.

---

## Task 7: Verify our custom CDP features are compiled in

Our patch adds `rawFilePath` and `directClip` parameters to `Page.captureScreenshot`. We verify they made it into the compiled protocol.

**Files:**
- Read: `/work/yichuan_wang/chromium-build/chromium/src/out/Release/gen/third_party/blink/public/devtools_protocol/protocol/page.json` (generated protocol JSON)

- [ ] **Step 7.1: Check the generated protocol JSON for our parameters**

```bash
ssh CentraliaB200 "grep -n 'rawFilePath\|directClip' /work/yichuan_wang/chromium-build/chromium/src/out/Release/gen/third_party/blink/public/devtools_protocol/protocol/page.json"
```

Expected: at least 2 lines mentioning `rawFilePath` and `directClip`.

- [ ] **Step 7.2: Check the compiled binary for our parameter strings**

```bash
ssh CentraliaB200 "strings /work/yichuan_wang/chromium-build/chromium/src/out/Release/chrome | grep -c 'rawFilePath'"
```

Expected: at least 1 (the string is embedded in the binary). If 0, the patch didn't compile into the binary — recheck that `git diff --stat` in Task 3 Step 6 was correct.

---

## Timing Estimates

| Task | Estimated Duration |
|---|---|
| Task 1: depot_tools clone | 2–5 min |
| Task 2: `fetch --no-history chromium` | 30–90 min (network-dependent) |
| Task 3: patch transfer + apply | 2 min |
| Task 4: args.gn setup | 1 min |
| Task 5: `gn gen` | 3–7 min |
| Task 6: `autoninja` build | 60–120 min (224 cores, no debug) |
| Task 7: verification | 2 min |
| **Total** | **~2–4 hours** |

---

## Recovery Notes

- **If screen session dies unexpectedly:** Re-attach with `screen -r chromium_build` to see any final error, then restart from the last completed step.
- **If fetch is interrupted:** Re-run `fetch --no-history chromium` from the same directory — gclient will resume.
- **If build is interrupted:** Re-run `autoninja -C out/Release chrome` — ninja tracks completed targets and resumes from where it left off.
- **If /work fills up:** `du -sh /work/yichuan_wang/chromium-build/chromium/src/out/Release/obj/` is usually the largest dir. Consider deleting `.o` files after build if you only need the final binary: `find out/Release/obj -name '*.o' -delete`.
