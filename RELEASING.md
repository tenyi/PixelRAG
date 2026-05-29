# Releasing PixelRAG to PyPI

PixelRAG ships as a single PyPI package, **`pixelrag`** — one distribution that bundles the
umbrella CLI plus every stage module (`pixelrag_render`, `pixelrag_embed`, `pixelrag_index`,
`pixelrag_serve`). The core install is light (rendering only); heavy ML stages are extras:

```bash
pip install pixelrag            # pixelshot + pixelrag umbrella (no torch)
pip install 'pixelrag[serve]'   # + search API   (also [embed], [index], [all])
```

`train/` is a separate local project and is **not** published.

Publishing is automated by [`.github/workflows/release.yml`](.github/workflows/release.yml)
using an **account-scoped PyPI API token** stored as a repo secret.

## One-time setup

1. **Create an account-scoped PyPI API token** at
   <https://pypi.org/manage/account/token/>. Copy the `pypi-...` value.

2. **Store it as the `PYPI_API_TOKEN` repo secret:**

   ```bash
   gh secret set PYPI_API_TOKEN --repo StarTrail-org/PixelRAG
   ```

   (or repo Settings → Secrets and variables → Actions → New repository secret).

## Cutting a release

1. **Bump the version:**

   ```bash
   uv version X.Y.Z
   ```

2. **Commit and push.**

3. **Publish a GitHub Release** with tag `vX.Y.Z` (matching the version). That fires
   `release.yml`, which builds and publishes `pixelrag` to PyPI.

## Dry run

Actions → **Release to PyPI** → Run workflow (leave `dry_run` checked) builds the package
and lists artifacts **without** uploading. Use it to sanity-check before a real release.

## Notes

- **README images on PyPI.** The README uses repo-relative image paths (`docs/assets/...`),
  which render on GitHub but **not** on the PyPI project page. To show them on PyPI, switch
  those `<img src>` to absolute `https://raw.githubusercontent.com/...` URLs.
- **sdist scope.** The repo root holds large data dirs (`.venv`, `tiles`, `arxiv`, …); the
  package restricts its sdist to the source dirs + `README.md` + `LICENSE`
  (`[tool.hatch.build.targets.sdist]` in `pyproject.toml`).
- **Superseded packages.** `pixelrag-render`, `pixelrag-embed`, `pixelrag-index`, and
  `pixelrag-serve` were published once (0.1.0) before consolidation and are now **yanked**;
  everything lives in `pixelrag`.
