# Hybrid RAG (雙路混合檢索與雙階段純文字增強) 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 為 PixelRAG 引入純文字檢索作為輔助（包含 OCR 對齊、BM25 檢索與 RRF 排名融合），並在生成端將圖片與對應文字一併送入多模態模型，以顯著提高 RAG 的精準度。

**Architecture:** 
1. 離線端在 `chunk.py` 中調用 EasyOCR，對每個 chunk 提取文字並寫入 `chunks.json`。
2. 線上端在 `serve` 模組中建立 `BaseTextRetriever` 抽象層與 `BM25TextRetriever` 實作。
3. 在 `/search` API 端點中使用 RRF (Reciprocal Rank Fusion) 融合視覺 FAISS 與 BM25 排名，並於回傳的 Hit 結構中附帶 `text` 欄位。

**Tech Stack:** Python 3.12, rank-bm25, easyocr, FAISS, FastAPI, Pytest

## Global Constraints

- 程式碼本體保留原始語言，但變數宣告、說明及 Docstring 必須使用詳細的繁體中文註解。
- 專案依賴管理必須使用 `uv`，嚴格在 `.venv` 虛擬環境中執行。
- Git Commit Messages 一律使用繁體中文台灣用語。

---

### Task 1: 安裝專案依賴套件

**Files:**
- Modify: [pyproject.toml](file:///Users/tenyi/Projects/PixelRAG/pyproject.toml:26-46)

**Interfaces:**
- Consumes: None
- Produces: 依賴 `rank-bm25` 與 `easyocr` 加入到專案中。

- [ ] **Step 1: 修改 pyproject.toml 檔案**

在 `[project.optional-dependencies]` 的 `embed` 中加入 `easyocr`，並在 `serve` 中加入 `rank-bm25`：

```toml
embed = [
    "torch>=2.9.0",
    "torchvision>=0.24.0",
    "transformers>=4.57.0",
    "faiss-cpu>=1.9.0",
    "numpy>=1.26.0",
    "tqdm>=4.60.0",
    "easyocr>=1.7.1",
]
serve = [
    "fastapi>=0.115.0",
    "uvicorn>=0.30.0",
    "numpy>=1.26.0",
    "faiss-cpu>=1.9.0",
    "transformers>=4.57.0",
    "torch>=2.9.0",
    "qwen-vl-utils",
    "pydantic>=2.0.0",
    "rank-bm25>=0.2.2",
]
```

- [ ] **Step 2: 使用 uv 進行環境同步**

執行命令安裝新套件並同步虛擬環境：

Run: `uv sync --all-extras`
Expected: 順利下載安裝並編譯 `easyocr` 與 `rank-bm25`。

- [ ] **Step 3: 驗證套件是否可正確導入**

在虛擬環境下執行 Python 單行命令測試：

Run: `uv run python -c "import easyocr; import rank_bm25; print('Import OK')"`
Expected: 輸出 "Import OK" 且無報錯。

- [ ] **Step 4: 提交變更**

Run: `git add pyproject.toml uv.lock`
Expected: 成功暫存檔案。

Run: `git commit -m "chore: 新增 rank-bm25 與 easyocr 依賴套件"`
Expected: 成功提交。

---

### Task 2: 在 Chunk 階段整合 EasyOCR 提取文字

**Files:**
- Modify: [embed/src/pixelrag_embed/chunk.py](file:///Users/tenyi/Projects/PixelRAG/embed/src/pixelrag_embed/chunk.py:46-243)

**Interfaces:**
- Consumes: `easyocr` 套件
- Produces: 切分後的 `chunks.json` 會多出每個 chunk 的 `"text"` 欄位。

- [ ] **Step 1: 在 chunk.py 中載入 EasyOCR 延遲載入器**

修改 [embed/src/pixelrag_embed/chunk.py](file:///Users/tenyi/Projects/PixelRAG/embed/src/pixelrag_embed/chunk.py)，在檔案頭部或適當位置加入 lazy reader 宣告：

```python
# 全域 OCR Reader 快取變數
_ocr_reader = None

def _get_ocr_reader():
    """延遲載入 EasyOCR 讀取器，避免在多進程初始化時重複加載權重。"""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            import torch
            use_gpu = torch.cuda.is_available()
            # 支援繁體中文與英文
            _ocr_reader = easyocr.Reader(["ch_tra", "en"], gpu=use_gpu)
        except Exception as e:
            import logging
            logging.getLogger("chunk_tiles").warning(f"無法初始化 EasyOCR: {e}")
            _ocr_reader = "FAILED"
    return _ocr_reader
```

- [ ] **Step 2: 修改 chunk_article 函式以對 chunk 執行 OCR 辨識**

修改 `chunk_article` 中的瓦片切分邏輯，在寫入圖片 chunk 後進行文字辨識。

對於單一瓦片 Fast Path:
```python
        if w <= viewport_width and h <= CHUNK_HEIGHT:
            chunk_name = f"chunk_{tile_idx:04d}_00.png"
            chunk_path = os.path.join(article_dir, chunk_name)
            chunk_text = ""
            if not dry_run:
                shutil.copy2(tile_path, chunk_path)
                files_written += 1
                # 執行 OCR 辨識
                reader = _get_ocr_reader()
                if reader and reader != "FAILED":
                    try:
                        results = reader.readtext(chunk_path, detail=0)
                        chunk_text = " ".join(results)
                    except Exception as e:
                        logger.warning(f"OCR 辨識失敗 {chunk_path}: {e}")
```

對於 2D 網架切分 Path:
```python
                chunk_name = f"chunk_{tile_idx:04d}_{chunk_idx:02d}.png"
                chunk_path = os.path.join(article_dir, chunk_name)
                chunk_text = ""
                if not dry_run:
                    img.crop((x, y, x + cw, y + ch)).save(chunk_path, format="PNG")
                    files_written += 1
                    # 執行 OCR 辨識
                    reader = _get_ocr_reader()
                    if reader and reader != "FAILED":
                        try:
                            results = reader.readtext(chunk_path, detail=0)
                            chunk_text = " ".join(results)
                        except Exception as e:
                            logger.warning(f"OCR 辨識失敗 {chunk_path}: {e}")
```

並在 `chunks_info` 記錄中加上 `"text"`：
```python
                chunks_info.append(
                    {
                        "tile": tile_name,
                        "tile_index": tile_idx,
                        "chunk_index": chunk_idx,
                        "file": chunk_name,
                        "x_offset": x,
                        "y_offset": y,
                        "height": ch,
                        "width": cw,
                        "text": chunk_text, # 寫入對齊的純文字
                    }
                )
```

- [ ] **Step 3: 寫入單元測試驗證 chunker OCR 輸出**

建立 [tests/test_ocr_chunk.py](file:///Users/tenyi/Projects/PixelRAG/tests/test_ocr_chunk.py)：

```python
import os
import json
import shutil
from PIL import Image
from pixelrag_embed.chunk import chunk_article

def test_chunk_article_with_ocr(tmp_path):
    # 建立一個測試用的 article 目錄與圖片
    article_dir = tmp_path / "test_article.png.tiles"
    article_dir.mkdir()
    
    # 建立一個模擬的 tile_0000.png，裡面繪製一些測試文字
    tile_path = article_dir / "tile_0000.png"
    img = Image.new("RGB", (200, 100), color="white")
    # 不寫實際文字以加速測試，只要確保能順利透過 OCR 函式即可
    img.save(tile_path)
    
    # 建立 tiles.json
    tiles_json = {
        "page_height": 100,
        "viewport_width": 200,
        "tile_height": 100,
        "tiles": ["tile_0000.png"]
    }
    with open(article_dir / "tiles.json", "w") as f:
        json.dump(tiles_json, f)
        
    res = chunk_article(str(article_dir), dry_run=False, force=True)
    assert res is not None
    
    # 檢查 chunks.json 是否包含 text 欄位
    chunks_json_path = article_dir / "chunks.json"
    assert chunks_json_path.exists()
    with open(chunks_json_path) as f:
        manifest = json.load(f)
    
    assert len(manifest["chunks"]) > 0
    assert "text" in manifest["chunks"][0]
```

- [ ] **Step 4: 執行測試**

Run: `uv run pytest tests/test_ocr_chunk.py -v`
Expected: 測試通過 (PASS)。

- [ ] **Step 5: 提交變更**

Run: `git add embed/src/pixelrag_embed/chunk.py tests/test_ocr_chunk.py`
Expected: 暫存檔案成功。

Run: `git commit -m "feat: 於 chunking 階段整合 EasyOCR 提取圖片文字並記錄於 chunks.json"`
Expected: 成功提交。

---

### Task 3: 實作文字檢索抽象層與 BM25 檢索器

**Files:**
- Create: [serve/src/pixelrag_serve/text_retriever.py](file:///Users/tenyi/Projects/PixelRAG/serve/src/pixelrag_serve/text_retriever.py)

**Interfaces:**
- Consumes: `rank_bm25` 套件
- Produces: `BaseTextRetriever` 抽象類別與 `BM25TextRetriever` 實體類別。

- [ ] **Step 1: 建立 text_retriever.py 並實作分詞與檢索邏輯**

建立檔案 [serve/src/pixelrag_serve/text_retriever.py](file:///Users/tenyi/Projects/PixelRAG/serve/src/pixelrag_serve/text_retriever.py)：

```python
import os
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any

logger = logging.getLogger("text_retriever")

def tokenize_chinese_english(text: str) -> List[str]:
    """極輕量中文與英文混合分詞。中文拆為單個字，英文保留單字。"""
    if not text:
        return []
    # 匹配英數字單字，或者單個中文字元
    pattern = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")
    return pattern.findall(text.lower())

class BaseTextRetriever(ABC):
    @abstractmethod
    def index(self, tiles_dir: str) -> None:
        """從 tiles_dir 中所有子目錄載入 chunks.json 以建立/載入文字索引"""
        pass

    @abstractmethod
    def search(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """搜尋最相關的 chunks"""
        pass

class BM25TextRetriever(BaseTextRetriever):
    def __init__(self):
        self.bm25 = None
        self.doc_metadata = []  # 儲存與 corpus 索引一對一的 metadata
        self.coordinate_to_meta = {}  # 快速鍵值對查詢：(aid, ti, ci) -> metadata

    def index(self, tiles_dir: str) -> None:
        from rank_bm25 import BM25Okapi

        logger.info(f"正在從 {tiles_dir} 建立 BM25 文字索引...")
        corpus = []
        doc_metadata = []
        coordinate_to_meta = {}

        # 遍歷尋找所有的 chunks.json
        for root, _, files in os.walk(tiles_dir):
            if "chunks.json" in files:
                chunks_json_path = os.path.join(root, "chunks.json")
                dir_name = os.path.basename(root)
                if not dir_name.endswith(".png.tiles"):
                    continue
                try:
                    article_id = int(dir_name.split(".")[0])
                except ValueError:
                    continue

                try:
                    with open(chunks_json_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)

                    for chunk in manifest.get("chunks", []):
                        text = chunk.get("text", "")
                        if not text:
                            continue

                        tokens = tokenize_chinese_english(text)
                        if not tokens:
                            continue

                        corpus.append(tokens)
                        meta = {
                            "article_id": article_id,
                            "tile_index": int(chunk.get("tile_index", 0)),
                            "chunk_index": int(chunk.get("chunk_index", 0)),
                            "text": text,
                            "y_offset": int(chunk.get("y_offset", 0)),
                            "tile_height": int(chunk.get("height", 0)),
                        }
                        doc_metadata.append(meta)
                        key = (meta["article_id"], meta["tile_index"], meta["chunk_index"])
                        coordinate_to_meta[key] = meta
                except Exception as e:
                    logger.warning(f"讀取/索引 {chunks_json_path} 失敗: {e}")

        if corpus:
            self.bm25 = BM25Okapi(corpus)
            self.doc_metadata = doc_metadata
            self.coordinate_to_meta = coordinate_to_meta
            logger.info(f"BM25 文字索引建立完成，共 {len(corpus)} 個 chunks。")
        else:
            logger.warning("未找到任何純文字進行索引。BM25 索引為空。")
            self.bm25 = None
            self.doc_metadata = []
            self.coordinate_to_meta = {}

    def search(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.doc_metadata:
            return []

        query_tokens = tokenize_chinese_english(query_text)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        
        # 取得排序前 top_k 的 indices
        import numpy as np
        # 避免 top_k 大於現有 document 數
        actual_k = min(top_k, len(scores))
        if actual_k <= 0:
            return []
            
        top_indices = np.argsort(scores)[::-1][:actual_k]

        hits = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            meta = self.doc_metadata[idx]
            hits.append({
                "article_id": meta["article_id"],
                "tile_index": meta["tile_index"],
                "chunk_index": meta["chunk_index"],
                "text": meta["text"],
                "score": score,
            })
        return hits
```

- [ ] **Step 2: 撰寫單元測試驗證 BM25 檢索與分詞**

建立 [tests/test_text_retriever.py](file:///Users/tenyi/Projects/PixelRAG/tests/test_text_retriever.py)：

```python
import os
import json
from serve.src.pixelrag_serve.text_retriever import tokenize_chinese_english, BM25TextRetriever

def test_tokenize():
    # 驗證中文與英文混合分詞
    tokens = tokenize_chinese_english("Hello 世界 123! 關鍵字")
    assert tokens == ["hello", "世", "界", "123", "關", "鍵", "字"]

def test_bm25_retriever(tmp_path):
    # 建立 mock tiles 檔案結構
    article_dir = tmp_path / "42.png.tiles"
    article_dir.mkdir()
    
    chunks_json = {
        "chunks": [
            {
                "tile_index": 0,
                "chunk_index": 0,
                "y_offset": 10,
                "height": 100,
                "text": "這是包含特斯拉與電動車的內容"
            },
            {
                "tile_index": 0,
                "chunk_index": 1,
                "y_offset": 110,
                "height": 100,
                "text": "這是關於太空探索與 SpaceX 的段落"
            }
        ]
    }
    with open(article_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_json, f)
        
    retriever = BM25TextRetriever()
    retriever.index(str(tmp_path))
    
    # 測試搜尋特斯拉
    hits = retriever.search("特斯拉", top_k=2)
    assert len(hits) == 1
    assert hits[0]["article_id"] == 42
    assert hits[0]["chunk_index"] == 0
    
    # 測試搜尋太空
    hits_space = retriever.search("SpaceX太空", top_k=2)
    assert len(hits_space) == 1
    assert hits_space[0]["chunk_index"] == 1
```

- [ ] **Step 3: 執行測試**

Run: `uv run pytest tests/test_text_retriever.py -v`
Expected: 測試通過 (PASS)。

- [ ] **Step 4: 提交變更**

Run: `git add serve/src/pixelrag_serve/text_retriever.py tests/test_text_retriever.py`
Expected: 暫存檔案成功。

Run: `git commit -m "feat: 實作 BaseTextRetriever 抽象類別與 BM25TextRetriever 文字檢索器"`
Expected: 成功提交。

---

### Task 4: 於 serve API 整合雙路檢索與 RRF 排名融合

**Files:**
- Modify: [serve/src/pixelrag_serve/api.py](file:///Users/tenyi/Projects/PixelRAG/serve/src/pixelrag_serve/api.py:168-520)

**Interfaces:**
- Consumes: `BM25TextRetriever` 類別，具備 `coordinate_to_meta` 的字典。
- Produces: 
  - `Hit` 模型新增 `text: str | None = None`。
  - `/search` 端點整合雙路 RRF 檢索。

- [ ] **Step 1: 修改 Hit 模型與載入邏輯**

修改 [serve/src/pixelrag_serve/api.py](file:///Users/tenyi/Projects/PixelRAG/serve/src/pixelrag_serve/api.py:168-185)，在 `Hit` Pydantic 類別中新增 `text` 屬性：

```python
class Hit(BaseModel):
    score: float
    vector_id: int
    article_id: int
    tile_index: int
    chunk_index: int
    y_offset: int
    tile_height: int
    path: str
    url: str
    article_pages: str | None = None
    image_base64: str | None = None
    text: str | None = None # 新增對應的對齊純文字欄位
```

在 `load(args)` 的末尾初始化並索引文字檢索器：
```python
    # 初始化文字檢索器並索引
    from .text_retriever import BM25TextRetriever
    text_retriever = BM25TextRetriever()
    text_retriever.index(args.tiles_dir)

    _state.update(
        {
            # ... 保留原有的屬性
            "text_retriever": text_retriever,
        }
    )
```

- [ ] **Step 2: 修改 search 函數以整合 RRF 排名融合**

修改 `search(req: SearchRequest)` 中的檢索邏輯，用 RRF 代替純視覺檢索。

修改 [serve/src/pixelrag_serve/api.py](file:///Users/tenyi/Projects/PixelRAG/serve/src/pixelrag_serve/api.py:398-502)：

```python
@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    t0 = time.time()

    # 1. 取得視覺查詢向量
    if req.queries and all(q.embedding is not None for q in req.queries):
        query_vectors = _normalize_query_embeddings(req.queries)
    else:
        if any(q.embedding is not None for q in req.queries):
            raise HTTPException(
                status_code=400,
                detail="Do not mix pre-computed embeddings with text/image queries in one request.",
            )
        query_vectors = _encode_queries(req.queries, req.instruction)
    t_encode = time.time() - t0

    # 2. 進行視覺 FAISS 檢索 (過度提取以預留過濾空間)
    index = _state["index"]
    default_nprobe = index.nprobe
    if req.nprobe is not None:
        index.nprobe = req.nprobe

    # 過度提取
    if req.articles_only:
        fetch_k = req.n_docs * 10
    elif req.min_tile_height:
        fetch_k = req.n_docs * 5
    else:
        fetch_k = req.n_docs * 3  # RRF 融合需要足夠的候選集
        
    distances, indices = index.search(query_vectors, fetch_k)

    if req.nprobe is not None:
        index.nprobe = default_nprobe
    t_search = time.time() - t0 - t_encode

    # 3. 排名融合與結果建構
    meta = _state["metadata"]
    article_ids = meta["article_ids"]
    tile_indices = meta["tile_indices"]
    chunk_indices = meta["chunk_indices"]
    y_offsets = meta["y_offsets"]
    tile_heights = meta["tile_heights"]
    tiles_dir = _state.get("tiles_dir", "")
    text_retriever = _state.get("text_retriever")

    RRF_K = 60 # RRF 排名平滑常數
    results = []

    for qi in range(len(req.queries)):
        # 3.1 收集視覺路候選 (過濾 -1)
        visual_candidates = []
        seen_keys = set()
        for j in range(fetch_k):
            vid = int(indices[qi, j])
            if vid == -1:
                continue
            aid = int(article_ids[vid])
            ti = int(tile_indices[vid])
            ci = int(chunk_indices[vid])
            key = (aid, ti, ci)
            if key not in seen_keys:
                seen_keys.add(key)
                visual_candidates.append({
                    "article_id": aid,
                    "tile_index": ti,
                    "chunk_index": ci,
                    "y_offset": int(y_offsets[vid]),
                    "tile_height": int(tile_heights[vid]),
                    "vector_id": vid,
                    "score": float(distances[qi, j])
                })

        # 3.2 收集文字路候選
        text_candidates = []
        if text_retriever and req.queries[qi].text:
            text_candidates = text_retriever.search(req.queries[qi].text, fetch_k)

        # 3.3 計算 RRF 得分
        rrf_scores = {} # (aid, ti, ci) -> score
        
        for rank, cand in enumerate(visual_candidates, 1):
            key = (cand["article_id"], cand["tile_index"], cand["chunk_index"])
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (RRF_K + rank))

        for rank, cand in enumerate(text_candidates, 1):
            key = (cand["article_id"], cand["tile_index"], cand["chunk_index"])
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (RRF_K + rank))

        # 3.4 依據 RRF 得分降序排序
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        # 3.5 建構 Hits 列表
        hits = []
        for aid, ti, ci in sorted_keys:
            key = (aid, ti, ci)
            
            # 從文字檢索器的記憶體中獲取對齊屬性
            meta_info = None
            if text_retriever and hasattr(text_retriever, "coordinate_to_meta"):
                meta_info = text_retriever.coordinate_to_meta.get(key)

            if meta_info:
                y_offset = meta_info["y_offset"]
                tile_height = meta_info["tile_height"]
                text_content = meta_info["text"]
            else:
                # 若無對應的文字記錄，回歸尋找視覺候選或預設值
                y_offset = 0
                tile_height = 0
                text_content = ""
                for vc in visual_candidates:
                    if vc["article_id"] == aid and vc["tile_index"] == ti and vc["chunk_index"] == ci:
                        y_offset = vc["y_offset"]
                        tile_height = vc["tile_height"]
                        break

            # 進行 min_tile_height 與 metadata 頁面過濾
            if req.min_tile_height and tile_height < req.min_tile_height:
                continue
            url = _resolve_url(aid)
            if req.articles_only and _is_meta(url):
                continue

            tile_path = _resolve_path(aid, ti, ci)
            img_b64 = None
            if req.include_images and tile_path and os.path.exists(tile_path):
                with open(tile_path, "rb") as fp:
                    img_b64 = base64.b64encode(fp.read()).decode()
            elif req.include_images and _state.get("ondemand") is not None:
                img_b64 = _ondemand_chunk_b64(aid, ti, ci, tile_height)

            rel_path = tile_path
            if tiles_dir:
                candidate = os.path.relpath(tile_path, tiles_dir)
                if not candidate.startswith(".."):
                    rel_path = candidate

            # 對應在 FAISS 中的 vector_id，若是從純文字路來且視覺無對應則標為 -1
            vid = -1
            for vc in visual_candidates:
                if vc["article_id"] == aid and vc["tile_index"] == ti and vc["chunk_index"] == ci:
                    vid = vc["vector_id"]
                    break

            hits.append(
                Hit(
                    score=float(rrf_scores[key]), # 傳回融合分數
                    vector_id=vid,
                    article_id=aid,
                    tile_index=ti,
                    chunk_index=ci,
                    y_offset=y_offset,
                    tile_height=tile_height,
                    path=rel_path,
                    url=url,
                    article_pages=_article_pages(aid),
                    image_base64=img_b64,
                    text=text_content
                )
            )

            if len(hits) >= req.n_docs:
                break
        
        results.append(QueryResult(hits=hits))

    logger.info(
        "RRF 混合搜尋: %d 個 queries, n_docs=%d, encode=%.3fs, search=%.3fs, total=%.3fs",
        len(req.queries),
        req.n_docs,
        t_encode,
        t_search,
        time.time() - t0,
    )

    return SearchResponse(results=results)
```

- [ ] **Step 3: 撰寫 RRF 融合邏輯與 API 整體單元測試**

建立 [tests/test_rrf_api.py](file:///Users/tenyi/Projects/PixelRAG/tests/test_rrf_api.py)：

```python
import pytest
from fastapi.testclient import TestClient
from serve.src.pixelrag_serve.api import app, _state
from unittest.mock import MagicMock

@pytest.fixture
def mock_state():
    # 模擬 FAISS index
    mock_index = MagicMock()
    mock_index.nprobe = 1
    mock_index.ntotal = 10
    
    # 模擬 metadata.npz
    mock_meta = {
        "article_ids": [10, 20],
        "tile_indices": [0, 0],
        "chunk_indices": [0, 1],
        "y_offsets": [0, 100],
        "tile_heights": [100, 100]
    }
    
    # 模擬 articles
    mock_articles = {
        10: {"url": "https://example.com/art10", "title": "Art10"},
        20: {"url": "https://example.com/art20", "title": "Art20"}
    }
    
    # 模擬 TextRetriever
    mock_text_retriever = MagicMock()
    mock_text_retriever.search.return_value = [
        {"article_id": 20, "tile_index": 0, "chunk_index": 1, "text": "SpaceX 太空", "score": 2.0}
    ]
    mock_text_retriever.coordinate_to_meta = {
        (20, 0, 1): {"article_id": 20, "tile_index": 0, "chunk_index": 1, "text": "SpaceX 太空", "y_offset": 100, "tile_height": 100}
    }
    
    # 模擬 Qwen3-VL embedder 輸出
    import numpy as np
    
    # 修改全局 _state
    _state.clear()
    _state.update({
        "index": mock_index,
        "metadata": mock_meta,
        "articles": mock_articles,
        "text_retriever": mock_text_retriever,
        "device": "cpu",
        "model_name": "mock-model",
        "tiles_dir": "",
        "dimension": 128,
        "nlist": 4096,
        "index_built_at": "2026-07-10T12:00:00Z",
        "index_size_bytes": 100,
        "metadata_size_bytes": 100
    })

def test_search_rrf_endpoint(mock_state, monkeypatch):
    import numpy as np
    # 模擬 _encode_queries 回傳 dummy 向量
    monkeypatch.setattr("serve.src.pixelrag_serve.api._encode_queries", lambda q, inst: np.zeros((len(q), 128)))
    
    # 模擬 FAISS index.search 回傳 indices=[[0, 1]], distances=[[0.9, 0.8]]
    # 0 號 vector 是 article 10, 1 號 vector 是 article 20
    _state["index"].search.return_value = (np.array([[0.9, 0.8]]), np.array([[0, 1]]))
    
    client = TestClient(app)
    
    # 發送搜尋請求
    response = client.post("/search", json={
        "queries": [{"text": "太空"}],
        "n_docs": 2
    })
    
    assert response.status_code == 200
    data = response.json()
    
    # SpaceX (article 20, chunk 1) 出現在視覺(第2名)與文字(第1名)，它的 RRF 得分應高於 article 10
    hits = data["results"][0]["hits"]
    assert len(hits) >= 1
    # 排名第一的應該是 article 20 (因為文字路與視覺路皆有召回)
    assert hits[0]["article_id"] == 20
    assert hits[0]["text"] == "SpaceX 太空"
```

- [ ] **Step 4: 執行測試**

Run: `uv run pytest tests/test_rrf_api.py -v`
Expected: 測試通過 (PASS)。

- [ ] **Step 5: 提交變更**

Run: `git add serve/src/pixelrag_serve/api.py tests/test_rrf_api.py`
Expected: 暫存檔案成功。

Run: `git commit -m "feat: 在 API 搜尋端點整合視覺與 BM25 文字雙路檢索，並引入 RRF 融合排序"`
Expected: 成功提交。
