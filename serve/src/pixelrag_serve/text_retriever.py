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
