import os
import json
from pixelrag_serve.text_retriever import tokenize_chinese_english, BM25TextRetriever

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
            },
            {
                "tile_index": 0,
                "chunk_index": 2,
                "y_offset": 210,
                "height": 100,
                "text": "這是一份完全無關的背景測試資料"
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
