import pytest
from fastapi.testclient import TestClient
from pixelrag_serve.api import app, _state
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
    monkeypatch.setattr("pixelrag_serve.api._encode_queries", lambda q, inst: np.zeros((len(q), 128)))
    
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
