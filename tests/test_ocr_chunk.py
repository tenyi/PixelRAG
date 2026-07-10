import os
import json
from PIL import Image
from pixelrag_embed.chunk import chunk_article

def test_chunk_article_with_ocr(tmp_path):
    # 建立一個測試用的 article 目錄與圖片
    article_dir = tmp_path / "test_article.png.tiles"
    article_dir.mkdir()
    
    # 建立一個模擬的 tile_0000.png
    tile_path = article_dir / "tile_0000.png"
    img = Image.new("RGB", (200, 100), color="white")
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
