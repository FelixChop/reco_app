import json
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "scraping" / "data"
STATIC_DIR = BASE_DIR / "backend" / "static" / "images"

def optimize_images(json_file, mode):
    print(f"Optimizing {mode} from {json_file}...")
    path = DATA_DIR / json_file
    if not path.exists():
        print(f"  File {path} not found.")
        return
    
    with open(path, "r") as f:
        items = json.load(f)
        
    mode_dir = STATIC_DIR / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    
    updated_items = []
    for idx, item in enumerate(items):
        url = item.get("image_url")
        if not url:
            updated_items.append(item)
            continue
            
        # Check if already local
        if "localhost:8000/static" in url:
            updated_items.append(item)
            continue
            
        try:
            # Download
            print(f"  Downloading {url}...")
            response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                
                # Resize/Convert
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.thumbnail((400, 400))
                
                # Sanitize filename
                safe_name = "".join([c for c in item['name'] if c.isalnum() or c in "._-"])
                filename = f"{idx}_{safe_name}.jpg"
                
                save_path = mode_dir / filename
                img.save(save_path, "JPEG", quality=80)
                
                item["image_url"] = f"http://localhost:8000/static/images/{mode}/{filename}"
                print(f"  ✅ Processed {item['name']}")
            else:
                print(f"  ❌ Failed to download {url} (Status {response.status_code})")
        except Exception as e:
            print(f"  ❌ Error {item['name']}: {e}")
            
        updated_items.append(item)
        
    with open(path, "w") as f:
        json.dump(updated_items, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # Disable DecompressionBombError for large images from Wikidata
    Image.MAX_IMAGE_PIXELS = None
    
    optimize_images("politicians.json", "politicians")
    optimize_images("movies.json", "movies")
    optimize_images("destinations.json", "destinations")
    optimize_images("songs.json", "songs") # Added songs just in case, though they usually have URLs
