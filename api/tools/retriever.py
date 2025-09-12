import json, faiss
from pathlib import Path

class Retriever:
    def __init__(self, base_dir: str):
        self.base = Path(base_dir)
        self.text_idx = faiss.read_index(str(self.base / "text.faiss"))
        self.img_idx = faiss.read_index(str(self.base / "image.faiss"))
        self.text_meta = json.loads((self.base / "text_meta.json").read_text())
        self.img_meta = json.loads((self.base / "image_meta.json").read_text())

    def search_text(self, q_vec, k=5):
        D, I = self.text_idx.search(q_vec, k)
        return [(float(D[0][i]), self.text_meta[I[0][i]]) for i in range(len(I[0]))]

    def search_image(self, q_vec, k=5):
        D, I = self.img_idx.search(q_vec, k)
        return [(float(D[0][i]), self.img_meta[I[0][i]]) for i in range(len(I[0]))]
