import argparse, os, json, yaml
from pathlib import Path
from sentence_transformers import SentenceTransformer
from PIL import Image
import pdfplumber
from pdf2image import convert_from_path
import faiss

# 簡化：文字 & 圖像雙索引
class VectorStore:
    def __init__(self, dim_text: int, dim_image: int, base_dir: str):
        self.text_index = faiss.IndexFlatIP(dim_text)
        self.image_index = faiss.IndexFlatIP(dim_image)
        self.text_meta = []
        self.image_meta = []
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def add_text(self, vec, meta):
        self.text_index.add(vec)
        self.text_meta.append(meta)

    def add_image(self, vec, meta):
        self.image_index.add(vec)
        self.image_meta.append(meta)

    def save(self):
        faiss.write_index(self.text_index, str(self.base_dir / "text.faiss"))
        faiss.write_index(self.image_index, str(self.base_dir / "image.faiss"))
        (self.base_dir / "text_meta.json").write_text(json.dumps(self.text_meta, ensure_ascii=False))
        (self.base_dir / "image_meta.json").write_text(json.dumps(self.image_meta, ensure_ascii=False))


def load_cfg():
    with open("configs/settings.yaml", "r") as f:
        return yaml.safe_load(f)


def parse_txt(path: Path):
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_docx(path: Path):
    from docx import Document
    return "\n".join(p.text for p in Document(path).paragraphs)


def parse_pdf_text(path: Path):
    texts = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            t = page.extract_text() or ""
            texts.append({"page": i, "text": t})
    return texts


def pdf_pages_to_images(path: Path):
    return convert_from_path(str(path))  # list[Image]


def chunk_text(txt: str, size=600, overlap=80):
    tokens = txt.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = " ".join(tokens[i:i + size])
        chunks.append(chunk)
        i += size - overlap
    return chunks


def main(docs_dir: str, rebuild: bool = False):
    cfg = load_cfg()
    vs = VectorStore(cfg["embed"]["dim_text"], cfg["embed"]["dim_image"], cfg["storage"]["vector_dir"])
    text_model = SentenceTransformer(cfg["embed"]["text_model"]).eval()
    img_model = SentenceTransformer(cfg["embed"]["image_model"]).eval()

    docs = list(Path(docs_dir).glob("**/*"))
    for fp in docs:
        if fp.suffix.lower() in [".txt", ".md"]:
            txt = parse_txt(fp)
            for ch in chunk_text(txt):
                emb = text_model.encode([ch], normalize_embeddings=True)
                vs.add_text(emb, {"type": "text", "file": str(fp.name)})
        elif fp.suffix.lower() == ".docx":
            txt = parse_docx(fp)
            for ch in chunk_text(txt):
                emb = text_model.encode([ch], normalize_embeddings=True)
                vs.add_text(emb, {"type": "docx", "file": str(fp.name)})
        elif fp.suffix.lower() == ".pdf":
            # text
            for rec in parse_pdf_text(fp):
                for ch in chunk_text(rec["text"] or ""):
                    emb = text_model.encode([ch], normalize_embeddings=True)
                    vs.add_text(emb, {"type": "pdf_text", "file": fp.name, "page": rec["page"]})
            # images (page-level)
            try:
                images = pdf_pages_to_images(fp)
                for idx, im in enumerate(images, start=1):
                    emb = img_model.encode([im], normalize_embeddings=True)
                    vs.add_image(emb, {"type": "pdf_image", "file": fp.name, "page": idx})
            except Exception:
                pass
        elif fp.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            im = Image.open(fp)
            emb = img_model.encode([im], normalize_embeddings=True)
            vs.add_image(emb, {"type": "image", "file": fp.name})

    vs.save()
    print("[Ingest] done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=str, default="data/docs")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    main(args.docs, args.rebuild)
