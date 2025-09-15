#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision-RAG Agent — Ingestion Pipeline (text / docx / pdf / image / csv)

功能：
- 讀取多種格式，抽取文本與圖像特徵，建立文字/圖像向量索引（FAISS）
- PDF：文字抽取（pdfplumber）+ 可選 OCR 備援（pytesseract），並將每頁轉圖以建立圖像向量
- CSV（CoinGecko / Binance / 合併後 markets_combined）：切成 30天/塊，寫入來源 metadata（source_set / origin_dir）

使用：
  python api/ingest.py --docs data/docs --rebuild
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import yaml
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image

# 可選依賴：pdf/docx/csv
import pdfplumber
from pdf2image import convert_from_path
from docx import Document
import pandas as pd

# 可選 OCR（若 settings.yaml ocr.enabled = true）
try:
    import pytesseract  # 需要本機有 tesseract 可執行檔
except Exception:
    pytesseract = None


# ----------------------------
# 工具類：向量庫（FAISS + metadata）
# ----------------------------
class VectorStore:
    def __init__(self, dim_text: int, dim_image: int, base_dir: str):
        self.dim_text = dim_text
        self.dim_image = dim_image

        self.text_index = faiss.IndexFlatIP(dim_text)
        self.image_index = faiss.IndexFlatIP(dim_image)

        self.text_meta: List[Dict] = []
        self.image_meta: List[Dict] = []

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def add_text(self, vec: np.ndarray, meta: Dict):
        # vec: shape (1, dim_text)
        self.text_index.add(vec.astype(np.float32))
        self.text_meta.append(meta)

    def add_image(self, vec: np.ndarray, meta: Dict):
        # vec: shape (1, dim_image)
        self.image_index.add(vec.astype(np.float32))
        self.image_meta.append(meta)

    def save(self):
        faiss.write_index(self.text_index, str(self.base_dir / "text.faiss"))
        faiss.write_index(self.image_index, str(self.base_dir / "image.faiss"))
        (self.base_dir / "text_meta.json").write_text(
            json.dumps(self.text_meta, ensure_ascii=False), encoding="utf-8"
        )
        (self.base_dir / "image_meta.json").write_text(
            json.dumps(self.image_meta, ensure_ascii=False), encoding="utf-8"
        )


# ----------------------------
# 設定與通用工具
# ----------------------------
def load_cfg():
    """
    依序尋找：
    - <project_root>/configs/settings.yaml
    - <project_root>/configs/settings.example.yaml
    找不到就回傳內建預設。
    """
    from pathlib import Path
    import yaml

    script_dir = Path(__file__).resolve().parent     # api/
    project_root = script_dir.parent                 # 專案根目錄
    candidates = [
        project_root / "configs" / "settings.yaml",
        project_root / "configs" / "settings.example.yaml",
        Path("configs/settings.yaml"),               # 萬一在根目錄執行
        Path("configs/settings.example.yaml"),
    ]
    for p in candidates:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if cfg:
                    return cfg

    # 內建預設（找不到檔案時也可跑）
    return {
        "embed": {
            "text_model": "sentence-transformers/all-MiniLM-L6-v2",
            "image_model": "clip-ViT-B-32",
            "dim_text": 384,
            "dim_image": 512,
        },
        "ocr": {"enabled": False, "lang": "eng"},
        "retrieval": {"top_k_text": 5, "top_k_image": 4, "fuse": "rrf"},
        "storage": {"vector_dir": "data/embeddings"},
        "ui": {"show_traces": True},
    }

def chunk_text(txt: str, size: int = 600, overlap: int = 80) -> List[str]:
    tokens = txt.split()
    chunks, i = [], 0
    while i < len(tokens):
        chunk = " ".join(tokens[i : i + size])
        if chunk.strip():
            chunks.append(chunk)
        i += max(size - overlap, 1)
    return chunks


# ----------------------------
# 解析：文字 / DOCX
# ----------------------------
def parse_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


# ----------------------------
# 解析：PDF（文字 + 圖像 + 可選 OCR 備援）
# ----------------------------
def parse_pdf_text(path: Path) -> List[Dict]:
    """
    回傳：List[{"page": int, "text": str}]
    """
    out = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            t = page.extract_text() or ""
            out.append({"page": i, "text": t})
    return out


def pdf_pages_to_images(path: Path) -> List[Image.Image]:
    """
    將 PDF 每頁轉為 PIL Image 列表
    需要系統已安裝 poppler（pdf2image 依賴）
    """
    return convert_from_path(str(path))


def ocr_image_to_text(img: Image.Image, lang: str = "eng") -> str:
    if pytesseract is None:
        return ""
    try:
        return pytesseract.image_to_string(img, lang=lang) or ""
    except Exception:
        return ""


# ----------------------------
# 解析：CSV（CoinGecko / Binance / 合併）
# ----------------------------
def ingest_csv_file(
    fp: Path,
    text_model: SentenceTransformer,
    vs: VectorStore,
):
    """
    期待欄位（盡量標準化）：date, price, market_cap, total_volume
    可選欄位：source（coingecko / binance / mixed）
    切塊規則：每 30 行為一塊，metadata 帶 source_set + origin_dir
    """
    origin_dir = fp.parent.name  # markets / markets_binance / markets_combined / ...
    try:
        df = pd.read_csv(fp)
        # 標準化欄位名到小寫
        df.columns = [str(c).strip().lower() for c in df.columns]
        # 確保必要欄位存在
        for col in ["date", "price", "market_cap", "total_volume"]:
            if col not in df.columns:
                df[col] = ""

        # 判定來源
        has_src_col = "source" in df.columns
        default_source = (
            "coingecko"
            if origin_dir in {"markets"}
            else "binance"
            if origin_dir in {"markets_binance"}
            else "mixed"
        )

        rows_txt: List[str] = []
        srcs: List[str] = []

        for _, r in df.iterrows():
            date = str(r.get("date", ""))
            price = r.get("price", "")
            mcap = r.get("market_cap", "")
            vol = r.get("total_volume", "")
            src = str(r.get("source")) if has_src_col else default_source
            src = src if src else default_source

            line = f"{fp.stem} on {date}: price={price}, market_cap={mcap}, volume={vol} (src={src})"
            rows_txt.append(line)
            srcs.append(src)

        # 每 30 行為一塊
        for i in range(0, len(rows_txt), 30):
            ch = "\n".join(rows_txt[i : i + 30])
            emb = text_model.encode([ch], normalize_embeddings=True)
            source_set = sorted(set(srcs[i : i + 30]))
            vs.add_text(
                emb,
                {
                    "type": "csv",
                    "file": fp.name,
                    "source_set": source_set,  # e.g., ["binance"] 或 ["coingecko","binance"]
                    "origin_dir": origin_dir,  # e.g., markets_combined
                },
            )
    except Exception as e:
        print(f"[warn] CSV ingest failed for {fp.name}: {e}")


# ----------------------------
# 主流程
# ----------------------------
def main(docs_dir: str, rebuild: bool = False):
    cfg = load_cfg()

    # 初始化向量庫
    vs = VectorStore(cfg["embed"]["dim_text"], cfg["embed"]["dim_image"], cfg["storage"]["vector_dir"])

    # 向量模型
    text_model = SentenceTransformer(cfg["embed"]["text_model"]).eval()
    img_model = SentenceTransformer(cfg["embed"]["image_model"]).eval()

    # OCR 設定
    ocr_enabled = bool(cfg.get("ocr", {}).get("enabled", False))
    ocr_lang = str(cfg.get("ocr", {}).get("lang", "eng"))

    # 掃描資料
    docs = sorted(Path(docs_dir).glob("**/*"))

    for fp in docs:
        suffix = fp.suffix.lower()

        # ---------- 純文字 ----------
        if suffix in [".txt", ".md"]:
            txt = parse_txt(fp)
            for ch in chunk_text(txt):
                emb = text_model.encode([ch], normalize_embeddings=True)
                vs.add_text(emb, {"type": "text", "file": fp.name})

        # ---------- DOCX ----------
        elif suffix == ".docx":
            txt = parse_docx(fp)
            for ch in chunk_text(txt):
                emb = text_model.encode([ch], normalize_embeddings=True)
                vs.add_text(emb, {"type": "docx", "file": fp.name})

        # ---------- PDF ----------
        elif suffix == ".pdf":
            # 1) 文字抽取（pdfplumber）
            text_pages = parse_pdf_text(fp)
            for rec in text_pages:
                page_no = rec["page"]
                txt = rec["text"] or ""
                if txt.strip():
                    for ch in chunk_text(txt):
                        emb = text_model.encode([ch], normalize_embeddings=True)
                        vs.add_text(
                            emb, {"type": "pdf_text", "file": fp.name, "page": page_no}
                        )

            # 2) 轉頁為圖片 → 建立圖像向量；若啟用 OCR，額外把該頁 OCR 文本也塞進文字索引
            try:
                images = pdf_pages_to_images(fp)
            except Exception as e:
                print(f"[warn] pdf2image failed for {fp.name}: {e}")
                images = []

            for idx, im in enumerate(images, start=1):
                try:
                    emb_img = img_model.encode([im], normalize_embeddings=True)
                    vs.add_image(
                        emb_img,
                        {"type": "pdf_image", "file": fp.name, "page": idx},
                    )
                except Exception as e:
                    print(f"[warn] image embedding failed ({fp.name} p.{idx}): {e}")

                # 可選 OCR：補充掃描型 PDF
                if ocr_enabled and pytesseract is not None:
                    try:
                        ocr_txt = ocr_image_to_text(im, lang=ocr_lang)
                        for ch in chunk_text(ocr_txt):
                            emb = text_model.encode([ch], normalize_embeddings=True)
                            vs.add_text(
                                emb,
                                {
                                    "type": "pdf_ocr",
                                    "file": fp.name,
                                    "page": idx,
                                    "lang": ocr_lang,
                                },
                            )
                    except Exception as e:
                        print(f"[warn] OCR failed ({fp.name} p.{idx}): {e}")

        # ---------- 圖片 ----------
        elif suffix in [".png", ".jpg", ".jpeg"]:
            try:
                im = Image.open(fp).convert("RGB")
                emb = img_model.encode([im], normalize_embeddings=True)
                vs.add_image(emb, {"type": "image", "file": fp.name})
            except Exception as e:
                print(f"[warn] image ingest failed for {fp.name}: {e}")

        # ---------- CSV（行情/合併資料） ----------
        elif suffix == ".csv":
            ingest_csv_file(fp, text_model, vs)

        # ---------- 其他格式忽略 ----------
        else:
            continue

    # 寫檔
    vs.save()
    print("[Ingest] done. Index saved to", vs.base_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=str, default="data/docs", help="Directory to ingest")
    ap.add_argument("--rebuild", action="store_true", help="Reserved for compatibility (overwrite indexes)")
    args = ap.parse_args()
    main(args.docs, args.rebuild)
