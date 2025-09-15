#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download a small pack of authoritative crypto PDFs for Vision-RAG testing.
Outputs to data/docs/papers/

- 若檔案已存在，預設不再重下載
- 可用 --force 強制覆蓋
"""
import argparse
from pathlib import Path
import urllib.request

OUT = Path("data/docs/papers")
OUT.mkdir(parents=True, exist_ok=True)

URLS = {
    "ethereum_yellow_paper.pdf": "https://ethereum.github.io/yellowpaper/paper.pdf",
    "uniswap_v3_whitepaper.pdf": "https://uniswap.org/whitepaper-v3.pdf",
    "chainlink_whitepaper_v1.pdf": "https://research.chain.link/whitepaper-v1.pdf",
    "eip1559_economic_analysis_roughgarden.pdf": "https://timroughgarden.org/papers/eip1559.pdf",
}

def fetch(url: str, dest: Path, force: bool = False):
    if dest.exists() and not force:
        print(f"[skip] {dest.name} already exists")
        return
    print(f"[download] {url} -> {dest.name}")
    urllib.request.urlretrieve(url, dest)

def main(force: bool = False):
    for name, url in URLS.items():
        dest = OUT / name
        try:
            fetch(url, dest, force)
        except Exception as e:
            print(f"  !! failed: {name}: {e}")
    print("[done] PDFs under data/docs/papers/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Force re-download and overwrite existing files")
    args = ap.parse_args()
    main(force=args.force)
