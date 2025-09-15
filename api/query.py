# api/query.py
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import yaml
import numpy as np
from sentence_transformers import SentenceTransformer

# 讓此模組不依賴執行路徑：推導專案根目錄
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent  # .../vision-rag-agent
CFG_CANDIDATES = [
    PROJECT_ROOT / "configs" / "settings.yaml",
    PROJECT_ROOT / "configs" / "settings.example.yaml",
]

# ---------------- 配置載入（含預設） ----------------
@lru_cache(maxsize=1)
def load_cfg() -> Dict[str, Any]:
    for p in CFG_CANDIDATES:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return cfg
    # 內建預設，讓最小版也能跑
    return {
        "embed": {
            "text_model": "sentence-transformers/all-MiniLM-L6-v2",
            "image_model": "clip-ViT-B-32",
            "dim_text": 384,
            "dim_image": 512,
        },
        "retrieval": {"top_k_text": 5},
        "storage": {"vector_dir": "data/embeddings"},
    }

# ---------------- 模型懶載入 ----------------
@lru_cache(maxsize=1)
def get_txt_model() -> SentenceTransformer:
    cfg = load_cfg()
    m = SentenceTransformer(cfg["embed"]["text_model"])
    m.eval()
    return m

@lru_cache(maxsize=1)
def get_img_model() -> SentenceTransformer:
    cfg = load_cfg()
    m = SentenceTransformer(cfg["embed"]["image_model"])
    m.eval()
    return m

@lru_cache(maxsize=1)
def get_retriever():
    # 延遲載入，避免匯入時就炸
    from api.tools.retriever import Retriever
    cfg = load_cfg()
    vecdir = cfg["storage"]["vector_dir"]
    vs_path = PROJECT_ROOT / vecdir
    if not vs_path.exists():
        # 讓呼叫端收到更易懂的錯誤
        raise FileNotFoundError(
            f"Vector index not found at '{vs_path}'. "
            f"請先建立索引：python api/ingest.py --docs data/docs --rebuild"
        )
    return Retriever(str(vs_path))

def encode_text(q: str) -> np.ndarray:
    """將文字查詢轉為 (1, dim) 的向量（已正規化）。"""
    model = get_txt_model()
    return model.encode([q], normalize_embeddings=True)

# ---------------- Router 與 TaskType ----------------
from api.graph import router, TaskType

# ---------------- 市場資料過濾輔助 ----------------
_COIN_ALIASES = {
    "eth": "ethereum",
    "ethereum": "ethereum",
    "btc": "bitcoin",
    "bitcoin": "bitcoin",
    "sol": "solana",
    "solana": "solana",
    "bnb": "binancecoin",
    "xrp": "ripple",
    "ada": "cardano",
    "doge": "dogecoin",
    "ton": "toncoin",
    "trx": "tron",
    "avax": "avalanche-2",
}

def _extract_coin_id(q: str) -> Optional[str]:
    ql = q.lower()
    for k, cid in _COIN_ALIASES.items():
        if k in ql:
            return cid
    return None

def _is_market_meta(meta: dict) -> bool:
    """是否為行情 CSV 來源（markets / markets_binance / markets_combined）。"""
    od = (meta.get("origin_dir") or "").lower()
    t = (meta.get("type") or "").lower()
    return t == "csv" or od in {"markets", "markets_binance", "markets_combined"}

def _wants_market_only(q: str) -> bool:
    kw = [
        "價格", "走勢", "均價", "均線", "移動平均", "ma", "成交量",
        "price", "trend", "volume", "moving average", "volatility"
    ]
    ql = q.lower()
    return any(k in ql for k in kw)

def _predicate_for_question(q: str):
    """
    回傳 predicate(meta)：
    - 若問題像是市場分析：先限制為 CSV 行情來源
    - 若同時提到幣別：再限制檔名 == <coin_id>.csv（避免命中到別的幣）
    """
    wants_market = _wants_market_only(q)
    coin_id = _extract_coin_id(q)

    if not wants_market and not coin_id:
        return None  # 不加限制

    def pred(meta: dict) -> bool:
        if not _is_market_meta(meta):
            return False
        if coin_id:
            fname = (meta.get("file") or "").lower()
            return fname == f"{coin_id}.csv"
        return True

    return pred

# ---------------- 主函式：ask() ----------------
def ask(question: str) -> Dict[str, Any]:
    """
    最小可行的問答流程：
    1) Router 判斷任務種類
    2) 依任務呼叫相應的檢索（目前先用文字檢索示範）
    3) 回傳：計畫、前幾筆命中、中間答案占位（後續可接 LLM/VLM 與 citation）
    """
    cfg = load_cfg()
    plan = router(question)

    # 目前先以文字檢索為主；VISION_QA 未來可改走影像/混合流程
    hits: List[Tuple[float, Dict[str, Any]]] = []
    try:
        if plan.task in [TaskType.TEXT_QA, TaskType.CALC, TaskType.VISION_QA, TaskType.TABLE_TO_CSV]:
            # 小技巧：若偵測到幣別，將其附加到查詢文字，提升語意命中
            cid = _extract_coin_id(question)
            q_for_embed = f"{question} [coin:{cid}]" if cid else question

            qv = encode_text(q_for_embed)  # (1, dim)
            RET = get_retriever()

            # 先取大一些的 k，再做本地 predicate 過濾，提高穩定度
            predicate = _predicate_for_question(question)
            raw_hits = RET.search_text(qv, k=max(20, int(cfg["retrieval"]["top_k_text"])))
            if predicate:
                raw_hits = [h for h in raw_hits if predicate(h[1])]

            k_final = int(cfg["retrieval"]["top_k_text"])
            hits = raw_hits[:k_final]

    except FileNotFoundError as e:
        # 索引未建立
        return {
            "plan": plan.model_dump(),
            "hits": [],
            "answer": f"[error] {e}"
        }
    except Exception as e:
        return {
            "plan": plan.model_dump(),
            "hits": [],
            "answer": f"[error] retrieval failed: {e}。請確認已建立索引與 settings.yaml"
        }

    # 產生占位答案（之後可替換為 LLM/VLM 生成與 citation）
    if hits:
        meta = hits[0][1] or {}
        origin = f"{meta.get('file', '')}"
        if "page" in meta:
            origin += f" (p.{meta['page']})"
        top_note = f"Top hit: {origin}"
    else:
        top_note = "目前沒有檢索命中，請確認已建立索引或換個問題再試。"

    answer = (
        "【占位答案】已完成最小檢索。下一步可接 LLM/VLM 生成內容與引用。\n"
        f"{top_note}"
    )

    return {
        "plan": plan.model_dump(),
        "hits": hits[:3],
        "answer": answer
    }

# -------------- CLI quick test --------------
if __name__ == "__main__":
    print("Vision-RAG Agent (minimal) — 請輸入問題，Ctrl+C 離開。")
    while True:
        q = input("Q> ").strip()
        if not q:
            continue
        res = ask(q)
        print("\n[Plan]")
        print(res["plan"])
        print("\n[Top Hits]")
        for d, m in res["hits"]:
            print(f"- score={round(float(d), 4)} meta={m}")
        print("\n[Answer]")
        print(res["answer"])
        print("-" * 40)
