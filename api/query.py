import yaml
from sentence_transformers import SentenceTransformer
from api.tools.retriever import Retriever
from api.graph import router, TaskType

# 載入設定
CFG = yaml.safe_load(open("configs/settings.yaml", "r"))

# 載入向量模型（文字／影像）
TXT_MODEL = SentenceTransformer(CFG["embed"]["text_model"]).eval()
IMG_MODEL = SentenceTransformer(CFG["embed"]["image_model"]).eval()

# 載入檢索器（FAISS 索引＋中繼資料）
RET = Retriever(CFG["storage"]["vector_dir"])

def encode_text(q: str):
    """將文字查詢轉為向量（單筆，已正規化）。"""
    return TXT_MODEL.encode([q], normalize_embeddings=True)

def ask(question: str) -> dict:
    """
    最小可行的問答流程：
    1) Router 判斷任務種類
    2) 依任務呼叫相應的檢索（目前皆先用文字檢索做示範）
    3) 回傳：計畫、前幾筆命中、中間答案占位（後續可接 LLM/VLM 與 citation）
    """
    plan = router(question)

    # 目前先以文字檢索為主；VISION_QA 之後可補 image-query 或 VLM
    hits = []
    try:
        if plan.task in [TaskType.TEXT_QA, TaskType.CALC, TaskType.VISION_QA, TaskType.TABLE_TO_CSV]:
            qv = encode_text(question)
            hits = RET.search_text(qv, k=CFG["retrieval"]["top_k_text"])
    except Exception as e:
        # 若索引不存在或讀取失敗，給出提示
        return {
            "plan": plan.model_dump(),
            "hits": [],
            "answer": f"[error] retrieval failed: {e}. 請先執行 ingestion：python api/ingest.py --docs data/docs --rebuild"
        }

    # 產生占位答案（之後可替換為 LLM/VLM 生成與 citation）
    top_note = ""
    if hits:
        meta = hits[0][1]
        # 盡量回傳來源線索，方便你檢查是否命中對的檔案與頁面
        origin = f"{meta.get('file', '')}"
        if "page" in meta:
            origin += f" (p.{meta['page']})"
        top_note = f"Top hit: {origin}"
    else:
        top_note = "目前沒有檢索命中，請確認已建立索引與資料。"

    answer = (
        "【占位答案】已完成最小檢索。下一步可接 LLM/VLM 生成內容與引用。\n"
        f"{top_note}"
    )

    return {
        "plan": plan.model_dump(),
        "hits": hits[:3],
        "answer": answer
    }

if __name__ == "__main__":
    # 簡易 CLI 測試
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
            print(f"- score={round(d, 4)} meta={m}")
        print("\n[Answer]")
        print(res["answer"])
        print("-" * 40)
