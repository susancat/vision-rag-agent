# app/app.py
import os, sys
import streamlit as st

# --- 確保能 import 專案根目錄下的 api/ ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from api.query import ask
except Exception as e:
    st.error(f"無法載入 api.query.ask：{e}\n請確認你是從專案根目錄執行或已建立 api/__init__.py")
    st.stop()

st.set_page_config(page_title="Vision-RAG Agent — 基礎版", layout="wide")
st.title("Vision-RAG Agent — 基礎版")
st.caption("多格式（txt/docx/pdf/img）最小 Agentic RAG 原型：Router → 檢索 →（占位回答）")

with st.sidebar:
    st.header("提示")
    st.markdown(
        """
1. 先建立索引：  
2. 測試檔放在 `data/docs/`
3. 範例問題：  
- *第 2 頁的圖表在描述什麼？*  
- *規格表中的尺寸有哪些，幫我整理成 CSV？*  
- *說明書提到的安全注意事項有哪些？*
     """
 )

q = st.text_input("輸入問題（可描述圖/表/文字）", placeholder="例如：請解釋第 3 頁示意圖中的資料流向")
ask_btn = st.button("送出", use_container_width=True)

if ask_btn and q:
 with st.spinner("思考中…"):
     try:
         res = ask(q)  # 期望回傳 dict: {plan:..., hits:..., answer:...}
     except Exception as e:
         st.error(f"ask() 執行失敗：{e}")
         st.stop()

 # 防呆：緩和缺欄位的情況
 plan = res.get("plan", {})
 hits = res.get("hits", [])
 answer = res.get("answer", "(暫無回答)")

 col1, col2 = st.columns([1, 1])
 with col1:
     st.subheader("計畫（Router / Plan）")
     st.json(plan)
 with col2:
     st.subheader("檢索命中（Top Hits）")
     if hits:
         st.json(hits)
     else:
         st.info("目前沒有命中，請確認已建立索引或換個問題再試。")

 st.subheader("回答（占位）")
 st.write(answer)
