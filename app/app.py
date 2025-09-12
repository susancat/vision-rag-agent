import streamlit as st
from api.query import ask

st.set_page_config(page_title="Vision-RAG Agent — 基礎版", layout="wide")
st.title("Vision-RAG Agent — 基礎版")
st.caption("多格式（txt/docx/pdf/img）最小 Agentic RAG 原型：Router → 檢索 →（占位回答）")

with st.sidebar:
    st.header("提示")
    st.markdown(
        """
        1. 請先執行 ingestion 建索引：  
           `python api/ingest.py --docs data/docs --rebuild`
        2. 將測試檔放在 `data/docs/`
        3. 在下方輸入問題，例如：  
           - *第 2 頁的圖表在描述什麼？*  
           - *規格表中的尺寸有哪些，幫我整理成 CSV？*  
           - *說明書提到的安全注意事項有哪些？*
        """
    )

q = st.text_input("輸入問題（可描述圖/表/文字）", placeholder="例如：請解釋第 3 頁示意圖中的資料流向")
ask_btn = st.button("送出", use_container_width=True)

if ask_btn and q:
    with st.spinner("思考中…"):
        res = ask(q)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("計畫（Router / Plan）")
        st.json(res["plan"])
    with col2:
        st.subheader("檢索命中（Top Hits）")
        if res["hits"]:
            st.json(res["hits"])
        else:
            st.info("目前沒有命中，請確認已建立索引或換個問題再試。")

    st.subheader("回答（占位）")
    st.write(res["answer"])
