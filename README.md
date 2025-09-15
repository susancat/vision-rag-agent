# Vision-RAG Agent 🚀

**Vision-RAG Agent** is an experimental project exploring how **Agentic AI** can enhance multi-modal retrieval (RAG, Retrieval-Augmented Generation).

### Current Capabilities
- 📄 **Multi-format ingestion**: TXT, DOCX, PDF (with optional OCR), CSV, JSON  
- 🖼 **Image retrieval**: Supports PNG/JPG, plus PDF-to-image embeddings  
- 🔎 **Vector search**: Built with `sentence-transformers` + `faiss`, supporting both text and image indices  
- ⚡ **Lightweight design**: Runs fully on local machine without cloud APIs, making it easy for prototyping and learning  

---

## 🛠 Tech Stack
- **Embedding**: `sentence-transformers` (MiniLM, CLIP)  
- **Vector DB**: `faiss`  
- **Document processing**: `pdfplumber`, `pdf2image`, `python-docx`, `pandas`  
- **OCR (optional)**: `pytesseract`  
- **Agent architecture**: prototype, extendable for tool-calling  

---

## 📂 Example Data
- `data/docs/markets/` → Crypto daily market data from CoinGecko / Binance (CSV)  
- `data/docs/papers/` → Key blockchain papers (Ethereum Yellow Paper, Chainlink, EIP-1559...)  

This allows the Agent to answer both **trend analysis questions** and **document knowledge questions**.  

## 🚀 How to Use

```bash
#setup
git clone https://github.com/yourname/vision-rag-agent.git
cd vision-rag-agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Blockchain whitepapers (skips if already downloaded)
python scripts/fetch_papers.py

# Crypto market data (CG recent year + Binance backfill)
python scripts/fetch_market_data.py --top 10 --vs usd --years 8
#Build Index
python api/ingest.py --docs data/docs --rebuild
#Start Interface
streamlit run app/app.py

#Daily Updates, when new data available
python scripts/fetch_market_data.py --top 10 --vs usd --years 8
python scripts/fetch_papers.py
python api/ingest.py --docs data/docs --rebuild

---

## 🌱 Why I Built This
- **Learn Agentic AI**: From classic RAG → agentive workflows  
- **Explore multi-modal**: Combine structured data (CSV) + text (PDF) + images  
- **PM/PO perspective**: As a Tech PM/PO, I want to better understand how AI can integrate into real workflows  

---

## 🔮 Next Steps
- **Tool Use** → Add calculation & data analysis, real-time API queries  
- **Workflow Integration** → Slack bot / Notion plugin for summaries & PRD drafts  
- **Metrics & ROI** → Track queries & tool usage, measure time saved  
- **Product Lifecycle Demos** → Show use cases in Discovery, Validation, Delivery  

---

## 💡 About Agentic AI
- Traditional RAG: retrieve → answer  
- Agentic RAG: AI not only answers but also **decides actions** (e.g. tool use, calculations, fetching data), and **integrates into workflows** (Slack, Notion).  

Vision-RAG Agent is just the **first step**. Future work will explore agents that can actively plan and act.  

---

👉 If you’re also interested in **Agentic AI + Multi-modal RAG**, stay tuned for updates!

**Vision-RAG Agent** 是一個正在開發中的實驗專案，目標是探索 **Agentic AI** 在多模態檢索 (RAG, Retrieval-Augmented Generation) 中的應用。

目前版本的核心能力：

- 📄 **多格式文件支援**：TXT、DOCX、PDF（含 OCR 備援）、CSV、JSON  
- 🖼 **圖像檢索**：支援 PNG/JPG 圖片，以及將 PDF 轉為頁面圖片嵌入  
- 🔎 **向量檢索**：基於 `sentence-transformers` + `faiss`，支援文字與圖像雙索引  
- ⚡ **輕量級設計**：完全本地可跑，不依賴雲端 API，方便快速原型與教學  

---

## 🛠 技術堆疊
- **Embedding**：`sentence-transformers`（MiniLM, CLIP）  
- **向量檢索**：`faiss`  
- **文件處理**：`pdfplumber`、`pdf2image`、`python-docx`、`pandas`  
- **OCR（可選）**：`pytesseract`  
- **Agent 架構**：目前為 prototype，已可擴展接入 tool calling  

---

## 📂 資料示例
- `data/docs/markets/` → 從 CoinGecko / Binance 擷取的加密貨幣日線數據 (CSV)  
- `data/docs/papers/` → 區塊鏈白皮書 (Ethereum Yellow Paper, Chainlink, EIP-1559...)  

這些內容讓 Agent 同時能回答 **「數據趨勢問題」** 與 **「研究文件問題」**。  

---

## 🌱 為什麼做這個？
- **學習 Agentic AI**：理解從單純 RAG → Agentic Workflow 的演進  
- **探索多模態應用**：結合結構化數據 (CSV) + 文本 (PDF) + 圖像  
- **產品管理思維**：作為 Tech PM/PO，我希望能更好地理解 AI 工具如何融入工作流  

---

## 🔮 下一步擴展方向
- **Tool Use 增強** → 自動計算與數據分析、即時 API 抓取  
- **Workflow Integration** → Slack bot / Notion 插件，自動摘要與需求文檔生成  
- **效益追蹤與 ROI** → 查詢/工具使用統計，量化 AI 節省的時間  
- **產品週期案例 Demo** → 探索、驗證、交付各階段的應用  

---

## 💡 關於 Agentic AI
- 傳統 RAG：回答問題前，檢索知識庫 → 回答  
- Agentic RAG：AI 不只回答，還能 **決定行動**（用工具、做計算、取資料），並能 **融入工作流**（如 Slack、Notion）  

Vision-RAG Agent 目前是 **第一步**，後續會探索「能主動規劃與行動」的 Agent。  

---

👉 如果你也對 **Agentic AI + 多模態 RAG** 有興趣，歡迎關注後續更新！