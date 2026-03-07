# Agentic RAG Chatbot

Replaces manual resume screening with an **Agentic RAG chatbot** built on a **ReAct architecture**: PDF resumes become a **digital resume** recruiters can chat with—asking questions in plain language and getting answers from the document, with the option to open profiles through **Model Context Protocol (MCP) Playwright** integration. Quality is measured by **tool-call accuracy**, **context relevance**, **precision/recall**, **faithfulness**, and **answer relevance**.

---

## What It Does

- **Business problem:** Reduces manual resume review by turning PDF resumes into a queryable, conversational interface.
- **Recruiter flow:** Upload a resume PDF → chat in natural language (e.g. “What is Nischitha’s email?”, “What are her technical skills?”, “Open her LinkedIn profile”) → get answers from the document and optionally open LinkedIn/GitHub via the browser.
- **Tech:** Agentic RAG (retriever + LLM) with a ReAct-style agent (LangGraph) that decides when to call tools; MCP (Playwright) for browser actions; evaluation metrics for accuracy.

---

## Architecture (High Level)

1. **Ingest:** PDF resume → `PyPDFLoader` → `RecursiveCharacterTextSplitter` → **FAISS** vector store (HuggingFace embeddings).
2. **Retriever tool:** `retriever_vector_db_resume` searches the vector store and returns relevant chunks.
3. **Agent (ReAct):** LangGraph agent (Groq LLM) with tools: resume retriever + MCP Playwright tools (e.g. `browser_navigate`). Agent chooses when to retrieve and when to open URLs.
4. **Graph flow:** `START → agent ⇄ retrieve → agent → … → END` (retrieve loops back to agent so it can chain retriever + browser).
5. **Evaluation:** Test set of questions; metrics: Tool Call Accuracy, Tool Context Relevance, Context Precision/Recall, Faithfulness, Answer Relevance.

---

## Prerequisites

- **Python** 3.10+
- **Resume PDF** at `data/pdf/` (e.g. `data/pdf/Nischitha.D.pdf`)
- **API keys** in `.env` (see below)

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Required for LLM and agent
GROQ_API_KEY=your_groq_api_key

# Optional
ELEVENLABS_API_KEY=your_elevenlabs_key
LANGCHAIN_API_KEY=your_langchain_key
```

- **GROQ_API_KEY:** [Groq Console](https://console.groq.com/keys) — used for the chat model (`openai/gpt-oss-120b` or similar).
- **LANGCHAIN_API_KEY:** Optional, for LangSmith tracing.

---

## Setup

1. **Clone / open the project** and go to the project root.

2. **Create a virtual environment and install dependencies:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Add your resume PDF** under `data/pdf/` (e.g. `data/pdf/YourResume.pdf`) and, if needed, update the path in the notebook (search for `PyPDFLoader("data/pdf/`).

4. **MCP (Playwright):** The notebook uses `langchain-mcp-adapters` and the Playwright MCP server. Ensure `@playwright/mcp` is available (e.g. `npx @playwright/mcp` or as configured in the notebook). Install Playwright browsers if required: `playwright install`.

5. **Optional:** Install `nest_asyncio` if you hit event-loop issues with async MCP tools in the notebook:  
   `pip install nest_asyncio`

---

## How to Run

1. Open **`chat.ipynb`** in Jupyter or VS Code.
2. Run cells **top to bottom**:
   - **Cells 1–2:** Env check and imports.
   - **PDF + vector store:** Load PDF, split, build FAISS index and retriever.
   - **Retriever tool:** Create `retriever_tool_resume`.
   - **MCP section:** Start MCP client, load Playwright tools, build `tools_mcp` (retriever + browser tools).
   - **Agent + graph:** Define agent, nodes, edges, then `workflow.compile()` → `graph`.
3. **Invoke the chatbot:**

   ```python
   from langchain_core.messages import HumanMessage
   result = graph.invoke({"messages": [HumanMessage(content="What is the email of the candidate?")]})
   # or
   result = graph.invoke({"messages": [HumanMessage(content="Open the LinkedIn profile of the candidate")]})
   ```

4. **Evaluation:** Run the “Agentic RAG Evaluation” cells to compute mean scores (Tool Call Accuracy, Context Relevance, Precision/Recall, Faithfulness, Answer Relevance).

---

## Main Dependencies (from requirements.txt)

- `langchain`, `langgraph`, `langchain-community`, `langchain-core`, `langchain-groq`
- `langchain-mcp-adapters`, `mcp`
- `faiss-cpu`, `sentence-transformers`
- `pypdf`, `python-dotenv`, `pydantic`
- `gradio`, `langgraph-cli` (optional)

---

## Evaluation Metrics (Summary)

| Metric                  | Meaning |
|-------------------------|--------|
| **Tool Call Accuracy**  | Correct tool(s) invoked with valid parameters. |
| **Tool Context Relevance** | Retrieved text is relevant to the question. |
| **Context Precision**   | Proportion of retrieved content that is relevant. |
| **Context Recall**      | Whether all needed information was retrieved. |
| **Faithfulness**       | Answer is grounded in retrieved docs (no hallucination). |
| **Answer Relevance**    | Final answer addresses the user’s question. |

Overall score is the mean of these metrics (e.g. 0.98 = 98%).

---

## Notes

- **Groq TPM limits:** If you see “Request too large” or token limits, the notebook truncates long tool (retriever) context before calling the model to stay within limits.
- **Input format:** Prefer `graph.invoke({"messages": [HumanMessage(content="…")]})` over passing a raw string for `messages`.
- **Browser/MCP:** Opening LinkedIn/GitHub requires the MCP Playwright server and a supported environment (e.g. desktop or configured browser).

---

## File Reference

| Item            | Purpose |
|-----------------|--------|
| `chat.ipynb`    | Main notebook: data load, retriever, agent, graph, MCP, evaluation. |
| `data/pdf/`     | Directory for resume PDF(s). |
| `.env`          | API keys (Groq, optional: ElevenLabs, LangChain). |
| `requirements.txt` | Python dependencies. |
