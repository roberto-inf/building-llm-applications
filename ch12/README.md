# UK Travel Assistant – LangGraph (ToolNode pattern)

This project is an example of the **"Add tools"** LangGraph tutorial [§4 Define the graph](https://langchain-ai.github.io/langgraph/tutorials/get-started/2-add-tools/#4-define-the-graph) applied to a travel assistant.

The assistant can answer questions about WikiVoyage pages for:

* Cornwall
* North Cornwall
* South Cornwall
* West Cornwall

It relies on a **single tool** (`search_travel_info`) and the standard pre-built components `tools_condition` + custom `CustomToolNode`.

---

## Setup (PowerShell)

```powershell
# 1 · Virtual environment
python -m venv env_ch12
.\env_ch12\Scripts\Activate.ps1

# 2 · Dependencies
pip install -r requirements.txt

# 3 · OpenAI key (session-only)
$Env:OPENAI_API_KEY = "sk-..."

# 4 · Run
python main.py
```

---

### Internals

* **Vector store:** Pages fetched with `AsyncHtmlLoader`, chunked and embedded with `OpenAIEmbeddings`, stored in **Chroma**.
* **Tool:** `search_travel_info` performs similarity search and returns top chunks.
* **LangGraph:**
  * `chatbot` node -> LLM (may emit tool_calls).
  * `tools` node -> `CustomToolNode` executes those calls.
  * `tools_condition` routes between them.
* **Loop:** After each tool call, control returns to the LLM until a final answer is produced. 