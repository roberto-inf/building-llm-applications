## Guardrails for AI Agents: What, Why, and How

Guardrails are application-level controls that constrain an AI agent’s behavior to a defined scope and policy. They help ensure that an agent remains safe, relevant, compliant, and cost-efficient by preventing undesired actions or responses. In practice, guardrails can be rule-based, retrieval-based, or model-based (e.g., small LLM classifiers), and they can be enforced at different points in the agent lifecycle: before a model call, after a model call, inside routing logic, or around tool execution.

Common uses of guardrails in agentic systems include:
- Preventing out-of-scope queries (e.g., a travel assistant refusing finance questions)
- Enforcing safety and policy (e.g., content moderation, data leakage prevention)
- Reducing latency/cost by failing fast on invalid inputs
- Steering to well-defined outcomes (e.g., fallback to a refusal or a help message)

The LangGraph framework supports lightweight hooks for orchestration. We leverage its pre-model hook to inject guardrails before the LLM call and we augment the router node to short-circuit non-compliant requests. See related LangGraph guidance on agent structure and hooks: [Agent overview](https://langchain-ai.github.io/langgraph/agents/overview/?utm_source=chatgpt.com#visualize-an-agent-graph), and pre-model hook examples in [managing message history](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent-manage-message-history/#keep-the-original-message-history-unmodified).

---

## What we added to main_09_01.py

We introduced a travel-only guardrail that screens user queries before invoking an LLM or tools. If the query is not travel-related, the system immediately returns a polite refusal. The guardrail is applied in two places:

1) A pre-model hook on each ReAct agent so the agent model won’t be called for out-of-scope inputs
2) The router node, which can short-circuit to a refusal node instead of invoking any agent

### 1) Structured-output classifier for guardrails
We added a Pydantic model and a structured-output variant of the base LLM to classify whether a message is travel-related.

```python
class GuardrailDecision(BaseModel):
    is_travel: bool
    reason: str

GUARDRAIL_SYSTEM_PROMPT = (
    "You are a strict classifier... Travel-related queries include destinations, attractions, lodging..."
)

llm_guardrail = llm_model.with_structured_output(GuardrailDecision)
```

- Why structured output? Reliable, typed responses (“true/false + reason”) avoid brittle string parsing and minimize prompt engineering overhead.

### 2) A reusable refusal policy
We centralized the refusal text as a constant for consistency across the codebase.

```python
REFUSAL_INSTRUCTION = (
    "You can only help with travel-related questions ... Politely refuse and briefly explain ..."
)
```

### 3) Pre-model guardrail hook
We added a `pre_model_guardrail(state)` that classifies the last user message. If out of scope, it injects a refusal instruction in `llm_input_messages`, which LangGraph passes to the model instead of the raw history. This guarantees a polite refusal without tool calls.

```python
def pre_model_guardrail(state: dict):
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None
    if not isinstance(last_msg, HumanMessage):
        return {}

    decision = llm_guardrail.invoke([
        SystemMessage(content=GUARDRAIL_SYSTEM_PROMPT),
        HumanMessage(content=last_msg.content),
    ])

    if decision.is_travel:
        return {}

    # Inject refusal ahead of the original messages
    return {"llm_input_messages": [SystemMessage(content=REFUSAL_INSTRUCTION), *messages]}
```

We then attached this hook to both agents created via `create_react_agent`:

```python
travel_info_agent = create_react_agent(
    model=llm_model,
    tools=TOOLS,
    state_schema=AgentState,
    prompt="...",
    pre_model_hook=pre_model_guardrail,
)

accommodation_booking_agent = create_react_agent(
    model=llm_model,
    tools=BOOKING_TOOLS,
    state_schema=AgentState,
    prompt="...",
    pre_model_hook=pre_model_guardrail,
)
```

Effect: If the user’s latest message is not travel-related, the agent never triggers tool calls or a normal generation. The refusal is deliberate and low-latency.

### 4) Router-level guardrail and direct refusal path
We extended the `router_agent_node` to run the same classification before deciding which agent to call. On non-travel queries, we immediately attach a refusal `AIMessage` and route to a dedicated node that ends execution.

```python
from langchain_core.messages import AIMessage

# inside router_agent_node(...)
decision = llm_guardrail.invoke([
    SystemMessage(content=GUARDRAIL_SYSTEM_PROMPT),
    HumanMessage(content=user_input),
])
if not decision.is_travel:
    refusal_text = (
        "Sorry, I can only help with travel-related questions... "
        "Please rephrase your request to be travel-related."
    )
    return Command(
        update={"messages": [AIMessage(content=refusal_text)]},
        goto="guardrail_refusal",
    )
```

We added and wired the no-op node:

```python
def guardrail_refusal_node(state: AgentState):
    return {}

graph.add_node("guardrail_refusal", guardrail_refusal_node)
graph.add_edge("guardrail_refusal", END)
```

Effect: The router never calls an agent for non-travel queries; it returns a refusal immediately and ends the graph. This saves cost and keeps the UX crisp.

---

## Why both places?
- Router guardrail: fails fast before any agent logic or tool invocation.
- Agent pre-model hook: belt-and-suspenders to guard against any future code paths or reuse where the agent may be invoked directly outside the router.

This layered approach ensures robust scope control without adding significant latency.

---

## Extending this pattern
- Replace the binary classifier with a multi-label policy checker (safety, PII, compliance).
- Log guardrail decisions and reasons for auditing and evaluation.
- Make refusal dynamic: point to allowed topics, or offer to hand off to another service.
- Add human-in-the-loop when blocked topics might be escalated safely.

---

## References
- LangGraph agent overview (prebuilt components and hooks): [link](https://langchain-ai.github.io/langgraph/agents/overview/?utm_source=chatgpt.com#visualize-an-agent-graph)
- Managing message history with a pre-model hook (pattern reused for guardrails): [link](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent-manage-message-history/#keep-the-original-message-history-unmodified) 