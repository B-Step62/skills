# MLflow Tracing - Python Guide

## Contents
- Quick Start
- Instrumentation Methods (AutoLogging, Decorator, Manual Spans)
- Selecting and Combining Autolog Integrations
- User/Session Tracking
- Combining AutoLogging with Custom Tracing
- Common Issues

---

## Quick Start

### Install and Configure

```bash
pip install mlflow>=3.8.0
```

**Check if `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_ID` are already set in the environment.** If both are set, skip the configuration below — MLflow will use them automatically.

Only call these if the environment is NOT pre-configured:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")  # skip if MLFLOW_TRACKING_URI is set
mlflow.set_experiment("my-agent")                 # skip if MLFLOW_EXPERIMENT_ID is set
```

> **On Databricks: the experiment name must be an absolute workspace path.**
> `mlflow.set_experiment("my-agent")` is rejected or lands in the wrong place. Use
> `mlflow.set_experiment("/Users/<your-email>/my-agent")` or, to attach to an existing
> experiment, look up its numeric ID in the UI and pass `set_experiment(experiment_id="<id>")`.
>
> **On Databricks: opt into UC trace storage**<br>
> Traces land in the experiment backend by default (capped at 100,000 per experiment). For production use, bind the experiment to a UC trace location when calling `set_experiment`. See [`databricks.md`](databricks.md) for the one-liner.

### Enable Tracing

**For supported frameworks** (LangChain, LangGraph, OpenAI, etc.):

```python
mlflow.langchain.autolog()  # LangChain / LangGraph orchestration
# Add other integrations used by the app when they add needed coverage or detail.
# See "Selecting and Combining Autolog Integrations" below.
```

**For custom code:**

```python
from mlflow.entities import SpanType

@mlflow.trace(span_type=SpanType.CHAIN)
def my_function(query: str) -> str:
    # Your code here
    return result
```

---

## Instrumentation Methods

### Method 1: AutoLogging (Recommended for Frameworks)

Zero-code instrumentation for supported libraries. See [Automatic Tracing](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/automatic.md) for supported integrations and a multi-framework example.

```python
import mlflow

# Enable the integrations needed for the app's actual execution path.
mlflow.langchain.autolog()  # LangChain and LangGraph
```

### Selecting and Combining Autolog Integrations

Inspect the application's imports, dependencies, and call sites to identify both the orchestration framework and the SDKs actually invoked. Choose integrations by that execution path, rather than by the model provider's name. Multiple autolog calls are appropriate when they contribute needed coverage or detail; there is no one-integration limit. Do not enable every integration speculatively.

- **Framework integration**: `mlflow.langchain.autolog()` captures LangChain / LangGraph execution, nodes, tools, and model calls. LangGraph uses this integration; there is no `mlflow.langgraph.autolog()`.
- **Provider SDK integration**: add `mlflow.openai.autolog()`, `mlflow.anthropic.autolog()`, or another matching integration for SDK operations the framework does not separately trace (such as a direct embeddings SDK call inside a tool), or to capture additional SDK request/response details. OpenAI-compatible endpoints are traced by OpenAI autolog when calls actually use the OpenAI SDK.
- **Framework coverage is already sufficient**: keep the framework integration alone when the additional SDK integration would not add useful information.

**ChatDatabricks example:** OpenAI-backed `databricks_langchain.ChatDatabricks` versions, such as [v0.21.0](https://github.com/databricks/databricks-ai-bridge/blob/databricks-langchain-v0.21.0/integrations/langchain/src/databricks_langchain/chat_models.py), use the OpenAI SDK internally. When SDK details are useful, combine LangChain and OpenAI autologging:

```python
import mlflow
from databricks_langchain import ChatDatabricks

# Enable before invoking the model or graph.
mlflow.langchain.autolog()  # Workflow, tools, and LangChain model spans
mlflow.openai.autolog()    # Underlying OpenAI SDK requests to Databricks

llm = ChatDatabricks(endpoint="<databricks-serving-endpoint>")
response = llm.invoke("What is MLflow tracing?")
```

Check the installed wrapper's version and client implementation before adding SDK autologging. Older or alternative wrappers may use a different client. Check integration compatibility with the installed MLflow and library versions. LangChain autologging already captures ChatDatabricks model calls; OpenAI autologging adds SDK detail and covers direct OpenAI SDK calls in the same app. See [Tracing Databricks](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/databricks.md) for OpenAI SDK calls to Databricks endpoints.

**Verify combined tracing:** run a representative request, flush with `mlflow.flush_trace_async_logging()`, and inspect the new trace's span tree, inputs/outputs, and usage. Confirm the needed operations share a connected trace. A framework model span containing an SDK model span can describe the same physical request at two layers; count provider requests separately from spans. MLflow's built-in trace token totals avoid counting usage again on descendants of a span with usage ([aggregation in MLflow 3.8.0](https://github.com/mlflow/mlflow/blob/v3.8.0/mlflow/tracing/utils/__init__.py#L179-L225)), but a custom sum over every model span can double count it. Correct parentage is required for this accounting; unrelated roots are not deduplicated by it. Check async, streaming, and batch paths if the app uses them.

**Do not decorate framework operations to recreate automatic spans.** Adding `@mlflow.trace` to nodes, tools, or model calls already captured by the framework produces redundant spans. Add a manual span only for app-specific work the framework does not capture, such as custom retrieval or an outer application boundary (see "Combining AutoLogging with Custom Tracing" below).

### Method 2: Decorator (Recommended for Custom Code)

**Prefer decorator over manual spans** - it auto-captures function name, inputs, and outputs.

```python
from mlflow.entities import SpanType

@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_documents(query: str) -> list[str]:
    return documents

@mlflow.trace(span_type=SpanType.TOOL)
def search_database(sql: str) -> dict:
    return results
```

**Span types**: `LLM`, `CHAIN`, `TOOL`, `AGENT`, `RETRIEVER`, `EMBEDDING`, `RERANKER`, `PARSER`, `UNKNOWN`

**Caution: the decorator auto-captures raw function arguments and return value.** Set `span.set_inputs()` or `span.set_outputs()` inside the body when the span should store a more useful, compact value; explicit values take precedence over the decorator's automatic capture.

For non-chat-shaped root inputs or outputs that need a custom trace-list preview or compact stored values, see [Root span previews and compact values](root-span-previews.md).

### Method 3: Manual Spans (When Decorator Not Possible)

Use only when you can't use a decorator:
- **Tracing code not wrapped in a function** (e.g., script-level code, loop bodies)
- **Dynamic span names** computed at runtime

```python
with mlflow.start_span(name=f"process_{item_id}") as span:
    span.set_inputs({"query": query})  # Must set manually
    result = process(query)
    span.set_outputs({"result": result})  # Must set manually
```

> **Warning:** A `start_span` context manager that never calls `span.set_inputs(...)`
> and `span.set_outputs(...)` produces a span with a name and a duration but no I/O. In
> the trace UI it looks identical to a span that legitimately has none, so a reviewer
> cannot tell instrumentation succeeded. Always set inputs and outputs on a manual span,
> or use the `@mlflow.trace` decorator, which captures both automatically.

### Span content: record full data, not a count

**Always record the actual retrieved content in span inputs and outputs, never a bare count or size summary.** A span that records `{"matches": 20}` instead of the actual documents is useless for debugging.

MLflow's trace store accepts large string attributes. Single attributes over 500K characters have been verified to round-trip correctly, so size is not a reason to truncate. Full content is the default. Any truncation must be a deliberate choice with a comment explaining why.

```python
# Wrong: a count tells you nothing about what the agent retrieved
span.set_outputs({"matches": len(docs)})

# Right: record the content so the trace is debuggable
# docs is the actual list of retrieved records, e.g. [{"text": ..., "id": ...}]
span.set_outputs({"documents": docs})
```

Apply this to all spans that fetch content: retrieval results, search hits, tool outputs, messages, and records.

---

## User/Session Tracking

For multi-turn applications, use standard metadata fields `mlflow.trace.user` and `mlflow.trace.session`.

```python
from fastapi import Request
from mlflow.entities import SpanType

@app.post("/chat")
def handle_chat(request: Request, body: ChatRequest):
    user_id = request.headers.get("X-User-ID", "anonymous")
    session_id = request.headers.get("X-Session-ID", "default")
    return chat(body.message, user_id, session_id)

@mlflow.trace(span_type=SpanType.CHAIN)
def chat(message: str, user_id: str, session_id: str) -> str:
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.user": user_id,
            "mlflow.trace.session": session_id,
        }
    )
    return response
```

**Query traces by user:**

```python
traces = mlflow.search_traces(
    filter_string="metadata.`mlflow.trace.user` = 'user123'"
)
```

---

## Combining AutoLogging with Custom Tracing

The `@mlflow.trace` decorator here marks a deliberate application boundary that wraps a custom retrieval step and the autologged LLM call under one root span. It does not re-decorate the framework's own nodes or model calls, which autolog already captures.

```python
import mlflow
from mlflow.entities import SpanType
from langchain_openai import ChatOpenAI

mlflow.langchain.autolog()

@mlflow.trace(name="rag_pipeline", span_type=SpanType.CHAIN)
def rag_query(question: str) -> str:
    docs = retrieve_documents(question)  # Custom function

    llm = ChatOpenAI()  # Auto-traced by autolog
    response = llm.invoke(format_prompt(docs, question))

    return response.content
```

### Adding descriptions to LangGraph node spans

When `mlflow.langchain.autolog()` traces a LangGraph graph, each node becomes a span named after the node function. The span shows inputs and outputs but nothing that explains what the node does. To make the trace readable without opening each node's code, write the node's docstring into the span's `description` attribute:

```python
import mlflow

def web_research(state: dict) -> dict:
    """Search the web for information relevant to the query."""
    span = mlflow.get_current_active_span()
    if span:
        span.set_attribute("description", web_research.__doc__)
    # node logic here
    return state

def enrich_findings(state: dict) -> dict:
    """Cross-reference web results with the internal knowledge base."""
    span = mlflow.get_current_active_span()
    if span:
        span.set_attribute("description", enrich_findings.__doc__)
    # node logic here
    return state
```

The `description` attribute appears in the span's Attributes tab in the MLflow trace viewer. Any string key works. `"description"` is a readable convention.

---

## Common Issues

**Traces not appearing?**
1. Verify tracking URI is correct (`MLFLOW_TRACKING_URI` env var or `mlflow.set_tracking_uri()`)
2. Ensure the matching autolog integrations are enabled before traced calls run; check setup warnings for patching failures
3. Check experiment is configured (`MLFLOW_EXPERIMENT_ID` env var or `mlflow.set_experiment()`)

**Nested spans not connected?**
- Use `@mlflow.trace` or context managers consistently
- For threading, see `advanced-patterns.md`
