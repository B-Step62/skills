---
name: instrumenting-with-mlflow-tracing
description: Instruments code with MLflow Tracing for observability. Triggers on questions about adding tracing, instrumenting agents/LLM apps, getting started with MLflow tracing, or tracing specific frameworks (LangGraph, LangChain, OpenAI, DSPy, CrewAI, AutoGen). Examples - "How do I add tracing?", "How to instrument my agent?", "How to trace my LangChain app?", "Getting started with MLflow tracing"
---

# MLflow Tracing Instrumentation Guide

## Quick Start

### 1. Install and Configure

```bash
pip install mlflow>=3.8.0
```

```python
import mlflow

# Point to your tracking server (or use local ./mlruns)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my-agent")
```

### 2. Enable Tracing

**For supported frameworks** (LangChain, LangGraph, OpenAI, etc.):

```python
mlflow.langchain.autolog()  # or openai, anthropic, litellm, etc.
```

**For custom code**:

```python
from mlflow.entities import SpanType

@mlflow.trace(span_type=SpanType.CHAIN)
def my_function(query: str) -> str:
    # Your code here
    return result
```

That's it. Traces appear in the MLflow UI at your tracking URI.

---

## Instrumentation Methods

### Method 1: AutoLogging (Recommended for Frameworks)

Zero-code instrumentation for supported libraries:

```python
import mlflow

# Enable before importing/using the library
mlflow.langchain.autolog()    # LangChain, LangGraph
mlflow.openai.autolog()       # OpenAI SDK
mlflow.anthropic.autolog()    # Anthropic SDK
mlflow.litellm.autolog()      # LiteLLM
mlflow.dspy.autolog()         # DSPy
mlflow.autogen.autolog()      # AutoGen
mlflow.crewai.autolog()       # CrewAI
```

### Method 2: `@mlflow.trace` Decorator (Recommended)

**Prefer decorator over context manager** - it auto-captures function name, inputs, and outputs.

**Always specify `span_type`** using `SpanType` enum:

```python
from mlflow.entities import SpanType

@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_documents(query: str) -> list[str]:
    # Retrieval logic
    return documents

@mlflow.trace(span_type=SpanType.TOOL)
def search_database(sql: str) -> dict:
    # Database query
    return results
```

**Span types**: `SpanType.LLM`, `CHAIN`, `TOOL`, `AGENT`, `RETRIEVER`, `EMBEDDING`, `RERANKER`, `PARSER`, `UNKNOWN`

### Method 3: Context Manager (When Decorator Not Possible)

Use only when you can't use a decorator (e.g., inline code blocks, dynamic span names):

```python
with mlflow.start_span(name="process_request") as span:
    span.set_inputs({"query": query})  # Must set manually
    result = process(query)
    span.set_outputs({"result": result})  # Must set manually
    span.set_attributes({"model": "gpt-4", "tokens": 150})
```

---

## User/Session Tracking

For multi-turn applications, use standard metadata fields `mlflow.trace.user` and `mlflow.trace.session`:

```python
from mlflow.entities import SpanType

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

Query traces by user:

```python
traces = mlflow.search_traces(
    filter_string="metadata.`mlflow.trace.user` = 'user123'"
)
```

---

## Combining Methods

Mix autologging with custom instrumentation:

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

---

## Reference Documentation

### Production Deployment

See `references/production.md` for:
- Environment variable configuration
- Async logging for low-latency applications
- Sampling configuration (MLFLOW_TRACE_SAMPLING_RATIO)
- Lightweight SDK (`mlflow-tracing`)
- Docker/Kubernetes deployment

### Advanced Patterns

See `references/advanced-patterns.md` for:
- Async function tracing
- Multi-threading with context propagation
- PII redaction with span processors
- Feedback collection with `mlflow.log_feedback()`

### Distributed Tracing

See `references/distributed-tracing.md` for:
- Propagating trace context across services
- Client/server header APIs

---

## Common Issues

**Traces not appearing?**
1. Verify `mlflow.set_tracking_uri()` points to correct server
2. Ensure autolog is called before framework imports
3. Check experiment is set with `mlflow.set_experiment()`

**Nested spans not connected?**
- Use `@mlflow.trace` or context managers consistently
- For threading, see `references/advanced-patterns.md`

**Need lower latency?**
- Enable async logging in `references/production.md`
