# Topic Categorisation and Tagging System (Exploratory Project)

> **Note**: This is an **exploratory project** focused on learning and experimenting with modular LLM orchestration, custom observability, and structured data extraction from notices of developments from notable platforms.

## Overview

This project provides a framework for processing, summarizing, and categorizing markdown-formatted documents such as news bulletins, notices, or reports. The primary goal is to automatically assign tags based on a user's specific topics of interest, allowing them to filter through high volumes of information and focus on what is truly important. It utilizes local LLMs via **Ollama** and implements a custom pipeline architecture to handle long-form text through summarization before performing final categorization.

Key features include:
- **Modular Prompt Chains**: Implementations of Sequential, Router, and Map-Reduce patterns.
- **Processing Cache**: A robust caching layer that prevents redundant LLM calls by hashing both file content and prompt parameters.
- **Custom Observability**: Integration with **OpenTelemetry** to track every step of the chain execution, including LLM prompts, results, and evaluation scores.
- **Trace Visualization**: Tools to convert OTel traces into interactive HTML Gantt charts.

## Project Structure

```text
CategorisationTagging/
├── ingestion/             # Logic for tracking and reading markdown files
├── prompt_chains/         # Core orchestration logic (BaseChain, Sequential, Map-Reduce, etc.)
├── tagging/               # Categorization logic
│   ├── evals/             # Evaluators for validation and embedding similarity
│   ├── category_tagging.py # The main categorization chain implementation
│   └── process_markdowns.py# Cache management logic
├── utils/                 # Shared utilities
│   ├── myOllama.py        # Custom wrappers for Ollama Chat and Embeddings API
│   ├── otel_setup.py      # OpenTelemetry provider, custom exporters, and Parquet writers
│   └── visualize_trace_html.py # Gantt chart generator for trace analysis
├── main.py                # Entry point
```

## How to Use

### 1. Prerequisites

- [Ollama](https://ollama.com/) installed and running locally.
- Required models pulled: `gemma4:e4b` (for chat) and `qwen3-embedding:8b` (for embeddings).
- Python dependencies installed (Pandas, Plotly, OpenTelemetry SDK, Requests, Pydantic).

### 2. Running the Batch Processor

To process the documents, run one of the main entry points. This will iterate through your ingest directory, check the cache, and execute the LLM chains if necessary.

```bash
python main.py
```

### 3. Visualizing Traces

Every execution generates a unique Trace ID. You can find these in the terminal output or by looking at the logs in `logs/traces/`.

To generate an interactive Gantt chart of a specific execution:

```bash
python utils/visualize_trace_html.py <your_trace_id> --log_dir logs/traces
```

The resulting HTML file will be saved in `logs/gantt_charts/`.

## Implementation Highlights

### Sequential & Map-Reduce Chains
The system automatically decides whether to summarize a document based on its length. 
- **Short documents**: Sent directly to `TradingSystemCategorisation`.
- **Long documents**: Passed through a `MapReduceSummarizeChain` first to distill key information, followed by the categorizer.

### Tracing
Unlike standard logging, our OpenTelemetry implementation:
1. Captures the full `system_prompt` and `user_input`.
2. Saves large LLM responses and "thoughts" to separate Parquet/JSON files to keep the main trace logs lightweight.
3. Injects `session_id` and directory context into every span for easy multi-run analysis.

### Caching Strategy
The `ProcessingCache` uses a `param_hash` which includes the list of `target_tags`. If you add a new tag to your configuration, the system recognizes that the parameters have changed and will re-process the files even if the file content itself hasn't changed.

## Key Takeaways

This exploratory project highlighted several critical strategies for working with LLMs, particularly when deploying smaller parameter models in multi-step pipelines:

- **Traceability and Observability**: As chains grow in complexity and length, observability becomes essential. Using OpenTelemetry to track every step's input and output allows effective debugging and refinement of a pipeline once it exceeds a few simple steps.
- **Granular Units of Work**: Smaller LLMs perform significantly better when complex tasks are broken down into small, specialized units. 
- **Semantic Distinctness in Categorization**: For classification tasks, using words that are as distinct and non-overlapping as possible is crucial. Smaller models have narrower semantic decision boundaries; clear, unique labels help gives better consistency  
- **Schema Enforcement via Structured Outputs**: Leveraging Pydantic for structured outputs saves you the trouble of trying different prompts to get the output to be in the schema you want
- **Working with Smaller Context Windows**:  If the context that is passed through is above the limit, the output will be unexpected even with schema enforcement. While the workaround is to pass it through summarization, some meaning will be lost. This is where deteministically inserting back semantically important parts of the input (e.g. title) will help
---