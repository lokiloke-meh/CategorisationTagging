# Topic Categorisation and Tagging System

> **Note**: This project focuses on modular LLM orchestration, custom observability, and structured data extraction from market notices, circulars, and exchange announcements.

## Overview

This project provides an end-to-end framework for ingesting, summarizing, and categorizing documents such as exchange circulars, API changelogs, and platform notices. Its primary objective is to evaluate incoming documents against customizable topics of interest (e.g., market structure changes, disaster recovery sessions, trading hour amendments, corporate actions) and score their relevance (e.g., High, Medium, Low, None) with auditable reasoning.

Key capabilities include:
- **Modular Ingestion Engine**: Automated scrapers for various financial exchanges and crypto platforms, handling PDF extraction and Markdown conversion.
- **Composable Prompt Chains**: Primitives for Sequential, Boolean Routing, Categorical Routing, and Map-Reduce / Refine Summarization workflows.
- **Hybrid LLM Adapters**: Support for both OpenAI-compatible endpoints (such as `llama.cpp` for local inference and specialized decision models) and `Ollama`.
- **Intelligent Caching Layer**: Additive, per-tag caching keyed by document content hash and pipeline identity—ensuring only missing or updated tags are re-evaluated.
- **Deep Observability**: OpenTelemetry tracing for every chain, decision, evaluation, and token metric, offloading large prompt payloads and thought logs to compact Parquet files.
- **Interactive Visualizers & Web Dashboard**: A Flask-based web application with multi-topic filtering, embedded interactive Mermaid pipeline DAGs, and Plotly Gantt execution timelines.

---

## How It Works

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Ingestion Layer                                                     │
│    Exchange Scrapers                                                     │
│    └─► Fetch HTML / Download PDF ─► Convert to Markdown               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ 2. ContentStore & Caching Layer                                        │
│    SHA-256 Hashing ─► file_hashes.json                                 │
│    Additive per-tag cache check (has_valid_cache)                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (Cache Miss)
┌───────────────────────────────────▼────────────────────────────────────┐
│ 3. Pipeline Orchestration (TagPipeline / TagPipelineDecisionModel)     │
│    Document Length Routing:                                            │
│      ├─► Short (< 4,000 chars) ─► Direct to Categorical Router         │
│      └─► Long  (>= 4,000 chars) ─► Sentence Splitter + Summarizer      │
│                                    (Refine / Map-Reduce)              │
│    Impact Categorization:                                              │
│      └─► Topic Bucketing (Market-Wide vs Segment vs Specific Security) │
│    Topic Distillation:                                                 │
│      └─► Few-shot evaluation against YAML tag definitions               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ 4. Observability & Persistence                                         │
│    OpenTelemetry Spans + Token Metrics                                │
│    Heavy payloads stored in Parquet (llm_details, chain_details)       │
│    Results saved to processing_results.json & pipeline_registry.json   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ 5. Access & Visualization                                              │
│    CLI Reports ─► tagging/generate_report.py (Markdown export)         │
│    Web Dashboard ─► tagging/flaskapp/run.py (Flask UI + Gantt + DAG)   │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Ingestion & Content Store
Scrapers inherit from `CircularScraper` (`ingestion/circular_scraper_base.py`) to fetch circulars, extract text from PDFs, sanitize content, and write standardized Markdown files to `data_store/<topic>/ingest/`. The `ContentStore` computes SHA-256 hashes of content, tracks duplicate downloads in `file_hashes.json`, and records ingestion timestamps.

### 2. Pipeline Orchestration
Pipelines are constructed from composable chain blocks in `prompt_chains/`:
- **Length Routing**: Short documents bypass summarization; long documents pass through a `SentenceTextSplitter` into a `RefineSummarizeChain` or `MapReduceSummarizeChain`.
- **Categorical Routing (`TopicBucketing`)**: Pre-filters documents by operational scope (e.g., `MarketWideImpact`, `SegmentAssetLevelImpact`, `SpecificSecurityImpact`).
- **Topic Distillation (`TradingTopicsDistiller`)**: Analyzes relevant documents against prompt definitions and few-shot examples defined in YAML (`tagging/tagging_examples_yaml/`), returning structured decisions and justification.
- **Supported Pipeline Architectures**:
  - `TagPipeline`: Pure generative chat pipeline utilizing a `ChatAdapter` (e.g., `llamaCppChat` or `ollamaChat`).
  - `TagPipelineDecisionModel`: Hybrid pipeline pairing a specialized `DecisionModel` (e.g., `llamacppDecisionModel` querying structured choice endpoints) for categorization with a `ChatAdapter` for generative summarization.

### 3. Additive Per-Tag Caching
Every pipeline produces a deterministic `pipeline_id` hashing its graph structure, component settings, and chunk sizes. When target tags change or new tags are added to a configuration profile, `ContentStore.has_valid_cache()` detects missing tags and triggers processing only for the missing tags, saving previous evaluations without re-running unchanged work.

### 4. Telemetry & Trace Storage
OpenTelemetry tracks every execution step (`utils/otel_setup.py`). To prevent log bloat and performance degradation, large attributes (system prompts, user inputs, model thoughts, and evaluation matrices) are offloaded to individual Parquet files under `traces/<run_timestamp>/`.

---

## Project Structure

```text
CategorisationTagging/
├── config.py                     # Profile configuration loader proxy
├── run_configs/                  # Configuration profiles
│   ├── config_default.py         # Default configuration profile & tag mappings
│   └── config_crypto_exchanges.py# Alternative exchange profile
├── ingestion/                    # Ingestion framework & scrapers
│   ├── circular_scraper_base.py  # Base scraper class with PDF/Markdown helpers
│   ├── content_store.py          # Central store for hashing, caching, and reporting
│   └── ingesters/                # Source-specific scrapers
│       ├── hkex_ingester.py
│       ├── binance_announcement_ingester.py
│       ├── binance_dev_changelog_ingester.py
│       ├── bybit_announcement_ingester.py
│       ├── bybit_api_v5_changelog_ingester.py
│       ├── okx_announcement_ingester.py
│       ├── nseix_ingester.py
│       └── mcx_ingester.py
├── prompt_chains/                # Orchestration primitives
│   ├── base_chains.py            # BaseChain, Document, ChainResult
│   ├── long_text_summariser.py   # Refine & Map-Reduce summarizers
│   ├── long_html_splitter.py     # HTML token/word splitters
│   ├── routing_chains_bool.py    # Boolean condition routers
│   ├── routing_chains_categorical.py # Multi-branch categorical routers
│   └── sequential_chains.py      # Linear chain executor
├── tagging/                      # Tagging logic & application layer
│   ├── category_tagging.py       # TradingTopicsDistiller
│   ├── independent_category_tagging.py # TopicBucketing
│   ├── pipelines/
│   │   ├── tag_pipeline.py       # Standard chat-based tag pipeline
│   │   └── tag_pipeline_decision_model.py # Hybrid decision + chat model pipeline
│   ├── tagging_examples_yaml/    # YAML tag definitions and decision criteria
│   ├── evals/                    # Result data and consistency evaluators
│   ├── generate_report.py        # CLI report generation
│   └── flaskapp/                 # Interactive visualizer web dashboard
│       ├── run.py                # Flask entry point
│       ├── app.py                # Route controllers
│       ├── services.py           # Data queries and multi-topic report aggregation
│       └── renderers/            # Mermaid DAG and Plotly Gantt HTML renderers
├── utils/                        # Shared adapters and utilities
│   ├── llm_models.py             # ChatAdapter, DecisionModel, LLMResult protocols
│   ├── myllamacpp.py             # llama.cpp Chat, Decision, and Embedding adapters
│   ├── myOllama.py               # Ollama Chat and Embedding wrappers
│   ├── otel_setup.py             # OpenTelemetry provider and Parquet detail exporter
│   ├── visualise_pipeline_html.py# Standalone pipeline DAG visualizer
│   └── visualize_trace_html.py   # Standalone trace Gantt visualizer
├── main_decision_model.py        # Main entry point (Decision + Chat model)
├── main_chat_model.py            # Main entry point (Chat model only)
├── data_store/                   # Storage root for topics, markdown, and traces
├── pyproject.toml                # Project metadata and dependencies
└── requirements.txt              # Pip requirements
```

---

## Access Points & How to Use

### 1. Prerequisites & Environment Setup

- **Python**: Python `>= 3.14` (or managed via `uv`).
- **Dependencies**: Install using `uv` or `pip`:
  ```bash
  pip install -r requirements.txt
  # or with uv
  uv sync
  ```
- **LLM Server**:
  - Run a local server such as **llama.cpp** exposing an OpenAI-compatible API:
    - Chat endpoint: `http://localhost:8080/v1` (e.g. `gemma-4-12b-it-qat-q4_0`)
    - Decision endpoint: `http://localhost:8080/v1/systemone` (e.g. `Cloudflare_clef-flash-Q4_K_M`)
    - Embeddings endpoint: `http://localhost:8082/v1/embeddings` (optional)
  - Or use **Ollama** running locally on default port `11434`.

---

### 2. Ingesting Documents (Scrapers)

Scrapers can be run directly from the command line to fetch notices and convert them to Markdown.

```bash
# Scrape exchange circulars for the past 7 days
python -m ingestion.ingesters.exchange_ingester --days 7

# Scrape crypto announcements for the past 30 days
python -m ingestion.ingesters.crypto_announcement_ingester --days 30

# Scrape API changelogs
python -m ingestion.ingesters.crypto_changelog_ingester --days 14

# Scrape announcements to a custom output directory
python -m ingestion.ingesters.alternative_exchange_ingester --days 3 --output data_store/announcements
```

All ingested files are stored in `data_store/<topic>/ingest/` with tracking metadata updated in `data_store/<topic>/file_hashes.json`.

---

### 3. Running Tagging Pipelines

Two primary entry points are available depending on your model setup:

#### Option A: Hybrid Pipeline (Decision Model + Chat Model)
Uses a structured decision model for high-throughput bucketing/scoring and a chat model for text summarization:

```bash
# Run against exchange circulars using default profile
python main_decision_model.py --base-dir data_store/exchange_circulars

# Run against alternative exchange data using the alternative profile
python main_decision_model.py --base-dir data_store/crypto_announcements --config crypto_exchanges

# Process a limited test batch of documents
python main_decision_model.py --base-dir data_store/exchange_circulars --num-docs 5
```

#### Option B: Chat-Only Pipeline
Uses a standard conversational model for both summarization and structured JSON extraction:

```bash
python main_chat_model.py --base-dir data_store/exchange_circulars --num-docs 10
```

---

### 4. Generating Reports

Aggregate and export high-relevance findings into a clean Markdown summary report:

```bash
# Generate report for High-impact items in a topic
python tagging/generate_report.py data_store/exchange_circulars --levels High

# Filter by specific tags and date processed
python tagging/generate_report.py data_store/exchange_circulars \
  --tags "TradingHoursAmendment,MarketWideBoardLotSizeFrameworkChanges" \
  --since 2026-10-01 \
  --output data_store/exchange_circulars/filtered_report.md
```

---

### 5. Web UI Visualizer & Dashboard

Launch the local Flask application to browse notices, inspect tag classifications, review trace timings, and view pipeline execution DAGs:

```bash
python tagging/flaskapp/run.py
```

Options:
- `--data-dir`: Custom path to data store (default: `data_store/`).
- `--port`: Port to listen on (default: `5000`).
- `--host`: Host interface to bind to (default: `127.0.0.1`).

**Features available at `http://127.0.0.1:5000`**:
- **Cross-Topic Search**: Filter notices across multiple ingested exchanges simultaneously.
- **Tag Filtering**: Select specific tags or target impact levels.
- **Article Detail View**:
  - **Trace Gantt Tab**: Embedded Plotly timeline displaying execution duration and input/output payloads of every span.
  - **Pipeline DAG Tab**: Dynamic Mermaid diagram showing the exact chain topology and configuration parameters for that run.
  - **Raw Content Tab**: Formatted preview of the ingested Markdown document.

---

### 6. Standalone Trace & Pipeline Visualizers

Generate standalone, sharable HTML files directly from the command line:

```bash
# Generate interactive OpenTelemetry Trace Gantt chart
python utils/visualize_trace_html.py <trace_id> --log_dir data_store/exchange_circulars/traces

# Generate standalone Mermaid Pipeline DAG visualization
python utils/visualise_pipeline_html.py <pipeline_id> --input data_store/exchange_circulars/pipeline_registry.json
```

---

## Key Takeaways

Key takeaways from this project highlight several critical strategies for working with LLMs, particularly when deploying smaller parameter models in multi-step pipelines:

- **Traceability and Observability**: As chains grow in complexity and length, observability becomes essential. Using OpenTelemetry to track every step's input and output allows effective debugging and refinement of a pipeline once it exceeds a few simple steps.
- **Granular Units of Work**: Smaller LLMs perform significantly better when complex tasks are broken down into small, specialized units. 
- **Semantic Distinctness in Categorization**: For classification tasks, using words that are as distinct and non-overlapping as possible is crucial. Smaller models have narrower semantic decision boundaries; clear, unique labels help gives better consistency  
- **Schema Enforcement via Structured Outputs**: Leveraging Pydantic for structured outputs saves you the trouble of trying different prompts to get the output to be in the schema you want
- **Working with Smaller Context Windows**:  If the context that is passed through is above the limit, the output will be unexpected even with schema enforcement. While the workaround is to pass it through summarization, some meaning will be lost. This is where deteministically inserting back semantically important parts of the input (e.g. title) will help
---
