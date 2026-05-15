# EvalPipeline

A production-grade LLM evaluation pipeline for multi-agent systems, built with Langfuse, PydanticAI, and Streamlit.

## Overview

EvalPipeline captures traces from your AI agents, evaluates outputs using LLM-as-a-Judge, and surfaces quality, cost, and latency metrics — all in one place.

```
Multi-Agent System → LiteLLM Proxy → Langfuse Traces → EvalPipeline → Metrics & Reports
```

## Features

| Module | Description |
|--------|-------------|
| **Experiment Studio** | Run prompt variants against datasets and compare results |
| **LLM-as-a-Judge** | Automated evaluation with customizable scoring criteria |
| **Run History** | Browse and diff past experiment runs |
| **LiteLLM Cost Builder** | Analyze cost and token usage from LiteLLM request logs |
| **Dataset Builder** | Curate evaluation datasets from Langfuse traces |
| **Prompt Analytics** | Track prompt performance over time |
| **Tool Judge** | Evaluate tool call correctness in agentic workflows |
| **OpenReward Runner** | Run reward model evaluations |

## Tech Stack

- **[PydanticAI](https://ai.pydantic.dev/)** — typed agent framework
- **[Langfuse](https://langfuse.com/)** — observability, tracing, and scoring
- **[LiteLLM](https://litellm.ai/)** — unified LLM proxy and cost tracking
- **[Streamlit](https://streamlit.io/)** — evaluation dashboard UI
- **PostgreSQL** — run history and LiteLLM log storage
- **boto3** — AWS integrations

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-org/evalpipeline.git
cd evalpipeline
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

Required variables:

```env
OPENAI_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
DATABASE_URL=postgresql://user:pass@localhost:5432/evalpipeline
```

See [`.env.example`](.env.example) for the full list including LiteLLM options.

### 3. Launch the dashboard

```bash
streamlit run src/frontend/streamlit_app.py
```

## Project Structure

```
src/
├── core/
│   ├── experiment_runner.py     # Orchestrates experiment runs
│   ├── judger.py                # LLM-as-a-Judge evaluator
│   ├── langfuse_client.py       # Langfuse API wrapper
│   ├── dataset_builder.py       # Trace-to-dataset pipeline
│   ├── litellm_ingestion.py     # LiteLLM log ingestion
│   ├── litellm_cost_builder.py  # Cost analytics
│   ├── metrics_analytics.py     # Metrics aggregation
│   ├── openreward_runner.py     # Reward model runner
│   ├── endpoint_runner.py       # HTTP endpoint evaluation
│   ├── prompt_coach_agent.py    # AI-assisted prompt optimization
│   └── run_history.py           # Run persistence
├── schemas/                     # Pydantic models
└── frontend/
    ├── streamlit_app.py         # App entrypoint
    └── pages/                   # One file per dashboard page
tests/                           # pytest test suite
```

## Running Tests

```bash
uv run python -m pytest tests/ -v
```

## LiteLLM Integration

If you use LiteLLM as a proxy, point `LITELLM_DATABASE_URL` at your LiteLLM Postgres database. EvalPipeline reads `litellm_request_logs` by default and joins cost/token data with Langfuse traces.

For non-canonical table schemas, use the `LITELLM_*_COLUMN` overrides in `.env.example`.

## Langfuse Setup

1. Create a project at [cloud.langfuse.com](https://cloud.langfuse.com)
2. Copy your public and secret keys to `.env`
3. Instrument your agents with the Langfuse SDK — traces will appear automatically in the dashboard

## License

MIT
