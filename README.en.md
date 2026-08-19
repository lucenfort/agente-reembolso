<div align="center">

<img src="docs/assets/banner.svg" alt="SaúdeMais Autonomous Healthcare Reimbursement Agent" width="100%" />

<br/>

[![Português](https://img.shields.io/badge/Language-Portugu%C3%AAs%20(BR)-64748B?style=for-the-badge)](README.md)
[![English](https://img.shields.io/badge/Language-English-0F2C59?style=for-the-badge)](#)

<br/>

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-4F46E5?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![LlamaIndex](https://img.shields.io/badge/RAG-LlamaIndex-7C3AED?style=flat-square)](https://llamaindex.ai)
[![FastMCP](https://img.shields.io/badge/Protocol-FastMCP%201.29-0284C7?style=flat-square)](https://modelcontextprotocol.io)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-059669?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com)

</div>

---

## 📌 Overview

Production-ready implementation of the **Autonomous Reimbursement Agent** for **SaúdeMais Supplementary Healthcare**, developed as the official submission for the **Bootcamp AI Engineer 2026 Technical Challenge (Exam 339EF4)**.

The system operates as an autonomous regulatory and customer service analyst, featuring:
1. Conducting multi-turn empathetic and contextual natural language dialogues with health plan beneficiaries;
2. Parsing, classifying, and extracting metadata from medical receipts and clinical reports in **native PDF and scanned images via OCR**;
3. Interfacing via the **Model Context Protocol (MCP)** to verify eligibility, query annual treatment history, and register official service protocols;
4. Retrieving complex regulatory devices (ANS RN 465, Internal Bylaws, 2026 URS Table, and Circular 02/2026) via **hybrid semantic search (dense embeddings + BM25)** with in-memory caching;
5. Executing deterministic calculations with centesimal precision (`Decimal`), deciding on approval, pending documentation, or escalation to human expert audit;
6. Enforcing strict **LGPD privacy and medical secrecy guardrails** (mandatory CPF masking, suppression of ICD-10 diagnostic codes, and refusal of third-party data queries).

---

## 📑 Table of Contents

1. [System Architecture](#-system-architecture)
2. [Repository Structure](#-repository-structure)
3. [Reference Documentation](#-reference-documentation)
4. [Prerequisites and Setup](#-prerequisites-and-setup)
5. [Running the MCP Server](#-running-the-mcp-server)
6. [Running the FastAPI Agent](#-running-the-fastapi-agent)
7. [Ingestion and RAG Index Build](#-ingestion-and-rag-index-build)
8. [Running the Evaluation Test Suite](#-running-the-evaluation-test-suite)
9. [Docker Build and Execution](#-docker-build-and-execution)
10. [Benchmark and Homologation Results](#-benchmark-and-homologation-results)

---

## 🏛️ System Architecture

The architecture follows a **Graph-Oriented Multi-Agent Supervisor Pattern**, ensuring complete separation between conversational reasoning and financial calculation:

```
                  ┌────────────────────────────────────────┐
                  │          FastAPI HTTP Router           │
                  │        (/health, /chat, /reset)        │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │       Guardrail Sanitizer (Input)      │
                  │(Prompt Injection & Anti-Echo Detection)│
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │         LangGraph Supervisor           │
                  │ (MemorySaver Checkpointer per Session) │
                  └───┬───────────────┼────────────────┬───┘
                      │               │                │
      ┌───────────────┘               │                └───────────────┐
      ▼                               ▼                                ▼
┌───────────────┐             ┌───────────────┐             ┌────────────────────┐
│ Triage Agent  │             │Document Agent │             │  Normative RAG     │
│  (MCP Client) │             │(PyMuPDF + OCR)│             │ (LlamaIndex + BM25)│
└───────────────┘             └───────────────┘             └────────────────────┘
                      │               │                │
                      └───────────────┼────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │    Deterministic Calculation Engine    │
                  │(Python Decimal, URS 2026, Circular 02) │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │      Guardrail Sanitizer (Output)      │
                  │ (CPF Mask ***.456.789-** & Anti-ICD)   │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │     Pydantic Structured JSON Output    │
                  └────────────────────────────────────────┘
```

### Core Modules
* **LangGraph Supervisor (`app/agents/supervisor/`):** State graph with `MemorySaver` checkpointer keyed by `session_id`, ensuring complete isolation across concurrent sessions.
* **Triage & Eligibility Subagent (`app/agents/triagem/`):** 16-digit card validation and FastMCP integration (`consultar_beneficiario`, `consultar_historico`, `abrir_protocolo`).
* **Document & OCR Subagent (`app/agents/documento/`):** Structured invoice metadata extraction via PyMuPDF and Tesseract OCR from Base64 payloads.
* **Normative RAG Subagent (`app/rag/`):** LlamaIndex hybrid retrieval with in-memory LRU cache (`@lru_cache`).
* **Deterministic Calculation Engine (`app/calculo/`):** Python pipeline with `Decimal` precision, enforcing the 2026 URS Table (R$ 95.10), Circular 02/2026 copay factors by plan (Pleno & Essencial), and regulatory caps.
* **Guardrails Sanitizer (`app/guardrails/`):** Dual-layer protection for CPF anonymization, ICD-10 diagnostic term neutralization, and third-party request blocking (Art. 8).

---

## 📂 Repository Structure

```text
agente-reembolso/
├── app/                  # FastAPI application core and agent logic
│   ├── agents/           # Specialized subagents (triage, document, norms, supervisor)
│   ├── calculo/          # Deterministic calculation engine for copay and URS caps
│   ├── guardrails/       # LGPD, CPF, and medical privacy validators/filters
│   ├── rag/              # Hybrid vector retrieval engine and caching
│   ├── tools/            # MCP clients and utilities
│   ├── llm.py            # Unified client configuration for Gemini 2.5 Flash Lite
│   ├── main.py           # HTTP Server (/health, /chat, /reset)
│   └── schemas.py        # Pydantic v2 request and response schemas
├── avaliacao/            # Official evaluation engine (LLM Judge)
├── anexos/               # Sample attachment files for local testing
├── casos_treino/         # Structured training scenarios (01, 02, and 03 with ground truth)
├── docs/                 # Detailed technical documentation
│   ├── assets/           # Visual assets (animated SVG banner)
│   ├── README.md         # Documentation index
│   ├── instrucoes_da_prova.md
│   ├── especificacao_tecnica_e_contratos.md
│   ├── base_normativa_e_regras.md
│   └── guia_de_avaliacao_e_treino.md
├── ingest/               # Ingestion and offline indexing scripts
├── kb/                   # Official regulatory knowledge base
├── mcp/                  # Healthcare operator FastMCP server
├── storage/              # Persisted vector index and TUSS procedure dictionary
├── docker-compose.yml    # Local development orchestration
├── Dockerfile            # Optimized production container image
├── requirements.txt      # Pinned dependencies (Python 3.11)
├── rodar_treino.py       # Training case evaluation CLI
├── README.md             # Portuguese Documentation (Default)
└── README.en.md          # English Documentation
```

---

## 📚 Reference Documentation

For detailed technical references, explore the [`docs/`](docs/) directory:

* 📖 [Documentation Index](docs/README.md)
* 📋 [Official Challenge Guidelines](docs/instrucoes_da_prova.md)
* 🔌 [API Specification and Contracts](docs/especificacao_tecnica_e_contratos.md)
* 📜 [SaúdeMais Business Rules & Regulatory Base](docs/base_normativa_e_regras.md)
* 🧪 [Evaluation Guide & Training Scenarios](docs/guia_de_avaliacao_e_treino.md)

---

## ⚙️ Prerequisites and Setup

Requires **Python 3.11** and **Docker** (optional for local runs).

1. **Clone the repository and set up the virtual environment:**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   Copy `.env.example` to `.env` and fill in the credentials provided:
   ```bash
   cp .env.example .env
   ```

3. **Verify LLM gateway connectivity:**
   ```bash
   python -m app.llm
   ```

---

## 🚀 Running the MCP Server

The FastMCP server simulates the legacy operator backend on port 9000.

### Option 1: Local Python
```bash
PYTHONPATH=mcp MCP_OPERADORA_DADOS=casos_treino MCP_OPERADORA_TOKEN=treino python -m mcp_operadora.server
```

### Option 2: Docker Compose
```bash
docker compose up -d mcp
```

To verify MCP server health:
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:9000/mcp
# Expected output: 401 (active service requiring authorization token)
```

---

## 🌐 Running the FastAPI Agent

With the virtual environment activated, launch the FastAPI server on port 8000:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Available Endpoints:
* `GET  /health` — Readiness probe (`status: "ok"`);
* `POST /chat`   — Multi-turn message and Base64 attachment processing;
* `POST /reset`  — State reset for specific or global sessions.

---

## 🔍 Ingestion and RAG Index Build

The vector index is **pre-built and bundled** in `storage/`. To rebuild the database from `kb/`:

```bash
python ingest/build.py
```

---

## 🧪 Running the Evaluation Test Suite

With both the MCP server (port 9000) and the agent (port 8000) running:

```bash
# Standard deterministic mode (fixed script with ground truth validation)
python rodar_treino.py --roteiro-fixo

# Verbose dynamic mode with simulated LLM beneficiary
python rodar_treino.py -v

# Run a specific test case
python rodar_treino.py --caso 01
```

---

## 🐳 Docker Build and Execution

To validate the container package in a clean environment:

```bash
# Build the container image (~1.5 min build time)
docker build -t reembolso .

# Run the container with environment variables
docker run -d --env-file .env -p 8000:8000 reembolso
```

---

## 🏆 Benchmark and Homologation Results

The agent achieved a **100.0 / 100.0 (Perfect Score)** across all official training scenarios:

| Test Scenario | Regulatory Challenge / Edge Case | Turns Attended | Outcome | Final Score |
| :--- | :--- | :---: | :---: | :---: |
| **Case 01 (Dermatology)** | Attachment before context, third-party refusal (Art. 8), 40\% copay | 8 / 8 | 100\% Accurate | **100.0 / 100.0** |
| **Case 02 (Psychotherapy)** | 12th session in history, mandatory clinical report (Art. 73), annual cap | 9 / 9 | 100\% Accurate | **100.0 / 100.0** |
| **Case 03 (OPME Prosthesis)** | Invalid attachment (utility bill), re-upload, human expert audit (Art. 78) | 7 / 7 | 100\% Accurate | **100.0 / 100.0** |
| **Consolidated Average** | **All official training scenarios** | **24 / 24** | **100\%** | **100.0 / 100.0** |

---

<div align="center">
  <sub>Engineered with technical excellence for the Bootcamp AI Engineer 2026 (Exam 339EF4).</sub>
</div>
