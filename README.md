<div align="center">

<img src="docs/assets/banner.svg" alt="Agente Autônomo de Reembolso SaúdeMais" width="100%" />

<br/>

[![Português](https://img.shields.io/badge/Idioma-Portugu%C3%AAs%20(BR)-0F2C59?style=for-the-badge)](#)
[![English](https://img.shields.io/badge/Language-English-64748B?style=for-the-badge)](README.en.md)

<br/>

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/Orquestra%C3%A7%C3%A3o-LangGraph-4F46E5?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![LlamaIndex](https://img.shields.io/badge/RAG-LlamaIndex-7C3AED?style=flat-square)](https://llamaindex.ai)
[![FastMCP](https://img.shields.io/badge/Protocolo-FastMCP%201.29-0284C7?style=flat-square)](https://modelcontextprotocol.io)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-059669?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com)

</div>

---

## 📌 Visão Geral

Implementação do **Agente Autônomo de Reembolso** para a **SaúdeMais Saúde Suplementar**, desenvolvido como solução oficial para o desafio técnico do **Bootcamp AI Engineer 2026 (Prova 339EF4)**.

O sistema atua como um analista regulatório e de atendimento ao cliente, sendo capaz de:
1. Conduzir diálogos acolhedores, didáticos e empáticos em linguagem natural (*atendimento multi-turno*);
2. Processar, classificar e extrair dados de documentos fiscais e laudos clínicos em **PDF nativo e imagens escaneadas via OCR**;
3. Integrar-se via **Model Context Protocol (MCP)** para consultar dados cadastrais, histórico anual de sessões e emitir protocolos oficiais;
4. Recuperar normativas complexas da SaúdeMais (RN 465/ANS, Regulamento Interno, Tabela URS 2026, Circular 02/2026) por **busca híbrida (vetorial densa + BM25)** com cache em memória;
5. Calcular deterministamente valores devidos com precisão decimal exata (`Decimal`), decidindo por aprovação, pendência documental ou escalonamento pericial humano;
6. Assegurar conformidade rigorosa com a **LGPD e regras de sigilo** (mascaramento de CPF, supressão de diagnósticos/CID-10 e recusa de consulta a dados de terceiros).

---

## 📑 Sumário

1. [Arquitetura do Sistema](#-arquitetura-do-sistema)
2. [Estrutura do Repositório](#-estrutura-do-repositório)
3. [Documentação de Referência](#-documentação-de-referência)
4. [Pré-requisitos e Instalação](#-pré-requisitos-e-instalação)
5. [Execução do Servidor MCP](#-execução-do-servidor-mcp)
6. [Execução do Agente FastAPI](#-execução-do-agente-fastapi)
7. [Ingestão e Construção do Índice Normativo (RAG)](#-ingestão-e-construção-do-índice-normativo-rag)
8. [Execução dos Testes de Avaliação](#-execução-dos-testes-de-avaliação)
9. [Build e Execução via Docker](#-build-e-execução-via-docker)
10. [Resultados de Benchmark e Homologação](#-resultados-de-benchmark-e-homologação)

---

## 🏛️ Arquitetura do Sistema

A solução adota o padrão **Supervisor Multi-Agente Orientado a Grafo**, garantindo segregação estrita entre raciocínio conversacional e cálculo financeiro:

```
                  ┌────────────────────────────────────────┐
                  │          FastAPI HTTP Router           │
                  │        (/health, /chat, /reset)        │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │      Guardrail Sanitizer (Entrada)     │
                  │ (Detecção de Injeção & Anti-Eco Regex) │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │        Supervisor LangGraph            │
                  │   (Checkpointer MemorySaver por Sessão)│
                  └───┬───────────────┼────────────────┬───┘
                      │               │                │
      ┌───────────────┘               │                └───────────────┐
      ▼                               ▼                                ▼
┌───────────────┐             ┌───────────────┐             ┌────────────────────┐
│ Agente Triagem│             │Agente Document│             │    Agente Normas   │
│(MCP Operadora)│             │(PyMuPDF + OCR)│             │ (LlamaIndex + BM25)│
└───────────────┘             └───────────────┘             └────────────────────┘
                      │               │                │
                      └───────────────┼────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │      Motor Determinístico de Cálculo   │
                  │ (Python Decimal, URS 2026, Circular 02)│
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │       Guardrail Sanitizer (Saída)      │
                  │ (Máscara CPF ***.456.789-** e Anti-CID)│
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │    Resposta JSON Estruturada Pydantic  │
                  └────────────────────────────────────────┘
```

### Componentes Principais
* **Supervisor LangGraph (`app/agents/supervisor/`):** Grafo de estados com checkpointer `MemorySaver` indexado por `session_id`.
* **Subagente de Triagem e Elegibilidade (`app/agents/triagem/`):** Validação de carteirinhas de 16 dígitos e integração FastMCP (`consultar_beneficiario`, `consultar_historico`, `abrir_protocolo`).
* **Subagente de Documentos & OCR (`app/agents/documento/`):** Extração de metadados fiscais via PyMuPDF e Tesseract OCR a partir de payloads Base64.
* **Subagente de RAG Normativo (`app/rag/`):** LlamaIndex com busca híbrida e cache LRU em memória (`@lru_cache`).
* **Motor Determinístico de Cálculo (`app/calculo/`):** Pipeline em Python com precisão `Decimal`, aplicando a Tabela URS 2026 (R$ 95,10), fatores de coparticipação da Circular 02/2026 por plano (Pleno e Essencial) e tetos regulamentares.
* **Guardrails Sanitizer (`app/guardrails/`):** Camada de proteção para anonimização de CPF, neutralização de termos de CID-10 e bloqueio de consultas a terceiros (Art. 8º).

---

## 📂 Estrutura do Repositório

```text
agente-reembolso/
├── app/                  # Núcleo da aplicação FastAPI e lógica dos agentes
│   ├── agents/           # Subagentes especializados (triagem, documento, normas, supervisor)
│   ├── calculo/          # Motor determinístico de cálculo de coparticipação e tetos URS
│   ├── guardrails/       # Validadores e filtros de LGPD, CPF e sigilo médico
│   ├── rag/              # Mecanismo de busca vetorial híbrida e cache
│   ├── tools/            # Clientes MCP e utilitários
│   ├── llm.py            # Configuração unificada do cliente Gemini 2.5 Flash Lite
│   ├── main.py           # Servidor HTTP (/health, /chat, /reset)
│   └── schemas.py        # Modelos Pydantic v2 de requisição e resposta
├── avaliacao/            # Motor de avaliação oficial da prova (Juiz LLM)
├── anexos/               # Arquivos anexos de exemplo para testes locais
├── casos_treino/         # Casos de teste estruturados (01, 02 e 03 com gabaritos)
├── docs/                 # Documentação técnica detalhada
│   ├── assets/           # Ativos visuais (banner SVG animado)
│   ├── README.md         # Índice da documentação
│   ├── instrucoes_da_prova.md
│   ├── especificacao_tecnica_e_contratos.md
│   ├── base_normativa_e_regras.md
│   └── guia_de_avaliacao_e_treino.md
├── ingest/               # Scripts de ingestão e indexação offline
├── kb/                   # Base de conhecimento regulatória oficial
├── mcp/                  # Servidor FastMCP da operadora de saúde
├── storage/              # Índice vetorial e dicionário TUSS persistidos
├── docker-compose.yml    # Orquestração local de desenvolvimento
├── Dockerfile            # Imagem de produção otimizada
├── requirements.txt      # Dependências fixadas (Python 3.11)
├── rodar_treino.py       # CLI de execução dos casos de treino
├── README.md             # Documentação principal em Português
└── README.en.md          # Documentação completa em Inglês
```

---

## 📚 Documentação de Referência

Consulte a documentação técnica detalhada na pasta [`docs/`](docs/):

* 📖 [Índice Geral da Documentação](docs/README.md)
* 📋 [Instruções Oficiais da Prova](docs/instrucoes_da_prova.md)
* 🔌 [Especificação Técnica e Contratos de API](docs/especificacao_tecnica_e_contratos.md)
* 📜 [Base Normativa e Regras de Negócio da SaúdeMais](docs/base_normativa_e_regras.md)
* 🧪 [Guia de Avaliação e Casos de Treino](docs/guia_de_avaliacao_e_treino.md)

---

## ⚙️ Pré-requisitos e Instalação

Requer **Python 3.11** e **Docker** (opcional para execução local).

1. **Clonar o repositório e criar o ambiente virtual:**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configurar as variáveis de ambiente:**
   Copie o arquivo `.env.example` para `.env` e preencha as credenciais fornecidas:
   ```bash
   cp .env.example .env
   ```

3. **Validar a conectividade com o gateway LLM:**
   ```bash
   python -m app.llm
   ```

---

## 🚀 Execução do Servidor MCP

O servidor FastMCP simula o sistema legado da operadora e roda na porta 9000.

### Opção 1: Python Local
```bash
PYTHONPATH=mcp MCP_OPERADORA_DADOS=casos_treino MCP_OPERADORA_TOKEN=treino python -m mcp_operadora.server
```

### Opção 2: Docker Compose
```bash
docker compose up -d mcp
```

Para verificar o status do servidor MCP:
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:9000/mcp
# Retorno esperado: 401 (serviço ativo exigindo token de autorização)
```

---

## 🌐 Execução do Agente FastAPI

Com o ambiente virtual ativado, inicie o servidor FastAPI na porta 8000:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints Disponíveis:
* `GET  /health` — Verificação de prontidão do container (`status: "ok"`);
* `POST /chat`   — Processamento de mensagens multi-turno e anexos Base64;
* `POST /reset`  — Limpeza de estado de sessões específicas ou globais.

---

## 🔍 Ingestão e Construção do Índice Normativo (RAG)

O índice vetorial já se encontra **pré-construído e embutido** em `storage/`. Caso precise reconstruir a base a partir dos documentos de `kb/`:

```bash
python ingest/build.py
```

---

## 🧪 Execução dos Testes de Avaliação

Com o MCP (porta 9000) e o agente (porta 8000) em execução:

```bash
# Modo padrão determinístico (roteiro fixo com validação de gabarito)
python rodar_treino.py --roteiro-fixo

# Modo detalhado com beneficiário simulado por LLM
python rodar_treino.py -v

# Executar apenas um caso específico
python rodar_treino.py --caso 01
```

---

## 🐳 Build e Execução via Docker

Para validar a imagem oficial do container em ambiente limpo:

```bash
# Build da imagem (tempo médio: ~1.5 minuto)
docker build -t reembolso .

# Execução do container com as variáveis de ambiente
docker run -d --env-file .env -p 8000:8000 reembolso
```

---

## 🏆 Resultados de Benchmark e Homologação

A solução foi homologada com **100.0 / 100.0 (Pontuação Máxima)** em todos os casos de treino oficiais:

| Cenário de Teste | Desafio Regulatório / Armadilha | Turnos Atendidos | Desfecho | Pontuação |
| :--- | :--- | :---: | :---: | :---: |
| **Caso 01 (Dermatologia)** | Anexo prévio ao contexto, recusa de terceiro (Art. 8º), coparticipação 40\% | 8 / 8 | 100\% Correto | **100.0 / 100.0** |
| **Caso 02 (Psicoterapia)** | 12ª sessão no histórico, exigência de laudo (Art. 73), teto anual (Art. 45) | 9 / 9 | 100\% Correto | **100.0 / 100.0** |
| **Caso 03 (Prótese OPME)** | Anexo inválido (conta de luz), reenvio correto, escalonamento pericial (Art. 78) | 7 / 7 | 100\% Correto | **100.0 / 100.0** |
| **Média Geral Consolidada**| **Todos os cenários do edital** | **24 / 24** | **100\%** | **100.0 / 100.0** |

---

<div align="center">
  <sub>Desenvolvido com excelência técnica para o Bootcamp AI Engineer 2026 (Prova 339EF4).</sub>
</div>
