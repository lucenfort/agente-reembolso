# Agente Autonomo de Reembolso - SaudeMais

Implementacao do Agente Autonomo de Reembolso para a SaudeMais Saude Suplementar (Desafio Tecnico Bootcamp 2026 - Prova 339EF4).

O sistema analisa solicitacoes de reembolso de beneficiarios de planos de saude em linguagem natural, valida elegibilidade e historico via Model Context Protocol (MCP), processa e classifica documentos fiscais/laudos (PDF/OCR), recupera dispositivos normativos aplicaveis via busca hibrida (RAG) e calcula deterministamente o valor a ser reembolsado, emitindo fundamentacao juridica ou escalonando para auditoria humana conforme as alcadas regulamentares.

---

## Sumario

1. [Arquitetura do Sistema](#arquitetura-do-sistema)
2. [Estrutura do Repositorio](#estrutura-do-repositorio)
3. [Documentacao Detalhada](#documentacao-detalhada)
4. [Pre-requisitos e Configuracao](#pre-requisitos-e-configuracao)
5. [Como Executar o Servidor MCP](#como-executar-o-servidor-mcp)
6. [Como Executar o Agente](#como-executar-o-agente)
7. [Construcao do Indice Normativo (Ingestao)](#construcao-do-indice-normativo-ingestao)
8. [Execucao dos Testes de Avaliacao](#execucao-dos-testes-de-avaliacao)
9. [Build e Execucao via Docker](#build-e-execucao-via-docker)

---

## Arquitetura do Sistema

A solucao e baseada em uma arquitetura multi-agente orquestrada com LangGraph e indexada via LlamaIndex:

- **Orquestracao:** Supervisor LangGraph com roteamento de estado e memoria persistente indexada por `session_id`.
- **Subagente de Triagem e Elegibilidade:** Comunicacao via HTTP Streamable com o servidor MCP da operadora (`consultar_beneficiario`, `consultar_historico`).
- **Subagente de Extracao e Analise Documental:** Processamento de PDFs e imagens via PyMuPDF / Tesseract OCR, identificacao de campos obrigatorios e codigos TUSS.
- **Subagente de Normas e RAG:** Busca hibrida (densa + BM25) sobre a base normativa (`kb/`) com citacao exata de artigos e circulares.
- **Motor Deterministico de Calculo:** Aplicacao de tetos URS, fatores de coparticipacao por plano (Prata, Ouro, Diamante) e controle de saldo anual de sessoes.
- **Guardrails de Seguranca e LGPD:** Mascaramento obrigatorio de CPF, supressao de codigos CID/diagnosticos clinicos e isolamento estrito contra acesso a dados de terceiros.

---

## Estrutura do Repositorio

```text
agente-reembolso/
├── app/                  # Nucleo da aplicacao FastAPI e logica dos agentes
│   ├── agents/           # Subagentes especializados (triagem, documento, normas, supervisor)
│   ├── calculo/          # Motor deterministico de calculo de coparticipacao e tetos URS
│   ├── guardrails/       # Validadores e filtros de LGPD, CPF e sigilo medico
│   ├── rag/              # Mecanismo de busca vetorial e hibrida
│   ├── tools/            # Clientes MCP e ferramentas auxiliares
│   ├── llm.py            # Configuracao unificada dos clientes LLM e Embeddings
│   ├── main.py           # Servidor HTTP (/health, /chat, /reset)
│   └── schemas.py        # Modelos Pydantic de requisicao e resposta
├── avaliacao/            # Motor de avaliacao oficial e juizes LLM
├── anexos/               # Arquivos anexos de exemplo para testes locais
├── casos_treino/         # Casos de teste estruturados (01, 02 e 03)
├── docs/                 # Documentacao tecnica tratada e sanitizada
│   ├── README.md         # Indice geral da documentacao
│   ├── instrucoes_da_prova.md
│   ├── especificacao_tecnica_e_contratos.md
│   ├── base_normativa_e_regras.md
│   └── guia_de_avaliacao_e_treino.md
├── ingest/               # Scripts de indexacao e preparacao do storage
├── kb/                   # Documentos oficiais da base de conhecimento
├── mcp/                  # Servidor Model Context Protocol da Operadora
├── storage/              # Indice vetorial persistido
├── docker-compose.yml    # Configuracao Docker Compose
├── Dockerfile            # Imagem de producao do container
├── requirements.txt      # Dependencias do projeto
└── rodar_treino.py       # Script de avaliacao local
```

---

## Documentacao Detalhada

Para consultar a documentacao completa e higienizada, acesse a pasta `docs/`:

- [Indice Geral da Documentacao](docs/README.md)
- [Instrucoes da Prova](docs/instrucoes_da_prova.md)
- [Especificacao Tecnica e Contratos de API](docs/especificacao_tecnica_e_contratos.md)
- [Base Normativa e Regras de Negocio](docs/base_normativa_e_regras.md)
- [Guia de Avaliacao e Casos de Treino](docs/guia_de_avaliacao_e_treino.md)

---

## Pre-requisitos e Configuracao

Requer **Python 3.11** e **Docker**.

1. Criar e ativar o ambiente virtual:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Validar conectividade com o gateway LLM:
   ```bash
   python -m app.llm
   ```

---

## Como Executar o Servidor MCP

O servidor MCP pode ser executado via Docker Compose ou diretamente em Python:

### Opcao 1: Docker Compose
```bash
docker compose up -d mcp
```

### Opcao 2: Python Local
```bash
cd mcp && MCP_OPERADORA_DADOS=../casos_treino MCP_OPERADORA_TOKEN=treino \
  ../.venv/bin/python -m mcp_operadora.server
```

Verifique se o servidor esta ativo na porta 9000:
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:9000/mcp
# Retorno esperado: 401 (indica servico ativo exigindo token de autenticacao)
```

---

## Como Executar o Agente

Inicie a API FastAPI localmente:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints disponiveis:
- `GET http://localhost:8000/health`
- `POST http://localhost:8000/chat`
- `POST http://localhost:8000/reset`

---

## Construcao do Indice Normativo (Ingestao)

Antes de construir o container Docker de entrega, execute o pipeline de ingestao para persistir o indice em `storage/`:
```bash
python ingest/build.py
```

---

## Execucao dos Testes de Avaliacao

Com o MCP (porta 9000) e o agente (porta 8000) em execucao:

```bash
# Executar todos os casos de treino com beneficiario simulado
python rodar_treino.py

# Modo detalhado (exibe mensagens de cada turno)
python rodar_treino.py -v

# Modo com roteiro fixo (determinista para depuracao)
python rodar_treino.py --roteiro-fixo

# Executar caso especifico
python rodar_treino.py --caso 01
```

---

## Build e Execucao via Docker

Para validar o pacote de entrega conforme exigido pela banca:

```bash
# Build da imagem (maximo 10 minutos)
docker build -t reembolso .

# Execucao do container na porta 8000
docker run -d --env-file .env -p 8000:8000 reembolso
```
