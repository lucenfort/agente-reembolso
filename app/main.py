"""API do agente de reembolso.

Endpoints obrigatórios:
- GET /health: Healthcheck da aplicação (< 5s cold start)
- POST /chat: Turno de conversa com orquestração LangGraph
- POST /reset: Limpeza de memória e sessões
"""

from __future__ import annotations

from fastapi import FastAPI

from app.agents.supervisor import executar_turno, resetar_estado
from app.llm import carregar_env
from app.schemas import ChatRequest, ChatResponse

# Lê o `.env` no start. O que já vem do ambiente vence — é assim que a banca
# injeta as credenciais dela no `docker run`.
carregar_env()

app = FastAPI(title="Agente de Reembolso - SaúdeMais")


@app.get("/health")
def health() -> dict:
    """Precisa responder 200 em até 60 segundos do start do container."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Um turno de conversa orquestrado pelo grafo LangGraph."""
    return executar_turno(req)


@app.post("/reset")
def reset() -> dict:
    """Limpa estado e sessões entre conversas."""
    return resetar_estado()
