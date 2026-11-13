"""API do agente de reembolso.

Endpoints obrigatórios:
- GET /health: Healthcheck da aplicação (< 5s cold start)
- POST /chat: Turno de conversa com orquestração LangGraph
- POST /reset: Limpeza de memória e sessões
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app.agents.supervisor import executar_turno, resetar_estado
from app.llm import carregar_env, detectar_provedor
from app.schemas import ChatRequest, ChatResponse

# Lê o `.env` no start. O que já vem do ambiente vence — é assim que a banca
# injeta as credenciais dela no `docker run`.
carregar_env()

RAIZ = Path(__file__).resolve().parents[1]
INDEX_HTML = Path(__file__).resolve().parent / "static" / "index.html"
DIR_DEMO_ANEXOS = RAIZ / "anexos" / "treino"

app = FastAPI(title="Agente de Reembolso - SaúdeMais")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> HTMLResponse:
    """Interface Web interativa (Chat + Painel de Auditoria em tempo real)."""
    if INDEX_HTML.exists():
        return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Agente de Reembolso - SaúdeMais</h1><p>Acesse <a href='/docs'>/docs</a>.</p>")


@app.get("/demo/anexos", include_in_schema=False)
def listar_anexos_demo() -> dict:
    """Lista os arquivos de demonstração disponíveis em anexos/treino e o provedor ativo."""
    arquivos = sorted(p.name for p in DIR_DEMO_ANEXOS.iterdir() if p.is_file()) if DIR_DEMO_ANEXOS.exists() else []
    return {"arquivos": arquivos, "provider": detectar_provedor()}


@app.get("/demo/anexos/{filename}", include_in_schema=False)
def obter_anexo_demo(filename: str) -> dict:
    """Retorna um arquivo de demonstração codificado em Base64 para anexo em 1 clique na UI."""
    nome_seguro = Path(filename).name
    caminho = (DIR_DEMO_ANEXOS / nome_seguro).resolve()
    if not DIR_DEMO_ANEXOS.exists() or caminho.parent != DIR_DEMO_ANEXOS.resolve() or not caminho.is_file():
        raise HTTPException(status_code=404, detail="Anexo de demonstração não encontrado.")
    mime, _ = mimetypes.guess_type(caminho.name)
    return {
        "filename": caminho.name,
        "mime_type": mime or "application/pdf",
        "base64": base64.b64encode(caminho.read_bytes()).decode("ascii"),
    }


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
