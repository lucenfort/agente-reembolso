"""Recuperador de normas e base de conhecimento (RAG).

Carrega o índice vetorial pré-compilado de storage/ e executa buscas cirúrgicas
para abastecer o Agente de Normas sem sobrecarregar a cota de tokens (máx ~3000 tokens).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.indices.vector_store import VectorStoreIndex

from app.llm import criar_embeddings_llamaindex

RAIZ = Path(__file__).resolve().parents[2]
DIR_STORAGE = RAIZ / "storage"

_INDICE: VectorStoreIndex | None = None
_TUSS_TABELA: dict[str, dict[str, Any]] | None = None


def carregar_tuss_tabela() -> dict[str, dict[str, Any]]:
    """Carrega o dicionário estruturado de procedimentos TUSS em memória."""
    global _TUSS_TABELA
    if _TUSS_TABELA is not None:
        return _TUSS_TABELA

    caminho = DIR_STORAGE / "tuss_tabela.json"
    if caminho.exists():
        with open(caminho, "r", encoding="utf-8") as f:
            _TUSS_TABELA = json.load(f)
    else:
        _TUSS_TABELA = {}
    return _TUSS_TABELA


def obter_indice() -> VectorStoreIndex:
    """Carrega o índice do storage de forma singleton."""
    global _INDICE
    if _INDICE is not None:
        return _INDICE

    Settings.embed_model = criar_embeddings_llamaindex()
    storage_context = StorageContext.from_defaults(persist_dir=str(DIR_STORAGE))
    _INDICE = load_index_from_storage(storage_context)
    return _INDICE


@lru_cache(maxsize=128)
def consultar_normas(query: str, top_k: int = 4) -> str:
    """Recupera trechos normativos relevantes para uma dúvida ou tópico (com cache LRU em memória)."""
    try:
        index = obter_indice()
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

        trechos = []
        for n in nodes:
            texto = n.node.get_content().strip()
            meta = n.node.metadata
            doc_name = meta.get("doc_name", "")
            dispositivo = meta.get("dispositivo", "")
            rotulo = f"[{doc_name} | {dispositivo}]" if (doc_name or dispositivo) else ""
            trechos.append(f"{rotulo}\n{texto}")

        return "\n\n---\n\n".join(trechos)
    except Exception as e:
        # Fallback gracioso caso haja erro de rede / embeddings
        return f"Não foi possível consultar a base vetorial no momento ({e})."


def consultar_tuss(codigo_tuss: str) -> dict[str, Any] | None:
    """Busca dados de um procedimento diretamente na tabela TUSS estruturada."""
    tabela = carregar_tuss_tabela()
    return tabela.get(codigo_tuss.strip())
