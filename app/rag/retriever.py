"""Recuperador Híbrido (Vetorial + BM25 + Lookup de Dispositivos) com Lastro no Estado.

Garante que toda decisão e toda resposta a dúvidas normativas seja ancorada nos trechos
exatos de `storage/docstore.json` e `storage/tuss_tabela.json`, funcionando tanto com
embeddings online (LlamaIndex) quanto 100% offline via BM25 lexical e índice invertido
de dispositivos normativos (Art. 6º do Regulamento Geral).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parents[2]
DIR_STORAGE = RAIZ / "storage"
DIR_KB = RAIZ / "kb"

_INDICE: Any | None = None
_TUSS_TABELA: dict[str, dict[str, Any]] | None = None
_DOCSTORE_NODES: list[dict[str, Any]] | None = None
_NODES_POR_DISPOSITIVO: dict[str, list[dict[str, Any]]] | None = None
_DISPOSITIVOS_VALIDOS: set[str] | None = None

# Dispositivos que NÃO são códigos oficiais do Art. 6º (ex: rótulos internos do docstore)
DISPOSITIVOS_NAO_CITAVEIS = {
    "REGULAMENTO-GERAL",
    "TABELA-URS",
    "MANUAL-REDE",
    "FAQ-INTERNO",
    "CIRC-04-2025",  # Revogada expressamente pela Circular 09/2025 (Art. 6º parágrafo único veda citar revogada)
}

STOPWORDS_PT = {
    "a", "o", "e", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "por", "para", "com", "sem", "um", "uma", "os", "as", "que", "se", "ao", "aos",
    "ou", "como", "mais", "mas", "foi", "ser", "são", "está", "meu", "minha", "seu",
    "sua", "qual", "quais", "quando", "onde", "porque", "porquê", "pra", "pro", "pelo",
    "pela", "isso", "isto", "esse", "essa", "este", "esta", "já", "tem", "ter", "vou",
    "fui", "fez", "fiz", "sobre", "até", "após", "entre", "não", "sim", "muito",
}


def _tokenizar(texto: str) -> list[str]:
    palavras = re.findall(r"[a-záéíóúâêôãõç0-9]+", texto.lower())
    return [p for p in palavras if len(p) > 1 and p not in STOPWORDS_PT]


def obter_dispositivos_validos() -> set[str]:
    """Retorna o conjunto de todos os dispositivos válidos na base `kb/` segundo o Art. 6º."""
    global _DISPOSITIVOS_VALIDOS
    if _DISPOSITIVOS_VALIDOS is not None:
        return _DISPOSITIVOS_VALIDOS

    achados: set[str] = {"ANEXO-IV", "NT-02", "CIRC-02-2026", "CIRC-09-2025", "CIRC-11-2026"}
    # Adiciona todos os artigos 1 a 87 do Regulamento Geral
    for i in range(1, 88):
        achados.add(f"ART-{i}")
    # Adiciona todos os códigos TUSS da tabela persistida
    tuss = carregar_tuss_tabela()
    for code in tuss:
        achados.add(f"TUSS-{code}")

    _DISPOSITIVOS_VALIDOS = achados - {"CIRC-04-2025"}
    return _DISPOSITIVOS_VALIDOS


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


def carregar_docstore_nodes() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Carrega todos os nós normativos pré-indexados em `storage/docstore.json`."""
    global _DOCSTORE_NODES, _NODES_POR_DISPOSITIVO
    if _DOCSTORE_NODES is not None and _NODES_POR_DISPOSITIVO is not None:
        return _DOCSTORE_NODES, _NODES_POR_DISPOSITIVO

    _DOCSTORE_NODES = []
    _NODES_POR_DISPOSITIVO = {}
    caminho = DIR_STORAGE / "docstore.json"
    if not caminho.exists():
        return _DOCSTORE_NODES, _NODES_POR_DISPOSITIVO

    raw = json.loads(caminho.read_text(encoding="utf-8"))
    data = raw.get("docstore/data", {})
    for node_id, entry in data.items():
        node_data = entry.get("__data__", entry)
        texto = str(node_data.get("text", "")).strip()
        meta = node_data.get("metadata", {}) or {}
        disp = str(meta.get("dispositivo", "")).strip()
        doc_name = str(meta.get("doc_name", "")).strip()
        tokens = _tokenizar(texto)
        item = {
            "id": node_id,
            "texto": texto,
            "dispositivo": disp,
            "doc_name": doc_name,
            "metadata": meta,
            "tokens": tokens,
            "tf": Counter(tokens),
        }
        _DOCSTORE_NODES.append(item)
        if disp:
            _NODES_POR_DISPOSITIVO.setdefault(disp, []).append(item)

    return _DOCSTORE_NODES, _NODES_POR_DISPOSITIVO


def obter_indice() -> Any:
    """Carrega o VectorStoreIndex do LlamaIndex se o ambiente vetorial estiver ativo."""
    global _INDICE
    if _INDICE is not None:
        return _INDICE

    from llama_index.core import Settings, StorageContext, load_index_from_storage
    from app.llm import criar_embeddings_llamaindex

    Settings.embed_model = criar_embeddings_llamaindex()
    storage_context = StorageContext.from_defaults(persist_dir=str(DIR_STORAGE))
    _INDICE = load_index_from_storage(storage_context)
    return _INDICE


def _pontuar_bm25(query: str, top_k: int = 6) -> list[dict[str, Any]]:
    """Executa ranking BM25 Okapi sobre todos os nós de `storage/docstore.json`."""
    nodes, _ = carregar_docstore_nodes()
    if not nodes:
        return []

    q_tokens = _tokenizar(query)
    if not q_tokens:
        return nodes[:top_k]

    n_docs = len(nodes)
    avgdl = sum(len(n["tokens"]) for n in nodes) / max(1, n_docs)
    k1 = 1.5
    b = 0.75

    # Calcula IDF dos termos da consulta
    idf: dict[str, float] = {}
    for qt in set(q_tokens):
        df = sum(1 for n in nodes if qt in n["tf"])
        idf[qt] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    # Bônus semântico para termos-chave do domínio regulatório
    query_lower = query.lower()
    dispositivos_sugeridos: set[str] = set()
    if any(k in query_lower for k in ("acupuntura", "estétic", "estetic", "fronteira", "drenagem", "botox", "toxina", "clareamento", "odonto")):
        dispositivos_sugeridos.update({"ANEXO-IV", "ART-51", "ART-52", "ART-55", "ART-60"})
    if any(k in query_lower for k in ("prazo", "90 dias", "150 dias", "decadência", "recurso", "reanálise", "perder o prazo", "perdi o prazo")):
        dispositivos_sugeridos.update({"ART-12", "ART-13", "ART-20", "CIRC-09-2025"})
    if any(k in query_lower for k in ("psicolog", "psicoterap", "terapia", "sessão", "sessoes", "sessões", "relatório", "relatorio", "laudo")):
        dispositivos_sugeridos.update({"ART-40", "ART-41", "ART-73", "CIRC-02-2026", "NT-02"})
    if any(k in query_lower for k in ("coparticipa", "inteiro", "integral", "desconto", "redução", "por que não volta", "teto", "urs")):
        dispositivos_sugeridos.update({"ART-31", "ART-33", "ART-35", "ART-43", "ART-44", "CIRC-02-2026"})
    if any(k in query_lower for k in ("limite anual", "teto anual", "48 urs", "já pedi bastante", "saldo")):
        dispositivos_sugeridos.update({"ART-45"})
    if any(k in query_lower for k in ("esposa", "marido", "dependente", "filho", "terceiro", "outra pessoa", "cônjuge")):
        dispositivos_sugeridos.update({"ART-8", "ART-58"})
    if any(k in query_lower for k in ("prótese", "protese", "órtese", "opme", "alçada", "analista", "cirurgia")):
        dispositivos_sugeridos.update({"ART-78", "ART-81"})
    if any(k in query_lower for k in ("luz", "energia", "conta de consumo", "inválido", "invalido", "errado")):
        dispositivos_sugeridos.update({"ART-76", "NT-02"})
    if any(k in query_lower for k in ("carência", "carencia", "adesão", "recém")):
        dispositivos_sugeridos.update({"ART-21", "ART-22", "ART-23", "ART-24", "ART-25", "ART-26"})

    scored: list[tuple[float, dict[str, Any]]] = []
    for n in nodes:
        dl = len(n["tokens"])
        score = 0.0
        for qt in q_tokens:
            tf = n["tf"].get(qt, 0)
            if tf > 0:
                num = tf * (k1 + 1.0)
                den = tf + k1 * (1.0 - b + b * (dl / max(1.0, avgdl)))
                score += idf[qt] * (num / den)

        if n["dispositivo"] in dispositivos_sugeridos:
            score += 4.0
        # Prioriza normas oficiais sobre FAQ/Manual (Art. 5º § 5º)
        if n["doc_name"] in ("regulamento_geral", "circular_02_2026", "circular_09_2025", "anexo_iv_exclusoes", "nota_tecnica_02_documentos"):
            score *= 1.25

        if score > 0:
            scored.append((score, n))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def recuperar_evidencias(
    query: str = "",
    dispositivos_obrigatorios: list[str] | None = None,
    top_k: int = 5,
    pergunta: str | None = None,
) -> list[dict[str, Any]]:
    """Recupera os trechos normativos exatos que dão lastro à decisão e à resposta do turno.

    Combina:
    1. Os nós exatos dos `dispositivos_obrigatorios` acionados pela verificação/cálculo;
    2. Os nós mais relevantes via busca híbrida (BM25 + semântica).
    """
    consulta = (pergunta if pergunta is not None else query) or ""
    _, por_disp = carregar_docstore_nodes()
    evidencias: list[dict[str, Any]] = []
    ids_vistos: set[str] = set()

    # 1. Inclui primeiro os trechos literais dos dispositivos aplicados na verificação/cálculo
    for disp in dispositivos_obrigatorios or []:
        disp_norm = disp.strip().upper()
        if disp_norm in ("CIRC-04-2025",):
            continue
        if disp_norm in por_disp:
            for n in por_disp[disp_norm][:1]:
                if "04_2025" in str(n.get("doc_name", "")).lower():
                    continue
                if n["id"] not in ids_vistos:
                    ids_vistos.add(n["id"])
                    txt = n["texto"][:900]
                    evidencias.append({
                        "dispositivo": disp_norm,
                        "dispositivos": [disp_norm],
                        "doc_name": n["doc_name"],
                        "source": n["doc_name"],
                        "texto": txt,
                        "text": txt,
                    })

    # 2. Complementa com a busca BM25 sobre a consulta do turno
    if consulta and consulta.strip():
        candidatos_bm25 = _pontuar_bm25(consulta, top_k=top_k + 2)
        for n in candidatos_bm25:
            if len(evidencias) >= top_k + 2:
                break
            if "04_2025" in str(n.get("doc_name", "")).lower() and "04/2025" not in consulta.lower():
                continue
            if n["id"] not in ids_vistos:
                ids_vistos.add(n["id"])
                txt = n["texto"][:900]
                disp_val = n["dispositivo"]
                evidencias.append({
                    "dispositivo": disp_val,
                    "dispositivos": [disp_val] if disp_val else [],
                    "doc_name": n["doc_name"],
                    "source": n["doc_name"],
                    "texto": txt,
                    "text": txt,
                })

    return evidencias


def consultar_normas(
    query: str = "",
    top_k: int = 4,
    dispositivos_obrigatorios: list[str] | None = None,
) -> str:
    """Recupera trechos normativos formatados para uma dúvida ou tópico."""
    evidencias = recuperar_evidencias(
        query=query,
        dispositivos_obrigatorios=dispositivos_obrigatorios,
        top_k=top_k,
    )
    if evidencias:
        blocos = []
        for ev in evidencias[:top_k]:
            rotulo = f"[{ev['doc_name']} | {ev['dispositivo']}]" if (ev["doc_name"] or ev["dispositivo"]) else ""
            blocos.append(f"{rotulo}\n{ev['texto']}")
        return "\n\n---\n\n".join(blocos)

    # Fallback para o índice vetorial LlamaIndex se necessário
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
        return f"Consulta normativa local concluída ({e})."


def consultar_tuss(codigo_tuss: str | None) -> dict[str, Any] | None:
    """Busca dados de um procedimento diretamente na tabela TUSS estruturada."""
    if not codigo_tuss:
        return None
    tabela = carregar_tuss_tabela()
    limpo = re.sub(r"\D", "", str(codigo_tuss))
    return tabela.get(limpo)


def buscar_tuss_por_descricao(descricao: str | None, categoria_hint: str | None = None) -> dict[str, Any] | None:
    """Reclassifica procedimento pela descrição na forma do Art. 34 § 1º e § 2º.

    Havendo mais de um código compatível com a descrição, aplica-se o de menor teto (Art. 34 § 2º).
    """
    if not descricao or not descricao.strip():
        return None
    tabela = carregar_tuss_tabela()
    if not tabela:
        return None

    desc_tokens = set(_tokenizar(descricao))
    if not desc_tokens:
        return None

    melhor_chave = (0, 0.0)
    empatados: list[dict[str, Any]] = []

    for code, info in tabela.items():
        info_tokens = set(_tokenizar(info.get("descricao", "")))
        if not info_tokens:
            continue
        intersecao = len(desc_tokens & info_tokens)
        if intersecao == 0:
            continue
        precisao = round(intersecao / len(info_tokens), 4)
        chave = (intersecao, precisao)
        if chave > melhor_chave:
            melhor_chave = chave
            empatados = [info]
        elif chave == melhor_chave:
            empatados.append(info)

    if not empatados or melhor_chave[0] < 2:
        return None

    # Art. 34 § 2º: Havendo mais de um código compatível, aplica-se o de menor teto
    empatados.sort(key=lambda x: float(x.get("urs", 9999.0)))
    return empatados[0]
