"""Recuperação — carga do índice, busca híbrida (Vetorial + BM25) e lastro normativo."""

from .retriever import (
    buscar_tuss_por_descricao,
    carregar_docstore_nodes,
    carregar_tuss_tabela,
    consultar_normas,
    consultar_tuss,
    obter_dispositivos_validos,
    obter_indice,
    recuperar_evidencias,
)

__all__ = [
    "buscar_tuss_por_descricao",
    "carregar_docstore_nodes",
    "carregar_tuss_tabela",
    "consultar_normas",
    "consultar_tuss",
    "obter_dispositivos_validos",
    "obter_indice",
    "recuperar_evidencias",
]
