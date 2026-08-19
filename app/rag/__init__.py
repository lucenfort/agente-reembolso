"""Recuperação — carga do índice e busca híbrida."""

from .retriever import carregar_tuss_tabela, consultar_normas, consultar_tuss, obter_indice

__all__ = [
    "carregar_tuss_tabela",
    "consultar_normas",
    "consultar_tuss",
    "obter_indice",
]
