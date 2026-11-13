"""Agente de Documento — decodificação, OCR, extração de campos e categorização NT-02."""

from __future__ import annotations

from typing import Any

from .extractor import extrair_informacoes_documento, processar_anexo
from .parser import extrair_texto_de_base64
from .schemas import CategoriaReembolso, DocumentoInfo, TipoDocumento

__all__ = [
    "processar_anexo",
    "extrair_texto_de_base64",
    "extrair_informacoes_documento",
    "DocumentoInfo",
    "TipoDocumento",
    "CategoriaReembolso",
]
