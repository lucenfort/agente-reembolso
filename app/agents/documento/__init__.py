"""Agente de Documento — decodificação, OCR, extração de campos e categorização NT-02."""

from __future__ import annotations

from typing import Any

from .extractor import extrair_informacoes_documento
from .parser import extrair_texto_de_base64
from .schemas import CategoriaReembolso, DocumentoInfo, TipoDocumento


def processar_anexo(anexo: Any) -> DocumentoInfo:
    """Processa anexo (seja string base64, dicionário ou objeto Anexo) e retorna os dados estruturados."""
    if hasattr(anexo, "base64"):
        b64_str = str(anexo.base64)
    elif isinstance(anexo, dict) and "base64" in anexo:
        b64_str = str(anexo["base64"])
    elif isinstance(anexo, str):
        b64_str = anexo
    else:
        b64_str = ""

    texto_extraido, tipo_mime = extrair_texto_de_base64(b64_str)
    return extrair_informacoes_documento(texto_extraido)


__all__ = [
    "processar_anexo",
    "extrair_texto_de_base64",
    "extrair_informacoes_documento",
    "DocumentoInfo",
    "TipoDocumento",
    "CategoriaReembolso",
]
