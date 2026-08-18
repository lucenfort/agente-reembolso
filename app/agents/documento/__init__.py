"""Agente de Documento — decodificação, OCR, extração de campos e categorização NT-02."""

from __future__ import annotations

from .extractor import extrair_informacoes_documento
from .parser import extrair_texto_de_base64
from .schemas import CategoriaReembolso, DocumentoInfo, TipoDocumento


def processar_anexo(anexo_base64: str) -> DocumentoInfo:
    """Processa payload Base64 de anexo (PDF ou imagem) e retorna os dados estruturados."""
    texto_extraido, tipo_mime = extrair_texto_de_base64(anexo_base64)
    return extrair_informacoes_documento(texto_extraido)


__all__ = [
    "processar_anexo",
    "extrair_texto_de_base64",
    "extrair_informacoes_documento",
    "DocumentoInfo",
    "TipoDocumento",
    "CategoriaReembolso",
]
