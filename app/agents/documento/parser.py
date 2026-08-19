"""Módulo de parsing e decodificação de anexos em Base64 (PDFs e Imagens)."""

from __future__ import annotations

import base64
import io
import logging
import re
import shutil
from typing import Any

import pymupdf
from PIL import Image

logger = logging.getLogger("app.agents.documento.parser")

_TESSERACT_DISPONIVEL: bool | None = None


def _verificar_tesseract() -> bool:
    global _TESSERACT_DISPONIVEL
    if _TESSERACT_DISPONIVEL is None:
        _TESSERACT_DISPONIVEL = bool(shutil.which("tesseract"))
        if not _TESSERACT_DISPONIVEL:
            logger.info("Executável 'tesseract' não encontrado no PATH; OCR em imagens será ignorado no host.")
    return _TESSERACT_DISPONIVEL


def extrair_texto_de_base64(anexo_input: Any) -> tuple[str, str]:
    """Decodifica payload Base64 (ou objeto Anexo / dict) e extrai texto de PDF ou imagem.

    Retorna:
        tuple[texto_extraido, tipo_detectado]
    """
    if not anexo_input:
        return "", "DESCONHECIDO"

    if hasattr(anexo_input, "base64"):
        anexo_base64 = str(anexo_input.base64)
    elif isinstance(anexo_input, dict) and "base64" in anexo_input:
        anexo_base64 = str(anexo_input["base64"])
    elif isinstance(anexo_input, str):
        anexo_base64 = anexo_input
    else:
        anexo_base64 = str(anexo_input)

    # Remove prefixo de data URI caso presente (ex: data:application/pdf;base64,...)
    anexo_limpo = anexo_base64.strip()
    if "," in anexo_limpo and anexo_limpo.startswith("data:"):
        anexo_limpo = anexo_limpo.split(",", 1)[1].strip()

    try:
        dados_bytes = base64.b64decode(anexo_limpo)
    except Exception as e:
        logger.error(f"Erro ao decodificar Base64: {e}")
        return "", "ERRO_DECODIFICACAO"

    if not dados_bytes:
        return "", "VAZIO"

    # 1. Detecção por Magic Bytes de PDF (%PDF)
    if dados_bytes.startswith(b"%PDF"):
        return _extrair_texto_pdf(dados_bytes), "PDF"

    # 2. Detecção por Imagem (PNG: \x89PNG, JPEG: \xff\xd8\xff, etc.)
    return _extrair_texto_imagem(dados_bytes), "IMAGEM"


def _extrair_texto_pdf(dados_bytes: bytes) -> str:
    """Extrai texto e tabelas de um arquivo PDF via PyMuPDF com fallback OCR."""
    textos: list[str] = []
    try:
        with pymupdf.open(stream=dados_bytes, filetype="pdf") as doc:
            for page_num, page in enumerate(doc):
                texto_pagina = page.get_text().strip()

                # Tenta extrair tabelas estruturadas da página se existirem
                tabs = page.find_tables()
                if tabs and len(tabs.tables) > 0:
                    for tab in tabs:
                        for row in tab.extract():
                            if row and any(row):
                                linha = " | ".join(str(c).strip() for c in row if c is not None)
                                if linha not in texto_pagina:
                                    textos.append(linha)

                if texto_pagina:
                    textos.append(texto_pagina)
                elif _verificar_tesseract():
                    # Fallback OCR para páginas que são imagens escaneadas
                    pix = page.get_pixmap(dpi=150)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    try:
                        import pytesseract

                        texto_ocr = pytesseract.image_to_string(img, lang="por+eng")
                        if texto_ocr.strip():
                            textos.append(texto_ocr.strip())
                    except Exception as e:
                        logger.warning(f"Falha no OCR da página {page_num}: {e}")

    except Exception as e:
        logger.error(f"Erro ao processar PDF com PyMuPDF: {e}")
        return ""

    return "\n\n".join(textos)


def _extrair_texto_imagem(dados_bytes: bytes) -> str:
    """Extrai texto de imagem usando Tesseract OCR se disponível."""
    if not _verificar_tesseract():
        logger.warning("OCR solicitado para imagem mas Tesseract não está disponível no PATH.")
        return ""

    try:
        import pytesseract

        img = Image.open(io.BytesIO(dados_bytes))
        texto_ocr = pytesseract.image_to_string(img, lang="por+eng")
        return texto_ocr.strip()
    except Exception as e:
        logger.error(f"Erro no OCR de imagem: {e}")
        return ""
