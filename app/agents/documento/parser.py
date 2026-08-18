"""Decodificação e extração de texto de anexos (PDF e Imagens/OCR)."""

from __future__ import annotations

import base64
import io
import logging
import re
import shutil
from pathlib import Path

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


def extrair_texto_de_base64(anexo_base64: str) -> tuple[str, str]:
    """Decodifica payload Base64 e extrai texto de PDF ou imagem.

    Retorna:
        tuple[texto_extraido, tipo_detectado]
    """
    if not anexo_base64:
        return "", "DESCONHECIDO"

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
    """Extrai texto de PDF com fallback para OCR em páginas escaneadas."""
    textos_paginas: list[str] = []
    try:
        doc = pymupdf.open(stream=dados_bytes, filetype="pdf")
        for num_pag, page in enumerate(doc):
            texto = page.get_text().strip()

            # Se a página contiver texto vetorial estruturado, utiliza diretamente
            if len(texto) >= 30:
                textos_paginas.append(texto)
                continue

            # Fallback para OCR caso a página seja imagem escaneada
            if _verificar_tesseract():
                try:
                    import pytesseract

                    pix = page.get_pixmap(dpi=150)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    texto_ocr = pytesseract.image_to_string(img, lang="por")
                    if texto_ocr.strip():
                        textos_paginas.append(texto_ocr.strip())
                        continue
                except Exception as e:
                    logger.warning(f"Falha ao executar OCR na página {num_pag + 1}: {e}")

            if texto:
                textos_paginas.append(texto)

    except Exception as e:
        logger.error(f"Erro ao processar PDF com PyMuPDF: {e}")

    return "\n\n".join(textos_paginas).strip()


def _extrair_texto_imagem(dados_bytes: bytes) -> str:
    """Extrai texto de imagem usando Tesseract OCR em português."""
    if not _verificar_tesseract():
        logger.warning("OCR requisitado para imagem, mas Tesseract não está instalado.")
        return ""

    try:
        import pytesseract

        img = Image.open(io.BytesIO(dados_bytes))
        texto_ocr = pytesseract.image_to_string(img, lang="por")
        return texto_ocr.strip()
    except Exception as e:
        logger.error(f"Erro ao executar OCR em imagem: {e}")
        return ""
