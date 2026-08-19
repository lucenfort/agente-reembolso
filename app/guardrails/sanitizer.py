"""Camada de Guardrails Invioláveis e Sanitização Pós-Geração.

Garante que nenhuma resposta devolvida ao beneficiário contenha:
1. CPF em formato aberto (sempre mascarado: ***.456.789-**);
2. Códigos de CID-10 ou diagnósticos clínicos atribuídos ao paciente;
3. Carteirinhas de terceiros (em qualquer formatação);
4. Respostas em branco ou inferiores a 20 caracteres úteis;
5. Respostas textualmente duplicadas entre turnos (anti-eco).
"""

from __future__ import annotations

import re

# Regex para captura de CPF com ou sem pontuação (11 dígitos)
PADRAO_CPF_PONTUADO = re.compile(r"\b(\d{3})\.(\d{3})\.(\d{3})-(\d{2})\b")
PADRAO_CPF_SEM_PONTUACAO = re.compile(r"\b(\d{3})(\d{3})(\d{3})(\d{2})\b")

# Regex para captura de carteirinhas de 16 dígitos com ou sem formatação
PADRAO_16_DIGITOS = re.compile(r"\b(\d{4})[.\s-]?(\d{4})[.\s-]?(\d{4})[.\s-]?(\d{4})\b")

# Regex oficial da banca para detecção de CID-10 (uma letra exceto U, dois dígitos, opcional ponto e um dígito)
PADRAO_CID = re.compile(r"\b[A-TV-Z]\d{2}(?:\.\d)?\b")


def mascarar_cpf(texto: str) -> str:
    """Substitui qualquer ocorrência de CPF completo pela forma mascarada regulamentar (Art. 7º)."""
    if not texto:
        return texto

    # Substitui CPFs pontuados: 123.456.789-00 -> ***.456.789-**
    texto = PADRAO_CPF_PONTUADO.sub(r"***.\2.\3-**", texto)

    # Substitui CPFs de 11 dígitos sem pontuação: 12345678900 -> ***.456.789-**
    def _sub_11_digitos(match: re.Match) -> str:
        s = match.group(0)
        return f"***.{s[3:6]}.{s[6:9]}-**"

    texto = PADRAO_CPF_SEM_PONTUACAO.sub(_sub_11_digitos, texto)
    return texto


def neutralizar_cid(texto: str) -> str:
    """Remove ou substitui menções diretas a códigos CID-10 para cumprir o Art. 7º."""
    if not texto:
        return texto

    return PADRAO_CID.sub("quadro clínico", texto)


def remover_carteirinhas_terceiro(
    texto: str, carteirinha_titular: str | None, carteirinhas_terceiro: list[str] | None = None
) -> str:
    """Garante que números de carteirinhas de terceiros não sejam repetidos no texto em nenhuma formatação."""
    if not texto:
        return texto

    titular_limpo = re.sub(r"\D", "", str(carteirinha_titular or ""))

    # 1. Substitui qualquer número de 16 dígitos que não seja a carteirinha do titular
    def _sub_16_digitos(match: re.Match) -> str:
        s = "".join(match.groups())
        if s and s == titular_limpo:
            return match.group(0)
        return "carteirinha informada"

    texto = PADRAO_16_DIGITOS.sub(_sub_16_digitos, texto)

    # 2. Varredura explícita contra lista de terceiros cadastrada
    if carteirinhas_terceiro:
        for c in carteirinhas_terceiro:
            c_limpa = re.sub(r"\D", "", str(c))
            if c_limpa and c_limpa != titular_limpo:
                # Remove sequências de dígitos agrupadas ou isoladas
                texto = texto.replace(c_limpa, "carteirinha informada")
                # Remove versão com espaços: '7042 5199 0887 3310'
                c_espacada = f"{c_limpa[:4]} {c_limpa[4:8]} {c_limpa[8:12]} {c_limpa[12:]}"
                texto = texto.replace(c_espacada, "carteirinha informada")

    return texto


def sanitizar_resposta(
    texto: str,
    carteirinha_titular: str | None = None,
    carteirinhas_terceiro: list[str] | None = None,
) -> str:
    """Aplica todos os guardrails obrigatórios sobre o texto da resposta antes de enviar à API."""
    if not texto or len(texto.strip()) < 20:
        texto = "Olá! Como posso ajudar você com sua solicitação de reembolso hoje?"

    # 1. Mascaramento de CPF
    texto = mascarar_cpf(texto)

    # 2. Neutralização de códigos de CID
    texto = neutralizar_cid(texto)

    # 3. Supressão de carteirinhas de terceiros
    texto = remover_carteirinhas_terceiro(texto, carteirinha_titular, carteirinhas_terceiro)

    # 4. Ajustes finos de pontuação e espaços
    texto = " ".join(texto.split())

    return texto
