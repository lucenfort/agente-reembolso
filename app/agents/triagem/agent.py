"""Agente de Triagem e Elegibilidade.

Responsável por:
1. Extração de número de carteirinha (16 dígitos) das mensagens do beneficiário;
2. Distinção precisa entre carteirinha do titular e carteirinhas de terceiros/dependentes (Art. 8º);
3. Consulta de elegibilidade e cadastro no servidor MCP da operadora;
4. Consulta de histórico de sinistros e contagem de sessões no servidor MCP;
5. Identificação e bloqueio de consultas sobre terceiros / dependentes em sessão alheia (Art. 8º);
6. Verificação de status cadastral ativo do plano (Art. 10, I e Art. 30).
"""

from __future__ import annotations

import re
from typing import Any

from app.tools.mcp_client import obter_cliente_mcp

# Regex para carteirinha de 16 dígitos com ou sem espaços/pontos/hífens
PADRAO_CARTEIRINHA = re.compile(r"\b(\d{4})[.\s-]?(\d{4})[.\s-]?(\d{4})[.\s-]?(\d{4})\b")

# Marcadores lexicais de menção a terceiros/dependentes (Art. 8º)
PADRAO_MENCAO_TERCEIRO = re.compile(
    r"\b(esposa|marido|c[oô]njuge|dependente|filho|filha|m[aã]e|pai|irm[aã]o|irm[aã]|"
    r"namorad[oa]|noiv[oa]|sogr[oa]|amig[oa]|colega|terceiro|outra\s+pessoa|"
    r"carteirinha\s+del[ea]|reembolso\s+del[ea]|consulta\s+del[ea]|exame\s+del[ea])\b",
    re.IGNORECASE,
)


def extrair_carteirinhas(texto: str) -> list[str]:
    """Extrai todos os números de carteirinha de 16 dígitos encontrados no texto."""
    if not texto:
        return []
    matches = PADRAO_CARTEIRINHA.findall(texto)
    vistos: list[str] = []
    for m in matches:
        c = "".join(m)
        if c not in vistos:
            vistos.append(c)
    return vistos


def _separar_titular_e_terceiros(
    texto: str,
    carteirinhas: list[str],
    carteirinha_titular_atual: str | None,
) -> tuple[str | None, list[str]]:
    """Separa qual carteirinha pertence ao titular e quais pertencem a terceiros."""
    if not carteirinhas:
        return carteirinha_titular_atual, []

    if carteirinha_titular_atual:
        terceiros = [c for c in carteirinhas if c != carteirinha_titular_atual]
        return carteirinha_titular_atual, terceiros

    # Ainda não temos titular cadastrado na sessão.
    # Caso haja mais de uma carteirinha na mesma mensagem (ex: "a minha é X ... a da minha esposa é Y"):
    if len(carteirinhas) >= 2:
        titular_cand = None
        terceiros_cand: list[str] = []
        for m in PADRAO_CARTEIRINHA.finditer(texto):
            c = "".join(m.groups())
            inicio = max(0, m.start() - 55)
            contexto_anterior = texto[inicio : m.start()].lower()
            if any(
                p in contexto_anterior
                for p in (
                    "esposa",
                    "marido",
                    "cônjuge",
                    "conjuge",
                    "dependente",
                    "filho",
                    "filha",
                    "dela",
                    "dele",
                    "mãe",
                    "mae",
                    "pai",
                    "terceiro",
                )
            ):
                if c not in terceiros_cand:
                    terceiros_cand.append(c)
            elif titular_cand is None:
                titular_cand = c
            else:
                if c not in terceiros_cand:
                    terceiros_cand.append(c)

        if titular_cand is None:
            titular_cand = carteirinhas[0]
            terceiros_cand = [c for c in carteirinhas[1:] if c != titular_cand]
        return titular_cand, terceiros_cand

    # Apenas 1 carteirinha na mensagem e ainda não temos titular:
    unica = carteirinhas[0]
    # Verifica se o usuário está declarando que essa única carteirinha é de terceiro (sem informar a própria)
    m = PADRAO_CARTEIRINHA.search(texto)
    if m:
        inicio = max(0, m.start() - 55)
        contexto_anterior = texto[inicio : m.start()].lower()
        tem_marcador_proprio = any(p in contexto_anterior for p in ("minha", "meu número", "meu numero", "sou "))
        tem_marcador_terceiro = any(
            p in contexto_anterior
            for p in ("esposa", "marido", "cônjuge", "conjuge", "dependente", "filho", "filha", "dela", "dele")
        )
        if tem_marcador_terceiro and not tem_marcador_proprio:
            return None, [unica]

    return unica, []


class ResultadoTriagem:
    def __init__(
        self,
        carteirinha_identificada: str | None = None,
        beneficiario: dict[str, Any] | None = None,
        historico: dict[str, Any] | None = None,
        eh_terceiro: bool = False,
        carteirinha_terceiro: str | None = None,
        carteirinhas_terceiros: list[str] | None = None,
        resposta_direta: str | None = None,
        sucesso: bool = True,
        motivo: str = "",
    ):
        self.carteirinha_identificada = carteirinha_identificada
        self.beneficiario = beneficiario
        self.historico = historico
        self.eh_terceiro = eh_terceiro
        self.carteirinha_terceiro = carteirinha_terceiro
        self.carteirinhas_terceiros = carteirinhas_terceiros or (
            [carteirinha_terceiro] if carteirinha_terceiro else []
        )
        self.resposta_direta = resposta_direta
        self.sucesso = sucesso
        self.motivo = motivo


def processar_triagem(
    mensagem: str,
    carteirinha_titular_atual: str | None = None,
    beneficiario_atual: dict[str, Any] | None = None,
    historico_atual: dict[str, Any] | None = None,
) -> ResultadoTriagem:
    """Executa a triagem da mensagem do usuário, validando carteirinhas e consultando MCP."""
    carteirinhas = extrair_carteirinhas(mensagem)
    titular_identificado, terceiros_identificados = _separar_titular_e_terceiros(
        mensagem, carteirinhas, carteirinha_titular_atual
    )

    menciona_terceiro_texto = bool(PADRAO_MENCAO_TERCEIRO.search(mensagem or ""))
    eh_terceiro = bool(terceiros_identificados) or menciona_terceiro_texto
    primeiro_terceiro = terceiros_identificados[0] if terceiros_identificados else None

    beneficiario = beneficiario_atual
    historico = historico_atual

    # Se identificamos uma nova carteirinha de titular neste turno, consultamos o MCP APENAS para o titular!
    # Nunca consultamos o MCP para carteirinhas de terceiros (Art. 8º).
    if titular_identificado and titular_identificado != carteirinha_titular_atual:
        mcp = obter_cliente_mcp()
        beneficiario = mcp.consultar_beneficiario(titular_identificado)
        historico = mcp.consultar_historico(titular_identificado)

        if not beneficiario or beneficiario.get("status") == "NAO_ENCONTRADO":
            return ResultadoTriagem(
                carteirinha_identificada=None,
                eh_terceiro=eh_terceiro,
                carteirinha_terceiro=primeiro_terceiro,
                carteirinhas_terceiros=terceiros_identificados,
                sucesso=False,
                resposta_direta=(
                    "Não consegui localizar um cadastro ativo para o número de carteirinha informado. "
                    "Por gentileza, confira os 16 dígitos e envie novamente."
                ),
                motivo="Carteirinha não localizada no MCP.",
            )

        if beneficiario.get("status") in ["CANCELADO", "SUSPENSO"]:
            return ResultadoTriagem(
                carteirinha_identificada=titular_identificado,
                beneficiario=beneficiario,
                historico=historico,
                eh_terceiro=eh_terceiro,
                carteirinha_terceiro=primeiro_terceiro,
                carteirinhas_terceiros=terceiros_identificados,
                sucesso=False,
                resposta_direta=(
                    f"Identifiquei o cadastro de {beneficiario.get('nome')}, contudo o plano consta como "
                    f"{beneficiario.get('status')}. Conforme o Art. 10, I e o Art. 30 do Regulamento Geral, "
                    "contratos suspensos ou cancelados não geram direito a reembolso."
                ),
                motivo="Plano inativo ou cancelado (Art. 30).",
            )

    motivo = "Elegibilidade confirmada via MCP." if titular_identificado else "Aguardando número de carteirinha."
    if eh_terceiro:
        motivo = "Menção/consulta de terceiro bloqueada pelo Art. 8º (LGPD e sigilo por sessão)."

    return ResultadoTriagem(
        carteirinha_identificada=titular_identificado,
        beneficiario=beneficiario,
        historico=historico,
        eh_terceiro=eh_terceiro,
        carteirinha_terceiro=primeiro_terceiro,
        carteirinhas_terceiros=terceiros_identificados,
        sucesso=True,
        motivo=motivo,
    )
