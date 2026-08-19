"""Agente de Triagem e Elegibilidade.

Responsável por:
1. Extração de número de carteirinha (16 dígitos) das mensagens do beneficiário;
2. Consulta de elegibilidade e cadastro no servidor MCP da operadora;
3. Consulta de histórico de sinistros e contagem de sessões no servidor MCP;
4. Identificação e bloqueio de consultas sobre terceiros / dependentes em sessão alheia (Art. 8º);
5. Verificação de status cadastral ativo do plano.
"""

from __future__ import annotations

import re
from typing import Any

from app.tools.mcp_client import obter_cliente_mcp

# Regex para carteirinha de 16 dígitos com ou sem espaços/pontos
PADRAO_CARTEIRINHA = re.compile(r"\b(\d{4})[.\s-]?(\d{4})[.\s-]?(\d{4})[.\s-]?(\d{4})\b")


def extrair_carteirinhas(texto: str) -> list[str]:
    """Extrai todos os números de carteirinha de 16 dígitos encontrados no texto."""
    if not texto:
        return []
    matches = PADRAO_CARTEIRINHA.findall(texto)
    return ["".join(m) for m in matches]


class ResultadoTriagem:
    def __init__(
        self,
        carteirinha_identificada: str | None = None,
        beneficiario: dict[str, Any] | None = None,
        historico: dict[str, Any] | None = None,
        eh_terceiro: bool = False,
        carteirinha_terceiro: str | None = None,
        resposta_direta: str | None = None,
        sucesso: bool = True,
        motivo: str = "",
    ):
        self.carteirinha_identificada = carteirinha_identificada
        self.beneficiario = beneficiario
        self.historico = historico
        self.eh_terceiro = eh_terceiro
        self.carteirinha_terceiro = carteirinha_terceiro
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

    # 1. Checagem de Terceiros (Art. 8º)
    if carteirinha_titular_atual and carteirinhas:
        for c in carteirinhas:
            if c != carteirinha_titular_atual:
                # O usuário mencionou carteirinha diferente da do titular em atendimento
                resposta_recusa = (
                    "Em atenção ao Art. 8º do Regulamento Geral e às normas de privacidade e sigilo, "
                    "este atendimento é estritamente vinculado à sua carteirinha e não é possível consultar, "
                    "tratar ou prestar informações sobre reembolsos de terceiros, dependentes ou cônjuges nesta sessão. "
                    "Para solicitações de outros beneficiários, por gentileza solicite a abertura de um atendimento "
                    "exclusivo com a respectiva carteirinha. Como posso prosseguir com a sua solicitação atual?"
                )
                return ResultadoTriagem(
                    carteirinha_identificada=carteirinha_titular_atual,
                    beneficiario=beneficiario_atual,
                    historico=historico_atual,
                    eh_terceiro=True,
                    carteirinha_terceiro=c,
                    resposta_direta=resposta_recusa,
                    sucesso=True,
                    motivo="Consulta de terceiro bloqueada pelo Art. 8º.",
                )

    # 2. Se já temos titular identificado e não há nova carteirinha válida
    if carteirinha_titular_atual and not carteirinhas:
        return ResultadoTriagem(
            carteirinha_identificada=carteirinha_titular_atual,
            beneficiario=beneficiario_atual,
            historico=historico_atual,
            eh_terceiro=False,
            sucesso=True,
        )

    # 3. Se foi fornecida uma carteirinha e ainda não tínhamos titular
    if carteirinhas:
        carteirinha = carteirinhas[0]
        mcp = obter_cliente_mcp()

        beneficiario = mcp.consultar_beneficiario(carteirinha)
        historico = mcp.consultar_historico(carteirinha)

        if not beneficiario or beneficiario.get("status") == "NAO_ENCONTRADO":
            return ResultadoTriagem(
                carteirinha_identificada=None,
                eh_terceiro=False,
                sucesso=False,
                resposta_direta=(
                    "Não consegui localizar um cadastro ativo para o número de carteirinha informado. "
                    "Por gentileza, confira os 16 dígitos e envie novamente."
                ),
                motivo="Carteirinha não localizada no MCP.",
            )

        if beneficiario.get("status") in ["CANCELADO", "SUSPENSO"]:
            return ResultadoTriagem(
                carteirinha_identificada=carteirinha,
                beneficiario=beneficiario,
                historico=historico,
                eh_terceiro=False,
                sucesso=False,
                resposta_direta=(
                    f"Identifiquei o cadastro de {beneficiario.get('nome')}, contudo o plano consta como "
                    f"{beneficiario.get('status')}. Conforme o Art. 30 do Regulamento, contratos não ativos "
                    "não geram direito a reembolso."
                ),
                motivo="Plano inativo ou cancelado (Art. 30).",
            )

        return ResultadoTriagem(
            carteirinha_identificada=carteirinha,
            beneficiario=beneficiario,
            historico=historico,
            eh_terceiro=False,
            sucesso=True,
            motivo="Elegibilidade confirmada via MCP.",
        )

    # 4. Nenhuma carteirinha foi enviada na mensagem
    return ResultadoTriagem(
        carteirinha_identificada=None,
        beneficiario=None,
        historico=None,
        eh_terceiro=False,
        sucesso=True,
        motivo="Aguardando número de carteirinha.",
    )
