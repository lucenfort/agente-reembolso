"""Definição do estado compartilhado do Supervisor Multi-Agente (LangGraph)."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Estado do agente mantido pelo checkpointer do LangGraph indexado por session_id."""

    session_id: str
    mensagens_historico: list[dict[str, str]]
    carteirinha_titular: str | None
    carteirinhas_terceiro: list[str]
    beneficiario: dict[str, Any] | None
    historico_mcp: dict[str, Any] | None
    documentos_processados: list[dict[str, Any]]
    documento_principal: dict[str, Any] | None
    protocolo_mcp: str | None
    categoria_documento: str | None
    decisao: str | None
    valor_solicitado_brl: float | None
    valor_reembolso_brl: float | None
    regras_aplicadas: list[str]
    pendencias: list[str]
    resposta_gerada: str
    respostas_anteriores: list[str]
    turno_atual: int
    mensagem_atual: str
    anexo_atual: dict[str, Any] | None
    recusa_terceiro_neste_turno: bool
    motivo_pendencia: str | None
    etapa_bloqueio: str | None
    apto_para_calculo: bool
    detalhes_calculo: dict[str, Any] | None
    evidencias_rag: list[dict[str, Any]]
    dispositivos_rag: list[str]
