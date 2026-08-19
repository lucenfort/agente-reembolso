"""Cálculo — a apuração do valor a reembolsar."""

from .motor import (
    ALCADA_AUTOMATIZADA_MAX,
    CARENCIA_DIAS,
    TETO_ANUAL_GLOBAL_BRL,
    TETO_CONSULTA_BRL,
    TETO_PSICOTERAPIA_BRL,
    VALOR_URS_2026,
    ResultadoCalculo,
    apurar_reembolso,
    calcular_dias_adesao,
    calcular_meses_adesao,
    obter_fator_coparticipacao,
    obter_teto_procedimento,
)

__all__ = [
    "ALCADA_AUTOMATIZADA_MAX",
    "CARENCIA_DIAS",
    "TETO_ANUAL_GLOBAL_BRL",
    "TETO_CONSULTA_BRL",
    "TETO_PSICOTERAPIA_BRL",
    "VALOR_URS_2026",
    "ResultadoCalculo",
    "apurar_reembolso",
    "calcular_dias_adesao",
    "calcular_meses_adesao",
    "obter_fator_coparticipacao",
    "obter_teto_procedimento",
]
