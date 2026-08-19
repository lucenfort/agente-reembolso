"""Motor de cálculo determinístico de reembolso.

Implementa integralmente as regras do Regulamento Geral de Reembolso da SaúdeMais,
as alterações da Circular Normativa 02/2026, a Tabela URS 2026 e os critérios da NT-02.
Toda matemática é feita com Decimal e precisão exata de centavos (tolerância R$ 0,01).
"""

from __future__ import annotations

import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from app.agents.documento.schemas import DocumentoInfo
from app.schemas import Categoria, Decisao

# Constantes Regulamentares
VALOR_URS_2026 = Decimal("95.10")
TETO_ANUAL_GLOBAL_URS = Decimal("48")
TETO_ANUAL_GLOBAL_BRL = (TETO_ANUAL_GLOBAL_URS * VALOR_URS_2026).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)  # R$ 4.564,80

TETO_CONSULTA_URS = Decimal("5.0")
TETO_CONSULTA_BRL = (TETO_CONSULTA_URS * VALOR_URS_2026).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)  # R$ 475,50

# Circular 02/2026 altera teto de psicoterapia individual para 2,8 URS
TETO_PSICOTERAPIA_URS = Decimal("2.8")
TETO_PSICOTERAPIA_BRL = (TETO_PSICOTERAPIA_URS * VALOR_URS_2026).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)  # R$ 266,28

# Limite de sessões no ano civil (Art. 40)
LIMITE_SESSAO_ANO = 40

# Alçadas da análise automatizada (Art. 78)
ALCADA_AUTOMATIZADA_MAX = Decimal("5000.00")
ALCADA_CIRURGICA_MAX = Decimal("7500.00")

# Carências contratuais em dias (Arts. 21 a 26)
CARENCIA_DIAS = {
    Categoria.CONSULTA_MEDICA: 30,      # Art. 22
    Categoria.EXAME_DIAGNOSTICO: 60,    # Art. 23
    Categoria.SESSAO_TERAPIA: 90,       # Art. 24
    Categoria.MATERIAL_OPME: 180,       # Art. 25
}


def mapear_categoria_oficial(doc: DocumentoInfo) -> Categoria:
    """Mapeia os campos do DocumentoInfo para o enum Categoria do contrato da API."""
    if not doc.valido_assistencial or doc.tipo_documento == "FATURA_CONSUMO":
        return Categoria.INVALIDO

    if doc.tipo_documento == "RELATORIO_CLINICO":
        return Categoria.RELATORIO_CLINICO

    cat_map = {
        "CONSULTA_MEDICA": Categoria.CONSULTA_MEDICA,
        "SESSAO_TERAPIA": Categoria.SESSAO_TERAPIA,
        "EXAME": Categoria.EXAME_DIAGNOSTICO,
        "MATERIAL_OPME": Categoria.MATERIAL_OPME,
        "PROCEDIMENTO_CIRURGICO": Categoria.CONSULTA_MEDICA,
    }
    return cat_map.get(doc.categoria_reembolso, Categoria.INVALIDO)


def calcular_meses_adesao(data_adesao_str: str, data_referencia_str: str | None = None) -> int:
    """Calcula meses completos de adesão até a data do atendimento/conversa."""
    try:
        dt_adesao = datetime.date.fromisoformat(data_adesao_str.strip())
    except (ValueError, AttributeError):
        return 0

    if data_referencia_str:
        try:
            dt_ref = datetime.date.fromisoformat(data_referencia_str.strip())
        except (ValueError, AttributeError):
            dt_ref = datetime.date.today()
    else:
        dt_ref = datetime.date.today()

    meses = (dt_ref.year - dt_adesao.year) * 12 + (dt_ref.month - dt_adesao.month)
    if dt_ref.day < dt_adesao.day:
        meses -= 1
    return max(0, meses)


def calcular_dias_adesao(data_adesao_str: str, data_referencia_str: str | None = None) -> int:
    """Calcula dias corridos de adesão até a data do atendimento."""
    try:
        dt_adesao = datetime.date.fromisoformat(data_adesao_str.strip())
    except (ValueError, AttributeError):
        return 0

    if data_referencia_str:
        try:
            dt_ref = datetime.date.fromisoformat(data_referencia_str.strip())
        except (ValueError, AttributeError):
            dt_ref = datetime.date.today()
    else:
        dt_ref = datetime.date.today()

    return max(0, (dt_ref - dt_adesao).days)


def obter_fator_coparticipacao(plano: str, meses_adesao: int) -> Decimal:
    """Retorna o percentual de coparticipação conforme Circular 02/2026 (Art. 44 § 2º).

    Pleno:
      - Até 12 meses: 40% (0.40)
      - De 12 a 36 meses: 30% (0.30)
      - Acima de 36 meses: 20% (0.20)
    Essencial:
      - Até 12 meses: 45% (0.45)
      - De 12 a 36 meses: 35% (0.35)
      - Acima de 36 meses: 25% (0.25)
    """
    plano_norm = (plano or "").strip().lower()
    eh_essencial = "essencial" in plano_norm

    if meses_adesao <= 12:
        return Decimal("0.45") if eh_essencial else Decimal("0.40")
    elif meses_adesao <= 36:
        return Decimal("0.35") if eh_essencial else Decimal("0.30")
    else:
        return Decimal("0.25") if eh_essencial else Decimal("0.20")


def obter_teto_procedimento(categoria: Categoria, codigo_tuss: str | None = None) -> tuple[Decimal, Decimal, str | None]:
    """Retorna (teto_urs, teto_brl, dispositivo_teto)."""
    if categoria == Categoria.CONSULTA_MEDICA or codigo_tuss == "10101012":
        return TETO_CONSULTA_URS, TETO_CONSULTA_BRL, "ART-35"
    elif categoria == Categoria.SESSAO_TERAPIA or codigo_tuss == "50000462":
        return TETO_PSICOTERAPIA_URS, TETO_PSICOTERAPIA_BRL, "ART-41"
    else:
        # Padrão ou procedimentos gerais
        return Decimal("0.0"), Decimal("0.0"), None


class ResultadoCalculo:
    def __init__(
        self,
        decisao: Decisao,
        categoria: Categoria,
        valor_solicitado: Decimal | None,
        valor_reembolso: Decimal | None,
        regras_aplicadas: list[str],
        pendencias: list[str] | None = None,
        protocolo_exigido: bool = False,
        motivo: str = "",
    ):
        self.decisao = decisao
        self.categoria = categoria
        self.valor_solicitado = valor_solicitado
        self.valor_reembolso = valor_reembolso
        self.regras_aplicadas = regras_aplicadas
        self.pendencias = pendencias or []
        self.protocolo_exigido = protocolo_exigido
        self.motivo = motivo

    def como_dict(self) -> dict[str, Any]:
        return {
            "decisao": self.decisao.value,
            "categoria_documento": self.categoria.value if self.categoria else None,
            "valor_solicitado_brl": float(self.valor_solicitado) if self.valor_solicitado is not None else None,
            "valor_reembolso_brl": float(self.valor_reembolso) if self.valor_reembolso is not None else None,
            "regras_aplicadas": self.regras_aplicadas,
            "pendencias": self.pendencias,
            "protocolo_exigido": self.protocolo_exigido,
            "motivo": self.motivo,
        }


def apurar_reembolso(
    doc: DocumentoInfo,
    beneficiario: dict[str, Any],
    historico: dict[str, Any] | None = None,
    documentos_adicionais: list[DocumentoInfo] | None = None,
) -> ResultadoCalculo:
    """Executa a apuração matemática completa e aplicação de normas regulamentares."""
    regras: list[str] = []
    pendencias: list[str] = []
    categoria = mapear_categoria_oficial(doc)
    valor_solicitado = Decimal(str(doc.valor_cobrado)) if doc.valor_cobrado is not None else None

    # 1. Validação de Documento Válido / Assistencial (Art. 76)
    if not doc.valido_assistencial or categoria == Categoria.INVALIDO:
        return ResultadoCalculo(
            decisao=Decisao.PENDENTE_DOCUMENTO,
            categoria=Categoria.INVALIDO,
            valor_solicitado=valor_solicitado,
            valor_reembolso=None,
            regras_aplicadas=[],
            pendencias=["Documento enviado não é comprovante assistencial válido (Art. 76)"],
            protocolo_exigido=False,
            motivo="O documento enviado não constitui comprovante de despesa assistencial de saúde.",
        )

    # 2. Validação de Alçada e Matéria Complexa — OPME (Art. 78)
    if categoria == Categoria.MATERIAL_OPME:
        regras.append("ART-78")
        return ResultadoCalculo(
            decisao=Decisao.ESCALADO_ANALISTA,
            categoria=Categoria.MATERIAL_OPME,
            valor_solicitado=valor_solicitado,
            valor_reembolso=None,
            regras_aplicadas=regras,
            protocolo_exigido=True,
            motivo="Procedimento de Material/OPME exige análise técnica humana obrigatória (Art. 78, § 2º).",
        )

    # 3. Validação de Alçada Financeira Geral (Art. 78 § 1º)
    if valor_solicitado and valor_solicitado > ALCADA_AUTOMATIZADA_MAX:
        regras.append("ART-78")
        return ResultadoCalculo(
            decisao=Decisao.ESCALADO_ANALISTA,
            categoria=categoria,
            valor_solicitado=valor_solicitado,
            valor_reembolso=None,
            regras_aplicadas=regras,
            protocolo_exigido=True,
            motivo=f"Valor solicitado (R$ {valor_solicitado}) excede a alçada automatizada de R$ 5.000,00.",
        )

    # 4. Checagem de Carência Contratual (Arts. 21 a 30)
    data_adesao = beneficiario.get("data_adesao") or "2020-01-01"
    data_atendimento = doc.data_atendimento or doc.data_emissao or "2026-05-20"
    dias_adesao = calcular_dias_adesao(data_adesao, data_atendimento)
    carencia_exigida = CARENCIA_DIAS.get(categoria, 30)

    if dias_adesao < carencia_exigida:
        return ResultadoCalculo(
            decisao=Decisao.NEGADO,
            categoria=categoria,
            valor_solicitado=valor_solicitado,
            valor_reembolso=None,
            regras_aplicadas=[f"ART-{carencia_exigida // 30 + 20}"],
            motivo=f"Atendimento realizado antes do término do período de carência ({dias_adesao} dias de {carencia_exigida} dias).",
        )

    # 5. Exigência de Relatório Clínico para Sessões de Terapia (Art. 73 § 3º e Circular 02/2026)
    if categoria == Categoria.SESSAO_TERAPIA:
        sessoes_anteriores = 0
        if historico and "sessoes_terapia_ano" in historico:
            sessoes_anteriores = int(historico["sessoes_terapia_ano"])

        sessao_atual_numero = sessoes_anteriores + 1
        teto_urs, teto_brl, _ = obter_teto_procedimento(categoria, doc.codigo_tuss)

        excede_75_pct = valor_solicitado and (valor_solicitado > Decimal("1.75") * teto_brl)
        exige_relatorio = (sessao_atual_numero >= 5) or bool(excede_75_pct)

        # Checa se já recebemos relatório clínico neste turno ou em documentos_adicionais
        tem_relatorio = False
        if documentos_adicionais:
            for d in documentos_adicionais:
                if mapear_categoria_oficial(d) == Categoria.RELATORIO_CLINICO:
                    tem_relatorio = True
                    break

        if exige_relatorio and not tem_relatorio:
            regras.extend(["ART-41", "CIRC-02-2026", "ART-73", "NT-02"])
            if doc.codigo_tuss:
                regras.append(f"TUSS-{doc.codigo_tuss}")
            return ResultadoCalculo(
                decisao=Decisao.PENDENTE_DOCUMENTO,
                categoria=Categoria.SESSAO_TERAPIA,
                valor_solicitado=valor_solicitado,
                valor_reembolso=None,
                regras_aplicadas=regras,
                pendencias=["Relatório clínico circunstanciado contemporâneo (Art. 73, § 3º e Circular 02/2026)"],
                protocolo_exigido=False,
                motivo=f"A {sessao_atual_numero}ª sessão de psicoterapia no ano civil exige apresentação de relatório clínico.",
            )

    # 6. Cálculo Financeiro e Aplicação dos Tetos e Coparticipação
    val_solic = valor_solicitado or Decimal("0.00")
    teto_urs, teto_brl, art_procedimento = obter_teto_procedimento(categoria, doc.codigo_tuss)

    # Adiciona dispositivos de procedimento
    if categoria == Categoria.CONSULTA_MEDICA:
        regras.append("ART-35")
    elif categoria == Categoria.SESSAO_TERAPIA:
        regras.extend(["ART-41", "CIRC-02-2026", "ART-73"])

    # Art. 33 (multiplicação da URS por R$ 95,10)
    regras.append("ART-33")

    # Art. 43: Base de Cálculo = min(valor_pago, teto_brl)
    base_calculo = min(val_solic, teto_brl)
    regras.append("ART-43")

    # Art. 44: Coparticipação por Plano e Tempo de Adesão (Circular 02/2026)
    plano = beneficiario.get("plano", "Pleno")
    meses_adesao = calcular_meses_adesao(data_adesao, data_atendimento)
    fator_copart = obter_fator_coparticipacao(plano, meses_adesao)
    fator_reembolso = Decimal("1.0") - fator_copart
    valor_provisorio = base_calculo * fator_reembolso
    regras.append("ART-44")

    # Art. 45: Teto Anual Global por Beneficiário (48 URS = R$ 4.564,80)
    total_ja_reembolsado = Decimal(str(historico.get("total_reembolsado_ano", 0.0))) if historico else Decimal("0.0")
    saldo_anual = max(Decimal("0.00"), TETO_ANUAL_GLOBAL_BRL - total_ja_reembolsado)

    limitado_por_saldo = valor_provisorio > saldo_anual
    if limitado_por_saldo or total_ja_reembolsado > 0:
        regras.append("ART-45")

    valor_antes_arredondamento = min(valor_provisorio, saldo_anual)

    # Art. 47: Arredondamento final de 2 casas decimais
    regras.append("ART-47")
    valor_final = valor_antes_arredondamento.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    # TUSS no regras_aplicadas
    if doc.codigo_tuss:
        tuss_code = f"TUSS-{doc.codigo_tuss}"
        if tuss_code not in regras:
            regras.append(tuss_code)

    # 7. Determinação da Decisão (APROVADO vs APROVADO_PARCIAL)
    # Se o valor solicitado foi maior que o teto, ou se houve redução por teto anual global
    if val_solic > teto_brl or limitado_por_saldo:
        decisao = Decisao.APROVADO_PARCIAL
    else:
        decisao = Decisao.APROVADO

    return ResultadoCalculo(
        decisao=decisao,
        categoria=categoria,
        valor_solicitado=val_solic,
        valor_reembolso=valor_final,
        regras_aplicadas=regras,
        pendencias=[],
        protocolo_exigido=False,
        motivo="Apuração concluída com aplicação dos tetos da URS 2026 e coparticipação regulamentar.",
    )
