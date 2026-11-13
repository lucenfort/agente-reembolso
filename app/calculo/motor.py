"""Motor determinístico de verificação de requisitos e cálculo de reembolso.

Implementa integralmente:
- A sequência obrigatória de verificação da NT-02 (Seção 6.1) e do Art. 10 do Regulamento Geral;
- A Tabela Oficial de Procedimentos e Limites em URS (storage/tuss_tabela.json - 16 códigos TUSS);
- As regras de vigência temporal entre a Circular 11/2026 (01/02/2026 a 19/04/2026) e a Circular 02/2026 (>= 20/04/2026);
- O teto reduzido para psicoterapia em continuidade (>= 8ª sessão no ano civil: 2,6 URS sob CIRC-02-2026 ou 1,4 URS sob CIRC-11-2026);
- Carências contratuais exatas (Arts. 21 a 26: Consulta 15d, Exame 60d, Terapia 150d, OPME 180d, Urgência 0d);
- Limite anual de 40 sessões de terapia (Art. 40) e teto anual global de 48 URS = R$ 4.564,80 (Art. 45);
- Dedução de taxas não assistenciais (Arts. 57 e 59) e bloqueio de recibos sem discriminação individualizada (NT-02 P16);
- Toda matemática com Decimal e arredondamento financeiro HALF_UP em 2 casas decimais (Art. 47).
"""

from __future__ import annotations

import datetime
import json
import os
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from app.agents.documento.schemas import DocumentoInfo
from app.rag.retriever import buscar_tuss_por_descricao
from app.schemas import Categoria, Decisao

# ==============================================================================
# Constantes Regulamentares (Vigência 2026)
# ==============================================================================
VALOR_URS_2026 = Decimal("95.10")
TETO_ANUAL_GLOBAL_URS = Decimal("48")
TETO_ANUAL_GLOBAL_BRL = (TETO_ANUAL_GLOBAL_URS * VALOR_URS_2026).quantize(
    Decimal("0.01"), rounding=ROUND_HALF_UP
)  # R$ 4.564,80

TETO_CONSULTA_URS = Decimal("5.0")
TETO_CONSULTA_BRL = (TETO_CONSULTA_URS * VALOR_URS_2026).quantize(
    Decimal("0.01"), rounding=ROUND_HALF_UP
)  # R$ 475,50

# Circular 02/2026 (vigência >= 2026-04-20)
TETO_PSICOTERAPIA_URS = Decimal("2.8")
TETO_PSICOTERAPIA_BRL = (TETO_PSICOTERAPIA_URS * VALOR_URS_2026).quantize(
    Decimal("0.01"), rounding=ROUND_HALF_UP
)  # R$ 266,28
TETO_PSICOTERAPIA_CONTINUIDADE_URS = Decimal("2.6")
TETO_PSICOTERAPIA_CONTINUIDADE_BRL = (TETO_PSICOTERAPIA_CONTINUIDADE_URS * VALOR_URS_2026).quantize(
    Decimal("0.01"), rounding=ROUND_HALF_UP
)  # R$ 247,26

# Circular 11/2026 (vigência 2026-02-01 a 2026-04-19)
TETO_PSICOTERAPIA_CIRC11_URS = Decimal("1.6")
TETO_PSICOTERAPIA_CIRC11_BRL = (TETO_PSICOTERAPIA_CIRC11_URS * VALOR_URS_2026).quantize(
    Decimal("0.01"), rounding=ROUND_HALF_UP
)  # R$ 152,16
TETO_PSICOTERAPIA_CIRC11_CONT_URS = Decimal("1.4")
TETO_PSICOTERAPIA_CIRC11_CONT_BRL = (TETO_PSICOTERAPIA_CIRC11_CONT_URS * VALOR_URS_2026).quantize(
    Decimal("0.01"), rounding=ROUND_HALF_UP
)  # R$ 133,14

# Limite de sessões no ano civil (Art. 40)
LIMITE_SESSAO_ANO = 40

# Prazo decadencial de solicitação de reembolso em dias corridos (Art. 12 e CIRC-09-2025)
PRAZO_DECADENCIAL_DIAS = 90

# Alçadas da análise automatizada (Art. 78)
ALCADA_AUTOMATIZADA_MAX = Decimal("5000.00")
ALCADA_CIRURGICA_MAX = Decimal("7500.00")

# Carências contratuais em dias corridos (Arts. 21 a 26 do Regulamento Geral)
CARENCIA_DIAS = {
    Categoria.CONSULTA_MEDICA: 15,      # Art. 22: Consultas médicas eletivas = 15 dias
    Categoria.EXAME_DIAGNOSTICO: 60,    # Art. 23: Exames diagnósticos ambulatoriais = 60 dias
    Categoria.SESSAO_TERAPIA: 150,      # Art. 24: Sessões de terapias seriadas = 150 dias
    Categoria.MATERIAL_OPME: 180,       # Art. 25: Internações, cirurgias e OPME = 180 dias
}

ARTIGO_CARENCIA = {
    Categoria.CONSULTA_MEDICA: "ART-22",
    Categoria.EXAME_DIAGNOSTICO: "ART-23",
    Categoria.SESSAO_TERAPIA: "ART-24",
    Categoria.MATERIAL_OPME: "ART-25",
}

_TUSS_JSON_PATH = Path(__file__).resolve().parents[2] / "storage" / "tuss_tabela.json"
_TABELA_TUSS_CACHE: dict[str, dict[str, Any]] | None = None


def _carregar_tabela_tuss() -> dict[str, dict[str, Any]]:
    """Carrega storage/tuss_tabela.json indexado pelo código TUSS."""
    global _TABELA_TUSS_CACHE
    if _TABELA_TUSS_CACHE is not None:
        return _TABELA_TUSS_CACHE

    tabela: dict[str, dict[str, Any]] = {}
    if _TUSS_JSON_PATH.exists():
        try:
            bruto = json.loads(_TUSS_JSON_PATH.read_text(encoding="utf-8"))
            if isinstance(bruto, dict):
                for cod_key, item in bruto.items():
                    if not isinstance(item, dict):
                        continue
                    cod = str(item.get("tuss") or cod_key).strip()
                    urs_val = Decimal(str(item.get("urs", "0")).replace(",", "."))
                    brl_val = (urs_val * VALOR_URS_2026).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                    tabela[cod] = {
                        "codigo": cod,
                        "descricao": str(item.get("descricao", "")),
                        "teto_urs": urs_val,
                        "teto_brl": brl_val,
                        "obs": str(item.get("observacao", "")),
                        "grupo": str(item.get("categoria", "")),
                    }
            elif isinstance(bruto, list):
                for item in bruto:
                    cod = str(item.get("Código TUSS") or item.get("tuss") or "").strip()
                    if not cod:
                        continue
                    urs_str = str(item.get("Teto URS") or item.get("urs") or "0").replace(",", ".").strip()
                    urs_val = Decimal(urs_str)
                    brl_val = (urs_val * VALOR_URS_2026).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                    tabela[cod] = {
                        "codigo": cod,
                        "descricao": str(item.get("Descrição") or item.get("descricao") or ""),
                        "teto_urs": urs_val,
                        "teto_brl": brl_val,
                        "obs": str(item.get("Obs") or item.get("observacao") or ""),
                        "grupo": str(item.get("Grupo") or item.get("categoria") or ""),
                    }
        except Exception:
            pass
    _TABELA_TUSS_CACHE = tabela
    return tabela


def mapear_categoria_oficial(doc: DocumentoInfo) -> Categoria:
    """Mapeia os campos do DocumentoInfo para o enum Categoria do contrato da API."""
    if not doc.valido_assistencial or doc.tipo_documento == "FATURA_CONSUMO" or doc.categoria_reembolso == "INVALIDO":
        return Categoria.INVALIDO

    if doc.tipo_documento in ("RELATORIO_CLINICO", "PEDIDO_MEDICO") or doc.categoria_reembolso == "RELATORIO_CLINICO":
        return Categoria.RELATORIO_CLINICO

    if doc.categoria_reembolso == "DESPESA_NAO_COBERTA" or doc.procedimento_excluido:
        return Categoria.DESPESA_NAO_COBERTA

    cat_map = {
        "CONSULTA_MEDICA": Categoria.CONSULTA_MEDICA,
        "SESSAO_TERAPIA": Categoria.SESSAO_TERAPIA,
        "EXAME": Categoria.EXAME_DIAGNOSTICO,
        "EXAME_DIAGNOSTICO": Categoria.EXAME_DIAGNOSTICO,
        "MATERIAL_OPME": Categoria.MATERIAL_OPME,
        "PROCEDIMENTO_CIRURGICO": Categoria.CONSULTA_MEDICA,
        "DESPESA_NAO_COBERTA": Categoria.DESPESA_NAO_COBERTA,
    }
    return cat_map.get(doc.categoria_reembolso, Categoria.INVALIDO)


def _parse_date(date_str: str | None) -> datetime.date | None:
    if not date_str:
        return None
    s = str(date_str).strip()
    try:
        return datetime.date.fromisoformat(s[:10])
    except ValueError:
        m = re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b", s)
        if m:
            try:
                return datetime.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            except ValueError:
                return None
    return None


def eh_vigencia_circular_11(data_atendimento_str: str | None) -> bool:
    """Verifica se a data do atendimento cai na vigência da Circular 11/2026 (01/02/2026 a 19/04/2026)."""
    dt = _parse_date(data_atendimento_str)
    if not dt:
        return False
    return datetime.date(2026, 2, 1) <= dt <= datetime.date(2026, 4, 19)


def calcular_meses_adesao(data_adesao_str: str, data_referencia_str: str | None = None) -> int:
    """Calcula meses completos de adesão até a data do atendimento/referência."""
    dt_adesao = _parse_date(data_adesao_str)
    if not dt_adesao:
        return 0
    dt_ref = _parse_date(data_referencia_str) or datetime.date(2026, 5, 20)

    meses = (dt_ref.year - dt_adesao.year) * 12 + (dt_ref.month - dt_adesao.month)
    if dt_ref.day < dt_adesao.day:
        meses -= 1
    return max(0, meses)


def calcular_dias_adesao(data_adesao_str: str, data_referencia_str: str | None = None) -> int:
    """Calcula dias corridos de adesão até a data do atendimento."""
    dt_adesao = _parse_date(data_adesao_str)
    if not dt_adesao:
        return 0
    dt_ref = _parse_date(data_referencia_str) or datetime.date(2026, 5, 20)
    return max(0, (dt_ref - dt_adesao).days)


def obter_fator_coparticipacao(
    plano: str,
    meses_adesao: int,
    data_atendimento_str: str | None = None,
) -> Decimal:
    """Retorna o percentual de coparticipação conforme a Circular vigente na data do atendimento.

    Circular 02/2026 (vigência >= 2026-04-20, Art. 44 § 2º):
      - Pleno: Até 12m = 40% | 13 a 36m = 30% | > 36m = 20%
      - Essencial: Até 12m = 45% | 13 a 36m = 35% | > 36m = 25%

    Circular 11/2026 (vigência 2026-02-01 a 2026-04-19, Art. 4º):
      - Pleno: Até 12m = 30% | 13 a 36m = 20% | > 36m = 10%
      - Essencial: Até 12m = 35% | 13 a 36m = 25% | > 36m = 15%
    """
    plano_norm = (plano or "").strip().lower()
    eh_essencial = "essencial" in plano_norm
    circ_11 = eh_vigencia_circular_11(data_atendimento_str)

    if circ_11:
        if meses_adesao <= 12:
            return Decimal("0.35") if eh_essencial else Decimal("0.30")
        elif meses_adesao <= 36:
            return Decimal("0.25") if eh_essencial else Decimal("0.20")
        else:
            return Decimal("0.15") if eh_essencial else Decimal("0.10")

    if meses_adesao <= 12:
        return Decimal("0.45") if eh_essencial else Decimal("0.40")
    elif meses_adesao <= 36:
        return Decimal("0.35") if eh_essencial else Decimal("0.30")
    else:
        return Decimal("0.25") if eh_essencial else Decimal("0.20")


def obter_teto_procedimento(
    categoria: Categoria,
    codigo_tuss: str | None = None,
    descricao: str | None = None,
    sessao_atual_numero: int = 1,
    data_atendimento_str: str | None = None,
) -> tuple[Decimal, Decimal, list[str], str | None]:
    """Retorna (teto_urs, teto_brl, artigos_aplicados, codigo_tuss_resolvido).

    Consulta dinamicamente storage/tuss_tabela.json e aplica as regras de:
    - Vigência temporal (Circular 11/2026 vs Circular 02/2026);
    - Psicoterapia em continuidade (>= 8ª sessão no ano civil: Art. 2º da Circular);
    - Enquadramento por descrição quando o código TUSS estiver ausente (Art. 34 § 2º).
    """
    tabela = _carregar_tabela_tuss()
    cod = (codigo_tuss or "").strip()

    # Art. 34 § 2º: se o código TUSS não constar no documento, busca por descrição na tabela
    if not cod and descricao:
        item_inferido = buscar_tuss_por_descricao(descricao, categoria.value if categoria else None)
        if item_inferido:
            cod = str(item_inferido.get("tuss") or item_inferido.get("Código TUSS") or "").replace("*", "").strip()

    circ_11 = eh_vigencia_circular_11(data_atendimento_str)
    circular_tag = "CIRC-11-2026" if circ_11 else "CIRC-02-2026"

    # Caso 1: Psicoterapia individual (50000462 ou terapia sem código específico na tabela)
    if cod in ("50000462", "50000462*") or (
        categoria == Categoria.SESSAO_TERAPIA and cod not in tabela
    ):
        cod_resolvido = "50000462" if (cod or categoria == Categoria.SESSAO_TERAPIA) else None
        if circ_11:
            if sessao_atual_numero >= 8:
                return (
                    TETO_PSICOTERAPIA_CIRC11_CONT_URS,
                    TETO_PSICOTERAPIA_CIRC11_CONT_BRL,
                    ["ART-41", circular_tag, "ART-73"],
                    cod_resolvido,
                )
            return (
                TETO_PSICOTERAPIA_CIRC11_URS,
                TETO_PSICOTERAPIA_CIRC11_BRL,
                ["ART-41", circular_tag, "ART-73"],
                cod_resolvido,
            )
        else:
            if sessao_atual_numero >= 8:
                return (
                    TETO_PSICOTERAPIA_CONTINUIDADE_URS,
                    TETO_PSICOTERAPIA_CONTINUIDADE_BRL,
                    ["ART-41", circular_tag, "ART-73"],
                    cod_resolvido,
                )
            return (
                TETO_PSICOTERAPIA_URS,
                TETO_PSICOTERAPIA_BRL,
                ["ART-41", circular_tag, "ART-73"],
                cod_resolvido,
            )

    # Caso 2: Código TUSS presente na tabela oficial (storage/tuss_tabela.json)
    if cod in tabela:
        info = tabela[cod]
        teto_urs = info["teto_urs"]
        teto_brl = info["teto_brl"]
        grupo = info["grupo"].lower()
        obs = info["obs"].lower()
        artigos: list[str] = []
        if "consulta" in grupo or "grupo i" in grupo or categoria == Categoria.CONSULTA_MEDICA:
            artigos.append("ART-35")
            if "art. 26" in obs:
                artigos.append("ART-26")
        elif "terapia" in grupo or "grupo iv" in grupo or categoria == Categoria.SESSAO_TERAPIA:
            if "art. 41" in obs:
                artigos.append("ART-41")
            else:
                artigos.append("ART-42")
            if sessao_atual_numero >= 5:
                artigos.append("ART-73")
        elif "exame" in grupo or "grupo ii" in grupo or "grupo iii" in grupo or categoria == Categoria.EXAME_DIAGNOSTICO:
            artigos.append("ART-37")
            if "art. 38" in obs or "pedido médico" in obs or "alta complexidade" in obs:
                artigos.append("ART-38")
        return teto_urs, teto_brl, artigos, cod

    # Caso 3: Fallbacks regulamentares por categoria (Art. 34 § 2º e Arts. 35, 37, 39, 42)
    if categoria == Categoria.CONSULTA_MEDICA:
        return TETO_CONSULTA_URS, TETO_CONSULTA_BRL, ["ART-35"], cod or "10101012"

    if categoria == Categoria.EXAME_DIAGNOSTICO:
        # Art. 39: Exame laboratorial simples sem código específico = teto agregado de 1,5 URS (R$ 142,65)
        teto_urs_ex = Decimal("1.5")
        teto_brl_ex = (teto_urs_ex * VALOR_URS_2026).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return teto_urs_ex, teto_brl_ex, ["ART-37", "ART-39"], cod or None

    return Decimal("0.0"), Decimal("0.0"), [], cod or None


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
        etapa_bloqueio: str | None = None,
        detalhes_calculo: dict[str, Any] | None = None,
    ):
        self.decisao = decisao
        self.categoria = categoria
        self.valor_solicitado = valor_solicitado
        self.valor_reembolso = valor_reembolso
        self.regras_aplicadas = regras_aplicadas
        self.pendencias = pendencias or []
        self.protocolo_exigido = protocolo_exigido
        self.motivo = motivo
        self.etapa_bloqueio = etapa_bloqueio
        self.detalhes_calculo = detalhes_calculo or {}

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
            "etapa_bloqueio": self.etapa_bloqueio,
            "detalhes_calculo": self.detalhes_calculo,
        }


def _extrair_sessoes_anteriores(
    beneficiario: dict[str, Any] | None,
    historico: dict[str, Any] | None,
) -> int:
    """Lê a quantidade de sessões de terapia acumuladas no ano civil de todas as fontes possíveis."""
    val_ben = int((beneficiario or {}).get("sessoes_terapia_ano") or 0)
    val_hist = int((historico or {}).get("sessoes_terapia_ano") or 0)
    val_hist_calc = int((historico or {}).get("sessoes_terapia_historico") or 0)
    return max(val_ben, val_hist, val_hist_calc)


def _extrair_total_reembolsado_ano(
    beneficiario: dict[str, Any] | None,
    historico: dict[str, Any] | None,
) -> Decimal:
    """Lê o total já reembolsado no ano civil (em R$) tanto de historico quanto de beneficiario."""
    bruto = (
        (historico or {}).get("total_reembolsado_ano")
        if (historico and "total_reembolsado_ano" in historico)
        else (beneficiario or {}).get("total_reembolsado_ano")
        or (beneficiario or {}).get("reembolsado_ano_brl")
        or 0.0
    )
    return Decimal(str(bruto)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _paciente_diverge_do_titular(doc: DocumentoInfo, beneficiario: dict[str, Any]) -> bool:
    """Verifica se o paciente discriminado no documento é terceiro distinto do titular da sessão."""
    cpf_doc = re.sub(r"\D", "", str(doc.paciente_cpf or ""))
    cpf_ben = re.sub(r"\D", "", str(beneficiario.get("cpf") or ""))
    if len(cpf_doc) == 11 and len(cpf_ben) == 11 and cpf_doc != cpf_ben:
        return True

    nome_doc = (doc.paciente_nome or "").strip().lower()
    nome_ben = (beneficiario.get("nome") or "").strip().lower()
    if nome_doc and nome_ben:
        partes_doc = [p for p in re.split(r"\s+", nome_doc) if len(p) > 2]
        partes_ben = [p for p in re.split(r"\s+", nome_ben) if len(p) > 2]
        if partes_doc and partes_ben and partes_doc[0] != partes_ben[0]:
            return True

    return False


def verificar_requisitos_e_impedimentos(
    doc: DocumentoInfo,
    beneficiario: dict[str, Any] | None = None,
    historico: dict[str, Any] | None = None,
    documentos_adicionais: list[DocumentoInfo] | None = None,
) -> ResultadoCalculo | None:
    """Executa os 11 portões de verificação prévia (Art. 10 e NT-02 Seção 6.1).

    Retorna `ResultadoCalculo` imediatamente se houver qualquer impedimento, pendência,
    negativa ou escalonamento para analista humano.
    Retorna `None` se todos os pré-requisitos estiverem cumpridos e o caso estiver apto
    para o cálculo financeiro (Portão 12).
    """
    categoria = mapear_categoria_oficial(doc)
    valor_solicitado = Decimal(str(doc.valor_cobrado)) if doc.valor_cobrado is not None else None
    docs_lista = documentos_adicionais or []

    # Verifica se há relatório clínico ou pedido médico anexo na sessão
    tem_relatorio_clinico = bool(doc.tem_indicacao_clinica)
    tem_pedido_medico = bool(doc.tem_indicacao_clinica)
    pendencias_relatorio: list[str] = []
    for d in docs_lista:
        if d.tipo_documento == "RELATORIO_CLINICO" or mapear_categoria_oficial(d) == Categoria.RELATORIO_CLINICO:
            tem_relatorio_clinico = True
            tem_pedido_medico = True
            if d.pendencias_relatorio:
                pendencias_relatorio.extend(d.pendencias_relatorio)
        elif d.tipo_documento == "PEDIDO_MEDICO":
            tem_pedido_medico = True

    # ------------------------------------------------------------------
    # PORTÃO 1: Documento inválido / não assistencial (Art. 76 e NT-02 Seção 1.3)
    # ------------------------------------------------------------------
    if not doc.valido_assistencial or categoria == Categoria.INVALIDO:
        return ResultadoCalculo(
            decisao=Decisao.PENDENTE_DOCUMENTO,
            categoria=Categoria.INVALIDO,
            valor_solicitado=valor_solicitado,
            valor_reembolso=None,
            regras_aplicadas=[],
            pendencias=["Documento enviado não é comprovante assistencial válido de saúde (Art. 76 e NT-02)"],
            protocolo_exigido=False,
            motivo="O documento enviado não constitui comprovante fiscal de despesa assistencial de saúde.",
            etapa_bloqueio="DOCUMENTO_INVALIDO_NAO_ASSISTENCIAL",
        )

    # ------------------------------------------------------------------
    # PORTÃO 2: Despesa não coberta / Exclusão assistencial (Arts. 51, 52, 55 e Anexo IV)
    # ------------------------------------------------------------------
    if categoria == Categoria.DESPESA_NAO_COBERTA or doc.procedimento_excluido:
        # Se for procedimento de fronteira (Grupo 3 do Anexo IV, Art. 55) e houver relatório/indicação clínica,
        # reclassifica para a categoria assistencial correspondente!
        if doc.procedimento_fronteira and (tem_relatorio_clinico or tem_pedido_medico):
            categoria = Categoria.SESSAO_TERAPIA if doc.codigo_tuss else Categoria.CONSULTA_MEDICA
        else:
            regras_nc = ["ART-51", "ANEXO-IV"]
            if doc.procedimento_fronteira:
                regras_nc.append("ART-55")
            else:
                regras_nc.append("ART-52")
            return ResultadoCalculo(
                decisao=Decisao.NEGADO,
                categoria=Categoria.DESPESA_NAO_COBERTA,
                valor_solicitado=valor_solicitado,
                valor_reembolso=None,
                regras_aplicadas=regras_nc,
                pendencias=[],
                protocolo_exigido=False,
                motivo=(
                    "Procedimento de fronteira sem indicação clínica formal (presunção estética/bem-estar - Art. 55 e Anexo IV)."
                    if doc.procedimento_fronteira
                    else "Despesa excluída da cobertura de reembolso conforme Art. 51 e Anexo IV do Regulamento Geral."
                ),
                etapa_bloqueio="DESPESA_NAO_COBERTA",
            )

    # ------------------------------------------------------------------
    # PORTÃO 3: Alçada e Matéria Complexa — OPME, Cirurgia > 7500 ou Geral > 5000 (Art. 78)
    # ------------------------------------------------------------------
    if categoria == Categoria.MATERIAL_OPME:
        return ResultadoCalculo(
            decisao=Decisao.ESCALADO_ANALISTA,
            categoria=Categoria.MATERIAL_OPME,
            valor_solicitado=valor_solicitado,
            valor_reembolso=None,
            regras_aplicadas=["ART-78"],
            pendencias=[],
            protocolo_exigido=True,
            motivo="Procedimento envolvendo próteses/órteses e materiais especiais (OPME) exige análise técnica humana obrigatória (Art. 78, § 2º, I).",
            etapa_bloqueio="ALCADA_OPME",
        )

    eh_cirurgico = doc.tipo_documento == "PROCEDIMENTO_CIRURGICO" or doc.categoria_reembolso == "PROCEDIMENTO_CIRURGICO"
    if eh_cirurgico and valor_solicitado and valor_solicitado > ALCADA_CIRURGICA_MAX:
        return ResultadoCalculo(
            decisao=Decisao.ESCALADO_ANALISTA,
            categoria=categoria,
            valor_solicitado=valor_solicitado,
            valor_reembolso=None,
            regras_aplicadas=["ART-78"],
            pendencias=[],
            protocolo_exigido=True,
            motivo=f"Procedimento cirúrgico (R$ {valor_solicitado}) excede a alçada de R$ 7.500,00 (Art. 78, § 2º, II).",
            etapa_bloqueio="ALCADA_CIRURGICA",
        )

    if valor_solicitado and valor_solicitado > ALCADA_AUTOMATIZADA_MAX:
        return ResultadoCalculo(
            decisao=Decisao.ESCALADO_ANALISTA,
            categoria=categoria,
            valor_solicitado=valor_solicitado,
            valor_reembolso=None,
            regras_aplicadas=["ART-78"],
            pendencias=[],
            protocolo_exigido=True,
            motivo=f"Valor solicitado (R$ {valor_solicitado}) excede a alçada máxima automatizada de R$ 5.000,00 (Art. 78, § 1º).",
            etapa_bloqueio="ALCADA_FINANCEIRA",
        )

    # Se o beneficiário ainda não foi identificado na sessão, não é possível validar elegibilidade ou calcular
    if not beneficiario:
        return None

    # ------------------------------------------------------------------
    # PORTÃO 4: Contrato Inativo / Suspenso / Cancelado (Art. 10, I e Art. 30)
    # ------------------------------------------------------------------
    status_plano = str(beneficiario.get("status") or "ATIVO").upper()
    if status_plano in ("CANCELADO", "SUSPENSO", "INATIVO", "NAO_ENCONTRADO"):
        return ResultadoCalculo(
            decisao=Decisao.NEGADO,
            categoria=categoria,
            valor_solicitado=valor_solicitado,
            valor_reembolso=None,
            regras_aplicadas=["ART-10", "ART-30"],
            pendencias=[],
            protocolo_exigido=False,
            motivo=f"Vínculo contratual do beneficiário encontra-se {status_plano}, sem cobertura para reembolso (Art. 10, I e Art. 30).",
            etapa_bloqueio="CONTRATO_INATIVO",
        )

    # ------------------------------------------------------------------
    # PORTÃO 5: Titularidade do Comprovante — Paciente diverge do Beneficiário (Art. 8º, Art. 58, NT-02)
    # ------------------------------------------------------------------
    if _paciente_diverge_do_titular(doc, beneficiario):
        return ResultadoCalculo(
            decisao=Decisao.NEGADO,
            categoria=categoria,
            valor_solicitado=valor_solicitado,
            valor_reembolso=None,
            regras_aplicadas=["ART-8", "ART-58", "NT-02"],
            pendencias=[],
            protocolo_exigido=False,
            motivo="O paciente identificado no comprovante fiscal diverge do beneficiário titular desta sessão de atendimento (Art. 8º e Art. 58).",
            etapa_bloqueio="PACIENTE_DIVERGENTE_TERCEIRO",
        )

    # ------------------------------------------------------------------
    # PORTÃO 6: Tempestividade — Prazo decadencial de 90 dias corridos (Arts. 12, 13 e CIRC-09-2025)
    # ------------------------------------------------------------------
    data_atendimento = doc.data_atendimento or doc.data_emissao or "2026-05-20"
    data_protocolo_ref = (
        beneficiario.get("data_protocolo")
        or (historico or {}).get("data_protocolo")
        or os.environ.get("DATA_REFERENCIA_PROTOCOLO", "2026-05-20")
    )
    dt_atend = _parse_date(data_atendimento)
    dt_prot = _parse_date(data_protocolo_ref) or datetime.date(2026, 5, 20)
    if dt_atend and dt_prot >= dt_atend:
        dias_decorridos = (dt_prot - dt_atend).days
        if dias_decorridos > PRAZO_DECADENCIAL_DIAS:
            return ResultadoCalculo(
                decisao=Decisao.NEGADO,
                categoria=categoria,
                valor_solicitado=valor_solicitado,
                valor_reembolso=None,
                regras_aplicadas=["ART-12", "ART-13", "CIRC-09-2025"],
                pendencias=[],
                protocolo_exigido=False,
                motivo=(
                    f"Prazo decadencial de 90 dias corridos ultrapassado ({dias_decorridos} dias desde o atendimento em {data_atendimento}). "
                    "Conforme Art. 12, Art. 13 e Circular 09/2025, pedidos intempestivos são indeferidos sem análise de mérito."
                ),
                etapa_bloqueio="PRAZO_DECADENCIAL_90_DIAS",
            )

    # ------------------------------------------------------------------
    # PORTÃO 7: Carência Contratual por Categoria (Arts. 21 a 26)
    # ------------------------------------------------------------------
    data_adesao = beneficiario.get("data_adesao") or "2020-01-01"
    dias_adesao = calcular_dias_adesao(data_adesao, data_atendimento)
    carencia_exigida = 0 if doc.carater_urgencia else CARENCIA_DIAS.get(categoria, 15)

    if dias_adesao < carencia_exigida:
        art_car = ARTIGO_CARENCIA.get(categoria, "ART-22")
        return ResultadoCalculo(
            decisao=Decisao.NEGADO,
            categoria=categoria,
            valor_solicitado=valor_solicitado,
            valor_reembolso=None,
            regras_aplicadas=["ART-21", art_car],
            pendencias=[],
            protocolo_exigido=False,
            motivo=(
                f"Atendimento realizado durante o período de carência contratual "
                f"({dias_adesao} dias corridos de adesão; exigido mínimo de {carencia_exigida} dias conforme {art_car})."
            ),
            etapa_bloqueio="CARENCIA_CONTRATUAL",
        )

    # ------------------------------------------------------------------
    # PORTÃO 8: Limite Anual de 40 Sessões (Art. 40) e Teto Global Anual Exaurido (Art. 45)
    # ------------------------------------------------------------------
    sessoes_anteriores = _extrair_sessoes_anteriores(beneficiario, historico)
    sessao_atual_numero = sessoes_anteriores + 1

    if categoria == Categoria.SESSAO_TERAPIA and sessoes_anteriores >= LIMITE_SESSAO_ANO:
        return ResultadoCalculo(
            decisao=Decisao.NEGADO,
            categoria=Categoria.SESSAO_TERAPIA,
            valor_solicitado=valor_solicitado,
            valor_reembolso=None,
            regras_aplicadas=["ART-40"],
            pendencias=[],
            protocolo_exigido=False,
            motivo=f"Limite anual máximo de {LIMITE_SESSAO_ANO} sessões de terapia no ano civil já atingido ({sessoes_anteriores} sessões anteriores utilizadas - Art. 40).",
            etapa_bloqueio="LIMITE_40_SESSOES_ANO",
        )

    total_ja_reembolsado = _extrair_total_reembolsado_ano(beneficiario, historico)
    saldo_anual = max(Decimal("0.00"), TETO_ANUAL_GLOBAL_BRL - total_ja_reembolsado)
    if saldo_anual <= Decimal("0.00"):
        return ResultadoCalculo(
            decisao=Decisao.NEGADO,
            categoria=categoria,
            valor_solicitado=valor_solicitado,
            valor_reembolso=None,
            regras_aplicadas=["ART-45"],
            pendencias=[],
            protocolo_exigido=False,
            motivo="O teto anual global de reembolso de 48 URS (R$ 4.564,80) para o ano civil já foi integralmente utilizado (Art. 45).",
            etapa_bloqueio="TETO_ANUAL_EXAURIDO",
        )

    # ------------------------------------------------------------------
    # PORTÃO 9: Exame de Alta Complexidade / Imagem sem Pedido Médico (Art. 38 e NT-02 Seção 4.2)
    # ------------------------------------------------------------------
    _, _, artigos_tuss_check, _ = obter_teto_procedimento(
        categoria,
        doc.codigo_tuss,
        doc.procedimento_descricao,
        sessao_atual_numero=1,
        data_atendimento_str=data_atendimento,
    )
    exige_pedido_por_tuss = "ART-38" in artigos_tuss_check
    if categoria == Categoria.EXAME_DIAGNOSTICO and (doc.exige_pedido_medico or exige_pedido_por_tuss) and not doc.carater_urgencia:
        if not tem_pedido_medico:
            regras_ex = ["ART-38", "NT-02"]
            if doc.codigo_tuss:
                regras_ex.append(f"TUSS-{doc.codigo_tuss}")
            return ResultadoCalculo(
                decisao=Decisao.PENDENTE_DOCUMENTO,
                categoria=Categoria.EXAME_DIAGNOSTICO,
                valor_solicitado=valor_solicitado,
                valor_reembolso=None,
                regras_aplicadas=regras_ex,
                pendencias=["Pedido médico com indicação clínica expressa (Art. 38 e NT-02 Seção 4.2)"],
                protocolo_exigido=False,
                motivo="Exame diagnóstico de imagem/alta complexidade exige apresentação do pedido médico com justificativa clínica (Art. 38).",
                etapa_bloqueio="FALTA_PEDIDO_MEDICO_EXAME",
            )

    # ------------------------------------------------------------------
    # PORTÃO 10: Requisitos Formais Obrigatórios da NT-02 (Art. 18, NT-02 Seção 2.1 e Seção 3 P16)
    # ------------------------------------------------------------------
    if doc.multiplos_itens_sem_discriminacao:
        return ResultadoCalculo(
            decisao=Decisao.PENDENTE_DOCUMENTO,
            categoria=categoria,
            valor_solicitado=valor_solicitado,
            valor_reembolso=None,
            regras_aplicadas=["ART-18", "NT-02"],
            pendencias=["Discriminação individualizada de datas e valores de cada atendimento no comprovante fiscal (NT-02 Seção 3, P16)"],
            protocolo_exigido=False,
            motivo="Comprovante fiscal agrupa múltiplos atendimentos em valor único sem discriminar data e valor individual de cada sessão (NT-02 P16).",
            etapa_bloqueio="MULTIPLOS_ITENS_SEM_DISCRIMINACAO",
        )

    if doc.pendencias_nt02:
        return ResultadoCalculo(
            decisao=Decisao.PENDENTE_DOCUMENTO,
            categoria=categoria,
            valor_solicitado=valor_solicitado,
            valor_reembolso=None,
            regras_aplicadas=["ART-18", "NT-02"],
            pendencias=list(doc.pendencias_nt02),
            protocolo_exigido=False,
            motivo=f"Comprovante fiscal apresenta pendências formais obrigatórias segundo a NT-02: {', '.join(doc.pendencias_nt02)}.",
            etapa_bloqueio="PENDENCIA_FORMAL_NT02",
        )

    # ------------------------------------------------------------------
    # PORTÃO 11: Exigência de Relatório Clínico para Sessões de Terapia (Art. 73 § 3º e Circular vigente)
    # ------------------------------------------------------------------
    if categoria == Categoria.SESSAO_TERAPIA:
        circ_11 = eh_vigencia_circular_11(data_atendimento)
        circular_tag = "CIRC-11-2026" if circ_11 else "CIRC-02-2026"
        limiar_sessao_relatorio = 10 if circ_11 else 5

        # Para verificar o gatilho de > 75% acima do teto (Art. 73 § 3º, II), usa o teto base da sessão
        _, teto_base_terapia, _, cod_resolvido = obter_teto_procedimento(
            categoria,
            doc.codigo_tuss,
            doc.procedimento_descricao,
            sessao_atual_numero=1,
            data_atendimento_str=data_atendimento,
        )
        excede_75_pct = bool(
            valor_solicitado and teto_base_terapia > 0 and (valor_solicitado > Decimal("1.75") * teto_base_terapia)
        )
        exige_relatorio = (sessao_atual_numero >= limiar_sessao_relatorio) or excede_75_pct

        if exige_relatorio and (not tem_relatorio_clinico or pendencias_relatorio):
            art_terapia = "ART-41" if (cod_resolvido in (None, "50000462")) else "ART-42"
            regras_rel = [art_terapia, circular_tag, "ART-73", "NT-02"]
            if doc.codigo_tuss:
                regras_rel.append(f"TUSS-{doc.codigo_tuss}")
            pend_msg = (
                pendencias_relatorio
                if pendencias_relatorio
                else [f"Relatório clínico circunstanciado contemporâneo do profissional assistente (Art. 73, § 3º e {circular_tag})"]
            )
            return ResultadoCalculo(
                decisao=Decisao.PENDENTE_DOCUMENTO,
                categoria=Categoria.SESSAO_TERAPIA,
                valor_solicitado=valor_solicitado,
                valor_reembolso=None,
                regras_aplicadas=regras_rel,
                pendencias=pend_msg,
                protocolo_exigido=False,
                motivo=(
                    f"A {sessao_atual_numero}ª sessão de terapia no ano civil (a partir da {limiar_sessao_relatorio}ª sessão "
                    f"ou quando o valor excede em mais de 75% o teto em URS) exige apresentação de relatório clínico circunstanciado."
                ),
                etapa_bloqueio="FALTA_RELATORIO_CLINICO_TERAPIA",
            )

    return None


def apurar_reembolso(
    doc: DocumentoInfo,
    beneficiario: dict[str, Any],
    historico: dict[str, Any] | None = None,
    documentos_adicionais: list[DocumentoInfo] | None = None,
) -> ResultadoCalculo:
    """Executa primeiro todos os portões de verificação de requisitos e, se aprovado, realiza o cálculo exato."""
    # Executa os Portões 1 a 11 antes de qualquer cálculo matemático
    impedimento = verificar_requisitos_e_impedimentos(
        doc=doc,
        beneficiario=beneficiario,
        historico=historico,
        documentos_adicionais=documentos_adicionais,
    )
    if impedimento is not None:
        return impedimento

    # ------------------------------------------------------------------
    # PORTÃO 12: Cálculo Financeiro Determinístico (Arts. 33, 35/37/41/42, 43, 44, 45, 47)
    # ------------------------------------------------------------------
    regras: list[str] = []
    categoria = mapear_categoria_oficial(doc)
    if categoria == Categoria.DESPESA_NAO_COBERTA and doc.procedimento_fronteira:
        categoria = Categoria.SESSAO_TERAPIA if doc.codigo_tuss else Categoria.CONSULTA_MEDICA

    val_solic = Decimal(str(doc.valor_cobrado or 0.0)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    taxa_nao_assist = Decimal(str(doc.taxa_nao_assistencial or 0.0)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    val_assistencial = max(Decimal("0.00"), val_solic - taxa_nao_assist)

    data_adesao = beneficiario.get("data_adesao") or "2020-01-01"
    data_atendimento = doc.data_atendimento or doc.data_emissao or "2026-05-20"
    sessoes_anteriores = _extrair_sessoes_anteriores(beneficiario, historico)
    sessao_atual_numero = sessoes_anteriores + 1

    teto_urs, teto_brl, artigos_proc, cod_resolvido = obter_teto_procedimento(
        categoria=categoria,
        codigo_tuss=doc.codigo_tuss,
        descricao=doc.procedimento_descricao,
        sessao_atual_numero=sessao_atual_numero,
        data_atendimento_str=data_atendimento,
    )

    for art in artigos_proc:
        if art not in regras:
            regras.append(art)

    if doc.carater_urgencia and "ART-26" not in regras:
        regras.append("ART-26")

    if doc.procedimento_fronteira and "ART-55" not in regras:
        regras.append("ART-55")

    if taxa_nao_assist > Decimal("0.00"):
        for art_tx in ("ART-57", "ART-59"):
            if art_tx not in regras:
                regras.append(art_tx)

    # Art. 33: Conversão de URS pela cotação vigente (R$ 95,10)
    if "ART-33" not in regras:
        regras.append("ART-33")

    # Art. 43: Base de cálculo = min(valor_efetivamente_pago_assistencial, teto_procedimento_brl)
    base_calculo = min(val_assistencial, teto_brl) if teto_brl > Decimal("0.00") else val_assistencial
    if "ART-43" not in regras:
        regras.append("ART-43")

    # Art. 44: Coparticipação conforme modalidade do plano, tempo de adesão e Circular vigente
    plano = beneficiario.get("plano", "Pleno")
    meses_adesao = calcular_meses_adesao(data_adesao, data_atendimento)
    fator_copart = obter_fator_coparticipacao(plano, meses_adesao, data_atendimento)
    fator_reembolso = Decimal("1.00") - fator_copart
    valor_provisorio = base_calculo * fator_reembolso
    if "ART-44" not in regras:
        regras.append("ART-44")

    # Art. 45: Teto Anual Global por Beneficiário (48 URS = R$ 4.564,80)
    total_ja_reembolsado = _extrair_total_reembolsado_ano(beneficiario, historico)
    saldo_anual = max(Decimal("0.00"), TETO_ANUAL_GLOBAL_BRL - total_ja_reembolsado)

    limitado_por_saldo = valor_provisorio > saldo_anual
    if limitado_por_saldo or total_ja_reembolsado > Decimal("0.00"):
        if "ART-45" not in regras:
            regras.append("ART-45")

    valor_antes_arredondamento = min(valor_provisorio, saldo_anual)

    # Art. 47: Arredondamento final HALF_UP em 2 casas decimais
    if "ART-47" not in regras:
        regras.append("ART-47")
    valor_final = valor_antes_arredondamento.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    # Código TUSS no regras_aplicadas
    cod_final = doc.codigo_tuss or cod_resolvido
    if cod_final:
        tuss_code = f"TUSS-{str(cod_final).replace('*', '')}"
        if tuss_code not in regras:
            regras.append(tuss_code)

    # Determinação da Decisão (APROVADO vs APROVADO_PARCIAL)
    # APROVADO_PARCIAL quando o valor solicitado excede o teto do procedimento, há glosa de taxa não assistencial
    # ou há limitação pelo saldo do teto anual global (Art. 45).
    if val_solic > teto_brl or limitado_por_saldo or taxa_nao_assist > Decimal("0.00"):
        decisao = Decisao.APROVADO_PARCIAL
    else:
        decisao = Decisao.APROVADO

    detalhes = {
        "teto_urs": float(teto_urs),
        "teto_brl": float(teto_brl),
        "base_calculo_brl": float(base_calculo),
        "meses_adesao": meses_adesao,
        "percentual_coparticipacao": int(fator_copart * 100),
        "percentual_reembolso": int(fator_reembolso * 100),
        "valor_provisorio_brl": float(valor_provisorio.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
        "total_ja_reembolsado_ano_brl": float(total_ja_reembolsado),
        "saldo_anual_disponivel_brl": float(saldo_anual),
        "limitado_por_saldo_anual": limitado_por_saldo,
        "sessoes_terapia_anteriores": sessoes_anteriores,
        "sessao_terapia_atual": sessao_atual_numero if categoria == Categoria.SESSAO_TERAPIA else None,
        "taxa_nao_assistencial_glosada_brl": float(taxa_nao_assist),
    }

    return ResultadoCalculo(
        decisao=decisao,
        categoria=categoria,
        valor_solicitado=val_solic,
        valor_reembolso=valor_final,
        regras_aplicadas=regras,
        pendencias=[],
        protocolo_exigido=False,
        motivo="Apuração concluída com aplicação dos tetos da Tabela URS 2026 e coparticipação regulamentar.",
        etapa_bloqueio=None,
        detalhes_calculo=detalhes,
    )
