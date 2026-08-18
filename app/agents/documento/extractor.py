"""Extrator estruturado e classificador de documentos fiscais e laudos de saúde."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .schemas import CategoriaReembolso, DocumentoInfo, TipoDocumento

logger = logging.getLogger("app.agents.documento.extractor")

RAIZ = Path(__file__).resolve().parents[3]
CAMINHO_TUSS_JSON = RAIZ / "storage" / "tuss_tabela.json"

_TABELA_TUSS: dict[str, dict[str, Any]] | None = None


def _carregar_tuss_tabela() -> dict[str, dict[str, Any]]:
    global _TABELA_TUSS
    if _TABELA_TUSS is None:
        if CAMINHO_TUSS_JSON.exists():
            try:
                _TABELA_TUSS = json.loads(CAMINHO_TUSS_JSON.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Não foi possível carregar tuss_tabela.json: {e}")
                _TABELA_TUSS = {}
        else:
            _TABELA_TUSS = {}
    return _TABELA_TUSS


def _formatar_data_iso(data_str: str) -> str | None:
    """Converte datas DD/MM/AAAA para ISO YYYY-MM-DD."""
    if not data_str:
        return None
    data_str = data_str.strip()
    match_br = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", data_str)
    if match_br:
        d, m, y = match_br.groups()
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    match_iso = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", data_str)
    if match_iso:
        y, m, d = match_iso.groups()
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    return None


def extrair_informacoes_documento(texto: str) -> DocumentoInfo:
    """Analisa o texto do anexo e extrai campos da NT-02 e categorização determinística."""
    if not texto or not texto.strip():
        return DocumentoInfo(
            tipo_documento="OUTRO",
            categoria_reembolso="INVALIDO",
            valido_assistencial=False,
            motivo_invalidez="Documento em branco ou texto não legível.",
            texto_extraido="",
        )

    texto_limpo = texto.strip()
    cabecalho = texto_limpo[:350].upper()

    # 1. Checagem de documento não assistencial (ex: conta de consumo / energia)
    padroes_nao_assistenciais = [
        r"FATURA DE ENERGIA",
        r"CONTA DE CONSUMO",
        r"DISTRIBUIÇÃO DE ENERGIA",
        r"ENERGIA ELÉTRICA",
        r"ILUMINAÇÃO PÚBLICA",
        r"TARIFA RESIDENCIAL",
        r"COMPANHIA METROPOLITANA DE ENERGIA",
        r"CONTA DE ÁGUA",
        r"FATURA DE GÁS",
        r"ESTACIONAMENTO",
    ]
    if any(re.search(p, texto_limpo, re.IGNORECASE) for p in padroes_nao_assistenciais):
        return DocumentoInfo(
            tipo_documento="FATURA_CONSUMO",
            categoria_reembolso="INVALIDO",
            valido_assistencial=False,
            motivo_invalidez="Documento não assistencial (fatura de consumo / energia). O reembolso exige documento fiscal comprobatório de despesa de saúde.",
            texto_extraido=texto_limpo,
        )

    # 2. Identificação do Tipo de Documento
    tipo_doc: TipoDocumento = "OUTRO"
    if "RELATÓRIO CLÍNICO" in cabecalho or "RELATORIO CLINICO" in cabecalho:
        tipo_doc = "RELATORIO_CLINICO"
    elif "NOTA FISCAL" in cabecalho or "DANFE" in cabecalho or "NFS-E" in cabecalho:
        tipo_doc = "NOTA_FISCAL"
    elif "RECIBO" in cabecalho:
        tipo_doc = "RECIBO"
    elif "PEDIDO MÉDICO" in cabecalho or "SOLICITAÇÃO" in cabecalho:
        tipo_doc = "PEDIDO_MEDICO"

    # 3. Extração de Dados do Paciente (C2 da NT-02)
    paciente_nome = None
    paciente_cpf = None

    match_pac = re.search(r"(?:Recebi de|Paciente|Cliente|Beneficiário):\s*([^\n\r_]+)", texto_limpo, re.IGNORECASE)
    if match_pac:
        paciente_nome = match_pac.group(1).strip()

    match_cpf_pac = re.search(r"CPF(?: do paciente)?:\s*(\d{3}\.?\d{3}\.?\d{3}-?\d{2})", texto_limpo, re.IGNORECASE)
    if match_cpf_pac:
        paciente_cpf = match_cpf_pac.group(1).strip()
    else:
        # Busca genérica por CPF se houver paciente
        todos_cpfs = re.findall(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", texto_limpo)
        if todos_cpfs:
            paciente_cpf = todos_cpfs[0]

    # 4. Extração de Dados do Prestador (C1 da NT-02)
    prestador_nome = None
    prestador_doc = None
    prestador_conselho = None

    match_prest = re.search(r"(?:Profissional|Emitente|Prestador|Dra?\.|Dr\.|Psicólogo\(a\)?|Médico\(a\)?):\s*([^\n\r_]+)", texto_limpo, re.IGNORECASE)
    if match_prest:
        prestador_nome = match_prest.group(1).strip()
    elif tipo_doc == "RELATORIO_CLINICO":
        match_laudo_prest = re.search(r"([A-ZÁÉÍÓÚ][a-záéíóú]+\s+[A-ZÁÉÍÓÚ][a-záéíóú]+(?:\s+[A-ZÁÉÍÓÚ][a-záéíóú]+)?)\s*—\s*Psicologia", texto_limpo)
        if match_laudo_prest:
            prestador_nome = match_laudo_prest.group(1).strip()

    match_cnpj = re.search(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b", texto_limpo)
    if match_cnpj:
        prestador_doc = match_cnpj.group(0)
    else:
        match_cpf_prest = re.search(r"(?:CPF/CNPJ|CNPJ/CPF|CPF do profissional):\s*(\d{3}\.?\d{3}\.?\d{3}-?\d{2})", texto_limpo, re.IGNORECASE)
        if match_cpf_prest:
            prestador_doc = match_cpf_prest.group(1).strip()

    match_conselho = re.search(
        r"\b(CRM|CRP|CRO|RMS|CREFITO|CRFa)(?:[-/\s]*(?:[A-Z]{2}|\d{2}))?[-/\s]*(\d{2,}[\.\d]*)\b",
        texto_limpo,
        re.IGNORECASE,
    )
    if match_conselho:
        prestador_conselho = match_conselho.group(0).strip()

    # 5. Extração de Datas (C3 da NT-02 - Data do Atendimento prevalece)
    data_atendimento = None
    data_emissao = None

    match_dt_atend = re.search(r"(?:Data do atendimento|Data do evento|Atendimento|Data da prestação):\s*([0-9/]+)", texto_limpo, re.IGNORECASE)
    if match_dt_atend:
        data_atendimento = _formatar_data_iso(match_dt_atend.group(1))

    match_dt_emiss = re.search(r"(?:Data de emissão|Emissão|Emitido em):\s*([0-9/]+)", texto_limpo, re.IGNORECASE)
    if match_dt_emiss:
        data_emissao = _formatar_data_iso(match_dt_emiss.group(1))

    # Se não especificou atendimento explicitamente, mas há emissão
    if not data_atendimento and data_emissao:
        data_atendimento = data_emissao

    # 6. Extração de Códigos TUSS e Descrição do Serviço (C4 e C5 da NT-02)
    tabela_tuss = _carregar_tuss_tabela()
    codigos_encontrados = re.findall(r"\b(1\d{7}|2\d{7}|3\d{7}|4\d{7}|5\d{7}|7\d{7})\b", texto_limpo)
    # Remove duplicatas mantendo ordem
    codigos_unicos = list(dict.fromkeys(codigos_encontrados))

    codigo_tuss_principal = codigos_unicos[0] if codigos_unicos else None
    descricao_procedimento = None

    match_desc = re.search(r"(?:Referente a|Descrição|Discriminação|Serviço prestado):\s*([^\n\r]+)", texto_limpo, re.IGNORECASE)
    if match_desc:
        descricao_procedimento = match_desc.group(1).strip()
    elif codigo_tuss_principal and codigo_tuss_principal in tabela_tuss:
        descricao_procedimento = tabela_tuss[codigo_tuss_principal].get("descricao")

    # 7. Extração de Valores Monetários (C6 da NT-02)
    valor_cobrado = None
    valores_reais = re.findall(r"R\$\s*([\d\.]+(?:,\d{2})?)", texto_limpo)
    if valores_reais:
        valores_float = []
        for v in valores_reais:
            try:
                v_num = float(v.replace(".", "").replace(",", "."))
                valores_float.append(v_num)
            except ValueError:
                pass
        if valores_float:
            # Se houver indicação de 'Valor total' ou 'importância de', pega o maior valor que representa o total
            valor_cobrado = max(valores_float)

    # 8. Classificação da Categoria Assistencial (Art. 74 c/c NT-02)
    categoria: CategoriaReembolso = "INVALIDO"

    if any(c.startswith("7") for c in codigos_unicos) or re.search(r"\b(prótese|órtese|osteossíntese|implante|haste|parafuso)\b", texto_limpo, re.IGNORECASE):
        categoria = "MATERIAL_OPME"
    elif any(c.startswith("50000462") or c.startswith("5") for c in codigos_unicos) or re.search(r"\b(psicoterapia|psicologia|sessão de terapia|fonoaudiologia|fisioterapia)\b", texto_limpo, re.IGNORECASE):
        categoria = "SESSAO_TERAPIA"
    elif any(c.startswith("10101") or c.startswith("1") for c in codigos_unicos) or re.search(r"\b(consulta médica|consulta em|dermatologia|cardiologia|clínica médica)\b", texto_limpo, re.IGNORECASE):
        categoria = "CONSULTA_MEDICA"
    elif any(c.startswith("3") for c in codigos_unicos) or re.search(r"\b(artroplastia|cirurgia|procedimento cirúrgico|laparoscopia)\b", texto_limpo, re.IGNORECASE):
        categoria = "PROCEDIMENTO_CIRURGICO"
    elif any(c.startswith("4") for c in codigos_unicos) or re.search(r"\b(exame|ressonância|tomografia|hemograma|ultrassonografia)\b", texto_limpo, re.IGNORECASE):
        categoria = "EXAME"
    elif tipo_doc == "RELATORIO_CLINICO":
        categoria = "SESSAO_TERAPIA"

    # 9. Verificação de Pendências da NT-02 (C1 a C8)
    pendencias: list[str] = []
    if tipo_doc in ("RECIBO", "NOTA_FISCAL"):
        if not prestador_conselho and not prestador_doc:
            pendencias.append("Identificação incompleta do prestador (sem registro de conselho ou CNPJ).")
        if not paciente_nome:
            pendencias.append("Nome do paciente ausente no documento fiscal.")
        if not data_atendimento:
            pendencias.append("Data da prestação do serviço/atendimento ausente.")
        if valor_cobrado is None:
            pendencias.append("Valor do procedimento não discriminado no documento.")

    return DocumentoInfo(
        tipo_documento=tipo_doc,
        categoria_reembolso=categoria,
        valido_assistencial=True,
        paciente_nome=paciente_nome,
        paciente_cpf=paciente_cpf,
        prestador_nome=prestador_nome,
        prestador_documento=prestador_doc,
        prestador_conselho=prestador_conselho,
        data_atendimento=data_atendimento,
        data_emissao=data_emissao,
        descricao_procedimento=descricao_procedimento,
        codigo_tuss=codigo_tuss_principal,
        codigos_tuss=codigos_unicos,
        valor_cobrado=valor_cobrado,
        tem_quitacao=True,
        tem_assinatura=True,
        pendencias_nt02=pendencias,
        texto_extraido=texto_limpo,
    )
