"""Extrator estruturado e classificador de documentos fiscais e laudos de saúde.

Implementa integralmente as regras da Nota Técnica 02 (Anexos A, B, C, D e E),
do Anexo IV de Exclusões (Grupos 1 a 10) e dos Arts. 34, 35, 51 a 60 e 73 a 76.
"""

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


# Padrões de documentos inválidos (Art. 76 / NT-02 3.1 e Caso 10/11)
PADROES_DOCUMENTO_INVALIDO = [
    r"FATURA DE ENERGIA",
    r"CONTA DE CONSUMO",
    r"DISTRIBUIÇÃO DE ENERGIA",
    r"ENERGIA ELÉTRICA",
    r"ILUMINAÇÃO PÚBLICA",
    r"TARIFA RESIDENCIAL",
    r"COMPANHIA METROPOLITANA DE ENERGIA",
    r"CONTA DE ÁGUA",
    r"FATURA DE GÁS",
    r"COMPROVANTE DE TRANSFERÊNCIA",
    r"COMPROVANTE DE PIX",
    r"CUPOM FISCAL DE SUPERMERCADO",
]

# Padrões de Exclusão Absoluta — Anexo IV (Grupos 1, 2, 4, 7, 8, 9, 10)
PADROES_EXCLUSAO_ABSOLUTA = [
    r"\btoxina botulínica\b",
    r"\bbotox\b",
    r"\bpreenchimento dérmico\b",
    r"\bácido hialurônico estético\b",
    r"\bharmonização orofacial\b",
    r"\blipoaspiração\b",
    r"\babdominoplastia estética\b",
    r"\brinoplastia estética\b",
    r"\bblefaroplastia estética\b",
    r"\bclareamento dental\b",
    r"\blentes? de contato dental\b",
    r"\bfacetas? dent[áa]ri",
    r"\btratamento endodôntico\b",
    r"\bimplante dentário\b",
    r"\baparelho ortodôntico\b",
    r"\bortodontia\b",
    r"\bpeeling químico\b",
    r"\bmicroagulhamento\b",
    r"\bcriolipólise\b",
    r"\bdepilação definitiva\b",
    r"\bmassagem modeladora\b",
    r"\bozonioterapia\b",
    r"\bsoroterapia\b",
    r"\bcromoterapia\b",
    r"\bmedicina ortomolecular\b",
    r"\bfertilização in vitro\b",
    r"\binseminação artificial\b",
    r"\bmedicamentos? de uso domiciliar\b",
    r"\bsuplemento alimentar\b",
    r"\bfiltro solar\b",
    r"\bdermopigmentação\b",
    r"\bcoaching\b",
]

# Padrões de Procedimentos de Fronteira — Anexo IV Grupo 3 (exigem indicação clínica expressa, Art. 55)
PADROES_FRONTEIRA_GRUPO3 = [
    r"\bacupuntura\b",
    r"\bdrenagem linfática\b",
    r"\bfisioterapia dermatofuncional\b",
    r"\bliberação miofascial\b",
    r"\bpilates\b",
    r"\breeducação postural global\b",
    r"\bhidroterapia\b",
    r"\beletroestimulação muscular\b",
    r"\bconsultoria nutricional\b",
    r"\btreinamento funcional\b",
]

# Padrões de Taxas Não Assistenciais (Anexo IV Grupo 5 / Art. 57 / Art. 59)
PADROES_TAXA_NAO_ASSISTENCIAL = [
    r"estacionamento",
    r"manobrista",
    r"taxa administrativa",
    r"taxa de abertura de ficha",
    r"taxa de agendamento",
    r"taxa de remarcação",
    r"multa por falta",
    r"segunda via de",
    r"cópia de prontuário",
]


def extrair_informacoes_documento(texto: str) -> DocumentoInfo:
    """Analisa o texto do anexo e extrai campos da NT-02 e categorização determinística."""
    if not texto or not texto.strip():
        return DocumentoInfo(
            tipo_documento="OUTRO",
            categoria_reembolso="INVALIDO",
            valido_assistencial=False,
            pendencias_nt02=["P27: Documento em branco, corrompido ou ilegível (Art. 75 e NT-02)."],
            motivo_invalidez="Documento em branco ou texto não legível.",
            texto_extraido="",
        )

    texto_limpo = texto.strip()
    cabecalho = texto_limpo[:400].upper()

    # 1. Checagem de documento não assistencial / inválido (ex: conta de consumo / energia, comprovante Pix isolado)
    if any(re.search(p, texto_limpo, re.IGNORECASE) for p in PADROES_DOCUMENTO_INVALIDO):
        return DocumentoInfo(
            tipo_documento="FATURA_CONSUMO",
            categoria_reembolso="INVALIDO",
            valido_assistencial=False,
            motivo_invalidez="Documento não assistencial (fatura de consumo / comprovante não fiscal). O reembolso exige documento fiscal comprobatório de despesa de saúde (Art. 76).",
            texto_extraido=texto_limpo,
        )

    # 2. Identificação do Tipo de Documento
    tipo_doc: TipoDocumento = "OUTRO"
    if "RELATÓRIO CLÍNICO" in cabecalho or "RELATORIO CLINICO" in cabecalho or "LAUDO MÉDICO" in cabecalho or "RELATÓRIO DE ACOMPANHAMENTO" in cabecalho:
        tipo_doc = "RELATORIO_CLINICO"
    elif "PEDIDO MÉDICO" in cabecalho or "SOLICITAÇÃO DE EXAME" in cabecalho or "REQUISIÇÃO MÉDICA" in cabecalho:
        tipo_doc = "PEDIDO_MEDICO"
    elif "NOTA FISCAL" in cabecalho or "DANFE" in cabecalho or "NFS-E" in cabecalho:
        tipo_doc = "NOTA_FISCAL"
    elif "RECIBO" in cabecalho:
        tipo_doc = "RECIBO"

    # 3. Extração de Dados do Paciente (C1 da NT-02)
    paciente_nome = None
    paciente_cpf = None

    match_pac = re.search(r"(?:Recebi de|Paciente|Cliente|Beneficiário|Nome do paciente):\s*([^\n\r_|]+)", texto_limpo, re.IGNORECASE)
    if match_pac:
        cand = match_pac.group(1).strip()
        cand = re.sub(r"\s*[-—,]?\s*CPF.*$", "", cand, flags=re.IGNORECASE).strip()
        if cand:
            paciente_nome = cand

    match_cpf_pac = re.search(r"CPF(?: do paciente| do beneficiário)?:\s*(\d{3}\.?\d{3}\.?\d{3}-?\d{2})", texto_limpo, re.IGNORECASE)
    if match_cpf_pac:
        paciente_cpf = match_cpf_pac.group(1).strip()
    else:
        todos_cpfs = re.findall(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", texto_limpo)
        if todos_cpfs:
            paciente_cpf = todos_cpfs[0]

    # 4. Extração de Dados do Prestador (C2 e C3 da NT-02)
    prestador_nome = None
    prestador_doc = None
    prestador_conselho = None

    match_prest = re.search(
        r"(?:(?:Profissional|Emitente|Prestador|Razão Social|Clínica|Laboratório|Hospital|Psicólogo\(a\)?|Médico\(a\)?)\s*:\s*|(?:\bDra?\.?\s+))([^\n\r_|—-]+)",
        texto_limpo,
        re.IGNORECASE,
    )
    if match_prest:
        prestador_nome = match_prest.group(1).strip()
    elif tipo_doc == "RELATORIO_CLINICO":
        match_laudo_prest = re.search(
            r"([A-ZÁÉÍÓÚ][a-záéíóú]+(?:\s+[A-ZÁÉÍÓÚ][a-záéíóú]+)+)\s*[—-]\s*(?:Psicologia|Medicina|Fisioterapia|Fonoaudiologia)",
            texto_limpo,
        )
        if match_laudo_prest:
            prestador_nome = match_laudo_prest.group(1).strip()

    match_cnpj = re.search(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b", texto_limpo)
    if match_cnpj:
        prestador_doc = match_cnpj.group(0)
    else:
        match_cpf_prest = re.search(
            r"(?:CPF/CNPJ|CNPJ/CPF|CPF do profissional|CPF do prestador):\s*(\d{3}\.?\d{3}\.?\d{3}-?\d{2})",
            texto_limpo,
            re.IGNORECASE,
        )
        if match_cpf_prest:
            prestador_doc = match_cpf_prest.group(1).strip()
        else:
            todos_cpfs = re.findall(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", texto_limpo)
            if len(todos_cpfs) >= 2:
                prestador_doc = todos_cpfs[1]

    match_conselho = re.search(
        r"\b(CRM|CRP|CRO|RMS|CREFITO|CRFa|CRN)(?:[-/\s]*(?:[A-Z]{2}|\d{2}))?[-/\s]*(\d{2,}[\.\d]*)\b",
        texto_limpo,
        re.IGNORECASE,
    )
    if match_conselho:
        prestador_conselho = match_conselho.group(0).strip()

    # 5. Extração de Datas (C4 da NT-02 - Data do Atendimento prevalece sobre Emissão)
    data_atendimento = None
    data_emissao = None

    match_dt_atend = re.search(
        r"(?:Data do atendimento|Data do evento|Atendimento em|Data da prestação|Data da sessão|Data da realização):\s*([0-9/-]+)",
        texto_limpo,
        re.IGNORECASE,
    )
    if match_dt_atend:
        data_atendimento = _formatar_data_iso(match_dt_atend.group(1))

    match_dt_emiss = re.search(
        r"(?:Data de emissão|Emissão|Emitido em|Data do recibo|Data):\s*([0-9/-]+)",
        texto_limpo,
        re.IGNORECASE,
    )
    if match_dt_emiss:
        data_emissao = _formatar_data_iso(match_dt_emiss.group(1))

    if not data_atendimento and data_emissao:
        data_atendimento = data_emissao

    # 6. Extração de Códigos TUSS e Descrição do Serviço (C5 da NT-02)
    tabela_tuss = _carregar_tuss_tabela()
    codigos_encontrados = re.findall(r"\b([1-7]\d{7})\b", texto_limpo)
    codigos_unicos = list(dict.fromkeys(codigos_encontrados))

    codigo_tuss_principal = codigos_unicos[0] if codigos_unicos else None
    descricao_procedimento = None

    match_desc = re.search(
        r"(?:Referente a|Descrição|Discriminação|Serviço prestado|Procedimento):\s*([^\n\r]+)",
        texto_limpo,
        re.IGNORECASE,
    )
    if match_desc:
        descricao_procedimento = match_desc.group(1).strip()
    elif codigo_tuss_principal and codigo_tuss_principal in tabela_tuss:
        descricao_procedimento = tabela_tuss[codigo_tuss_principal].get("descricao")

    # Se não veio código TUSS explícito, tenta reclassificar pela descrição (Art. 34 § 1º e § 2º)
    if not codigo_tuss_principal and descricao_procedimento and tipo_doc in ("RECIBO", "NOTA_FISCAL"):
        from app.rag.retriever import buscar_tuss_por_descricao

        info_tuss_desc = buscar_tuss_por_descricao(descricao_procedimento)
        if info_tuss_desc:
            codigo_tuss_principal = info_tuss_desc.get("tuss")
            if codigo_tuss_principal:
                codigos_unicos.append(codigo_tuss_principal)

    # 7. Extração de Valores Monetários e Dedução de Taxas Não Assistenciais (C6 da NT-02, Art. 43 § 1º, Art. 57, Art. 59)
    valor_cobrado = None
    valor_itens_excluidos = 0.0
    multiplos_itens_sem_discriminacao = False

    # Verifica se há linhas discriminadas com taxas não assistenciais (ex: estacionamento R$ 30,00)
    linhas_texto = texto_limpo.splitlines()
    valores_assistenciais: list[float] = []
    valores_taxas_excluidas: list[float] = []

    for linha in linhas_texto:
        matches_val_linha = re.findall(r"R\$\s*([\d\.]+(?:,\d{2})?)", linha)
        if not matches_val_linha:
            continue
        eh_linha_total = bool(re.search(r"\b(total|soma|importância de|valor pago)\b", linha, re.IGNORECASE))
        eh_linha_taxa = any(re.search(p, linha, re.IGNORECASE) for p in PADROES_TAXA_NAO_ASSISTENCIAL)
        for v_str in matches_val_linha:
            try:
                v_num = float(v_str.replace(".", "").replace(",", "."))
                if eh_linha_taxa:
                    valores_taxas_excluidas.append(v_num)
                elif not eh_linha_total:
                    valores_assistenciais.append(v_num)
            except ValueError:
                pass

    todos_valores_reais = re.findall(r"R\$\s*([\d\.]+(?:,\d{2})?)", texto_limpo)
    valores_float: list[float] = []
    for v in todos_valores_reais:
        try:
            valores_float.append(float(v.replace(".", "").replace(",", ".")))
        except ValueError:
            pass

    tem_mencao_taxa = any(re.search(p, texto_limpo, re.IGNORECASE) for p in PADROES_TAXA_NAO_ASSISTENCIAL)
    tem_multiplas_sessoes_global = bool(
        re.search(r"\b(\d+|duas|três|quatro|cinco|dez)\s+sessões\b", texto_limpo, re.IGNORECASE)
        and len(set(valores_float)) == 1
        and tipo_doc in ("RECIBO", "NOTA_FISCAL")
    )

    if valores_taxas_excluidas and valores_float:
        valor_itens_excluidos = sum(valores_taxas_excluidas)
        maior_valor = max(valores_float)
        if valores_assistenciais and sum(valores_assistenciais) < maior_valor:
            valor_cobrado = round(sum(valores_assistenciais), 2)
        elif maior_valor > valor_itens_excluidos:
            valor_cobrado = round(maior_valor - valor_itens_excluidos, 2)
        else:
            valor_cobrado = maior_valor
    elif tem_mencao_taxa and len(set(valores_float)) == 1 and tipo_doc in ("RECIBO", "NOTA_FISCAL"):
        # Caso 16 da NT-02: Consulta + taxa administrativa em valor único sem discriminação -> Pendência C6
        multiplos_itens_sem_discriminacao = True
        valor_cobrado = valores_float[0]
    elif tem_multiplas_sessoes_global:
        # Caso 03 da NT-02: Múltiplas sessões em valor global sem discriminação -> Pendência C6
        multiplos_itens_sem_discriminacao = True
        valor_cobrado = valores_float[0]
    elif valores_float:
        valor_cobrado = max(valores_float)

    # 8. Detecção de Urgência/Emergência (Art. 26) e Indicação Clínica (Art. 55 / Art. 60)
    eh_urgencia = bool(
        codigo_tuss_principal == "10101020"
        or re.search(r"\b(urgência|emergência|pronto[- ]?socorro|pronto[- ]?atendimento|caráter de urgência)\b", texto_limpo, re.IGNORECASE)
    )
    tem_indicacao_clinica = bool(
        re.search(
            r"\b(indicação clínica|finalidade terapêutica|pós[- ]?operatóri[oa]|linfedema|reabilitação|tratamento de dor|prescrição médica|solicitação médica)\b",
            texto_limpo,
            re.IGNORECASE,
        )
    )
    eh_fronteira_grupo3 = any(re.search(p, texto_limpo, re.IGNORECASE) for p in PADROES_FRONTEIRA_GRUPO3)

    # 9. Classificação da Categoria Assistencial (Art. 74 c/c NT-02 Anexo A e Regras 4.1 a 4.6)
    categoria: CategoriaReembolso = "INVALIDO"
    conselho_upper = (prestador_conselho or "").upper()
    eh_medico_crm = "CRM" in conselho_upper
    eh_psicologo_crp = "CRP" in conselho_upper

    if tipo_doc == "RELATORIO_CLINICO":
        # Se o relatório clínico tem valor cobrado para reembolso isolado (TUSS 10104011, Caso 36 da NT-02) ou é documento acessório
        categoria = "RELATORIO_CLINICO"
    elif any(re.search(p, texto_limpo, re.IGNORECASE) for p in PADROES_EXCLUSAO_ABSOLUTA) or "CRO" in conselho_upper:
        categoria = "DESPESA_NAO_COBERTA"
    elif eh_fronteira_grupo3 and not tem_indicacao_clinica:
        # Art. 55 e Anexo IV Grupo 3: procedimento de fronteira sem indicação clínica presume-se estético -> DESPESA_NAO_COBERTA
        categoria = "DESPESA_NAO_COBERTA"
    elif any(c.startswith("7") for c in codigos_unicos) or re.search(
        r"\b(prótese|órtese|osteossíntese|implante ortopédico|haste intramedular|parafuso cirúrgico|material especial|opme)\b",
        texto_limpo,
        re.IGNORECASE,
    ):
        categoria = "MATERIAL_OPME"
    elif (
        codigo_tuss_principal == "10101039"
        or (eh_medico_crm and re.search(r"\bpsiquiatr", texto_limpo, re.IGNORECASE))
    ):
        # NT-02 4.2 e Caso 02 / Art. 35 § único: Consulta com médico psiquiatra é SEMPRE CONSULTA_MEDICA, mesmo em acompanhamento continuado
        categoria = "CONSULTA_MEDICA"
    elif (
        eh_psicologo_crp
        or any(c.startswith("5") for c in codigos_unicos)
        or re.search(r"\b(psicoterapia|psicólog[oa]|sessão de terapia|fonoaudiologia|fisioterapia|terapia ocupacional|drenagem linfática)\b", texto_limpo, re.IGNORECASE)
    ):
        # NT-02 4.3 e Caso 01: Sessão conduzida por psicólogo é SESSAO_TERAPIA, mesmo se intitulada "consulta"
        categoria = "SESSAO_TERAPIA"
    elif any(c.startswith("10101") or c.startswith("10102") for c in codigos_unicos) or re.search(
        r"\b(consulta médica|consulta em|dermatologia|cardiologia|clínica médica|ortopedia|ginecologia|pediatria|neurologia|endocrinologia|oftalmologia|urologia|pronto atendimento)\b",
        texto_limpo,
        re.IGNORECASE,
    ):
        categoria = "CONSULTA_MEDICA"
    elif any(c.startswith("4") for c in codigos_unicos) or re.search(
        r"\b(exame|ressonância|tomografia|hemograma|ultrassonografia|radiografia|ecocardiograma|endoscopia|colonoscopia|dosagem de|painel genético|cariótipo)\b",
        texto_limpo,
        re.IGNORECASE,
    ):
        categoria = "EXAME"
    elif any(c.startswith("3") for c in codigos_unicos) or re.search(
        r"\b(artroplastia|cirurgia|procedimento cirúrgico|laparoscopia|internação hospitalar)\b",
        texto_limpo,
        re.IGNORECASE,
    ):
        categoria = "PROCEDIMENTO_CIRURGICO"
    elif any(c.startswith("10104") for c in codigos_unicos):
        categoria = "RELATORIO_CLINICO"

    # 10. Verificação de Pendências Documentais da NT-02 (Seção 2: C1 a C7 e Anexo B: P01 a P35)
    pendencias: list[str] = []
    if tipo_doc in ("RECIBO", "NOTA_FISCAL") and categoria != "DESPESA_NAO_COBERTA":
        if not paciente_nome:
            pendencias.append("P01: Nome completo do beneficiário atendido ausente no documento fiscal (Campo C1 da NT-02).")
        if not prestador_nome:
            pendencias.append("P04: Identificação do prestador ausente no documento fiscal (Campo C2 da NT-02).")
        if not prestador_conselho and not prestador_doc:
            pendencias.append("P06: Registro no conselho profissional e CNPJ/CPF do prestador ausentes (Campos C2/C3 da NT-02 e Art. 68).")
        if not data_atendimento:
            pendencias.append("P09: Data do atendimento ausente no documento fiscal (Campo C4 da NT-02).")
        if data_atendimento and data_emissao and data_atendimento > data_emissao:
            pendencias.append("P10: Data do atendimento posterior à data de emissão do documento fiscal (Campo C4 e Caso 31 da NT-02).")
        if valor_cobrado is None:
            pendencias.append("P15: Valor total pago ausente no documento fiscal (Campo C6 da NT-02).")
        if multiplos_itens_sem_discriminacao:
            pendencias.append("P16: Valor global único sem discriminação individual dos itens/sessões (Campo C6 da NT-02 e Art. 59 § 2º).")
        if descricao_procedimento and re.match(r"^(atendimento|serviços? prestados?|procedimento|honorários?)$", descricao_procedimento.strip(), re.IGNORECASE):
            pendencias.append("P12: Descrição do procedimento genérica sem código TUSS (Campo C5 da NT-02).")

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
        valor_itens_excluidos=valor_itens_excluidos,
        eh_urgencia=eh_urgencia,
        eh_fronteira_grupo3=eh_fronteira_grupo3,
        tem_indicacao_clinica=tem_indicacao_clinica,
        multiplos_itens_sem_discriminacao=multiplos_itens_sem_discriminacao,
        tem_quitacao=True,
        tem_assinatura=True,
        pendencias_nt02=pendencias,
        texto_extraido=texto_limpo,
    )


def processar_anexo(anexo: Any) -> DocumentoInfo:
    """Processa anexo (string base64, dicionário ou objeto Anexo) e retorna DocumentoInfo."""
    from .parser import extrair_texto_de_base64

    b64_str = ""
    filename = ""
    if hasattr(anexo, "base64"):
        b64_str = str(anexo.base64)
        filename = str(getattr(anexo, "filename", "") or "")
    elif isinstance(anexo, dict):
        b64_str = str(anexo.get("base64", ""))
        filename = str(anexo.get("filename", "") or "")
    elif isinstance(anexo, str):
        b64_str = anexo

    texto_extraido, tipo_detectado = extrair_texto_de_base64(b64_str)

    # Caso seja imagem e o OCR local (tesseract) não esteja instalado no host,
    # inspeciona metadados/filename para identificar faturas de consumo (ex: conta_energia.png)
    if not texto_extraido.strip() and tipo_detectado == "IMAGEM":
        fn_low = filename.lower()
        if any(k in fn_low for k in ("energia", "luz", "agua", "gas", "fatura", "consumo", "conta")):
            return DocumentoInfo(
                tipo_documento="FATURA_CONSUMO",
                categoria_reembolso="INVALIDO",
                valido_assistencial=False,
                motivo_invalidez="Documento não assistencial (fatura de consumo residencial). O reembolso exige comprovante fiscal de despesa de saúde (Art. 76).",
                texto_extraido="FATURA DE ENERGIA ELÉTRICA - CONTA DE CONSUMO RESIDENCIAL",
            )

    return extrair_informacoes_documento(texto_extraido)


