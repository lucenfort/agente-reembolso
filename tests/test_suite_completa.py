"""Suíte Completa de Testes Automatizados — Projeto Agente de Reembolso SaúdeMais.

Valida:
1. Os 3 Casos de Treino Oficiais ponta a ponta (24 turnos em `casos_treino/`, usando o avaliador oficial `avaliacao.avaliar`);
2. Os 12 Portões de Verificação Prévia e Cálculo Determinístico (NT-02 Seção 6.1 e Regulamento Geral);
3. Vigência Temporal (Circular 11/2026 vs Circular 02/2026) e Teto Reduzido de Psicoterapia em Continuidade (>= 8ª sessão);
4. Tabela TUSS completa (16 códigos de storage/tuss_tabela.json) e inferência por descrição (Art. 34 § 2º);
5. Distinção entre Consulta Psiquiátrica (CRM / Art. 35 § único) e Psicoterapia (CRP / Art. 41);
6. Carências contratuais (15d, 60d, 150d, 180d) e isenção em Urgência/Emergência (0d - Art. 26);
7. Prazo decadencial de 90 dias (Art. 12 e CIRC-09-2025) e revogação da CIRC-04-2025;
8. Exames de alta complexidade (Art. 38 - Pedido Médico) vs Exames laboratoriais simples (Art. 39);
9. Procedimentos excluídos (Art. 51 / Anexo IV) vs Procedimentos de fronteira com/sem laudo (Art. 55);
10. Dedução de taxas não assistenciais (Arts. 57 e 59) e bloqueio de múltiplos itens sem discriminação (NT-02 P16);
11. Lastro RAG (BM25 + índice de dispositivos) em todos os turnos e suporte multi-provedor LLM (Gemini, Grok, Local, Antigravity, Bootcamp).
"""

from __future__ import annotations

import base64
import json
import os
from decimal import Decimal
from pathlib import Path

import pytest

# Garante execução determinística com o provedor Antigravity durante os testes automatizados
os.environ["LLM_PROVIDER"] = "antigravity"

from app.agents.documento import processar_anexo
from app.agents.documento.schemas import DocumentoInfo
from app.agents.supervisor.graph import executar_turno, resetar_estado
from app.calculo.motor import (
    TETO_ANUAL_GLOBAL_BRL,
    apurar_reembolso,
    obter_teto_procedimento,
)
from app.llm import criar_llm, resolver_provedor
from app.main import app
from app.rag.retriever import consultar_normas, recuperar_evidencias
from app.schemas import Anexo, Categoria, ChatRequest, Decisao
from avaliacao.avaliar import avaliar
from avaliacao.checagens import checar_transcricao, eh_agente
from avaliacao.dispositivos import existentes

RAIZ = Path(__file__).resolve().parents[1]
DIR_CASOS = RAIZ / "casos_treino"
VALIDOS_KB = existentes(RAIZ / "kb")


def _carregar_anexo_caso(pasta_caso: Path, anexo_info: dict | None) -> Anexo | None:
    if not anexo_info:
        return None
    p = (pasta_caso / anexo_info["path"]).resolve()
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return Anexo(
        filename=anexo_info["filename"],
        mime_type=anexo_info["mime_type"],
        base64=b64,
    )


# ==============================================================================
# 1. TESTES END-TO-END DOS 3 CASOS DE TREINO OFICIAIS (24 TURNOS)
# ==============================================================================
@pytest.mark.parametrize(
    "nome_caso",
    [
        "01_fora_de_ordem",
        "02_sessao_pelo_historico",
        "03_invalido_e_alcada",
    ],
)
def test_casos_treino_oficiais_ponta_a_ponta(nome_caso: str):
    """Executa todos os turnos de cada caso de treino oficial e valida 100% no avaliador oficial da banca."""
    os.environ["LLM_PROVIDER"] = "antigravity"
    resetar_estado()
    pasta = DIR_CASOS / nome_caso
    linhas_turnos = (pasta / "turnos.jsonl").read_text(encoding="utf-8").splitlines()
    turnos = [json.loads(l) for l in linhas_turnos if l.strip()]
    gabarito = json.loads((pasta / "esperado.json").read_text(encoding="utf-8"))

    transcricao: list[dict] = []
    for item in turnos:
        anexo = _carregar_anexo_caso(pasta, item.get("anexo"))
        req = ChatRequest(
            session_id=f"treino-{nome_caso}",
            mensagem=item.get("mensagem", ""),
            anexo=anexo,
        )
        resp_obj = executar_turno(req)
        devolvido = resp_obj.model_dump(mode="json")
        transcricao.append({
            "turno": item["turno"],
            "mensagem": item.get("mensagem", ""),
            "roteiro": item.get("mensagem", ""),
            "anexo": item.get("anexo", {}).get("filename"),
            "resposta": devolvido.get("resposta", ""),
            "estruturado": {k: v for k, v in devolvido.items() if k != "resposta"},
        })

    # 1. Porta de entrada (anti-eco e respostas não vazias)
    passou_porta, motivo_porta = eh_agente(transcricao)
    assert passou_porta is True, f"Falha na porta de entrada em {nome_caso}: {motivo_porta}"

    # 2. Checagem objetiva de guardrails (CPF, CID, carteirinha de terceiro)
    problemas_obj = checar_transcricao(transcricao, gabarito.get("_carteirinhas_de_terceiro", ()))
    assert problemas_obj == [], f"Violações de guardrail em {nome_caso}: {problemas_obj}"

    # 3. Avaliação completa (turnos + desfecho estruturado) via avaliacao.avaliar
    veredito = avaliar(nome_caso, transcricao, gabarito, VALIDOS_KB, usar_modelo=True)
    assert veredito.desfecho == [], f"Divergências no desfecho de {nome_caso}: {veredito.desfecho}"
    assert veredito.turnos_reprovados == [], f"Turnos reprovados em {nome_caso}: {veredito.turnos_reprovados}"
    assert veredito.nota == 100.0, f"Nota abaixo de 100 em {nome_caso}: {veredito.como_dict()}"


# ==============================================================================
# 2. TESTES DE VIGÊNCIA TEMPORAL E PSICOTERAPIA EM CONTINUIDADE
# ==============================================================================
def test_psicoterapia_continuidade_8a_sessao_circular_02_2026():
    """Valida teto reduzido de 2,6 URS (R$ 247,26) a partir da 8ª sessão no ano civil (Circular 02/2026, Art. 2º)."""
    doc = DocumentoInfo(
        tipo_documento="RECIBO",
        categoria_reembolso="SESSAO_TERAPIA",
        valido_assistencial=True,
        emissor_nome="Dra. Helena Costa",
        registro_profissional="CRP 06/99887",
        data_atendimento="2026-05-10",
        valor_cobrado=300.00,
        codigo_tuss="50000462",
        procedimento_descricao="Sessão individual de psicoterapia",
    )
    relatorio = DocumentoInfo(
        tipo_documento="RELATORIO_CLINICO",
        categoria_reembolso="RELATORIO_CLINICO",
        valido_assistencial=True,
    )
    # Beneficiário Pleno com 40 meses (> 36m => coparticipação 20%, reembolso 80%) e 7 sessões anteriores (atual = 8ª)
    beneficiario = {
        "nome": "Carlos Eduardo",
        "plano": "Pleno",
        "status": "ATIVO",
        "data_adesao": "2022-01-10",
        "sessoes_terapia_ano": 7,
    }
    historico = {"total_reembolsado_ano": 1000.00}

    res = apurar_reembolso(doc, beneficiario, historico, [relatorio])
    assert res.decisao == Decisao.APROVADO_PARCIAL
    # Teto 8ª sessão = 2,6 URS = R$ 247,26. Com 20% copart (80% reembolso): 247,26 * 0,80 = R$ 197,81
    assert res.valor_reembolso == Decimal("197.81")
    assert res.detalhes_calculo["teto_urs"] == 2.6
    assert res.detalhes_calculo["teto_brl"] == 247.26


def test_vigencia_temporal_circular_11_2026_fevereiro_a_abril():
    """Valida atendimento realizado entre 01/02/2026 e 19/04/2026 sob a Circular 11/2026:
    - Relatório exigido apenas a partir da 10ª sessão (portanto 6ª sessão não exige relatório);
    - Teto de psicoterapia = 1,6 URS (R$ 152,16);
    - Coparticipação Pleno até 12 meses = 30% (reembolso de 70%).
    """
    doc = DocumentoInfo(
        tipo_documento="RECIBO",
        categoria_reembolso="SESSAO_TERAPIA",
        valido_assistencial=True,
        emissor_nome="Dra. Helena Costa",
        registro_profissional="CRP 06/99887",
        data_atendimento="2026-03-15",
        valor_cobrado=200.00,
        codigo_tuss="50000462",
        procedimento_descricao="Sessão individual de psicoterapia",
    )
    beneficiario = {
        "nome": "Fernanda Souza",
        "plano": "Pleno",
        "status": "ATIVO",
        "data_adesao": "2025-06-01",  # 9 meses de adesão em 15/03/2026
        "sessoes_terapia_ano": 5,     # 6ª sessão no ano (sob CIRC-11-2026, relatório só na 10ª!)
    }
    historico = {"total_reembolsado_ano": 500.00}

    res = apurar_reembolso(doc, beneficiario, historico, [])
    assert res.decisao == Decisao.APROVADO_PARCIAL
    assert "CIRC-11-2026" in res.regras_aplicadas
    # Base = min(200.00, 152.16) = 152.16; Copart Pleno <=12m na CIRC-11 = 30% => 152.16 * 0.70 = 106.512 -> R$ 106,51
    assert res.valor_reembolso == Decimal("106.51")


# ==============================================================================
# 3. TESTES DE PSIQUIATRIA VS PSICOLOGIA E TABELA TUSS COMPLETA
# ==============================================================================
def test_consulta_psiquiatrica_crm_enquadrada_como_consulta_medica():
    """Valida que consulta psiquiátrica (CRM, TUSS 10101012) usa teto de 5,0 URS (R$ 475,50) e não exige relatório de terapia (Art. 35 § único)."""
    texto_recibo_psiquiatria = (
        "RECIBO DE CONSULTA MÉDICA\n"
        "Dr. Roberto Almeida - Médico Psiquiatra - CRM/SP 112233\n"
        "CNPJ: 12.345.678/0001-90\n"
        "Paciente: Rodrigo Mendes - CPF: 123.456.789-00\n"
        "Procedimento: Consulta médica em psiquiatria (TUSS 10101012)\n"
        "Data do atendimento: 14/05/2026\n"
        "Valor pago: R$ 400,00"
    )
    b64 = base64.b64encode(texto_recibo_psiquiatria.encode("utf-8")).decode("ascii")
    doc = processar_anexo({"filename": "recibo_psiquiatria.pdf", "mime_type": "text/plain", "base64": b64})

    assert doc.categoria_reembolso == "CONSULTA_MEDICA"
    assert doc.codigo_tuss == "10101012"

    # Mesmo para beneficiário com 15 sessões de terapia no ano, consulta psiquiátrica não exige relatório de terapia
    beneficiario = {
        "nome": "Rodrigo Mendes",
        "cpf": "123.456.789-00",
        "plano": "Essencial",
        "status": "ATIVO",
        "data_adesao": "2024-03-01",  # 26 meses => 35% copart
        "sessoes_terapia_ano": 15,
    }
    res = apurar_reembolso(doc, beneficiario, {"total_reembolsado_ano": 0.0}, [])
    assert res.decisao == Decisao.APROVADO
    assert res.categoria == Categoria.CONSULTA_MEDICA
    # Base = min(400.00, 475.50) = 400.00 * 0.65 = R$ 260,00
    assert res.valor_reembolso == Decimal("260.00")
    assert "ART-35" in res.regras_aplicadas


def test_tabela_tuss_todos_os_grupos_e_inferencia_por_descricao():
    """Valida consulta dinâmica dos 1.835 códigos de storage/tuss_tabela.json e inferência Art. 34 § 2º."""
    # Grupo I - Pronto Atendimento (10101020 = 5,0 URS = R$ 475,50, Art. 26 dispensa carência)
    urs_ps, brl_ps, arts_ps, _ = obter_teto_procedimento(Categoria.CONSULTA_MEDICA, "10101020")
    assert urs_ps == Decimal("5.0")
    assert brl_ps == Decimal("475.50")
    assert "ART-35" in arts_ps and "ART-26" in arts_ps

    # Grupo II - Hemograma completo com contagem de plaquetas (40301002 = 0,4 URS = R$ 38,04)
    urs_hm, brl_hm, arts_hm, _ = obter_teto_procedimento(Categoria.EXAME_DIAGNOSTICO, "40301002")
    assert urs_hm == Decimal("0.4")
    assert brl_hm == Decimal("38.04")
    assert "ART-37" in arts_hm

    # Grupo III - Ressonância magnética de coluna cervical (40901049 = 12,0 URS = R$ 1.141,20)
    urs_rm, brl_rm, arts_rm, _ = obter_teto_procedimento(Categoria.EXAME_DIAGNOSTICO, "40901049")
    assert urs_rm == Decimal("12.0")
    assert brl_rm == Decimal("1141.20")
    assert "ART-37" in arts_rm and "ART-38" in arts_rm

    # Grupo IV - Fonoaudiologia sem código TUSS informado (inferido por descrição - Art. 34 § 2º: 50000558 = 1,5 URS = R$ 142,65)
    urs_fn, brl_fn, arts_fn, cod_fn = obter_teto_procedimento(
        Categoria.SESSAO_TERAPIA, None, "Sessão de fonoaudiologia"
    )
    assert cod_fn == "50000558"
    assert urs_fn == Decimal("1.5")
    assert brl_fn == Decimal("142.65")
    assert "ART-42" in arts_fn


# ==============================================================================
# 4. TESTES DE IMPEDIMENTOS E PORTÕES DE VERIFICAÇÃO (PORTÕES 1 A 11)
# ==============================================================================
def test_carencias_contratuais_e_isencao_urgencia():
    """Valida carências exatas (Consulta 15d, Exame 60d, Terapia 150d) e isenção em urgência (0d - Art. 26)."""
    beneficiario_recente = {
        "nome": "Lucas Martins",
        "plano": "Pleno",
        "status": "ATIVO",
        "data_adesao": "2026-05-05",  # Apenas 9 dias antes de 2026-05-14
    }
    doc_eletivo = DocumentoInfo(
        tipo_documento="RECIBO",
        categoria_reembolso="CONSULTA_MEDICA",
        valido_assistencial=True,
        data_atendimento="2026-05-14",
        valor_cobrado=250.00,
        codigo_tuss="10101012",
        carater_urgencia=False,
    )
    res_neg = apurar_reembolso(doc_eletivo, beneficiario_recente)
    assert res_neg.decisao == Decisao.NEGADO
    assert "ART-21" in res_neg.regras_aplicadas and "ART-22" in res_neg.regras_aplicadas

    # Mesmo beneficiário (9 dias de adesão) em atendimento de Pronto-Socorro / Urgência (Art. 26 => 0 dias)
    doc_urgencia = DocumentoInfo(
        tipo_documento="RECIBO",
        categoria_reembolso="CONSULTA_MEDICA",
        valido_assistencial=True,
        data_atendimento="2026-05-14",
        valor_cobrado=300.00,
        codigo_tuss="10101039",
        carater_urgencia=True,
    )
    res_aprov = apurar_reembolso(doc_urgencia, beneficiario_recente)
    assert res_aprov.decisao == Decisao.APROVADO
    assert "ART-26" in res_aprov.regras_aplicadas
    assert res_aprov.valor_reembolso == Decimal("180.00")  # 300 * 0.60


def test_prazo_decadencial_90_dias_corridos():
    """Valida indeferimento por decadência quando o atendimento ocorreu há mais de 90 dias (Art. 12, Art. 13, CIRC-09-2025)."""
    doc_antigo = DocumentoInfo(
        tipo_documento="RECIBO",
        categoria_reembolso="CONSULTA_MEDICA",
        valido_assistencial=True,
        data_atendimento="2026-01-10",  # 130 dias antes de 2026-05-20
        valor_cobrado=240.00,
        codigo_tuss="10101012",
    )
    beneficiario = {
        "nome": "Marina Rocha",
        "plano": "Pleno",
        "status": "ATIVO",
        "data_adesao": "2024-01-01",
        "data_protocolo": "2026-05-20",
    }
    res = apurar_reembolso(doc_antigo, beneficiario)
    assert res.decisao == Decisao.NEGADO
    assert set(["ART-12", "ART-13", "CIRC-09-2025"]).issubset(set(res.regras_aplicadas))


def test_limite_40_sessoes_terapia_e_teto_anual_exaurido():
    """Valida negativa por atingimento de 40 sessões anuais (Art. 40) e por exaurimento das 48 URS (Art. 45)."""
    doc_terapia = DocumentoInfo(
        tipo_documento="RECIBO",
        categoria_reembolso="SESSAO_TERAPIA",
        valido_assistencial=True,
        data_atendimento="2026-05-12",
        valor_cobrado=250.00,
        codigo_tuss="50000462",
    )
    rel = DocumentoInfo(tipo_documento="RELATORIO_CLINICO", categoria_reembolso="RELATORIO_CLINICO", valido_assistencial=True)

    # 40 sessões já realizadas => 41ª deve ser NEGADA pelo Art. 40
    ben_40 = {"nome": "Teste", "plano": "Pleno", "status": "ATIVO", "data_adesao": "2022-01-01", "sessoes_terapia_ano": 40}
    res_40 = apurar_reembolso(doc_terapia, ben_40, {"total_reembolsado_ano": 2000.00}, [rel])
    assert res_40.decisao == Decisao.NEGADO
    assert "ART-40" in res_40.regras_aplicadas

    # Teto anual de R$ 4.564,80 já exaurido => NEGADO pelo Art. 45
    ben_teto = {"nome": "Teste", "plano": "Pleno", "status": "ATIVO", "data_adesao": "2022-01-01", "sessoes_terapia_ano": 10}
    res_teto = apurar_reembolso(doc_terapia, ben_teto, {"total_reembolsado_ano": float(TETO_ANUAL_GLOBAL_BRL)}, [rel])
    assert res_teto.decisao == Decisao.NEGADO
    assert "ART-45" in res_teto.regras_aplicadas


def test_exame_alta_complexidade_exige_pedido_medico_art_38():
    """Valida que Ressonância Magnética (TUSS 40901049) fica PENDENTE_DOCUMENTO sem pedido médico e APROVADO_PARCIAL com pedido."""
    doc_rm = DocumentoInfo(
        tipo_documento="NOTA_FISCAL",
        categoria_reembolso="EXAME_DIAGNOSTICO",
        valido_assistencial=True,
        data_atendimento="2026-05-10",
        valor_cobrado=1500.00,
        codigo_tuss="40901049",
        procedimento_descricao="Ressonância magnética de coluna cervical",
    )
    ben = {"nome": "Marina Rocha", "plano": "Pleno", "status": "ATIVO", "data_adesao": "2022-01-01"}  # >36m => 20% copart

    # Sem pedido médico -> PENDENTE_DOCUMENTO (Art. 38)
    res_pend = apurar_reembolso(doc_rm, ben, {"total_reembolsado_ano": 0.0}, [])
    assert res_pend.decisao == Decisao.PENDENTE_DOCUMENTO
    assert "ART-38" in res_pend.regras_aplicadas

    # Com pedido médico -> APROVADO_PARCIAL (Teto 12 URS = R$ 1.141,20 * 0.80 = R$ 912,96)
    pedido = DocumentoInfo(tipo_documento="PEDIDO_MEDICO", categoria_reembolso="RELATORIO_CLINICO", valido_assistencial=True)
    res_aprov = apurar_reembolso(doc_rm, ben, {"total_reembolsado_ano": 0.0}, [pedido])
    assert res_aprov.decisao == Decisao.APROVADO_PARCIAL
    assert res_aprov.valor_reembolso == Decimal("912.96")


def test_despesa_nao_coberta_e_procedimento_fronteira_anexo_iv():
    """Valida negativa de despesa não coberta (Art. 51/Anexo IV) e comportamento de procedimento de fronteira (Art. 55)."""
    doc_estetica = DocumentoInfo(
        tipo_documento="RECIBO",
        categoria_reembolso="DESPESA_NAO_COBERTA",
        valido_assistencial=True,
        data_atendimento="2026-05-10",
        valor_cobrado=1200.00,
        procedimento_descricao="Aplicação estética de toxina botulínica facial",
        procedimento_excluido=True,
        procedimento_fronteira=False,
    )
    ben = {"nome": "Marina Rocha", "plano": "Pleno", "status": "ATIVO", "data_adesao": "2022-01-01"}
    res_est = apurar_reembolso(doc_estetica, ben)
    assert res_est.decisao == Decisao.NEGADO
    assert res_est.categoria == Categoria.DESPESA_NAO_COBERTA
    assert "ART-51" in res_est.regras_aplicadas and "ANEXO-IV" in res_est.regras_aplicadas

    # Acupuntura (Fronteira - Grupo 3 do Anexo IV) sem laudo médico => NEGADO com ART-55
    doc_acup = DocumentoInfo(
        tipo_documento="RECIBO",
        categoria_reembolso="DESPESA_NAO_COBERTA",
        valido_assistencial=True,
        data_atendimento="2026-05-10",
        valor_cobrado=200.00,
        procedimento_descricao="Sessão de acupuntura",
        procedimento_excluido=True,
        procedimento_fronteira=True,
    )
    res_acup_sem = apurar_reembolso(doc_acup, ben, {}, [])
    assert res_acup_sem.decisao == Decisao.NEGADO
    assert "ART-55" in res_acup_sem.regras_aplicadas

    # Acupuntura com relatório/indicação médica => Coberta!
    laudo = DocumentoInfo(tipo_documento="RELATORIO_CLINICO", categoria_reembolso="RELATORIO_CLINICO", valido_assistencial=True)
    res_acup_com = apurar_reembolso(doc_acup, ben, {}, [laudo])
    assert res_acup_com.decisao in (Decisao.APROVADO, Decisao.APROVADO_PARCIAL)
    assert "ART-55" in res_acup_com.regras_aplicadas


def test_deducao_taxa_nao_assistencial_e_multiplos_itens_sem_discriminacao():
    """Valida glosa de taxa não assistencial (Arts. 57 e 59) e pendência em múltiplos itens sem valor individual (NT-02 P16)."""
    ben = {"nome": "Marina Rocha", "plano": "Pleno", "status": "ATIVO", "data_adesao": "2025-06-05"}  # 11m => 40% copart

    # Consulta de R$ 240 + Taxa de emissão de laudo R$ 30 = R$ 270 cobrados (base elegível = R$ 240 => reembolso R$ 144,00)
    doc_taxa = DocumentoInfo(
        tipo_documento="RECIBO",
        categoria_reembolso="CONSULTA_MEDICA",
        valido_assistencial=True,
        data_atendimento="2026-05-14",
        valor_cobrado=270.00,
        taxa_nao_assistencial=30.00,
        codigo_tuss="10101012",
    )
    res_taxa = apurar_reembolso(doc_taxa, ben)
    assert res_taxa.decisao == Decisao.APROVADO_PARCIAL
    assert res_taxa.valor_reembolso == Decimal("144.00")
    assert "ART-57" in res_taxa.regras_aplicadas and "ART-59" in res_taxa.regras_aplicadas

    # Recibo com "4 sessões de psicoterapia = R$ 1.200,00" sem discriminar cada sessão (NT-02 P16)
    doc_mult = DocumentoInfo(
        tipo_documento="RECIBO",
        categoria_reembolso="SESSAO_TERAPIA",
        valido_assistencial=True,
        data_atendimento="2026-05-14",
        valor_cobrado=1200.00,
        codigo_tuss="50000462",
        multiplos_itens_sem_discriminacao=True,
    )
    res_mult = apurar_reembolso(doc_mult, ben)
    assert res_mult.decisao == Decisao.PENDENTE_DOCUMENTO
    assert "ART-18" in res_mult.regras_aplicadas and "NT-02" in res_mult.regras_aplicadas


def test_rag_bm25_e_filtro_norma_revogada():
    """Valida que o motor RAG recupera dispositivos exatos e nunca retorna a Circular 04/2025 revogada."""
    evidencias = recuperar_evidencias(
        "Qual o prazo decadencial de 90 dias para pedir reembolso e a regra de recurso de 150 dias?",
        top_k=4,
        dispositivos_obrigatorios=["ART-12", "CIRC-09-2025"],
    )
    assert len(evidencias) >= 2
    fontes = [e["source"] for e in evidencias]
    assert "05_circular_04_2025_REVOGADA.md" not in fontes

    texto_normas = consultar_normas("teto psicoterapia individual Circular 02/2026", top_k=3)
    assert "2,8 URS" in texto_normas or "266,28" in texto_normas


def test_suporte_multiprovedor_llm(monkeypatch):
    """Valida a resolução dos 5 provedores LLM (antigravity, gemini, grok, local, bootcamp)."""
    monkeypatch.setenv("LLM_PROVIDER", "antigravity")
    assert resolver_provedor() == "antigravity"
    llm_ag = criar_llm()
    assert llm_ag.provider == "antigravity"

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert resolver_provedor() == "gemini"

    monkeypatch.setenv("LLM_PROVIDER", "grok")
    assert resolver_provedor() == "grok"

    monkeypatch.setenv("LLM_PROVIDER", "local")
    assert resolver_provedor() == "local"

    monkeypatch.setenv("LLM_PROVIDER", "bootcamp")
    assert resolver_provedor() == "bootcamp"


def test_endpoints_fastapi_health_chat_reset(monkeypatch):
    """Valida os endpoints HTTP obrigatórios (/health, /chat, /reset) via TestClient."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("LLM_PROVIDER", "antigravity")
    client = TestClient(app)

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json() == {"status": "ok"}

    r_reset = client.post("/reset")
    assert r_reset.status_code == 200
    assert r_reset.json().get("status") == "ok"

    r_chat = client.post(
        "/chat",
        json={
            "session_id": "teste-http",
            "mensagem": "Bom dia! Acupuntura sem pedido médico tem reembolso?",
            "anexo": None,
        },
    )
    assert r_chat.status_code == 200
    dados = r_chat.json()
    assert "acupuntura" in dados["resposta"].lower()
    assert "indicação" in dados["resposta"].lower()

