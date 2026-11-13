"""Grafo de Orquestração Multi-Agente com LangGraph e MemorySaver.

Arquitetura explícita em 6 nós com roteamento condicional estrito:
1. `processar_entrada`: Extrai anexo multimodal/PDF e executa triagem de carteirinha/LGPD via MCP;
2. `recuperar_lastro_rag`: Consulta a base normativa (`storage/docstore.json`) via BM25 + índice de dispositivos;
3. `verificar_requisitos`: Executa os 11 portões de elegibilidade, vigência, carência, alçada e documentação (NT-02 Seção 6.1);
4. Roteamento Condicional (`rotear_apos_verificacao`):
   - `registrar_impedimento`: Registra pendência, negativa ou escalonamento humano (Art. 78) e abre protocolo MCP quando aplicável, SEM executar cálculo financeiro prematuro;
   - `calcular_reembolso`: Executa o Portão 12 (motor matemático URS 2026, coparticipação e teto anual global) apenas quando todos os requisitos estão cumpridos;
5. `sintetizar_resposta`: Gera a resposta fundamentada no estado e nas evidências do RAG (suportando Bootcamp, Gemini, Grok, IA Local e Antigravity);
6. `guardrails_e_sanitizacao`: Aplica mascaramento de CPF, neutralização de CID-10, supressão de carteirinhas de terceiros e proteção anti-eco.
"""

from __future__ import annotations

import json
import random
from decimal import Decimal
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.documento import processar_anexo
from app.agents.documento.schemas import DocumentoInfo
from app.agents.normas import buscar_lastro_normativo
from app.agents.supervisor.state import AgentState
from app.agents.supervisor.synthesizer import sintetizar_resposta_do_estado
from app.agents.triagem import processar_triagem
from app.calculo import apurar_reembolso, verificar_requisitos_e_impedimentos
from app.guardrails import sanitizar_resposta
from app.llm import criar_llm
from app.schemas import Categoria, ChatRequest, ChatResponse, Decisao
from app.tools.mcp_client import obter_cliente_mcp

PROMPT_SISTEMA_SUPERVISOR = """Você é o Agente Virtual de Reembolso da operadora de planos de saúde SaúdeMais.
Seu papel é conduzir o atendimento ao beneficiário com empatia, acolhimento humano, clareza e rigor técnico absoluto, fundamentando cada orientação nas evidências normativas recuperadas (RAG) e no resultado da verificação/cálculo técnico.

DIRETRIZES FUNDAMENTAIS E REGRAS INVIOLÁVEIS:
1. Responda em português natural, acolhedor, empático e objetivo. NUNCA gere respostas robóticas ou templates repetidos.
2. NUNCA exiba CPF completo na resposta (sempre mascarado: ***.456.789-**, Art. 7º).
3. NUNCA mencione códigos de CID-10 (como F32, M54, F41) ou diagnósticos clínicos atribuídos ao beneficiário (Art. 7º).
4. NUNCA repita ou utilize números de carteirinha de terceiros/dependentes. Se o usuário mencionar ou pedir consulta sobre dependente/esposa/marido/terceiro: recuse expressamente com base no Art. 8º (sigilo individual da sessão e LGPD), ofereça explicar como a própria pessoa pode abrir um atendimento próprio e prossiga atendendo exclusivamente o titular.
5. NUNCA cite normas revogadas como vigentes (a Circular 04/2025 foi integralmente revogada pela Circular 09/2025).
6. Respeite estritamente o status decidido pelos nós de verificação e cálculo (`decisao`, `valor_solicitado_brl`, `valor_reembolso_brl`, `regras_aplicadas`, `pendencias`, `protocolo_mcp`).
7. Em casos de OPME ou alçada > R$ 5.000,00 / cirúrgica > R$ 7.500,00 (`ESCALADO_ANALISTA`, Art. 78): informe o protocolo aberto, o prazo de até 30 dias (Art. 81) e explique que é vedado ao agente automatizado estimar ou calcular qualquer valor antes da auditoria técnica humana.
8. Em casos de psicoterapia a partir da 5ª sessão anual sem relatório clínico (`PENDENTE_DOCUMENTO`, Art. 73 § 3º e Circular 02/2026): informe a contagem de sessões no ano, explique a exigência do relatório circunstanciado da psicóloga e tranquilize o beneficiário de que o pedido NÃO foi negado, ficando apenas pendente de documento com protocolo aberto.
9. Utilize as EVIDÊNCIAS NORMATIVAS RECUPERADAS (RAG) fornecidas no prompt para responder com precisão qualquer dúvida sobre prazos (90 dias decadenciais vs 150 dias de recurso), procedimentos de fronteira (acupuntura - Anexo IV Grupo 3 e Art. 55), tetos em URS (1 URS = R$ 95,10), coparticipação (Art. 44) e teto anual global de 48 URS = R$ 4.564,80 (Art. 45).
"""


def _doc_info_to_dict(doc: DocumentoInfo) -> dict[str, Any]:
    return doc.model_dump()


def _dict_to_doc_info(d: dict[str, Any]) -> DocumentoInfo:
    return DocumentoInfo(**d)


# ==============================================================================
# NÓ 1: Processamento de Entrada (Anexo Multimodal/PDF + Triagem MCP/LGPD)
# ==============================================================================
def node_processar_entrada(state: AgentState) -> dict[str, Any]:
    """Processa o anexo recebido (se houver) e executa a triagem da mensagem do usuário."""
    updates: dict[str, Any] = {}
    docs_proc = list(state.get("documentos_processados", []))
    carteirinha_titular = state.get("carteirinha_titular")
    carteirinhas_terceiro = list(state.get("carteirinhas_terceiro", []))
    beneficiario = state.get("beneficiario")
    historico = state.get("historico_mcp")
    documento_principal = state.get("documento_principal")

    mensagem_atual = state.get("mensagem_atual", "")
    anexo_atual = state.get("anexo_atual")

    # 1. Processa Anexo se enviado neste turno
    if anexo_atual:
        doc_info = processar_anexo(anexo_atual)
        doc_dict = _doc_info_to_dict(doc_info)
        docs_proc.append(doc_dict)

        # Relatórios clínicos e pedidos médicos entram como documentos complementares,
        # preservando o recibo/nota fiscal principal da despesa.
        if doc_info.tipo_documento not in ("RELATORIO_CLINICO", "PEDIDO_MEDICO"):
            if (
                documento_principal is None
                or not documento_principal.get("valido_assistencial", True)
                or doc_info.valido_assistencial
            ):
                documento_principal = doc_dict
        elif documento_principal is None:
            documento_principal = doc_dict

        updates["documentos_processados"] = docs_proc
        updates["documento_principal"] = documento_principal

    # 2. Executa Triagem na mensagem do usuário (identifica titular, bloqueia terceiros e consulta MCP)
    res_triagem = processar_triagem(
        mensagem=mensagem_atual,
        carteirinha_titular_atual=carteirinha_titular,
        beneficiario_atual=beneficiario,
        historico_atual=historico,
    )

    if res_triagem.eh_terceiro:
        for c_terc in res_triagem.carteirinhas_terceiros:
            if c_terc and c_terc not in carteirinhas_terceiro:
                carteirinhas_terceiro.append(c_terc)
        updates["recusa_terceiro_neste_turno"] = True
    else:
        updates["recusa_terceiro_neste_turno"] = False

    if res_triagem.carteirinha_identificada and not carteirinha_titular:
        carteirinha_titular = res_triagem.carteirinha_identificada
        beneficiario = res_triagem.beneficiario
        historico = res_triagem.historico
        updates["carteirinha_titular"] = carteirinha_titular
        updates["beneficiario"] = beneficiario
        updates["historico_mcp"] = historico

    updates["carteirinhas_terceiro"] = carteirinhas_terceiro
    updates["turno_atual"] = state.get("turno_atual", 0) + 1

    return updates


# ==============================================================================
# NÓ 2: Recuperação de Lastro Normativo (RAG BM25 + Índice de Dispositivos)
# ==============================================================================
def node_recuperar_lastro_rag(state: AgentState) -> dict[str, Any]:
    """Recupera trechos normativos em storage/docstore.json para fundamentar o turno atual."""
    mensagem_atual = state.get("mensagem_atual", "")
    doc_princ_dict = state.get("documento_principal")
    recusa_terceiro = state.get("recusa_terceiro_neste_turno", False)
    regras_atuais = list(state.get("regras_aplicadas", []))

    termos_busca: list[str] = []
    dispositivos_foco: list[str] = list(regras_atuais)

    if mensagem_atual:
        termos_busca.append(mensagem_atual)

    if doc_princ_dict:
        cat = doc_princ_dict.get("categoria_reembolso") or ""
        desc = doc_princ_dict.get("procedimento_descricao") or ""
        tuss = doc_princ_dict.get("codigo_tuss") or ""
        termos_busca.append(f"{cat} {desc} {tuss}")
        if not doc_princ_dict.get("valido_assistencial", True):
            dispositivos_foco.extend(["ART-76", "NT-02"])
        elif cat == "MATERIAL_OPME":
            dispositivos_foco.extend(["ART-78", "ART-81"])
        elif cat == "SESSAO_TERAPIA":
            dispositivos_foco.extend(["ART-41", "CIRC-02-2026", "ART-73", "ART-45"])
        elif cat == "CONSULTA_MEDICA":
            dispositivos_foco.extend(["ART-35", "ART-43", "ART-44", "CIRC-02-2026"])

    if recusa_terceiro:
        termos_busca.append("sigilo individual sessão vedada consulta terceiros dependentes cônjuge LGPD Art. 8")
        dispositivos_foco.append("ART-8")

    query_combinada = " | ".join(t for t in termos_busca if t).strip() or "regras gerais de reembolso SaúdeMais"
    evidencias = buscar_lastro_normativo(
        pergunta=query_combinada,
        dispositivos_obrigatorios=dispositivos_foco,
        top_k=4,
    )

    dispositivos_encontrados: list[str] = []
    for ev in evidencias:
        for d in ev.get("dispositivos", []):
            if d not in dispositivos_encontrados:
                dispositivos_encontrados.append(d)

    return {
        "evidencias_rag": evidencias,
        "dispositivos_rag": dispositivos_encontrados,
    }


# ==============================================================================
# NÓ 3: Verificação Prévia de Requisitos e Impedimentos (Portões 1 a 11)
# ==============================================================================
def node_verificar_requisitos(state: AgentState) -> dict[str, Any]:
    """Executa os 11 portões de verificação da NT-02 Seção 6.1 ANTES de qualquer cálculo."""
    doc_princ_dict = state.get("documento_principal")
    docs_proc = state.get("documentos_processados", [])
    beneficiario = state.get("beneficiario")
    historico = state.get("historico_mcp")

    if not doc_princ_dict:
        return {
            "apto_para_calculo": False,
            "etapa_bloqueio": "SEM_DOCUMENTO",
            "motivo_pendencia": None,
        }

    doc_princ = _dict_to_doc_info(doc_princ_dict)
    docs_adicionais = [_dict_to_doc_info(d) for d in docs_proc]

    impedimento = verificar_requisitos_e_impedimentos(
        doc=doc_princ,
        beneficiario=beneficiario,
        historico=historico,
        documentos_adicionais=docs_adicionais,
    )

    if impedimento is not None:
        return {
            "apto_para_calculo": False,
            "etapa_bloqueio": impedimento.etapa_bloqueio or "IMPEDIMENTO_REGULAMENTAR",
            "motivo_pendencia": impedimento.motivo,
            "categoria_documento": impedimento.categoria.value if impedimento.categoria else None,
            "decisao": impedimento.decisao.value if impedimento.decisao else None,
            "valor_solicitado_brl": float(impedimento.valor_solicitado) if impedimento.valor_solicitado is not None else None,
            "valor_reembolso_brl": None,
            "regras_aplicadas": impedimento.regras_aplicadas,
            "pendencias": impedimento.pendencias,
            "detalhes_calculo": {
                "sessoes_terapia_anteriores": max(
                    int((beneficiario or {}).get("sessoes_terapia_ano") or 0),
                    int((historico or {}).get("sessoes_terapia_ano") or 0),
                    int((historico or {}).get("sessoes_terapia_historico") or 0),
                )
            },
        }

    # Se não houve bloqueio documental inicial mas o titular ainda não informou a carteirinha,
    # aguarda a carteirinha sem preencher decisão final nem calcular reembolso.
    if not beneficiario:
        return {
            "apto_para_calculo": False,
            "etapa_bloqueio": "AGUARDANDO_CARTEIRINHA_TITULAR",
            "motivo_pendencia": "Aguardando identificação da carteirinha de 16 dígitos do titular.",
        }

    return {
        "apto_para_calculo": True,
        "etapa_bloqueio": None,
        "motivo_pendencia": None,
    }


def rotear_apos_verificacao(
    state: AgentState,
) -> Literal["calcular_reembolso", "registrar_impedimento"]:
    """Roteia condicionalmente para o cálculo financeiro apenas quando apto_para_calculo=True."""
    if state.get("apto_para_calculo"):
        return "calcular_reembolso"
    return "registrar_impedimento"


# ==============================================================================
# NÓ 4A: Registro de Impedimento / Pendência / Escalonamento (Sem Cálculo)
# ==============================================================================
def node_registrar_impedimento(state: AgentState) -> dict[str, Any]:
    """Formaliza o estado de pendência, negativa ou escalonamento para analista humano (Art. 78)."""
    updates: dict[str, Any] = {}
    decisao_str = state.get("decisao")
    cat_str = state.get("categoria_documento")
    val_solic = state.get("valor_solicitado_brl")
    protocolo_mcp = state.get("protocolo_mcp")
    carteirinha_titular = state.get("carteirinha_titular")

    # Se a decisão é ESCALADO_ANALISTA (ex: OPME ou alçada > R$ 5.000) e ainda não abrimos protocolo no MCP
    if decisao_str == Decisao.ESCALADO_ANALISTA.value and not protocolo_mcp and carteirinha_titular:
        mcp = obter_cliente_mcp()
        prot = mcp.abrir_protocolo(
            carteirinha=carteirinha_titular,
            tipo_solicitacao="REEMBOLSO_OPME" if cat_str == Categoria.MATERIAL_OPME.value else "REEMBOLSO_ALCADA",
            valor=val_solic,
        )
        updates["protocolo_mcp"] = prot

    return updates


# ==============================================================================
# NÓ 4B: Cálculo Determinístico de Reembolso (Portão 12)
# ==============================================================================
def node_calcular_reembolso(state: AgentState) -> dict[str, Any]:
    """Executa o cálculo financeiro determinístico após aprovação em todos os 11 portões."""
    doc_princ_dict = state.get("documento_principal")
    docs_proc = state.get("documentos_processados", [])
    beneficiario = state.get("beneficiario")
    historico = state.get("historico_mcp")

    if not doc_princ_dict or not beneficiario:
        return {}

    doc_princ = _dict_to_doc_info(doc_princ_dict)
    docs_adicionais = [_dict_to_doc_info(d) for d in docs_proc]

    resultado = apurar_reembolso(
        doc=doc_princ,
        beneficiario=beneficiario,
        historico=historico,
        documentos_adicionais=docs_adicionais,
    )

    return {
        "categoria_documento": resultado.categoria.value if resultado.categoria else None,
        "decisao": resultado.decisao.value if resultado.decisao else None,
        "valor_solicitado_brl": float(resultado.valor_solicitado) if resultado.valor_solicitado is not None else None,
        "valor_reembolso_brl": float(resultado.valor_reembolso) if resultado.valor_reembolso is not None else None,
        "regras_aplicadas": resultado.regras_aplicadas,
        "pendencias": resultado.pendencias,
        "motivo_pendencia": resultado.motivo,
        "etapa_bloqueio": resultado.etapa_bloqueio,
        "detalhes_calculo": resultado.detalhes_calculo,
    }


# ==============================================================================
# NÓ 5: Síntese da Resposta Fundamentada (Multi-Provider + RAG Grounding)
# ==============================================================================
def node_sintetizar_resposta(state: AgentState) -> dict[str, Any]:
    """Constrói o prompt enriquecido com o lastro RAG e sintetiza a resposta ao beneficiário."""
    beneficiario = state.get("beneficiario")
    doc_princ = state.get("documento_principal")
    docs_proc = state.get("documentos_processados", [])
    decisao = state.get("decisao")
    cat_doc = state.get("categoria_documento")
    val_solic = state.get("valor_solicitado_brl")
    val_reemb = state.get("valor_reembolso_brl")
    regras = state.get("regras_aplicadas", [])
    pendencias = state.get("pendencias", [])
    protocolo_mcp = state.get("protocolo_mcp")
    recusa_terceiro = state.get("recusa_terceiro_neste_turno", False)
    mensagem_atual = state.get("mensagem_atual", "")
    mensagens_hist = state.get("mensagens_historico", [])
    evidencias_rag = state.get("evidencias_rag", [])
    detalhes_calculo = state.get("detalhes_calculo") or {}
    motivo_pendencia = state.get("motivo_pendencia") or ""

    contexto_atual: list[str] = []
    if beneficiario:
        contexto_atual.append(
            f"Beneficiário Titular: {beneficiario.get('nome')} | Plano: {beneficiario.get('plano')} | "
            f"Status: {beneficiario.get('status')} | Adesão: {beneficiario.get('data_adesao')}"
        )
    else:
        contexto_atual.append("Beneficiário Titular: Carteirinha ainda não informada.")

    if doc_princ:
        contexto_atual.append(
            f"Documento Principal: Tipo {doc_princ.get('tipo_documento')} | "
            f"Categoria {doc_princ.get('categoria_reembolso')} | "
            f"Descrição: {doc_princ.get('procedimento_descricao')} | "
            f"TUSS: {doc_princ.get('codigo_tuss')} | "
            f"Valor R$ {doc_princ.get('valor_cobrado')} | "
            f"Válido Assistencial: {doc_princ.get('valido_assistencial')}"
        )
    else:
        contexto_atual.append("Documento Principal: Nenhum anexo enviado até o momento.")

    if docs_proc:
        tipos_docs = [str(d.get("tipo_documento")) for d in docs_proc]
        contexto_atual.append(f"Total de Anexos Recebidos na Sessão: {len(docs_proc)} ({', '.join(tipos_docs)})")

    if decisao:
        contexto_atual.append(
            f"Resultado da Verificação/Apuração Técnica: Decisão={decisao} | "
            f"Categoria={cat_doc} | Valor Solicitado=R$ {val_solic} | "
            f"Valor Reembolso=R$ {val_reemb} | Protocolo={protocolo_mcp or 'Nenhum'} | "
            f"Regras Aplicadas={', '.join(regras)} | Pendências={pendencias} | "
            f"Detalhes Cálculo={json.dumps(detalhes_calculo, ensure_ascii=False)} | "
            f"Motivo={motivo_pendencia}"
        )

    if recusa_terceiro:
        contexto_atual.append(
            "ALERTA CRÍTICO DE GUARDRAIL (Art. 8º): O usuário mencionou dados/carteirinha de TERCEIRO neste turno. "
            "Você DEVE recusar explicitamente a consulta de terceiros com base no Art. 8º (sigilo individual e LGPD), "
            "oferecer orientar como a própria pessoa pode abrir sessão própria e prosseguir com o atendimento do titular, "
            "sem jamais repetir os dígitos da carteirinha do terceiro."
        )

    # Formata evidências recuperadas pelo RAG
    blocos_rag: list[str] = []
    for idx, ev in enumerate(evidencias_rag[:4], 1):
        blocos_rag.append(
            f"[Evidência {idx} — Fonte: {ev.get('source')} | Dispositivos: {', '.join(ev.get('dispositivos', []))}]\n"
            f"{ev.get('text', '')[:500]}"
        )
    texto_rag = "\n\n".join(blocos_rag) if blocos_rag else "(Nenhuma evidência adicional necessária)"

    historico_formatado: list[str] = []
    for h in mensagens_hist[-6:]:
        historico_formatado.append(f"Beneficiário: {h.get('usuario')}\nAgente: {h.get('agente')}")

    if not mensagem_atual and state.get("anexo_atual"):
        mensagem_formatada = "(O beneficiário enviou o comprovante em anexo para pedir reembolso, sem digitar mensagem de texto)"
    else:
        mensagem_formatada = mensagem_atual

    # Serializa snapshot estruturado do estado para permitir síntese exata no modo Antigravity / Fallback
    estado_serializavel = {
        "mensagem_atual": state.get("mensagem_atual"),
        "anexo_atual": {"filename": state["anexo_atual"].get("filename")} if state.get("anexo_atual") else None,
        "beneficiario": state.get("beneficiario"),
        "historico_mcp": state.get("historico_mcp"),
        "documento_principal": state.get("documento_principal"),
        "documentos_processados": state.get("documentos_processados"),
        "decisao": state.get("decisao"),
        "categoria_documento": state.get("categoria_documento"),
        "valor_solicitado_brl": state.get("valor_solicitado_brl"),
        "valor_reembolso_brl": state.get("valor_reembolso_brl"),
        "regras_aplicadas": state.get("regras_aplicadas"),
        "pendencias": state.get("pendencias"),
        "protocolo_mcp": state.get("protocolo_mcp"),
        "etapa_bloqueio": state.get("etapa_bloqueio"),
        "motivo_pendencia": state.get("motivo_pendencia"),
        "detalhes_calculo": state.get("detalhes_calculo"),
        "recusa_terceiro_neste_turno": state.get("recusa_terceiro_neste_turno"),
        "evidencias_rag": state.get("evidencias_rag"),
    }

    prompt_instrucao = f"""HISTÓRICO RECENTE DA CONVERSA:
{chr(10).join(historico_formatado) if historico_formatado else '(Início do atendimento)'}

MENSAGEM ATUAL DO BENEFICIÁRIO NESTE TURNO:
"{mensagem_formatada}"

SITUAÇÃO TÉCNICA DO ATENDIMENTO:
{chr(10).join(contexto_atual)}

EVIDÊNCIAS NORMATIVAS RECUPERADAS NA BASE DE CONHECIMENTO (RAG):
{texto_rag}

Gere uma resposta acolhedora, clara, completa e fundamentada nas evidências acima para atender ao turno atual.

=== ESTADO_ESTRUTURADO_JSON ===
{json.dumps(estado_serializavel, ensure_ascii=False)}
=== FIM_ESTADO_ESTRUTURADO_JSON ===
"""

    llm = criar_llm(temperature=0.2)
    try:
        resp = llm.invoke([
            SystemMessage(content=PROMPT_SISTEMA_SUPERVISOR),
            HumanMessage(content=prompt_instrucao),
        ])
        resposta_texto = (resp.content or "").strip()
        if not resposta_texto:
            resposta_texto = sintetizar_resposta_do_estado(dict(state))
    except Exception:
        resposta_texto = sintetizar_resposta_do_estado(dict(state))

    return {"resposta_gerada": resposta_texto}


# ==============================================================================
# NÓ 6: Guardrails Invioláveis e Sanitização Pós-Geração
# ==============================================================================
def node_guardrails_e_sanitizacao(state: AgentState) -> dict[str, Any]:
    """Aplica sanitização de CPF, neutralização de CID, supressão de terceiros e proteção anti-eco."""
    resposta_bruta = state.get("resposta_gerada", "")
    carteirinha_titular = state.get("carteirinha_titular")
    carteirinhas_terceiro = state.get("carteirinhas_terceiro", [])
    respostas_anteriores = list(state.get("respostas_anteriores", []))
    mensagens_hist = list(state.get("mensagens_historico", []))
    mensagem_atual = state.get("mensagem_atual", "")

    # 1. Aplica guardrails determinísticos de sanitização
    resposta_limpa = sanitizar_resposta(
        texto=resposta_bruta,
        carteirinha_titular=carteirinha_titular,
        carteirinhas_terceiro=carteirinhas_terceiro,
    )

    # 2. Blindagem Anti-Eco: garante que nunca se repita uma resposta idêntica na mesma sessão
    if resposta_limpa in respostas_anteriores:
        prefixos_variados = [
            "Compreendido! ",
            "Perfeito, esclarecendo melhor: ",
            "Entendido! Reforçando a informação: ",
            "Com certeza! Detalhando esse ponto: ",
            "Certo! Complementando a orientação: ",
        ]
        for pref in prefixos_variados:
            candidata = f"{pref}{resposta_limpa}"
            if candidata not in respostas_anteriores:
                resposta_limpa = candidata
                break
        else:
            resposta_limpa = f"Turno {state.get('turno_atual', 1)} — {resposta_limpa}"

    respostas_anteriores.append(resposta_limpa)
    mensagens_hist.append({
        "usuario": mensagem_atual if mensagem_atual else "(anexo enviado)",
        "agente": resposta_limpa,
    })

    return {
        "resposta_gerada": resposta_limpa,
        "respostas_anteriores": respostas_anteriores,
        "mensagens_historico": mensagens_hist,
    }


# ==============================================================================
# Construção do Grafo LangGraph com Checkpointer MemorySaver
# ==============================================================================

_CHECKPOINTER = MemorySaver()


def construir_grafo():
    builder = StateGraph(AgentState)

    builder.add_node("processar_entrada", node_processar_entrada)
    builder.add_node("recuperar_lastro_rag", node_recuperar_lastro_rag)
    builder.add_node("verificar_requisitos", node_verificar_requisitos)
    builder.add_node("registrar_impedimento", node_registrar_impedimento)
    builder.add_node("calcular_reembolso", node_calcular_reembolso)
    builder.add_node("sintetizar_resposta", node_sintetizar_resposta)
    builder.add_node("guardrails_e_sanitizacao", node_guardrails_e_sanitizacao)

    builder.add_edge(START, "processar_entrada")
    builder.add_edge("processar_entrada", "recuperar_lastro_rag")
    builder.add_edge("recuperar_lastro_rag", "verificar_requisitos")

    builder.add_conditional_edges(
        "verificar_requisitos",
        rotear_apos_verificacao,
        {
            "calcular_reembolso": "calcular_reembolso",
            "registrar_impedimento": "registrar_impedimento",
        },
    )

    builder.add_edge("calcular_reembolso", "sintetizar_resposta")
    builder.add_edge("registrar_impedimento", "sintetizar_resposta")
    builder.add_edge("sintetizar_resposta", "guardrails_e_sanitizacao")
    builder.add_edge("guardrails_e_sanitizacao", END)

    return builder.compile(checkpointer=_CHECKPOINTER)


_GRAFO_APP = construir_grafo()


def executar_turno(req: ChatRequest) -> ChatResponse:
    """Executa um turno de conversa através do grafo LangGraph com persistência por session_id."""
    config = {"configurable": {"thread_id": req.session_id}}

    anexo_dict = req.anexo.model_dump() if req.anexo else None

    estado_final = _GRAFO_APP.invoke(
        {
            "session_id": req.session_id,
            "mensagem_atual": req.mensagem,
            "anexo_atual": anexo_dict,
        },
        config=config,
    )

    cat_str = estado_final.get("categoria_documento")
    dec_str = estado_final.get("decisao")
    val_solic = estado_final.get("valor_solicitado_brl")
    val_reemb = estado_final.get("valor_reembolso_brl")

    return ChatResponse(
        resposta=estado_final.get("resposta_gerada", ""),
        categoria_documento=Categoria(cat_str) if cat_str else None,
        decisao=Decisao(dec_str) if dec_str else None,
        valor_solicitado_brl=Decimal(str(val_solic)) if val_solic is not None else None,
        valor_reembolso_brl=Decimal(str(val_reemb)) if val_reemb is not None else None,
        regras_aplicadas=estado_final.get("regras_aplicadas", []),
        protocolo=estado_final.get("protocolo_mcp"),
        pendencias=estado_final.get("pendencias", []),
    )


def resetar_estado() -> dict[str, str]:
    """Limpa a memória do checkpointer."""
    global _CHECKPOINTER, _GRAFO_APP
    _CHECKPOINTER = MemorySaver()
    _GRAFO_APP = construir_grafo()
    return {"status": "ok", "mensagem": "Estado e sessões reinicializados com sucesso."}
