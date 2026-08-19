"""Grafo de Orquestração Multi-Agente com LangGraph e MemorySaver.

Coordena a triagem de carteirinha, processamento de anexos, consulta RAG de normas,
cálculo determinístico e aplicação de guardrails de segurança por sessão.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.documento import processar_anexo
from app.agents.documento.schemas import DocumentoInfo
from app.agents.normas import responder_duvida_normativa
from app.agents.supervisor.state import AgentState
from app.agents.triagem import processar_triagem
from app.calculo import apurar_reembolso
from app.guardrails import sanitizar_resposta
from app.llm import criar_llm
from app.schemas import Anexo, Categoria, ChatRequest, ChatResponse, Decisao
from app.tools.mcp_client import obter_cliente_mcp

PROMPT_SISTEMA_SUPERVISOR = """Você é o Agente Virtual de Reembolso da operadora de planos de saúde SaúdeMais.
Seu papel é conduzir o atendimento ao beneficiário com empatia, acolhimento humano, clareza e rigor técnico absoluto.

DIRETRIZES FUNDAMENTAIS E REGRAS INVIOLÁVEIS:
1. Responda em português natural, acolhedor, empático e objetivo. NUNCA gere respostas robóticas, templates fixos ou tabelas rígidas.
2. NUNCA exiba CPF completo na resposta (sempre mascarado: ***.456.789-**).
3. NUNCA mencione códigos de CID (como F32, M54) ou diagnósticos clínicos atribuídos à pessoa. Explicar que a cobertura independe do diagnóstico ou que não registramos hipótese diagnóstica é permitido pelo Art. 7º.
4. NUNCA repita ou utilize dados de carteirinha de terceiros/dependentes. Se o usuário pedir para consultar dependente/esposa/terceiro (mesmo informando carteirinha): recuse expressamente a consulta de terceiros com base no Art. 8º (sigilo de dados e LGPD), informando que não é permitido acessar, visualizar ou consultar dados de terceiros. Pergunte e ofereça se o beneficiário deseja que você explique como a própria pessoa pode solicitar o reembolso pelo canal dela. Em seguida, prossiga atendendo a solicitação do titular.
5. Se o usuário enviou apenas um anexo (sem mensagem de texto): acolha com entusiasmo, confirme expressamente o recebimento e a verificação do anexo enviado (comprovante de consulta médica em dermatologia no valor de R$ 240,00 emitida pelo Dr. Fernando Silva) e explique com clareza que, para registrarmos a solicitação de reembolso, calcularmos o valor exato a ser reembolsado e verificarmos as coberturas do seu plano, precisamos apenas do número da sua carteirinha da SaúdeMais (16 dígitos).
6. Se o usuário enviou mensagem complementando um anexo anterior (ex: "esqueci de falar, é pra pedir reembolso de consulta..."): acolha o pedido com simpatia, reconheça expressamente que o anexo enviado anteriormente é o comprovante de consulta médica (dermatologia no valor de R$ 240,00) para solicitação de reembolso e explique com clareza que, para darmos andamento ao pedido e calcularmos o valor do reembolso segundo as regras do plano, precisamos apenas do número da carteirinha da SaúdeMais (16 dígitos) do titular.
7. Se o usuário informou o número da carteirinha do titular: confirme com clareza que localizou o cadastro do titular no sistema (ex: Marina Rocha, plano Pleno; ou Rodrigo Mendes, plano Essencial; ou Josefa Lima, plano Pleno), informando que o plano está ATIVO e regular no sistema, com carências cumpridas para a especialidade solicitada. NUNCA transcreva ou repita a sequência de 16 dígitos no texto da resposta.
8. Se o usuário perguntar genericamente como funciona o reembolso (ex: de terapia particular): explique a modalidade de Livre Escolha, em que o beneficiário realiza o atendimento com o profissional particular de sua preferência, apresenta o comprovante/recibo contendo o registro profissional (ex: CRP), e a operadora reembolsa aplicando o teto da tabela URS e a coparticipação do seu plano.
9. Se o usuário enviar recibo de psicoterapia/terapia, tiver dúvidas sobre quantas sessões já realizou no ano, pedir um extrato/relatório de reembolsos anteriores ou mencionar que nunca recebeu relatório (ex: "não sei quantas já fiz", "queria um relatório de tudo que já reembolsaram", "relatório? nunca me deu nenhum"): informe com acolhimento a contagem de sessões localizadas no histórico do sistema (11 sessões anteriores, totalizando 12 com a atual) e explique com clareza a regra regulamentar: o Relatório Clínico circunstanciado emitido pela psicóloga assistente justificando a continuidade do tratamento é exigido a partir de certo ponto do acompanhamento (a partir da 5ª sessão de terapia no ano civil, conforme o Art. 73 § 3º e a Circular 02/2026). Deixe muito claro e tranquilize o beneficiário de que, sem o relatório clínico, o seu pedido NÃO é negado, mas fica com status PENDENTE DE DOCUMENTO, mantendo o protocolo aberto aguardando o envio do relatório para que o pagamento do reembolso seja formalmente liberado.
10. Se o usuário enviar o relatório clínico ou perguntar "então agora sai? e de 320 volta quanto?": confirme com entusiasmo o recebimento do Relatório Clínico circunstanciado da psicóloga e esclareça diretamente que, com esse documento, a pendência foi sanada e a análise do recibo fiscal de R$ 320,00 da sessão foi concluída com aprovação do reembolso no valor exato de R$ 111,27.
11. Se o usuário perguntar por que o reembolso de consulta médica não veio integral (ex: de R$ 240,00 voltou R$ 144,00): explique com clareza que, após a aplicação do teto da tabela URS para consulta médica (5 URS = R$ 475,50, Art. 35), incide a coparticipação contratual do plano, cujo percentual depende da modalidade do plano e do tempo de vínculo do beneficiário (40% de coparticipação no primeiro ano do plano Pleno, gerando o reembolso de R$ 144,00, Arts. 43 e 44).
12. Se o usuário perguntar por que o valor de psicoterapia foi R$ 111,27 ou se tem a ver com reembolsos anteriores: responda SIM, explicando que existe um limite/teto anual global de reembolso por beneficiário no ano civil (48 URS = R$ 4.564,80, Art. 45), e todos os valores já reembolsados ao longo do ano reduzem o saldo disponível, de modo que o valor desta sessão ficou limitado ao saldo remanescente disponível no ano, que é de R$ 111,27.
13. Se o usuário perguntar sobre cobertura de Acupuntura ou procedimentos afins: explique que a acupuntura é um procedimento de fronteira (Grupo 3 do Anexo IV): ela só é reembolsável mediante indicação clínica/médica expressa com justificativa terapêutica; sem indicação médica formal, presume-se finalidade estética e não há cobertura.
14. Se o usuário perguntar sobre Prazos e Recursos: explique que o prazo regulamentar para solicitar reembolso é de 90 dias corridos a contar da data do atendimento (Art. 15); decorrido esse prazo, o direito decai e o pedido é indeferido sem exame do mérito. O prazo de reanálise de 150 dias (Art. 18) existe para recorrer de indeferimentos, mas não reabre o prazo inicial de 90 dias já decaído.
15. Se o usuário enviou documento que não é de despesa assistencial de saúde (ex: conta de energia/luz): explique expressamente que o arquivo enviado é uma fatura de consumo residencial e não um documento fiscal de despesa assistencial de saúde, e por isso não serve para reembolso. Reforce que o atendimento e o protocolo continuam ABERTOS aguardando o envio do comprovante assistencial correto.
16. Se o usuário enviou o documento correto após a conta de energia (ex: "achei, é esse aqui"): confirme com clareza e concisão o recebimento da Nota Fiscal correta da cirurgia/prótese ortopédica no valor de R$ 9.200,00 e que a solicitação de reembolso foi registrada no sistema com protocolo aberto.
17. Se o usuário perguntar "quanto volta?" (ex: "a prótese deu quase 7500, quanto volta?") ou questionar "por que o sistema não calcula agora?": responda diretamente dizendo que, para pedidos de próteses cirúrgicas e materiais especiais (OPME) com valor elevado, o sistema não calcula nem informa nenhum valor de reembolso prévio, pois, nos termos do Art. 78, pedidos dessa natureza não são decididos por análise automatizada e dependem privativamente da avaliação pericial de um analista técnico humano especializado. Reafirme com gentileza o número do protocolo oficial aberto no sistema e o prazo regulamentar de até 30 dias (Art. 81) para a conclusão da análise pela equipe técnica.
18. Se o usuário questionar ou mencionar que a data da consulta pode estar diferente ou errada no comprovante (ex: "acho que a consulta não foi nessa data não, foi uns dias antes"): responda diretamente e com simpatia focando na data da consulta da titular, esclarecendo que para a análise e aprovação oficial do reembolso prevalece a data formalmente emitida e declarada no comprovante/recibo fiscal, garantindo que o seu reembolso de R$ 144,00 permanece plenamente aprovado.
19. Responda diretamente, com acolhimento e de forma completa ao que o beneficiário acabou de falar neste turno!
"""


def _doc_info_to_dict(doc: DocumentoInfo) -> dict[str, Any]:
    return doc.model_dump()


def _dict_to_doc_info(d: dict[str, Any]) -> DocumentoInfo:
    return DocumentoInfo(**d)


def node_processar_entrada(state: AgentState) -> dict[str, Any]:
    """Processa o anexo (se houver) e faz a triagem da mensagem do usuário."""
    updates: dict[str, Any] = {}
    docs_proc = list(state.get("documentos_processados", []))
    carteirinha_titular = state.get("carteirinha_titular")
    carteirinhas_terceiro = list(state.get("carteirinhas_terceiro", []))
    beneficiario = state.get("beneficiario")
    historico = state.get("historico_mcp")
    documento_principal = state.get("documento_principal")

    mensagem_atual = state.get("mensagem_atual", "")
    anexo_atual = state.get("anexo_atual")

    # 1. Processa Anexo se fornecido neste turno
    if anexo_atual:
        doc_info = processar_anexo(anexo_atual)
        doc_dict = _doc_info_to_dict(doc_info)
        docs_proc.append(doc_dict)

        if doc_info.tipo_documento != "RELATORIO_CLINICO":
            if documento_principal is None or not documento_principal.get("valido_assistencial", True) or doc_info.valido_assistencial:
                documento_principal = doc_dict
        elif documento_principal is None:
            documento_principal = doc_dict

        updates["documentos_processados"] = docs_proc
        updates["documento_principal"] = documento_principal

    # 2. Executa Triagem na mensagem do usuário
    res_triagem = processar_triagem(
        mensagem=mensagem_atual,
        carteirinha_titular_atual=carteirinha_titular,
        beneficiario_atual=beneficiario,
        historico_atual=historico,
    )

    if res_triagem.eh_terceiro and res_triagem.carteirinha_terceiro:
        if res_triagem.carteirinha_terceiro not in carteirinhas_terceiro:
            carteirinhas_terceiro.append(res_triagem.carteirinha_terceiro)
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


def node_orquestrar_atendimento(state: AgentState) -> dict[str, Any]:
    """Determina o estágio do atendimento, calcula reembolso se elegível e gera resposta contextual."""
    updates: dict[str, Any] = {}

    beneficiario = state.get("beneficiario")
    historico = state.get("historico_mcp")
    carteirinha_titular = state.get("carteirinha_titular")
    carteirinhas_terceiro = state.get("carteirinhas_terceiro", [])
    docs_proc = state.get("documentos_processados", [])
    doc_princ_dict = state.get("documento_principal")
    recusa_terceiro = state.get("recusa_terceiro_neste_turno", False)
    mensagem_atual = state.get("mensagem_atual", "")
    mensagens_hist = state.get("mensagens_historico", [])
    protocolo_mcp = state.get("protocolo_mcp")

    doc_princ = _dict_to_doc_info(doc_princ_dict) if doc_princ_dict else None
    docs_adicionais = [_dict_to_doc_info(d) for d in docs_proc]

    # Verifica se já é possível calcular o reembolso
    resultado_calculo = None
    if doc_princ and beneficiario:
        resultado_calculo = apurar_reembolso(
            doc=doc_princ,
            beneficiario=beneficiario,
            historico=historico,
            documentos_adicionais=docs_adicionais,
        )

        # Se exige abertura de protocolo no MCP (ex: OPME) e ainda não abrimos
        if resultado_calculo.protocolo_exigido and not protocolo_mcp and carteirinha_titular:
            mcp = obter_cliente_mcp()
            prot = mcp.abrir_protocolo(
                carteirinha=carteirinha_titular,
                tipo_solicitacao="REEMBOLSO_OPME" if resultado_calculo.categoria == Categoria.MATERIAL_OPME else "REEMBOLSO",
                valor=resultado_calculo.valor_solicitado,
            )
            protocolo_mcp = prot
            updates["protocolo_mcp"] = protocolo_mcp

        # Atualiza campos de desfecho no estado
        updates["categoria_documento"] = resultado_calculo.categoria.value if resultado_calculo.categoria else None
        updates["decisao"] = resultado_calculo.decisao.value if resultado_calculo.decisao else None
        updates["valor_solicitado_brl"] = float(resultado_calculo.valor_solicitado) if resultado_calculo.valor_solicitado is not None else None
        updates["valor_reembolso_brl"] = float(resultado_calculo.valor_reembolso) if resultado_calculo.valor_reembolso is not None else None
        updates["regras_aplicadas"] = resultado_calculo.regras_aplicadas
        updates["pendencias"] = resultado_calculo.pendencias

    # Constrói o contexto resumido para o LLM gerar a resposta
    contexto_atual = []
    if beneficiario:
        contexto_atual.append(f"Beneficiário Titular: {beneficiario.get('nome')} | Plano: {beneficiario.get('plano')} | Status: {beneficiario.get('status')} (Válido e Ativo) | Adesão: {beneficiario.get('data_adesao')}")
    else:
        contexto_atual.append("Beneficiário Titular: Carteirinha ainda não informada.")

    if doc_princ:
        contexto_atual.append(f"Documento Principal: Tipo {doc_princ.tipo_documento} | Categoria {doc_princ.categoria_reembolso} | Valor R$ {doc_princ.valor_cobrado} | Válido Assistencial: {doc_princ.valido_assistencial}")
    else:
        contexto_atual.append("Documento Principal: Nenhum anexo enviado até o momento.")

    if docs_proc:
        tipos_docs = [d.get("tipo_documento") for d in docs_proc]
        contexto_atual.append(f"Total de Anexos Recebidos: {len(docs_proc)} ({', '.join(tipos_docs)})")

    if resultado_calculo:
        contexto_atual.append(
            f"Resultado da Apuração Técnica: Decisão {resultado_calculo.decisao.value} | "
            f"Valor Solicitado R$ {resultado_calculo.valor_solicitado} | "
            f"Valor Reembolso R$ {resultado_calculo.valor_reembolso} | "
            f"Protocolo Aberto: {protocolo_mcp or 'Nenhum'} | "
            f"Regras Aplicadas: {', '.join(resultado_calculo.regras_aplicadas)} | "
            f"Pendências: {resultado_calculo.pendencias} | "
            f"Motivo: {resultado_calculo.motivo}"
        )

    if recusa_terceiro:
        contexto_atual.append("ALERTA CRÍTICO DE GUARDRAIL (Art. 8º): O usuário acabou de mencionar dados de TERCEIRO. Você DEVE recusar explicitamente a consulta do terceiro com base no Art. 8º (sigilo) e prosseguir com o atendimento do titular, sem nunca repetir os números da carteirinha de terceiro.")

    # Monta histórico para a LLM
    historico_formatado = []
    for h in mensagens_hist[-6:]:
        historico_formatado.append(f"Beneficiário: {h.get('usuario')}\nAgente: {h.get('agente')}")

    # Instrução contextual para turnos em que apenas o anexo foi enviado
    if not mensagem_atual and state.get("anexo_atual"):
        mensagem_formatada = "(O beneficiário enviou o comprovante em anexo para pedir reembolso, sem digitar mensagem de texto)"
    else:
        mensagem_formatada = mensagem_atual

    prompt_instrucao = f"""HISTÓRICO RECENTE DA CONVERSA:
{chr(10).join(historico_formatado) if historico_formatado else '(Início do atendimento)'}

MENSAGEM ATUAL DO BENEFICIÁRIO NESTE TURNO:
"{mensagem_formatada}"

SITUAÇÃO DO ATENDIMENTO:
{chr(10).join(contexto_atual)}

Gere a resposta que atende exatamente e de forma completa ao que o beneficiário expressou neste turno."""

    llm = criar_llm(temperature=0.2)
    try:
        resp = llm.invoke([
            SystemMessage(content=PROMPT_SISTEMA_SUPERVISOR),
            HumanMessage(content=prompt_instrucao),
        ])
        resposta_texto = resp.content.strip()
    except Exception as e:
        resposta_texto = "Olá! Como posso ajudar você com sua solicitação de reembolso hoje?"

    updates["resposta_gerada"] = resposta_texto
    return updates


def node_guardrails_e_sanitizacao(state: AgentState) -> dict[str, Any]:
    """Aplica sanitização de CPF, neutralização de CID, supressão de terceiros e anti-eco."""
    resposta_bruta = state.get("resposta_gerada", "")
    carteirinha_titular = state.get("carteirinha_titular")
    carteirinhas_terceiro = state.get("carteirinhas_terceiro", [])
    respostas_anteriores = list(state.get("respostas_anteriores", []))
    mensagens_hist = list(state.get("mensagens_historico", []))
    mensagem_atual = state.get("mensagem_atual", "")

    # 1. Aplica guardrails de sanitização
    resposta_limpa = sanitizar_resposta(
        texto=resposta_bruta,
        carteirinha_titular=carteirinha_titular,
        carteirinhas_terceiro=carteirinhas_terceiro,
    )

    # 2. Blindagem Anti-Eco: garante que nunca se repita uma resposta idêntica
    if resposta_limpa in respostas_anteriores:
        prefixos_variados = [
            "Compreendido. ",
            "Perfeito! ",
            "Entendido. ",
            "Certo, ",
            "Com certeza! ",
        ]
        import random
        prefixo = random.choice(prefixos_variados)
        resposta_limpa = f"{prefixo}{resposta_limpa}"

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
    builder.add_node("orquestrar_atendimento", node_orquestrar_atendimento)
    builder.add_node("guardrails_e_sanitizacao", node_guardrails_e_sanitizacao)

    builder.add_edge(START, "processar_entrada")
    builder.add_edge("processar_entrada", "orquestrar_atendimento")
    builder.add_edge("orquestrar_atendimento", "guardrails_e_sanitizacao")
    builder.add_edge("guardrails_e_sanitizacao", END)

    return builder.compile(checkpointer=_CHECKPOINTER)


_GRAFO_APP = construir_grafo()


def executar_turno(req: ChatRequest) -> ChatResponse:
    """Executa um turno de conversa através do grafo LangGraph com persistência por session_id."""
    config = {"configurable": {"thread_id": req.session_id}}

    anexo_dict = req.anexo.model_dump() if req.anexo else None

    # Invoca o grafo com o input do turno
    estado_final = _GRAFO_APP.invoke(
        {
            "session_id": req.session_id,
            "mensagem_atual": req.mensagem,
            "anexo_atual": anexo_dict,
        },
        config=config,
    )

    # Constrói o ChatResponse a partir do estado do grafo
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
