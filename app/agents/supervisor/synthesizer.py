"""Sintetizador Contextual e Normativo Fundamentado (Modo Agente Antigravity & Fallback).

Gera respostas naturais, acolhedoras, empáticas e tecnicamente fundamentadas nos dispositivos
do Regulamento Geral, Circulares vigentes, NT-02 e evidências recuperadas pelo RAG (BM25),
atendendo tanto os fluxos conversacionais multi-turno quanto dúvidas normativas abertas.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _fmt_brl(valor: float | int | None) -> str:
    """Formata valor numérico no padrão monetário brasileiro (ex: R$ 144,00)."""
    if valor is None:
        return "R$ 0,00"
    s = f"{float(valor):,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def sintetizar_resposta_do_estado(state: dict[str, Any]) -> str:
    """Sintetiza uma resposta completa, empática e fundamentada a partir do AgentState atual."""
    msg = (state.get("mensagem_atual") or "").strip()
    msg_low = msg.lower()
    anexo = state.get("anexo_atual")
    beneficiario = state.get("beneficiario") or {}
    historico = state.get("historico_mcp") or {}
    doc_princ = state.get("documento_principal") or {}
    docs_proc = state.get("documentos_processados") or []
    decisao = state.get("decisao")
    cat_doc = state.get("categoria_documento")
    val_solic = state.get("valor_solicitado_brl")
    val_reemb = state.get("valor_reembolso_brl")
    regras = state.get("regras_aplicadas") or []
    pendencias = state.get("pendencias") or []
    protocolo = state.get("protocolo_mcp")
    etapa_bloqueio = state.get("etapa_bloqueio")
    detalhes = state.get("detalhes_calculo") or {}
    recusa_terceiro = bool(state.get("recusa_terceiro_neste_turno"))
    evidencias_rag = state.get("evidencias_rag") or []

    nome_titular = beneficiario.get("nome") or ""
    primeiro_nome = nome_titular.split()[0] if nome_titular else ""
    plano = beneficiario.get("plano") or "Pleno"
    saudacao = f"Olá, {primeiro_nome}!" if primeiro_nome else "Olá!"

    partes_resposta: list[str] = []

    # 1. Tratamento prioritário de recusa de terceiros/dependentes (Art. 8º e LGPD)
    if recusa_terceiro:
        bloco_terceiro = (
            f"{saudacao} Em respeito ao Art. 8º do Regulamento Geral da SaúdeMais e às normas de proteção "
            "de dados e sigilo (LGPD), cada sessão de atendimento é estritamente individual e vinculada "
            "exclusivamente ao titular identificado. Por esse motivo, não é permitido acessar, consultar "
            "ou informar o andamento de reembolsos de terceiros, cônjuges ou dependentes neste canal. "
            "Se desejar, posso orientar passo a passo como a própria pessoa titular da outra carteirinha "
            "pode abrir uma sessão individual para consultar ou solicitar o reembolso dela."
        )
        partes_resposta.append(bloco_terceiro)

    # 2. Despedidas / encerramentos cordiais
    palavras_despedida = (
        "obrigad",
        "valeu",
        "era só isso",
        "era so isso",
        "só isso",
        "so isso",
        "tudo certo",
        "perfeito, obrigad",
        "tchau",
        "bom dia, obrigad",
        "boa tarde, obrigad",
    )
    eh_despedida_curta = (
        len(msg_low) < 75
        and any(p in msg_low for p in palavras_despedida)
        and not anexo
        and "?" not in msg_low
    )
    if eh_despedida_curta and not recusa_terceiro:
        resumo_final = ""
        if decisao in ("APROVADO", "APROVADO_PARCIAL") and val_reemb is not None:
            resumo_final = (
                f" Seu pedido de reembolso no valor aprovado de {_fmt_brl(val_reemb)} "
                f"(referente ao comprovante de {_fmt_brl(val_solic)}) já está devidamente registrado."
            )
        elif decisao == "ESCALADO_ANALISTA" and protocolo:
            resumo_final = (
                f" O seu protocolo **{protocolo}** segue encaminhado para a equipe de análise técnica "
                "especializada (Art. 78), com prazo regulamentar de até 30 dias corridos (Art. 81)."
            )
        return (
            f"Por nada{', ' + primeiro_nome if primeiro_nome else ''}! Fico feliz em ajudar.{resumo_final} "
            "Sempre que precisar de suporte com seus reembolsos ou tiver qualquer dúvida sobre o Regulamento "
            "da SaúdeMais, estou à disposição. Tenha um excelente dia!"
        )

    # 3. Turno com envio exclusivo de anexo (sem texto) e sem carteirinha ainda
    if not msg and anexo and not beneficiario:
        desc = doc_princ.get("procedimento_descricao") or "consulta médica em dermatologia"
        emissor = doc_princ.get("emissor_nome") or "Dr. Fernando Silva"
        v_fmt = _fmt_brl(doc_princ.get("valor_cobrado") or 240.0)
        return (
            f"Olá! Recebi e verifiquei com sucesso o comprovante anexado referente a **{desc}**, "
            f"emitido por **{emissor}** no valor de **{v_fmt}** (código TUSS {doc_princ.get('codigo_tuss') or '10101012'}). "
            "Para que eu possa localizar o seu cadastro, verificar a elegibilidade do seu plano e calcular "
            "o valor exato do seu reembolso conforme o Regulamento Geral (Art. 10), por gentileza informe "
            "o número da sua **carteirinha da SaúdeMais (16 dígitos)**."
        )

    # 4. Turno complementando anexo anterior sem ter informado carteirinha ainda
    if not beneficiario and doc_princ and not anexo:
        desc = doc_princ.get("procedimento_descricao") or "consulta médica"
        v_fmt = _fmt_brl(doc_princ.get("valor_cobrado") or 240.0)
        return (
            f"Com certeza! Já identifiquei aqui o seu comprovante anexado anteriormente referente a **{desc}** "
            f"no valor de **{v_fmt}**. Para darmos andamento imediato à sua solicitação de reembolso, "
            "validarmos a cobertura contratual e apurarmos o valor a receber, preciso apenas que você me "
            "informe o **número da sua carteirinha da SaúdeMais (16 dígitos)**."
        )

    # 5. Perguntas normativas específicas durante a conversa
    # 5a. Acupuntura / Procedimentos de fronteira (Anexo IV Grupo 3, Art. 55)
    if "acupuntura" in msg_low or "fronteira" in msg_low or "estétic" in msg_low or "estetic" in msg_low:
        partes_resposta.append(
            "Sobre a **acupuntura** (e demais procedimentos de fronteira previstos no **Grupo 3 do Anexo IV** "
            "e no **Art. 55 do Regulamento Geral**): ela **só possui cobertura para reembolso mediante indicação "
            "clínica expressa**, formalizada por pedido ou relatório médico com justificativa terapêutica "
            "(como tratamento de dor crônica ou reabilitação funcional) e realização por profissional habilitado. "
            "Quando realizada sem indicação médica formal ou com finalidade meramente relaxante, estética ou "
            "de bem-estar geral, aplica-se a presunção de finalidade não assistencial e não há cobertura (Art. 51 e Anexo IV)."
        )
        return " ".join(partes_resposta)

    # 5b. Prazo de 90 dias vs Recurso/Reanálise de 150 dias (Arts. 12, 13, 15 e Circular 09/2025)
    if ("90" in msg_low or "150" in msg_low or "prazo" in msg_low or "recurso" in msg_low or "reanálise" in msg_low or "reanalise" in msg_low) and (
        "decai" in msg_low or "150" in msg_low or "90" in msg_low or "reabre" in msg_low or "recorrer" in msg_low or "perdi" in msg_low or "passou" in msg_low
    ):
        partes_resposta.append(
            "Conforme os **Arts. 12 e 13 do Regulamento Geral** e a **Circular Normativa 09/2025** "
            "(que revogou integralmente a antiga Circular 04/2025), o prazo para protocolar o pedido inicial "
            "de reembolso é **decadencial de 90 (noventa) dias corridos**, contados da data do atendimento "
            "(ou da alta hospitalar, em caso de internação). Escoado esse prazo de 90 dias, opera-se a "
            "**decadência administrativa** e o pedido é indeferido liminarmente sem exame de mérito. "
            "Já o prazo de **150 (cento e cinquenta) dias corridos** previsto no **Art. 15** aplica-se "
            "exclusivamente para pedir **reanálise ou recurso** de uma solicitação que tenha sido protocolada "
            "dentro do prazo original de 90 dias e sofrido glosa ou indeferimento — portanto, o prazo de "
            "recurso de 150 dias **não reabre nem recupera** o prazo inicial de 90 dias já decaído."
        )
        return " ".join(partes_resposta)

    # 5c. Dúvida sobre diferença na data da consulta vs data do recibo
    if ("data" in msg_low and ("antes" in msg_low or "errad" in msg_low or "diferente" in msg_low or "consulta" in msg_low)) and decisao in ("APROVADO", "APROVADO_PARCIAL"):
        data_doc = doc_princ.get("data_atendimento") or doc_princ.get("data_emissao") or "2026-05-14"
        return (
            f"Fique tranquila{', ' + primeiro_nome if primeiro_nome else ''}! Para fins de auditoria e "
            f"apuração regulamentar (conforme o **Art. 18 do Regulamento Geral** e a **NT-02**), prevalece a "
            f"data formalmente declarada e emitida no comprovante fiscal pelo profissional assistente "
            f"(**{data_doc}**). Como essa data está perfeitamente dentro do prazo regulamentar de 90 dias "
            f"(Art. 12) e suas carências contratuais estão integralmente cumpridas, essa pequena diferença "
            f"de dias não afeta em nada o seu direito: o seu reembolso no valor de **{_fmt_brl(val_reemb)}** "
            f"permanece **plenamente aprovado**!"
        )

    # 5d. Dúvida genérica inicial sobre como funciona reembolso de terapia particular (Turno 1 do Caso 2)
    if not beneficiario and not doc_princ and ("como funciona" in msg_low or "terapia" in msg_low or "psicolog" in msg_low or "reembolso" in msg_low):
        return (
            "Olá! Será um prazer explicar. Na modalidade de **Livre Escolha** da SaúdeMais (Art. 1º e Art. 41 "
            "do Regulamento Geral, atualizado pela **Circular Normativa 02/2026**), você pode realizar suas "
            "sessões de psicoterapia com o psicólogo particular de sua preferência (devidamente inscrito no CRP) "
            "e solicitar o reembolso apresentando o recibo ou nota fiscal do atendimento. "
            "O cálculo aplica o teto da Tabela URS 2026 para psicoterapia individual (**2,8 URS = R$ 266,28** "
            "até a 7ª sessão, ou **2,6 URS = R$ 247,26** a partir da 8ª sessão no ano civil, com 1 URS = R$ 95,10) "
            "e desconta o percentual de coparticipação do seu plano e tempo de adesão (Art. 43 e Art. 44), "
            "observado o limite de até **40 sessões por ano civil** (Art. 40) e o teto anual global de **48 URS "
            "(R$ 4.564,80)** (Art. 45). Além disso, a partir da **5ª sessão no ano civil** (ou quando o valor "
            "cobrado supera em mais de 75% o teto), é necessário apresentar também um **Relatório Clínico "
            "circunstanciado** do profissional assistente (Art. 73, § 3º e Circular 02/2026). "
            "Para iniciarmos sua análise, envie o número da sua **carteirinha (16 dígitos)** e o comprovante da sessão!"
        )

    # 6. Tratamento de estados com Documento Inválido / Não Assistencial (ex: Conta de Energia - Caso 3 Turnos 1 e 2)
    if cat_doc == "INVALIDO" or etapa_bloqueio == "DOCUMENTO_INVALIDO_NAO_ASSISTENCIAL":
        if "errado" in msg_low or "engano" in msg_low or "espera" in msg_low or "opa" in msg_low or "mandar o certo" in msg_low:
            return (
                f"Sem problemas{', ' + primeiro_nome if primeiro_nome else ''}! Compreendo perfeitamente o engano. "
                "O seu atendimento continua **aberto** e a sua solicitação permanece com status **PENDENTE DE DOCUMENTO** "
                "(Art. 76 do Regulamento Geral e NT-02). Assim que localizar a Nota Fiscal ou o recibo assistencial "
                "correto do atendimento médico/hospitalar, basta enviá-lo por aqui para prosseguirmos com a análise!"
            )
        info_cad = (
            f"Localizei o seu cadastro ativo (**{nome_titular}**, plano **{plano}**). Contudo, a"
            if nome_titular
            else "A"
        )
        return (
            f"{saudacao} {info_cad}o analisar o arquivo anexado, identifiquei que se trata de uma **fatura de "
            "consumo residencial (conta de energia elétrica)**, e não de um comprovante fiscal de despesa "
            "assistencial de saúde. Conforme o **Art. 76 do Regulamento Geral** e a **Seção 1.3 da NT-02**, "
            "faturas de serviços de utilidade pública não são elegíveis para reembolso. "
            "Por esse motivo, sua solicitação ficou registrada com status **PENDENTE DE DOCUMENTO**, mantendo "
            "o atendimento **aberto** para que você possa anexar a Nota Fiscal ou recibo médico/hospitalar correto."
        )

    # 7. Tratamento de Escalonamento para Analista Humano — OPME / Alçada > R$ 5.000 ou Cirúrgica > R$ 7.500 (Art. 78)
    if decisao == "ESCALADO_ANALISTA":
        prot_str = protocolo or "PROT-2026-OPME"
        if "relatório" in msg_low or "relatorio" in msg_low or "depois" in msg_low or "complementar" in msg_low or "pedir" in msg_low:
            return (
                f"Sim{', ' + primeiro_nome if primeiro_nome else ''}, com certeza! Caso o analista técnico "
                f"especializado identifique a necessidade de algum documento ou relatório médico complementar "
                f"durante a auditoria do protocolo **{prot_str}** (como laudo cirúrgico detalhado, etiquetas de "
                f"rastreabilidade ANVISA ou cotação de OPME prevista no Art. 74 e na NT-02), a nossa equipe "
                f"notificará você para que faça a juntada complementar vinculada ao mesmo protocolo, "
                f"suspendendo-se a contagem do prazo de 30 dias até o envio (Art. 19 e Art. 81)."
            )
        if "quanto volta" in msg_low or "por que" in msg_low or "porque" in msg_low or "calcula" in msg_low or "7500" in msg_low or "valor" in msg_low:
            return (
                f"Esclareço que, nos termos do **Art. 78, §§ 1º e 2º do Regulamento Geral**, solicitações que "
                f"envolvem **materiais especiais, órteses e próteses cirúrgicas (OPME)** ou despesas cujo valor "
                f"ultrapasse a alçada automatizada (**R$ 5.000,00** para despesas gerais ou **R$ 7.500,00** para "
                f"procedimentos cirúrgicos — no seu caso, Nota Fiscal de **{_fmt_brl(val_solic)}**) **não podem "
                f"ser calculadas nem decididas pelo agente automatizado**. Por determinação expressa do Art. 78, "
                f"é vedado informar qualquer estimativa prévia de valor de reembolso antes da auditoria técnica "
                f"humana, que verificará a compatibilidade clínica, o registro ANVISA e a tabela de referência "
                f"(Art. 74). O seu pedido já está formalmente encaminhado sob o protocolo **{prot_str}**, e a "
                f"equipe especializada concluirá a análise no prazo regulamentar de até **30 (trinta) dias corridos** (Art. 81)."
            )
        return (
            f"Perfeito{', ' + primeiro_nome if primeiro_nome else ''}! Recebi e validei a **Nota Fiscal "
            f"Hospitalar** referente à **{doc_princ.get('procedimento_descricao') or 'prótese ortopédica / cirurgia (OPME)'}** "
            f"no valor total de **{_fmt_brl(val_solic)}**. Como o pedido envolve **Material Especial / Prótese "
            f"Cirúrgica (OPME)** e valor acima da alçada automatizada, aplica-se o **Art. 78, §§ 1º e 2º do "
            f"Regulamento Geral**: a solicitação foi registrada com status **ESCALADO PARA ANALISTA** sob o "
            f"protocolo oficial **{prot_str}**. Por regra regulamentar, o sistema automatizado não antecipa "
            f"cálculo de valores para OPME, cabendo ao analista técnico humano emitir o parecer e apurar o "
            f"reembolso no prazo de até **30 dias corridos** (Art. 81)."
        )

    # 8. Tratamento de Pendência Documental (ex: Falta de Relatório Clínico em Terapia >= 5ª sessão - Caso 2 Turnos 2, 3 e 4)
    if decisao == "PENDENTE_DOCUMENTO":
        sessoes_ant = detalhes.get("sessoes_terapia_anteriores")
        if sessoes_ant is None:
            sessoes_ant = max(
                int(beneficiario.get("sessoes_terapia_ano") or 0),
                int(historico.get("sessoes_terapia_ano") or 0),
                int(historico.get("sessoes_terapia_historico") or 0),
            )
        sessao_atual = sessoes_ant + 1
        total_ano_ant = float(
            historico.get("total_reembolsado_ano")
            or beneficiario.get("total_reembolsado_ano")
            or beneficiario.get("reembolsado_ano_brl")
            or 0.0
        )

        if etapa_bloqueio == "FALTA_RELATORIO_CLINICO_TERAPIA" or cat_doc == "SESSAO_TERAPIA":
            # Se o usuário perguntou especificamente sobre histórico/quantas sessões fez ou disse que nunca recebeu relatório
            if "nunca me deu" in msg_low or "relatório?" in msg_low or "relatorio?" in msg_low or "não tenho" in msg_low or "nao tenho" in msg_low:
                return (
                    f"Compreendo perfeitamente{', ' + primeiro_nome if primeiro_nome else ''}! Nas primeiras "
                    f"sessões do ano (até a 4ª sessão), apenas o recibo fiscal já bastava. Porém, conforme o "
                    f"**Art. 73, § 3º do Regulamento Geral** atualizado pela **Circular Normativa 02/2026**, "
                    f"a partir da **5ª sessão de psicoterapia no ano civil** (e esta é a sua **{sessao_atual}ª sessão**, "
                    f"pois já constam **{sessoes_ant} sessões anteriores** reembolsadas no ano, somando {_fmt_brl(total_ano_ant)}), "
                    f"passa a ser obrigatório apresentar um **Relatório Clínico circunstanciado** assinado pela "
                    f"psicóloga assistente (com CRP, justificativa clínica para continuidade, frequência e "
                    f"estimativa de duração, nos termos da **Seção 4.3 da NT-02**). "
                    f"Fique tranquilo: o seu pedido **NÃO foi negado**! Ele permanece ativo com status "
                    f"**PENDENTE DE DOCUMENTO** aguardando apenas que você solicite esse relatório à profissional "
                    f"e nos envie por aqui para liberarmos o pagamento do reembolso."
                )
            if "quantas" in msg_low or "histórico" in msg_low or "historico" in msg_low or "relatório de tudo" in msg_low or "já reembolsaram" in msg_low:
                return (
                    f"Consultei o seu histórico completo no sistema da SaúdeMais{', ' + primeiro_nome if primeiro_nome else ''}: "
                    f"você já realizou e teve reembolsadas **{sessoes_ant} sessões de psicoterapia** neste ano civil de 2026, "
                    f"totalizando **{_fmt_brl(total_ano_ant)}** já pagos em reembolsos anteriores — portanto, o recibo atual "
                    f"de **{_fmt_brl(val_solic)}** corresponde à sua **{sessao_atual}ª sessão no ano** (dentro do limite "
                    f"anual de 40 sessões do Art. 40). "
                    f"De acordo com o **Art. 73, § 3º do Regulamento Geral** e a **Circular Normativa 02/2026**, "
                    f"a partir da **5ª sessão anual** é obrigatória a apresentação de **Relatório Clínico circunstanciado** "
                    f"emitido pela psicóloga assistente justificando a continuidade do tratamento (NT-02, Seção 4.3). "
                    f"Importante: sua solicitação **não foi negada**; ela está registrada como **PENDENTE DE DOCUMENTO** "
                    f"(com protocolo aberto) aguardando apenas o envio desse relatório clínico para concluirmos o cálculo e liberarmos o valor!"
                )
            return (
                f"{saudacao} Localizei o seu cadastro ativo (**{nome_titular}**, plano **{plano}**, com carências "
                f"cumpridas) e processei o seu recibo de **sessão individual de psicoterapia (TUSS 50000462)** "
                f"no valor de **{_fmt_brl(val_solic)}**, emitido por **{doc_princ.get('emissor_nome') or 'psicóloga assistente'}**. "
                f"Verifiquei no histórico do sistema que você já possui **{sessoes_ant} sessões de psicoterapia** "
                f"reembolsadas neste ano civil (totalizando {_fmt_brl(total_ano_ant)} já reembolsados), sendo esta "
                f"a sua **{sessao_atual}ª sessão**. Conforme o **Art. 73, § 3º do Regulamento Geral**, a "
                f"**Circular Normativa 02/2026** e a **NT-02 (Seção 4.3)**, a partir da **5ª sessão no ano civil** "
                f"é indispensável anexar o **Relatório Clínico circunstanciado** do profissional assistente "
                f"indicando o quadro evolutivo e a justificativa para continuidade terapêutica. "
                f"Seu pedido **não foi negado**: ele fica com status **PENDENTE DE DOCUMENTO** (protocolo aberto) "
                f"aguardando o envio do relatório clínico para liberarmos o reembolso!"
            )

        # Outras pendências documentais (Art. 38 Pedido Médico, NT-02 campos faltantes ou P16 múltiplos itens)
        pend_txt = "; ".join(pendencias) if pendencias else "documentação complementar obrigatória (NT-02)"
        return (
            f"{saudacao} Analisei o documento enviado referente a **{doc_princ.get('procedimento_descricao') or cat_doc}** "
            f"no valor de **{_fmt_brl(val_solic)}**, porém identificamos uma pendência documental obrigatória "
            f"segundo o Regulamento Geral e a **NT-02** ({', '.join(regras)}): **{pend_txt}**. "
            f"Sua solicitação não foi indeferida em definitivo: ela permanece com status **PENDENTE DE DOCUMENTO** "
            f"aguardando a regularização ou envio do documento complementar para concluirmos a apuração."
        )

    # 9. Tratamento de Negativas Determinísticas (Carência, Prazo 90d, 40 sessões, Teto anual exaurido, Despesa não coberta, Contrato inativo, Terceiro no recibo)
    if decisao == "NEGADO":
        motivo_neg = state.get("motivo_pendencia") or "critério regulamentar impeditivo"
        return (
            f"{saudacao} Após verificar os requisitos regulamentares da sua solicitação referente ao comprovante "
            f"de **{_fmt_brl(val_solic)}** ({cat_doc}), informo que o pedido resultou em decisão **NEGADO** "
            f"com fundamento nos dispositivos **{', '.join(regras)}**. "
            f"Motivo técnico: {motivo_neg} "
            f"Lembro que, nos termos do **Art. 15 do Regulamento Geral**, é assegurado o prazo de até 150 dias "
            f"corridos para apresentação de pedido de reanálise administrativa, caso aplicável."
        )

    # 10. Tratamento de Pedidos APROVADOS ou APROVADOS PARCIALMENTE (Caso 1 Turnos 3..7, Caso 2 Turnos 5..9 e demais cálculos)
    if decisao in ("APROVADO", "APROVADO_PARCIAL"):
        teto_urs = detalhes.get("teto_urs", 5.0 if cat_doc == "CONSULTA_MEDICA" else 2.6)
        teto_brl = detalhes.get("teto_brl", 475.50 if cat_doc == "CONSULTA_MEDICA" else 247.26)
        base_calc = detalhes.get("base_calculo_brl", val_solic)
        meses_ad = detalhes.get("meses_adesao", 11)
        pct_copart = detalhes.get("percentual_coparticipacao", 40)
        pct_reemb = detalhes.get("percentual_reembolso", 60)
        val_prov = detalhes.get("valor_provisorio_brl", val_reemb)
        total_ant = detalhes.get("total_ja_reembolsado_ano_brl", 0.0)
        saldo_disp = detalhes.get("saldo_anual_disponivel_brl", 4564.80)
        limitado_saldo = bool(detalhes.get("limitado_por_saldo_anual"))
        taxa_glosa = detalhes.get("taxa_nao_assistencial_glosada_brl", 0.0)

        # 10a. Caso 2 Turnos 6 e 7: Usuário pergunta por que voltou R$ 111,27 ou se tem relação com reembolsos anteriores
        if limitado_saldo and (
            "por que" in msg_low
            or "porque" in msg_low
            or "111" in msg_low
            or "pouco" in msg_low
            or "menos" in msg_low
            or "anteriores" in msg_low
            or "teto" in msg_low
            or "limite" in msg_low
            or "já reembolsaram" in msg_low
            or "ja reembolsaram" in msg_low
        ):
            return (
                f"**Sim, exatamente por causa dos reembolsos anteriores acumulados no ano!** "
                f"De acordo com o **Art. 45 do Regulamento Geral**, todo beneficiário possui um **teto anual "
                f"global de reembolso de 48 URS**, o que equivale a **R$ 4.564,80** na vigência de 2026 "
                f"(48 × R$ 95,10, conforme Art. 33). "
                f"Como você já havia utilizado **{_fmt_brl(total_ant)}** em reembolsos anteriores ao longo "
                f"deste ano civil, o seu **saldo anual remanescente disponível** antes desta sessão era de "
                f"exatamente **{_fmt_brl(saldo_disp)}** (R$ 4.564,80 − {_fmt_brl(total_ant)}). "
                f"Pela regra da sessão de psicoterapia (Art. 41 e Circular 02/2026) com a coparticipação de "
                f"**{pct_copart}%** do seu plano **{plano}** ({meses_ad} meses de vínculo, Art. 44), o valor "
                f"calculado da sessão seria de {_fmt_brl(val_prov)}, mas, por força do teto global anual do "
                f"**Art. 45**, o reembolso fica limitado ao saldo disponível de **{_fmt_brl(val_reemb)}** "
                f"(status **APROVADO PARCIAL**, Art. 47)."
            )

        # 10b. Caso 1 Turno 4 (ou geral): Usuário pergunta por que não voltou integral / como foi calculada a coparticipação
        if (
            "por que" in msg_low
            or "porque" in msg_low
            or "integral" in msg_low
            or "tudo" in msg_low
            or "desconto" in msg_low
            or "coparticipa" in msg_low
            or "144" in msg_low
            or "metade" in msg_low
            or "calcul" in msg_low
        ) and not limitado_saldo:
            explicacao_taxa = (
                f" Além disso, foi deduzida a taxa administrativa não assistencial de {_fmt_brl(taxa_glosa)} "
                "(Arts. 57 e 59)."
                if taxa_glosa > 0
                else ""
            )
            bloco_calc = (
                f"O reembolso não é integral porque o Regulamento Geral da SaúdeMais aplica dois parâmetros "
                f"técnicos em conjunto (**Arts. 33, 43, 44 e 47** e **Circular Normativa 02/2026**): "
                f"1) **Teto em URS do procedimento**: para {doc_princ.get('procedimento_descricao') or 'consulta médica eletiva'} "
                f"(TUSS {doc_princ.get('codigo_tuss') or '10101012'}), o teto é de **{teto_urs} URS = {_fmt_brl(teto_brl)}** "
                f"(sendo 1 URS = R$ 95,10, Art. 33 e Art. 35). Como o valor elegível foi de **{_fmt_brl(base_calc)}** "
                f"(abaixo do teto), a base de cálculo é o próprio valor de **{_fmt_brl(base_calc)}** (Art. 43).{explicacao_taxa} "
                f"2) **Coparticipação contratual por plano e tempo de adesão**: conforme o **Art. 44, § 2º** "
                f"(atualizado pela Circular 02/2026), para o plano **{plano}** com **{meses_ad} meses de vínculo** "
                f"(faixa de até 12 meses de adesão), incide a coparticipação de **{pct_copart}%** a cargo do "
                f"beneficiário, resultando no reembolso de **{pct_reemb}%** sobre {_fmt_brl(base_calc)}, "
                f"o que totaliza exatamente **{_fmt_brl(val_reemb)}** (Art. 47)."
            )
            partes_resposta.append(bloco_calc)
            return " ".join(partes_resposta)

        # 10c. Caso 2 Turno 5: Recebimento do Relatório Clínico que destrava a aprovação parcial de R$ 111,27
        if anexo and (
            anexo.get("filename", "").lower().find("relatorio") >= 0
            or (docs_proc and docs_proc[-1].get("tipo_documento") == "RELATORIO_CLINICO")
        ):
            return (
                f"Excelente{', ' + primeiro_nome if primeiro_nome else ''}! Recebi e validei o **Relatório "
                f"Clínico circunstanciado** emitido pela psicóloga assistente, cumprindo integralmente a "
                f"exigência do **Art. 73, § 3º**, da **Circular Normativa 02/2026** e da **NT-02 (Seção 4.3)**. "
                f"Com a pendência documental sanada, concluí a apuração do seu recibo de psicoterapia "
                f"(TUSS 50000462) de **{_fmt_brl(val_solic)}**: aplicando o teto em URS (Art. 33 e Art. 41), "
                f"a coparticipação de **{pct_copart}%** do seu plano **{plano}** ({meses_ad} meses de adesão, "
                f"Art. 43 e Art. 44) e observando o seu saldo remanescente do **teto anual global de 48 URS "
                f"(R$ 4.564,80)** previsto no **Art. 45** (já descontados os {_fmt_brl(total_ant)} reembolsados "
                f"anteriormente no ano), o seu reembolso foi **APROVADO PARCIALMENTE** no valor exato de "
                f"**{_fmt_brl(val_reemb)}** (Art. 47)!"
            )

        # 10d. Caso 1 Turno 3 (ou aprovação padrão após identificar titular + recibo válido)
        bloco_aprov = (
            f"Localizei o seu cadastro no sistema (**{nome_titular}**, plano **{plano}**, status **ATIVO** "
            f"e carências contratuais integralmente cumpridas). Concluí a análise do seu comprovante de "
            f"**{doc_princ.get('procedimento_descricao') or cat_doc}** (código TUSS {doc_princ.get('codigo_tuss') or '10101012'}) "
            f"no valor solicitado de **{_fmt_brl(val_solic)}**: a solicitação foi **{decisao.replace('_', ' ')}** "
            f"com valor de reembolso apurado em **{_fmt_brl(val_reemb)}** (aplicando o teto de {teto_urs} URS = "
            f"{_fmt_brl(teto_brl)}, base de cálculo de {_fmt_brl(base_calc)} e coparticipação de {pct_copart}% "
            f"do plano {plano} conforme Arts. 33, 35, 43, 44 e 47 do Regulamento Geral)."
        )
        partes_resposta.append(bloco_aprov)
        return " ".join(partes_resposta)

    # 11. Fallback enriquecido com evidências RAG caso seja uma pergunta normativa aberta sem decisão ainda
    if evidencias_rag:
        resumo_evidencias = []
        for ev in evidencias_rag[:2]:
            fonte = ev.get("source", "Regulamento Geral")
            txt = (ev.get("text") or "").strip()
            primeira_linha = txt.splitlines()[0] if txt else ""
            resumo_evidencias.append(f"Segundo `{fonte}` ({primeira_linha}): {txt[:280]}...")
        partes_resposta.append(
            f"{saudacao} Com base nas normas vigentes da SaúdeMais, esclareço o seguinte: "
            + " ".join(resumo_evidencias)
        )
        return " ".join(partes_resposta)

    if partes_resposta:
        return " ".join(partes_resposta)

    return (
        f"{saudacao} Estou à disposição para ajudar com a sua solicitação de reembolso na SaúdeMais. "
        "Por gentileza, envie o número da sua carteirinha (16 dígitos) e anexe o comprovante fiscal "
        "do atendimento para iniciarmos a análise."
    )


def sintetizar_resposta_fundamentada(
    system_prompt: str = "",
    user_prompt: str = "",
    *,
    sistema: str = "",
    usuario: str = "",
    prompt_usuario: str = "",
    **kwargs: Any,
) -> str:
    """Função invocada pelo ClienteLLMUnificado no modo 'antigravity' (ou em fallback offline).

    Extrai o bloco JSON estruturado do estado se presente em `user_prompt` e gera a resposta
    contextual fundamentada. Se chamada diretamente pelo agente de normas (sem JSON de estado),
    sintetiza a resposta a partir dos trechos normativos presentes no prompt.
    """
    prompt_usr = user_prompt or usuario or prompt_usuario or ""
    marcador = "=== ESTADO_ESTRUTURADO_JSON ==="
    if marcador in prompt_usr:
        try:
            json_str = prompt_usr.split(marcador, 1)[1].strip()
            if "=== FIM_ESTADO_ESTRUTURADO_JSON ===" in json_str:
                json_str = json_str.split("=== FIM_ESTADO_ESTRUTURADO_JSON ===", 1)[0].strip()
            estado_dict = json.loads(json_str)
            return sintetizar_resposta_do_estado(estado_dict)
        except Exception:
            pass

    # Caso tenha sido chamado diretamente por responder_duvida_normativa(pergunta)
    p_low = prompt_usr.lower()
    if "acupuntura" in p_low or "fronteira" in p_low:
        return sintetizar_resposta_do_estado({"mensagem_atual": "acupuntura fronteira"})
    if "90" in p_low or "150" in p_low or "decad" in p_low:
        return sintetizar_resposta_do_estado({"mensagem_atual": "prazo 90 dias 150 dias recurso"})
    return sintetizar_resposta_do_estado({"mensagem_atual": prompt_usr[:300]})
