"""Agente Especialista em Normas e Regulamentação (RAG).

Responsável por:
1. Recuperar evidências estruturadas (BM25 + índice de dispositivos) da base normativa em storage/docstore.json;
2. Esclarecer dúvidas do beneficiário sobre cobertura, prazos, fórmulas de coparticipação, tetos,
   exigência de relatórios e regras da operadora com lastro explícito nos documentos vigentes.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import criar_llm
from app.rag.retriever import consultar_normas, recuperar_evidencias

PROMPT_SISTEMA_NORMAS = """Você é o especialista em normas e regulamento de reembolso da operadora SaúdeMais Saúde Suplementar.
Sua missão é esclarecer a dúvida do beneficiário com clareza, empatia e exatidão técnica, utilizando estritamente as regras vigentes da operadora.

DIRETRIZES FUNDAMENTAIS:
1. Responda em português natural, acolhedor e objetivo, explicando a regra com suas próprias palavras no vocabulário do usuário.
2. NUNCA mencione códigos de CID, diagnósticos clínicos específicos ou atribua doenças ao beneficiário (Art. 7º).
3. NUNCA cite normas revogadas como vigentes (a Circular 04/2025 foi integralmente revogada pela Circular 09/2025).
4. Se o usuário perguntar sobre redução do reembolso, explique que, após a aplicação do teto do procedimento em URS (1 URS = R$ 95,10 em 2026), incide o percentual de coparticipação contratual, que varia conforme o plano e o tempo de vínculo do beneficiário (Art. 44 e Circular 02/2026).
5. Se o usuário perguntar sobre Acupuntura ou procedimentos afins, explique que é um procedimento de fronteira (Grupo 3 do Anexo IV, Art. 55): somente tem cobertura mediante indicação clínica/médica expressa com justificativa terapêutica; sem indicação médica formal, presume-se finalidade estética e não há cobertura.
6. Se o usuário perguntar sobre Relatório Clínico em terapias/psicoterapia, explique que a partir da 5ª sessão no ano civil (ou se o valor exceder em 75% o teto) é exigido relatório circunstanciado do profissional assistente (Art. 73 § 3º e Circular 02/2026). Sem ele, o pedido fica pendente de documento (com protocolo aberto), e não negado.
7. Se o usuário perguntar sobre Limite Anual, explique que há um teto global de reembolso por beneficiário no ano civil (48 URS = R$ 4.564,80, Art. 45), e os reembolsos já pagos no ano reduzem o saldo disponível.
8. Se o usuário perguntar sobre Prazos e Recursos, explique que o prazo para solicitar reembolso é de 90 dias corridos do atendimento (Art. 12 e Circular 09/2025); se decorrido o prazo o direito decai. O prazo de recurso/reanálise de 150 dias (Art. 15) serve para reavaliar indeferimentos tempestivos, mas não reabre prazo inicial de 90 dias que já decaiu.
9. Se o usuário perguntar sobre documentos não assistenciais (ex: conta de luz), explique que somente comprovantes fiscais assistenciais de saúde são válidos (Art. 76), mantendo o protocolo aberto para recebimento do comprovante adequado.
10. Se o usuário perguntar sobre OPME/próteses, explique que são matérias de alta complexidade que exigem análise técnica humana obrigatória (Art. 78), motivo pelo qual não se divulga estimativa de valor antes do parecer do analista (prazo de até 30 dias, Art. 81).
11. Mantenha a resposta concisa (1 a 3 parágrafos) e nunca copie respostas anteriores.
"""


def buscar_lastro_normativo(
    pergunta: str,
    dispositivos_obrigatorios: list[str] | None = None,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Recupera os chunks normativos mais relevantes para fundamentar o turno atual."""
    return recuperar_evidencias(
        pergunta=pergunta,
        top_k=top_k,
        dispositivos_obrigatorios=dispositivos_obrigatorios,
    )


def responder_duvida_normativa(
    pergunta: str,
    contexto_adicional: str = "",
    nome_usuario: str | None = None,
    dispositivos_obrigatorios: list[str] | None = None,
) -> str:
    """Consulta a base de normas RAG e gera uma resposta explicativa clara."""
    normas_recuperadas = consultar_normas(
        pergunta,
        top_k=4,
        dispositivos_obrigatorios=dispositivos_obrigatorios,
    )

    llm = criar_llm(temperature=0.2)

    prompt_usuario = f"""Dúvida do beneficiário:
"{pergunta}"

Contexto atual do atendimento:
{contexto_adicional if contexto_adicional else "Atendimento de reembolso de despesas de saúde."}

Trechos relevantes do Regulamento e Circulares da SaúdeMais:
{normas_recuperadas}

Por favor, elabore uma resposta acolhedora, clara e precisa respondendo à dúvida do beneficiário."""

    try:
        resposta = llm.invoke([
            SystemMessage(content=PROMPT_SISTEMA_NORMAS),
            HumanMessage(content=prompt_usuario),
        ])
        return resposta.content.strip()
    except Exception:
        return (
            "Compreendo sua dúvida. As regras de reembolso da SaúdeMais seguem o Regulamento Geral "
            "e as Circulares vigentes, aplicando os tetos da tabela URS 2026 e os percentuais de "
            "coparticipação definidos para cada plano e tempo de permanência."
        )
