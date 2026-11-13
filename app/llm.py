"""Camada unificada Multi-Provedor de LLM e Embeddings.

Suporta 5 modos de operação selecionáveis via `LLM_PROVIDER` (ou detecção automática):
1. `bootcamp`    : Gateway oficial da banca (`BOOTCAMP_LLM_ENDPOINT` + `BOOTCAMP_API_KEY`).
2. `gemini`      : API oficial do Google Gemini (`GEMINI_API_KEY` / `GOOGLE_API_KEY`).
3. `grok`        : API da xAI Grok compatível com OpenAI (`GROK_API_KEY` / `XAI_API_KEY`).
4. `local`       : IA Local via Ollama, LM Studio, vLLM ou Llama.cpp (`LOCAL_LLM_URL`).
5. `antigravity` : Modo Agente Antigravity / Sintetizador Normativo Contextual (opera 100% local
                   com base nos trechos recuperados do RAG e no veredito determinístico do estado).
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("app.llm")

MODELO = "gemini-2.5-flash-lite"
MODELO_EMBEDDING = "gemini-embedding-2"
DIMENSOES = 1536

RAIZ = Path(__file__).resolve().parents[1]


class FaltaConfiguracao(RuntimeError):
    pass


@dataclass
class RespostaLLM:
    """Objeto de resposta padronizado compatível com LangChain (`resp.content`)."""

    content: str
    provider: str = "unknown"
    model: str = "unknown"


def carregar_env() -> None:
    """Lê o `.env` da raiz do repositório, sem depender de biblioteca.

    O que já está no ambiente vence o arquivo — é assim que a banca injeta as
    credenciais dela no `docker run`, sem que o seu `.env` atrapalhe.
    """
    env = RAIZ / ".env"
    if not env.exists():
        return
    for linha in env.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, valor = linha.split("=", 1)
        os.environ.setdefault(chave.strip(), valor.strip().strip('"').strip("'"))


def detectar_provedor() -> str:
    """Detecta o provedor ativo respeitando a prioridade oficial da banca."""
    carregar_env()
    explicito = os.getenv("LLM_PROVIDER", "auto").strip().lower()

    # Se as credenciais da banca estiverem presentes e o modo for auto ou bootcamp, prioriza bootcamp
    if explicito in ("", "auto"):
        if os.getenv("BOOTCAMP_LLM_ENDPOINT", "").strip() and os.getenv("BOOTCAMP_API_KEY", "").strip():
            return "bootcamp"
        if os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip():
            return "gemini"
        if os.getenv("GROK_API_KEY", "").strip() or os.getenv("XAI_API_KEY", "").strip():
            return "grok"
        if os.getenv("LOCAL_LLM_ENABLED", "").strip().lower() in ("1", "true", "yes"):
            return "local"
        if os.getenv("ANTIGRAVITY_MODE", "").strip().lower() in ("1", "true", "enabled", "yes"):
            return "antigravity"
        return "antigravity"

    return explicito


resolver_provedor = detectar_provedor



def _ambiente() -> tuple[str, str]:
    """Compatibilidade direta com o contrato original da banca."""
    carregar_env()
    endpoint = os.getenv("BOOTCAMP_LLM_ENDPOINT", "").strip().rstrip("/")
    chave = os.getenv("BOOTCAMP_API_KEY", "").strip()
    if not endpoint or not chave:
        raise FaltaConfiguracao(
            "preencha BOOTCAMP_LLM_ENDPOINT e BOOTCAMP_API_KEY no .env "
            "(ou configure GEMINI_API_KEY, GROK_API_KEY, LOCAL_LLM_URL ou LLM_PROVIDER=antigravity)"
        )
    return re.sub(r"/v1(beta)?$", "", endpoint), chave


def _extrair_mensagens(messages: Any) -> tuple[str, str]:
    """Converte lista de mensagens (LangChain ou dicts ou str) em (system_prompt, user_prompt)."""
    if isinstance(messages, str):
        return "", messages
    sistema_partes: list[str] = []
    usuario_partes: list[str] = []
    if isinstance(messages, (list, tuple)):
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", "user").lower()
                content = str(m.get("content", ""))
            else:
                tipo = type(m).__name__.lower()
                role = "system" if "system" in tipo else "user"
                content = str(getattr(m, "content", m))
            if role == "system":
                sistema_partes.append(content)
            else:
                usuario_partes.append(content)
    return "\n\n".join(sistema_partes), "\n\n".join(usuario_partes)


def _http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float = 45.0) -> dict[str, Any]:
    """Executa requisição HTTP POST JSON usando httpx (se instalado) ou urllib.request nativo."""
    try:
        import httpx

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except ImportError:
        dados = json.dumps(payload).encode("utf-8")
        req_headers = {"Content-Type": "application/json", **headers}
        req = urllib.request.Request(url, data=dados, headers=req_headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


class ClienteLLMUnificado:
    """Cliente de chat multi-provedor compatível com `.invoke(messages)` do LangChain."""

    def __init__(self, provider: str | None = None, temperature: float = 0.0, **extra: Any) -> None:
        self.provider = (provider or detectar_provedor()).lower()
        self.temperature = temperature
        self.extra = extra

    def invoke(self, messages: Any, **kwargs: Any) -> RespostaLLM:
        sistema, usuario = _extrair_mensagens(messages)
        prov = self.provider

        try:
            if prov == "bootcamp":
                return self._chamar_bootcamp(sistema, usuario)
            if prov == "gemini":
                return self._chamar_gemini(sistema, usuario)
            if prov == "grok":
                return self._chamar_grok(sistema, usuario)
            if prov == "local":
                return self._chamar_local(sistema, usuario)
            if prov == "antigravity":
                return self._chamar_antigravity(sistema, usuario)
        except Exception as e:
            logger.warning(f"Falha no provedor '{prov}' ({e}); acionando sintetizador Antigravity de contingência.")
            return self._chamar_antigravity(sistema, usuario)

        return self._chamar_antigravity(sistema, usuario)

    def _chamar_bootcamp(self, sistema: str, usuario: str) -> RespostaLLM:
        endpoint, chave = _ambiente()
        modelo = os.getenv("BOOTCAMP_MODEL", MODELO).strip() or MODELO
        # Tenta usar ChatGoogleGenerativeAI se disponível, senão REST direto
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = ChatGoogleGenerativeAI(
                model=modelo,
                google_api_key=chave,
                base_url=endpoint,
                temperature=self.temperature,
                **self.extra,
            )
            msgs = []
            if sistema:
                msgs.append(SystemMessage(content=sistema))
            msgs.append(HumanMessage(content=usuario))
            res = llm.invoke(msgs)
            return RespostaLLM(content=str(res.content), provider="bootcamp", model=modelo)
        except Exception:
            url = f"{endpoint}/v1beta/models/{modelo}:generateContent"
            corpo: dict[str, Any] = {
                "contents": [{"role": "user", "parts": [{"text": usuario}]}],
                "generationConfig": {"temperature": self.temperature},
            }
            if sistema:
                corpo["systemInstruction"] = {"parts": [{"text": sistema}]}
            dados = _http_post_json(url, corpo, {"x-goog-api-key": chave})
            texto = self._extrair_texto_gemini(dados)
            return RespostaLLM(content=texto, provider="bootcamp", model=modelo)

    def _chamar_gemini(self, sistema: str, usuario: str) -> RespostaLLM:
        chave = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        if not chave:
            raise FaltaConfiguracao("Defina GEMINI_API_KEY no arquivo .env para usar o provedor gemini.")
        modelo = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        endpoint = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").strip().rstrip("/")
        url = f"{endpoint}/v1beta/models/{modelo}:generateContent"
        corpo: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": usuario}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        if sistema:
            corpo["systemInstruction"] = {"parts": [{"text": sistema}]}
        dados = _http_post_json(url, corpo, {"x-goog-api-key": chave})
        texto = self._extrair_texto_gemini(dados)
        return RespostaLLM(content=texto, provider="gemini", model=modelo)

    def _chamar_grok(self, sistema: str, usuario: str) -> RespostaLLM:
        chave = (os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY") or "").strip()
        if not chave:
            raise FaltaConfiguracao("Defina GROK_API_KEY no arquivo .env para usar o provedor grok.")
        base_url = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1").strip().rstrip("/")
        modelo = os.getenv("GROK_MODEL", "grok-3-mini").strip()
        url = f"{base_url}/chat/completions"
        msgs = []
        if sistema:
            msgs.append({"role": "system", "content": sistema})
        msgs.append({"role": "user", "content": usuario})
        corpo = {
            "model": modelo,
            "messages": msgs,
            "temperature": self.temperature,
        }
        dados = _http_post_json(
            url,
            corpo,
            {"Authorization": f"Bearer {chave}", "User-Agent": "SaudeMais-Reembolso-Agent/2.0"},
        )
        texto = dados["choices"][0]["message"]["content"]
        return RespostaLLM(content=str(texto).strip(), provider="grok", model=modelo)

    def _chamar_local(self, sistema: str, usuario: str) -> RespostaLLM:
        base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1").strip().rstrip("/")
        modelo = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b").strip()
        chave = os.getenv("LOCAL_LLM_API_KEY", "ollama").strip()

        msgs = []
        if sistema:
            msgs.append({"role": "system", "content": sistema})
        msgs.append({"role": "user", "content": usuario})

        # Suporta tanto endpoint OpenAI-compatible (/v1/chat/completions) quanto nativo Ollama (/api/chat)
        if base_url.endswith("/api"):
            url = f"{base_url}/chat"
            corpo = {"model": modelo, "messages": msgs, "stream": False, "options": {"temperature": self.temperature}}
            dados = _http_post_json(url, corpo, {})
            texto = dados.get("message", {}).get("content", "")
        else:
            if not base_url.endswith("/v1"):
                url = f"{base_url}/v1/chat/completions"
            else:
                url = f"{base_url}/chat/completions"
            corpo = {"model": modelo, "messages": msgs, "temperature": self.temperature}
            dados = _http_post_json(url, corpo, {"Authorization": f"Bearer {chave}"})
            texto = dados["choices"][0]["message"]["content"]

        return RespostaLLM(content=str(texto).strip(), provider="local", model=modelo)

    def _chamar_antigravity(self, sistema: str, usuario: str) -> RespostaLLM:
        """Sintetizador Normativo Determinístico & Modo Agente Antigravity.

        Constrói a resposta fundamentada diretamente a partir do estado de verificação,
        dos cálculos determinísticos e dos trechos recuperados do RAG presentes no prompt.
        """
        from app.agents.supervisor.synthesizer import sintetizar_resposta_fundamentada

        texto = sintetizar_resposta_fundamentada(sistema=sistema, usuario=usuario)
        return RespostaLLM(content=texto, provider="antigravity", model="antigravity-normative-v2")

    @staticmethod
    def _extrair_texto_gemini(devolvido: dict[str, Any]) -> str:
        for candidato in devolvido.get("candidates") or []:
            partes = (candidato.get("content") or {}).get("parts") or []
            return "".join(p.get("text", "") for p in partes if isinstance(p, dict)).strip()
        return ""


# ------------------------------------------------------- LangChain / LangGraph
def criar_llm(**extra: Any) -> ClienteLLMUnificado:
    """Retorna o cliente de chat unificado compatível com todos os provedores configurados."""
    provider = extra.pop("provider", None)
    temperature = extra.pop("temperature", 0.0)
    return ClienteLLMUnificado(provider=provider, temperature=temperature, **extra)


def criar_embeddings(**extra: Any):
    """Embeddings do provedor ativo (Bootcamp, Gemini ou Local)."""
    prov = detectar_provedor()
    if prov == "bootcamp":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        endpoint, chave = _ambiente()
        return GoogleGenerativeAIEmbeddings(
            model=f"models/{MODELO_EMBEDDING}",
            google_api_key=chave,
            base_url=endpoint,
            **extra,
        )
    if prov == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        chave = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        modelo_emb = os.getenv("GEMINI_EMBEDDING_MODEL", MODELO_EMBEDDING).strip()
        if not modelo_emb.startswith("models/"):
            modelo_emb = f"models/{modelo_emb}"
        return GoogleGenerativeAIEmbeddings(model=modelo_emb, google_api_key=chave, **extra)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    endpoint, chave = _ambiente()
    return GoogleGenerativeAIEmbeddings(
        model=f"models/{MODELO_EMBEDDING}",
        google_api_key=chave,
        base_url=endpoint,
        **extra,
    )


# ------------------------------------------------------------------ LlamaIndex
def criar_llm_llamaindex(**extra: Any):
    from llama_index.llms.google_genai import GoogleGenAI

    prov = detectar_provedor()
    if prov == "gemini":
        chave = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        modelo = os.getenv("GEMINI_MODEL", MODELO).strip()
        return GoogleGenAI(model=modelo, api_key=chave, **extra)

    endpoint, chave = _ambiente()
    return GoogleGenAI(model=MODELO, api_key=chave, http_options={"base_url": endpoint}, **extra)


def criar_embeddings_llamaindex(**extra: Any):
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

    prov = detectar_provedor()
    if prov == "gemini":
        chave = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        modelo_emb = os.getenv("GEMINI_EMBEDDING_MODEL", MODELO_EMBEDDING).strip()
        return GoogleGenAIEmbedding(model_name=modelo_emb, api_key=chave, **extra)

    endpoint, chave = _ambiente()
    return GoogleGenAIEmbedding(
        model_name=MODELO_EMBEDDING,
        api_key=chave,
        http_options={"base_url": endpoint},
        **extra,
    )


def saldo() -> dict[str, Any]:
    """Quanto ainda resta do seu crédito (ou status do provedor ativo)."""
    prov = detectar_provedor()
    if prov != "bootcamp":
        return {"provider": prov, "status": "ativo", "nota": "Provedor externo/local sem limite de saldo da banca."}

    import httpx

    endpoint, chave = _ambiente()
    with httpx.Client(timeout=30) as c:
        r = c.get(f"{endpoint}/saldo", headers={"Authorization": f"Bearer {chave}"})
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    prov_ativo = detectar_provedor()
    print(f"provedor  : {prov_ativo}")
    try:
        resp = criar_llm().invoke("Responda apenas: pronto.")
        print(f"chat      : {resp.content.strip()} (model={resp.model})")
        print(f"saldo     : {saldo()}")
    except Exception as erro:
        raise SystemExit(f"Erro ao validar provedor '{prov_ativo}': {erro}")
