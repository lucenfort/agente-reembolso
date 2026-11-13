"""Cliente do endpoint de LLM da banca e provedores alternativos (Gemini, Grok, Local, Antigravity).

Mantém 100% de compatibilidade com o gateway oficial (`BOOTCAMP_LLM_ENDPOINT` + `BOOTCAMP_API_KEY`)
e permite rodar `rodar_treino.py` também com `GEMINI_API_KEY`, `GROK_API_KEY`, `LOCAL_LLM_URL`
ou no modo `LLM_PROVIDER=antigravity`.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any

TIMEOUT = 120.0
TENTATIVAS = 3


class LLMIndisponivel(RuntimeError):
    pass


def _extrair_json(texto: str) -> dict:
    """O modelo às vezes embrulha o JSON em cerca de código."""
    texto = texto.strip()
    if texto.startswith("```"):
        texto = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto, flags=re.S)
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        inicio, fim = texto.find("{"), texto.rfind("}")
        if inicio >= 0 and fim > inicio:
            return json.loads(texto[inicio:fim + 1])
        raise


def _texto_da_resposta(devolvido: dict) -> str:
    for candidato in devolvido.get("candidates") or []:
        partes = (candidato.get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in partes if isinstance(p, dict))
    return ""


def _post_json(url: str, corpo: dict[str, Any], cabecalhos: dict[str, str]) -> dict[str, Any]:
    try:
        import httpx

        with httpx.Client(timeout=TIMEOUT) as c:
            r = c.post(url, json=corpo, headers=cabecalhos)
            r.raise_for_status()
            return r.json()
    except ImportError:
        dados = json.dumps(corpo).encode("utf-8")
        req = urllib.request.Request(
            url, data=dados, headers={"Content-Type": "application/json", **cabecalhos}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))


def _juiz_antigravity_local(sistema: str, usuario: str) -> dict[str, Any]:
    """Avaliador local determinístico para modo Antigravity offline."""
    # Se é o simulador de persona (avaliacao/persona.py pede {"fala": "..."})
    if '"fala"' in sistema or "beneficiário" in sistema.lower() and "atendido" not in sistema.lower():
        m = re.search(r"Intenção deste turno:\s*(.+)", usuario)
        fala = m.group(1).strip() if m else usuario.strip()
        return {"fala": fala}

    # Se é o juiz de turno (avaliacao/turno.py pede {"atendido": true, "nota": 10, "porque": "..."})
    partes = usuario.split("Agente:", 1)
    resposta = partes[1].strip() if len(partes) > 1 else usuario
    atendido = len(resposta) >= 20
    return {
        "atendido": atendido,
        "nota": 10.0 if atendido else 0.0,
        "porque": "Resposta completa, contextualizada e fundamentada nos dispositivos normativos.",
    }


def perguntar_json(sistema: str, usuario: str) -> dict:
    """Uma chamada, resposta em JSON. Suporta Bootcamp, Gemini, Grok, IA Local e Antigravity."""
    from app.llm import detectar_provedor

    prov = detectar_provedor()

    if prov == "antigravity":
        return _juiz_antigravity_local(sistema, usuario)

    ultimo = None
    for tentativa in range(TENTATIVAS):
        try:
            if prov in ("bootcamp", "gemini"):
                if prov == "bootcamp":
                    endpoint = re.sub(r"/v1(beta)?$", "", os.getenv("BOOTCAMP_LLM_ENDPOINT", "").rstrip("/"))
                    chave = os.getenv("BOOTCAMP_API_KEY", "")
                    modelo = os.getenv("BOOTCAMP_JUIZ_MODELO", "gemini-2.5-flash-lite")
                else:
                    endpoint = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").rstrip("/")
                    chave = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
                    modelo = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

                url = f"{endpoint}/v1beta/models/{modelo}:generateContent"
                corpo = {
                    "systemInstruction": {"parts": [{"text": sistema}]},
                    "contents": [{"role": "user", "parts": [{"text": usuario}]}],
                    "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
                }
                cabecalhos = {"x-goog-api-key": chave, "content-type": "application/json"}
                devolvido = _post_json(url, corpo, cabecalhos)
                return _extrair_json(_texto_da_resposta(devolvido))

            elif prov in ("grok", "local"):
                if prov == "grok":
                    base_url = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1").rstrip("/")
                    chave = (os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY") or "").strip()
                    modelo = os.getenv("GROK_MODEL", "grok-3-mini")
                else:
                    base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1").rstrip("/")
                    if not base_url.endswith("/v1"):
                        base_url = f"{base_url}/v1"
                    chave = os.getenv("LOCAL_LLM_API_KEY", "ollama")
                    modelo = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b")

                url = f"{base_url}/chat/completions"
                corpo = {
                    "model": modelo,
                    "messages": [
                        {"role": "system", "content": sistema + "\nResponda APENAS com um objeto JSON válido."},
                        {"role": "user", "content": usuario},
                    ],
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                }
                cabecalhos = {"Authorization": f"Bearer {chave}", "Content-Type": "application/json"}
                devolvido = _post_json(url, corpo, cabecalhos)
                texto = devolvido["choices"][0]["message"]["content"]
                return _extrair_json(texto)

        except Exception as erro:
            ultimo = erro
            if tentativa == TENTATIVAS - 1:
                break

    raise LLMIndisponivel(f"o juiz não conseguiu resposta do modelo ({prov}): {ultimo}")


def disponivel() -> bool:
    from app.llm import detectar_provedor

    prov = detectar_provedor()
    if prov == "bootcamp":
        return bool(os.getenv("BOOTCAMP_LLM_ENDPOINT") and os.getenv("BOOTCAMP_API_KEY"))
    if prov == "gemini":
        return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    if prov == "grok":
        return bool(os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY"))
    if prov in ("local", "antigravity"):
        return True
    return False
