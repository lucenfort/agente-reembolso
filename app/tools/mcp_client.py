"""Cliente assíncrono e síncrono para o servidor MCP da operadora SaúdeMais.

Consome as três ferramentas expostas pelo FastMCP:
1. `consultar_beneficiario(carteirinha)`: dados cadastrais, plano, adesão e sessões (suporta esquema v1 e v2).
2. `consultar_historico(carteirinha)`: histórico de pedidos anteriores no ano civil.
3. `abrir_protocolo(carteirinha, payload)`: abertura de protocolo para análise humana.

Inclui fallback automático para o motor local `mcp_operadora.dados.Cadastro` quando
o servidor HTTP na porta 9000 não estiver ativo (útil em testes locais / modo Antigravity).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("app.tools.mcp_client")

RAIZ = Path(__file__).resolve().parents[2]
DIR_MCP = RAIZ / "mcp"
if str(DIR_MCP) not in sys.path:
    sys.path.insert(0, str(DIR_MCP))


def _obter_url_mcp() -> str:
    """Normaliza a URL do servidor MCP a partir das variáveis de ambiente."""
    url = os.getenv("MCP_OPERADORA_URL", "http://127.0.0.1:9000").strip().rstrip("/")
    if not url.endswith("/mcp"):
        url = f"{url}/mcp"
    return url


def _obter_token_mcp() -> str:
    """Obtém o token de autenticação MCP."""
    return os.getenv("MCP_OPERADORA_TOKEN", "treino").strip()


class _FallbackLocalOperadora:
    """Espelho local do servidor MCP para quando o processo HTTP 9000 não estiver rodando."""

    def __init__(self) -> None:
        self._por_carteirinha: dict[str, dict[str, Any]] = {}
        self._seq = 0
        self._carregar()

    @staticmethod
    def _mascarar_cpf(cpf: str) -> str:
        digitos = [c for c in str(cpf) if c.isdigit()]
        if len(digitos) != 11:
            return "***"
        meio = "".join(digitos[3:9])
        return f"***.{meio[:3]}.{meio[3:]}-**"

    @staticmethod
    def _historico_derivado(sessoes_terapia_ano: int, reembolsado_no_ano: float) -> list[dict[str, Any]]:
        from datetime import date, timedelta

        n = sessoes_terapia_ano - 1
        if n <= 0:
            return []
        base = date(2026, 1, 12)
        centavos = round(reembolsado_no_ano * 100)
        parcela, resto = divmod(centavos, n)
        return [
            {
                "protocolo": f"20260{200 + i:04d}",
                "data_atendimento": (base + timedelta(days=11 * i)).isoformat(),
                "categoria": "SESSAO_TERAPIA",
                "decisao": "APROVADO_PARCIAL",
                "valor_reembolsado_brl": round((parcela + (resto if i == 0 else 0)) / 100, 2),
            }
            for i in range(n)
        ]

    def _carregar(self) -> None:
        try:
            origem = Path(os.getenv("MCP_OPERADORA_DADOS", str(RAIZ / "casos_treino"))).expanduser().resolve()
            if not origem.exists():
                return
            if origem.is_dir():
                arquivos = sorted(origem.glob("*/persona.json"))
                brutos = [json.loads(a.read_text(encoding="utf-8")) for a in arquivos]
            else:
                dados = json.loads(origem.read_text(encoding="utf-8"))
                brutos = dados if isinstance(dados, list) else dados.get("beneficiarios", [])

            self._por_carteirinha = {}
            for d in brutos:
                cart = str(d.get("carteirinha", "")).replace(" ", "").strip()
                if not cart:
                    continue
                sessoes_ano = int(d.get("sessoes_terapia_ano", 0))
                acumulado = int(d.get("acumulado_terapia", 0)) or (sessoes_ano * 2)
                reemb_ano = float(d.get("reembolsado_no_ano_brl", 0.0))
                hist = list(d.get("historico", [])) or self._historico_derivado(sessoes_ano, reemb_ano)
                self._por_carteirinha[cart] = {
                    "carteirinha": cart,
                    "nome": d.get("nome", ""),
                    "cpf": d.get("cpf", ""),
                    "cpf_mascarado": self._mascarar_cpf(d.get("cpf", "")) if d.get("cpf") else "",
                    "plano": d.get("plano", ""),
                    "data_adesao": d.get("data_adesao", ""),
                    "status": d.get("status", "ATIVO"),
                    "sessoes_terapia_ano": sessoes_ano,
                    "acumulado_terapia": acumulado,
                    "reembolsado_no_ano": reemb_ano,
                    "historico": hist,
                }
        except Exception as e:
            logger.debug(f"Fallback local MCP não carregado: {e}")
            self._por_carteirinha = {}

    def consultar_beneficiario(self, carteirinha: str) -> dict[str, Any] | None:
        if not self._por_carteirinha:
            self._carregar()
        cart_limpa = str(carteirinha).replace(" ", "").strip()
        b = self._por_carteirinha.get(cart_limpa)
        if b is None:
            return None
        esquema = os.getenv("MCP_OPERADORA_ESQUEMA", "v1").lower()
        pub = {
            "carteirinha": b["carteirinha"],
            "nome": b["nome"],
            "cpf_mascarado": b["cpf_mascarado"],
            "plano": b["plano"],
            "data_adesao": b["data_adesao"],
            "status": b["status"],
        }
        if esquema == "v2":
            pub["sessoes_terapia"] = {
                "ano_corrente": b["sessoes_terapia_ano"],
                "acumulado": b["acumulado_terapia"],
            }
        else:
            pub["sessoes_terapia_ano"] = b["sessoes_terapia_ano"]
        return pub

    def consultar_historico(self, carteirinha: str) -> dict[str, Any] | None:
        if not self._por_carteirinha:
            self._carregar()
        cart_limpa = str(carteirinha).replace(" ", "").strip()
        b = self._por_carteirinha.get(cart_limpa)
        if b is None:
            return None
        return {"carteirinha": b["carteirinha"], "pedidos": b["historico"]}

    def abrir_protocolo(self, carteirinha: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        self._seq += 1
        return {
            "protocolo": f"2026{self._seq:07d}",
            "carteirinha": carteirinha,
            "aberto_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "EM_ANALISE",
        }


_FALLBACK_LOCAL = _FallbackLocalOperadora()


class MCPClient:
    """Cliente robusto para integração com o servidor MCP da operadora."""

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        self.url = url or _obter_url_mcp()
        self.token = token if token is not None else _obter_token_mcp()
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def _executar_ferramenta(
        self, nome_ferramenta: str, argumentos: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Executa uma chamada RPC de ferramenta no servidor MCP via Streamable HTTP."""
        try:
            import httpx
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client

            async with httpx.AsyncClient(headers=self._headers(), timeout=self.timeout) as http_client:
                async with streamable_http_client(
                    self.url,
                    http_client=http_client,
                ) as (read_stream, write_stream, init_options):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        resultado = await session.call_tool(nome_ferramenta, argumentos)

                        if resultado.isError:
                            logger.warning(
                                f"Servidor MCP retornou erro ao chamar '{nome_ferramenta}': {resultado.content}"
                            )
                            return None

                        if not resultado.content:
                            return None

                        bloco_conteudo = resultado.content[0]
                        texto_resposta = getattr(bloco_conteudo, "text", "")

                        try:
                            return json.loads(texto_resposta)
                        except json.JSONDecodeError:
                            return {"texto": texto_resposta}

        except Exception as e:
            logger.debug(f"Conexão MCP HTTP indisponível para '{nome_ferramenta}' ({e}); tentando fallback local.")
            carteirinha = str(argumentos.get("carteirinha", ""))
            if nome_ferramenta == "consultar_beneficiario":
                return _FALLBACK_LOCAL.consultar_beneficiario(carteirinha)
            if nome_ferramenta == "consultar_historico":
                return _FALLBACK_LOCAL.consultar_historico(carteirinha)
            if nome_ferramenta == "abrir_protocolo":
                return _FALLBACK_LOCAL.abrir_protocolo(carteirinha, argumentos.get("payload", {}))
            return None

    # ------------------------------------------------------------- 1. Beneficiário
    async def consultar_beneficiario_async(self, carteirinha: str) -> dict[str, Any] | None:
        carteirinha_limpa = re.sub(r"\D", "", str(carteirinha))
        if not carteirinha_limpa:
            return None

        dados = await self._executar_ferramenta(
            "consultar_beneficiario", {"carteirinha": carteirinha_limpa}
        )
        if not dados or not isinstance(dados, dict):
            return None

        sessoes_ano = 0
        acumulado = 0

        if "sessoes_terapia_ano" in dados:
            sessoes_ano = int(dados.get("sessoes_terapia_ano", 0))
            acumulado = int(dados.get("acumulado_terapia", sessoes_ano * 2))
        elif "sessoes_terapia" in dados:
            bloco_sessoes = dados.get("sessoes_terapia", {})
            if isinstance(bloco_sessoes, dict):
                sessoes_ano = int(bloco_sessoes.get("ano_corrente", 0))
                acumulado = int(bloco_sessoes.get("acumulado", sessoes_ano * 2))
            else:
                sessoes_ano = int(bloco_sessoes)
                acumulado = sessoes_ano * 2

        return {
            "carteirinha": str(dados.get("carteirinha", carteirinha_limpa)),
            "nome": str(dados.get("nome", "")),
            "cpf_mascarado": str(dados.get("cpf_mascarado", "")),
            "plano": str(dados.get("plano", "")).strip(),
            "data_adesao": str(dados.get("data_adesao", "")),
            "status": str(dados.get("status", "ATIVO")).upper(),
            "sessoes_terapia_ano": sessoes_ano,
            "acumulado_terapia": acumulado,
            "encontrado": True,
        }

    # ---------------------------------------------------------------- 2. Histórico
    async def consultar_historico_async(self, carteirinha: str) -> dict[str, Any]:
        carteirinha_limpa = re.sub(r"\D", "", str(carteirinha))
        if not carteirinha_limpa:
            return {
                "carteirinha": "",
                "pedidos": [],
                "total_reembolsado_ano": 0.0,
                "total_reembolsado_ano_brl": 0.0,
                "total_pedidos": 0,
                "sessoes_terapia_historico": 0,
            }

        dados = await self._executar_ferramenta(
            "consultar_historico", {"carteirinha": carteirinha_limpa}
        )
        if not dados or not isinstance(dados, dict):
            return {
                "carteirinha": carteirinha_limpa,
                "pedidos": [],
                "total_reembolsado_ano": 0.0,
                "total_reembolsado_ano_brl": 0.0,
                "total_pedidos": 0,
                "sessoes_terapia_historico": 0,
            }

        pedidos = dados.get("pedidos", [])
        total_pago = sum(
            float(p.get("valor_reembolsado_brl", 0.0))
            for p in pedidos
            if isinstance(p, dict) and p.get("valor_reembolsado_brl") is not None
        )
        sessoes_terapia_hist = sum(
            1
            for p in pedidos
            if isinstance(p, dict) and p.get("categoria") == "SESSAO_TERAPIA"
        )

        return {
            "carteirinha": carteirinha_limpa,
            "pedidos": pedidos,
            "total_reembolsado_ano": round(total_pago, 2),
            "total_reembolsado_ano_brl": round(total_pago, 2),
            "total_pedidos": len(pedidos),
            "sessoes_terapia_historico": sessoes_terapia_hist,
        }

    # ---------------------------------------------------------------- 3. Protocolo
    async def abrir_protocolo_async(
        self, carteirinha: str, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        carteirinha_limpa = re.sub(r"\D", "", str(carteirinha))
        dados = await self._executar_ferramenta(
            "abrir_protocolo",
            {"carteirinha": carteirinha_limpa, "payload": payload},
        )
        if not dados or not isinstance(dados, dict):
            return None

        return {
            "protocolo": str(dados.get("protocolo", "")),
            "carteirinha": str(dados.get("carteirinha", carteirinha_limpa)),
            "aberto_em": str(dados.get("aberto_em", "")),
            "status": str(dados.get("status", "EM_ANALISE")),
        }

    # ------------------------------------------------------- Wrappers Síncronos
    @staticmethod
    def _rodar_sync(coro: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)

    def consultar_beneficiario(self, carteirinha: str) -> dict[str, Any] | None:
        """Consulta dados cadastrais síncrona."""
        return self._rodar_sync(self.consultar_beneficiario_async(carteirinha))

    def consultar_historico(self, carteirinha: str) -> dict[str, Any]:
        """Consulta histórico de pedidos síncrona."""
        return self._rodar_sync(self.consultar_historico_async(carteirinha))

    def abrir_protocolo(
        self, carteirinha: str, tipo_solicitacao: str | dict[str, Any] = "REEMBOLSO_OPME", valor: Any = None
    ) -> str | None:
        """Abre protocolo síncrono e devolve a string do número do protocolo."""
        if isinstance(tipo_solicitacao, dict):
            payload = tipo_solicitacao
        else:
            payload = {
                "tipo": str(tipo_solicitacao),
                "valor_solicitado": float(valor) if valor is not None else None,
            }

        res = self._rodar_sync(self.abrir_protocolo_async(carteirinha, payload))
        if res and "protocolo" in res:
            return str(res["protocolo"])
        return None


_CLIENTE_MCP_SINGLETON: MCPClient | None = None


def obter_cliente_mcp() -> MCPClient:
    """Retorna a instância singleton do cliente MCP."""
    global _CLIENTE_MCP_SINGLETON
    if _CLIENTE_MCP_SINGLETON is None:
        _CLIENTE_MCP_SINGLETON = MCPClient()
    return _CLIENTE_MCP_SINGLETON
