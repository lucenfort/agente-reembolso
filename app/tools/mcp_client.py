"""Cliente assíncrono e síncrono para o servidor MCP da operadora SaúdeMais.

Consome as três ferramentas expostas pelo FastMCP:
1. `consultar_beneficiario(carteirinha)`: dados cadastrais, plano, adesão e sessões.
2. `consultar_historico(carteirinha)`: histórico de pedidos anteriores no ano civil.
3. `abrir_protocolo(carteirinha, payload)`: abertura de protocolo para análise humana.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

logger = logging.getLogger("app.tools.mcp_client")


def _obter_url_mcp() -> str:
    """Normaliza a URL do servidor MCP a partir das variáveis de ambiente."""
    url = os.getenv("MCP_OPERADORA_URL", "http://127.0.0.1:9000").strip().rstrip("/")
    if not url.endswith("/mcp"):
        url = f"{url}/mcp"
    return url


def _obter_token_mcp() -> str:
    """Obtém o token de autenticação MCP."""
    return os.getenv("MCP_OPERADORA_TOKEN", "").strip()


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
        headers = self._headers()
        try:
            async with httpx.AsyncClient(headers=headers, timeout=self.timeout) as http_client:
                async with streamable_http_client(self.url, http_client=http_client) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        resultado = await session.call_tool(nome_ferramenta, argumentos)

                        if resultado.isError:
                            msg_erro = ""
                            if resultado.content:
                                msg_erro = " ".join(
                                    getattr(c, "text", str(c)) for c in resultado.content
                                )
                            logger.warning(
                                f"Erro retornado pela ferramenta MCP '{nome_ferramenta}': {msg_erro}"
                            )
                            return None

                        if not resultado.content:
                            return None

                        # O conteúdo retornado pelo FastMCP vem como JSON no primeiro bloco de texto
                        primeiro_bloco = resultado.content[0]
                        texto_json = getattr(primeiro_bloco, "text", "")
                        if not texto_json:
                            return None

                        return json.loads(texto_json)

        except httpx.HTTPStatusError as e:
            logger.error(f"Erro HTTP {e.response.status_code} ao conectar no MCP ({self.url}): {e}")
            return None
        except httpx.ConnectError:
            logger.warning(f"Não foi possível conectar ao servidor MCP em {self.url}")
            return None
        except asyncio.TimeoutError:
            logger.warning(f"Timeout ({self.timeout}s) ao chamar ferramenta '{nome_ferramenta}' no MCP")
            return None
        except Exception as e:
            logger.error(f"Falha inesperada ao executar ferramenta MCP '{nome_ferramenta}': {e}")
            return None

    # ------------------------------------------------------------- 1. Beneficiário
    async def consultar_beneficiario(self, carteirinha: str) -> dict[str, Any] | None:
        """Consulta dados cadastrais do beneficiário normalizando os esquemas v1 e v2.

        Retorna dicionário unificado:
            carteirinha: str
            nome: str
            cpf_mascarado: str
            plano: str ("Pleno" ou "Essencial")
            data_adesao: str ("YYYY-MM-DD")
            status: str ("ATIVO", "SUSPENSO", etc.)
            sessoes_terapia_ano: int (sessões realizadas no ano civil corrente)
            acumulado_terapia: int (total histórico acumulado)
            encontrado: bool
        """
        carteirinha_limpa = re.sub(r"\D", "", str(carteirinha))
        if not carteirinha_limpa:
            return None

        dados = await self._executar_ferramenta(
            "consultar_beneficiario", {"carteirinha": carteirinha_limpa}
        )
        if not dados or not isinstance(dados, dict):
            return None

        # Trata polimorfismo de esquema (v1 vs v2)
        sessoes_ano = 0
        acumulado = 0

        if "sessoes_terapia_ano" in dados:
            # Esquema v1
            sessoes_ano = int(dados.get("sessoes_terapia_ano", 0))
            acumulado = int(dados.get("acumulado_terapia", sessoes_ano * 2))
        elif "sessoes_terapia" in dados:
            # Esquema v2
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
    async def consultar_historico(self, carteirinha: str) -> dict[str, Any]:
        """Consulta os pedidos de reembolso anteriores do beneficiário no ano.

        Retorna dicionário:
            carteirinha: str
            pedidos: list[dict]
            total_reembolsado_ano_brl: float (soma de reembolsos aprovados no ano)
            total_pedidos: int
        """
        carteirinha_limpa = re.sub(r"\D", "", str(carteirinha))
        if not carteirinha_limpa:
            return {"carteirinha": "", "pedidos": [], "total_reembolsado_ano_brl": 0.0, "total_pedidos": 0}

        dados = await self._executar_ferramenta(
            "consultar_historico", {"carteirinha": carteirinha_limpa}
        )
        if not dados or not isinstance(dados, dict):
            return {
                "carteirinha": carteirinha_limpa,
                "pedidos": [],
                "total_reembolsado_ano_brl": 0.0,
                "total_pedidos": 0,
            }

        pedidos = dados.get("pedidos", [])
        total_pago = sum(
            float(p.get("valor_reembolsado_brl", 0.0))
            for p in pedidos
            if isinstance(p, dict) and p.get("valor_reembolsado_brl") is not None
        )

        return {
            "carteirinha": carteirinha_limpa,
            "pedidos": pedidos,
            "total_reembolsado_ano_brl": round(total_pago, 2),
            "total_pedidos": len(pedidos),
        }

    # ---------------------------------------------------------------- 3. Protocolo
    async def abrir_protocolo(
        self, carteirinha: str, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Abre protocolo para análise humana e devolve o registro oficial."""
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
    def consultar_beneficiario_sync(self, carteirinha: str) -> dict[str, Any] | None:
        return asyncio.run(self.consultar_beneficiario(carteirinha))

    def consultar_historico_sync(self, carteirinha: str) -> dict[str, Any]:
        return asyncio.run(self.consultar_historico(carteirinha))

    def abrir_protocolo_sync(
        self, carteirinha: str, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        return asyncio.run(self.abrir_protocolo(carteirinha, payload))
