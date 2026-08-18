"""Schemas Pydantic para extração e categorização de documentos e anexos."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

TipoDocumento = Literal[
    "RECIBO",
    "NOTA_FISCAL",
    "RELATORIO_CLINICO",
    "FATURA_CONSUMO",
    "PEDIDO_MEDICO",
    "OUTRO",
]

CategoriaReembolso = Literal[
    "CONSULTA_MEDICA",
    "SESSAO_TERAPIA",
    "EXAME",
    "PROCEDIMENTO_CIRURGICO",
    "MATERIAL_OPME",
    "INVALIDO",
]


class DocumentoInfo(BaseModel):
    """Estrutura consolidada de informações extraídas de um anexo."""

    tipo_documento: TipoDocumento = Field(
        default="OUTRO",
        description="Tipo do documento (RECIBO, NOTA_FISCAL, RELATORIO_CLINICO, FATURA_CONSUMO, OUTRO)",
    )
    categoria_reembolso: CategoriaReembolso = Field(
        default="INVALIDO",
        description="Categoria assistencial de reembolso (CONSULTA_MEDICA, SESSAO_TERAPIA, EXAME, PROCEDIMENTO_CIRURGICO, MATERIAL_OPME, INVALIDO)",
    )
    valido_assistencial: bool = Field(
        default=True,
        description="Indica se o documento é de despesa assistencial médica/hospitalar (False para faturas de consumo, energia, etc.)",
    )
    paciente_nome: Optional[str] = Field(
        default=None, description="Nome completo do paciente atendido"
    )
    paciente_cpf: Optional[str] = Field(
        default=None, description="CPF do paciente atendido"
    )
    prestador_nome: Optional[str] = Field(
        default=None, description="Nome do profissional ou clínica prestadora"
    )
    prestador_documento: Optional[str] = Field(
        default=None, description="CNPJ ou CPF do prestador de serviço"
    )
    prestador_conselho: Optional[str] = Field(
        default=None, description="Registro no conselho de classe (CRM, CRP, CRO, etc.)"
    )
    data_atendimento: Optional[str] = Field(
        default=None, description="Data da prestação do serviço/atendimento em formato ISO (YYYY-MM-DD)"
    )
    data_emissao: Optional[str] = Field(
        default=None, description="Data de emissão do documento em formato ISO (YYYY-MM-DD)"
    )
    descricao_procedimento: Optional[str] = Field(
        default=None, description="Descrição do serviço ou procedimento realizado"
    )
    codigo_tuss: Optional[str] = Field(
        default=None, description="Código TUSS do procedimento principal (8 dígitos)"
    )
    codigos_tuss: list[str] = Field(
        default_factory=list, description="Lista de todos os códigos TUSS identificados"
    )
    valor_cobrado: Optional[float] = Field(
        default=None, description="Valor cobrado no documento fiscal (em Reais)"
    )
    tem_quitacao: bool = Field(
        default=True, description="Indica se há comprovação de quitação / recebimento"
    )
    tem_assinatura: bool = Field(
        default=True, description="Indica se há assinatura ou identificação do emissor"
    )
    pendencias_nt02: list[str] = Field(
        default_factory=list, description="Lista de pendências obrigatórias conforme a Nota Técnica 02"
    )
    motivo_invalidez: Optional[str] = Field(
        default=None, description="Motivo pelo qual o documento é inválido para reembolso, se aplicável"
    )
    texto_extraido: str = Field(
        default="", description="Texto bruto extraído do PDF ou imagem via OCR"
    )
