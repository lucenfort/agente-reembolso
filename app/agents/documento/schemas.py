"""Schemas Pydantic para extração e categorização de documentos e anexos."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, model_validator

TipoDocumento = str
CategoriaReembolso = str


class DocumentoInfo(BaseModel):
    """Estrutura consolidada de informações extraídas de um anexo."""

    tipo_documento: str = Field(
        default="OUTRO",
        description="Tipo do documento (RECIBO, NOTA_FISCAL, RELATORIO_CLINICO, FATURA_CONSUMO, PEDIDO_MEDICO, OUTRO)",
    )
    categoria_reembolso: str = Field(
        default="INVALIDO",
        description="Categoria assistencial de reembolso (CONSULTA_MEDICA, SESSAO_TERAPIA, EXAME, EXAME_DIAGNOSTICO, RELATORIO_CLINICO, PROCEDIMENTO_CIRURGICO, MATERIAL_OPME, DESPESA_NAO_COBERTA, INVALIDO)",
    )
    valido_assistencial: bool = Field(
        default=True,
        description="Indica se o documento é de despesa assistencial médica/hospitalar",
    )
    paciente_nome: Optional[str] = Field(default=None, description="Nome completo do paciente atendido")
    paciente_cpf: Optional[str] = Field(default=None, description="CPF do paciente atendido")
    prestador_nome: Optional[str] = Field(default=None, description="Nome do profissional ou clínica prestadora")
    emissor_nome: Optional[str] = Field(default=None, description="Alias para prestador_nome")
    prestador_documento: Optional[str] = Field(default=None, description="CNPJ ou CPF do prestador de serviço")
    emissor_documento: Optional[str] = Field(default=None, description="Alias para prestador_documento")
    prestador_conselho: Optional[str] = Field(default=None, description="Registro no conselho de classe (CRM, CRP, etc.)")
    registro_profissional: Optional[str] = Field(default=None, description="Alias para prestador_conselho")
    data_atendimento: Optional[str] = Field(default=None, description="Data do atendimento (YYYY-MM-DD)")
    data_emissao: Optional[str] = Field(default=None, description="Data de emissão (YYYY-MM-DD)")
    descricao_procedimento: Optional[str] = Field(default=None, description="Descrição do procedimento realizado")
    procedimento_descricao: Optional[str] = Field(default=None, description="Alias para descricao_procedimento")
    codigo_tuss: Optional[str] = Field(default=None, description="Código TUSS do procedimento principal (8 dígitos)")
    codigos_tuss: list[str] = Field(default_factory=list, description="Lista de todos os códigos TUSS identificados")
    valor_cobrado: Optional[float] = Field(default=None, description="Valor cobrado no documento fiscal (em Reais)")
    valor_itens_excluidos: float = Field(default=0.0, description="Valor de taxas não assistenciais discriminadas")
    taxa_nao_assistencial: float = Field(default=0.0, description="Alias para valor_itens_excluidos")
    eh_urgencia: bool = Field(default=False, description="Indica se o atendimento foi de urgência/emergência (Art. 26)")
    carater_urgencia: bool = Field(default=False, description="Alias para eh_urgencia")
    eh_fronteira_grupo3: bool = Field(default=False, description="Indica se pertence ao Grupo 3 do Anexo IV")
    procedimento_fronteira: bool = Field(default=False, description="Alias para eh_fronteira_grupo3")
    procedimento_excluido: bool = Field(default=False, description="Indica se o procedimento é excluído da cobertura")
    exige_pedido_medico: bool = Field(default=False, description="Indica se o exame exige pedido médico (Art. 38)")
    tem_indicacao_clinica: bool = Field(default=False, description="Indica se há justificativa/indicação clínica expressa")
    multiplos_itens_sem_discriminacao: bool = Field(
        default=False, description="Indica se há múltiplos itens/sessões sem discriminação individual (NT-02 P16)"
    )
    tem_quitacao: bool = Field(default=True, description="Indica se há comprovação de quitação")
    tem_assinatura: bool = Field(default=True, description="Indica se há assinatura do emissor")
    pendencias_nt02: list[str] = Field(default_factory=list, description="Pendências formais conforme NT-02")
    pendencias_relatorio: list[str] = Field(default_factory=list, description="Pendências de relatório clínico (NT-02)")
    motivo_invalidez: Optional[str] = Field(default=None, description="Motivo de invalidez ou não cobertura")
    texto_extraido: str = Field(default="", description="Texto bruto extraído do PDF ou imagem")

    @model_validator(mode="after")
    def _sincronizar_aliases(self) -> "DocumentoInfo":
        # Sincroniza prestador_nome <-> emissor_nome
        if self.prestador_nome and not self.emissor_nome:
            self.emissor_nome = self.prestador_nome
        elif self.emissor_nome and not self.prestador_nome:
            self.prestador_nome = self.emissor_nome

        # Sincroniza prestador_documento <-> emissor_documento
        if self.prestador_documento and not self.emissor_documento:
            self.emissor_documento = self.prestador_documento
        elif self.emissor_documento and not self.prestador_documento:
            self.prestador_documento = self.emissor_documento

        # Sincroniza prestador_conselho <-> registro_profissional
        if self.prestador_conselho and not self.registro_profissional:
            self.registro_profissional = self.prestador_conselho
        elif self.registro_profissional and not self.prestador_conselho:
            self.prestador_conselho = self.registro_profissional

        # Sincroniza descricao_procedimento <-> procedimento_descricao
        if self.descricao_procedimento and not self.procedimento_descricao:
            self.procedimento_descricao = self.descricao_procedimento
        elif self.procedimento_descricao and not self.descricao_procedimento:
            self.descricao_procedimento = self.procedimento_descricao

        # Sincroniza eh_urgencia <-> carater_urgencia
        if self.eh_urgencia or self.carater_urgencia:
            self.eh_urgencia = True
            self.carater_urgencia = True

        # Sincroniza eh_fronteira_grupo3 <-> procedimento_fronteira
        if self.eh_fronteira_grupo3 or self.procedimento_fronteira:
            self.eh_fronteira_grupo3 = True
            self.procedimento_fronteira = True

        # Sincroniza valor_itens_excluidos <-> taxa_nao_assistencial
        maior_taxa = max(self.valor_itens_excluidos or 0.0, self.taxa_nao_assistencial or 0.0)
        self.valor_itens_excluidos = maior_taxa
        self.taxa_nao_assistencial = maior_taxa

        # Marca procedimento_excluido se categoria_reembolso == DESPESA_NAO_COBERTA
        if self.categoria_reembolso == "DESPESA_NAO_COBERTA":
            self.procedimento_excluido = True

        # Marca exige_pedido_medico para exames de alta complexidade (41001010, 41101014)
        if self.codigo_tuss in ("41001010", "41101014"):
            self.exige_pedido_medico = True

        return self
