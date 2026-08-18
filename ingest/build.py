"""Constrói o índice a partir de kb/ e grava em storage/.

    python -m ingest.build

Roda FORA do container, na sua máquina. Depois você commita `storage/` e o
Dockerfile só copia — o build da imagem precisa ser offline e determinístico, e
o `/health` tem 60 segundos para responder.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import docx
import pymupdf
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.schema import TextNode

from app.llm import criar_embeddings_llamaindex

RAIZ = Path(__file__).resolve().parents[1]
DIR_KB = RAIZ / "kb"
DIR_STORAGE = RAIZ / "storage"


def extrair_tabela_urs(caminho_pdf: Path) -> tuple[dict[str, dict[str, Any]], list[TextNode]]:
    """Extrai tabela estruturada de procedimentos TUSS e gera dicionário e nós de busca."""
    doc = pymupdf.open(caminho_pdf)
    tuss_dict: dict[str, dict[str, Any]] = {}
    nodes: list[TextNode] = []

    # Extrai introdução / URS 2026 (página 1)
    texto_intro = doc[0].get_text()
    nodes.append(
        TextNode(
            text=f"TABELA URS 2026 - DISPOSIÇÕES GERAIS E VALOR DA URS\n\n{texto_intro.strip()}",
            metadata={
                "doc_name": "tabela_urs_2026",
                "dispositivo": "TABELA-URS",
                "tipo": "tabela_precos",
                "pagina": 1,
            },
            id_=f"tabela_urs_intro_p1",
        )
    )

    # Extrai procedimentos por tabela
    linhas_por_categoria: dict[str, list[str]] = {}

    for p_idx, page in enumerate(doc):
        tabs = page.find_tables()
        if not tabs or len(tabs.tables) == 0:
            continue
        for tab in tabs:
            for row in tab.extract()[1:]:
                if not row or len(row) < 4:
                    continue
                code = str(row[0]).strip().replace("\n", "")
                if re.match(r"^\d{8}$", code):
                    desc = str(row[1]).strip().replace("\n", " ")
                    urs = str(row[2]).strip().replace("\n", "").replace(",", ".")
                    reais = str(row[3]).strip().replace("\n", "")
                    cat = str(row[4]).strip().replace("\n", " ") if len(row) > 4 else "Geral"
                    obs = str(row[5]).strip().replace("\n", " ") if len(row) > 5 else ""

                    try:
                        urs_val = float(urs)
                    except ValueError:
                        urs_val = 0.0

                    tuss_dict[code] = {
                        "tuss": code,
                        "descricao": desc,
                        "urs": urs_val,
                        "reais": reais,
                        "categoria": cat,
                        "observacao": obs,
                        "pagina": p_idx + 1,
                    }

                    linha = f"- TUSS {code}: {desc} | Categoria: {cat} | Teto URS: {urs} ({reais}) {f'| Obs: {obs}' if obs else ''}"
                    linhas_por_categoria.setdefault(cat, []).append(linha)

    # Cria nós textuais agrupados por categoria para permitir RAG semântico
    for cat, linhas in linhas_por_categoria.items():
        chunk_size = 25
        for i in range(0, len(linhas), chunk_size):
            bloco = "\n".join(linhas[i : i + chunk_size])
            nodes.append(
                TextNode(
                    text=f"TABELA URS 2026 — PROCEDIMENTOS E TETOS ({cat.upper()} - Parte {i//chunk_size + 1}):\n\n{bloco}",
                    metadata={
                        "doc_name": "tabela_urs_2026",
                        "dispositivo": "TABELA-URS",
                        "categoria": cat,
                        "tipo": "tabela_precos",
                    },
                    id_=f"tabela_urs_{cat.lower().replace(' ', '_')}_{i//chunk_size + 1}",
                )
            )

    return tuss_dict, nodes


def extrair_regulamento_geral(caminho_pdf: Path) -> list[TextNode]:
    """Extrai os 87 artigos do Regulamento Geral como nós semânticos individuais."""
    doc = pymupdf.open(caminho_pdf)
    texto_completo = ""
    for page in doc:
        texto_completo += page.get_text() + "\n"

    # Quebra por artigo: 'Art. 1º', 'Art. 2º', etc.
    segmentos = re.split(r"(?=\n\s*Art\.\s*\d+[º°]?)", texto_completo)
    nodes: list[TextNode] = []

    # Preâmbulo / Título preliminar
    if segmentos and not re.match(r"^\s*Art\.\s*\d+", segmentos[0]):
        preambulo = segmentos[0].strip()
        if preambulo:
            nodes.append(
                TextNode(
                    text=f"REGULAMENTO GERAL DE REEMBOLSO — DISPOSIÇÕES INICIAIS:\n\n{preambulo}",
                    metadata={
                        "doc_name": "regulamento_geral",
                        "dispositivo": "REGULAMENTO-GERAL",
                        "tipo": "regulamento",
                    },
                    id_="reg_geral_preambulo",
                )
            )
        artigos = segmentos[1:]
    else:
        artigos = segmentos

    for art_texto in artigos:
        match_art = re.search(r"Art\.\s*(\d+)", art_texto)
        if not match_art:
            continue
        num_art = int(match_art.group(1))
        art_clean = art_texto.strip()

        nodes.append(
            TextNode(
                text=f"REGULAMENTO GERAL DE REEMBOLSO — ARTIGO {num_art}:\n\n{art_clean}",
                metadata={
                    "doc_name": "regulamento_geral",
                    "dispositivo": f"ART-{num_art}",
                    "artigo": num_art,
                    "tipo": "regulamento",
                },
                id_=f"reg_geral_art_{num_art:02d}",
            )
        )

    return nodes


def extrair_circulares(caminho_pdf: Path, doc_name: str, dispositivo: str) -> list[TextNode]:
    """Extrai circulares normativas preservando ementa e artigos."""
    doc = pymupdf.open(caminho_pdf)
    texto_completo = ""
    for page in doc:
        texto_completo += page.get_text() + "\n"

    nodes: list[TextNode] = []
    # Cria nó com o texto integral da circular (são curtas, 2 a 3 páginas)
    nodes.append(
        TextNode(
            text=f"CIRCULAR NORMATIVA {dispositivo.replace('CIRC-', '').replace('-', '/')}:\n\n{texto_completo.strip()}",
            metadata={
                "doc_name": doc_name,
                "dispositivo": dispositivo,
                "tipo": "circular",
            },
            id_=f"circular_{dispositivo.lower().replace('-', '_')}_completa",
        )
    )

    # Também divide por artigos internos da circular
    sub_artigos = re.split(r"(?=\n\s*Art\.\s*\d+[º°]?)", texto_completo)
    for idx, sub in enumerate(sub_artigos[1:], start=1):
        match_sub = re.search(r"Art\.\s*(\d+)", sub)
        sub_num = int(match_sub.group(1)) if match_sub else idx
        nodes.append(
            TextNode(
                text=f"CIRCULAR NORMATIVA {dispositivo.replace('CIRC-', '').replace('-', '/')} — ARTIGO {sub_num}:\n\n{sub.strip()}",
                metadata={
                    "doc_name": doc_name,
                    "dispositivo": dispositivo,
                    "tipo": "circular",
                    "artigo_circular": sub_num,
                },
                id_=f"circular_{dispositivo.lower().replace('-', '_')}_art_{sub_num}",
            )
        )

    return nodes


def extrair_anexo_iv(caminho_pdf: Path) -> list[TextNode]:
    """Extrai os 10 grupos de exclusão de cobertura do Anexo IV."""
    doc = pymupdf.open(caminho_pdf)
    texto_completo = ""
    for page in doc:
        texto_completo += page.get_text() + "\n"

    nodes: list[TextNode] = []
    grupos = re.split(r"(?=\n\s*Grupo\s+\d+|ANEXO\s+IV\s+—\s+EXCLUSÕES)", texto_completo)

    for idx, grupo in enumerate(grupos):
        g_clean = grupo.strip()
        if not g_clean:
            continue
        match_g = re.search(r"Grupo\s+(\d+)", g_clean, re.IGNORECASE)
        g_num = int(match_g.group(1)) if match_g else idx

        nodes.append(
            TextNode(
                text=f"ANEXO IV — EXCLUSÕES DE COBERTURA (GRUPO {g_num}):\n\n{g_clean}",
                metadata={
                    "doc_name": "anexo_iv_exclusoes",
                    "dispositivo": "ANEXO-IV",
                    "grupo": g_num,
                    "tipo": "exclusao",
                },
                id_=f"anexo_iv_grupo_{g_num}",
            )
        )

    return nodes


def extrair_nota_tecnica_02(caminho_pdf: Path) -> list[TextNode]:
    """Extrai os requisitos documentais e campos da NT-02."""
    doc = pymupdf.open(caminho_pdf)
    texto_completo = ""
    for page in doc:
        texto_completo += page.get_text() + "\n"

    nodes: list[TextNode] = []
    secoes = re.split(r"(?=\n\s*\d+\.\s+[A-ZÁÉÍÓÚ])", texto_completo)

    for idx, sec in enumerate(secoes):
        s_clean = sec.strip()
        if not s_clean:
            continue
        nodes.append(
            TextNode(
                text=f"NOTA TÉCNICA 02 — REQUISITOS DOCUMENTAIS (SEÇÃO {idx + 1}):\n\n{s_clean}",
                metadata={
                    "doc_name": "nota_tecnica_02_documentos",
                    "dispositivo": "NT-02",
                    "tipo": "requisitos_documentais",
                },
                id_=f"nt_02_secao_{idx + 1}",
            )
        )

    return nodes


def extrair_manual_rede(caminho_pdf: Path) -> list[TextNode]:
    """Extrai o Manual da Rede Credenciada e Livre Escolha."""
    doc = pymupdf.open(caminho_pdf)
    texto_completo = ""
    for page in doc:
        texto_completo += page.get_text() + "\n"

    nodes: list[TextNode] = []
    secoes = re.split(r"(?=\n\s*\d+\.\s+[A-ZÁÉÍÓÚ])", texto_completo)

    for idx, sec in enumerate(secoes):
        s_clean = sec.strip()
        if not s_clean:
            continue
        nodes.append(
            TextNode(
                text=f"MANUAL DA REDE CREDENCIADA E REEMBOLSO (SEÇÃO {idx + 1}):\n\n{s_clean}",
                metadata={
                    "doc_name": "manual_rede_credenciada",
                    "dispositivo": "MANUAL-REDE",
                    "tipo": "manual",
                },
                id_=f"manual_rede_secao_{idx + 1}",
            )
        )

    return nodes


def extrair_faq_interno(caminho_docx: Path) -> list[TextNode]:
    """Extrai perguntas e respostas do FAQ Interno."""
    doc = docx.Document(caminho_docx)
    paragrafos = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    nodes: list[TextNode] = []
    chunk_tamanho = 4
    for i in range(0, len(paragrafos), chunk_tamanho):
        bloco = "\n\n".join(paragrafos[i : i + chunk_tamanho])
        nodes.append(
            TextNode(
                text=f"FAQ INTERNO DE ATENDIMENTO (PARTE {i//chunk_tamanho + 1}):\n\n{bloco}",
                metadata={
                    "doc_name": "faq_interno",
                    "dispositivo": "FAQ-INTERNO",
                    "tipo": "faq",
                },
                id_=f"faq_interno_bloco_{i//chunk_tamanho + 1}",
            )
        )

    return nodes


def construir_indice() -> tuple[VectorStoreIndex, dict[str, Any]]:
    """Carrega todos os 10 documentos de kb/, chunka, gera embeddings e persiste em storage/."""
    print("=" * 70)
    print("🚀 INICIANDO PIPELINE DE INGESTÃO NORMATIVA E INDEXAÇÃO HÍBRIDA")
    print("=" * 70)

    DIR_STORAGE.mkdir(parents=True, exist_ok=True)
    all_nodes: list[TextNode] = []

    # 1. Tabela URS 2026 + TUSS
    print("📊 1/10 Processando Tabela URS 2026 (PDF)...")
    tuss_dict, tuss_nodes = extrair_tabela_urs(DIR_KB / "tabela_urs_2026.pdf")
    all_nodes.extend(tuss_nodes)
    print(f"   -> {len(tuss_dict)} procedimentos TUSS extraídos.")
    print(f"   -> {len(tuss_nodes)} nós de texto da tabela gerados.")

    # Salva o dicionário estruturado TUSS em JSON
    caminho_tuss_json = DIR_STORAGE / "tuss_tabela.json"
    with open(caminho_tuss_json, "w", encoding="utf-8") as f:
        json.dump(tuss_dict, f, ensure_ascii=False, indent=2)
    print(f"   -> Tabela TUSS persistida em {caminho_tuss_json}")

    # 2. Regulamento Geral
    print("📜 2/10 Processando Regulamento Geral (87 Artigos)...")
    reg_nodes = extrair_regulamento_geral(DIR_KB / "regulamento_geral.pdf")
    all_nodes.extend(reg_nodes)
    print(f"   -> {len(reg_nodes)} nós de artigos gerados.")

    # 3. Circular 02/2026 (Precedência Máxima)
    print("📜 3/10 Processando Circular 02/2026...")
    c02_nodes = extrair_circulares(DIR_KB / "circular_02_2026.pdf", "circular_02_2026", "CIRC-02-2026")
    all_nodes.extend(c02_nodes)
    print(f"   -> {len(c02_nodes)} nós gerados.")

    # 4. Circular 04/2025
    print("📜 4/10 Processando Circular 04/2025...")
    c04_nodes = extrair_circulares(DIR_KB / "circular_04_2025.pdf", "circular_04_2025", "CIRC-04-2025")
    all_nodes.extend(c04_nodes)
    print(f"   -> {len(c04_nodes)} nós gerados.")

    # 5. Circular 09/2025
    print("📜 5/10 Processando Circular 09/2025...")
    c09_nodes = extrair_circulares(DIR_KB / "circular_09_2025.pdf", "circular_09_2025", "CIRC-09-2025")
    all_nodes.extend(c09_nodes)
    print(f"   -> {len(c09_nodes)} nós gerados.")

    # 6. Circular 11/2026
    print("📜 6/10 Processando Circular 11/2026...")
    c11_nodes = extrair_circulares(DIR_KB / "circular_11_2026.pdf", "circular_11_2026", "CIRC-11-2026")
    all_nodes.extend(c11_nodes)
    print(f"   -> {len(c11_nodes)} nós gerados.")

    # 7. Anexo IV Exclusões
    print("🚫 7/10 Processando Anexo IV — Exclusões de Cobertura...")
    anexo_nodes = extrair_anexo_iv(DIR_KB / "anexo_iv_exclusoes.pdf")
    all_nodes.extend(anexo_nodes)
    print(f"   -> {len(anexo_nodes)} nós gerados.")

    # 8. Nota Técnica 02
    print("📋 8/10 Processando Nota Técnica 02 — Requisitos Documentais...")
    nt_nodes = extrair_nota_tecnica_02(DIR_KB / "nota_tecnica_02_documentos.pdf")
    all_nodes.extend(nt_nodes)
    print(f"   -> {len(nt_nodes)} nós gerados.")

    # 9. Manual da Rede Credenciada
    print("🏥 9/10 Processando Manual da Rede Credenciada...")
    manual_nodes = extrair_manual_rede(DIR_KB / "manual_rede_credenciada.pdf")
    all_nodes.extend(manual_nodes)
    print(f"   -> {len(manual_nodes)} nós gerados.")

    # 10. FAQ Interno
    print("❓ 10/10 Processando FAQ Interno (DOCX)...")
    faq_nodes = extrair_faq_interno(DIR_KB / "faq_interno.docx")
    all_nodes.extend(faq_nodes)
    print(f"   -> {len(faq_nodes)} nós gerados.")

    print(f"\n📦 TOTAL DE NÓS SEMÂNTICOS: {len(all_nodes)}")

    # Configura modelo de embeddings do Gemini via LlamaIndex
    print("\n🧠 Configurando modelo de embeddings (Gemini 1536d)...")
    embed_model = criar_embeddings_llamaindex()
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024

    # Constrói o VectorStoreIndex e persiste em disco
    print("⚙️  Gerando embeddings vetoriais e construindo VectorStoreIndex...")
    inicio_emb = time.time()
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex(nodes=all_nodes, storage_context=storage_context)
    tempo_emb = time.time() - inicio_emb
    print(f"✅ Indexação concluída em {tempo_emb:.2f} segundos.")

    print(f"💾 Persistindo StorageContext em {DIR_STORAGE}...")
    storage_context.persist(persist_dir=str(DIR_STORAGE))
    print("✅ StorageContext persistido com sucesso!")

    return index, tuss_dict


def validar_indice(tuss_dict: dict[str, Any]) -> None:
    """Executa testes de validação semântica e BM25 contra o índice recém-construído."""
    print("\n" + "=" * 70)
    print("🧪 EXECUTANDO TESTES DE HOMOLOGAÇÃO DO ÍNDICE (QUALITY GATES)")
    print("=" * 70)

    # 1. Carrega o índice do disco
    storage_context = StorageContext.from_defaults(persist_dir=str(DIR_STORAGE))
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever(similarity_top_k=3)

    # Teste 1: Consulta médica dermatologia
    q1 = "qual é o teto de reembolso para consulta médica em dermatologia?"
    t0 = time.time()
    res1 = retriever.retrieve(q1)
    lat1 = (time.time() - t0) * 1000
    print(f"\n[Teste 1] '{q1}' (Latência: {lat1:.1f}ms):")
    for r in res1:
        print(f"   • [{r.node.metadata.get('dispositivo')}] {r.node.text[:120]}... (Score: {r.score:.3f})")

    # Teste 2: Psicoterapia e Circular 02/2026
    q2 = "a partir de qual sessão de psicoterapia é obrigatório apresentar relatório clínico circunstanciado?"
    t0 = time.time()
    res2 = retriever.retrieve(q2)
    lat2 = (time.time() - t0) * 1000
    print(f"\n[Teste 2] '{q2}' (Latência: {lat2:.1f}ms):")
    for r in res2:
        print(f"   • [{r.node.metadata.get('dispositivo')}] {r.node.text[:120]}... (Score: {r.score:.3f})")

    # Teste 3: Alçada e OPME (Art. 78)
    q3 = "como funciona a alçada de aprovação e o reembolso de prótese ortopédica OPME?"
    t0 = time.time()
    res3 = retriever.retrieve(q3)
    lat3 = (time.time() - t0) * 1000
    print(f"\n[Teste 3] '{q3}' (Latência: {lat3:.1f}ms):")
    for r in res3:
        print(f"   • [{r.node.metadata.get('dispositivo')}] {r.node.text[:120]}... (Score: {r.score:.3f})")

    # Teste 4: Consulta direta na tabela TUSS
    print("\n[Teste 4] Validação de Códigos TUSS:")
    assert "10101012" in tuss_dict, "TUSS 10101012 não encontrado!"
    assert "10101063" in tuss_dict, "TUSS 10101063 não encontrado!"
    assert "50000462" in tuss_dict, "TUSS 50000462 não encontrado!"
    print(f"   • TUSS 10101012: {tuss_dict['10101012']['descricao']} -> {tuss_dict['10101012']['urs']} URS")
    print(f"   • TUSS 10101063: {tuss_dict['10101063']['descricao']} -> {tuss_dict['10101063']['urs']} URS")
    print(f"   • TUSS 50000462: {tuss_dict['50000462']['descricao']} -> {tuss_dict['50000462']['urs']} URS")

    print("\n" + "=" * 70)
    print("🎉 HOMOLOGAÇÃO ETAPA 1.1 CONCLUÍDA COM 100% DE SUCESSO!")
    print("=" * 70)


def main() -> int:
    index, tuss_dict = construir_indice()
    validar_indice(tuss_dict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
