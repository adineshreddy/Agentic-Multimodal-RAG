"""
Tests for the document ingestion pipeline.
"""
import base64
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document


def test_supported_extensions():
    from app.rag.ingestion import SUPPORTED_EXTENSIONS

    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".docx" in SUPPORTED_EXTENSIONS
    assert ".txt" in SUPPORTED_EXTENSIONS
    assert ".xlsx" in SUPPORTED_EXTENSIONS
    assert ".csv" in SUPPORTED_EXTENSIONS


def test_load_and_chunk_txt_stamps_metadata(sample_txt_file):
    from app.rag.ingestion import load_and_chunk

    with patch("app.rag.ingestion._summarize_text_block", return_value=None):
        chunks, summaries, document_id = load_and_chunk(sample_txt_file, user_id=1)

    assert chunks, "Expected at least one chunk from sample text file."
    assert isinstance(document_id, str) and document_id
    assert summaries == []

    for i, chunk in enumerate(chunks):
        assert chunk.page_content.strip()
        assert chunk.metadata.get("source") == Path(sample_txt_file).name
        assert chunk.metadata.get("user_id") == 1
        assert chunk.metadata.get("node_type") == "raw_chunk"
        assert chunk.metadata.get("document_id") == document_id
        assert chunk.metadata.get("chunk_index") == i


def test_load_and_chunk_csv_structures_rows(sample_csv_file):
    from app.rag.ingestion import load_and_chunk

    with patch("app.rag.ingestion._summarize_text_block", return_value=None):
        chunks, summaries, _document_id = load_and_chunk(sample_csv_file, user_id=2)

    assert chunks, "Expected at least one chunk from sample CSV file."
    assert summaries == []

    joined = "\n".join(c.page_content for c in chunks)
    assert "quarter:" in joined.lower()
    assert "revenue:" in joined.lower()
    assert any(chunk.metadata.get("source") == "sample.csv" for chunk in chunks)


def test_load_docx_multimodal_adds_image_summaries(tmp_path, monkeypatch):
    from app.rag import ingestion

    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jx90AAAAASUVORK5CYII="
    )
    docx_path = tmp_path / "with-image.docx"
    with zipfile.ZipFile(docx_path, "w") as archive:
        archive.writestr("word/media/image1.png", png_bytes)

    monkeypatch.setattr(ingestion.settings, "enable_image_summaries", True)
    monkeypatch.setattr(ingestion.settings, "vision_max_images_per_document", 3)

    with (
        patch(
            "app.rag.ingestion._load_docx_text",
            return_value=[Document(page_content="Quarterly report text.", metadata={"source": "with-image.docx"})],
        ),
        patch("app.rag.ingestion._summarize_image", return_value="A chart showing growth by quarter."),
    ):
        bundle = ingestion._load_docx(str(docx_path), document_id="doc-xyz", user_id=4)

    assert bundle.raw_docs
    assert bundle.used_multimodal is True
    assert len(bundle.summary_docs) == 1

    summary = bundle.summary_docs[0]
    assert summary.metadata["node_type"] == "image_summary"
    assert summary.metadata["extraction_method"] == "docx_media"
    assert summary.metadata["source"] == "with-image.docx"
    assert Path(summary.metadata["image_path"]).exists()


def test_tabular_rows_to_docs_includes_date_variants_for_matching():
    from app.rag.ingestion import _tabular_rows_to_docs

    rows = [
        ["Name", "Medical Condition", "Date of Admission"],
        ["Alice Johnson", "Arthritis", date(2020, 7, 14)],
    ]

    docs = _tabular_rows_to_docs(rows, filename="healthcare_dataset.xlsx", metadata={"sheet": "Sheet1"})

    assert len(docs) == 1
    content = docs[0].page_content
    assert "Medical Condition: Arthritis" in content
    assert "Date of Admission:" in content
    assert "7/14/20" in content
    assert "2020-07-14" in content
    assert "was admitted for Arthritis on 2020-07-14" in content


def test_build_section_summaries_generates_nodes(monkeypatch):
    from app.rag import ingestion

    raw_docs = [
        Document(
            page_content="Revenue increased 12% due to enterprise deals.",
            metadata={"section_title": "Financial Overview", "page": 1},
        ),
        Document(
            page_content="Gross margin expanded while opex remained flat.",
            metadata={"section_title": "Financial Overview", "page": 2},
        ),
        Document(
            page_content="Churn reduced after onboarding updates.",
            metadata={"section_title": "Customer Retention", "page": 3},
        ),
    ]

    monkeypatch.setattr(ingestion.settings, "enable_summary_index", True)
    monkeypatch.setattr(ingestion.settings, "enable_section_summaries", True)

    with patch(
        "app.rag.ingestion._summarize_text_block",
        side_effect=["Financial summary", "Retention summary"],
    ):
        summaries = ingestion._build_section_summaries(
            source="report.pdf",
            file_path="/tmp/report.pdf",
            document_id="doc-123",
            user_id=7,
            raw_docs=raw_docs,
        )

    assert len(summaries) == 2
    assert all(s.metadata["node_type"] == "section_summary" for s in summaries)
    assert all(s.metadata["document_id"] == "doc-123" for s in summaries)
    assert summaries[0].metadata["section_title"] == "Financial Overview"
    assert summaries[1].metadata["section_title"] == "Customer Retention"


def test_ingest_file_summary_only_warning(tmp_path):
    from app.rag.ingestion import ingest_file

    fpath = tmp_path / "image_only.pdf"
    fpath.write_bytes(b"fake")
    summary_doc = Document(
        page_content="Image-based summary only.",
        metadata={"source": fpath.name, "node_type": "image_summary"},
    )

    with patch("app.rag.ingestion.load_and_chunk", return_value=([], [summary_doc], "doc-x")), \
         patch("app.rag.ingestion.delete_document_family", return_value=0), \
         patch("app.rag.ingestion.add_raw_documents", return_value=[]), \
         patch("app.rag.ingestion.add_summary_documents", return_value=["sum-1"]):
        result = ingest_file(str(fpath), user_id=1)

    assert result["file"] == "image_only.pdf"
    assert result["chunks"] == 0
    assert result["ids"] == 1
    assert "summary-only" in (result.get("warning") or "").lower()


def test_ingest_directory_skips_unsupported(tmp_path):
    (tmp_path / "valid.txt").write_text("Valid document text")
    (tmp_path / "invalid.bin").write_bytes(b"\x00\x01")

    from app.rag import ingestion

    with patch("app.rag.ingestion.ingest_file", return_value={"file": "valid.txt", "chunks": 1}):
        results = ingestion.ingest_directory(str(tmp_path), user_id=99)

    assert [r["file"] for r in results] == ["valid.txt"]


def test_remove_multimodal_assets_deletes_expected_dir(tmp_path):
    from app.rag import ingestion

    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"fake-pdf")
    document_id = ingestion._document_id(str(file_path), user_id=12)
    assets_dir = tmp_path / ".multimodal_assets" / document_id
    assets_dir.mkdir(parents=True)
    (assets_dir / "img-1.jpg").write_bytes(b"fake")

    removed = ingestion.remove_multimodal_assets(str(file_path), user_id=12)
    assert removed == 1
    assert not assets_dir.exists()

    removed_again = ingestion.remove_multimodal_assets(str(file_path), user_id=12)
    assert removed_again == 0
