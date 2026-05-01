"""
Shared test fixtures.
"""
import os
import tempfile

import pytest

# Point all storage to temp dirs during tests so nothing is written to disk
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")


@pytest.fixture(scope="session")
def tmp_chroma_dir():
    """Create a temporary ChromaDB directory for the test session."""
    with tempfile.TemporaryDirectory() as d:
        os.environ["CHROMA_DB_PATH"] = d
        yield d


@pytest.fixture(scope="session")
def sample_txt_file(tmp_path_factory):
    """Write a small text file for ingestion tests."""
    p = tmp_path_factory.mktemp("docs") / "sample.txt"
    p.write_text(
        "Artificial Intelligence (AI) is the simulation of human intelligence by machines.\n"
        "Machine learning is a subset of AI that learns from data.\n"
        "Deep learning uses neural networks with many layers.\n"
        "Natural language processing enables computers to understand human language.\n"
    )
    return str(p)


@pytest.fixture(scope="session")
def sample_csv_file(tmp_path_factory):
    """Write a small CSV file for ingestion tests."""
    p = tmp_path_factory.mktemp("docs") / "sample.csv"
    p.write_text(
        "quarter,revenue,region\n"
        "Q1,120000,NA\n"
        "Q2,135000,EMEA\n"
    )
    return str(p)


@pytest.fixture(scope="session")
def db_engine():
    from sqlalchemy import create_engine
    from app.database import Base
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine
