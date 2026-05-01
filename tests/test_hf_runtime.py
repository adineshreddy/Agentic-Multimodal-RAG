"""Tests for Hugging Face runtime setup."""

import os

from app.hf_runtime import configure_hf_runtime


def test_configure_hf_runtime_sets_tokens_and_xet_flag(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "true")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    configure_hf_runtime()

    assert "HF_TOKEN" in os.environ
    assert os.environ["HF_TOKEN"] == "hf_test_token"
    assert os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf_test_token"
    assert os.environ["HF_HUB_DISABLE_XET"] in {"true", "1"}


def test_configure_hf_runtime_respects_disable_xet_false(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "false")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    configure_hf_runtime()

    # When disabled in config/env, we should not force-enable it.
    assert os.environ.get("HF_HUB_DISABLE_XET") == "false"
