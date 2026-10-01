"""Offline tests for the two harvesters (no network: arXiv is mocked)."""

import sys
import types
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class FakeResult:
    def __init__(self, short_id, title):
        self._id = short_id
        self.entry_id = f"http://arxiv.org/abs/{short_id}"
        self.title = title
        self.summary = "An abstract " * 20
        self.published = datetime(2025, 1, 1)
        self.categories = ["cs.AI"]

    def get_short_id(self):
        return self._id


def test_harvest_marks_papers_retrieved_by_both_queries(monkeypatch):
    results = {
        "A": [FakeResult("2501.00001v1", "Indigenous data"), FakeResult("2501.00003v2", "Indigenous AI ethics")],
        "B": [FakeResult("2501.00002v1", "AI safety"), FakeResult("2501.00003v1", "Indigenous AI ethics")],
    }
    calls = iter(["A", "B"])

    fake = types.SimpleNamespace(
        Client=lambda **kw: types.SimpleNamespace(results=lambda search: results[search.which]),
        Search=lambda **kw: types.SimpleNamespace(which=next(calls)),
        SortCriterion=types.SimpleNamespace(Relevance=None),
    )
    monkeypatch.setitem(sys.modules, "arxiv", fake)
    import fetch_arxiv

    df = fetch_arxiv.harvest(10).set_index("arxiv_id")
    assert df.loc["2501.00001v1", "label"] == "Field_A"
    assert df.loc["2501.00002v1", "label"] == "Field_B"
    assert df.loc["2501.00003v2", "label"] == "Both"
    assert df.loc["2501.00003v2", "retrieved_by"] == "Field_A;Field_B"
    assert len(df) == 3


def test_ingest_pdfs_builds_corpus(tmp_path, monkeypatch):
    fitz = pytest.importorskip("fitz")
    for field, title, body in [("a", "Indigenous knowledge of fire", "Traditional ecological knowledge " * 20),
                               ("b", "Explainable AI for clinicians", "We study explainable AI " * 20)]:
        folder = tmp_path / field
        folder.mkdir()
        doc = fitz.open()
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 800), f"{title}\n{body}")
        doc.save(folder / "paper.pdf")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["ingest_pdfs.py", "--field-a", str(tmp_path / "a"),
                                      "--field-b", str(tmp_path / "b"), "--out", str(out)])
    import ingest_pdfs

    ingest_pdfs.main()
    df = pd.read_csv(out / "corpus.csv")
    assert set(df["label"]) == {"Field_A", "Field_B"}
    assert df["on_topic"].all()
    assert df.loc[df.label == "Field_A", "title"].iloc[0] == "Indigenous knowledge of fire"
