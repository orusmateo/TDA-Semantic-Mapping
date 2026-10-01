import pandas as pd

from bridge_tda.corpus import ARXIV_QUERIES, FIELD_A, FIELD_B, audit_relevance, mentions_field_a, mentions_field_b


def test_field_a_lexicon_hits():
    for text in ["Indigenous data sovereignty", "traditional ecological knowledge of fire",
                 "Two-eyed seeing in practice", "decolonizing methodologies", "Māori language revitalisation",
                 "TEK and climate", "Anishinaabe governance", "tribal sovereignty and water"]:
        assert mentions_field_a(text), text


def test_field_a_lexicon_rejects_generic_knowledge_papers():
    for text in ["Knowledge graph question answering", "native speakers of English",
                 "tribal dynamics in online political debate", "tek is a token", "domain knowledge for IoT"]:
        assert not mentions_field_a(text), text


def test_field_b_lexicon():
    assert mentions_field_b("Towards explainable AI for clinicians")
    assert mentions_field_b("AI safety via debate")
    assert mentions_field_b("algorithmic fairness in lending")
    assert not mentions_field_b("Deep learning for dental X-ray segmentation")
    assert not mentions_field_b("the bias-variance trade-off")


def test_queries_use_quoted_phrases():
    for q in ARXIV_QUERIES.values():
        for clause in q.split(" OR "):
            assert clause.startswith(("abs:", "ti:"))
            if " " in clause.split(":", 1)[1]:
                assert clause.split(":", 1)[1].startswith('"')


def test_audit_flags_on_topic_and_dual():
    df = pd.DataFrame({
        "title": ["Indigenous knowledge and AI ethics", "Knowledge graphs", "AI safety", "Dental AI"],
        "text_clean": ["", "", "", ""],
        "label": [FIELD_A, FIELD_A, FIELD_B, FIELD_B],
    })
    out = audit_relevance(df)
    assert out["on_topic"].tolist() == [True, False, True, False]
    assert out["dual"].tolist() == [True, False, False, False]


def test_field_a_lexicon_rejects_non_ethnic_senses():
    for text in ["the space of dormant indigenous bundles", "Algebraic properties of Indigenous semirings",
                 "an indigenously developed sputtering system", "an indigenous web based platform",
                 "Diagnosis of Indigenous Crops", "indigenous and invasive grass species",
                 "Prior Indigenous Technological Species", "Traditional knowledge tracing (KT) models",
                 "beyond traditional knowledge management models"]:
        assert not mentions_field_a(text), text


def test_field_a_lexicon_keeps_ethnic_senses():
    for text in ["Indigenous Astronomy", "Indigenizing the next decade", "Machine Perception of Indigeneity",
                 "Indigenization of Urban Mobility", "indigenous peoples in Nigeria",
                 "Indigenous knowledge for drought forecasting", "traditional knowledge of healers"]:
        assert mentions_field_a(text), text


def test_field_b_broad_rule_needs_both_an_ai_term_and_a_safety_term():
    assert mentions_field_b("Liability and insurance for catastrophic losses from AI")
    assert mentions_field_b("Assuring EU AI Act compliance and adversarial robustness of LLMs")
    assert not mentions_field_b("Liability and insurance for catastrophic losses in nuclear power")
    assert not mentions_field_b("AI platform for edge devices")
