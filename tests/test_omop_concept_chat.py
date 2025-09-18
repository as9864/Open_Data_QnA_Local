import asyncio
import pytest

from services import omop_concept_chat


def test_run_includes_concepts_in_prompt_and_response(monkeypatch):
    # Ensure cached factories do not leak between tests
    omop_concept_chat._get_embedder.cache_clear()
    omop_concept_chat._get_agent.cache_clear()

    captured_prompt = {}

    class FakeEmbedder:
        def create(self, question):
            assert question == "What is concept 123?"
            return [0.1, 0.2, 0.3]

    class FakeAgent:
        def generate_llm_response(self, prompt):
            captured_prompt["value"] = prompt
            return "fake answer"

    sample_concepts = [
        {
            "concept_id": 123,
            "concept_name": "Hypertension",
            "domain_id": "Condition",
            "vocabulary_id": "SNOMED",
            "concept_class_id": "Clinical Finding",
            "description": "concept_id 123 — Hypertension\nDomain: Condition\nVocabulary: SNOMED\nClass: Clinical Finding\nSynonyms: High blood pressure",
            "similarity": 0.95,
        }
    ]

    monkeypatch.setattr(omop_concept_chat, "_get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(omop_concept_chat, "_get_agent", lambda: FakeAgent())
    monkeypatch.setattr(
        omop_concept_chat,
        "_query_similar_concepts",
        lambda embedding, limit=5: sample_concepts,
    )

    payload = asyncio.run(omop_concept_chat.run("What is concept 123?", top_k=3))

    assert payload["answer"] == "fake answer"
    assert payload["concepts"] == sample_concepts
    assert payload["prompt"] == captured_prompt["value"]
    assert "concept_id 123" in payload["prompt"]
    assert "Hypertension" in payload["prompt"]
    assert "[Retrieved OMOP Concepts]" in payload["prompt"]
