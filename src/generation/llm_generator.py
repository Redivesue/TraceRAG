"""Grounded answer generation with hosted LLM support and offline fallback."""

from __future__ import annotations

import re

from src.core.config import Settings
from src.core.schemas import Citation, GeneratedAnswer, RerankResult
from src.generation.base import AnswerGenerator
from src.generation.openai_compatible import LLMClientError, OpenAICompatibleClient

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
ENTITY_PATTERN = re.compile(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)")
CITATION_PATTERN = re.compile(r"\[(\d+)\]")
QUESTION_STARTERS = {"were", "what", "is", "are", "was", "who", "which", "did", "does", "do"}


class LLMAnswerGenerator(AnswerGenerator):
    def __init__(
        self,
        client: OpenAICompatibleClient | None = None,
        provider_name: str = "offline",
    ) -> None:
        self.client = client
        self.provider_name = provider_name

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMAnswerGenerator":
        provider = settings.llm_provider.strip().lower()
        if provider in {"openai", "openai_compatible"} and settings.llm_api_key:
            client = OpenAICompatibleClient(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                model=settings.llm_model,
                timeout_seconds=settings.llm_timeout_seconds,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
            return cls(client=client, provider_name="openai_compatible")
        return cls(client=None, provider_name="offline")

    def generate(self, query: str, evidence: list[RerankResult]) -> GeneratedAnswer:
        """Generate a grounded answer from top evidence, preferring hosted LLMs."""
        citations = [
            Citation(label=index + 1, chunk_id=item.chunk.chunk_id, title=item.chunk.title)
            for index, item in enumerate(evidence)
        ]
        if not evidence:
            return GeneratedAnswer(
                answer="I do not know based on the retrieved evidence.",
                citations=[],
                generator_mode="offline",
                path_reason="no_evidence",
            )

        fast_answer = self._build_fast_answer(query=query, evidence=evidence)
        if fast_answer is not None:
            return GeneratedAnswer(
                answer=fast_answer,
                citations=_select_citations(fast_answer, citations),
                generator_mode="fast_path",
                path_reason="fast_path_match",
            )

        if self.client is not None:
            try:
                answer = self._generate_with_llm(query=query, evidence=evidence)
                return GeneratedAnswer(
                    answer=answer,
                    citations=_select_citations(answer, citations),
                    generator_mode=self.provider_name,
                    path_reason="hosted_llm_success",
                )
            except LLMClientError:
                fallback_reason = "hosted_llm_error_offline_fallback"
        else:
            fallback_reason = "llm_not_configured_offline_fallback"

        answer = self._build_fallback_answer(query=query, evidence=evidence)
        return GeneratedAnswer(
            answer=answer,
            citations=_select_citations(answer, citations),
            generator_mode="offline",
            path_reason=fallback_reason if "fallback_reason" in locals() else "offline_fallback",
        )

    def _build_fast_answer(self, query: str, evidence: list[RerankResult]) -> str | None:
        comparison_query = _is_comparison_query(query)
        query_lower = query.lower()
        nationality_query = "nationality" in query_lower
        birthplace_query = (
            "birthplace" in query_lower
            or "where was" in query_lower
            or "born in which city" in query_lower
            or "born in which town" in query_lower
            or "what city was" in query_lower
        )
        profession_query = (
            "profession" in query_lower
            or "what does" in query_lower
            or "what is" in query_lower
            or "what was" in query_lower
            or "job" in query_lower
        )
        older_younger_query = "older" in query_lower or "younger" in query_lower
        same_profession_query = (
            "same profession" in query_lower
            or "same job" in query_lower
            or "both work as" in query_lower
        )
        same_birthplace_query = (
            "same birthplace" in query_lower
            or "same city" in query_lower
            or "same town" in query_lower
            or "same place" in query_lower
        )

        if comparison_query and older_younger_query:
            return self._build_fast_age_comparison_answer(query, evidence)

        if comparison_query and same_profession_query:
            return self._build_fast_profession_comparison_answer(query, evidence)

        if comparison_query and same_birthplace_query:
            return self._build_fast_birthplace_comparison_answer(query, evidence)

        if comparison_query:
            return self._build_fast_comparison_answer(query, evidence)

        if nationality_query:
            entity = _extract_entities(query)
            if not entity:
                return None
            nationality = _extract_nationality_from_evidence(entity[0], evidence)
            if nationality is None:
                return None
            label = nationality[1]
            return f"{entity[0]} is {nationality[0]} [{label}]."

        if birthplace_query:
            entity = _extract_entities(query)
            if not entity:
                return None
            birthplace = _extract_birthplace_from_evidence(entity[0], evidence)
            if birthplace is None:
                return None
            return f"{entity[0]} was born in {birthplace[0]} [{birthplace[1]}]."

        if profession_query:
            entity = _extract_entities(query)
            if not entity:
                return None
            profession = _extract_profession_from_evidence(entity[0], evidence)
            if profession is None:
                return None
            return f"{entity[0]} is {_format_descriptor(profession[0])} [{profession[1]}]."

        return None

    def _build_fast_comparison_answer(self, query: str, evidence: list[RerankResult]) -> str | None:
        entities = _extract_entities(query)
        if len(entities) < 2:
            return None

        facts: list[tuple[str, str, int]] = []
        for entity in entities[:2]:
            nationality = _extract_nationality_from_evidence(entity, evidence)
            if nationality is None:
                return None
            facts.append((entity, nationality[0], nationality[1]))

        if facts[0][1].lower() == facts[1][1].lower():
            return (
                f"{facts[0][0]} is {facts[0][1]} [{facts[0][2]}]. "
                f"{facts[1][0]} is {facts[1][1]} [{facts[1][2]}]. "
                f"They share the same nationality. [{facts[0][2]}][{facts[1][2]}]"
            )
        return (
            f"{facts[0][0]} is {facts[0][1]} [{facts[0][2]}]. "
            f"{facts[1][0]} is {facts[1][1]} [{facts[1][2]}]. "
            f"They do not share the same nationality. [{facts[0][2]}][{facts[1][2]}]"
        )

    def _build_fast_age_comparison_answer(self, query: str, evidence: list[RerankResult]) -> str | None:
        entities = _extract_entities(query)
        if len(entities) < 2:
            return None

        facts: list[tuple[str, int, int]] = []
        for entity in entities[:2]:
            birth_year = _extract_birth_year_from_evidence(entity, evidence)
            if birth_year is None:
                return None
            facts.append((entity, birth_year[0], birth_year[1]))

        older, younger = sorted(facts, key=lambda item: item[1])
        if "younger" in query.lower():
            return f"{younger[0]} is younger than {older[0]} [{younger[2]}][{older[2]}]."
        return f"{older[0]} is older than {younger[0]} [{older[2]}][{younger[2]}]."

    def _build_fast_profession_comparison_answer(self, query: str, evidence: list[RerankResult]) -> str | None:
        entities = _extract_entities(query)
        if len(entities) < 2:
            return None

        facts: list[tuple[str, str, int]] = []
        for entity in entities[:2]:
            profession = _extract_profession_from_evidence(entity, evidence)
            if profession is None:
                return None
            facts.append((entity, profession[0], profession[1]))

        if _normalize_phrase(facts[0][1]) == _normalize_phrase(facts[1][1]):
            return (
                f"{facts[0][0]} is {_format_descriptor(facts[0][1])} [{facts[0][2]}]. "
                f"{facts[1][0]} is {_format_descriptor(facts[1][1])} [{facts[1][2]}]. "
                f"They share the same profession. [{facts[0][2]}][{facts[1][2]}]"
            )
        return (
            f"{facts[0][0]} is {_format_descriptor(facts[0][1])} [{facts[0][2]}]. "
            f"{facts[1][0]} is {_format_descriptor(facts[1][1])} [{facts[1][2]}]. "
            f"They do not share the same profession. [{facts[0][2]}][{facts[1][2]}]"
        )

    def _build_fast_birthplace_comparison_answer(self, query: str, evidence: list[RerankResult]) -> str | None:
        entities = _extract_entities(query)
        if len(entities) < 2:
            return None

        facts: list[tuple[str, str, int]] = []
        for entity in entities[:2]:
            birthplace = _extract_birthplace_from_evidence(entity, evidence)
            if birthplace is None:
                return None
            facts.append((entity, birthplace[0], birthplace[1]))

        if _normalize_phrase(facts[0][1]) == _normalize_phrase(facts[1][1]):
            return (
                f"{facts[0][0]} was born in {facts[0][1]} [{facts[0][2]}]. "
                f"{facts[1][0]} was born in {facts[1][1]} [{facts[1][2]}]. "
                f"They share the same birthplace. [{facts[0][2]}][{facts[1][2]}]"
            )
        return (
            f"{facts[0][0]} was born in {facts[0][1]} [{facts[0][2]}]. "
            f"{facts[1][0]} was born in {facts[1][1]} [{facts[1][2]}]. "
            f"They do not share the same birthplace. [{facts[0][2]}][{facts[1][2]}]"
        )

    def _generate_with_llm(self, query: str, evidence: list[RerankResult]) -> str:
        comparison_query = _is_comparison_query(query)
        system_prompt = (
            "You are a grounded search answer generator. "
            "Answer only from the provided evidence. "
            "If the evidence is insufficient, say you do not know. "
            "Every factual claim must include citations like [1] or [2]. "
            "Do not cite chunks that are not provided. "
            "Keep the answer concise."
        )
        numbered_evidence = []
        for index, item in enumerate(evidence, start=1):
            numbered_evidence.append(
                f"[{index}] Title: {item.chunk.title}\n"
                f"Text: {item.chunk.text}"
            )
        evidence_block = "\n\n".join(numbered_evidence)

        if comparison_query:
            instruction = (
                "Use this structure: "
                "Entity A evidence; Entity B evidence; Final comparison. "
                "If one side is missing, say there is insufficient evidence."
            )
        else:
            instruction = "Write a short answer using only the evidence above."

        user_prompt = (
            f"Question: {query}\n\n"
            f"Evidence:\n\n{evidence_block}\n\n"
            f"{instruction}"
        )
        return self.client.generate(system_prompt=system_prompt, user_prompt=user_prompt)

    def _build_fallback_answer(self, query: str, evidence: list[RerankResult]) -> str:
        if _is_comparison_query(query):
            comparison = self._build_comparison_fallback(query, evidence)
            if comparison:
                return comparison

        query_tokens = {token.lower() for token in TOKEN_PATTERN.findall(query)}
        best_sentences: list[tuple[float, int, str]] = []

        for index, item in enumerate(evidence):
            sentences = _split_sentences(item.chunk.text)
            for sentence in sentences:
                sentence_tokens = {token.lower() for token in TOKEN_PATTERN.findall(sentence)}
                overlap = len(query_tokens & sentence_tokens)
                if overlap == 0:
                    continue
                best_sentences.append((overlap + item.rerank_score, index + 1, sentence))

        if not best_sentences:
            fallback = evidence[0].chunk.text[:220].rstrip()
            return f"Based on the retrieved evidence, the most relevant passage says: {fallback} [1]"

        best_sentences.sort(key=lambda item: item[0], reverse=True)
        used_labels: set[int] = set()
        normalized_sentences: set[str] = set()
        snippets: list[str] = []
        for _, label, sentence in best_sentences:
            if label in used_labels:
                continue
            normalized = _normalize_sentence(sentence)
            if normalized in normalized_sentences:
                continue
            used_labels.add(label)
            normalized_sentences.add(normalized)
            snippets.append(f"{sentence.strip()} [{label}]")
            if len(snippets) == 2:
                break

        if len(snippets) == 2 and snippets[0].split("[")[0].strip() == snippets[1].split("[")[0].strip():
            snippets = snippets[:1]

        return " ".join(snippets)

    def _build_comparison_fallback(self, query: str, evidence: list[RerankResult]) -> str | None:
        entities = _extract_entities(query)
        if len(entities) < 2:
            return None

        entity_evidence: list[tuple[str, int, str]] = []
        for entity in entities[:2]:
            entity_lower = entity.lower()
            found = None
            for index, item in enumerate(evidence, start=1):
                if entity_lower in item.chunk.title.lower() or entity_lower in item.chunk.text.lower():
                    found = (entity, index, item.chunk.text)
                    break
            if found is None:
                return "I do not know based on the retrieved evidence."
            entity_evidence.append(found)

        summaries = []
        for entity, label, text in entity_evidence:
            sentence = _split_sentences(text)[0].strip()
            summaries.append(f"{entity}: {sentence} [{label}]")
        return " ".join(summaries)


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence for sentence in sentences if sentence]


def _extract_entities(query: str) -> list[str]:
    entities: list[str] = []
    for match in ENTITY_PATTERN.findall(query):
        cleaned = _clean_entity_match(match)
        if cleaned and cleaned not in entities:
            entities.append(cleaned)
    return entities


def _is_comparison_query(query: str) -> bool:
    query_lower = query.lower()
    markers = ["same ", " both ", "compare", "older", "younger", "than", "difference"]
    return any(marker in query_lower for marker in markers)


def _select_citations(answer: str, citations: list[Citation]) -> list[Citation]:
    used_labels = {int(match) for match in CITATION_PATTERN.findall(answer)}
    if not used_labels:
        return citations[:1]
    return [citation for citation in citations if citation.label in used_labels]


def _extract_nationality_from_evidence(
    entity: str,
    evidence: list[RerankResult],
) -> tuple[str, int] | None:
    patterns = [
        re.compile(r"\bis an? ([A-Z][a-z]+)\b"),
        re.compile(r"\bwas an? ([A-Z][a-z]+)\b"),
        re.compile(r"\b([A-Z][a-z]+) filmmaker\b"),
        re.compile(r"\b([A-Z][a-z]+) director\b"),
        re.compile(r"\b([A-Z][a-z]+) actor\b"),
        re.compile(r"\b([A-Z][a-z]+) writer\b"),
    ]
    entity_lower = entity.lower()
    for index, item in enumerate(evidence, start=1):
        text = item.chunk.text
        if entity_lower not in item.chunk.title.lower() and entity_lower not in text.lower():
            continue
        for sentence in _split_sentences(text):
            for pattern in patterns:
                match = pattern.search(sentence)
                if match:
                    return match.group(1), index
    return None


def _extract_birthplace_from_evidence(
    entity: str,
    evidence: list[RerankResult],
) -> tuple[str, int] | None:
    patterns = [
        re.compile(r"\bwas born in ([A-Z][A-Za-z]+(?:,\s*[A-Z][A-Za-z]+)*)"),
        re.compile(r"\bborn in ([A-Z][A-Za-z]+(?:,\s*[A-Z][A-Za-z]+)*)"),
        re.compile(r"\bborn at ([A-Z][A-Za-z]+(?:,\s*[A-Z][A-Za-z]+)*)"),
        re.compile(r"\bfrom ([A-Z][A-Za-z]+(?:,\s*[A-Z][A-Za-z]+)*)"),
        re.compile(r"\(([A-Z][A-Za-z]+,\s*[A-Z][A-Za-z]+)[^)]*\)"),
    ]
    entity_lower = entity.lower()
    for index, item in enumerate(evidence, start=1):
        text = item.chunk.text
        if entity_lower not in item.chunk.title.lower() and entity_lower not in text.lower():
            continue
        for sentence in _split_sentences(text):
            for pattern in patterns:
                match = pattern.search(sentence)
                if match:
                    return _clean_location(match.group(1)), index
    return None


def _extract_profession_from_evidence(
    entity: str,
    evidence: list[RerankResult],
) -> tuple[str, int] | None:
    patterns = [
        re.compile(r"\bis an? ([a-zA-Z,\-\s]+?)(?:\.|,| who| and| with)"),
        re.compile(r"\bwas an? ([a-zA-Z,\-\s]+?)(?:\.|,| who| and| with)"),
    ]
    entity_lower = entity.lower()
    for index, item in enumerate(evidence, start=1):
        text = item.chunk.text
        if entity_lower not in item.chunk.title.lower() and entity_lower not in text.lower():
            continue
        for sentence in _split_sentences(text):
            for pattern in patterns:
                match = pattern.search(sentence)
                if match:
                    profession = _clean_profession(match.group(1))
                    return profession, index
    return None


def _extract_birth_year_from_evidence(
    entity: str,
    evidence: list[RerankResult],
) -> tuple[int, int] | None:
    patterns = [
        re.compile(r"\(.*?([12][0-9]{3}).*?\)"),
        re.compile(r"\bborn .*?([12][0-9]{3})"),
    ]
    entity_lower = entity.lower()
    for index, item in enumerate(evidence, start=1):
        text = item.chunk.text
        if entity_lower not in item.chunk.title.lower() and entity_lower not in text.lower():
            continue
        for sentence in _split_sentences(text):
            for pattern in patterns:
                match = pattern.search(sentence)
                if match:
                    return int(match.group(1)), index
    return None


def _clean_entity_match(match: str) -> str:
    parts = match.split()
    while parts and parts[0].lower() in QUESTION_STARTERS:
        parts = parts[1:]
    return " ".join(parts)


def _clean_profession(text: str) -> str:
    cleaned = text.strip(" .,-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace("  ", " ")
    cleaned = cleaned.lower()
    cleaned = re.sub(r"^(an?|the)\s+", "", cleaned)
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if parts:
        cleaned = parts[0]
    return cleaned


def _clean_location(text: str) -> str:
    cleaned = text.strip(" .")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _normalize_phrase(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _normalize_sentence(text: str) -> str:
    normalized = _normalize_phrase(text)
    normalized = re.sub(r"\b(the|a|an|is|was|were|of|for|to|in)\b", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _format_descriptor(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return text
    for adjective in ["american", "british", "french", "canadian", "german", "italian"]:
        if cleaned.startswith(adjective + " "):
            cleaned = adjective.capitalize() + cleaned[len(adjective):]
            break
    if cleaned.startswith(("a ", "an ", "the ")):
        return cleaned
    article = "an" if cleaned[0].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {cleaned}"
