"""SlopTotal's six regex/counting engines, ported line for line.

Exact ports of ``app/engines/{linguistic,formulaic,structural,vocabulary,
readability,sentiment}.py`` from SlopTotal (https://github.com/pablocaeg/sloptotal,
MIT License, Copyright (c) 2025-2026 SlopTotal). Phrase and pattern lists are
copied verbatim and the code is followed even where its comments disagree
(e.g. the formulaic opener window is 300 chars, the closer window 400).
Each ``score`` returns the raw 0-1 score, before SlopTotal's ``round(score, 3)``.
``tests/test_detectors_heuristic.py`` checks parity against SlopTotal's output.

``HEURISTIC_ENGINES`` is a list of engine *instances*, in SlopTotal's order.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter

# --- linguistic -------------------------------------------------------------

# (word_or_phrase, weight) — higher weight = stronger AI signal
AI_MARKERS = [
    # Tier 1: Very strong AI signals (weight 3)
    ("delve", 3),
    ("tapestry", 3),
    ("multifaceted", 3),
    ("ever-evolving", 3),
    ("thought-provoking", 3),
    ("it's important to note", 3),
    ("it's worth noting", 3),
    ("in today's digital age", 3),
    ("in today's rapidly evolving", 3),
    ("let's dive in", 3),
    # Tier 2: Strong signals (weight 2)
    ("landscape", 2),
    ("navigate", 2),
    ("leverage", 2),
    ("foster", 2),
    ("pivotal", 2),
    ("nuanced", 2),
    ("robust", 2),
    ("holistic", 2),
    ("synergy", 2),
    ("paradigm", 2),
    ("encompass", 2),
    ("intricate", 2),
    ("comprehensive", 2),
    ("underscores", 2),
    ("underscore", 2),
    ("realm", 2),
    ("cornerstone", 2),
    ("underpinning", 2),
    ("facilitating", 2),
    ("harnessing", 2),
    ("spearheading", 2),
    ("revolutionize", 2),
    ("groundbreaking", 2),
    ("cutting-edge", 2),
    ("game-changer", 2),
    ("deep dive", 2),
    ("at the forefront", 2),
    ("at its core", 2),
    ("in the realm of", 2),
    ("it is crucial", 2),
    ("it is essential", 2),
    ("plays a crucial role", 2),
    ("serves as a testament", 2),
    ("is a testament to", 2),
    ("a myriad of", 2),
    ("plethora", 2),
    # Tier 3: Moderate signals (weight 1)
    ("crucial", 1),
    ("enhance", 1),
    ("dynamic", 1),
    ("innovative", 1),
    ("streamline", 1),
    ("optimize", 1),
    ("elevate", 1),
    ("empower", 1),
    ("stakeholder", 1),
    ("ecosystem", 1),
    ("actionable", 1),
    ("seamless", 1),
    ("seamlessly", 1),
    ("furthermore", 1),
    ("moreover", 1),
    ("consequently", 1),
    ("nevertheless", 1),
    ("in conclusion", 1),
    ("ultimately", 1),
    ("in essence", 1),
    ("it is important to", 1),
    ("not only", 1),
    ("but also", 1),
    ("on the other hand", 1),
    ("having said that", 1),
    ("that being said", 1),
    ("with that in mind", 1),
    ("in this context", 1),
    ("in light of", 1),
    ("as we navigate", 1),
    ("when it comes to", 1),
    ("it goes without saying", 1),
    ("needless to say", 1),
    ("at the end of the day", 1),
    ("the bottom line", 1),
    ("key takeaway", 1),
    ("food for thought", 1),
    ("resonate", 1),
    ("aligns with", 1),
    ("bolster", 1),
    ("catalyst", 1),
    ("testament", 1),
    ("arguably", 1),
    ("notably", 1),
    ("specifically", 1),
    ("essentially", 1),
    ("fundamentally", 1),
    ("inherently", 1),
    ("intricacies", 1),
    # Spanish markers
    ("en el panorama actual", 2),
    ("es importante destacar", 2),
    ("cabe señalar", 2),
    ("en este contexto", 2),
    ("sin lugar a dudas", 2),
    ("en la era digital", 2),
    ("resulta fundamental", 2),
    ("no cabe duda", 2),
]


class LinguisticEngine:
    key = "linguistic"
    name = "Linguistic Markers"
    engine_type = "linguistic"

    def score(self, text: str) -> float:
        text_lower = text.lower()
        word_count = max(len(text.split()), 1)
        total_weight = 0
        for marker, weight in AI_MARKERS:
            count = len(re.findall(r"\b" + re.escape(marker) + r"\b", text_lower))
            if count > 0:
                total_weight += count * weight
        # Density: weighted hits per 1000 words; 15+ maps to 1.0
        density = (total_weight / word_count) * 1000
        return min(density / 15.0, 1.0)


# --- formulaic --------------------------------------------------------------

AI_OPENERS = [
    r"in today'?s (?:rapidly )?(?:evolving|changing|digital|modern)",
    r"in (?:the|a) (?:world|era|age|landscape) (?:where|of|that)",
    r"when it comes to",
    r"in the realm of",
    r"(?:are you |ever )?(?:looking|wondering|struggling|trying) to",
    r"(?:have you ever )?wondered (?:what|how|why)",
    r"let'?s (?:dive|explore|take a (?:closer )?look|unpack|break down)",
    r"(?:imagine|picture) (?:this|a world)",
    r"(?:whether you'?re|if you'?re) (?:a |an )?(?:seasoned|beginner|new)",
    r"(?:in|throughout) (?:recent years|the past decade|today'?s society)",
    r"(?:it'?s no (?:secret|surprise)|there'?s no denying) that",
    r"as (?:technology|the world|we|society) continues to",
    r"the (?:rise|emergence|advent|proliferation) of",
]

AI_CLOSERS = [
    r"in conclusion,?",
    r"(?:in summary|to summarize|to sum up),?",
    r"(?:ultimately|at the end of the day),?",
    r"as we (?:navigate|move forward|look ahead|continue)",
    r"(?:the (?:future|road ahead) (?:is|looks|holds))",
    r"(?:by|through) (?:embracing|leveraging|harnessing|adopting)",
    r"(?:only time will tell|the possibilities are (?:endless|limitless))",
    r"(?:it'?s clear|one thing is (?:clear|certain)) that",
    r"(?:in this ever|in our ever|in an ever)",
    r"(?:remember|keep in mind),? (?:it'?s|the)",
]

FILLER_PATTERNS = [
    r"not only\b.{3,60}\bbut also",
    r"on (?:the )?one hand\b.{3,120}\bon the other(?: hand)?",
    r"it(?:'?s| is) (?:important|worth|crucial|essential|interesting) to (?:note|remember|mention|highlight|consider|understand|recognize)",
    r"(?:this|that|which) (?:is to say|means that|implies that|suggests that|indicates that)",
    r"(?:needless to say|it goes without saying)",
    r"(?:first and foremost|last but not least)",
    r"(?:without (?:a )?doubt|beyond (?:a )?shadow of (?:a )?doubt)",
    r"(?:serves as|acts as) (?:a )?(?:powerful |key |critical )?(?:reminder|testament|example|illustration)",
]


class FormulaicEngine:
    key = "formulaic"
    name = "Formulaic Patterns"
    engine_type = "linguistic"

    def score(self, text: str) -> float:
        text_lower = text.lower()
        word_count = max(len(text.split()), 1)
        # SlopTotal decides opener/closer by substring-searching these display
        # strings for "opener"/"closer", so a match like "let's take a closer
        # look" also sets the closer. They are rebuilt exactly to keep that.
        found = []

        opener_text = text_lower[:300]
        for pattern in AI_OPENERS:
            match = re.search(pattern, opener_text)
            if match:
                found.append(f'opener: "{match.group()}"')
                break

        closer_text = text_lower[-400:]
        for pattern in AI_CLOSERS:
            match = re.search(pattern, closer_text)
            if match:
                found.append(f'closer: "{match.group()}"')
                break

        filler_count = 0
        for pattern in FILLER_PATTERNS:
            matches = re.findall(pattern, text_lower)
            filler_count += len(matches)
            for m in matches[:2]:
                snippet = m if isinstance(m, str) else m[0]
                found.append(f'filler: "{snippet[:50]}"')

        # Sentence starter repetition
        sents = re.split(r"(?<=[.!?])\s+", text)
        first_words = []
        for s in sents:
            s = s.strip()
            if s:
                word = re.match(r"[A-Za-z]+", s)
                if word:
                    first_words.append(word.group().lower())
        if len(first_words) >= 5:
            starter_counts = Counter(first_words)
            repeated = sum(c for c in starter_counts.values() if c > 1)
            repeat_ratio = repeated / len(first_words)
            starter_score = min(max(repeat_ratio - 0.55, 0) / 0.25, 1.0)
            if repeat_ratio > 0.55:
                top_starters = starter_counts.most_common(3)
                top_str = ", ".join(f'"{w}"x{c}' for w, c in top_starters if c > 1)
                found.append(f"starter repetition: {repeat_ratio:.0%} ({top_str})")
        else:
            starter_score = 0.0

        # Heading -> bullet-list pattern
        lines = text.split("\n")
        heading_list_count = 0
        i = 0
        while i < len(lines) - 1:
            line = lines[i].strip()
            if (
                line
                and len(line) < 60
                and not line.endswith(".")
                and not re.match(r"\s*[-•*]\s", line)
            ):
                j = i + 1
                bullet_run = 0
                while j < len(lines) and re.match(r"\s*[-•*]\s", lines[j].strip()):
                    bullet_run += 1
                    j += 1
                if bullet_run >= 2:
                    heading_list_count += 1
                    i = j
                    continue
            i += 1
        heading_list_score = min(heading_list_count / 3.0, 1.0)

        opener_score = 1.0 if any("opener" in f for f in found) else 0.0
        closer_score = 1.0 if any("closer" in f for f in found) else 0.0
        filler_density = (filler_count / word_count) * 1000
        filler_score = min(filler_density / 5.0, 1.0)

        return (
            opener_score * 0.20
            + closer_score * 0.15
            + filler_score * 0.30
            + starter_score * 0.05
            + heading_list_score * 0.30
        )


# --- structural -------------------------------------------------------------


class StructuralEngine:
    key = "structural"
    name = "Structural Analysis"
    engine_type = "linguistic"

    def score(self, text: str) -> float:
        # 1. Em-dash frequency (word—word without spaces)
        em_dashes = len(re.findall(r"\w—\w", text))
        word_count = max(len(text.split()), 1)
        em_dash_rate = (em_dashes / word_count) * 1000
        em_score = min(em_dash_rate / 6.0, 1.0)

        # 2. Sentence length uniformity
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
        if len(sentences) >= 3:
            lengths = [len(s.split()) for s in sentences]
            mean_len = statistics.mean(lengths)
            stdev = statistics.stdev(lengths)
            cv = stdev / mean_len if mean_len > 0 else 0
            uniformity_score = max(0, 1.0 - (cv / 0.6))
        else:
            uniformity_score = 0.0

        # 3. List/bullet density per 1k words
        lines = text.split("\n")
        list_lines = sum(
            1
            for l in lines  # noqa: E741
            if re.match(r"\s*[-•*]\s", l) or re.match(r"\s*\d+[.)]\s", l)
        )
        list_per_1k = (list_lines / word_count) * 1000
        list_score = min(list_per_1k / 30.0, 1.0)

        # 4. Paragraph length uniformity
        paragraphs = [
            p.strip() for p in text.split("\n\n") if len(p.strip().split()) >= 5
        ]
        if len(paragraphs) >= 3:
            para_lengths = [len(p.split()) for p in paragraphs]
            para_mean = statistics.mean(para_lengths)
            para_stdev = statistics.stdev(para_lengths)
            para_cv = para_stdev / para_mean if para_mean > 0 else 0
            para_score = max(0, 1.0 - (para_cv / 0.5))
        else:
            para_score = 0.0

        # 5. Bold/heading density
        bold_count = len(re.findall(r"\*\*[^*]+\*\*", text))
        heading_count = len(re.findall(r"^#{1,6}\s", text, re.MULTILINE))
        format_density = ((bold_count + heading_count) / word_count) * 1000
        format_score = min(format_density / 10.0, 1.0)

        # 6. Colon density per 1k words; > 8 starts scoring
        colon_count = text.count(":")
        colon_density = (colon_count / word_count) * 1000
        colon_score = min(max(colon_density - 8, 0) / 15.0, 1.0)

        # 7. Colon-introduced definitions ("Term: Explanation")
        colon_defs = len(
            re.findall(r"[A-Z][a-z]+(?:\s[A-Z]?[a-z]+){0,3}:\s[A-Z]", text)
        )
        colon_def_score = min(colon_defs / 3.0, 1.0)

        return (
            em_score * 0.08
            + uniformity_score * 0.08
            + list_score * 0.12
            + para_score * 0.07
            + format_score * 0.05
            + colon_score * 0.30
            + colon_def_score * 0.30
        )


# --- vocabulary -------------------------------------------------------------


class VocabularyEngine:
    key = "vocabulary"
    name = "Vocabulary Richness"
    engine_type = "linguistic"

    def score(self, text: str) -> float:
        words = re.findall(r"[a-zA-Z]+", text.lower())
        if len(words) < 20:
            return 0.0

        total = len(words)
        freq = Counter(words)
        unique = len(freq)

        # 1. Root type-token ratio; lower -> more AI
        root_ttr = unique / math.sqrt(total)
        ttr_score = max(0, 1.0 - ((root_ttr - 3.5) / 4.0))
        ttr_score = min(max(ttr_score, 0.0), 1.0)

        # 2. Hapax legomena ratio
        hapax = sum(1 for w, c in freq.items() if c == 1)
        hapax_ratio = hapax / unique if unique > 0 else 0
        hapax_score = max(0, 1.0 - ((hapax_ratio - 0.3) / 0.4))
        hapax_score = min(max(hapax_score, 0.0), 1.0)

        # 3. Yule's K (lower = richer vocabulary)
        freq_of_freq = Counter(freq.values())
        m1 = total
        m2 = sum(i * i * v for i, v in freq_of_freq.items())
        yules_k = 10000 * (m2 - m1) / (m1 * m1) if m1 > 0 else 0
        k_score = min(max((yules_k - 80) / 120, 0.0), 1.0)

        return ttr_score * 0.35 + hapax_score * 0.35 + k_score * 0.3


# --- readability ------------------------------------------------------------


def _syllable_count(word: str) -> int:
    word = word.lower().rstrip("e")
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(count, 1)


def _flesch_kincaid(text: str) -> float:
    """Flesch Reading Ease (SlopTotal's name for it notwithstanding)."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
    if not sentences:
        return 0.0
    words = re.findall(r"[a-zA-Z]+", text)
    if not words:
        return 0.0
    total_sentences = len(sentences)
    total_words = len(words)
    total_syllables = sum(_syllable_count(w) for w in words)
    return (
        206.835
        - 1.015 * (total_words / total_sentences)
        - 84.6 * (total_syllables / total_words)
    )


class ReadabilityEngine:
    """Cross-paragraph CV of Flesch Reading Ease; higher CV scores as more AI
    (SlopTotal recalibrated the direction on RAID, 2026-07-25)."""

    key = "readability"
    name = "Readability Uniformity"
    engine_type = "linguistic"

    def score(self, text: str) -> float:
        paragraphs = [
            p.strip() for p in text.split("\n\n") if len(p.strip().split()) >= 15
        ]
        if len(paragraphs) < 3:
            # Fall back to word chunks
            words = text.split()
            chunk_size = max(len(words) // 4, 20)
            paragraphs = []
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size])
                if len(chunk.split()) >= 15:
                    paragraphs.append(chunk)
        if len(paragraphs) < 3:
            return 0.0

        fk_scores = [_flesch_kincaid(p) for p in paragraphs]
        mean_fk = statistics.mean(fk_scores)
        stdev_fk = statistics.stdev(fk_scores)
        cv = stdev_fk / abs(mean_fk) if mean_fk != 0 else 0
        CV_LOW, CV_HIGH = 0.35, 1.30
        uniformity_score = (cv - CV_LOW) / (CV_HIGH - CV_LOW)
        return min(max(uniformity_score, 0.0), 1.0)


# --- sentiment --------------------------------------------------------------

HEDGING_PHRASES = [
    "it's worth noting",
    "it is worth noting",
    "it's important to remember",
    "it is important to remember",
    "it's important to note",
    "it is important to note",
    "it's crucial to",
    "it is crucial to",
    "it's essential to",
    "it is essential to",
    "it should be noted",
    "one might argue",
    "it could be argued",
    "some might say",
    "there are those who",
    "many experts believe",
    "studies suggest",
    "research indicates",
    "while there are",
    "although there are",
    "that said",
    "having said that",
    "that being said",
    "with that being said",
    "be that as it may",
    "regardless",
    "irrespective of",
]

BALANCE_PATTERNS = [
    r"on (?:the )?one hand\b.{5,150}\bon the other(?: hand)?",
    r"while .{5,80}, (?:it |there |we |they )",
    r"although .{5,80}, (?:it |there |we |they )",
    r"(?:pros|advantages|benefits) .{5,200}(?:cons|disadvantages|drawbacks)",
]

VAGUE_POSITIVITY = [
    "incredibly important",
    "extremely valuable",
    "truly remarkable",
    "absolutely essential",
    "deeply meaningful",
    "truly transformative",
    "game-changing",
    "world-class",
    "best-in-class",
    "make a difference",
    "positive impact",
    "meaningful change",
    "better future",
    "brighter future",
    "greater good",
    "incredible potential",
    "exciting opportunity",
    "powerful tool",
    "valuable resource",
    "essential guide",
]

FILLER_QUALIFIERS = [
    "very",
    "really",
    "extremely",
    "incredibly",
    "absolutely",
    "definitely",
    "certainly",
    "undoubtedly",
    "undeniably",
    "quite",
    "rather",
    "fairly",
    "somewhat",
    "relatively",
]


class SentimentEngine:
    key = "sentiment"
    name = "Sentiment & Hedging"
    engine_type = "linguistic"

    def score(self, text: str) -> float:
        text_lower = text.lower()
        word_count = max(len(text.split()), 1)

        # 1. Hedging phrases (plain substring counts, no word boundaries)
        hedge_count = 0
        for phrase in HEDGING_PHRASES:
            hedge_count += text_lower.count(phrase)
        hedge_density = (hedge_count / word_count) * 1000
        hedge_score = min(hedge_density / 4.0, 1.0)

        # 2. Balance patterns
        balance_count = 0
        for pattern in BALANCE_PATTERNS:
            balance_count += len(re.findall(pattern, text_lower, re.DOTALL))
        balance_score = min(balance_count / 3.0, 1.0)

        # 3. Vague positivity
        vague_count = 0
        for phrase in VAGUE_POSITIVITY:
            vague_count += text_lower.count(phrase)
        vague_density = (vague_count / word_count) * 1000
        vague_score = min(vague_density / 3.0, 1.0)

        # 4. Qualifier overuse
        qualifier_count = 0
        for word in FILLER_QUALIFIERS:
            qualifier_count += len(re.findall(r"\b" + word + r"\b", text_lower))
        qualifier_density = (qualifier_count / word_count) * 1000
        qualifier_score = max(0, min((qualifier_density - 8) / 15.0, 1.0))

        # 5. Explanatory "this/these" pattern
        explanatory_verbs = len(
            re.findall(
                r"\bThis (?:means|ensures|allows|enables|helps|provides|creates|offers|makes|includes|involves|requires)\b",
                text,
            )
        )
        explanatory_nouns = len(
            re.findall(
                r"\b[Tt]his (?:approach|method|process|tool|feature|system|framework|strategy|solution)\b",
                text,
            )
        )
        explanatory_these = len(
            re.findall(
                r"\b[Tt]hese (?:tools|features|systems|methods|approaches|strategies|solutions|techniques|elements|factors|components|resources)\b",
                text,
            )
        )
        explanatory_count = explanatory_verbs + explanatory_nouns + explanatory_these
        explanatory_score = min(explanatory_count / 4.0, 1.0)

        return (
            hedge_score * 0.30
            + balance_score * 0.15
            + vague_score * 0.20
            + qualifier_score * 0.15
            + explanatory_score * 0.20
        )


# SlopTotal's "Tier E" order in app/analyzer.py's engine list.
HEURISTIC_ENGINES: list = [
    LinguisticEngine(),
    StructuralEngine(),
    VocabularyEngine(),
    FormulaicEngine(),
    ReadabilityEngine(),
    SentimentEngine(),
]
