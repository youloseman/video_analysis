"""What a chat question is about, in one word, for counting.

Artur wants to know what people ask the report chat most (2026-11-22). The
text stays in our own table (chat_messages) and on the admin screen; what
goes to PostHog is this label, so the analytics service never holds an
athlete's words -- a question can carry an injury, a name, a race.

Keyword rules, English and Russian, first match wins in the order below: the
specific topics before the broad ones ("my knee hurts" is pain, not knee).
A question that matches nothing is "other"; the admin screen lists the
recent ones, which is how a missing topic gets noticed.
"""

from __future__ import annotations

import re

# Russian letters as a class, for patterns that must not match inside a
# longer word ("боль" sits inside "больше").
_RU = "а-яё"

TOPICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pain", ("pain", "hurt", "injur", "numb", "sore", "ache", "tingl",
              f"re:(?<![{_RU}])бол(ь|и|ит|ят|ел[{_RU}]*|езн[{_RU}]*)(?![{_RU}])",
              "травм", "онем", "немеет")),
    ("filming", ("film", "camera", "record", "video", "phone", "сним", "камер", "видео", "съёмк", "съемк")),
    ("cda_test", ("cda", "test", "tunnel", "notio", "aerolab", "тест", "трубе")),
    ("progress", ("last clip", "previous", "before", "since", "improv", "progress", "changed", "compare",
                  "прогресс", "улучш", "прошл", "сравн", "раньше")),
    ("parts", ("extension", "pads", "riser", "wedge", "crank", "cleat", "shoe", "helmet", "part", "buy",
               "шатун", "шип", "туфл", "шлем", "купить", "запчаст", "удлинит")),
    ("saddle", ("saddle", "seat", "setback", "седл", "сидень")),
    ("cockpit", ("stem", "bar", "reach", "stack", "spacer", "cockpit", "вынос", "руль", "проставк", "кокпит")),
    ("aero_time", ("aero", "drag", "faster", "speed", "time", "minute", "second", "watt",
                   "аэро", "быстр", "скорост", "врем", "минут", "ватт")),
    ("drills", ("drill", "exercise", "stretch", "mobility", "strength", "train",
                "упражн", "растяж", "мобильн", "трениров", "силов")),
    ("knee", ("knee", "колен")),
    ("hip", ("hip", "pelvi", "бедр", "таз")),
    ("trunk", ("trunk", "torso", "back", "lean", "корпус", "спин", "наклон")),
    ("arms", ("elbow", "arm", "shoulder", "forearm", "hand", "wrist", "локот", "плеч", "рук", "кист", "предплеч")),
    ("head", ("head", "neck", "look", "голов", "шея", "шею", "шеи")),
    ("run_form", ("cadence", "stride", "overstrid", "landing", "foot strike", "heel", "contact", "bounce", "oscillat",
                  "каденс", "шаг", "приземл", "пятк", "контакт")),
    ("score", ("score", "grade", "rating", "оценк", "балл", "грейд")),
    ("first_fix", ("first", "priorit", "most important", "сначала", "приоритет", "главн")),
)


def _match(word: str, text: str) -> bool:
    # "re:" -- a pattern. A short English stem must start a word ("arm" not
    # in "warm"); other stems match anywhere.
    if word.startswith("re:"):
        return re.search(word[3:], text) is not None
    if re.fullmatch(r"[a-z]{2,4}", word):
        return re.search(r"\b" + word, text) is not None
    return word in text


def classify(text: str) -> str:
    t = (text or "").lower()
    for topic, words in TOPICS:
        if any(_match(w, t) for w in words):
            return topic
    return "other"


def language(text: str) -> str:
    """"ru" when the question is mostly Cyrillic, else "en" (a hint, not a detector)."""
    letters = [c for c in text or "" if c.isalpha()]
    if not letters:
        return "en"
    cyr = sum(1 for c in letters if "Ѐ" <= c <= "ӿ")
    return "ru" if cyr / len(letters) > 0.5 else "en"
