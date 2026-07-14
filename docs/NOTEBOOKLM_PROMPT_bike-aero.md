# NotebookLM prompt — Bike aero position (JSON extraction)

Загрузи книги по велопосадке/аэро (*Bike Fit* — Phil Burt, *Faster* — Michael
Hutchinson, *The Midlife Cyclist* — Phil Cavell, опц. *Triathlon Science*),
затем отправь **COMPACT** промпт ниже. Сохрани ответ как
`bike-aero-position.json` и передай мне.

Если и COMPACT не влезает — используй **ULTRA-SHORT** (в самом низу): отправь
его, потом вторым сообщением отправь «Now output the JSON.»

---

## COMPACT (основной)

```
Using ONLY the loaded sources, extract everything about cyclist BODY POSITION and
AERODYNAMIC DRAG (torso/back angle; position type: upright/hoods/drops/aero_bars/
tt_tuck; head & arm position; CdA; frontal area; % of total drag from the rider's
body; comfort/power vs aero trade-off).

Output ONE valid JSON object, nothing else (no prose, no code fences). Schema:

{
 "topic":"cycling aerodynamic position",
 "key_facts":[{"claim":"","value":"num/range or null","unit":"or null","context":"speed/mass/wind conditions","has_citation":true,"source":{"author":"","year":0,"title":"","type":"study|book|review","where":"book+chapter if quoting a study"},"confidence":"high|medium|low","note":""}],
 "position_data":[{"position":"upright|hoods|drops|aero_bars|tt_tuck","cda_m2":null,"drag_relative_pct":null,"power_at_40kmh_watts":null,"context":"","has_citation":true,"source":{"author":"","year":0,"title":"","type":"","where":""}}],
 "mechanism":"2-4 sentences: why body position dominates drag at road speed (drag ~ speed^2; rider = most of frontal area; CdA explained simply)",
 "tradeoffs":"2-4 sentences: why the most aero position can be slower in reality (power, breathing, fit, handling); road vs triathlon",
 "practical_rules":[""],
 "myths":[{"myth":"","reality":"","has_citation":true}],
 "quotable_numbers":[""],
 "gaps":[""]
}

RULES:
1. Only facts from the loaded sources. No outside knowledge.
2. Never invent a citation. A fact with no source: still include it, set has_citation=false, confidence=low, explain in note. Don't drop it — flag it.
3. position_data: fill a row only if the sources give a real CdA / drag % / power for that position. Leave numeric fields null otherwise. Never fabricate CdA to fill the table.
4. Prefer the PRIMARY study a book cites over the book's own summary (study in author/year/title, book in "where").
5. Every number needs its conditions in "context" (speed, mass, wind) or set value null.
6. Each "claim" = one self-contained sentence.
7. Valid parseable JSON only: escaped quotes, no trailing commas, no text outside the object.
8. If sources lack measured CdA, say so in "gaps" instead of inventing numbers.
```

---

## ULTRA-SHORT (запасной, если COMPACT не влезает)

Отправь это, дождись, потом вторым сообщением: **`Now output the JSON.`**

```
From the loaded sources only, extract facts on cyclist body position & aero drag
(back/torso angle, position types upright/hoods/drops/aero_bars/tt_tuck, CdA,
frontal area, % of drag from the body, comfort/power vs aero). For each fact give:
claim, value+unit, conditions (speed/mass/wind), source (author/year/title; if a
book quotes a study, cite the study + name the book), and has_citation true/false.
Never invent a citation — flag unsourced facts as has_citation=false. Also give:
per-position CdA/drag%/power numbers ONLY where the sources state them (never
fabricate), the mechanism, the aero-vs-power tradeoffs, practical rules, common
myths+reality, quotable cited numbers, and gaps where sources are silent/disagree.
I will then ask you to output all of this as one JSON object.
```
```
```
