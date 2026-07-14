# NotebookLM prompt — Bike aero position (JSON extraction)

Скопируй **весь блок между линиями** в NotebookLM (после того как загрузил книги
по велопосадке/аэродинамике). Он вернёт один JSON — сохрани его как
`bike-aero-position.json` и передай мне. Я обработаю его и напишу статью +
интерактивный калькулятор.

Книги для этого notebook: *Bike Fit* (Phil Burt), *Faster* (Michael
Hutchinson), *The Midlife Cyclist* (Phil Cavell), опц. *Triathlon Science*.

---

You are a research-extraction assistant. Using ONLY the sources loaded in this
notebook, extract everything relevant to **cyclist body position and
aerodynamic drag** (torso/back angle, position type — upright vs hoods vs drops
vs aero bars/TT, head and arm position, CdA, frontal area, the share of total
drag caused by the rider's body, and the comfort/power vs aero trade-off).

Return a SINGLE valid JSON object and NOTHING else — no prose before or after,
no markdown code fences. Use this exact schema. Follow every rule below it.

{
  "topic": "cycling aerodynamic position",
  "key_facts": [
    {
      "claim": "one-sentence factual statement, self-contained",
      "value": "the number/range if any, e.g. '70-80%' or '0.30 m^2', else null",
      "unit": "e.g. 'percent', 'm^2 (CdA)', 'watts', 'km/h', else null",
      "context": "conditions the fact holds under, e.g. 'at 40 km/h on flat road'",
      "has_citation": true,
      "source": {
        "author": "study author(s) as the book cites them, else the book author",
        "year": 2011,
        "title": "study or book title",
        "type": "study | book | review",
        "where": "book name + chapter/page if the fact is from a book quoting a study"
      },
      "confidence": "high | medium | low",
      "note": "optional — e.g. 'rule of thumb, no primary study cited' "
    }
  ],
  "position_data": [
    {
      "position": "upright | hoods | drops | aero_bars | tt_tuck",
      "cda_m2": 0.00,
      "drag_relative_pct": 0,
      "power_at_40kmh_watts": 0,
      "context": "e.g. '75 kg rider, 40 km/h, flat, no wind'",
      "has_citation": true,
      "source": { "author": "", "year": 0, "title": "", "type": "", "where": "" }
    }
  ],
  "mechanism": "2-4 sentences, plain English: why body position dominates drag at road speed (drag ~ speed squared, rider's body is most of frontal area, CdA explained simply).",
  "tradeoffs": "2-4 sentences: why the most aero position on paper can be slower in reality (sustainable power, breathing, fit, handling); how triathletes differ from road racers.",
  "practical_rules": [
    "actionable rule a recreational athlete can apply, grounded in the sources"
  ],
  "myths": [
    { "myth": "commonly repeated but weak/false claim", "reality": "what the evidence actually says", "has_citation": true }
  ],
  "quotable_numbers": [
    "short, striking, cited numbers good for a pull-quote or stat, e.g. 'The rider is ~80% of aerodynamic drag at 40 km/h'"
  ],
  "gaps": [
    "things the sources do NOT establish, or where they disagree"
  ]
}

RULES — follow all:

1. Use ONLY facts supported by the loaded sources. Do NOT add outside knowledge.
2. NEVER invent a citation. If a fact has no clear source in the material, still
   include it but set "has_citation": false, "confidence": "low", and explain in
   "note" (e.g. "coaching rule of thumb, no study cited"). Do not drop such facts
   — flag them.
3. For "position_data": fill a row ONLY if the sources give a real CdA, a
   relative-drag percentage, or a power figure for that position. Leave a numeric
   field null if the sources don't give it. Do NOT fabricate CdA numbers to fill
   the table — an incomplete but honest table is the goal.
4. Prefer PRIMARY studies the books cite over the books' own summaries. When a
   book quotes a study, put the study in "author/year/title/type" and the book in
   "where".
5. Numbers must carry their conditions in "context" (speed, rider mass, wind).
   A CdA or wattage without conditions is useless — include the conditions or
   set the value to null.
6. Keep every "claim" self-contained and one sentence. No pronouns referring to
   earlier items.
7. Output MUST be valid parseable JSON. Escape quotes. No trailing commas. No
   text outside the JSON object.

If the sources are thin on measured CdA values, say so honestly in "gaps" rather
than inventing numbers.
