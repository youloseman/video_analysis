# Parts compatibility data

One file per frame GENERATION in `frames/<id>.json`. Read by
`app/services/parts.py`, checked by `scripts/check_parts_db.py` (schema, age,
dead links) and by `tests/test_parts.py`. Plan: `docs/AERO_ADVISOR_PLAN_RU.md`,
stage 3.

The question a file answers: *"I ride this frame, this generation, this size.
I want my pads further forward / higher / tilted. Will the stock adjustment
get me there, and if not, which parts fit this frame?"*

## Rules

- **No source, no record.** Every part carries at least one evidence URL. A
  part we only suspect fits is not listed.
- **Status says how sure we are**, and the app shows it:
  - `manufacturer` — the part maker or the frame maker says it fits this
    generation (product page, fitting guide, compatibility list).
  - `owners` — two or more independent owners/fitters report it fitted.
  - `single_report` — one owner/fitter report.
  - `incompatible` — documented as NOT fitting, with the reason. Worth as much
    as a fit: it stops a wasted purchase.
- **Paraphrase** forum posts and reviews; never copy more than a short phrase.
- **Prices** carry the currency and the day checked; `null` when not found.
  Manufacturer pages only for the `url`; no shops, no affiliate links.
- **Generations are separate files.** A part that fits two generations is
  listed in both files, each with its own evidence.
- **Pad reference point** is stated with every pad stack/reach number
  (`back_of_pad`, `pad_center`, `elbow_cup_center`, `unspecified`) — back vs
  centre differs 30–60 mm.
- Anything not found goes into `gaps`, never into a guessed value.

## Shape

```json
{
  "id": "giant-trinity-advanced-pro-2016",
  "family": "giant-trinity",
  "generation": "Trinity Advanced Pro (rim brake, 2016–2024)",
  "years": [2016, 2024],
  "brakes": "rim",
  "identify": "How a rider tells this generation from the next one, in one or two lines.",
  "cockpit": {
    "type": "integrated_proprietary | proprietary_stem_standard_bars | standard",
    "steerer": "e.g. 1-1/8 standard / proprietary aero steerer",
    "stem": "what holds the bars; proprietary or standard; clamp size",
    "basebar_clamp_mm": 31.8,
    "extension_clamp_mm": 22.2,
    "pad_bolt_spacing": "e.g. 15/30 mm two-bolt pattern, or null",
    "stack_system": "how pad height changes stock: riser kit, spacer sizes, max",
    "reach_system": "how pad reach changes stock: slots, flip, stems",
    "angle_system": "how extension tilt changes stock (wedges), or 'none'",
    "notes": "known limits fitters mention"
  },
  "stock_range": [
    {"size": "M", "pad_stack_mm": [600, 680], "pad_reach_mm": [380, 455],
     "pad_reference": "back_of_pad", "source": "URL"}
  ],
  "parts": [
    {
      "id": "fasttt-trinity-riser-adapter",
      "brand": "FastTT",
      "model": "Trinity riser adapter plate",
      "category": "adapter_plate | riser | spacer_kit | extensions | armrests | wedge | stem | basebar | full_cockpit | other",
      "effect": {"reach_mm": [0, 35], "stack_mm": null, "angle_deg": null,
                 "text": "Moves the armrest bridge ~35 mm forward."},
      "status": "manufacturer | owners | single_report | incompatible",
      "reason_incompatible": null,
      "evidence": [
        {"url": "https://...", "kind": "manufacturer | fitter | forum | review | retailer",
         "says": "short paraphrase of what the source states"}
      ],
      "url": "manufacturer product page or null",
      "price": {"amount": 249, "currency": "USD", "checked": "2026-11-16"},
      "needs": "what else the install needs (bolts, cable reroute, cutting), or null",
      "checked": "2026-11-16"
    }
  ],
  "gaps": ["what could not be found or verified"],
  "checked": "2026-11-16"
}
```

`effect` ranges are signed millimetres relative to the stock part they
replace or add to: reach + = pads further forward, stack + = pads higher,
angle + = extension tips higher. `null` = the part does not change that axis.
