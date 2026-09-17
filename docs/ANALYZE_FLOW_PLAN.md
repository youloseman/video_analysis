# Analyze page — flow transformation plan

Status: Phases 1–3 SHIPPED (17 Sep 2026). Open: the both-sides figure for the step-03 tile.

## Why the current page fails

The five steps are numbered, but the numbers lie about the order in which a
decision actually has to be made.

| now | step | what is wrong |
|---|---|---|
| 01 | Video / Photo | Fine on its own — but Photo is "bike fit only", so 01 and 02 are one decision split in two, and the invalid combination (Photo + Run) has to be hidden by code rather than by the design. |
| 02 | Sport | See above. |
| 03 | Your clip | The heaviest action (upload / record) comes BEFORE two choices that change what a clip means. |
| 04 | Position (bike) | A choice about the ride, asked after the ride has already been uploaded. |
| 05 | Details → Camera side → **Both sides** | A session type disguised as a fold inside "all optional" details. Choosing it reaches back up the page and rebuilds step 03 into two slots — the clip just loaded becomes "Left", a second card appears above the fold, and the athlete is told to upload again. Nobody discovers it; those who do think it broke. |

Two structural mistakes: **decisions after the action**, and **a mode switch
hidden in an options fold**.

## The rule for the new page

1. Every decision that changes what the footage *is* comes before the footage.
2. The footage step takes its shape from the decisions above it — one slot or
   two — and never changes shape after a file is in it.
3. Optional details stay optional: folded, after the footage, with the summary
   line stating what is set.
4. Nothing moves that the athlete has already filled.

## The new steps

```
01  WHAT ARE WE ANALYSING?          one choice, three tiles
    [ Run · video ]  [ Bike · video ]  [ Bike · photo — static fit ]

02  POSITION                        bike only (video and photo); hidden for run
    Road | Aero | Upright  →  Hoods · Drops / Time Trial · Triathlon / Casual
    (unchanged two-level control)

03  HOW WAS IT FILMED?              two tiles, each with a figure and one sentence
    [ One side ]                    "Camera on one side of you. We measure the leg
                                     nearest the camera."      · sub-control (bike only):
                                     Auto-detect | My left | My right
    [ Both sides — two clips ]      "Same ride/run filmed from each side. Merged into
                                     one score and a left-vs-right answer.
                                     Spends 2 of your monthly analyses."
    Photo: this step is hidden (a still has one side).

04  YOUR FOOTAGE                    shape follows 03, fixed once a file is in
    one side  → drop zone + Record with camera (as today)
    both      → two cards side by side: LEFT · RIGHT, each with its own
                drop/browse (Record per card: phase 2), "1 of 2 clips" line
    photo     → drop zone (image), no camera tile
    + pre-flight notes (duration, orientation)
    + the sticky Analyze bar ("Analyze" / "Analyze both clips")

05  DETAILS                         all optional, all folded, summaries state the value
    Setup & body (which bike/shoes; height for run) · Render overlay video ·
    Off-bike mobility (bike) · "Want us to look at something specific?"
```

Numerals keep the current behaviour: filled with the speed gradient once the
step's choice is in place, outlined while waiting. With defaults, a fresh
screen shows 01, 02, 03 filled and 04 as the one outline that matters.

### What "Both sides" looks like when chosen

```
03  HOW WAS IT FILMED?
    ┌───────────────────────┐ ┌─────────────────────────────┐
    │ ◉ One side            │ │ ○ Both sides — two clips     │
    │  [figure: one phone]  │ │  [figure: two phones]        │
    │  Camera on one side…  │ │  Same ride, each side, one   │
    │  Auto | Left | Right  │ │  merged score. Spends 2.     │
    └───────────────────────┘ └─────────────────────────────┘

04  YOUR FOOTAGE   ·   1 of 2 clips
    ┌── LEFT side ──────────┐ ┌── RIGHT side ─────────────┐
    │ [thumb]  IMG_9981.MOV │ │  ⬆ Drop or click to browse │
    │  Change clip          │ │  Your RIGHT side faces the │
    └───────────────────────┘ │  camera                    │
                              └────────────────────────────┘
    [ Analyze both clips ]  Add the right-side clip to run the session
```

Switching 03 after a file is loaded: the loaded file stays in the LEFT card
(both → one) or becomes the single clip (one → both keeps it as LEFT). No
re-upload, ever.

## State model (unchanged underneath)

`state.mode` (`video|photo`), `state.sport` (`run|bike`), `state.position`,
`state.cameraSide` (`auto|left|right|both`), `state.file`, `state.fileB` stay
exactly as they are; `isPair()`, `syncPairSlot()`, `renderPairCards()`,
`analyze()` / `analyzePair()` / the photo path keep working. The new tiles are
new *inputs* to the same state:

| tile | sets |
|---|---|
| Run · video | mode=video, sport=run |
| Bike · video | mode=video, sport=bike |
| Bike · photo | mode=photo, sport=bike, cameraSide=auto |
| One side | cameraSide = sub-control value (auto/left/right) |
| Both sides | cameraSide=both |

No backend change: `/analyze`, `/analyze-pair` (with `sport`) and
`/analyze-photo` already exist.

## Copy

- 01 lede: "One clip from the side, or one photo for a static bike fit."
- 03 lede: "A side view measures the leg nearest the camera. Film both sides
  if you want the whole athlete."
- Both-sides tile, second line: "Uses 2 of your monthly analyses." (quota is
  said where it is spent, before the upload — not in the hint after).
- Analyze button in pair mode: "Analyze both clips"; hint: "Add the
  right-side clip to run the session" / "Both clips loaded".
- Free plan on Both sides: the tile stays clickable and the quota line under
  the bar says "2 of N left this month" so the cost is visible before upload.

## Figures

Step 03 tiles carry two small figures in the "How to film it" line style:
one phone from the side (exists: `film-side.webp`) and two phones facing each
other (new: one Nano Banana prompt, same spec as Prompt 16 — "two phones in
portrait, left and right, a small runner between them facing sideways").
Until it is drawn, the tile uses the icon set.

## Tests that pin the current shape (and what changes)

- `test_uploader_invariants::test_the_camera_side_control_is_not_inside_the_bike_only_block`
  — intent kept (the run session must be reachable); `#sideField` moves from
  a `<details>` in 05 to tiles in 03; the assertion changes from "is a
  details with `#sideNow`" to "is outside `#positionField` and offers
  `data-side="both"` for run".
- `test_the_folded_controls_state_what_they_are_set_to` — `#optBox`
  (profile + height) unchanged; the camera-side half of the test is replaced
  by "the One-side tile shows its sub-choice".
- `test_camera_spa` — `#camOpen` outside the drop label: unchanged.
- `test_report_structure`, `test_free_limit_copy`, `test_design_rules` — untouched.

## Phases

**Phase 1 — the flow (one commit).** Markup reorder, the three 01 tiles, the
two 03 tiles with the side sub-control, footage step fixed in shape, details
folded last, step numerals, copy, tests updated, desktop + phone screenshots
of: fresh screen, bike + one side + clip, run + both sides with 1 and 2 clips,
photo. Estimated at one working session.

**Phase 2 — record per side.** SHIPPED: "Record this side" inside each empty
LEFT/RIGHT card; `openCam(slot)` aims the recorder, `camKeep` lands the clip
in that card through the same setFile/setFileB path a picked file takes.

**Phase 3 — measure it.** SHIPPED: `analyze_step_done` with `step` = what /
position / film / clip (`value`, `slot`, `source:'record'` when filmed
here); the pair session now sends `analysis_started` with `session:'pair'`
like the single-clip path, so the funnel counts it.

## Not doing

- Wizard with Next/Back buttons — the page stays one scroll; the sticky bar
  is the only button.
- Auto-advancing or collapsing completed steps — every choice stays visible
  and editable.
- Moving Details above the footage — optional things do not get to sit
  between the decision and the action.
