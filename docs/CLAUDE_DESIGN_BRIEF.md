# Flapp — бриф и промпты для редизайна в Claude Design

Цель: абстрагироваться от текущего UI, нарисовать приложение с нуля в Claude Design,
сравнить с тем что есть, выбрать.

Файл состоит из трёх частей:

1. **Как работать** (модель, порядок, что вставлять куда) — по-русски.
2. **CONTEXT** — блок на английском, вставляется первым сообщением (или в инструкции проекта).
3. **PROMPTS 1–12** — пошаговые промпты на английском, каждый — отдельное сообщение.

Промпты на английском намеренно: продукт англоязычный, все тексты в макетах должны быть
английскими, и модель даёт заметно более «продуктовый» результат, когда весь контекст на
одном языке с интерфейсом.

---

## 1. Как работать

### Модель

- **Основная — Claude Opus 5.** Дизайн-система и экран отчёта требуют вкуса и удержания
  десятка ограничений одновременно (одна CTA, три состояния доступа, честность,
  контраст) — это задачи для самой сильной модели, которую даёт Claude Design.
- **Если в пикере есть Claude Fable 5.1** — используй её для Prompt 1 (направления)
  и Prompt 5 (отчёт): это два самых тяжёлых шага. Остальные — Opus 5.
- **Sonnet 5** — только для быстрых вариаций уже утверждённого экрана
  («сделай три версии этой карточки»). Не для системы и не для отчёта.
- Не бери Haiku — там теряется именно то, ради чего затевается редизайн.

Что бы ни было выбрано — держи **одну модель на весь прогон** от Prompt 2 до Prompt 11,
иначе токены и стиль поплывут между экранами.

### Порядок

1. Новый проект/чат в Claude Design. Первым сообщением — блок **CONTEXT** целиком
   **+ четыре PNG из `docs/design-assets/`** (иллюстрации «How to film it» — они
   утверждены и остаются как есть, см. раздел *Keep as-is* в CONTEXT).
   Если есть «project instructions» — контекст туда, тогда он не съедает окно чата.
2. **Prompt 1** (три направления) — здесь ты принимаешь главное решение. Выбери одно,
   напиши «Go with direction B» и переходи дальше. Не смешивай направления.
3. **Prompt 2** (дизайн-система) — это фундамент; проверь его руками (см. чек-лист)
   до того, как идти к экранам. Ошибка тут размножится на все 10 экранов.
4. **Prompts 3–10** по одному, в этом же чате. После каждого — правки в 1–2 сообщения,
   не больше. Если экран не получается с третьей попытки — это сигнал, что проблема
   в системе (Prompt 2), а не в экране.
5. **Prompt 11** (мобильные) и **Prompt 12** (handoff) — в конце.
6. Сравнение с текущим: скриншоты прода (getflapp.com/app) рядом с артбордами,
   по экранам из таблицы в Prompt 12.

### Два трека (рекомендация)

Аудиты августа и сентября независимо пришли к одному: **лендинг и айдентика
конкурентоспособны, слабые места — экраны приложения** (первый экран, длинный отчёт,
плотность). Поэтому честное сравнение — два прогона:

- **Track A — «с чистого листа».** CONTEXT как есть, без упоминания текущих цветов
  и шрифтов. Это то, что ты просил.
- **Track B — «структура новая, бренд старый».** Тот же CONTEXT + абзац
  *Brand constraint* (внизу CONTEXT, закомментирован). Даёт новую структуру
  экранов без потери узнаваемости лендинга.

Запусти A первым. Если понравятся структура и плотность, но не цвет — B даст
ответ за один прогон, а не за переделку всего.

### Чек-лист приёмки Prompt 2 (перед экранами)

- [ ] Ровно один «filled» primary button в системе; secondary — outline; третий уровень — text.
- [ ] Шкала статусов **зелёный → янтарный → красный** (grade A→F, in range / check / out of range).
  Если модель нарисовала «синий = хорошо» — исправить сразу.
- [ ] Компонент **range bar** (значение на шкале с зелёной зоной) — существует и читается на 120px ширины.
- [ ] Компонент **score badge** с четырьмя состояниями: full / preview / teaser / withheld.
- [ ] Контраст текста на **тонированных** поверхностях проверен, не только на белом.
- [ ] Есть dark theme токены (даже если экраны рисуются в light).

---

## 2. CONTEXT (вставить первым сообщением)

```
You are designing the web app UI for Flapp (getflapp.com) from scratch. Treat this brief as
the complete product spec. Ignore any current visual style — we are exploring, not iterating.

## What Flapp is

Flapp is a running & cycling technique analysis app. An athlete films themselves from the
side with a phone (5–15 s), uploads the clip, and gets back a biomechanical report: joint
angles frame by frame from pose estimation, a 0–100 technique score with a letter grade,
what is inside or outside physiological ranges, an AI coach reading, corrective drills, and
(for cycling) bike-fit tradeoffs and an aerodynamic position estimate. It also accepts a
single side-view PHOTO for cycling (static bike fit).

It is the only consumer product that does BOTH running and cycling in one app — the target
user is the triathlete / serious amateur endurance athlete who cannot afford a $150–300
in-person gait or bike-fit session every time they change something, but wants real numbers.

Positioning line: "The triathlete's honest technique lab."
The word that matters most is HONEST — see the Honesty rules below.

## Who uses it

- Age-group triathletes and cyclists/runners, 25–55, data-literate (they know what cadence,
  ground contact time, knee angle at bottom dead centre mean). They own a Garmin, use
  Strava/TrainingPeaks/Intervals.icu. They tolerate density; they hate fluff and gamification.
- They use it in two moments: (1) right after filming, on a phone, often outdoors or in a
  garage on the trainer; (2) later at a desk, comparing before/after a change (new shoes,
  saddle moved 5 mm, cadence drill for two weeks).
- Solo founder product (the founder is himself a triathlete and reads Expert Reviews
  personally). Tone: a knowledgeable training partner, not a brand and not a coach-bot.

## The analysis pipeline (what the UI has to show)

1. Upload (video ≤ ~200 MB / photo). Sport toggle: Run / Bike. For bike: position picker
   (Road hoods / Road drops / TT-aero / Upright-casual / Trainer). Camera side: Left / Right /
   Both (two clips, one per side, merged). Optional free-text "focus question"
   ("does my left knee collapse?"). Optional two mobility-screen photos (hip / hamstring
   floor tests) that gate "get lower" advice for cycling.
2. Processing: 20–90 s. Pose model → biomechanics → score. Stages: uploading → detecting
   pose → measuring → writing the coach reading. Show progress honestly; no fake percent.
3. Report. Sections in this ARGUMENT order (score first, evidence, then what to do):
   - Score card: 0–100 + grade (A–F), sport, date, footage thumbnail (annotated keyframe
     with skeleton), confidence, "what the score was built from" (coverage).
   - Findings: 2–5 detected issues, each quoting the measurement that fired
     ("Overstride 1.18× — foot lands 18% ahead of the hip").
   - Aero position estimate (bike only): relative CdA zone (1–5), from trunk angle.
   - Coach reading: 3–6 short paragraphs, plain English, ends with ONE thing to change.
   - Action plan: 2–4 drills, each with dose ("3 × 30 s strides, 2×/week"), a re-test date.
   - Fit tradeoffs (bike only): "raise saddle 5 mm → knee 141°→145°, hip opens 3°,
     aero worsens".
   - Kinogram (run only): five stride positions in one strip image (ALTIS style).
   - Key metric tiles WITH range bars (value on a bar, green zone marked, p5–p95 whiskers,
     artefact count pill).
   - Joint-angle table: joint, mean, min, max, range, status pill (in range / check / out),
     near-side vs far-side rows (far side folded behind a count).
   - Lab / advanced (folded): phase stability %, waveform similarity, leg-identity
     stability — with a sport-aware verdict line.
   - Export: printable PDF report; "AI export" (paste-into-ChatGPT text block); share card
     (the annotated frame + score + getflapp.com).
   A sticky rail lists the visible sections and the score.
4. Overlay video: the clip re-rendered with skeleton + angle chips, only on MATERIAL joints
   (the two headline joints + up to two out-of-range ones, never all of them). Slow-mo.

## Real metric names & realistic sample values (use these, never lorem ipsum)

RUN (sample: score 74 / B): cadence 168 spm (target 170–185) · ground contact 248 ms
(target < 250) · flight time 118 ms · vertical oscillation 8.1 cm (6–9) · trunk lean 7°
(5–10) · overstride 1.12× (< 1.10) · foot strike: heel (midfoot preferred) · knee angle at
contact 158° · knee opening speed 412 °/s · knee closing speed 388 °/s · phase stability
71 % · waveform similarity 0.83.

BIKE (sample: score 81 / A−, Road hoods): knee angle at BDC 141° (140–150) · hip angle
(min) 43° (40–55) · trunk angle 38° (road: 35–45) · elbow 155° (150–165) · shoulder 88°
(80–95) · pelvic ratio 1.05× · head alignment: neutral · aero zone 3 / 5 ("moderate —
trunk 38°; zone 2 needs ≈ 30° and a passed hip screen") · knee opening speed 305 °/s.

Trend series (Progress screen): 4–8 points per metric, dated, oldest → newest, with a delta
and direction ("−0.4 cm · improving").

## Access states (every report screen exists in FOUR states — design all four)

- FULL (paid): everything.
- PREVIEW (first analysis on a free account): score + findings + coach reading shown;
  metric table, drills, fit, kinogram, export locked. The lock says exactly what is behind it.
- TEASER (later free analyses): score + one keyframe with skeleton, angle NUMBERS masked,
  watermark "FLAPP · FREE"; everything else blurred with real layout underneath.
  One offer on the blurred area: "Unlock this report — $4, once, no card kept" +
  a secondary link to plans. Never more than ONE upsell per screen.
- WITHHELD (quality gate fired: bad framing, leg swap, too short): NO score anywhere.
  A calm explanation of what the footage lacked, a link to the capture guide, no upsell.

## Plans (USD)

- Starter $0 — 10 analyses / month, teaser output, 1 equipment profile.
- Enthusiast $9 / mo or $69 / yr — 30 / mo, full report, overlay video, history & trends,
  1 profile. "Most popular".
- Full $99 / yr — 120 / mo, 10 equipment profiles (every bike / pair of shoes), compare one
  setup against another, PDF, 1 Expert Review included ($39 value).
- Unlock this report — $4 once (offered on the report, NOT on the pricing page).
- Expert Review — $39 add-on, any plan: a real triathlete reads your analysis and names the
  single thing to change (verdict, what's already working, ranked priorities with drills,
  re-test date).

## App screens (the full inventory)

Signed-out: landing (out of scope here) · sign in / register / password reset.
Signed-in:
1. Analyze (the first screen — the TOOL, not a slogan): uploader with sport toggle,
   dropzone, bike position picker, camera side, focus question, capture guide (three
   drawn panels: fill the frame / phone upright / from the side), trust strip (privacy
   claims), quota line ("7 of 10 analyses left this month").
2. Processing state.
3. Report (video) and Report (photo, bike fit only) — all four access states.
4. Dashboard / "My technique" (signed-in home): summary stats (analyses total / this
   month / last), per-sport snapshot card with score sparkline, "current form" (last
   analysis' key numbers vs previous), focus areas (recurring findings), active plan
   progress, 4 recent analyses, Expert Review status card.
5. Progress / trends: sport segment, one trend card per metric (value, delta, mini line
   chart with the green range band); free users see the score trend real and the rest as
   locked previews with one CTA.
6. History: list of analyses (thumb, sport, score, date, profile, position), filters,
   detail view (= saved report), delete.
7. Compare: pick two analyses → before/after wipe slider on the annotated frames,
   metric-by-metric delta table with direction, "two setups" mode (profile A vs B).
8. Profiles: equipment profiles (bike: road/TT/trainer; run: shoes), limits per plan.
9. Pricing (cards above + Expert Review strip + monthly/yearly toggle; current plan badge).
10. Expert Review: request form (which analysis, question), status (queued / delivered),
    the delivered review page.
11. Account: email, plan, manage subscription, delete account.
12. Secondary: What's new (changelog with unread dot), Feedback strip, Examples gallery,
    Academy (articles, server-rendered, out of scope).

## Honesty rules (these are product rules, the UI must express them)

- The score says what it was built from. A guessed time base is not graded.
- When the quality gate fires there is no score anywhere — not on the card, not on the
  kinogram badge, not in the free-tier teaser.
- Confidence and coverage are visible, not hidden in a tooltip.
- "Nothing to fix" is a legitimate coach verdict and must look fine, not like an error.
- A locked section says precisely what is behind it; the $4 unlock lists only what that
  one report can deliver (no video, no trends — those are subscription features).
- Far-side (away from camera) angles are less reliable → visually secondary.
- Artefact frames (physically impossible angles) are counted and shown as a pill, not
  silently averaged.
- No points, streaks, badges, confetti, "running style" archetypes. No dark patterns.

## Interaction & platform constraints

- Web app, responsive: desktop 1366×768 is the primary desk size; phone 390×844 is the
  primary mobile size (the app is also wrapped as an iOS app via Capacitor, so mobile is
  not secondary). Design both.
- The report is long (4.5–5.5k px). Keep it scannable: sticky section rail, folds for
  lab/far-side, no accordions hiding the argument.
- Charts: small, quiet, one accent; range band drawn as a soft green zone; no 3D, no
  gradients on data. Sparkline for dashboard, line chart with band for progress,
  horizontal range bar for metric tiles.
- Motion: minimal. Colour-step hovers, 150 ms transitions, no lifts/shadows on hover.
- Accessibility: AA contrast on every surface INCLUDING tinted panels; 44 px touch targets;
  status never conveyed by colour alone (always a word or an icon too).
- One filled primary button per screen. One upsell per screen, max.

## Keep as-is: the "How to film it" capture guide

Attached are four 512×512 line illustrations (film-fill-run, film-fill-bike,
film-portrait, film-side). They are approved and stay in the product unchanged: two-tone
stroke drawings, no fills, no text inside the image. Use them verbatim in the Analyze
screen and anywhere the capture guide appears; do not redraw them. The visual system you
propose has to sit well next to them — that is a constraint on the system, not on the
figures. Their two colours (a dark ink and an accent blue) may be remapped to the new
palette's ink and accent tokens, nothing else.

The guide's copy is fixed too (three rules, in this order, each beside its figure):
1. "Fill the frame. Head to feet should span about two thirds of it. Sky above and ground
   below are what wreck the measurement — this rule is worth more than every other one
   here combined."  (figure: fill-run or fill-bike, by sport)
2. "Hold the phone upright. Portrait puts nearly twice as many pixels on your legs as
   landscape."  (figure: portrait)
3. "Film from the side, camera at hip height. A few degrees off exactly 90° actually helps
   separate the legs. Turning to follow you is fine — it's up-and-down shake that hurts,
   not panning."  (figure: side)
Below them a folded "A few more things that help" list of six one-liners (stand back
6–8 m and zoom 2–3×; non-drive side on the bike; 5–10 s with you in shot; a treadmill is
easiest; normal speed or slow-mo both work; light behind the camera).

## What NOT to do

- No slogan hero on the first app screen — the dropzone is the hero.
- No gamification, no motivational copy, no emoji in UI.
- No "AI" sparkle iconography everywhere; AI is one section, not the brand.
- Don't hide the score in a gauge you can't read at a glance. A number and a grade.
- Don't design the landing page — out of scope.

<!-- Brand constraint — Track B only. Uncomment to keep the existing identity:
## Brand constraint
Keep the existing identity: wordmark "Flapp" (lowercase, italic, Archivo Black-like),
primary electric blue #2F6DE0, ink navy #0B1B3A, accent coral #F1553F used ONLY for the
one "Go" action; fonts Archivo (display) / Manrope (text) / IBM Plex Mono (numbers).
Status scale stays green → amber → red. Everything else — layout, density, components,
charts — is free.
-->
```

---

## 3. PROMPTS (по одному сообщению)

### Prompt 1 — три направления

```
Before any screens: propose THREE distinct visual directions for the Flapp app, as three
artboards side by side. Each artboard = a moodboard-style sheet showing: the report score
card, one metric tile with range bar, a primary + secondary button, a status pill trio
(in range / check / out of range), a small trend chart, the nav, and a one-line rationale.

Directions to explore (make them genuinely different, not three shades of one idea):
A. "Instrument" — lab / measurement-tool feel: mono numerals, hairline rules, cool neutrals,
   one accent. Dense, precise, quiet.
B. "Field notebook" — warm paper neutrals, strong editorial typography, generous whitespace,
   data drawn like a coach's notebook; friendlier but still exact.
C. "Sport product" — high-contrast dark-first UI like a modern training app (dark ground,
   bright accent, big score), built for phone-in-hand use outdoors.

For each: palette (with hex), type pairing, radius/border logic, how status colours work
(green → amber → red is fixed — do not change the semantics), and what it would be bad at.
Pick nothing; I will choose.
```

После ответа: `Go with direction <X>. Keep everything else from the brief.`

### Prompt 2 — дизайн-система

```
Using the chosen direction, build the design system as one artboard ("Foundations") and one
artboard ("Components").

Foundations: colour tokens (surfaces incl. two tinted panels, ink scale, one accent, status
scale good/warn/bad with a TEXT variant of each that passes AA on its tinted background),
type scale (display / h1–h3 / body / small / mono-numeric), spacing scale, three radii,
two shadows max, dark-theme token values next to light.

Components (each with states — default / hover / active / disabled / focus where relevant):
- Buttons: primary filled (the only filled button), secondary outline, tertiary text,
  destructive; sizes md/sm; with icon.
- Score badge: number + grade, in FULL / PREVIEW / TEASER / WITHHELD states.
- Status pill: in range / check / out of range, and "artefacts: 3".
- Metric tile: label, value+unit, range bar (bar with green zone, marker, p5–p95 whiskers),
  delta vs previous, at 220 px and 160 px widths.
- Range bar alone at 120 px.
- Confidence / coverage indicator.
- Trend card with a mini line chart and range band.
- Sparkline.
- Locked section overlay (blur + one CTA + "what's behind this" list).
- Section rail (sticky) item, active/inactive.
- Table row with status pill; folded "far side" row group.
- Input, select, segmented control (Run / Bike), file dropzone (idle / drag-over / file
  chosen), toggle, toast, modal, empty state, error state.
- Nav: sidebar item (desktop), bottom bar item (mobile), quota line.
Label every token and component. Real copy from the brief, no lorem ipsum.
```

### Prompt 3 — Analyze (первый экран)

**Приложи к этому сообщению четыре PNG из `docs/design-assets/`** (film-fill-run,
film-fill-bike, film-portrait, film-side). Если в Claude Design можно приложить файлы
к первому сообщению (CONTEXT) — лучше туда, тогда они будут видны с самого начала.

```
Attached: the four approved capture-guide figures (see "Keep as-is" in the brief). Use them
unchanged; recolour strokes to the system's ink + accent tokens only.

Screen: Analyze — the signed-in first screen. Desktop 1366×768 AND phone 390×844.
The dropzone is the hero and must be fully above the fold on both.
Contents, top to bottom: sport segmented control (Run / Bike); dropzone (video or photo,
with a "record with camera" alternative on mobile); bike-only row: position picker (Road
hoods / Road drops / TT-aero / Upright / Trainer) — one row with inline group labels; camera
side (Left / Right / Both sides); a folded "Ask a focus question" field; folded "Mobility
screen (2 photos)"; the primary "Analyze" button; quota line "7 of 10 analyses left this
month"; trust strip (3 short privacy claims); capture guide as three small drawn panels
(fill the frame / phone upright / from the side) BELOW the button.
Show two variants: sport = Run (position row hidden) and sport = Bike (position row shown).
Also one state with a file chosen (preview thumb, filename, duration, a landscape-orientation
warning linking to the guide).
```

### Prompt 4 — Processing

```
Screen: Processing, desktop + phone. Stages: uploading (real %) → detecting pose →
measuring → writing the coach reading. Show elapsed time, what is happening now, and the
one thing the user can read meanwhile (a 2-line "while you wait" note about the metric we
are about to measure). No fake progress, no spinner-only. Include the failure state:
"We couldn't read this clip" with the specific reason (too short / athlete too small in
frame / two people) and a retry that goes back to Analyze with the guide open.
```

### Prompt 5 — Report (главный экран)

```
Screen: Report — RUN video, FULL access, desktop 1366 wide (long artboard, full height) and
phone 390 wide. Use the run sample values from the brief exactly.
Sections in argument order: score card (74 / B, confidence, coverage, annotated keyframe,
date, profile) → Findings (3 items quoting measurements) → Coach reading (4 short
paragraphs ending in ONE thing to change) → Action plan (3 drills with dose + re-test date)
→ Kinogram (5-frame strip) → Key metrics (tiles with range bars; the range-less tiles —
foot strike, cadence category — in a shorter second row) → Joint angles table (near side
rows; far side folded "4 far-side rows") → Lab (folded; phase stability 71 %, waveform
similarity 0.83, verdict line) → Export (PDF / AI export / share card) → per-result
micro-feedback (👍 👎 + one line).
Sticky section rail on the left or right with the score at top. Overlay video player at
the top with slow-mo control, poster = annotated keyframe.
Then a second artboard: the SAME report in BIKE variant (81 / A−, Road hoods) adding the
Aero position card (zone 3/5 with the "what zone 2 would need" line) and the Fit tradeoffs
block (3 tradeoffs), no kinogram.
```

### Prompt 6 — состояния доступа

```
Take the RUN report from Prompt 5 and produce three more artboards (desktop):
1. PREVIEW: score + findings + coach reading fully visible; action plan, kinogram, metric
   tiles, angle table, export are locked — each lock states what is behind it. One CTA to
   plans, no $4 here.
2. TEASER: score + one keyframe with skeleton, angle numbers masked ("[locked]"), watermark
   "FLAPP · FREE"; the rest blurred with the real layout underneath; ONE offer panel:
   "Unlock this report — $4, once, no card kept" listing exactly: every joint angle vs its
   range, the corrective drills, the kinogram, AI export; and greyed "not included: trends,
   before/after, overlay video (with a plan)"; a small secondary link "See plans".
3. WITHHELD: no score anywhere. Calm explanation (e.g. "The athlete filled less than 30 %
   of the frame, so stride timing could not be trusted"), what we could still show (pose
   detected, cadence estimate marked as unreliable), link to the capture guide, "Analyze
   another clip". No upsell.
Also a phone version of TEASER.
```

### Prompt 7 — Dashboard

```
Screen: Dashboard ("My technique"), desktop + phone. Signed-in home.
Blocks: welcome-less header (no greeting hero); summary row (12 analyses · 3 this month ·
last 2 days ago); two sport cards (Run: score 74, sparkline of last 6, "since last: −0.4 cm
oscillation"; Bike: 81, sparkline); "Current form" — last analysis' 4 key numbers vs the
previous one with deltas; "Focus areas" — 2 recurring findings across the last 5 analyses
with a link to the report that last showed them; "Active plan" — drill checklist progress
and re-test date; "Recent" — 4 analysis cards (thumb, sport, score, date, position);
Expert Review status card (queued / delivered / "1 credit available").
Also the empty state for a brand-new account (one CTA: Analyze a clip).
```

### Prompt 8 — Progress / trends

```
Screen: Progress, desktop + phone. Sport segment (Run / Bike). One trend card per metric
from the brief (run: score, cadence, vertical oscillation, trunk lean, ground contact,
flight time, overstride, knee angle, knee opening/closing speed; bike: score, knee, hip,
trunk, elbow, shoulder, pelvic ratio, knee speeds). Each card: label, latest value, delta
since first with direction word ("improving" / "drifting" / "stable"), N analyses, a line
chart with dated points and a soft green range band; hover state showing one point's value
and date. Categorical metric (foot strike) shown as changed/unchanged chips over time.
Profile filter (which bike / shoes). FREE state: score trend real, every other card a
locked preview, ONE CTA panel. Empty state: "One more to go — analyze 2 runs to unlock".
Charts: quiet, one accent, no gradients, legible at 160 px card width on phone.
```

### Prompt 9 — History и Compare

```
Two screens, desktop + phone.
History: list of analyses — thumbnail (annotated frame), sport icon, score badge, date,
position / profile, access state (full / preview / teaser / withheld) as a subtle mark;
filters by sport and profile; delete with confirm. Detail = the saved report (reuse
Prompt 5), with the $4 unlock panel when the saved entry is a teaser.
Compare: pick two analyses (pickers show thumb + date + score); before/after wipe slider on
the two annotated frames (aspect from footage); metric delta table (metric, before, after,
delta, direction); "Two setups" mode toggle where the pickers become Profile A / Profile B
(e.g. Road bike vs TT bike) and the table gains a "which setup is closer to range" column.
Compare summary line at top: "3 metrics improved, 1 drifted, 4 unchanged".
```

### Prompt 10 — Pricing, Expert Review, Account

```
Three screens, desktop + phone.
Pricing: three cards (Starter $0 · Enthusiast $9/mo or $69/yr "Most popular" · Full
$99/yr) with monthly/yearly toggle (Full is yearly-only), feature lists from the brief,
"Your current plan" badge on the user's tier, manage-subscription for paid; below, the
Expert Review add-on strip ($39, "included with Full") with a sample review excerpt. The $4
unlock is NOT on this page.
Expert Review: request form (choose analysis, one free-text question, what happens next,
turnaround), status page (queued / in review / delivered), and the delivered review as a
readable document: verdict → what's already working → ranked priorities with drills →
re-test date, signed by a person, with a "reply" affordance.
Account: email, plan + renewal date, manage subscription, equipment profiles (list with
limit "1 of 10"), data & privacy (export / delete account), sign out.
```

### Prompt 11 — мобильные версии

```
Consolidate mobile: one long artboard per screen at 390×844 for Analyze, Report (FULL, run),
Report (TEASER), Dashboard, Progress, History, Compare, Pricing — in the system's bottom-nav
pattern (Home / Analyze / Progress / History / More). Rules: 44 px targets, tables reflow to
cards (never clip a column), the report rail becomes a horizontal chip strip under the score,
the wipe slider is thumb-driven, charts stay legible at 160 px. Show the sticky "Analyze"
bar behaviour when a file is chosen.
```

### Prompt 12 — handoff

```
Produce a handoff artboard: the final token table (light + dark values), the component list
with names, and a screen inventory table with columns: screen · states designed · desktop
artboard · mobile artboard · notes. Then a short "what changed vs a typical technique app"
list — 8 bullets max, each one a design decision and why (e.g. "score is a number, not a
gauge, because…"). Plain, no marketing language.
```

### Prompt 13 — дополнение: применить «How to film it» к уже готовому дизайну

Для случая, когда прогон уже завершён без картинок. Отправляется в тот же чат,
**с четырьмя PNG из `docs/design-assets/` во вложении**.

```
Addendum to the finished design. Attached are four approved illustrations from the current
product — the "How to film it" capture guide: film-fill-run, film-fill-bike, film-portrait,
film-side (512×512, two-tone stroke drawings, no fills, no text inside). They are kept
as-is. Do NOT redraw, restyle, add fills, shadows, backgrounds or text to them. The only
change allowed: remap their two stroke colours (dark ink, accent blue) to this system's
ink token and accent token.

Apply them to the existing artboards:

1. Analyze screen (desktop + phone): add a "How to film it" section BELOW the Analyze
   button and the trust strip, using the system's section/eyebrow style. Layout: an
   ordered list of three steps, each = figure on the left (fixed 96–120 px on desktop,
   72–88 px on phone) + rule text on the right. Copy is fixed, keep it word for word:
     1. Fill the frame. Head to feet should span about two thirds of it. Sky above and
        ground below are what wreck the measurement — this rule is worth more than every
        other one here combined.   (figure: fill-run when sport = Run, fill-bike when Bike)
     2. Hold the phone upright. Portrait puts nearly twice as many pixels on your legs as
        landscape.   (figure: portrait)
     3. Film from the side, camera at hip height. A few degrees off exactly 90° actually
        helps separate the legs. Turning to follow you is fine — it's up-and-down shake
        that hurts, not panning.   (figure: side)
   Lede above the list: "Three things decide whether the numbers are worth reading. Don't
   overthink the rest — a phone propped against a water bottle works."
   Under the list a folded row "A few more things that help" with six one-liners (stand
   back 6–8 m and zoom 2–3×; on the bike film from the non-drive side; 5–10 s with you in
   shot, one person in frame; a treadmill is easiest; normal speed or slow-mo both work;
   light behind the camera, not behind you). Show it closed; one variant open.
   Show the section in both sport variants (Run / Bike) — only the first figure changes.

2. File-chosen state on Analyze: when the chosen clip is landscape, the warning line
   "Landscape clip — portrait puts twice the pixels on your legs" links to this section and
   the `portrait` figure gets a brief highlight ring (one 150 ms colour step, no motion).

3. Processing screen failure state ("We couldn't read this clip"): reuse the ONE figure
   that matches the reason — too small in frame → fill-run/fill-bike; wrong orientation →
   portrait; not from the side → side — next to the explanation, same size as in step 1.

4. Report WITHHELD state: the same single matching figure beside the explanation of what
   the footage lacked, with the link "See how to film it".

5. Add the three figures to the Components artboard as a component "Capture figure"
   (sizes 120 / 88 / 64) with the recolour rule written next to it, and to the handoff
   inventory as fixed assets (file names above).

Keep everything else in the design untouched. Output: updated Analyze (desktop + phone,
Run + Bike), Processing-failure, Report-withheld, Components, Handoff.
```

### Prompts 14a–14e — правки по ревью (2026-09-13, Track A / Instrument)

Ревью отрендеренного экспорта: `docs/design-export/track-a/png/`. Структура принята,
ниже только исправления. Пять отдельных сообщений в том же чате, **строго в этом
порядке**: 14a трогает каждый артборд (логотип, навигация, иконки), поэтому идёт первым —
иначе всё, что нарисуют 14b–14e, придётся перекрашивать.

После каждого промпта — правки в 1–2 сообщения, потом следующий.

#### Prompt 14a — бренд, навигация, иконки (все артборды)

```
Review fixes, part 1 of 5 — brand and navigation. The structure is approved; change
nothing but what is listed.

1. Wordmark. "FLAPP" in tracked caps is a label, not a logo. Restore the existing
   wordmark: "Flapp" lowercase, heavy italic grotesk (Archivo Black-like), followed by a
   6 px round dot in coral #F1553F. The dot is the only coral in the app — a brand mark,
   not a UI colour, outside the accent rule. Letters in ink/1. Apply on every artboard:
   sidebar, phone header, share card, PDF masthead.

2. Sidebar active item. The solid ink/1 block is the loudest thing on the Analyze screen.
   Two meanings are conflated — WHERE I AM (navigation) and WHAT I CHOSE (a control).
   Split them:
   - nav active = panel/accent tint #EDF2FD + ink/1 text + a 2 px accent bar on the left
     edge, no fill; hover = surface/sunken.
   - segmented controls (Run / Bike, Left / Right, Two analyses / Two setups) keep the
     ink/1 fill for the selected value.
   - mobile bottom bar: active = accent glyph + ink/1 label, no block.

3. Icons. Nav icons are identical placeholder squares everywhere. Draw a 16 px stroke icon
   set in the Instrument style (1.5 px, square caps): Analyze, My technique, Progress,
   History, Compare, Profiles, What's new, Account; status glyphs (● △ ✕); the sport pair
   (run / bike). Add them to Components and apply across all artboards.

Output: Components (wordmark, nav states, icon set) and every artboard that carries the
sidebar, the phone header or the bottom bar — re-issued with the new wordmark, nav
treatment and icons. Nothing else moves.
```

#### Prompt 14b — фактические правки текста

```
Review fixes, part 2 of 5 — copy the product cannot say. Fix the words only; no layout
changes.

1. Analyze (Bike), camera-side hint: "Drive side gives a cleaner knee track" contradicts
   the guide (film from the NON-drive side). Replace with: "Non-drive side keeps the
   chainring away from the ankle; both sides catches asymmetry."
2. Withheld (5c) and its "What to change" strip: "3–5 m" / "move the phone closer". The
   rule is: stand back 6–8 m and zoom in 2–3×. Fix both places.
3. Processing failure 3f-3: "camera ≈ 40° off perpendicular". We do not measure camera
   angle. Rewrite around what we do measure: "the near and far legs overlap for most of
   the stride — leg identity 0.42, unstable". No degrees.
4. Pricing: the sample review is signed "M. Halvorsen, reviewer since 2024". No invented
   people. Sign it "— Artur, founder · age-group triathlete" or leave it unsigned.
5. Score card: "Nike Vaporfly 3 — 412 km". We do not track shoe mileage. Drop the km.
6. Action plan: "Add to calendar" does not exist. Remove it; keep the re-test date line.
7. Dashboard summary: replace the filler tile "Sports tracked · Run & Bike" with
   "Re-test · in 27 days" (or the quota line when no plan is active).

Output: updated 3b, 3f-3, 5c, 4a (score card + action plan), 6a, 9a.
```

#### Prompt 14c — реальный кадр и тёмная тема

```
Review fixes, part 3 of 5 — prove the system on real footage and on a dark ground. Two
new artboards; nothing existing changes.

1. "Overlay on real footage". Every video frame so far is a hatched placeholder. Show the
   report's player and the score-card keyframe on a realistic photo-like background — a
   runner on an asphalt path in daylight, and a cyclist on a trainer in a garage — with
   the skeleton and 3 angle chips on top. Chip = solid card with a 1 px hairline, never a
   transparent label; skeleton stroke must read on both grass and asphalt. Show the
   kinogram strip the same way (five real-looking frames). Also the share card with a
   real frame.
2. "Report — dark". Render the RUN report (4a) once in the dark tokens from Foundations.
   Same layout, tokens only. The point is to check every status text/tint pair on a dark
   ground and the blue accent against surface/card dark.

Output: two artboards. If anything in the token set fails on the dark ground or on
footage, say so under the artboard and propose the token change — do not silently
adjust a screen.
```

#### Prompt 14d — недостающие экраны

```
Review fixes, part 4 of 5 — two screens the brief lists that are not in the set. Same
system, same components; desktop 1366 + phone 390 each.

1. Report — photo (bike fit only). Input is a single side-view photo, so: score card
   with the annotated photo as the hero (larger than the video keyframe), findings, coach
   reading, fit tradeoffs, aero zone, joint-angle table (one value per joint — no mean/
   min/max, no timing metrics, no kinogram, no phase stability). Access states: FULL and
   TEASER only.
2. Profiles. List of equipment profiles (bike: name, type road/TT/trainer, crank length,
   saddle height note; run: shoe name, model) with the plan limit "1 of 10" for Full and
   "1 of 1" for Starter/Enthusiast; add / edit / archive; which analyses use each
   profile; the upsell when the limit is reached (one panel, plan CTA).

Output: four artboards (two screens × two sizes).
```

#### Prompt 14e — мобильный отчёт и иерархия

```
Review fixes, part 5 of 5 — mobile report and entry points.

1. Mobile report (10b): draw the bottom nav at the bottom of the artboard, not floating
   mid-page; label it "sticky".
2. Mobile report: "Findings 3 · Coach · Plan 3 · Kinogram" — say whether this is a set of
   tabs or a horizontal chip strip of anchors. The brief asked for anchors: the whole
   report stays on one page and a chip scrolls to its section. Make it that and label it.
3. Hierarchy — one entry point per non-report screen. Analyze: the three capture figures
   at 120 px are it; keep them large on desktop. Dashboard: the two sport cards — score
   numeral 56 px, sparkline 160 px wide. Pricing: the price numerals. Section headings
   (h2) go one step up: 18 → 20 px everywhere.

Output: updated 10b, 3a/3b, 6a, 9a, and the type-scale row in Foundations.
```

### Prompt 15 — консолидация (после ревью v2, 2026-09-13)

Ревью `docs/design-export/track-a-v2/png/`: 14a, 14b, 14c, 14d выполнены; 14e — частично.
Остались: предложенные на 12a/12b токены не внесены в Foundations, Handoff не обновлён
вообще, два новых экрана принесли новые выдумки. Один промпт, тот же чат — последний
перед заморозкой.

```
Consolidation — the last pass before the design is frozen for implementation.

1. Fold the token proposals from "Overlay on real footage" and "Report — dark" into
   Foundations and Components. They are accepted as written:
   - overlay/halo #0E1116 @ 55 %, width = stroke + 4 (part of the skeleton, not optional)
   - shadow/on-footage 0 2 6 rgba(20,24,31,.25–.30), only over video
   - over video: chip border = ink/1 or the status line, never a hairline; joint dots
     carry a 1.5 px #0E1116 ring; no status chip over footage without glyph + border
   - status/good/band-dark #1B4A2C for range bands on dark (tint stays for pills)
   - primary button on dark: ink-inverse #0E1116 label at 600, never white
   - panel/quiet dropped on dark → surface/sunken #1F242C + top hairline
   - ink/4-dark #7C8698 for masked "[locked]" values
   Write them into the token tables with their contrast ratios, add an "over footage"
   row to the button/chip/skeleton components, and re-issue 12a/12b with the tokens
   applied so the artboards and the tables agree.

2. Photo report (13a) says "crank position: verified at BDC". A single photo cannot
   verify crank position — we ask for the photo at BDC and trust it. Replace with
   "crank at BDC: as photographed, not verified" and keep the ±2° caveat.

3. Profiles (13d) shows fields the product does not have: crank length, saddle-height
   note, default position. A profile is name · sport · kind (road / TT / trainer /
   gravel / MTB / shoes / other) · archived. Decide per field:
   - crank length — remove, and remove the sentence "used to check the knee angle at
     BDC"; no angle depends on it.
   - default position — keep as a proposed field, label it "proposed · not built".
   - saddle-height note — keep as proposed free text, same label.
   Also the "3 of 10 used" pill is a count, not a status — plain mono, no green.

4. Mobile report (10b): the bottom bar is still drawn mid-page. Move it to the bottom
   edge of the artboard; keep the "sticky" label.

5. Handoff (11a) is unchanged since v1. Rebuild it: final token tables (light + dark +
   over-footage), the icon set, the wordmark rule, nav states, the four capture figures
   as fixed assets, and the screen inventory including Photo report, Profiles, Overlay
   on footage, Report dark, and every access state. Add a "proposed, not built" list:
   default position, saddle note, camera-reason failure copy.

Output: Foundations, Components, 12a, 12b, 13a, 13d, 10b, Handoff.
```

### Prompt 16 — иллюстрации пустых состояний (шаг 5 «характера», 2026-09-13)

Отдельная задача в Claude Design. Приложи четыре PNG из `docs/design-assets/` как
образец стиля. Результат — PNG 512×512 в `backend/app/static/media/` (нужен ещё
webp-экспорт, как у четырёх существующих).

```
Five illustrations in the exact style of the four attached "How to film it"
figures: 512×512, two-tone stroke only (dark ink #14181F and accent blue
#2457C5), 12–14 px stroke at 512, round caps, no fills except the small green
tick / grey cross already used in film-portrait, no text, no gradients, no
shadows, generous empty space, subject centred. Each must read at 120 px and
still at 64 px. They are drawn for these empty and edge states of the app:

1. "No analyses yet" — a phone propped against a water bottle, filming: the
   phone upright, the bottle beside it, a small runner figure in the frame.
2. "Processing" — the same runner figure with the skeleton being drawn over
   it: three joints connected, two still open circles (the analysis mid-way).
3. "Score withheld" — the athlete too small in a large frame: the phone frame
   big, the figure tiny at the bottom, two thirds of the frame empty sky.
4. "Nothing to fix" — the runner figure with every joint a small green tick
   (the same tick as in film-portrait), calm, no drama.
5. "No trends yet" — two phone frames side by side with a dashed arrow
   between them, the second frame empty: "one more to go".

Deliver each as a single artboard with the figure alone on white, plus one
sheet showing all nine figures (the four existing and the five new) at 64 px
in a row to prove they read as one set. Name the files empty-first,
empty-processing, empty-withheld, empty-nothing-to-fix, empty-trends.
```

---

## 4. Как сравнивать (после прогона)

Скриншоты прода по тем же экранам (можно взять из сентябрьского аудита, там 46 комбинаций)
и рядом артборды. Критерии — те, по которым проседает текущий UI, а не «нравится / не
нравится»:

| Критерий | Текущий (сент. 2026) | Новый |
|---|---|---|
| Analyze выше фолда на 1366×768 и 390×844 | да (после wave C) | |
| Длина run-отчёта, px | ~5.2k desktop / ~8.2k mobile | |
| Понятно ли за 5 секунд, что заперто и за сколько | одна панель | |
| Range bar читается на телефоне | да | |
| Контраст на тонированных панелях (AA) | да, после трёх коммитов | |
| Один filled-button на экране | да (искл. unlock) | |
| Withheld-состояние без оценки | да | |
| Dark theme | нет | |

Если новый вариант выигрывает по структуре, но проигрывает по узнаваемости —
это ровно случай для Track B.
