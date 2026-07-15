# Advanced-биомеханика (бег и велопосадка) + визуализация диапазонов метрик

> Документ описывает две вещи:
>
> 1. **Как реализована «advanced»-биомеханика** для бега (`run`, side view) и
>    велопосадки (`bike`, side view): фазовые портреты, координация, симметрия,
>    сравнение с эталонной кривой — вся математика, пороги и структуры данных
>    на бэкенде, и как это рисуется во фронтенде (Recharts).
> 2. **Как реализована визуализация диапазонов метрик** — та самая ось «min…max
>    с зелёной/жёлтой/красной зоной и треугольником-маркером текущего значения».
>
> Дополняет `docs/VIDEO_ANALYSIS_STANDALONE_RU.md` (общая архитектура) и
> `docs/VIDEO_ANALYSIS_ARCHITECTURE.md` (ранний англоязычный обзор).

---

## Часть A. Advanced-биомеханика

### A.0. Где это живёт и когда запускается

Бэкенд, модуль `backend/app/services/video_analysis/biomechanics/`:

```
advanced_pipeline.py       # оркестратор run_advanced_biomechanics(...)
butterworth_filter.py      # Модуль 1: фильтр по углам (мутирует angle_history)
phase_portrait.py          # Модуль 2: фазовые портреты (угол vs угл. скорость)
symmetry_analyzer.py       # Модуль 3: симметрия CRP/BSI (для run/bike ПРОПУСКАЕТСЯ)
coordination_analyzer.py   # Модуль 4: координация proximal-distal (vector coding)
waveform_comparator.py     # Модуль 5: сравнение с эталонной кривой (RMSD/corr)
reference_data/            # эталонные кривые (running_reference.json, bike_reference.json)
```

Вызывается из `pipeline.py` **после** обработки всех кадров, но **до** `compute_summary()`, и **только если не rear-ракурс**:

```python
is_rear_view = camera_view == "rear" and sport_type in ("bike", "run")
if not is_rear_view:
    biomechanics_data = run_advanced_biomechanics(analyzer, sport_type)
# ...
summary["biomechanics"] = biomechanics_data
```

То есть весь этот блок работает для **side-view бега и велопосадки** (и плавания). Rear-виды используют свои собственные анализаторы и сюда не попадают.

Результат складывается в `sport_specific_metrics["biomechanics"]` и оттуда читается фронтендом.

---

### A.1. Откуда берутся углы (ключевой контекст run vs bike)

Всё поведение advanced-модулей определяется тем, **как анализатор записал `angle_history`**. Базовый `SportAnalyzer` (`base_analyzer.py`) хранит:

- `angle_history: dict[str, list[float]]` — временной ряд угла по имени сустава (по одному значению на кадр);
- `angle_timestamps: list[float]` — время в **секундах**;
- `camera_side: "left" | "right"` — сторона, обращённая к камере (детектится по Z-глубине MediaPipe, финализируется голосованием);
- `get_effective_fps()` = `(N−1) / (t[-1] − t[0])`.

**Ключевое различие:**

| | **RUN** (side) | **BIKE** (side) |
|---|---|---|
| Ключи в `angle_history` | **непрефиксные**: `knee`, `hip`, `ankle`, `elbow`, `trunk` (индексы выбраны по near-side) | **префиксные оба**: `left_knee`+`right_knee`, `left_hip`+`right_hip`, `left_ankle`+`right_ankle`, `left_shoulder`+`right_shoulder`, `left_elbow`+`right_elbow` |
| Far-side | вообще не хранится | хранится, но **весь = NaN** (`_strip_z` + far помечается NaN) |
| Итог | один «всегда надёжный» near-канал без префикса | два канала, near — реальный, far — NaN |

Из этого следует общий механизм отсева: у bike far-side = NaN → детект циклов даёт 0 → сустав/пара выпадает из выходных данных. То есть, даже несмотря на присутствие `right_*` ключей, до графиков они не доходят.

---

### A.2. Оркестратор — `run_advanced_biomechanics(analyzer, sport_type)`

```python
def run_advanced_biomechanics(analyzer, sport_type) -> dict | None
```

- Константа `MIN_FRAMES_FOR_ANALYSIS = 10`. Меньше 10 кадров → `None`.
- `effective_fps = analyzer.get_effective_fps()`, `camera_side = analyzer.camera_side`.
- Каждый модуль в отдельном `try/except` — падение одного не рушит остальные (его ключ = `None`).
- **Порядок важен** из-за мутации и передачи циклов:

```
1. Butterworth        → мутирует angle_history in-place (все дальше читают гладкие ряды)
2. Phase portraits    → даёт "cycles" по суставам
3. Symmetry / CRP     → (для run/bike сразу пустой результат)
4. Coordination       → использует cycles из шага 2 (phase_data)
5. Waveform           → использует cycles из шага 2 (phase_data)
```

**Возврат (это и есть `summary["biomechanics"]`):**

```python
{
  "effective_fps": float,
  "frame_count": int,
  "camera_side": "left" | "right" | None,
  "butterworth":     {...} | None,
  "phase_portraits": {...} | None,
  "symmetry":        {...} | None,
  "coordination":    {...} | None,
  "waveform":        {...} | None,
}
```

---

### A.3. Модуль 1 — Butterworth (`butterworth_filter.py`)

```python
apply_butterworth_filter(angle_history, effective_fps, sport_type) -> dict
```

Zero-phase low-pass по углам. **Мутирует `angle_history` in-place.**

**Константы:** `SPORT_CUTOFFS = {"run": 4.0, "bike": 3.0, "swim": 3.5}` Гц; `MIN_SAMPLES = 13`; `FILTER_ORDER = 2` (через `sosfiltfilt` эффективно 4-й порядок); `MAX_INTERP_GAP = 5`, `MARGIN = 2`.

**Шаги:**
1. `nyquist = fps / 2`; `cutoff = min(SPORT_CUTOFFS[sport], 0.9 * nyquist)`.
2. Если `cutoff <= 0` → скип всех (`reason: "invalid_fps"`).
3. `wn = cutoff / nyquist`; `sos = butter(2, wn, "low", output="sos")`.
4. По каждому углу: линейная интерполяция NaN (`_interpolate_nans`), затем `sosfiltfilt`.
5. **Восстановление больших дыр:** промежутки NaN длиной `> 5` кадров возвращаются в NaN в центре (края шириной 2 остаются интерполированными) — чтобы не выдавать «синтетику» на длинных пропусках.

**Возврат:** `{filtered: [...], skipped: [...], cutoff_hz, effective_fps, filter_order: 2}`.

**run vs bike:** отличается только cutoff (4.0 vs 3.0 Гц — велосипед статичнее). У bike far-каналы (NaN) имеют <13 валидных точек → попадают в `skipped`.

---

### A.4. Модуль 2 — Фазовые портреты (`phase_portrait.py`)

```python
compute_phase_portraits(angle_history, timestamps, sport_type, camera_side=None) -> dict
```

Строит для каждого сустава портрет «угол (X) vs угловая скорость (Y)». Стабильность = площадь выпуклой оболочки облака точек: чем компактнее (повторяемее движение) — тем меньше площадь и выше `stability_score`.

**Суставы (`SPORT_JOINTS`):**
- `run`: `["knee", "hip"]` — оба near, оба reliable.
- `bike`: `["left_knee","right_knee","left_hip","right_hip","left_ankle","right_ankle"]` — near-side ставятся первыми в обработке; far (NaN) выпадают.

**Константы:** `MIN_POINTS = 15`, `MIN_CYCLES = 2`.
`EXPECTED_ROM` (для ROM-штрафа): run `{knee: 80, hip: 45}`; bike `{knee: 70, hip: 40, ankle: 30}` (обе стороны).

**Математика:**

*Угловая скорость* `_compute_angular_velocity` — центральные разности (град/с), NaN пропагируется:
- внутри: `v[i] = (a[i+1] − a[i−1]) / (t[i+1] − t[i−1])`;
- на краях — forward/backward diff.

*Детект циклов* `_detect_cycles` (`scipy.signal.find_peaks` по сигналу угла):
- `prominence = max(5.0, nanstd(angles) * 0.5)`;
- 1-я попытка: `find_peaks(distance=5, prominence=prominence)`;
- если пиков < 2 → 2-я: `find_peaks(distance=4, prominence=prominence*0.5)`;
- цикл = интервал между соседними пиками.

*Площадь оболочки* `_compute_hull_area`:
- `ConvexHull(points).volume` (в 2D `.volume` = площадь); при <4 валидных точках или вырожденности (std <1e-6) → `0.0`.

*Площадь → stability_score* (0–100) `_hull_area_to_score`:
```
если angle_range < 1.0:        score = 50.0     # нет движения
normalized = area / angle_range**2
score = max(0, 100 − normalized * 0.5)
rom_factor = min(1.0, angle_range / EXPECTED_ROM[sport][joint])
score = score * rom_factor
```

**Важно:** если у сустава **0 циклов** — он скипается целиком (`PHASE_PORTRAIT_SKIP reason=0_cycles`). Именно так отсеивается far-side bike (NaN → 0 циклов).

**Тегирование стороны:** run → `side="near"`, `reliable=True` всегда. bike → `side = "near" if joint.startswith(camera_side) else "far"`, `reliable = (side=="near")`.

**Возврат:**
```python
{
  "joints": {
    "<joint>": {
      "data_points": [{"angle","velocity"}, ...],  # ≤200 (downsample)
      "hull_area": float,
      "stability_score": float,
      "cycles": [{start_idx,end_idx,start_time,end_time,duration}, ...],
      "angle_range": float,
      "velocity_range": float,
      "side": "near"|"far"|"both",
      "reliable": bool,
    }, ...
  },
  "overall_stability_score": float,   # mean или 50.0
}
```

---

### A.5. Модуль 3 — Симметрия / CRP (`symmetry_analyzer.py`)

```python
compute_symmetry(angle_history, timestamps, sport_type, camera_side=None) -> dict
```

**★ Для run и bike ВСЕГДА пропускается.** Первая же строка:

```python
if sport_type in ("run", "bike"):
    return {"pairs": [], "bilateral_symmetry_index": None}
```

**Почему:** в side-view far-side landmarks недостоверны (run их не хранит, bike пишет NaN), поэтому билатеральную симметрию L/R посчитать нельзя. Полноценный CRP/BSI работает **только для плавания** (там видны обе стороны).

Для полноты — формулы (актуальны для swim): нормализация сигнала в [−1,1] → фазовый угол `phase = degrees(arctan2(velocity_norm, angle_norm))` → `CRP = wrap(phase_left − phase_right)` → `BSI = max(0, 100·(1 − |L_mean − R_mean|/avg))`; lagging side по знаку `mean_crp` (порог `LAG_THRESHOLD = 5°`).

**Для run/bike выход всегда:** `{"pairs": [], "bilateral_symmetry_index": None}` → фронтенд эту панель просто не рисует.

---

### A.6. Модуль 4 — Координация (`coordination_analyzer.py`)

```python
compute_coordination(angle_history, timestamps, sport_type, phase_data=None, camera_side=None) -> dict
```

Диаграммы «угол-угол» (proximal vs distal) с оценкой стабильности координации через **vector coding**.

**Пары (`SPORT_COORD_PAIRS`, proximal→distal):**
- `run`: `[("hip","knee"), ("knee","ankle")]` — непрефиксные near-каналы.
- `bike`: `[("left_hip","left_knee"), ("left_knee","left_ankle")]` — но `left_` заменяется на `{near}_`, т.е. реально берутся `{near}_hip→{near}_knee`, `{near}_knee→{near}_ankle`.

**Константы:** `NORM_POINTS = 101`, `MIN_CYCLE_SAMPLES = 6`.

**Математика:**
- Циклы берутся из `phase_data` **проксимального** сустава (переиспользуются из Модуля 2).
- Каждый цикл тайм-нормализуется к 101 точке (`CubicSpline` при n≥8, иначе линейно).
- **Coupling angle (vector coding):** `coupling = degrees(arctan2(diff(distal), diff(proximal))) % 360`. (0/360 — только проксимальный, 90 — только дистальный, 45 — 1:1.)
- **Variability score:** `variability_score = max(0, 100 − mean(coupling_angle_std) · (100/90))` (100 при нулевой вариабельности).
- **Cross-correlation:** Pearson r при lag=0 (реализация считает только нулевой лаг → `lag` всегда 0), предпочтительно на ensemble mean-цикле.

**Если у пары 0 нормализованных циклов — пара скипается.**

**Возврат:**
```python
{
  "pairs": [
    {
      "pair": "Hip-Knee",          # лейбл с очищенными префиксами
      "proximal": "...", "distal": "...",
      "num_cycles": int,
      "cross_correlation": {"r": float, "lag": 0},
      "variability_score": float | None,
      "coupling_angle_mean": float | None,
      "cycle_data": [[{proximal,distal,pct}...] ...],  # ≤5 циклов
      "mean_cycle": [{proximal,distal,pct}...] | None,
    }, ...
  ]
}
```

**run vs bike:** одинаковые пары hip→knee и knee→ankle; разница только в источнике (непрефиксный near у run vs `{near}_*` у bike). Far-side не участвует.

---

### A.7. Модуль 5 — Сравнение с эталоном (`waveform_comparator.py`)

```python
compute_waveform_comparison(angle_history, timestamps, sport_type, phase_data=None, camera_side=None) -> dict
```

Сравнивает усреднённый цикл атлета с эталонной кривой из литературы (101 точка).

**Суставы (`SPORT_COMPARE_JOINTS`, joint→ref_key):**
- `run`: `{"knee":"knee", "hip":"hip"}`.
- `bike`: `{"left_knee":"knee", "right_knee":"knee", "left_hip":"hip", "right_hip":"hip", "left_ankle":"knee", ... "left_ankle":"ankle", "right_ankle":"ankle"}` → near-фильтр оставляет `{near}_knee/hip/ankle` (3 сустава).

> **Нюанс кода для run:** near-side фильтр сравнивает `key.startswith(near)` с непрефиксными ключами `"knee"/"hip"`, которые не начинаются с `left/right`. Поэтому при заданном `camera_side` map для run получается пустым, и waveform-сравнение для бега фактически отрабатывает только когда `camera_side is None` (тогда возвращается полный непрефиксный map). Для bike префиксные near-ключи проходят фильтр штатно.

**Эталонные данные** (`reference_data/`, кэшируются):
- `running_reference.json` → суставы `knee`, `hip` (Novacheck 1998).
- `bike_reference.json` → `knee`, `hip`, `ankle` (Bini 2011 + Fonda & Sarabon 2012).
- Формат: `{joints: {key: {description, unit, mean:[101], std:[101]}}}`, 101 точка на цикл, геометрические внутренние углы (180 = разогнут).

**Константы:** `NORM_POINTS = 101`, `Z_THRESHOLD = 2.0`, `MIN_CONSECUTIVE = 3`, порог автокоррекции `original_r < 0.3`, RMSD-нормировка `30.0`, веса similarity `0.7·r + 0.3·rmsd`.

**Математика:**
1. **Ensemble average** — усреднённая 101-точечная кривая атлета (по циклам; фолбэк — весь сигнал как один цикл).
2. **Корреляция** `original_r = corrcoef(athlete, ref_mean)`.
3. **Auto-correct** (только если `r < 0.3`): пробует сдвиги фазы (`np.roll` на 25/50/75%), инверсию конвенции (`180 − curve`), их комбинации; оставляет вариант с лучшим r. Записывает `correction_applied` + `original_r`.
4. **RMSD** = `sqrt(mean((athlete − ref)²))`.
5. **Deviation zones** — участки, где `|z| = |(athlete − ref)/ref_std| > 2` подряд ≥3 точек: `{start_pct, end_pct, mean_z_score, direction: "above"|"below"}`.
6. **Similarity score** = `100·(0.7·max(0,r) + 0.3·max(0, 1 − rmsd/30))`, зажат [0,100].

**Возврат:**
```python
{
  "comparisons": [
    {
      "joint": "...", "ref_key": "...",
      "rmsd": float, "correlation_r": float, "similarity_score": float,
      "deviation_zones": [{start_pct,end_pct,mean_z_score,direction}, ...],
      "chart_data": [{pct, athlete, ref_mean, ref_upper, ref_lower}, ...],  # ~50 точек
      "correction_applied": "...", "original_r": float,   # опционально
    }, ...
  ],
  "overall_similarity_score": float | None,
}
```

**run vs bike:** run сравнивает 2 сустава (knee, hip, только при `camera_side=None`); bike — 3 сустава (near knee/hip/ankle). Разные эталонные файлы.

---

### A.8. Сводка «run vs bike» по advanced-модулям

| Аспект | RUN (side) | BIKE (side) |
|---|---|---|
| Ключи `angle_history` | непрефиксные near | `left_*`+`right_*`, far = NaN |
| Butterworth cutoff | 4.0 Гц | 3.0 Гц (far → skipped) |
| Phase portraits | `knee`, `hip` — near, reliable | 6 ключей, far (NaN)→0 циклов→скип |
| Symmetry/CRP | **скип** (пусто) | **скип** (пусто) |
| Coordination | `hip→knee`, `knee→ankle` (непрефикс) | `{near}_hip→{near}_knee`, `{near}_knee→{near}_ankle` |
| Waveform | knee, hip (только при `camera_side=None`) | near knee/hip/ankle |
| Reference | `running_reference.json` | `bike_reference.json` |
| EXPECTED_ROM | knee 80, hip 45 | knee 70, hip 40, ankle 30 |

---

### A.9. Визуализация advanced-биомеханики (фронтенд)

Компоненты: `frontend/src/components/analysis/biomechanics/`. Типы и `getBiomechanicsData` — в **`frontend/src/types/biomechanics.ts`** (не в `analysis.ts`!). Библиотека графиков — **Recharts**. Тема цветов — `frontend/src/lib/chart-theme.tsx`.

**Поток данных:**
```ts
// types/biomechanics.ts
getBiomechanicsData(metrics) → metrics.biomechanics as BiomechanicsData | null
```
Вызывается в `analysis-results/full-analysis-view.tsx` (секция 7) → `<BiomechanicsSection data={bioData} />`.

#### Контейнер `biomechanics-section.tsx`
Пропс `{ data: BiomechanicsData }`. Считает `has*` для каждого модуля (портреты/симметрия/координация/waveform); если `moduleCount === 0` → `null`. Раскрываемая секция (тоггл), протаскивает `data.camera_side` вниз в `PhasePortraitChart` и `WaveformChart` (для L./R.-меток). Внизу — инфо-строка Butterworth (`cutoff_hz`, `filter_order`, число отфильтрованных).

#### `phase-portrait-chart.tsx` — **ScatterChart**
- Фильтр `cycles.length > 0 && angle_range > 0`; максимум 4 сустава.
- Ось X = угол, ось Y = угловая скорость; точки `data_points` (`Scatter`, r=2, opacity 0.6, цвет по индексу карточки).
- **Бейдж `stability_score`** с цветом: `≥70` emerald / `≥40` amber / иначе rose.
- Имя сустава: при заданном `cameraSide` (run/bike) префикс срезается → «Knee»; без него (swim) → «L.Knee»/«R.Elbow».
- Эталонных зон нет (это scatter сырых точек).

#### `coordination-chart.tsx` — **LineChart**
- Диаграмма угол-угол (X = proximal, Y = distal). До 5 индивидуальных циклов (полупрозрачные линии) + средний цикл (cyan, жирнее).
- Метрики: `variability_score`, `cross_correlation.r`, `coupling_angle_mean` — muted-текст **без** цветовой градации.
- Различий run/bike нет (`cameraSide` не принимает; имя пары приходит готовым).

#### `symmetry-panel.tsx` — **LineChart + ReferenceLine**
- Для run/bike пусто (модуль скипнут на бэке), панель не рисуется.
- (Для swim) SVG-кольцо BSI + список пар с бейджем `bsi%` (пороги **90/75**, не 70/40); CRP-таймлайн с `YAxis domain=[-180,180]` и `ReferenceLine y=0`.

#### `waveform-chart.tsx` — **AreaChart + Line + ReferenceArea** (самый богатый по «диапазонам»)
- Ось X = `% цикла` (pct 0–100), ось Y = градусы.
- **Эталонный коридор** = две наложенные `Area`: `ref_upper` (серая заливка) поверх `ref_lower` (цветом фона карточки) — визуально «вырезает» полосу между `ref_lower` и `ref_upper` (это `mean ± std` эталона).
- **Эталонное среднее** — пунктирная серая `Line dataKey="ref_mean"`.
- **Линия атлета** — сплошная cyan `Line dataKey="athlete"`.
- **Зоны отклонения** — красные `ReferenceArea x1..x2` (`rgba(239,68,68,0.15)`) там, где атлет вышел за эталон.
- **Бейдж `similarity_score`** (и overall): пороги `≥70` emerald / `≥40` amber / иначе rose. RMSD и r — muted-текст.

**Пороги цветовых бейджей (сводка):**

| Бейдж | Компонент | Пороги | Цвета |
|---|---|---|---|
| `stability_score` | phase-portrait | 70 / 40 | emerald / amber / rose |
| `similarity_score` | waveform | 70 / 40 | emerald / amber / rose |
| `bsi` | symmetry | 90 / 75 | emerald / amber / rose |
| `variability_score`, `r`, `rmsd`, `coupling` | coordination/waveform | — | без градации (muted) |

> Два разных «Butterworth» в типах: `ButterworthInfo` (`types/biomechanics.ts`, для инфо-строки под графиками) и `ButterworthDiagnostics` (`types/analysis.ts`, для `DiagnosticsPanel`/`ConfidenceBadge`, swim). Не путать.

---

## Часть B. Визуализация диапазонов метрик (ось min…max + маркер)

Это компонент «оценочной шкалы»: горизонтальная градиентная полоса красный→жёлтый→зелёный→жёлтый→красный, где **зелёная зона = оптимальный диапазон [min, max]**, а **треугольник/белая черта = текущий показатель атлета** на этой оси.

### B.1. Компонент `GradientRangeBar`

Файл: `frontend/src/components/analysis/gradient-range-bar.tsx`.

**Пропсы:**
```ts
interface GradientRangeBarProps {
  label: string;
  value: number;        // текущий показатель
  unit?: string;        // "deg" по умолчанию
  optimalLow: number;   // нижняя граница оптимума (min)
  optimalHigh: number;  // верхняя граница оптимума (max)
  compact?: boolean;    // true = только полоса+треугольник, без заголовка
}
```

### B.2. Как строится ось (математика раскладки)

Ось не заканчивается ровно на min/max оптимума — вокруг оптимума добавляются **зона предупреждения** и **критическая зона**, чтобы показатель, вышедший за оптимум, было куда «поставить». Все производные границы считаются от ширины оптимума:

```ts
const optRange   = optimalHigh - optimalLow;   // ширина оптимума
const margin     = optRange * 0.5;             // warning-поле = 50% ширины оптимума

const warningLow  = optimalLow  - margin;      // начало жёлтой зоны слева
const warningHigh = optimalHigh + margin;      // конец жёлтой зоны справа

// Полоса тянется ещё на один margin за warning с каждой стороны:
const barMin = warningLow  - margin;           // левый край всей оси
const barMax = warningHigh + margin;           // правый край всей оси
const barRange = barMax - barMin;
```

Итого ось состоит из 5 сегментов (слева направо):
`critical (red) | warning (amber) | OPTIMAL (green) | warning (amber) | critical (red)`.

**Позиция маркера** текущего значения (в процентах ширины оси, зажата в [0,100]):
```ts
const markerPct = clamp(((value - barMin) / barRange) * 100, 0, 100);
```

**Границы цветовых сегментов** (в процентах — для CSS-градиента):
```ts
warnLowPct  = ((warningLow  - barMin) / barRange) * 100;
optLowPct   = ((optimalLow  - barMin) / barRange) * 100;
optHighPct  = ((optimalHigh - barMin) / barRange) * 100;
warnHighPct = ((warningHigh - barMin) / barRange) * 100;
```

CSS-градиент полосы:
```
red 0%..warnLowPct → amber warnLowPct..optLowPct → green optLowPct..optHighPct
→ amber optHighPct..warnHighPct → red warnHighPct..100%
```

### B.3. Определение зоны (цвет значения)

```ts
function getZone() {
  if (value >= optimalLow && value <= optimalHigh)  return "optimal";  // #22c55e зелёный
  if ((value >= warningLow && value < optimalLow) ||
      (value >  optimalHigh && value <= warningHigh)) return "warning"; // #f59e0b жёлтый
  return "critical";                                                    // #ef4444 красный
}
```

Этот же цвет используется для треугольника-маркера и для числа в заголовке.

### B.4. Что рисуется (DOM)

1. **Заголовок** (только в non-compact): слева `label`, справа `value.toFixed(1) + unit` цветом зоны.
2. **Градиентная полоса** `h-2.5 rounded-full` с фоном-градиентом; поверх — **белая вертикальная черта** маркера на `left: markerPct%`.
3. **Треугольник** под полосой на той же позиции `markerPct%`, цвет = цвет зоны (указывает на текущее значение).
4. **Подписи диапазона** (только non-compact): слева `warningLow`, по центру зелёным `optimalLow–optimalHigh`, справа `warningHigh`.

В `compact`-режиме (используется в карточках метрик) остаётся только полоса + белая черта + треугольник, без текста.

### B.5. Пример (чтобы «пощупать» цифры)

Метрика «Trunk Lean» у бега, оптимум `[4, 8]°`, показатель `value = 11°`:

```
optRange = 4;  margin = 2
warningLow = 2;  warningHigh = 10
barMin = 0;  barMax = 12;  barRange = 12
markerPct = ((11 - 0) / 12) * 100 ≈ 91.7%     → маркер почти у правого края
value=11 > warningHigh=10 → зона "critical" → красный треугольник
```

Ось: `0…2` красная, `2…4` жёлтая, `4…8` зелёная, `8…10` жёлтая, `10…12` красная; маркер на ~92% (в правой красной зоне).

---

## Часть C. Как диапазоны попадают в карточки метрик

### C.1. Таблицы оптимальных диапазонов

Файл: `frontend/src/components/analysis/analysis-results/helpers.ts`.

`BASE_OPTIMAL_RANGES` — дефолтные диапазоны по видам спорта, например:
```ts
run: {
  trunk_lean_avg:        [4, 8],
  cadence_spm:           [170, 190],
  knee_mean:             [140, 160],
  elbow_mean:            [85, 100],
  vertical_oscillation_m:[0.06, 0.10],
},
bike: {   // дефолт = road_hoods
  knee_at_bdc:      [135, 145],
  knee_at_tdc:      [65, 75],
  trunk_angle_avg:  [40, 55],
  elbow_angle_avg:  [145, 165],
  hip_angle_avg:    [55, 65],
  ...
},
```

**Для велопосадки** диапазон зависит от **выбранной позиции** — берётся из `CYCLING_POSITION_RANGES` (5 профилей: road_hoods / road_drops / tt_aero / triathlon / casual), которые зеркалят backend `cycling_positions.py`:

```ts
export function getOptimalRangesForAnalysis(sportType, metrics) {
  if (sportType === "bike" && metrics) {
    const pos = metrics.cycling_position;
    if (pos && CYCLING_POSITION_RANGES[pos]) return CYCLING_POSITION_RANGES[pos];
  }
  return BASE_OPTIMAL_RANGES[sportType] || {};
}
```

### C.2. Цвет значения (независимо от полосы)

`getMetricColor(key, value, sportType, ranges)` — тот же принцип «зелёный/жёлтый/красный», что и в `GradientRangeBar`, с warning-полем `margin = (max - min) * 0.5`:
```ts
if (value >= min && value <= max)                    return emerald;  // оптимум
// спец-случай: hip_angle_avg асимметричен — выше max = комфорт (зелёный)
if (value >= min - margin && value <= max + margin)  return amber;    // warning
return rose;                                                          // critical
```

### C.3. Отрисовка карточки метрики

Файл: `frontend/src/components/analysis/analysis-results/sport-metric-cards.tsx`, компонент `SportMetricCards`. По каждой метрике из определения (`sport-metric-defs.ts`):

1. Достаёт значение из `metrics[key]`, парсит в число.
2. **Отображение значения с полосой погрешности:** если у метрики задан `uncertainty` (в `sport-metric-defs.ts`), в заголовке показывается **диапазон** `value−u … value+u`, а не ложно-точная десятая доля (2D-угол не точен до 0.1°). Пример: `knee_at_bdc` с `uncertainty: 3` → «142–148».
3. Цвет числа = `getMetricColor(...)`.
4. Диапазон оптимума = `getOptimalRangeForKey(key, optimalRanges)`.
5. Если диапазон найден и значение числовое → рендерит **`<GradientRangeBar compact value={...} optimalLow optimalHigh />`** (та самая ось). Если диапазона нет — просто текст «optimal: min–max».
6. Для плавания — дополнительно `TrendStrip` (тренд метрики по третям видео, усталость).

**Определения метрик** (`sport-metric-defs.ts`): `CYCLING_CORE_METRICS`, `CYCLING_AERO_METRICS`, `RUNNING_METRICS`, `SWIM_*_METRICS` — каждый элемент `{key, label, unit?, tooltip?, precision?, uncertainty?}`. `uncertainty` задан там, где 2D-угол шумит (напр. вело: knee ±3°, ankle ±4°, trunk ±2°, pelvic_ratio ±0.2).

---

## Приложение: карта файлов этого документа

| Файл | Ответственность |
|---|---|
| `biomechanics/advanced_pipeline.py` | Оркестратор advanced-биомеханики |
| `biomechanics/butterworth_filter.py` | Модуль 1: фильтр по углам |
| `biomechanics/phase_portrait.py` | Модуль 2: фазовые портреты + stability_score |
| `biomechanics/symmetry_analyzer.py` | Модуль 3: CRP/BSI (run/bike → пусто) |
| `biomechanics/coordination_analyzer.py` | Модуль 4: координация (vector coding) |
| `biomechanics/waveform_comparator.py` | Модуль 5: сравнение с эталоном (RMSD/corr) |
| `biomechanics/reference_data/*.json` | Эталонные кривые цикла (101 точка) |
| фронт `biomechanics/biomechanics-section.tsx` | Контейнер графиков |
| фронт `biomechanics/phase-portrait-chart.tsx` | ScatterChart портретов |
| фронт `biomechanics/coordination-chart.tsx` | LineChart координации |
| фронт `biomechanics/symmetry-panel.tsx` | BSI-кольцо + CRP (swim) |
| фронт `biomechanics/waveform-chart.tsx` | AreaChart эталон vs атлет |
| фронт `types/biomechanics.ts` | Типы биомеханики + `getBiomechanicsData` |
| фронт `gradient-range-bar.tsx` | **Ось min…max + маркер значения** |
| фронт `analysis-results/sport-metric-cards.tsx` | Карточки метрик (рендер полосы) |
| фронт `analysis-results/sport-metric-defs.ts` | Определения метрик + `uncertainty` |
| фронт `analysis-results/helpers.ts` | Таблицы диапазонов + `getMetricColor` |

---

*Документ сгенерирован на основе анализа кодовой базы Motus. При рефакторинге ядра сверяйтесь с исходниками.*
