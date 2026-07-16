# Расчёт Confidence (уровня доверия) в видео-анализе

> Документ описывает, как в Motus (CoachPowerBoost) считается **confidence** —
> итоговый уровень доверия к результату видео-анализа (`high` / `medium` / `low`),
> который показывается пользователю бейджем. Разбираются все входные сигналы,
> пороги, логика понижения уровня и per-metric бейджи для плавания.
>
> Дополняет `docs/VIDEO_ANALYSIS_STANDALONE_RU.md` (общая архитектура).

---

## 0. Три уровня «доверия», которые не надо путать

В пайплайне есть **три разных механизма оценки качества**. Confidence — только один из них:

| Механизм | Файл | Что делает | Влияет на |
|---|---|---|---|
| **Quality gate** | `quality_gate.py` | Жёсткий продуктовый гейт: если данных слишком мало → **partial-режим** (скрыть оценку и AI-разбор) | Показывать/не показывать результат |
| **Confidence** ← *этот документ* | `confidence_scorer.py` | Мягкий уровень доверия `high/medium/low` для всего отчёта | Информационный бейдж (ничего не скрывает) |
| **Per-metric confidence** (swim) | `metric_reliability.py` | Бейдж доверия на **каждую отдельную метрику** плавания | Кавеат рядом с конкретным числом |

Confidence **никогда ничего не скрывает** — он лишь честно сообщает пользователю, насколько можно верить числам. Скрытие метрик — это работа quality gate (partial-режим), который отрабатывает раньше.

---

## 1. Главная функция — `compute_analysis_confidence`

Файл: `backend/app/services/video_analysis/biomechanics/confidence_scorer.py`.

```python
compute_analysis_confidence(
    angle_statistics: dict[str, dict],       # статистика по каждому углу (в т.ч. nan_pct)
    frames_processed: int,                   # сколько кадров обработано
    butterworth_meta: dict | None,           # мета Butterworth-фильтра
    analysis_warnings: list[str],            # объединённые предупреждения
    landmark_quality: dict | None = None,    # качество детекции точек по регионам
    phase_diagnostics: dict | None = None,   # диагностика фаз гребка (только swim)
) -> {
    "level": "high" | "medium" | "low",
    "factors": {...},        # прозрачная раскладка причин
    "explanation": str,      # человекочитаемое объяснение
}
```

**Принцип:** уровень стартует как `"high"` и **только понижается** по мере срабатывания факторов. Понижение делается функцией `_downgrade(current, target)`, которая возвращает **худший** из двух уровней (порядок `high=0 < medium=1 < low=2`). То есть ни один фактор не может *повысить* уровень — итог = самый плохой сигнал среди всех.

```python
def _downgrade(current, target):
    order = {"high": 0, "medium": 1, "low": 2}
    return current if order[current] >= order[target] else target
```

Результат кладётся в `summary["analysis_confidence"]` и уходит во фронтенд.

---

## 2. Пороги (все в одном месте)

```python
THRESHOLDS = {
    "nan_pct_angle_strict": 40.0,      # угол с >40% пропусков считается «дырявым»
    "nan_pct_angle_high":   60.0,      # угол с >60% пропусков → сразу medium
    "valid_frames_ratio_medium": 0.70, # (объявлено; см. прим.)
    "valid_frames_ratio_low":    0.40, # (объявлено; см. прим.)
    "max_warnings_for_high":  0,       # 0 предупреждений → можно остаться high
    "max_warnings_for_medium": 2,      # >2 предупреждений → low
    "cutoff_reduction_pct_medium": 40.0, # фильтр урезал точность >40% → medium
    "unknown_phase_pct_medium": 40.0,  # swim: >40% неопознанных фаз → medium
    "unknown_phase_pct_low":    60.0,  # swim: >60% неопознанных фаз → low
}
```

> Примечание: `valid_frames_ratio_*` объявлены в таблице, но в текущей реализации `compute_analysis_confidence` напрямую не используются — долю валидных кадров учитывает более жёсткий quality gate. Здесь роль «мало кадров» играют другие факторы (landmark quality, Butterworth fallback, warnings).

---

## 3. Пять факторов, понижающих уровень

Все проверяются последовательно, каждый может дёрнуть `_downgrade`.

### Фактор 1 — Качество детекции точек (landmark quality)
Вход: `landmark_quality["confidence"]` (см. §4 — как он считается).
```
confidence == "low"    → _downgrade(→ low)    + «Body tracking was poor…»
confidence == "medium" → _downgrade(→ medium) + «Some landmarks had intermittent detection»
```
Также сохраняется `landmark_quality["overall_pct"]` в factors.

### Фактор 2 — Процент пропусков по углам (per-angle NaN)
Идёт по всем углам из `angle_statistics`, смотрит `nan_pct` каждого:
```
для каждого угла:
    если nan_pct > 60 → добавить в angles_with_high_nan, _downgrade(→ medium)
    если nan_pct > 40 → angles_above_strict += 1

majority_gated = (angles_above_strict > total_angles / 2)
если majority_gated → _downgrade(→ medium) + «More than half of angles had gaps»
```
То есть: один сильно дырявый угол (>60%) → medium; либо больше половины углов дырявые (>40%) → medium.

### Фактор 3 — Butterworth: fallback / урезание частоты среза
Вход: `butterworth_meta`.
```
если fallback_triggered → _downgrade(→ medium) + «fallback smoother used (clip too short)»
если reduction_pct > 40 → _downgrade(→ medium) + «filter had to reduce precision»
```
(`fallback_triggered` = фильтр не смог отработать штатно на коротком клипе; `reduction_pct` = насколько пришлось урезать частоту среза из-за низкого fps.)

### Фактор 4 — Количество предупреждений (analysis warnings)
Вход: объединённый список `analysis_warnings + quality_warnings` (см. §4).
```
warning_count > 2  → _downgrade(→ low)    + «Multiple quality issues detected»
warning_count > 0  → _downgrade(→ medium) + «A quality warning was raised»
```

### Фактор 5 — Качество классификации фаз (ТОЛЬКО плавание)
Вход: `phase_diagnostics` (dict `diagnostics.phase_thresholds`, только для swim).
```
unknown_phase_pct > 60 → _downgrade(→ low)
unknown_phase_pct > 40 → _downgrade(→ medium)
```
Плюс формируется **разное объяснение** в зависимости от источника калибровки (`source`):
- `source == "calibrated"` — адаптивная калибровка была применена, но фаз всё равно много неопознанных → проблема в качестве видео (совет: чище подводный ракурс, свет, меньше пузырей/бликов).
- `source == "fixed"` + `fallback_reason` — пытались калибровать, но откатились к фиксированным порогам (причина через `_humanize_fallback_reason`).
- иначе — техника нетипична; совет включить адаптивную калибровку фаз в advanced-опциях.

`_humanize_fallback_reason` переводит машинные причины отката в человеческие:
```
insufficient_samples → "not enough data"
low_variance         → "low motion variance"
narrow_range         → "narrow angle range"
sanity_violation     → "unstable thresholds"
out_of_bounds        → "out-of-bounds thresholds"
```

---

## 4. Откуда берутся входные сигналы

Всё собирается в `pipeline.py` перед вызовом `compute_analysis_confidence`.

### 4.1. `landmark_quality` — `_compute_landmark_quality(...)` (pipeline.py)

Считает **процент кадров, где точки региона тела были детектированы** (visibility ≥ `threshold = 0.3`), по 4 регионам:

| Регион | Индексы точек |
|---|---|
| `upper_body` | 11,12,13,14,15,16 (плечи, локти, кисти) |
| `core` | 23,24 (бёдра) |
| `lower_body` | колени+голеностопы (см. ниже) |
| `head` | 0 (нос) |

**Нижняя часть (`lower_body`) — важный нюанс односторонних видов:**
- Для **bike/run side-view** с известным `camera_side` — считается только по **ближней** ноге (2 индекса: колено+голеностоп ближней стороны). Причина: дальняя нога перекрыта телом/рамой, и AND-gate по 4 точкам ложно показал бы ~0%.
- Для swim и rear-видов — по **обеим** ногам (4 индекса, bilateral).

```
overall_pct = среднее по 4 регионам

# «критический» регион зависит от спорта:
swim → critical = upper_body
bike → critical = (upper_body + lower_body) / 2
run  → critical = overall_pct

confidence = "high"   если critical >= 60
             "medium" если critical >= 30
             "low"    иначе
```

Возвращает: `overall_pct`, `regions{}`, `upper_body_detection_ratio`, `lower_body_detection_ratio`, `confidence`, `lower_body_measurement_basis` (unilateral/bilateral).

Именно этот `confidence` (high/medium/low) идёт в **Фактор 1**, а `overall_pct` — в per-metric бейджи плавания.

### 4.2. `quality_warnings` — `_build_quality_warnings(...)` (pipeline.py)

Список пользовательских предупреждений (питает **Фактор 4**):
```
confidence == "low"                          → «Low landmark detection quality…»
swim и upper_body < 40%                       → «Upper body detected in only X%…»
head < 30%                                    → «Head position not reliably detected»
skeleton_jumps > 3                            → «Multiple people may be visible…»
```
`skeleton_jumps` = `_detect_skeleton_jumps(...)`: число кадров, где центр таза «прыгнул» >0.3 (в нормализованных координатах) между соседними кадрами — признак того, что MediaPipe переключился на другого человека.

### 4.3. `analysis_warnings`
Отдельный список предупреждений анализатора; сюда же в run/bike добавляется предупреждение об **адаптивном сэмплинге** (если длинный клип был прорежен — `frame_idx` дельта >1 → «Long clip downsampled… cadence may be less precise»).

### 4.4. Слияние перед вызовом
```python
_merged_warnings = summary["analysis_warnings"] + summary["quality_warnings"]

summary["analysis_confidence"] = compute_analysis_confidence(
    angle_statistics = angle_stats,
    frames_processed = len(raw_frame_data),
    butterworth_meta = butter_meta,
    analysis_warnings = _merged_warnings,     # ← оба списка влияют на уровень
    landmark_quality = landmark_quality,
    phase_diagnostics = phase_diagnostics,     # только swim
)
```
Оба списка предупреждений сливаются локально **только для скоринга** — в UI у них остаются свои отдельные панели.

---

## 5. Итоговый объект и объяснение

```python
{
  "level": "high" | "medium" | "low",
  "factors": {
    "landmark_quality_pct": float | None,
    "angles_with_high_nan": [str, ...],       # какие углы дырявые
    "majority_angles_gated": bool,            # >половины углов дырявые
    "fallback_triggered": bool,               # Butterworth откат
    "cutoff_reduced": bool,                   # урезание частоты среза
    "warning_count": int,
    # только для swim:
    "high_unknown_phases": bool,
    "unknown_phase_pct": float,
    "phase_calibration_source": "fixed" | "calibrated",
  },
  "explanation": str,
}
```

**Объяснение (`explanation`):**
- если `level == "high"` → «Analysis ran on clean data with good landmark detection.»
- иначе → все накопленные `reasons` через пробел + рекомендация «Try a clearer video with the whole body in frame, or film from a different angle.»

`factors` даёт прозрачную раскладку — по нему инженер поддержки (или сам пользователь через тултип) видит, **почему** упал уровень.

---

## 6. Per-metric confidence (только плавание, FULL-режим)

Файл: `backend/app/services/video_analysis/biomechanics/metric_reliability.py`.

Отдельно от общего confidence, для плавания в полном (не partial) режиме на **каждую метрику** вешается свой бейдж `high/medium/low`. Главная задача — не дать подводным углам захвата/EVF выглядеть полностью надёжными (BlazePose обучен на «сухих» позах, вода искажает подводные точки).

### 6.1. Классы надёжности метрик
```python
class MetricReliability:
    ALWAYS            # покадровые агрегаты — переживают даже плохие фазы
    PHASE_DEPENDENT   # зависят от детекции фаз CATCH/PULL/PUSH/ENTRY
    LANDMARK_SENSITIVE# зависят от чистоты конкретных точек тела
```
Примеры (`SWIM_METRIC_RELIABILITY`): `head_alignment_avg`, `kick_amplitude_avg`, `breath_count` → ALWAYS; `streamline_avg`, `body_line_angle_avg` → LANDMARK_SENSITIVE; `elbow_at_catch_avg`, `evf_angle_avg`, `stroke_rate_spm` → PHASE_DEPENDENT. Неизвестный ключ → LANDMARK_SENSITIVE (консервативно).

### 6.2. Пороги
```python
ALWAYS_LANDMARK_FLOOR_PCT          = 40.0   # ниже — даже ALWAYS-метрики падают в low
LANDMARK_SENSITIVE_FLOOR_PCT       = 60.0   # ниже — LANDMARK_SENSITIVE → low
PHASE_DEPENDENT_UNKNOWN_CEILING_PCT = 70.0  # выше unknown% — PHASE_DEPENDENT → low
_UNDERWATER_PHASE_CEILING = "medium"        # подводные фазовые метрики не выше medium
```

### 6.3. Логика `metric_confidence(metric_key, unknown_phase_pct, landmark_quality_pct, camera_angle)`
```
ALWAYS:
    landmark_quality < 40 → low, иначе → high

LANDMARK_SENSITIVE:
    landmark_quality < 60 → low
    иначе если under_water → medium («подводный трекинг приблизителен»)
    иначе → high

PHASE_DEPENDENT:
    unknown_phase_pct >= 70 → low
    unknown_phase_pct >= 40 → medium
    иначе → high
    + если under_water → уровень «прижимается» к medium (не выше)
```
Возврат `{"level": ..., "reason": ...}` (reason пустой для чистого high — тултип тогда не показывается).

`build_swim_metric_confidence(summary, camera_angle, unknown_phase_pct, landmark_quality_pct)` пробегает все известные swim-метрики, присутствующие в summary (non-None), и выдаёт бейдж на каждую. Кладётся в summary и рендерится рядом с числами.

> Есть также **устаревший** путь `is_reliable_for_partial` (v1) — он *скрывал* метрики. В v2 partial-режим скрывает **все** метрики целиком (потому что деградация landmark quality заражает всё, а не только фазозависимое), поэтому suppression-логика больше не используется — остались только информационные бейджи.

---

## 7. Фронтенд — `ConfidenceBadge`

Файл: `frontend/src/components/analysis/confidence-badge.tsx`.

Пропсы: `{ confidence: AnalysisConfidence, phaseDiagnostics? }`. Рендерит кликабельную «пилюлю» с цветом и иконкой по уровню:

| Уровень | Цвет | Иконка |
|---|---|---|
| high | emerald | `CheckCircle2` |
| medium | amber | `Info` |
| low | red | `AlertTriangle` |

По клику раскрывается панель с `confidence.explanation` (из бэкенда) и, для плавания, дополнительной строкой о калибровке фаз:
- `source == "calibrated"` → «персонализировано (N образцов)»;
- есть `fallback_reason` → «откат к стандартным порогам (причина)» (через `useHumanizeFallbackReason` → i18n);
- иначе → «стандартные пороги».

Плюс токен «% неопознанных фаз» с цветом по `unclassifiedColorClass`:
```
< 15%   → зелёный
15–30%  → жёлтый
> 30%   → красный
```

**Важно про синхронизацию:** фронтенд держит собственную копию карты `FALLBACK_REASON_LABELS`/`FALLBACK_REASON_KEYS`, зеркалящую `_FALLBACK_REASON_LABELS` из `confidence_scorer.py`. При добавлении новой причины отката её надо править **в обоих местах** (бэкенд формирует `explanation`, фронтенд — тултип калибровки).

---

## 8. Сводная схема

```
                       ┌─ Фактор 1: landmark_quality.confidence (low→low / medium→medium)
                       ├─ Фактор 2: per-angle nan_pct (>60%→medium; >половины >40%→medium)
  level = "high" ──────┼─ Фактор 3: butterworth (fallback→medium; reduction>40%→medium)
    (только ↓)         ├─ Фактор 4: warnings (>2→low; >0→medium)
                       └─ Фактор 5: swim unknown_phase_pct (>60→low; >40→medium)
                                    │
                                    ▼
                        _downgrade = худший из всех
                                    │
                                    ▼
        { level, factors{прозрачная раскладка}, explanation }
                                    │
                       ┌────────────┴────────────┐
                       ▼                         ▼
            summary["analysis_confidence"]   (swim, full) build_swim_metric_confidence
                       │                         → бейдж high/medium/low на КАЖДУЮ метрику
                       ▼
              ConfidenceBadge (пилюля + тултип)
```

---

## Приложение: карта файлов

| Файл | Ответственность |
|---|---|
| `biomechanics/confidence_scorer.py` | Главная функция `compute_analysis_confidence` + пороги |
| `biomechanics/metric_reliability.py` | Per-metric бейджи плавания (`build_swim_metric_confidence`) |
| `pipeline.py` → `_compute_landmark_quality` | Качество детекции точек по регионам (вход Фактора 1) |
| `pipeline.py` → `_build_quality_warnings` | Предупреждения (вход Фактора 4) |
| `pipeline.py` → `_detect_skeleton_jumps` | Детект «прыжков» скелета (предупреждение) |
| `pipeline.py` (~стр. 1133) | Сборка входов + вызов `compute_analysis_confidence` |
| фронт `confidence-badge.tsx` | Бейдж + тултип с объяснением |

---

*Документ сгенерирован на основе анализа кодовой базы Motus. При рефакторинге ядра сверяйтесь с исходниками.*
