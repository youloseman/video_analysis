"""The coach's brief for a Coach Notes order: one self-contained Russian HTML page.

Bogdan writes the notes in Russian, from Telegram, often on a phone. What he
needs is the clip, the annotated frame, the kinogram, the numbers against
their bands, what the automatic report already told the athlete, and a form
whose answer he can copy back -- in a file that opens with no login and no
connection. Until 2026-10-28 that page was assembled by hand per order, from a
prod dump nobody but Artur could run. This renders the same page from the
stored analysis, so the admin queue can hand it over in one click.

Pure: takes the stored result/entry/media and the clip bytes, returns HTML.
Every string that came from the athlete or the pipeline is escaped.
"""

from __future__ import annotations

import base64
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime, timedelta
from html import escape
from pathlib import Path
from typing import Any

import structlog

from app.services.notify import NOTES_DUE_HOURS

logger = structlog.get_logger(__name__)

ENCODE_TIMEOUT_S = 120
# Without ffmpeg the original goes in only if it is small enough to send.
MAX_RAW_CLIP_BYTES = 25 * 2**20
# A saddle step bigger than this at once breaks the review rule (5-10 mm).
MAX_SADDLE_STEP_MM = 10

_WEEKDAY_GEN = ["понедельника", "вторника", "среды", "четверга", "пятницы", "субботы", "воскресенья"]
_MONTH_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
              "сентября", "октября", "ноября", "декабря"]

# Russian label and unit for every number the brief can show. Keys are the
# summary fields reference_bands() serves, plus the unbanded extras.
_LABELS: dict[str, tuple[str, str]] = {
    # bike
    "knee_at_bdc": ("Колено внизу (НМТ, высота седла)", "°"),
    "knee_at_tdc": ("Колено вверху (ВМТ)", "°"),
    "hip_at_tdc": ("Бедро вверху (ВМТ)", "°"),
    "pelvic_ratio": ("Тазовое соотношение", ""),
    "trunk_angle_avg": ("Корпус к горизонту", "°"),
    "shoulder_angle_avg": ("Плечо", "°"),
    "elbow_angle_avg": ("Локоть", "°"),
    "forearm_tilt_avg": ("Наклон предплечий", "°"),
    "helmet_above_back": ("Голова над спиной, % длины корпуса", "%"),
    "ankle_at_bdc": ("Голеностоп внизу", "°"),
    "hip_at_bdc": ("Бедро внизу (НМТ)", "°"),
    # run
    "cadence_spm": ("Каденс", " шаг/мин"),
    "trunk_lean_avg": ("Наклон корпуса", "°"),
    "ground_contact_ms": ("Контакт с землёй", " мс"),
    "flight_time_ms": ("Фаза полёта", " мс"),
    "overstride_ratio": ("Вынос стопы вперёд (overstride)", ""),
    "vertical_oscillation_cm": ("Вертикальные колебания", " см"),
    "knee_max": ("Колено при касании", "°"),
    "knee_min": ("Колено в махе", "°"),
}
_BIKE_EXTRAS = ("helmet_above_back", "ankle_at_bdc", "hip_at_bdc")

# What to look for on the clip when a number is out of its band, by direction.
_LOOK: dict[tuple[str, str], str] = {
    ("knee_at_bdc", "hi"): "Обычно это высоковатое седло. Тянет ли носок вниз, качается ли таз по седлу, "
                           "«дотягивается» ли нога? Если опускать, на сколько первым шагом? Наше правило: "
                           "5–10 мм за раз, не больше.",
    ("knee_at_bdc", "lo"): "Обычно это низкое седло: нога внизу заметно согнута. Видно ли это? Если поднимать, "
                           "то по 5–10 мм за раз.",
    ("knee_at_tdc", "hi"): "Колено мало закрывается вверху: высоковатое седло или короткие шатуны.",
    ("knee_at_tdc", "lo"): "Колено сильно закрыто вверху: низкое седло или длинные шатуны.",
    ("hip_at_tdc", "lo"): "Бедро вверху сильно закрыто. Для триатлона это может мешать бежать после велосипеда.",
    ("trunk_angle_avg", "lo"): "Корпус очень низкий. Держит ли он позу спокойно (дыхание, плечи, шея)? "
                               "Или стоит приподнять подлокотники?",
    ("trunk_angle_avg", "hi"): "Корпус высоко. Есть ли запас опустить перёд?",
    ("pelvic_ratio", "hi"): "Низкий корпус получается скорее за счёт наклона таза вперёд или круглой спины. "
                            "Видно ли это на видео?",
    ("shoulder_angle_avg", "hi"): "Руки далеко вытянуты вперёд. Не тянется ли он к удлинителям?",
    ("shoulder_angle_avg", "lo"): "Руки близко к корпусу, посадка сжатая.",
    ("elbow_angle_avg", "hi"): "Руки почти прямые. Расслаблены ли плечи?",
    ("cadence_spm", "lo"): "Каденс низкий. Длинный шаг, «тянется» ли нога вперёд?",
    ("overstride_ratio", "hi"): "Стопа ставится далеко впереди корпуса. Видно ли торможение при касании?",
    ("vertical_oscillation_cm", "hi"): "Много прыгает вверх. Куда уходит толчок?",
    ("trunk_lean_avg", "hi"): "Сильный наклон корпуса. Из голеностопа или ломается в поясе?",
    ("trunk_lean_avg", "lo"): "Корпус почти вертикальный или отклонён назад.",
    ("ground_contact_ms", "hi"): "Долгий контакт с землёй. Мягкая, «проваливающаяся» постановка?",
    ("knee_max", "hi"): "Нога почти прямая при касании. Ставит ли стопу впереди себя?",
}

_PRO_NAMES = {"trunk_angle_avg": "корпус", "shoulder_angle_avg": "плечо", "elbow_angle_avg": "локоть",
              "forearm_tilt_avg": "предплечья", "helmet_above_back": "голова над спиной"}


def _num(v: Any, digits: int = 1) -> str:
    return f"{v:.{digits}f}".replace(".", ",") if isinstance(v, float) and not v.is_integer() else \
        str(int(v)) if isinstance(v, (int, float)) else "—"


def _fmt(key: str, v: Any) -> str:
    unit = _LABELS.get(key, ("", ""))[1]
    digits = 0 if key in ("cadence_spm", "ground_contact_ms", "flight_time_ms") else \
        2 if key == "overstride_ratio" else 1
    return _num(round(float(v), digits), digits) + unit if isinstance(v, (int, float)) else "—"


def _band(lo: Any, hi: Any, key: str) -> str:
    unit = _LABELS.get(key, ("", ""))[1]
    return f"{_num(lo)}–{_num(hi)}{unit.strip() if unit in ('°', '%') else ''}"


def _status(v: float, lo: float, hi: float) -> tuple[str, str, str]:
    """(row class, tag text, direction) for a value against its band."""
    width = max(hi - lo, 1e-9)
    if v > hi:
        return ("bad" if v - hi > width / 2 else "warn"), "выше нормы", "hi"
    if v < lo:
        return ("bad" if lo - v > width / 2 else "warn"), "ниже нормы", "lo"
    if hi - v <= width * 0.1:
        return "warn", "у верхней границы", "edge_hi"
    if v - lo <= width * 0.1:
        return "warn", "на нижней границе", "edge_lo"
    return "", "ок", ""


def _deadline(paid_ms: int) -> tuple[str, str]:
    """(coach deadline, client deadline), in Vancouver time, in Russian.

    The coach's is the evening before the athlete's day: the answer still has
    to be translated, pasted and published.
    """
    try:
        from zoneinfo import ZoneInfo
        tz: Any = ZoneInfo("America/Vancouver")
    except Exception:  # noqa: BLE001 -- no tz database in the image: UTC is still a date
        tz = UTC
    due = datetime.fromtimestamp(paid_ms / 1000, UTC).astimezone(tz) + timedelta(hours=NOTES_DUE_HOURS)
    coach = due - timedelta(days=1)
    c = f"{_WEEKDAY_GEN[coach.weekday()]} {coach.day} {_MONTH_GEN[coach.month - 1]}, вечер"
    d = f"{_WEEKDAY_GEN[due.weekday()]} {due.day} {_MONTH_GEN[due.month - 1]}"
    return c, d


def _md(text: str) -> str:
    """The athlete's report is light markdown: bold and line breaks are enough."""
    out = escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    return out.replace("\n", "<br>")


def _rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    sport = result.get("sport_type") or "run"
    m = result.get("sport_specific_metrics") or {}
    bands = result.get("reference_bands") or {}
    pro = ((result.get("pro_field") or {}).get("metrics") or {})
    rows: list[dict[str, Any]] = []
    for key, b in bands.items():
        v = m.get(key)
        if not isinstance(v, (int, float)) or key not in _LABELS:
            continue
        cls, tag, direction = _status(float(v), float(b["lo"]), float(b["hi"]))
        rows.append({"key": key, "value": _fmt(key, v), "band": _band(b["lo"], b["hi"], key),
                     "cls": cls, "tag": tag, "tagcls": cls or "ok", "dir": direction, "raw": float(v),
                     "lo": b["lo"], "hi": b["hi"]})
    if sport == "bike":
        for key in _BIKE_EXTRAS:
            v = m.get(key)
            if not isinstance(v, (int, float)):
                continue
            row = {"key": key, "value": _fmt(key, v), "band": "—", "cls": "", "tag": "без оценки",
                   "tagcls": "na", "dir": "", "raw": float(v)}
            if key == "helmet_above_back" and key in pro:
                p = pro[key]
                row["band"] = f"профи {round(p['lo'])}–{round(p['hi'])}%"
                if m.get("helmet_check") is False:
                    row["tag"] = "читается неуверенно"
            rows.append(row)
    order = list(_LABELS)
    return sorted(rows, key=lambda r: order.index(r["key"]))


def _look_items(result: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    sport = result.get("sport_type") or "run"
    items = []
    for r in rows:
        if r["dir"] not in ("hi", "lo", "edge_lo", "edge_hi"):
            continue
        direction = {"edge_lo": "lo", "edge_hi": "hi"}.get(r["dir"], r["dir"])
        hint = _LOOK.get((r["key"], direction))
        if r["dir"].startswith("edge") and not hint:
            continue
        label = _LABELS[r["key"]][0]
        where = r["tag"]
        items.append(f"<b>{escape(label)}</b>: {escape(r['value'])} при норме {escape(r['band'])}, "
                     f"{escape(where)}. {escape(hint or 'Что видно на видео?')}")
    tail = ("Всё, чего цифры не видят: круглость педалирования, стопа, плечи, руки на удлинителях."
            if sport == "bike" else
            "Всё, чего цифры не видят: постановка стопы, работа рук, плечи, расслабленность.")
    if not items:
        items.append("Все оцениваемые цифры в норме. Тем ценнее взгляд глазами: что бы ты поменял первым?")
    items.append(escape(tail))
    return items


def _saddle_flag(result: dict[str, Any]) -> str | None:
    """The automatic report's saddle step, when it is bigger than a coach would make at once."""
    for d in ((result.get("fit_plan") or {}).get("diagnostics") or []):
        if d.get("component") != "saddle_height":
            continue
        mm = re.search(r"(\d+)\s*mm", str(d.get("amount") or ""))
        if mm and int(mm.group(1)) > MAX_SADDLE_STEP_MM:
            return (f"Автоматический отчёт советует изменить высоту седла на {escape(str(d['amount']))} сразу. "
                    f"Это больше нашего шага 5–10 мм за раз. Твой ответ про седло для клиента главный: "
                    f"сколько менять и надо ли вообще.")
    return None


def _video_caption(result: dict[str, Any], entry: dict[str, Any]) -> str:
    m = result.get("sport_specific_metrics") or {}
    sampling = m.get("sampling") or {}
    bits = []
    dur = sampling.get("duration_s")
    if isinstance(dur, (int, float)):
        bits.append(f"Около {round(dur)} секунд")
    if (result.get("sport_type") or entry.get("sport")) == "bike":
        revs = (m.get("diagnostics") or {}).get("bdc_tdc", {}).get("pedal_strokes") \
            or (result.get("bilateral_geometry") or {}).get("revolutions")
        if revs:
            bits.append(f"примерно {revs} оборотов")
    side = result.get("camera_side") or entry.get("cameraSide")
    text = ", ".join(bits) + "." if bits else ""
    if side:
        text += f" Снято {'справа' if side == 'right' else 'слева'}."
    if (m.get("saddle_estimate") or {}).get("drive_side") or any(
            c.get("id") == "drive_side" for c in ((m.get("capture_report") or {}).get("checks") or [])):
        text += (" <b>Со стороны звёзд</b>: звезда мешает распознаванию лодыжки в нижней точке, "
                 "поэтому колено внизу здесь наименее надёжная цифра.")
    return text.strip()


def _title(result: dict[str, Any], entry: dict[str, Any]) -> str:
    sport = result.get("sport_type") or entry.get("sport")
    if sport != "bike":
        return "техника бега"
    pos = str(result.get("cycling_position") or entry.get("position") or "")
    if pos in ("triathlon", "tt", "time_trial"):
        return "посадка на триатлонном велосипеде"
    return "посадка на велосипеде"


def clip_for_brief(path: Path) -> bytes | None:
    """The stored clip, re-encoded small enough to travel inside a Telegram file.

    720p H.264 without sound is ~0.5-3 MB for a 10-30 s clip; the stored 1080p
    original can be ten times that. Falls back to the original when ffmpeg is
    missing or fails and the original is small, else None. Never raises.
    """
    try:
        if not path.is_file():
            return None
        if shutil.which("ffmpeg"):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "brief.mp4"
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-map", "0:v:0",
                         "-vf", "scale=-2:'min(720,ih)'", "-an", "-c:v", "libx264", "-crf", "28",
                         "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                        check=True, capture_output=True, timeout=ENCODE_TIMEOUT_S,
                    )
                    return out.read_bytes()
                except (subprocess.SubprocessError, OSError) as e:
                    logger.warning("BRIEF_CLIP_ENCODE_FAILED", path=str(path), err=str(e))
        return path.read_bytes() if path.stat().st_size <= MAX_RAW_CLIP_BYTES else None
    except OSError:
        return None


def render_brief_html(*, order_id: int, paid_ms: int, result: dict[str, Any] | None,
                      entry: dict[str, Any] | None, media: dict[str, str | None],
                      clip: bytes | None) -> str:
    result = result or {}
    entry = entry or {}
    sport = result.get("sport_type") or entry.get("sport") or "run"
    rows = _rows(result)
    coach_due, client_due = _deadline(paid_ms)
    focus = result.get("focus")
    question = (f"Атлет задал вопрос: <b>«{escape(str(focus))}»</b>. Ответ на него для клиента главный."
                if focus else "Отдельного вопроса атлет не задал, так что главное выбираешь ты.")
    score = result.get("technique_score")
    grade = result.get("letter_grade")

    media_html = []
    if clip:
        media_html.append(
            '<div class="card"><video controls playsinline preload="metadata" src="data:video/mp4;base64,'
            + base64.b64encode(clip).decode() + f'"></video><p class="cap">{_video_caption(result, entry)}</p></div>')
    elif entry.get("kind") == "video":
        media_html.append('<div class="card"><p class="cap">Видео в файл не попало (не сохранилось или '
                          'слишком большое). Попроси Артура прислать его отдельно.</p></div>')
    if media.get("keyframe"):
        media_html.append(f'<div class="card"><img src="{escape(media["keyframe"])}" alt="Кадр с измеренными углами">'
                          '<p class="cap">Кадр с углами, которые посчитало приложение.</p></div>')
    top = f'<div class="media two">{"".join(media_html)}</div>' if media_html else ""
    kino = ""
    if media.get("kinogram"):
        cap = "Один оборот: верх, низ и по 45° вокруг." if sport == "bike" else "Ключевые положения одного шага."
        kino = f'<div class="card"><img src="{escape(media["kinogram"])}" alt="Кинограмма"><p class="cap">{cap}</p></div>'

    table = "".join(
        f'<tr class="{r["cls"]}"><td>{escape(_LABELS[r["key"]][0])}</td><td class="v">{escape(r["value"])}</td>'
        f'<td>{escape(r["band"])}</td><td><span class="tag {r["tagcls"]}">{escape(r["tag"])}</span></td></tr>'
        for r in rows)
    pro = ((result.get("pro_field") or {}).get("metrics") or {})
    pro_line = ", ".join(f"{name} {round(pro[k]['median'])}{'%' if k == 'helmet_above_back' else '°'}"
                         for k, name in _PRO_NAMES.items()
                         if isinstance(pro.get(k), dict) and "median" in pro[k])
    cap = ("ВМТ — педаль в самом верху, НМТ — в самом низу. " if sport == "bike" else "") + \
        "Углы считает модель распознавания позы, точность ±2–3°."
    if score is not None:
        cap += f" Общая оценка приложения {score}/100" + (f" ({escape(str(grade))})" if grade else "") + "."
    if pro_line:
        cap += f" Для сравнения, медиана 32 профи на Коне: {pro_line}."
    numbers = (f'<h2>Цифры</h2><div class="card"><table><tr><th>Что</th><th>Значение</th><th>Норма</th><th></th></tr>'
               f'{table}</table><p class="cap">{cap}</p></div>') if rows else ""

    look = "".join(f"<li>{i}</li>" for i in _look_items(result, rows))
    report = entry.get("report") or ((result.get("ai_recommendations") or {}).get("report"))
    flag = _saddle_flag(result)
    told = ""
    if report or flag:
        told = ('<details><summary>Что клиенту уже написал автоматический отчёт</summary>'
                + (f'<blockquote>{_md(str(report))}</blockquote>' if report else "")
                + '<p class="cap">Текст по-английски, как его видел клиент.'
                + (f" {flag}" if flag else "") + '</p></details>')

    return (_HEAD.replace("{{ORDER}}", str(order_id))
            + f"""<body>
<main>
  <h1>Coach Notes: {escape(_title(result, entry))}</h1>
  <p class="sub">{question} Приложение посчитало углы, теперь нужен взгляд живого тренера.</p>

  <div class="card deadline">
    <b>Что нужно:</b> одна главная фраза и три короткие заметки, внизу страницы есть форма.<br>
    <b>Когда:</b> до <b>{coach_due}</b>. Клиенту обещан ответ в течение 72 часов, до {client_due}.<br>
    <b>Сколько времени:</b> минут 20–30.
  </div>

  <h2>Материалы</h2>
  {top}
  {kino}
  {numbers}

  <h2>На что я бы посмотрел</h2>
  <div class="card">
    <ul>{look}</ul>
    {told}
  </div>
"""
            + _TAIL.replace("{{ORDER}}", str(order_id)))


_HEAD = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Разбор для тренера — заказ №{{ORDER}}</title>
<style>
  :root{--bg:#f6f4ef;--card:#fff;--ink:#1b2430;--muted:#5d6672;--line:#e3dfd6;
        --red:#c0392b;--red-bg:#fbe9e7;--amber:#a86b00;--amber-bg:#fff4dc;--green:#2e7d4f;--accent:#1f3a5f}
  @media (prefers-color-scheme:dark){:root{--bg:#12161c;--card:#1b2129;--ink:#e8eaed;--muted:#9aa3ad;--line:#2c343e;
        --red:#ff7b6b;--red-bg:#3a1f1c;--amber:#f0b44c;--amber-bg:#3a2e15;--green:#6fcf97;--accent:#8fb4e3}}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
  main{max-width:860px;margin:0 auto;padding:20px 16px 60px}
  h1{font-size:24px;margin:0 0 4px}
  h2{font-size:18px;margin:28px 0 10px}
  .sub{color:var(--muted);margin:0 0 16px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:12px 0;overflow-x:auto}
  .deadline{border-left:4px solid var(--accent)}
  video,img{width:100%;border-radius:8px;display:block;background:#000}
  .media{display:grid;gap:12px}
  @media (min-width:760px){.media.two{grid-template-columns:1fr 1fr}}
  .media.two video,.media.two img{max-height:80vh;object-fit:contain}
  .cap{color:var(--muted);font-size:13px;margin:6px 0 0}
  table{width:100%;border-collapse:collapse;font-size:15px}
  th,td{text-align:left;padding:8px 6px;border-bottom:1px solid var(--line);vertical-align:top}
  th{font-size:13px;color:var(--muted);font-weight:600}
  td.v{font-weight:700;white-space:nowrap}
  tr.bad td{background:var(--red-bg)} tr.bad td.v{color:var(--red)}
  tr.warn td{background:var(--amber-bg)} tr.warn td.v{color:var(--amber)}
  .tag{font-size:12px;font-weight:700;border-radius:10px;padding:1px 8px;white-space:nowrap}
  .tag.bad{color:var(--red);border:1px solid var(--red)} .tag.warn{color:var(--amber);border:1px solid var(--amber)}
  .tag.ok{color:var(--green);border:1px solid var(--green)} .tag.na{color:var(--muted);border:1px solid var(--line)}
  ul{padding-left:20px;margin:6px 0} li{margin:4px 0}
  details summary{cursor:pointer;color:var(--accent);font-weight:600}
  blockquote{margin:10px 0 0;padding:8px 12px;border-left:3px solid var(--line);color:var(--muted);font-size:14px}
  label{display:block;font-weight:700;margin:16px 0 4px}
  .hint{color:var(--muted);font-size:13px;font-weight:400}
  textarea{width:100%;min-height:84px;padding:10px;border:1px solid var(--line);border-radius:8px;
           background:var(--bg);color:var(--ink);font:inherit;resize:vertical}
  .count{font-size:12px;color:var(--muted);text-align:right;margin-top:2px}
  .count.over{color:var(--red);font-weight:700}
  button{margin-top:18px;width:100%;padding:14px;border:0;border-radius:10px;background:var(--accent);
         color:#fff;font:600 16px/1 inherit;cursor:pointer}
  @media (prefers-color-scheme:dark){button{color:#12161c}}
  .ok-msg{color:var(--green);text-align:center;margin-top:8px;min-height:1.5em}
  .ex{font-size:14px} .ex b{display:block;margin-top:6px}
</style>
</head>
"""

_TAIL = """
  <h2>Как писать</h2>
  <div class="card ex">
    Обращайся на «ты», как на тренировке. Не пересказывай цифры: таблицу клиент уже видел. Нужно то, что видит глаз.
    <b>✅ Хорошо:</b> «В нижней точке ты тянешь носок вниз, будто дотягиваешься до педали. Опусти седло на 5 мм и пересними с той же стороны».
    <b>❌ Бесполезно:</b> «Угол в колене 150°, выше нормы».
    <p class="cap">Без медицинских диагнозов. Если нужен физиотерапевт или байкфиттер, так и напиши.</p>
  </div>

  <h2>Твой ответ</h2>
  <div class="card">
    <label for="f0">Главное, что изменить <span class="hint">одно действие на эту неделю, до 25 слов</span></label>
    <textarea id="f0" data-max="25"></textarea><div class="count" id="c0"></div>

    <label for="f1">Заметка 1 <span class="hint">до 35 слов</span></label>
    <textarea id="f1" data-max="35"></textarea><div class="count" id="c1"></div>

    <label for="f2">Заметка 2 <span class="hint">до 35 слов</span></label>
    <textarea id="f2" data-max="35"></textarea><div class="count" id="c2"></div>

    <label for="f3">Заметка 3 <span class="hint">до 35 слов</span></label>
    <textarea id="f3" data-max="35"></textarea><div class="count" id="c3"></div>

    <button id="copy" type="button">Скопировать ответ</button>
    <div class="ok-msg" id="msg"></div>
    <p class="cap">Нажми, вставь в Telegram и отправь Артуру. Черновик сохраняется в этом браузере, так что можно закрыть страницу и вернуться.</p>
  </div>

  <p class="cap">🔒 Видео и картинки — личные материалы клиента. Пожалуйста, не пересылай их и удали этот файл после отправки ответа.</p>
</main>
<script>
(function(){
  var KEY='flapp-coach-order{{ORDER}}-draft';
  var names=['Главное, что изменить','Заметка 1','Заметка 2','Заметка 3'];
  var fields=[0,1,2,3].map(function(i){return document.getElementById('f'+i);});
  var saved={};
  try{saved=JSON.parse(localStorage.getItem(KEY)||'{}')||{};}catch(e){}
  function words(s){var t=s.trim();return t?t.split(/\\s+/).length:0;}
  function update(){
    var draft={};
    fields.forEach(function(f,i){
      draft[i]=f.value;
      var c=document.getElementById('c'+i); if(!c) return;
      var n=words(f.value), max=+f.dataset.max;
      c.textContent=n+' / '+max+' слов';
      c.className='count'+(n>max?' over':'');
    });
    try{localStorage.setItem(KEY,JSON.stringify(draft));}catch(e){}
  }
  fields.forEach(function(f,i){ if(saved[i]) f.value=saved[i]; f.addEventListener('input',update); });
  update();
  document.getElementById('copy').addEventListener('click',function(){
    var text='Coach Notes, заказ №{{ORDER}}\\n\\n'+fields.map(function(f,i){
      return names[i]+':\\n'+(f.value.trim()||'—');}).join('\\n\\n');
    var msg=document.getElementById('msg');
    function done(){msg.textContent='Скопировано. Вставь в чат с Артуром.';}
    function fallback(){
      var ta=document.createElement('textarea'); ta.value=text; document.body.appendChild(ta);
      ta.select(); try{document.execCommand('copy'); done();}catch(e){msg.textContent='Не получилось, выдели текст в полях вручную.';}
      document.body.removeChild(ta);
    }
    if(navigator.clipboard&&navigator.clipboard.writeText){navigator.clipboard.writeText(text).then(done,fallback);}else{fallback();}
  });
})();
</script>
</body>
</html>
"""
