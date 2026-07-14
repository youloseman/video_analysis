# Academy — конвейер создания статьи (NotebookLM → готовая статья)

Рабочий процесс: **книга(и) → NotebookLM вытаскивает факты и цитаты → ты
проверяешь по первоисточникам → пишешь короткую статью → заливаешь в движок.**

Движок уже готов (см. [ACADEMY_HOW_IT_WORKS_RU.md](ACADEMY_HOW_IT_WORKS_RU.md) для
Motus-версии; в этом проекте — server-rendered, файлы в
`backend/content/academy/*.md`). Этот документ — про *контент*.

---

## Золотое правило

> Книга даёт **тему и каркас**. Каждый ключевой факт статьи опирается на
> **первичный источник** (рецензируемое исследование), который ты лично
> проверил и цитируешь в `sources`. NotebookLM — помощник, не автор.

Причина: спорт/здоровье = YMYL-ниша, Google строго оценивает достоверность.
Устаревший или выдуманный факт вредит и доверию, и ранжированию.

---

## Шаг 1. Собрать источники в NotebookLM

Один notebook = одна статья (не мешай темы). Загрузи книги, затем спрашивай.

**Промпт A — вытащить цитаты (это главное, не «дай факты»):**

```
List every scientific study, researcher and year these sources cite about
[ТЕМА, напр. "aerodynamic drag and rider position in cycling"]. For each,
give: (1) the specific claim, (2) the citation (author, year, journal if
available), (3) a direct quote or close paraphrase of what the source says.
Do not invent citations — if a claim has no citation, say so.
```

**Промпт B — механизм и что реально доказано:**

```
Explain the mechanism of [ТЕМА] in plain English. Then separate: what does the
peer-reviewed evidence actually support, versus what is coaching folklore or
tradition repeated without strong evidence? Be explicit about the difference.
```

**Промпт C — цифры для таблицы:**

```
Give me concrete numbers with their sources for [ТЕМА]: e.g. percentages,
ranges, effect sizes. For each number, name the study it comes from. Flag any
number that seems to be an estimate or rule-of-thumb rather than a measured
finding.
```

**Промпт D — черновик (после проверки фактов!):**

```
Draft a 700–1000 word article for recreational endurance athletes on [ТЕМА].
Tone: practical, evidence-based, no hype. Structure: hook → mechanism → what
the evidence shows → the trade-offs/nuance → practical rules → bottom line.
Use only the verified facts I provide. Keep sport terms in English.
```

---

## Шаг 2. Проверка фактов (ТЫ, не NotebookLM)

Для каждого факта, который попадёт в статью:

- [ ] Нашёл первоисточник (гугл названия → **Google Scholar / PubMed**).
- [ ] Прочитал хотя бы **abstract** — claim реально там есть?
- [ ] Исследование не древнее ~15 лет **или** подтверждено более свежим.
- [ ] Выборка вменяемая (не n=6 на одном велоклубе).
- [ ] Цитата **существует** (NotebookLM иногда галлюцинирует авторов/годы).

Выжившие факты → в блок `sources`. Не выжившие → выкинуть или пометить в
тексте как «rule of thumb / debated», а не как факт.

---

## Шаг 3. Написать статью

- Пиши **своими словами**. Не копипасть NotebookLM и не копируй структуру книги
  (авторские права + Google не любит рерайт).
- Твоя ценность как редактора: отфильтровать, структурировать, добавить связь с
  продуктом («что видно на side-view видео»).
- Длина: 700–1200 слов. Первый абзац = hook (движок рендерит его крупным lede).

---

## Шаг 4. Залить в движок

1. Возьми черновик-заготовку `*.DRAFT.md` (напр.
   `bike-aero-position.DRAFT.md`) — файлы с суффиксом `.DRAFT.md`
   **не публикуются**, можно спокойно писать.
2. Заполни frontmatter (slug, title, description, category, sport, sources).
3. Вставь текст в markdown (## заголовки, таблицы `| a | b |`, списки, **жирный**).
4. Проверь локально:
   ```bash
   cd backend
   python -c "import sys; sys.path.insert(0,'.'); from app.services.academy import get_article, invalidate_cache; invalidate_cache(); a=get_article('ТВОЙ-slug'); print('OK' if a else 'НЕ НАЙДЕНА'); print(a.meta.title if a else ''); print('sources:', len(a.sources) if a else 0)"
   ```
5. **Опубликовать:** переименуй `slug.DRAFT.md` → `slug.md`.
6. Коммит + пуш → Railway передеплоит → статья живая на `/academy/slug`.

Категории: `running-form` · `running-economy` · `bike-position` · `bike-aero`.
`sport`: `run` | `bike` (задаёт цвет акцента).

---

## Черновой процесс = безопасно

- `*.DRAFT.md` в папке — игнорируется движком (не попадёт в поиск).
- Или в frontmatter: `draft: true` / `published: false` — тоже скрывает.
- Публикация — это одно переименование файла.

---

## Первая статья пайплайна

**Тема:** велопосадка / аэродинамика (угол корпуса и CdA).
**Категория:** `bike-aero`. **Заготовка:** `backend/content/academy/bike-aero-position.DRAFT.md`.

**Книги-источники:**
- *Bike Fit* — Phil Burt (ex-физио British Cycling / Team Sky) — ядро по биомеханике посадки.
- *Faster: The Obsessive Cyclist's Guide* — Michael Hutchinson — аэродинамика, CdA, «поза важнее железа».
- *The Midlife Cyclist* — Phil Cavell — компромисс аэро vs. устойчивая мощность/здоровье.
- (опц.) *Triathlon Science* — Friel & Vance (ред.) — глава про аэро и позицию, ближе к академическому.

**Ключевой факт для проверки:** на скоростях шоссе тело гонщика даёт основную
долю аэродинамического сопротивления (часто цитируют ~70–80%), а позиция корпуса
влияет на CdA сильнее апгрейдов железа. Подтвердить первичным ветротуннельным
исследованием, не только книгой.
