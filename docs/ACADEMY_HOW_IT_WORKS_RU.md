# Как работает раздел Academy

> Практическое руководство по устройству образовательного раздела **Academy** в Motus / CoachPowerBoost: как хранятся и создаются статьи, как они парсятся на бэкенде, как рендерятся на фронтенде и как устроен механизм интерактивности (визуализации, персонализация, Science Bites).

---

## 1. Что такое Academy

Academy — образовательный блок приложения, доступный по маршруту `/learn` (пункт сайдбара **«Academy»**, иконка `BookOpen`). Состоит из двух независимых частей:

1. **Science Academy** — научные статьи о тренировках. Хранятся как **Markdown-файлы**, а не в БД.
2. **News Blog** (`/blog`) — агрегация RSS-лент с AI-фильтрацией через Gemini. Отдельная подсистема, здесь не рассматривается подробно (см. [ACADEMY_ARCHITECTURE.md](ACADEMY_ARCHITECTURE.md)).

Ключевая идея Academy: **статья = один `.md`-файл**. Никакой CMS, никакой админки, никакой таблицы в базе. Чтобы добавить статью — кладёшь файл в папку, чтобы отредактировать — правишь файл. Всё остальное (парсинг, кеш, API, рендер) происходит автоматически.

---

## 2. Общая схема потока данных

```
Markdown-файлы                     Бэкенд (FastAPI)                    Фронтенд (Next.js)
backend/content/academy/*.md  ──►  article_parser.py  ──►  /science/academy/*  ──►  React Query  ──►  Компоненты
       (контент)                   (парсинг + кеш)          (JSON API)              (кеш 10 мин)      (рендер)
                                          │
                                   Science KB  ──►  visualization_data
                                   (данные для графиков)
```

- **Контент** живёт в файлах.
- **Парсер** превращает Markdown в структуру секций + метаданные, кеширует в памяти.
- **API** отдаёт JSON и подмешивает данные для интерактивных визуализаций из Science KB.
- **Фронтенд** запрашивает JSON, рендерит секции по типам и подключает интерактивные компоненты.

---

## 3. Как хранятся статьи (файловая система)

**Путь:** `backend/content/academy/`

На данный момент там ~30 статей на английском (`slug.md`) + их русские переводы (`slug.ru.md`) — всего ~60 файлов.

### Правила именования

| Файл | Язык | Назначение |
|------|------|------------|
| `polarized_training.md` | EN | Каноническая (базовая) версия |
| `polarized_training.ru.md` | RU | Русский перевод той же статьи |

- Английская версия — **обязательна** и является фолбэком.
- Русская версия (`.ru.md`) — опциональна. Если её нет, для русской локали показывается английский текст с плашкой-предупреждением «статья доступна только на английском».
- Определение языка файла — по количеству точек в имени: `slug.md` = EN, `slug.ru.md` = локализованный (см. `filepath.name.count(".") > 1` в парсере).

---

## 4. Структура файла статьи

Каждый `.md`-файл = **YAML frontmatter** + **тело, разбитое на секции HTML-комментариями**.

### 4.1 YAML frontmatter (метаданные)

```yaml
---
slug: "race-day-nutrition"                     # уникальный идентификатор (= имя в URL)
title: "Race Day Nutrition: How Many Carbs..."  # заголовок
category: "special_topics"                      # одна из 6 категорий (см. ниже)
category_label: "Special Topics"                # человекочитаемая метка (опц.)
read_time: 8                                    # время чтения в минутах
visualization_type: "nutrition_calculator"      # тип интерактивной визуализации (опц.)
featured: false                                 # выводить как featured-статью на хабе
related:                                         # slug'и связанных статей (блок "Related")
  - nutrition-basics
sources:                                         # список научных источников
  - author: "Jeukendrup"
    year: 2014
    title: "A step towards personalized sports nutrition..."
    finding: "Carbohydrate intake should match duration..."
---
```

**Что делает каждое поле:**

| Поле | Обязательно | Роль |
|------|-------------|------|
| `slug` | да | Уникальный ID и часть URL `/learn/{slug}`. Если не указан — берётся из имени файла |
| `title` | да | Заголовок статьи |
| `category` | да | Категория (одна из 6). По ней определяются иконка, цвет, персонализация |
| `read_time` | нет (по умолч. 5) | «X min read» |
| `visualization_type` | нет | Включает интерактивную визуализацию внизу статьи |
| `featured` | нет (false) | Показывает статью крупным блоком в начале хаба |
| `related` | нет | Массив `slug`, из которых строится блок «Related articles» |
| `sources` | нет | Библиография внизу статьи |

### 4.2 Тело статьи: секции-маркеры

Тело разбивается на типизированные секции с помощью HTML-комментариев `<!-- marker -->`. Парсер знает следующие маркеры:

| Маркер | Тип секции | Заголовок по умолчанию | Свёрнута? | Как отображается |
|--------|-----------|------------------------|-----------|------------------|
| `<!-- hook -->` | `hook` | — | нет | Крупный курсивный вводный абзац (teal-акцент), без markdown-обёртки |
| `<!-- summary -->` | `text` | «The Simple Version» | нет | Обычный текст-конспект |
| `<!-- theory -->` | `theory` | «How It Works» | нет | Основная теория (markdown → HTML: заголовки, таблицы, списки) |
| `<!-- worked_example -->` | `worked_example` | «Example» | нет | Карточка с примером-расчётом (amber-иконка, тёмный фон) |
| `<!-- practical_rules -->` | `practical_rules` | «Practical Rules» | нет | Чеклист правил с иконкой галочки |
| `<!-- evidence -->` | `evidence_detail` | «Evidence Base» | **да** | Сворачиваемый блок с научными деталями |

> Обрати внимание: маркер в файле пишется `<!-- evidence -->`, но внутренний тип секции — `evidence_detail`.

Пример тела:

```markdown
<!-- hook -->
Nutrition is the fourth discipline of triathlon...

<!-- summary -->
For age-group triathletes the evidence-based sweet spot is 60-90 g/hour...

<!-- theory -->
## The proven zone: 60-90 g/hour
Decades of research confirm...

| Exercise Duration | Carbohydrate Recommendation |
|---|---|
| Under 1 hour | Mouth rinse or small amounts |

<!-- worked_example -->
## Example: 75 kg athlete, Half Ironman, 5:30 target
...

<!-- practical_rules -->
1. **Do not copy pro strategies.** ...

<!-- evidence -->
## Evidence base
Jeukendrup (2014) provided the definitive framework...
```

**Внутри секций работает полноценный Markdown** (расширения `tables` и `fenced_code`): заголовки `##`/`###`, таблицы, списки, `**жирный**`, код. Исключение — секция `hook`: она остаётся простым текстом без markdown-обёртки.

Если в файле **нет ни одного маркера**, всё тело считается одной `text`-секцией.

---

## 5. Категории статей

Определены в `article_parser.py` (`_CATEGORY_META`). Каждая задаёт метку и иконку; на фронтенде добавляется свой цвет:

| Ключ категории | Метка | Иконка | Цвет полоски (фронт) |
|----------------|-------|--------|----------------------|
| `training_models` | Training Models | `Layers` | blue |
| `load_management` | Load Management | `Shield` | amber |
| `performance_factors` | Performance Factors | `TrendingUp` | emerald |
| `periodization` | Periodization | `Calendar` | purple |
| `sport_specific` | Sport-Specific | `Activity` | cyan |
| `special_topics` | Special Topics | `BookOpen` | rose |

Категория влияет на: иконку/цвет карточки, фильтр на хабе, prev/next-навигацию (внутри категории) и **логику персонализации** (см. §9).

---

## 6. Как создаётся статья (пошагово)

Чтобы добавить новую статью, **не нужно трогать код** — достаточно создать файл:

1. **Создать файл** `backend/content/academy/<slug>.md` (подчёркивания в имени файла, дефисы в `slug` — так исторически сложилось; slug из frontmatter — источник истины для URL).
2. **Заполнить frontmatter** — обязательно `slug`, `title`, `category`. Категория должна быть одной из 6 существующих.
3. **Написать тело** с секциями `<!-- hook -->`, `<!-- summary -->`, `<!-- theory -->`, `<!-- worked_example -->`, `<!-- practical_rules -->`, `<!-- evidence -->`.
4. **(Опционально) добавить визуализацию** — указать `visualization_type` (см. §8) и убедиться, что бэкенд умеет строить для неё данные.
5. **(Опционально) связать** с другими статьями через `related: [slug1, slug2]`.
6. **(Опционально) перевести** — создать `<slug>.ru.md` с тем же `slug`, но русским содержимым.
7. **Перезапустить бэкенд** (или дождаться, пока сбросится in-memory кеш) — статья появится в списке автоматически.

Тон, длина и правила написания подробно описаны в [ACADEMY_ARTICLES_WRITING_PROMPT.md](ACADEMY_ARTICLES_WRITING_PROMPT.md) (напр. 800–1200 слов, аналогии из жизни, числовой пример в `worked_example`, научные термины на английском при первом упоминании).

**Проверки при добавлении:** валидный YAML, наличие обязательных секций, отсутствие «битых» `related`-ссылок (slug, которого нет). Скрипты валидации приведены в конце writing-prompt-документа.

---

## 7. Бэкенд: парсер и API

### 7.1 Парсер — `backend/app/services/science/article_parser.py`

Что делает:

- Читает `.md`-файлы через библиотеки `frontmatter` (YAML) и `markdown` (тело).
- `_parse_sections(body)` — режет тело по маркерам `<!-- ... -->` на типизированные секции, конвертирует markdown → HTML.
- Из секции `summary`/`text` вытягивает plain-text превью (первые 300 символов) для карточек в списке.
- **Кеширует всё в памяти** по языкам: `_articles_cache = { "en": {...}, "ru": {...} }`.
- Для RU: сначала грузит полный EN-набор как фолбэк, затем «наслаивает» переводы из `*.ru.md`. Непереведённые статьи остаются на EN и помечаются `language: "en"`.

Публичные функции:

| Функция | Назначение |
|---------|------------|
| `get_all_articles_metadata(language)` | Лёгкие метаданные всех статей (для списка/хаба) — без полного тела |
| `get_article(slug, language)` | Полная статья: секции + источники + метаданные |
| `get_articles_by_category(category, language)` | Фильтр по категории |
| `get_categories(language)` | Категории с количеством статей |
| `invalidate_cache()` | Сброс in-memory кеша (для тестов / hot-reload) |

### 7.2 API — `backend/app/api/academy.py`

Роутер регистрируется с префиксом `/science` (тег `Academy`) в `backend/app/main.py`:

```python
app.include_router(academy.router, prefix="/science", tags=["Academy"])
```

Эндпоинты:

| Метод | Путь | Что возвращает |
|-------|------|----------------|
| `GET` | `/science/academy/articles?language=ru` | `{ articles: [...], categories: {...} }` — список метаданных + категории со счётчиками |
| `GET` | `/science/academy/articles/{slug}?language=ru` | `{ found: true, article: {...} }` — полная статья + `visualization_data` |

При запросе **конкретной** статьи бэкенд дополнительно:

1. Берёт `visualization_type` из frontmatter.
2. Через `_build_visualization_data(...)` строит JSON-данные для интерактивной визуализации, подтягивая цифры из **Science KB** (`Science/knowledge_base.py`, ленивая загрузка).
3. Кладёт результат в поле `article["visualization_data"]`.

Так статья и данные для её графика приходят на фронт **одним запросом**.

---

## 8. Механизм интерактивности: визуализации

Это ключевая «фишка» Academy — многие статьи заканчиваются **интерактивным блоком** (график/калькулятор/таймлайн), а не просто текстом.

### 8.1 Как это работает end-to-end

```
frontmatter: visualization_type  ──►  backend _build_visualization_data()  ──►  visualization_data (JSON)
                                                    │                                    │
                                          Science KB (зоны, TID-модели)                  ▼
                                                                          ArticleVisualization (dispatcher)
                                                                                         │
                                                                          конкретный React-компонент графика
```

### 8.2 Поддерживаемые типы визуализаций

Диспетчер — `frontend/.../learn/[slug]/_components/article-visualization.tsx`. Он смотрит на `type` и рендерит нужный компонент:

| `visualization_type` | Компонент | Что показывает | Источник данных |
|----------------------|-----------|----------------|-----------------|
| `tid_comparison` | `TIDComparisonChart` | Stacked-бары распределения интенсивности: Polarized / Pyramidal / Norwegian / Threshold | Science KB (`get_tid_model`) |
| `zone_chart` | `ZoneBarChart` | Тренировочные зоны с min/max и 7 цветами (мощность/темп/ЧСС) | Science KB (`get_zones`) |
| `training_week` | `TrainingWeekHeatmap` | Тепловая карта AM/PM на 7 дней для выбранной модели | slug → модель |
| `periodization_timeline` | `PeriodizationTimelineGeneric` | Таймлайн фаз (напр. «Build 10 нед → Maintain 15 нед») | статичные фазы по slug |
| `acwr_gauge` | `ACWRGaugeArc` | Датчик Acute:Chronic Workload Ratio | значение (заготовка) |
| `pmc_mini` | `PMCMiniChart` | Мини-график CTL/ATL/TSB | данные (опц.) |
| `nutrition_calculator` | `NutritionCalculator` | **Полностью интерактивный калькулятор** углеводов/натрия/кофеина по весу, дистанции, целевому времени | клиентский, без Science KB |

**Важные детали механизма:**

- Если `visualization_type` не указан — блок визуализации просто не рендерится.
- `nutrition_calculator` — единственный полностью клиентский: бэкенд отдаёт лишь `{"type": "nutrition_calculator"}` (непустой payload, чтобы фронт отрисовал блок), а все расчёты идут в браузере. Его оборачивают в тёмную тему принудительно, т.к. компонент свёрстан под dark.
- Остальным нужен Science KB. Если KB недоступен — визуализация тихо не показывается.
- Компоненты графиков лежат в `frontend/src/components/academy/` и `frontend/src/components/features/training/`.

---

## 9. Механизм интерактивности: персонализация (Personal Connection)

Под текстом каждой статьи может появиться карточка **«Your data»** — она подставляет **реальные показатели пользователя** в контекст статьи.

**Компонент:** `frontend/src/components/academy/personal-connection.tsx`

Как работает:

- Берёт профиль из `useProfileStore()` (FTP, CSS, LTHR, VO2max, пороговый темп, зоны, возраст).
- По паре **(`category`, `slug`)** решает, какие метрики релевантны, и формирует строки. Примеры:
  - Категория `training_models` + есть FTP → показывает FTP и границы «лёгкой»/«тяжёлой» зон (55–75% и 106–120% FTP).
  - Статья `ftp-power-zones` → FTP + разбивку велозон из профиля.
  - Статья `swim-css` → CSS-темп; `lactate-threshold` → LTHR; `run-vdot` → пороговый беговой темп.
  - Статья `training-after-40` → возраст (если ≥ 40).
- Если релевантных данных нет (нет профиля или нет нужной метрики) — карточка **не рендерится**.

Результат: обезличенная научная статья превращается в «про тебя» — «Твой FTP: 287 W → лёгкая зона 158–215 W».

---

## 10. Механизм интерактивности: Science Bites (на хабе)

На странице-хабе `/learn` над списком статей есть горизонтальная лента **Science Bites** — короткие карточки-подсказки двух видов:

- **Concept** (научное понятие) и **Workout** (тип тренировки).
- Тянутся отдельными эндпоинтами (`getConceptCards`, `getWorkoutCards`), **чередуются** (concept / workout / concept / …).
- Карточка кликабельна: по клику **разворачивается** в блок с подробностями (`ScienceBiteExpanded`) — простое объяснение, научная деталь, аналогия / «почему это важно» / fun fact.

Это отдельная от статей подсистема (Science Cards), но живёт на той же странице Academy и усиливает ощущение интерактивности.

---

## 11. Фронтенд: страницы и рендер

### 11.1 Хаб — `/learn`

Файл: `frontend/src/app/[locale]/(dashboard)/learn/page.tsx`

Содержит:

- Hero-заголовок и подзаголовок.
- Ленту **Science Bites** (см. §10).
- **Featured-статью** крупным блоком (если у какой-то `featured: true`), только на вкладке «All».
- **Табы-фильтры** по 6 категориям + «All», со счётчиками. Категория синхронизируется с URL-параметром `?category=` (для перехода из хлебных крошек).
- **Грид карточек** статей (3 колонки на десктопе). На карточке: цветная полоска категории, иконка, метка, время чтения, бейдж «Science», бейдж «Interactive chart» (если есть визуализация), для RU-локали — бейдж «EN», если перевода нет.
- Кросс-ссылку на Blog.

Данные грузятся через React Query (`queryKey: ["academy-articles", locale]`, `staleTime: 10 мин`).

### 11.2 Статья — `/learn/[slug]`

Файл: `frontend/src/app/[locale]/(dashboard)/learn/[slug]/page.tsx`

Структура рендера:

1. **Хлебные крошки**: Academy → Категория → Заголовок.
2. **TOC (оглавление)**:
   - Десктоп — sticky-сайдбар слева, активный пункт подсвечивается через `IntersectionObserver`.
   - Мобайл — дропдаун «Jump to».
3. **Шапка**: бейдж категории, время чтения, заголовок, (для RU без перевода) плашка «только на английском».
4. **Тело** — цикл по `article.sections`, каждая секция рендерится своим компонентом по `section.type` (см. таблицу ниже). Каждой секции ставится `id="section-<type>"` и `ref` для TOC-обсервера.
5. **Визуализация** (если есть `visualization_type` + `visualization_data`) — через `ArticleVisualization`.
6. **Personal Connection** — карточка с данными пользователя.
7. **Sources** — нумерованный список источников.
8. **Related articles** — грид из связанных статей.
9. **Prev/Next** — навигация внутри той же категории.

Рендереры секций — `frontend/.../learn/[slug]/_components/section-renderers.tsx`:

| Тип секции | Компонент | Оформление |
|-----------|-----------|------------|
| `hook` | `SectionHook` | Крупный курсив, teal, без рамки |
| `text` | `SectionText` | Обычная проза (`prose-academy`) |
| `theory` | `SectionTheory` | Проза с заголовками/таблицами/списками |
| `worked_example` | `SectionWorkedExample` | Карточка с тёмным фоном и amber-иконкой |
| `practical_rules` | `SectionPracticalRules` | Чеклист с иконкой галочки |
| `evidence_detail` | `SectionEvidenceDetail` | **Сворачиваемый** блок (`CollapsibleSection`), свёрнут по умолчанию |

HTML секций рендерится через `dangerouslySetInnerHTML` с набором Tailwind-стилей (`HtmlContent`), которые стилизуют `h2/h3/p/ul/ol/table` под единый вид Academy.

> Legacy-режим: если у статьи вдруг нет `sections`, есть фолбэк `LegacyArticleBody`, собирающий тело из отдельных полей (hook/analogy/why_it_matters/…). Для новых markdown-статей он не нужен.

---

## 12. Локализация (i18n)

- Язык передаётся в API параметром `language` (берётся из `locale` next-intl).
- Бэкенд: `.ru.md` перекрывает `.md`; при отсутствии перевода — EN-фолбенд с пометкой `language: "en"`.
- Фронтенд для RU-локали с EN-фолбэком показывает бейдж «EN» на карточке и плашку-предупреждение на странице статьи.
- UI-обёртка (заголовки, кнопки, метки) переводится через namespace `pages.learn` и `pages.academyComponents` (файлы `frontend/messages/*.json`).
- Согласно правилу проекта, **спортивные термины** (FTP, TSS, CTL, threshold, taper, brick, zone и т.п.) в русском тексте остаются на латинице — переводится только окружающий UI.

---

## 13. Кеширование

- **Бэкенд**: in-memory кеш статей по языкам. Первый запрос парсит все файлы, дальше отдаёт из памяти. Сброс — `invalidate_cache()` или рестарт процесса.
- **Фронтенд**: React Query, `staleTime` 10 минут для статей, 30 минут для Science Bites.

Практический вывод: после правки `.md`-файла на проде нужен рестарт бэкенда (или инвалидация кеша), а у пользователя обновление подтянется по истечении staleTime.

---

## 14. Ключевые файлы (шпаргалка)

| Слой | Файл | Роль |
|------|------|------|
| Контент | `backend/content/academy/*.md` | Сами статьи (EN + `.ru.md`) |
| Парсер | `backend/app/services/science/article_parser.py` | Markdown → секции + метаданные + кеш |
| API | `backend/app/api/academy.py` | Эндпоинты + сборка `visualization_data` |
| Регистрация | `backend/app/main.py` | `include_router(..., prefix="/science")` |
| Данные графиков | `Science/knowledge_base.py` | Зоны, TID-модели для визуализаций |
| API-клиент | `frontend/src/lib/api-client.ts` | `scienceApi.getAcademyArticles/getAcademyArticle` |
| Хаб | `frontend/src/app/[locale]/(dashboard)/learn/page.tsx` | Список, фильтры, Science Bites |
| Статья | `frontend/src/app/[locale]/(dashboard)/learn/[slug]/page.tsx` | Рендер статьи + TOC |
| Секции | `.../learn/[slug]/_components/section-renderers.tsx` | Рендер каждого типа секции |
| Визуализации | `.../learn/[slug]/_components/article-visualization.tsx` | Диспетчер графиков |
| Компоненты графиков | `frontend/src/components/academy/*` | TID/Zone/PMC/heatmap/PersonalConnection |
| Персонализация | `frontend/src/components/academy/personal-connection.tsx` | Данные пользователя в контексте статьи |
| Калькулятор | `frontend/src/components/learn/nutrition-calculator.tsx` | Интерактивный расчёт питания |

---

## 15. Итог одним абзацем

Academy — это «файловый движок статей»: каждая статья — Markdown-файл с YAML-метаданными и секциями, размеченными HTML-комментариями. Бэкенд парсит эти файлы в структуру секций, кеширует в памяти и отдаёт по `/science/academy/*`, попутно подмешивая данные для интерактивных графиков из Science KB. Фронтенд рендерит секции разными компонентами по их типу и добавляет три слоя интерактивности: **визуализации** (графики/калькулятор внизу статьи), **персонализацию** (реальные FTP/CSS/LTHR пользователя в контексте) и **Science Bites** (разворачиваемые карточки на хабе). Чтобы добавить или изменить статью, достаточно положить/отредактировать `.md`-файл — код трогать не нужно.
