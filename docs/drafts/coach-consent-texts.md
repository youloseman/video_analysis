# Coach area · тексты согласия и правки политики (черновик на согласование)

> Статус: черновик, 2026-11-06. Ничего из этого не опубликовано. Публикуется только после прочтения Артуром.
> Контекст: docs/COACH_AREA_PLAN_RU.md §1 (две модели) и §1a (комментарии тренеров).

Принцип: атлет всегда знает, кто видит его материалы, и может это прекратить одной кнопкой. Текст без юридического тона, в голосе приложения.

---

## 1. Экран атлета «My coach» (модель A) — чекбокс согласия

**Заголовок:** Link a coach

**Над полем кода:**
Your coach gives you a code. Enter it here and they can see what you film — the analyses, the frames and the clips — and leave notes on your reports.

**Чекбокс (обязательный, снят по умолчанию):**
☐ I want **{Coach name}** to see my analyses, my annotated frames and my uploaded clips, and to write notes on my reports. I can switch this off at any time on this screen.

**Под чекбоксом, мелко:**
Your coach sees exactly what your plan shows you — no more. Clips are visible for as long as they are kept on your plan (7 days free, 30 days paid). Unlinking stops their access at once; notes they already wrote stay on your reports.

**Кнопка:** Link {Coach name}

**После привязки (та же карточка):**
{Coach name} can see your analyses and clips since {date}.
[Unlink] — stops their access right away.

**Письмо атлету после привязки (service mail, без цен):**
Subject: {Coach name} is now linked to your Flapp
You linked {Coach name} as your coach on Flapp. They can see your analyses, annotated frames and clips, and leave notes on your reports. You can unlink them any time from Account → My coach.

---

## 2. Первая чужая загрузка тренером (модель B) — подтверждение

Показывается один раз на аккаунт, когда тренер впервые выбирает атлета из ростера на Analyze (или грузит пакет), и повторяется только если тренер снимет галочку в Account.

**Заголовок:** Someone else's clip

**Текст:**
You're about to analyse footage of another person. Flapp processes it the same way as yours, and the report lands in your account under their name.

**Чекбокс (обязательный):**
☐ I have this person's permission to upload their footage and to keep their report in my account, and I'll delete it if they ask me to.

**Кнопка:** Got it, continue

---

## 3. Правка privacy.html

### §2 «Your uploaded video or photo» — срок хранения

Сейчас: «…**7 days** on the free plan and **30 days** on a paid plan…»

Предлагаемый текст:
> …**7 days** on the free plan, **30 days** on a paid plan and **60 days** on the Coach plan — a coach reviews a clip next to the one filmed after the change, weeks later…

### §3 — новый абзац после «If you do not buy an Expert Review or Coach Notes, no human reviews your footage»

Предлагаемый абзац:
> **Your coach, if you link one.** If you enter a coach's code in your account and tick the box, that coach can see your analyses, your annotated frames and your uploaded clips, and can write notes on your reports — for as long as the link is on. You can unlink them at any time from Account → My coach, and their access stops immediately. Flapp does not add a coach to your account on its own, and a coach never sees footage you did not choose to share this way.

> **If a coach uploads your clip.** A coach or bike fitter on the Coach plan may upload footage of the people they work with, into their own account. That footage and its report belong to the coach's account, are kept on the Coach plan's schedule (section 2) and are deleted when the coach deletes them. The coach confirms to us that they have your permission; if you believe a coach uploaded your footage without it, write to support@getflapp.com and we will remove it.

### §10 таблица сроков — добавить строку

| Uploaded video or photo (Coach plan) | 60 days from the analysis, then deleted. |

---

## 4. Terms — один пункт для тренеров

> **Coach plan.** If you upload footage of other people, you confirm you have their permission to do so and to keep their reports in your account, and you will remove them on request. Reports you hand to your athletes carry your name and mark; Flapp is named as the instrument that measured them. Flapp's measurements are a tool for your judgement, not a diagnosis.

---

## 5. Онбординг тренера — что он видит через месяц (одна карточка на экране Athletes, закрывается)

> **What stays, what goes.** Numbers, annotated frames, kinograms and the AI notes stay with every analysis for as long as the entry exists. Uploaded clips and overlay videos are kept for 60 days on this plan. A photo analysis keeps its frame and numbers but has no server-side result to reopen later — so correct its joints while it is fresh.

---

## Открытые вопросы к Артуру

1. §2: 60 дней для Coach — подтвердить (стоимость ≈ центы, см. план §4a).
2. §3, абзац про загрузку тренером: оставить фразу про «write to support@ and we will remove it»? Это обязательство удалять по запросу третьего лица.
3. Терминология: «coach» везде, или «coach or fitter»? Предлагаю «coach» в тексте, «Coach plan» как имя тарифа.
