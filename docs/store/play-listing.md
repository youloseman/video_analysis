# Google Play — листинг Flapp

Готовые тексты и ответы на формы Play Console. Ассеты (иконка, feature
graphic, скриншоты) в репозитории не лежат — они перерисовываются из
`mobile/brand/*.svg` и из продакшена, см. раздел «Графика».

Аккаунт **personal**, поэтому для production-доступа Google требует закрытый
тест: 12 тестировщиков, непрерывно подписанных 14 дней. Порядок: аккаунт →
приложение → закрытый тест с первым AAB → собрать 12 человек → через 14 дней
заявка на production.

---

## 1. Store listing

**App name** (27/30)

```
Flapp: Run & Bike Technique
```

**Short description** (69/80)

```
One side-view clip, measured: joint angles, cadence, and what to fix.
```

**Full description** (~2100/4000)

```
Flapp turns one side-view clip of your run or ride into a measurement, not an opinion.

Film 5-15 seconds from the side with the phone upright. Flapp finds you in every frame, measures the joints through the whole stride or pedal stroke, and reports what it actually saw - with the frames to prove it.

RUNNING
• Joint angles through the stride: knee, hip, ankle, elbow, trunk, against reference ranges whose source is named
• Cadence, ground contact, flight time, vertical oscillation, overstride and foot strike
• A kinogram: the five key positions of one stride in a single image

CYCLING
• Knee angle at the top and bottom of the pedal stroke, saddle height in millimetres
• Hip over the crank axle, found automatically and confirmable by hand
• Trunk angle and an aero read-out in minutes over your race distance, not an abstract score
• Your position next to the field measured at the 2026 Ironman World Championship

THE PLAYER
Step frame by frame, jump to each foot contact or bottom of the stroke, and read the angle curve under the video. Compare two clips side by side, locked to the same point in the cycle.

OVER TIME
Every analysis is kept, so a change in your form is something you can see rather than remember. Progress charts, personal bests, and one ranked list of what to fix first.

HONEST BY DESIGN
A single phone camera has limits and Flapp says where they are. When the footage cannot support a number it is withheld rather than guessed - no score, no invented figure. Every reading says how much of the clip it was measured from, and frames the tracker got wrong are marked as artefacts instead of averaged in.

WHO IT IS FOR
Runners, cyclists and triathletes who want to know what their body is doing, and coaches who want a measured second opinion.

HOW IT WORKS
The analysis runs on our servers, so the app needs a network connection. One clip takes 30-60 seconds. Your footage is deleted automatically after a fixed retention period, and is never sold or used to train models.

Your clips and your account are yours: delete the account in the app and every analysis goes with it.
```

**Прочие поля**

| Поле | Значение |
|---|---|
| App category | Health & Fitness |
| Tags (до 5) | Running, Cycling, Fitness, Sports, Training |
| Email | support@getflapp.com |
| Website | https://getflapp.com |
| Privacy policy | https://getflapp.com/privacy |
| Phone | необязательно |

---

## 2. Графика

| Что | Размер | Файл |
|---|---|---|
| App icon | 512×512 PNG | `icon-512.png` |
| Feature graphic | 1024×500 PNG | `feature-1024x500.png` |
| Phone screenshots | 1080×1920 PNG, нужно ≥4 | `play-1-report`, `play-6-coach`, `play-3-analyze`, `play-4-dashboard`, `play-5-metrics` |

Планшетные скриншоты не загружаем: приложение телефонное, как и в App Store.
Play пометит его «not optimised for tablets» — публикацию это не блокирует.

---

## 3. Data safety

Самая муторная форма. Ответы ниже точны для текущей сборки, в которой покупок
внутри приложения нет вообще.

**Общее**

- Does your app collect or share any of the required user data types? → **Yes**
- Is all of the user data collected by your app encrypted in transit? → **Yes** (всё по HTTPS)
- Do you provide a way for users to request that their data be deleted? → **Yes**
  - URL: `https://getflapp.com/privacy#delete`
  - Удаление аккаунта также self-serve внутри приложения
- Advertising ID → **не используется**

**Собираем (collected), третьим лицам не передаём (not shared)**

| Тип | Linked to user | Обязательно | Назначение |
|---|---|---|---|
| Personal info → Email address | Да | Да | Account management |
| Photos and videos → Videos | Да | Да | App functionality |
| Photos and videos → Photos | Да | Нет (только bike-photo) | App functionality |
| Health and fitness → Fitness info | Да | Да | App functionality |
| App activity → App interactions | Да | Нет | Analytics |
| App info and performance → Crash logs | Да | Нет | Analytics |
| App info and performance → Diagnostics | Да | Нет | Analytics |
| Device or other IDs | Да | Нет | Analytics |

**Чего НЕ собираем** — явное «нет»: местоположение, контакты, календарь, SMS,
аудио, файлы, финансовая информация, платёжные данные, история поиска, список
установленных приложений.

Почему нет финансовых данных: в Android-сборке весь UI покупок скрыт
(`mobile/bridge/bridge.js`, раздел 4) — ни цен, ни кнопок, ни ссылок.

Почему «not shared»: PostHog, Railway и Google (Gemini) — обработчики,
работающие по нашему поручению. У Play это «collected», а не «shared».

---

## 4. Content rating

Анкета IARC. Категория: **Utility, Productivity, Communication, or Other**.

Насилие, секс, наркотики, азартные игры, ненормативная лексика, пугающий
контент → везде **No**.

Отдельно стоит ответить аккуратно:

- Does the app allow users to interact or exchange content with other users? → **No**
  («Share result» отдаёт картинку системному share-листу; общения внутри
  приложения нет)
- Does the app share the user's location with other users? → **No**
- Does the app allow users to purchase digital goods? → **No** (в этой сборке)
- Does the app contain user-generated content? → **No**

Ожидаемый результат: **Everyone / 3+**.

---

## 5. App content — остальные декларации

| Форма | Ответ |
|---|---|
| Target audience and content | **18 and over** (любая группа младше включает Families policy с отдельными требованиями) |
| Ads | **No ads** |
| App access | **All functionality is available without special access**, но дать ревьюверу тестовый аккаунт — см. ниже |
| Government apps | No |
| Financial features | None |
| Health apps declaration | Не применимо: Health Connect не используем |
| Data deletion | URL из раздела 3 |
| News app | No |

**App access — тестовый аккаунт**

Тот же, что отдавали Apple: `appreview@getflapp.com`. Пароль — в
`secrets/store-review-account.md` (в репозиторий не коммитится).

Текст в поле «Any other instructions»:

```
Sign in with the account above. It already has analyses in History, so Home,
Progress and History are populated.

To run a new one, download this 8-second sample clip on the device and save it
to the gallery: https://getflapp.com/media/review-sample-run.mp4
Then: Analyze -> "Run - video" -> choose the clip -> Analyze. 30-60 seconds.

The analysis runs on our server, so the device needs a network connection.

This build contains no in-app purchases, no subscriptions, no pricing screen
and no link to any purchase flow. The review account has full access so every
feature is visible.

Account deletion: Menu -> Delete account.
```

---

## 6. Закрытый тест: 12 человек на 14 дней

Google засчитывает только **closed testing**; internal testing не считается.

1. Play Console → Testing → **Closed testing** → создать трек (подойдёт
   дефолтный «Alpha»).
2. Testers → Create email list → Gmail-адреса. Нужно **минимум 12**, и они
   должны оставаться подписанными все 14 дней — если кто-то отпишется, счётчик
   может сброситься. Брать с запасом: **15–16**.
3. Загрузить AAB в этот трек и опубликовать.
4. Скопировать opt-in ссылку и разослать. Каждый должен **принять приглашение
   и установить приложение**: Google смотрит на вовлечённость, а не только на
   факт подписки.
5. Через 14 дней появится **Apply for production access** — анкета о том, как
   шло тестирование и что по его итогам изменилось.

Где брать людей: активные пользователи Flapp с Android, триатлон-клуб,
велоклуб. Это не «поставьте галочку» — приложение нужно реально открыть хотя
бы раз, иначе заявку могут отклонить за недостаточное тестирование.

---

## 7. Сборка

AAB собирает Codemagic, workflow `android-release`, по тому же тегу `mobile/*`,
что и iOS (`scripts/mobile-release.sh`). Артефакт скачивается из билда и
загружается в Play Console руками; автоматическая публикация через
service account расписана закомментированной в `codemagic.yaml`.

Keystore: `secrets/android/flapp-upload.jks`, реквизиты в
`secrets/android/KEYSTORE.md`. Один раз загрузить в Codemagic → Teams → Code
signing identities → Android keystores под именем `flapp_upload`.
