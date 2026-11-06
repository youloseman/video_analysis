# Stripe: тариф Coach ($25 / мес, $199 / год)

> Продолжение `docs/STRIPE_GO_LIVE_RU.md`. Те же правила: нужен **id цены** `price_…`, не продукта `prod_…`; переменные на Railway менять разом; после деплоя смотреть лог.
> Код уже готов (коммит `4a96bda`): план-ключи `coach_monthly` / `coach_yearly`, переменные `STRIPE_PRICE_COACH_M` / `STRIPE_PRICE_COACH_Y`. Пока обе пустые, тариф в продаже не участвует и лог пишет `BILLING_COACH_OFF` (info, не ошибка).

## 1. Создать продукт

Stripe Dashboard → переключатель **Live mode** включён (не Test) → **Product catalogue** → **+ Create product**.

- **Name:** `Flapp Coach`
- **Description:** `Everything in Full, plus a roster of athletes, their intake links, the squad table and your name and mark on every report. 15 athletes included.` (это видно в Checkout, пишем по-английски)
- **Image:** можно пропустить
- **Pricing model:** Standard pricing
- **Price:** **25.00** · **USD** · **Recurring** · Billing period **Monthly**
- **Save product**

## 2. Добавить годовую цену

Открыть созданный `Flapp Coach` → блок **Pricing** → **+ Add another price**:

- **25.00** → заменить на **199.00** · USD · Recurring · **Yearly**
- **Save**

Теперь у продукта две строки цен.

## 3. Скопировать id цен

В таблице Pricing у каждой строки `⋯` → **Copy price ID**. Получится два `price_1…`:

| Цена | Переменная |
|---|---|
| $25.00 / month | `STRIPE_PRICE_COACH_M` |
| $199.00 / year | `STRIPE_PRICE_COACH_Y` |

Проверка: id начинается с `price_`, а не `prod_`. В Live mode (в Test id другие и в проде не сработают).

## 4. Railway: переменные

Railway → проект → сервис **`video-analysis`** → **Variables** → **+ New Variable** дважды:

```
STRIPE_PRICE_COACH_M = price_…
STRIPE_PRICE_COACH_Y = price_…
```

Добавлять обе за один раз и сохранять одним действием: сервис перезапустится один раз. Если задать только одну, лог деплоя скажет `BILLING_PRICES_MISSING plans=['coach_yearly']` (или monthly), и это ошибка по замыслу: одна цена из пары это копипаст-ошибка.

## 5. Проверить после перезапуска

В логе Railway при старте:

| Строка | Значит |
|---|---|
| `BILLING_READY plans=[…, 'coach_monthly', 'coach_yearly', …]` | обе цены подхвачены |
| `BILLING_COACH_OFF` | обе переменные пустые, тариф спит (так было до шага 4) |
| `BILLING_PRICES_MISSING plans=['coach_…']` | задана одна из двух |
| `BILLING_PRICE_REUSED` | один и тот же `price_` вставлен в две переменные |

И `GET https://getflapp.com/health` → `"plans_configured": 10` (было 8).

## 6. Что ещё НЕ произойдёт после шага 4

Карточки Coach на странице цен пока нет: это фаза 4 плана (`docs/COACH_AREA_PLAN_RU.md`), отдельная полоса «For coaches and fitters», а не четвёртая колонка. То есть цены можно создать заранее, покупать их никто не сможет, пока не выложим карточку. Webhook при этом уже знает тариф: если кто-то купит по прямой ссылке Stripe Payment Link на эту цену, аккаунт получит `tier=coach`.

## 7. Опционально: Payment Link для пилота

Если захочется взять деньги с первого тренера до появления карточки: в Stripe открыть цену → **Create payment link** → Checkout с email. Webhook сопоставит оплату с аккаунтом по email покупателя так же, как обычный Checkout (`customer_email` должен совпадать с email аккаунта Flapp). Это запасной путь, не основной.

## 8. Expert Review внутри Coach

Годовой Coach, как и годовой Full, включает одну Expert Review в год (`includes_review`), кредит выдаётся при покупке и при продлении. Месячный Coach ревью не включает, но даёт member-цену $29. Отдельных цен в Stripe для этого не нужно.
