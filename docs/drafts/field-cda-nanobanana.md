# Field CdA: промпты для Nano Banana

Статья `/academy/field-cda`. Графики (scale, loop, protocol, position, cancels) уже собраны кодом в стиле дома.
Не хватает рисованных иллюстраций, как в статьях про hip over crank и шлемы: силуэт райдера navy, велосипед тонкой серой линией, светлый тёплый фон.

**Как генерировать:**
- К каждому промпту прикладывать два референса стиля:
  `backend/app/static/media/academy/kona-2026-hip-over-crank/illustration.webp` и
  `backend/app/static/media/academy/triathlon-helmets-2026/head-nb.webp`.
- Текст на картинке модель рисует плохо, поэтому везде «no text». Подписи (IBM Plex) добавляю я поверх, как на референсах.
- Формат 16:9, не меньше 2400×1350.

---

## 1. Обложка: «ветер, который не чувствуешь»

Куда: обложка статьи и карточка в хабе (заменит карточку с цифрами; карточка с цифрами уйдёт в пост для FB).

```
Flat editorial illustration, side view, 16:9. A triathlete in a low time-trial tuck on a time-trial bike, moving to the right. The rider is a solid deep navy (#14294B) silhouette with a few thin lighter-navy seam lines, aero helmet, no facial detail. The bike is drawn in thin light-grey outline (#C9C5BC) with spoked wheels, no colour fill. Plain warm off-white background (#FAF9F6). In front of the rider, thin pale-blue airflow streamlines (#9DB5E6) come from the right edge, flow toward the rider and wrap over the helmet and back, with two or three small curls behind the hips. On the roadside ahead of the rider, a small windsock on a thin pole, drawn in the same thin grey line, only slightly lifted: a breeze too light to feel. The road is one thin grey horizontal line with a faint soft ground shadow under the wheels. Rider and bike take the right 60% of the frame; keep the left 40% and the top third completely empty for a headline. No text, no numbers, no logos, no brand marks, no gradients, no extra objects. Match the attached reference images exactly: the same navy, the same line weight, the same calm, technical look.
```

Подписи поверх: заголовок «A breeze you can't feel outweighs any helmet» слева, у ветроуказателя `1 m/s`, у спины `+23% CdA, one ride` (coral).

---

## 2. Схема маршрута туда-обратно

Куда: раздел «How to ride it», над графиком A-B-A-B. Сейчас правило «катай туда-обратно» есть только в тексте и в столбиках.

```
Flat minimal vector map seen from directly above, 16:9. A straight road runs left to right across the frame and ends at the right in a small round turnaround loop. The road is a light-grey band (#DEDBD4) with a thin white dashed centre line, on a plain warm off-white background (#FAF9F6). On the upper lane, a small coral (#F1553F) arrow-shaped rider marker heads right, away from the start. On the lower lane, a small green (#2E8B57) arrow-shaped rider marker heads left, back toward the start. Three large, soft, pale-blue (#9DB5E6) wind arrows blow from right to left across the whole scene, above and below the road. A small start/finish tick mark at the left end of the road. Lots of empty space around the road for labels. No text, no numbers, no trees, no buildings, no terrain, no shadows, no logos.
```

Подписи поверх: у красной стрелки `out: into the wind, +23%`, у зелёной `back: with it`, у старта `together: -0.2%`.

---

## 3. Что Flapp читает с фото

Куда: раздел «What a photo sees that a ride can't», над графиком position. Это связка с Flapp: на картинке те же две метрики, что в результате анализа (наклон корпуса и высота шлема над спиной).

```
Flat editorial illustration, side view, 16:9, the same style as the attached references. A triathlete in a time-trial tuck on a time-trial bike, facing right: solid deep navy (#14294B) silhouette, bike in thin light-grey outline (#C9C5BC), plain warm off-white background (#FAF9F6). Behind the rider, a second ghost silhouette of the same rider in warm grey (#C9C4BA), sitting slightly more upright with the back tilted a few degrees higher and the head and helmet clearly higher: the "before" position. Over the navy rider, clean measurement graphics in royal blue (#2457C5), thin and precise: a straight line along the back from the shoulder to the hip; a short horizontal reference line from the hip pointing back; a small arc between the two lines marking the torso angle; and a thin vertical bracket from the line of the back up to the top of the helmet marking helmet height. Small round blue dots at the shoulder and hip. Leave clean empty space next to the arc and next to the bracket for labels. No text, no numbers, no logos, no faces detail.
```

Подписи поверх: у дуги `torso 24° → 21°`, у скобки `helmet +30% → +18%`, внизу `≈ 6 W at 35 km/h, Flapp estimate`.

---

## Что не генерировать

- Скриншот журнала CdA и графики: они из кода и данных, ИИ-картинка их только испортит.
- Реальных райдеров и бренды велосипедов/шлемов: только обезличенные силуэты.
