# Geometry of Reasoning Trajectories in Recurrent-Depth Transformers

Inference-only анализ латентных траекторий рекуррентного блока Huginn-3.5B (Geiping et al., arXiv:2502.05171). Модель не обучается: прогоняем, снимаем скрытое состояние на каждом анролле (хук на `core_block[-1]`), считаем геометрические/динамические/топологические величины. MVP проверяет гипотезы H1–H3 из proposal.

## Статус
Рабочая заготовка (MVP). Реализованы: winding, steps-to-settle, classify_shape (settle/loop/drift), персистентная гомология H1 (ripser), объективные метрики сходимости, корректность через `generate`. Не реализованы: спектральный радиус ρ(∂ₕR)/показатель Ляпунова, Q–K проба, полный accuracy-vs-depth. Большинство прогонов — один сид инициализации и только токен ответа (см. Ограничения).

## Гипотезы
> **H1 (A few shapes).** Each token's latent path falls into one of a few regimes—settling to a point, looping, or drifting—that can be told apart by simple measurements: how quickly steps shrink (the Lyapunov exponent λ), whether the path returns on itself, and its persistent-homology signature.
>
> **H2 (Loops track reasoning depth).** When the path loops, its number of turns—the winding number—grows with the number of reasoning steps the problem requires.
>
> **H3 (Why looping can be necessary).** If the update is forced to always settle (strict contraction, ρ(∂ₕR)<1), the model cannot hold a running count or memory, so tasks needing persistent state must loop or drift.

Протокол: Inference-only on the open Huginn-3.5B model, no fine-tuning. Metrics: Lyapunov exponent and spectral radius ρ(∂ₕR); persistent homology H₁ and winding number; the depth-vs-winding fit; accuracy versus recurrence depth.

## Эксперименты
- E1 — PARARULE-Plus (глубины 2–5): форма и winding траектории токена ответа vs глубина.
- E2 — синтетический счёт ±1: winding/steps vs длина счёта.
- E3 — length-matched контроль (track/local одной длины): single-seed и multi-seed (5 стартов × 10 длин).
- E4 — phase / force-loop: форма при урезанном бюджете шагов (16/24/32/64).
- E5 — обобщение: чётность (switch), бегущий максимум (maxtask).
- Гомология H1 (ripser) на 15 забанканных траекториях.
- Корректность: точность ответа через `generate`.
- Объективные метрики сходимости (по Pappone et al., arXiv:2509.23314) на траекториях.

## Результаты
Все корреляции длины/глубины — Spearman rho по групповым средним (per-level), N = число уровней; per-row (по отдельным траекториям) завышает значимость из-за псевдорепликации.

- PARARULE: все 40 траекторий → settle. Тренда с глубиной нет: winding~depth rho=+0.20, steps~depth rho=+0.80 (per-level, N=4 глубины; оба n.s. — при N=4 порог значимости ≈1.0, магнитуда не интерпретируема, см. flat scatter). Для полноты per-row (N=40): −0.04 / +0.12.
- Счёт: всё settle; корреляция winding с длиной съедается длиной промпта.
- Length-matched, multi-seed (10 длин × 5 инициализаций, per-level): track steps~длина rho=+0.84 (N=10, p<.05), но эффект ~1 шаг и разваливается по инициализациям — per-seed rho=[0.87, 0.79, 0.79, 0.17, 0.45] (диапазон 0.17–0.87); local rho=−0.21 (n.s.), знак пляшет по сидам [−0.41 … +0.28]; winding n.s.
- Обобщение (per-level, N=6): switch steps~длина rho=+0.78, maxtask rho=−0.77 — обе не проходят порог значимости (0.886 при N=6): устойчивого эффекта длины нет ни там, ни там.
- Корректность: accuracy счёта 38% (len 2), 0% (len ≥ 8).
- Гомология: макс. норм-персистентность H1 ≈ 0.003 (ns=64) / 0.000 (ns=16). На 15 траекториях и на одномерных кривых H1≈0 ожидаем by construction.
- Объективная сходимость: размер шага к концу ужимается ×52; conv_rate<0 у 9/9; drift-to-loop ratio ≈12; косинус соседних шагов ≈−0.28.

## Выводы (с привязкой к источникам)
- На токене ответа траектории экспоненциально сходятся к фикс-точке. Это заложено в дизайн Huginn: Geiping et al. (2502.05171) инициализируют латент случайно, что «promotes convergence to a steady state independent of initialization, i.e. path independence». Независимое измерение на той же модели — Blayney et al. (2604.11791, App. C): «the vast majority of tokens reach a fixed-point».
- Синтетический счёт модель не решает (0% при len ≥ 8; проверено одним форматом промпта). Согласуется с теорией невозможности: Merrill et al. (2404.08819) — SSM «cannot solve simple state-tracking problems like permutation composition», и «the ‘state’ in an SSM is an illusion»; Grazzi et al. (2411.12537) — «finite precision LRNNs with state-transition matrices having only positive eigenvalues cannot solve parity».
- Роста winding с глубиной/длиной не обнаружено. Оговорка: смотрели только токен ответа, малые выборки, один сид — это непокрытие H2 (она условная: «when the path loops»), а не её опровержение.
- Формулировка «петель нет» некорректна по мощности: Blayney et al. (2604.11791, App. C) на Huginn-0125 — «only approximately 0.02% of tokens exhibit non-fixed-point behavior» (до 2.81% на токенах вопроса с их system-prompt). При ставке 0.02% 15 траекторий дают ожидаемо ≈0 даже если петли есть. Корректно: «ниже порога обнаружения».

## Ограничения
- Только токен ответа (index −1); у Geiping et al. орбиты наблюдаются на токенах вопроса/цифр, не на токене ответа.
- Малые выборки; нет мощности на явление частотой 0.02–2.8% (Blayney).
- Метрики: winding по 2D-PCA (знак произволен); steps-to-settle не отличает «малый шаг» от «сошёлся»; H1 одной кривой ≈0 by construction.
- Один сид инициализации в большинстве прогонов; множественные сравнения без поправки.
- Не воспроизведён режим Geiping (его system-prompt, глобальный PCA 6D, r=128).
- ρ(∂ₕR)/Ляпунов, персистентная гомология на масштабе, Q–K проба — не сделаны.
- Метрики сходимости взяты у Pappone et al. (2509.23314), но их работа — на собственной GPT-2-масштаба модели, не на Huginn; перенос требует проверки.

## Дальнейшая работа
- Все позиции токенов; system-prompt и режим Geiping et al. (2502.05171); r=128.
- Спектральный радиус ρ(J) через power-iteration + JVP (метод Yang et al., 2605.26733); дистанция до фикс-точки.
- Мощностный скан частоты форм против ставок Blayney et al. (2604.11791).
- Корректная TDA-конструкция (популяционное облако / delay-embedding).
- Q–K проба по Tulchinskii et al. (2502.17017) на PARARULE-Plus.

## Воспроизведение
```
uv sync --extra dev ; uv run pytest ; uv run ruff check .
uv sync --extra dev --extra tda ; uv run python scripts/run_homology.py     # ripser
uv sync --extra dev --extra model                                # torch, transformers==4.53.3
```
`uv sync` приводит окружение в *точное* соответствие запрошенным extras, то есть
доустанавливает нужное и удаляет лишнее — поэтому каждая строка перечисляет все
extras, которые ей нужны, а не только новый. `pytest` и `ruff` живут в `dev`: без
`--extra dev` команда `uv run pytest` поднимает pytest в отдельном окружении без
зависимостей проекта и падает на `ModuleNotFoundError: numpy/pandas` при сборе тестов.
Пины (в `src/traj_geom/constants.py`): `transformers==4.53.3` (рабочее окно 4.50–4.53), `revision="bb6621b65e90b6a4b9b29ef88dc83866d450470c"`. Замечания: `num_steps` — int; хук на `core_block[-1]` с `_forward_hooks.clear()` + try/finally; для траектории `forward`, не `generate`.

## Структура
```
src/traj_geom/   extraction/ metrics/ shapes/ data/ analysis/ + constants.py, types.py
scripts/         воспроизводимые эксперименты (python -m scripts.<name>)
results/         CSV со всеми прогонами
trajectories/    забанканные .npy + manifest.csv
figures/         фигуры экспериментов
notebooks/       00_smoke_extract.ipynb, 01_mvp_h2.ipynb
```

## Ссылки
- Geiping et al. Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach. NeurIPS 2025. arXiv:2502.05171.
- Blayney et al. A Mechanistic Analysis of Looped Reasoning Language Models. arXiv:2604.11791.
- Lu et al. Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer. arXiv:2507.02199.
- Pappone et al. Two-Scale Latent Dynamics for Recurrent-Depth Transformers. NeurIPS 2025. arXiv:2509.23314.
- Merrill, Petty, Sabharwal. The Illusion of State in State-Space Models. arXiv:2404.08819.
- Grazzi et al. Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues. arXiv:2411.12537.
- Sarrof, Veitsman, Hahn. The Expressive Capacity of State Space Models: A Formal Language Perspective. NeurIPS 2024. arXiv:2405.17394.
- Movahedi et al. Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers. arXiv:2606.18206.
- Yang et al. Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models. arXiv:2605.26733.
- Geshkovski et al. A Mathematical Perspective on Transformers. arXiv:2312.10794.
- Tulchinskii et al. Quantifying Logical Consistency in Transformers via Query–Key Alignment. EMNLP 2025. arXiv:2502.17017.
- Bai, Kolter, Koltun. Deep Equilibrium Models. NeurIPS 2019. arXiv:1909.01377.
- Barannikov. The framed Morse complex and its invariants. Advances in Soviet Mathematics 21 (1994) 93–115.
