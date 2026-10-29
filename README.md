# Condition-gated spiking neural P forecasters

Code for the paper *Restoring the firing condition: condition-gated spiking neural P models for sparse, closed-form univariate time-series forecasting*.

Everything is in one notebook, `trace_snp.ipynb`. Run top to bottom, it takes the 11 raw series, trains and fits every model, runs every statistical test and control experiment, and writes every figure and table of the paper and its supplementary material. It needs no account, token or remote storage. All results are written to a local folder.

## Contents

| Path | What it is |
|---|---|
| `trace_snp.ipynb` | the complete experiment (33 code cells, numbered 0 to 32) |
| `requirements.txt` | Python packages |
| `datasets.zip` | **all 11 datasets used in the paper**, exactly as used (unzip into `data/`) |
| `data/` | where the notebook looks for the series first (see [Data](#data)) |
| `LICENSE` | MIT License |

## Models

**Proposed family.** These are spiking neural P (SNP) forecasters with an explicit firing condition:

- **ALSTM-SNP:** the learned member. It is a recurrent SNP cell with spiking, adaptive-threshold neurons, trained by surrogate gradients.
- **TRACE-SNP v1:** the closed-form member. It is a tapped delay line feeding triangular receptive fields (exactly two of M fire per delay), plus k-nearest state neurons on the recent embedding. The readout is linear in its weights, so it is fitted by one ridge solve. Selection uses exact leave-one-out residuals plus a validation block. A binary-spike version is also fitted.
- **TRACE-SNP v2:** v1 with two additions. The reference point is either the newest value or the window mean. Multi-step forecasts are either direct or iterated, chosen on validation with a margin rule.
- **TRACE-SNP-AF:** v2 with an adaptive firing condition. It chooses among triangular and Gaussian receptive fields with 2, 4 or 8 active fields, per fold and before any test data is seen.

**Baselines.**

- **Networks, all at hidden size 8:** LSTM, GRU, LSTM-SNP, BiLSTM-SNP, an attention model, Informer, Autoformer and FEDformer.
- **Closed-form and simple models:** NG-RC, DLinear, ridge AR(L), persistence and the window mean.
- **Control experiments:** an echo-type SNP reservoir, plus tuned and directly trained network variants.
- **Classical statistical forecasters (extended comparison):** ARIMA, simple and damped Holt exponential smoothing, the seasonal naive forecast and the training median, on the same walk-forward folds, targets and origins. They enter a second ranking with their own tables and figures; the 16-model comparison of the paper is unchanged.

## Requirements

- Python 3.10 or newer, with the packages in `requirements.txt`. `nvidia-ml-py` is optional: it gives the GPU energy measurements, and without it the notebook reports time only.
- **Full run:** one CUDA GPU. The settings in cell 0 assume about 70 GB of VRAM and 30 GB of system RAM, which is an RTX 6000-class card. On a smaller GPU, set `RUN_BATCH` in cell 0 (for example `RUN_BATCH = 60`) so that fewer models train at once. Results are the same up to floating-point rounding; only the speed changes.
- **Smoke test:** a CPU is enough.
- **Classical forecasters (cell 28):** `statsmodels` (installed by the notebook if it is missing) and a CPU; no GPU is needed.

```
pip install -r requirements.txt
```

## Data

**All 11 datasets used in the paper are included in this repository, in `datasets.zip`.** These are the exact copies used for every reported result. Unzip it into `data/` before running:

```
unzip datasets.zip -d data
```

Cell 3 then reads each series from `data/<name>.csv`. Only if a file is missing does it fall back to the public source or generator listed below, so the table also documents where every series comes from.

The series actually used are then frozen in `results/data_main/` and `results/extra_data/`, and every later run reuses those copies.

| Name in the notebook | Series | Points | How cell 3 obtains it |
|---|---|---|---|
| `EPL` | English Premier League match data 2000–2025, a per-match series of 3,800 consecutive matches | 3,800 | **Local file only:** `data/EPL.csv`. Source: [Kaggle, English Premier League match data 2000–2025](https://www.kaggle.com/datasets/marcohuiii/english-premier-league-epl-match-data-2000-2025) (a Kaggle sign-in is needed to download). |
| `Sunspots` | monthly mean total sunspot number, January 1749 to January 2021 | 3,265 | Downloaded from [WDC-SILSO, Royal Observatory of Belgium](https://www.sidc.be/SILSO/datafiles). |
| `etth1`, `etth2` | electricity transformer oil temperature (`OT`), hourly, last 4,000 points | 4,000 each | Downloaded from the [ETDataset](https://github.com/zhouhaoyi/ETDataset) repository. |
| `melbourne_temp` | Melbourne daily minimum temperature, 1981–1990 (Australian Bureau of Meteorology) | 3,650 | Downloaded from a [public mirror](https://github.com/jbrownlee/Datasets). |
| `co2_weekly` | Mauna Loa CO2, weekly | 2,284 | `statsmodels.datasets.co2` |
| `elnino_monthly` | El Niño sea-surface temperature, monthly | 732 | `statsmodels.datasets.elnino` |
| `MackeyGlass` | Mackey–Glass delay equation, $`\tau = 17`$ | 1,082 | Generated (see below). |
| `lorenz` | Lorenz-63 $`x(t)`$, $`\sigma = 10`$, $`\rho = 28`$, $`\beta = 8/3`$ | 3,000 | Generated by RK4 with step 0.01; every 3rd step is kept after 500 samples of burn-in. |
| `lorenz_noisy` | the same series plus Gaussian noise of 1 % of its standard deviation (seed 1) | 3,000 | Generated. |
| `rossler` | Rössler $`x(t)`$, $`a = b = 0.2`$, $`c = 5.7`$ | 3,000 | Generated by RK4 with step 0.05; every 2nd step is kept after 1,000 samples of burn-in. |

**Notes on individual series:**

**Mackey–Glass.** The series is generated by the unit-step recursion

```math
x(t+1) = x(t) + \frac{0.2\,x(t-17)}{1 + x(t-17)^{10}} - 0.1\,x(t),
```

with $`x(0) = 1.2`$, $`x(1) = 1.0`$ and $`x(t) = 0`$ for $`t < 0`$. Cell 3 checks the series against its reference persistence and naive RMSE (0.0344 and 0.2305) and prints whether they match.

**Sunspots.** SILSO revises its series from time to time, so a fresh download may differ slightly. The copy in `datasets.zip` is the one used in the paper.

**EPL.** This series has no download without a sign-in. Use the copy in `datasets.zip`.

## Mathematical summary

This section states the models and every mathematical result of the paper. The full proofs are in the supplementary material. Notation:

- $`\sigma`$ is the logistic function, $`\mathrm{softplus}(x) = \log(1 + e^{x})`$, and $`\Theta`$ is the Heaviside step.
- $`\odot`$ is the entry-wise product.
- $`L`$ is the window length (number of delays), $`M`$ the number of receptive fields per delay, and $`C`$ the number of state neurons.
- Each series ends in $`J = 5`$ test blocks. Test block $`j`$ is preceded by a validation block $`\mathcal{V}_j`$ of the same length, and every earlier point is the training set $`\mathcal{D}_j`$.

### Standing assumptions

| | Assumption |
|---|---|
| A1 (ALSTM-SNP) | $`\alpha, \rho \in (0,1)`$ through $`\sigma`$; $`\beta = \mathrm{softplus}(\cdot) \ge 0`$; $`b_0 = \mathrm{softplus}(\cdot) + 10^{-3} > 0`$; $`u_0 = \eta_0 = h_0 = 0`$. |
| A2 (grid) | The centres are $`c_m = c_1 + (m-1)\Delta`$, with $`c_1 = -1.1`$, $`c_M = 1.1`$ and $`\Delta = 2.2/(M-1)`$. The scaling maps the training range onto $`[-1, 1]`$. |
| A3 (ridge) | Every TRACE-SNP candidate has the penalty $`P = \lambda_0 I + P'`$, with $`\lambda_0 > 0`$ and $`P' \succeq 0`$. |
| A4 (information set) | Every quantity used for test block $`j`$ (scaling, prototypes, weights, hyperparameters, field family, multi-step mode) is computed from $`\mathcal{D}_j \cup \mathcal{V}_j`$ only. |
| A5 (independent datasets) | The 11 series are independent units for the dataset-level tests. |

### The condition-gated SNP neuron

**Definition.** A neuron has:

- a state $`u_t`$;
- a charge $`q_t`$, computed from its inputs and its previous state;
- a threshold $`B_t`$.

It fires when $`z_t = \mathbf{1}[q_t \ge B_t]`$, and updates

```math
u_t = r_t\,u_{t-1} + i_t - z_t\,\kappa_t, \qquad h_t = z_t\,e_t,
```

with reset factor $`r_t`$, input current $`i_t`$, consumption $`\kappa_t \ge 0`$ and emission $`e_t`$. Its **activity** is the share of neurons that fire.

| Result | Statement |
|---|---|
| LSTM-SNP has no firing condition | LSTM-SNP, $`u_t = r_t \odot u_{t-1} - c_t \odot a_t`$ with $`h_t = o_t \odot a_t`$, is the neuron above with $`i_t = 0`$, $`\kappa_t = c_t a_t`$, $`e_t = o_t a_t`$ and $`B_t = -\infty`$. Every neuron fires at every step, so its activity is identically 1. |
| A finite threshold changes the model | A gated neuron with a finite $`B`$ reproduces LSTM-SNP on every input **if and only if** $`B \le \inf_t q_t`$ along every trajectory, that is, if and only if its condition never fails. |
| LSTM-SNP costs as much as a dense network | Every synapse is used at every step, so no event-driven saving is possible. |

### ALSTM-SNP (the learned member)

The gate pre-activations use no self-recurrence (the recurrent matrices $`U`$ have zero diagonals). The cell is:

```math
\begin{aligned}
\tilde n_t &= \tanh\!\left(W_n x_t + U_n h_{t-1}\right), &
o_t &= \sigma\!\left(W_o x_t + U_o h_{t-1} + v_o \odot u_{t-1}\right),\\
\tilde u_t &= \alpha \odot r_t \odot u_{t-1} + n_t, &
B_t &= b_0 + \beta \odot \eta_{t-1},\\
v_t &= (\tilde u_t - B_t) \oslash B_t, &
z_t &= \Theta(v_t),\\
u_t &= \tilde u_t - z_t \odot c_t \odot B_t, &
\eta_t &= \rho \odot \eta_{t-1} + (1-\rho) \odot z_t,\\
h_t &= o_t \odot z_t \odot \mathrm{softplus}(\tilde u_t).
\end{aligned}
```

Training uses the surrogate derivative $`\psi(v) = \gamma \max\left(0,\, 1 - \lvert v \rvert\right)`$.

| Result | Statement (under A1) |
|---|---|
| Firing trace | $`0 \le \eta_t \le 1 - \rho^{t} < 1`$. |
| The firing condition is well defined | $`b_0 \le B_t \le b_0 + \beta\left(1 - \rho^{t-1}\right) < b_0 + \beta`$. Hence $`v_t`$ is finite, and $`z_t = 1`$ if and only if $`\tilde u_t \ge B_t`$. |
| $`b_0 > 0`$ is necessary | If $`b_0 = 0`$, then $`v_1`$ is undefined. As $`b_0 \to 0`$, a neuron that has not fired receives no learning signal on almost every input. |
| Unconditional boundedness | For every input sequence, bounded or not: $`\lvert u_t \rvert \le \lvert \tilde u_t \rvert < \dfrac{1 - \bar\alpha^{t}}{1 - \bar\alpha} < \dfrac{1}{1 - \bar\alpha}`$, where $`\bar\alpha = \max_j \alpha_j`$. |
| Bounded outputs and forecasts | $`0 \le h_t < \mathrm{softplus}\!\left(\frac{1}{1-\bar\alpha}\right)`$, and $`h_t = 0`$ whenever $`z_t = 0`$. Every iterated $`h`$-step forecast stays in a fixed bounded interval. |
| Bounded surrogate gradients | $`0 \le \dfrac{\partial z_t}{\partial \tilde u_t} \le \dfrac{\gamma}{b_0}`$ and $`\left\lvert \dfrac{\partial z_t}{\partial B_t} \right\rvert \le \dfrac{2\gamma}{b_0}`$. |
| Silent neurons cost nothing | If $`z_{t,j} = 0`$, then $`h_{t,j} = 0`$ exactly. The recurrent synaptic operations at step $`t+1`$ are $`(H-1)\times`$ the number of neurons that fired at step $`t`$, per gate. |

### TRACE-SNP (the closed-form members)

**Forecast.** The forecast of $`x_{o+h-1}`$ is computed from the scaled delays $`u_k = s(x_{o-k})`$, $`k = 1, \dots, L`$, as

```math
\hat x = \text{anchor} + s^{-1}\!\left[\, b + \sum_{k} a_k u_k + \sum_{k}\sum_{m} w_{k,m}\,\mu_m(u_k) + \sum_{c} \nu_c(z)\left(\alpha_c + \gamma_c^{\top} z\right) \right],
\qquad z = (u_1, \dots, u_d).
```

- **Anchor:** v1 always uses the newest value $`x_{o-1}`$. v2 and AF choose between $`x_{o-1}`$ and the window mean.
- **Pathways:** there are three kinds of neurons.
  - *Linear neurons* $`a_k u_k`$ are always active.
  - *Receptive-field neurons* use triangular fields $`\mu_m(u) = \max\left(0,\, 1 - \lvert u - c_m \rvert / \Delta\right)`$ (Wang–Mendel).
  - *State neurons* $`\nu_c`$ fire for the $`k_a = 4`$ prototypes nearest to $`z`$ (k-means++ with a fixed seed). They use a normalised Gaussian kernel and Takagi–Sugeno consequents.

**Fit.** Every candidate is fitted by

```math
\hat\theta = \arg\min_{\theta} J(\theta), \qquad
J(\theta) = \frac{1}{N}\left\lVert r - \Phi\theta \right\rVert^2 + \theta^{\top} P\,\theta,
```

```math
P = \lambda_0 I + \lambda_s \sum_{k} E_k^{\top} D_2^{\top} D_2 E_k + \lambda_{st}\, E_{st}^{\top} E_{st},
```

with $`A = \frac{1}{N}\Phi^{\top}\Phi + P`$ and $`g = \frac{1}{N}\Phi^{\top} r`$. Here $`D_2`$ is the second-difference (curvature) operator on the field weights of one delay.

**Selection.** For block $`j`$, the chosen candidate minimises

```math
\mathrm{Crit}_j = \left( \frac{\displaystyle\sum_{i \in \mathcal{D}_j} \left(\frac{e_i}{1 - H_{ii}}\right)^{2} + \sum_{v \in \mathcal{V}_j} e_v^{2}}{\lvert \mathcal{D}_j \rvert + \lvert \mathcal{V}_j \rvert} \right)^{1/2}.
```

Persistence, the window mean and ridge AR($`L`$) are always among the candidates.

| Result | Statement |
|---|---|
| Partition of unity (A2) | $`\sum_m \mu_m(u) = 1`$ on $`[c_1, c_M]`$. Exactly two fields fire (one at a centre), and finding them costs $`O(1)`$ operations. |
| Piecewise-linear representation | $`w \mapsto \sum_m w_m \mu_m`$ is a linear bijection onto the continuous functions that are affine on each grid cell, with $`g(c_m) = w_m`$. Each delay contributes a one-dimensional function $`f_k(x_{t-k})`$. |
| Approximation rate | For an $`L_f`$-Lipschitz $`f`$, $`\lVert f - \mathcal{I}f \rVert_\infty \le L_f \Delta / 2`$. For $`f \in C^2`$, $`\lVert f - \mathcal{I}f \rVert_\infty \le \Delta^2 \lVert f'' \rVert_\infty / 8`$. |
| Additive targets | For $`F(u) = g_0 + \sum_k g_k(u_k)`$, the delay pathway reaches $`\sup \lvert F - \hat F \rvert \le \frac{\Delta}{2} \sum_k L_k`$. $`\Delta = 0.147`$ for $`M = 16`$ and $`\Delta = 0.071`$ for $`M = 32`$. |
| Curvature penalty | $`(D_2 w)_m = \Delta^2 f''(\xi_m)`$ when $`w`$ samples a $`C^2`$ function. The null space of $`D_2`$ is the affine functions, so the penalty never shrinks a linear trend. |
| State neurons | Exactly $`\min(k_a, C)`$ state neurons fire, with $`\nu \ge 0`$ and $`\sum \nu = 1`$. The state pathway is a convex combination of $`k_a`$ local affine models: a first-order Takagi–Sugeno system that is linear in its weights. |
| Exact activity | With inputs inside the grid, $`2L + k_a`$ of the $`ML + C`$ event neurons fire, for example $`44/704 = 6.25\,\%`$ at $`L = 20`$, $`M = 32`$, $`C = 64`$. Without the state pathway the share is $`2/M`$. |
| The ridge is necessary | $`\mathrm{rank} \Phi \le p - 2L`$: each delay has two null directions (the partition of unity and linear reproduction), and both lie in the null space of the curvature penalty. With $`\lambda_0 = 0`$, $`J`$ therefore has infinitely many minimisers. |
| Existence, uniqueness, stability (A3) | $`A \succeq \lambda_0 I \succ 0`$, and $`J`$ is $`2\lambda_0`$-strongly convex. The unique solution is $`\hat\theta = A^{-1} g`$, with a unique Cholesky factor, $`\lVert \hat\theta \rVert_2 \le \lVert g \rVert_2 / \lambda_0`$, and smooth dependence on the data. |
| Leverages | $`0 \le H_{ii} \le \dfrac{\lVert \varphi_i \rVert^2 / N}{\lambda_0 + \lVert \varphi_i \rVert^2 / N} < 1`$. |
| Exact leave-one-out | The residual of the model refitted without row $`i`$ equals $`e_i / (1 - H_{ii})`$. All $`N`$ of them follow from one factorisation. The identity is exact for $`\theta`$ given $`\Phi`$; the scaling and the k-means prototypes are not refitted per row. |
| Never worse than the fallbacks | $`\mathrm{Crit}_j(\text{selected}) \le \min_{m \in \{\text{persistence},\ \text{window mean},\ \text{ridge AR}(L)\}} \mathrm{Crit}_j(m)`$. |
| What leave-one-out estimates | For exchangeable pairs, $`\mathbb{E}\!\left[\left(\frac{e_i}{1 - H_{ii}}\right)^{2}\right]`$ is the expected error on a new pair of the model fitted on $`N-1`$ pairs. Time series are not exchangeable, which is why the criterion adds the validation block. |
| No test leakage (A4) | Every test forecast is a fixed function of $`\mathcal{D}_j \cup \mathcal{V}_j`$ and of the observed window. |
| Determinism | The fit is a deterministic function of the training data. There is no seed variance, and one evaluation is the final forecast. |
| Cost of the fit | With at most $`q = 1 + 3L + k_a(d+1)`$ non-zeros per row, a fit costs $`O(Nq^2 + p^3 + Np^2)`$, with no iterative optimisation. |
| The 25 % margin (v2) | v2 iterates if and only if $`R_{\mathrm{it}} \le 0.75\, R_{\mathrm{dir}}`$. When both modes are equally good and $`\log(R_{\mathrm{it}}/R_{\mathrm{dir}}) \sim \mathcal{N}(0, s^2)`$, it iterates with probability $`\Phi(\log 0.75 / s)`$: 0.2 % for $`s = 0.1`$ and 7.5 % for $`s = 0.2`$. An iterated forecast costs $`h`$ one-step evaluations. |

**Parameter counts.** The model has $`p = 1 + L + LM + C(d+1)`$ fitted parameters. Per forecast, only these are active:

- $`p_{\mathrm{act}} = 1 + 3L + k_a(d+1)`$ for the triangular condition;
- $`p_{\mathrm{act}} = 1 + (k+1)L + k_a(d+1)`$ for a $`k`$-field Gaussian condition.

### What a firing condition removes (Gaussian fields, TRACE-SNP-AF)

The Gaussian fields have width $`w\Delta`$, and the $`k`$ fields nearest to $`u`$ fire.

| Result | Statement |
|---|---|
| $`n`$-th nearest centre | $`\delta_{(n)} \ge (n-1)\Delta/2`$. |
| Upper bound on the dropped activation | $`\varepsilon_k(u; w) \le \sum_{n>k} \exp\!\left(-\frac{(n-1)^2}{8w^2}\right)`$. For $`w = 1`$ the bound is 1.124 at $`k = 2`$, 0.193 at $`k = 4`$ and $`3.8\times 10^{-4}`$ at $`k = 8`$. |
| Total dense activation | $`\left\lvert S_w(u) - \sqrt{2\pi}\,w \right\rvert \le \sqrt{2\pi}\,w \cdot \dfrac{2e^{-2\pi^2 w^2}}{1 - e^{-2\pi^2 w^2}}`$. The relative error is below $`6\times 10^{-9}`$ for $`w \ge 1`$. |
| Lower bound | At the midpoint between two centres, two fields remove exactly $`2\sum_j \exp\!\left(-\frac{(2j+1)^2}{8w^2}\right)`$. That is $`\ge 0.742`$ (29.6 % of the dense activation) for $`w = 1`$, and $`\ge 61\,\%`$ for $`w = 2`$. |
| Gated against dense forecasts | For the same weights, $`\lvert F_{\mathrm{dense}} - F_k \rvert \le \bar\varepsilon_k(w) \sum_k \lVert w_k \rVert_\infty`$. The same bound holds for the difference in RMSE. |
| Eight fields reproduce the dense basis | For $`w = 1`$ the gap is at most 0.04 % of $`\sum_k \lVert w_k \rVert_\infty`$ at $`k = 8`$, against a factor of 1.12 at $`k = 2`$. This is consistent with the measured RMSE ratio of 0.996 for $`k = 8`$. |
| Guarantees of the adaptive condition | On every fold, TRACE-SNP-AF has the following properties: (1) its criterion is within a factor $`1+\tau`$ of the best candidate; (2) it is within the same factor of the TRACE-SNP v2 choice; (3) no candidate within that tolerance has fewer firing neurons; (4) at most $`8L + k_a`$ event neurons fire; (5) the choice uses the training and validation data only. |
| Activity of TRACE-SNP-AF | $`\dfrac{kL + k_a}{ML + C}`$. This is never above the dense family, and it equals v2's activity when the triangular family is chosen. |

### Operations and energy

| Result | Statement |
|---|---|
| Exact counts (TRACE family) | $`\mathrm{MAC} = 2L + L + kL + kL\,\mathbf{1}[\text{Gauss}] + \left(Cd + k_a + k_a(d+1)\right)\mathbf{1}[\text{state}]`$. $`\mathrm{AC} = L\,\mathbf{1}[\text{mean anchor}] + L\,\mathbf{1}[\text{difference encoding}]`$. |
| Recurrent networks | A cell with $`G`$ gates and $`H`$ units needs $`\approx G\,H(H+1)\,L`$ MACs per window. With $`H = 8`$ that is $`288L`$ for the LSTM and LSTM-SNP and $`216L`$ for the GRU. |
| Members and ensembles | The cost of a forecast averaged over $`n`$ members is $`n`$ times the cost of one member. The networks use $`n = 60`$ and the closed-form models $`n = 1`$. |
| Benchmark overhead | If a call costs $`E(n) = E_0 + n\,e_1`$, the energy per forecast is $`e_1 + E_0/n`$. Implementations measured with different batch sizes therefore differ even when their arithmetic is identical. Operation counts are the architectural comparison. |

### Uncertainty of the learned delay functions

| Result | Statement |
|---|---|
| Identification | The centred delay function $`\tilde f_k`$ is invariant to every null direction of $`\Phi`$, so it is determined by the forecast function. |
| Sandwich covariance | $`\mathrm{Cov}(\hat\theta) = \frac{\sigma^2}{N} A^{-1}\left(\frac{1}{N}\Phi^{\top}\Phi\right) A^{-1}`$, and $`\mathrm{Var}\!\left(\tilde f_k(u)\right) = \tilde\ell_k(u)^{\top} \mathrm{Cov}(\hat\theta)\, \tilde\ell_k(u)`$. |
| Combining refits | $`\mathrm{Var} = \mathbb{E}_j\!\left[\mathrm{se}_j^2\right] + \mathrm{Var}_j\!\left[\tilde f_k^{(j)}\right]`$. The plotted pointwise 95 % band is $`\bar f_k \pm 1.96\left(\overline{\mathrm{se}^2} + \text{between-refit variance}\right)^{1/2}`$. |

### Statistics over the 11 datasets

**Score.** $`S_{m,d}`$ is the geometric mean, over the 18 (lag, horizon) problems of dataset $`d`$, of model $`m`$'s RMSE relative to a reference.

| Result | Statement |
|---|---|
| Properties of the score | The score is invariant to shifting and rescaling a series. Ratios of scores do not depend on the reference model. Ranks use one observation per dataset. |
| Friedman and Iman–Davenport | $`F_F = \dfrac{(N-1)\chi^2_F}{N(k-1) - \chi^2_F}`$. The Nemenyi critical difference is $`\mathrm{CD} = q_\alpha \sqrt{\dfrac{k(k+1)}{6N}}`$. For $`k = 16`$, $`N = 11`$ and $`\chi^2_F = 86.28`$: $`F_F = 10.96`$ on $`(15, 150)`$ degrees of freedom, and $`\mathrm{CD} = 6.96`$. |
| Holm | Holm's step-down procedure controls the family-wise error at $`\alpha`$ for valid $`p`$-values, under any dependence. |
| Power ceiling with $`N = 11`$ | The exact two-sided Wilcoxon $`p`$-value is at least 0.00098 with 0 losses, 0.0020 with 1 and 0.0049 with 2. This bounds what any Holm-corrected comparison over 15 rivals can reach. |
| Bayesian signed-rank | Each posterior probability is an average of 50,000 draws, so its Monte Carlo standard error is $`\le 0.0022`$. |

## How to run

**Full experiment:**

1. Open `trace_snp.ipynb` on a GPU machine.
2. Unzip `datasets.zip` into `data/`.
3. Run all cells.

Cell 7 (network training) and cells 21–23 (the control networks) take most of the time. Cells 26–31 (the extended comparison) take about 6 minutes on a CPU; only cell 28 fits models, and it reuses `results/revisions/extended_checks/statistical_baselines*.csv` when they exist (set `EXT_REFIT_CLASSICAL = True` in cell 28 to refit). Every long cell saves its progress continuously and stops cleanly when `TIME_BUDGET_H` (cell 0, 11 h by default) is reached. To continue, run all cells again: finished lags, folds and seeds are read back from `results/` and are not retrained.

**Smoke test:**

```
SMOKE=1 jupyter nbconvert --to notebook --execute trace_snp.ipynb --output smoke_run.ipynb
```

This uses 3 seeds, 3 epochs, lags 1 and 5, and the last 400 points of each series. All outputs carry the suffix `_smoke`, so they never mix with real results.

**Useful settings in cell 0:**

| Setting | Effect |
|---|---|
| `DATASETS_TO_RUN = ['MackeyGlass', 'lorenz']` | run a subset of the datasets |
| `MODELS_TO_RUN = ['ALSTM-SNP', 'gru_39']` | train only some of the networks |
| `RUN_TRAINING = False` | train nothing and only rebuild the analysis from `results/` |
| `FRESH_START = True` | delete and retrain the runs that this session starts |
| `RESULTS_DIR`, `DATA_DIR` | environment variables that move the two folders |

## Cells

| Cell | Purpose | Rough run time on one GPU |
|---|---|---|
| 0 | configuration | – |
| 1 | imports, device, memory guards, time budget, GPU energy meter | seconds |
| 2 | the local results store | – |
| 3 | the 11 datasets and their walk-forward folds (5 test blocks, each preceded by a validation block of the same length) | seconds to minutes (downloads) |
| 4 | ALSTM-SNP and the 8 baseline networks | – |
| 5 | helpers: windows, vmapped ensembles, Adam, metrics, Diebold–Mariano, DeLong, Holm | – |
| 6 | the walk-forward training harness | – |
| 7 | trains the 9 networks: 11 datasets × 6 lags × 5 folds × 60 seeds | the longest cell: several 11 h sessions |
| 8 | TRACE-SNP v1 and binary spikes, 6 ablations, NG-RC, DLinear, ridge, persistence; horizons 1, 6 and 24; input-noise robustness; GPU energy | minutes to about an hour |
| 9 | the networks at horizons 6 and 24, iterating their saved checkpoints | minutes to about an hour |
| 10 | TRACE-SNP v2 and binary spikes | minutes to about an hour |
| 11 | every model on identical targets: leaderboards, DM / DeLong with Holm, win/tie/loss, horizon profile, noise, energy, operation counts | minutes |
| 12 | inventory of the results and the parameters actually selected | seconds |
| 13 | firing-condition ablation: gated fields against a dense, non-gated RBF basis with the same parameters | minutes to about an hour |
| 14 | ranking over N = 11 datasets: Friedman, Iman–Davenport, Nemenyi CD, Wilcoxon–Holm | minutes |
| 15 | inference energy per ensemble member and per ensemble forecast | minutes |
| 16 | the analysis figures | minutes |
| 17 | receptive-field shape × gating, and accuracy against the number of active neurons | minutes to about an hour |
| 18 | TRACE-SNP-AF | minutes to about an hour |
| 19 | variance over seeds and test blocks; uncertainty bands of the learned delay functions | minutes |
| 20 | settings and helpers of the control experiments | – |
| 21 | every network tuned on the validation block (width × learning rate) | 15–25 GPU hours for cells 21–23 together |
| 22 | networks trained directly for h = 6 and h = 24 | (in the above) |
| 23 | ALSTM-SNP diagnostics: matched protocol, no firing condition, no threshold adaptation, surrogate slope 1, measured firing activity | (in the above) |
| 24 | echo-type SNP reservoir baseline (closed form) | minutes to about an hour |
| 25 | inference energy of TRACE-SNP v1, v2 and TRACE-SNP-AF with one common code path | minutes |
| 26 | extended comparison: settings, data manifest (SHA-256 of every series, the EPL source column), check that every stored forecast matches its series and that the persistence RMSE of the leaderboard is reproduced | seconds |
| 27 | constant forecasts at the training mean and median, against persistence | seconds |
| 28 | classical statistical forecasters: ARIMA (order by AIC, coefficients by maximum likelihood before each test block), simple and damped Holt exponential smoothing, seasonal naive, training median | about 5 min (CPU) |
| 29 | largest Lyapunov exponent of the three chaotic generators (two-trajectory method) and the horizons in Lyapunov times | about 1 min (CPU) |
| 30 | operation counts: the counting rule of the energy model against the counts stored with every TRACE-SNP-AF forecast | seconds |
| 31 | how often TRACE-SNP-AF selects a fallback candidate; summary of cells 26–31 | seconds |
| 32 | every figure and table of the paper and of the supplementary material, including the extended comparison | 1–2 min |

## Outputs

```
results/
  data_main/, extra_data/               the 11 series exactly as used
  <dataset>/<network>/                  checkpoints, per-seed metrics, 60-seed ensemble predictions (preds_*.csv),
                                        walk-forward summaries, energy records, forecast figures
  <dataset>/closed_form/                TRACE-SNP v1, ablations, NG-RC, DLinear, ridge: predictions, choices, energy
  <dataset>/trace_snp_v2b/              TRACE-SNP v2: predictions, choices, operation counts, energy
  comparison/final_all11_*              leaderboards, tests, energy and noise tables, figures (cell 11)
  revisions/                            cells 12–25: ablation, ranking, basis study, TRACE-SNP-AF, control experiments
                                        (r2_*.csv), figures and LaTeX tables
  revisions/extended_checks/            cells 26–31: data manifest, stored-forecast checks, classical forecasters
                                        (statistical_baselines*.csv), constant forecasts, Lyapunov times,
                                        operation-count check, fallback shares, SUMMARY.md
paper_final/
  figures/*.pdf, *.png                  every figure of the paper and the supplement
  tables/*.tex, tables/r2_macros.tex    every generated table, and the numbers of cells 21–25 as LaTeX macros
  tables/ext_macros.tex                 the numbers of the extended comparison (cells 26–31) as LaTeX macros
  numbers.md                            every number the text quotes
paper_final.zip                         the paper_final/ folder in one archive
```

## Where each table and figure comes from

Cell 32 draws every file in `paper_final/`. It computes nothing new: it reads the results of the cells listed here.

| Output of cell 32 | Content | Results from cells |
|---|---|---|
| `fig_arch_alstm`, `fig_arch_trace`, `fig1_family`, `figS1_bound` | architecture and family diagrams; the bound illustration | – |
| `fig2_datasets` | the 11 series with their test blocks | 3 |
| `tab_scores`, `tab_scores_h1/h6/h24`, `fig4_scores`, `figS3_lags_*`, `figS4_horizons` | accuracy of every model per dataset, lag and horizon (`figS4_horizons`: the chaotic panels also give h = 24 in Lyapunov times) | 7–11, 18, 29 |
| `fig3_cd`, `figS2_cd_settings`, `tab_friedman`, `tab_pairwise_*`, `tab_winrate`, `tab_r2_bayes`, `tab_r2_rank17` | ranking over the 11 datasets and pairwise tests | 11, 14, 18, 21–24 |
| `fig5_ablation`, `figS5_rbf_problems`, `tab_basis_by_dataset` | firing-condition ablation and basis study | 13, 17 |
| `fig6_af`, `figS6_af_by_lag`, `figS7_dm_af_v2`, `tab_dm_af_v2` | TRACE-SNP-AF: chosen conditions, activity, tests against v2 | 18 |
| `fig7_variance`, `figS8_seed_cv`, `figS9_blocks`, `tab_seed_by_dataset`, `tab_blocks_full_*` | variance over seeds and over test blocks | 7, 19 |
| `fig8_energy` | accuracy against measured inference energy | 7, 8, 10, 15, 25 |
| `fig9_noise` | robustness to input noise | 7, 8, 11 |
| `fig10_delay`, `figS10_delay_maps` | learned delay functions with uncertainty bands | 8, 19 |
| `figS11_forecasts` | example forecasts | 7–10, 18 |
| `figS12_v2_choices`, `tab_v2_params`, `tab_nets` | selected configurations, parameter counts, training settings | 10, 12 |
| `tab_r2_tuning`, `tab_r2_direct`, `tab_r2_alstm`, `tab_r2_echo`, `tab_r2_energy`, `r2_macros` | control experiments | 21–25 |
| `tab_ext_ranks`, `figS18_extended_cd` | ranking over the 11 datasets with the classical forecasters added, with and without the Transformers | 7–11, 18, 28 |
| `tab_ext_classical` | score of every classical forecaster per dataset, with the selected ARIMA order | 18, 28 |
| `tab_ext_pairwise`, `figS19_extended_h2h` | TRACE-SNP-AF against each classical forecaster, DLinear and ridge AR(L): Wilcoxon–Holm and Bayesian signed-rank | 18, 28 |
| `tab_ext_constant` | constant forecasts at the training mean and median | 27 |
| `tab_ext_lyapunov` | Lyapunov exponents and the horizons in Lyapunov times | 29 |
| `ext_macros` | every number of the extended comparison that the text quotes | 26–31 |

## Reproducibility

- **Network seeds:** every network is trained with seeds 0–59 on every fold. The reported forecast is the mean over the 60 seeds.
- **Closed-form members:** they have no seeds. Their fits are deterministic up to floating-point summation order on the GPU.
- **Training precision:** fp32 throughout (`AMP_DTYPE = None`, `ALLOW_TF32 = False`).
- **Test blocks:** they are used only for the reported scores. Every choice of configuration is made on the training rows and the validation blocks.
- **Classical forecasters:** the ARIMA fits are maximum-likelihood optimisations, so a different `statsmodels` / SciPy version can move a few ARIMA scores in the third decimal. Cell 28 reuses `statistical_baselines.csv` and `statistical_baselines_by_h.csv` from `results/revisions/extended_checks/` when they are present, which reproduces the reported numbers exactly.
- **Stored-forecast check:** cell 26 stops if any stored forecast does not match its series (`EXT_STRICT = True`), so every number of the extended comparison is tied to the series of cell 3.

## References

The works cited in the paper, grouped by topic. Entries follow APA (7th edition) style.

### Spiking neural P systems

Guo, Y., Wang, J., Peng, H., Ramírez-de-Arellano, A., & Hu, H. (2026). Time series prediction based on multi-head gated fusion spiking neural P systems. *Knowledge-Based Systems*, *347*, 116223. <https://doi.org/10.1016/j.knosys.2026.116223>

Guo, Y., Wang, J., Peng, H., Wang, T., Hu, H., & Ramírez-de-Arellano, A. (2026). A multi-scale spatiotemporal spiking neural model for power load forecasting considering extreme weather impact. *International Journal of Electrical Power & Energy Systems*, *175*, 111604. <https://doi.org/10.1016/j.ijepes.2026.111604>

He, J., Peng, H., Wang, J., & Ramírez-de-Arellano, A. (2024). A deep echo-like spiking neural P systems for time series prediction. *Knowledge-Based Systems*, *304*, 112560. <https://doi.org/10.1016/j.knosys.2024.112560>

He, J., Peng, H., Wang, J., Yang, Q., & Ramírez-de-Arellano, A. (2024). K-order echo-type spiking neural P systems for time series forecasting. *Neurocomputing*, *610*, 128613. <https://doi.org/10.1016/j.neucom.2024.128613>

Huang, Y., Liu, Q., Peng, H., Wang, J., Yang, Q., & Orellana-Martín, D. (2023). Sentiment classification using bidirectional LSTM-SNP model and attention mechanism. *Expert Systems with Applications*, *221*, 119730. <https://doi.org/10.1016/j.eswa.2023.119730>

Ionescu, M., Păun, G., & Yokomori, T. (2006). Spiking neural P systems. *Fundamenta Informaticae*, *71*(2–3), 279–308. <https://doi.org/10.3233/FUN-2006-712-308>

Liu, Q., Long, L., Peng, H., Wang, J., Yang, Q., Song, X., Riscos-Núñez, A., & Pérez-Jiménez, M. J. (2023). Gated spiking neural P systems for time series forecasting. *IEEE Transactions on Neural Networks and Learning Systems*, *34*(9), 6227–6236. <https://doi.org/10.1109/TNNLS.2021.3134792>

Liu, Q., Long, L., Yang, Q., Peng, H., Wang, J., & Luo, X. (2022). LSTM-SNP: A long short-term memory model inspired from spiking neural P systems. *Knowledge-Based Systems*, *235*, 107656. <https://doi.org/10.1016/j.knosys.2021.107656>

Liu, Q., Peng, H., Long, L., Wang, J., Yang, Q., Pérez-Jiménez, M. J., & Orellana-Martín, D. (2024). Nonlinear spiking neural systems with autapses for predicting chaotic time series. *IEEE Transactions on Cybernetics*, *54*(3), 1841–1853. <https://doi.org/10.1109/TCYB.2023.3270873>

Long, L., Liu, Q., Peng, H., Yang, Q., Luo, X., Wang, J., & Song, X. (2022). A time series forecasting approach based on nonlinear spiking neural systems. *International Journal of Neural Systems*, *32*(8), 2250020. <https://doi.org/10.1142/S0129065722500204>

Long, L., Lugu, R., Xiong, X., Liu, Q., Peng, H., Wang, J., Orellana-Martín, D., & Pérez-Jiménez, M. J. (2022). Echo spiking neural P systems. *Knowledge-Based Systems*, *253*, 109568. <https://doi.org/10.1016/j.knosys.2022.109568>

Paul, P., Sosík, P., & Ciencialová, L. (2024). *A survey on learning models of spiking neural membrane systems and spiking neural networks* (arXiv:2403.18609) [Preprint]. arXiv. <https://doi.org/10.48550/arXiv.2403.18609>

Păun, G. (2000). Computing with membranes. *Journal of Computer and System Sciences*, *61*(1), 108–143. <https://doi.org/10.1006/jcss.1999.1693>

Peng, H., Lv, Z., Li, B., Luo, X., Wang, J., Song, X., Wang, T., Pérez-Jiménez, M. J., & Riscos-Núñez, A. (2020). Nonlinear spiking neural P systems. *International Journal of Neural Systems*, *30*(10), 2050008. <https://doi.org/10.1142/S0129065720500082>

Xu, J., Pan, J., Su, S., Li, H., Zhao, R., & Ding, F. (2026). Multivariate time series forecasting with a numerical spiking neural P system-inspired network. *Neurocomputing*, *700*, 134476. <https://doi.org/10.1016/j.neucom.2026.134476>

Zhang, Y., Yang, Q., Liu, Z., Peng, H., & Wang, J. (2023). A prediction model based on gated nonlinear spiking neural systems. *International Journal of Neural Systems*, *33*(6), 2350029. <https://doi.org/10.1142/S0129065723500296>

### Spiking neural networks and neuromorphic computing

Bakhshaliyev, J., & Landwehr, N. (2026). *SpikF-GO: Spiking Fourier graph operators for multivariate time series forecasting* (arXiv:2606.13901) [Preprint]. arXiv. Accepted at ECML PKDD 2026. <https://doi.org/10.48550/arXiv.2606.13901>

Bellec, G., Salaj, D., Subramoney, A., Legenstein, R., & Maass, W. (2018). Long short-term memory and learning-to-learn in networks of spiking neurons. In *Advances in Neural Information Processing Systems* (Vol. 31, pp. 787–797). <https://doi.org/10.48550/arXiv.1803.09574>

Chakraborty, B., & Mukhopadhyay, S. (2023). Brain-inspired spiking neural network for online unsupervised time series prediction. In *2023 International Joint Conference on Neural Networks (IJCNN)* (pp. 1–8). IEEE. <https://doi.org/10.1109/IJCNN54540.2023.10191645>

Davies, M., et al. (2018). Loihi: A neuromorphic manycore processor with on-chip learning. *IEEE Micro*, *38*(1), 82–99. <https://doi.org/10.1109/MM.2018.112130359>

Feng, S., Feng, W., Gao, X., Zhao, P., & Shen, Z. (2025). TS-LIF: A temporal segment spiking neuron network for time series forecasting. In *International Conference on Learning Representations (ICLR)* (pp. 28064–28084). <https://doi.org/10.48550/arXiv.2503.05108>

Horowitz, M. (2014). Computing's energy problem (and what we can do about it). In *IEEE International Solid-State Circuits Conference (ISSCC)* (pp. 10–14). IEEE. <https://doi.org/10.1109/ISSCC.2014.6757323>

Hu, B., Lv, C., Li, M., Liu, Y., Zheng, X., Zhang, F., Cao, W., & Zhang, F. (2025). *SpikeSTAG: Spatial-temporal forecasting via GNN-SNN collaboration* (arXiv:2508.02069) [Preprint]. arXiv. <https://doi.org/10.48550/arXiv.2508.02069>

Lucas, S., & Portillo, E. (2024). Methodology based on spiking neural networks for univariate time-series forecasting. *Neural Networks*, *173*, 106171. <https://doi.org/10.1016/j.neunet.2024.106171>

Lv, C., Wang, Y., Han, D., Zheng, X., Huang, X., & Li, D. (2024). Efficient and effective time-series forecasting with spiking neural networks. In *Proceedings of the 41st International Conference on Machine Learning* (Proceedings of Machine Learning Research, Vol. 235, pp. 33624–33637). PMLR. <https://doi.org/10.48550/arXiv.2402.01533>

Maass, W. (1997). Networks of spiking neurons: The third generation of neural network models. *Neural Networks*, *10*(9), 1659–1671. <https://doi.org/10.1016/S0893-6080(97)00011-7>

Meshram, N., Anand, V., & Tiwari, V. (2026). TSA-Net+: A communication-aware hybrid ANN-SNN framework for energy-efficient IoT inference. *Internet of Things*, *40*, 102117. <https://doi.org/10.1016/j.iot.2026.102117>

Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks. *IEEE Signal Processing Magazine*, *36*(6), 51–63. <https://doi.org/10.1109/MSP.2019.2931595>

Tang, K., Zheng, J., Jin, Y., Qiu, Y., Sun, G., Yan, Z., & Wong, W.-F. (2026). *SpikySpace: A spiking state space model for energy-efficient time series forecasting* (arXiv:2601.02411) [Preprint]. arXiv. <https://doi.org/10.48550/arXiv.2601.02411>

Wu, W., Huo, D., & Chen, H. (2025). SpikF: Spiking Fourier network for efficient long-term prediction. In *Proceedings of the 42nd International Conference on Machine Learning* (Proceedings of Machine Learning Research, Vol. 267, pp. 67342–67368). PMLR. <https://proceedings.mlr.press/v267/wu25m.html>

### Fuzzy rule bases, closed-form and reservoir models

Allen, D. M. (1974). The relationship between variable selection and data augmentation and a method for prediction. *Technometrics*, *16*(1), 125–127. <https://doi.org/10.1080/00401706.1974.10489157>

Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. *Statistical Science*, *11*(2), 89–121. <https://doi.org/10.1214/ss/1038425655>

Gauthier, D. J., Bollt, E., Griffith, A., & Barbosa, W. A. S. (2021). Next generation reservoir computing. *Nature Communications*, *12*, 5564. <https://doi.org/10.1038/s41467-021-25801-2>

Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, *12*(1), 55–67. <https://doi.org/10.1080/00401706.1970.10488634>

Jaeger, H., & Haas, H. (2004). Harnessing nonlinearity: Predicting chaotic systems and saving energy in wireless communication. *Science*, *304*(5667), 78–80. <https://doi.org/10.1126/science.1091277>

Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based fuzzy inference system. *IEEE Transactions on Systems, Man, and Cybernetics*, *23*(3), 665–685. <https://doi.org/10.1109/21.256541>

Park, J., & Sandberg, I. W. (1991). Universal approximation using radial-basis-function networks. *Neural Computation*, *3*(2), 246–257. <https://doi.org/10.1162/neco.1991.3.2.246>

Takagi, T., & Sugeno, M. (1985). Fuzzy identification of systems and its applications to modeling and control. *IEEE Transactions on Systems, Man, and Cybernetics*, *SMC-15*(1), 116–132. <https://doi.org/10.1109/TSMC.1985.6313399>

Wang, L.-X., & Mendel, J. M. (1992). Generating fuzzy rules by learning from examples. *IEEE Transactions on Systems, Man, and Cybernetics*, *22*(6), 1414–1427. <https://doi.org/10.1109/21.199466>

### Neural forecasting baselines

Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. In *International Conference on Learning Representations (ICLR)* (pp. 1–15). <https://doi.org/10.48550/arXiv.1409.0473>

Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 1724–1734). Association for Computational Linguistics. <https://doi.org/10.3115/v1/D14-1179>

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, *9*(8), 1735–1780. <https://doi.org/10.1162/neco.1997.9.8.1735>

Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. In *Advances in Neural Information Processing Systems* (Vol. 34, pp. 22419–22430). <https://doi.org/10.48550/arXiv.2106.13008>

Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are transformers effective for time series forecasting? In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 37, pp. 11121–11128). <https://doi.org/10.1609/aaai.v37i9.26317>

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 35, pp. 11106–11115). <https://doi.org/10.1609/aaai.v35i12.17325>

Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., & Jin, R. (2022). FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. In *Proceedings of the 39th International Conference on Machine Learning* (Proceedings of Machine Learning Research, Vol. 162, pp. 27268–27286). PMLR. <https://doi.org/10.48550/arXiv.2201.12740>

### Classical forecasting and dynamical systems

Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, *19*(6), 716–723. <https://doi.org/10.1109/TAC.1974.1100705>

Benettin, G., Galgani, L., Giorgilli, A., & Strelcyn, J.-M. (1980). Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems; a method for computing all of them. Part 1: Theory. *Meccanica*, *15*(1), 9–20. <https://doi.org/10.1007/BF02128236>

Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time series analysis: Forecasting and control* (5th ed.). Wiley.

Gardner, E. S., Jr., & McKenzie, E. (1985). Forecasting trends in time series. *Management Science*, *31*(10), 1237–1246. <https://doi.org/10.1287/mnsc.31.10.1237>

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTexts. <https://otexts.com/fpp3/>

### Datasets

Lorenz, E. N. (1963). Deterministic nonperiodic flow. *Journal of the Atmospheric Sciences*, *20*(2), 130–141. [https://doi.org/10.1175/1520-0469(1963)020&lt;0130:DNF&gt;2.0.CO;2](https://doi.org/10.1175/1520-0469%281963%29020%3C0130%3ADNF%3E2.0.CO%3B2)

Mackey, M. C., & Glass, L. (1977). Oscillation and chaos in physiological control systems. *Science*, *197*(4300), 287–289. <https://doi.org/10.1126/science.267326>

Rössler, O. E. (1976). An equation for continuous chaos. *Physics Letters A*, *57*(5), 397–398. <https://doi.org/10.1016/0375-9601(76)90101-8>

### Evaluation and statistical testing

Benavoli, A., Corani, G., Demšar, J., & Zaffalon, M. (2017). Time for a change: A tutorial for comparing multiple classifiers through Bayesian analysis. *Journal of Machine Learning Research*, *18*(77), 1–36. <https://doi.org/10.48550/arXiv.1606.04316>

Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, *191*, 192–213. <https://doi.org/10.1016/j.ins.2011.12.028>

Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, *7*, 1–30. <https://www.jmlr.org/papers/v7/demsar06a.html>

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, *13*(3), 253–263. <https://doi.org/10.1080/07350015.1995.10524599>

Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *Journal of the American Statistical Association*, *32*(200), 675–701. <https://doi.org/10.1080/01621459.1937.10503522>

García, S., Fernández, A., Luengo, J., & Herrera, F. (2010). Advanced nonparametric tests for multiple comparisons in the design of experiments in computational intelligence and data mining: Experimental analysis of power. *Information Sciences*, *180*(10), 2044–2064. <https://doi.org/10.1016/j.ins.2009.12.010>

Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, *13*(2), 281–291. <https://doi.org/10.1016/S0169-2070(96)00719-4>

Hollander, M., Wolfe, D. A., & Chicken, E. (2015). *Nonparametric statistical methods* (3rd ed.). Wiley. <https://doi.org/10.1002/9781119196037>

Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, *6*(2), 65–70. <https://doi.org/10.2307/4615733>

Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, *22*(4), 679–688. <https://doi.org/10.1016/j.ijforecast.2006.03.001>

Iman, R. L., & Davenport, J. M. (1980). Approximations of the critical region of the Friedman statistic. *Communications in Statistics – Theory and Methods*, *9*(6), 571–595. <https://doi.org/10.1080/03610928008827904>

Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. *International Journal of Forecasting*, *16*(4), 437–450. <https://doi.org/10.1016/S0169-2070(00)00065-0>

Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, *1*(6), 80–83. <https://doi.org/10.2307/3001968>

### Mathematical and statistical foundations

Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. In *Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA)* (pp. 1027–1035). Society for Industrial and Applied Mathematics. <https://dl.acm.org/doi/10.5555/1283383.1283494>

de Boor, C. (2001). *A practical guide to splines* (Rev. ed.). Springer. <https://doi.org/10.1007/978-1-4612-6333-3>

Golub, G. H., & Van Loan, C. F. (2013). *Matrix computations* (4th ed.). Johns Hopkins University Press. <https://doi.org/10.56021/9781421407944>

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning* (2nd ed.). Springer. <https://doi.org/10.1007/978-0-387-84858-7>

Horn, R. A., & Johnson, C. R. (2012). *Matrix analysis* (2nd ed.). Cambridge University Press. <https://doi.org/10.1017/CBO9781139020411>

Katznelson, Y. (2004). *An introduction to harmonic analysis* (3rd ed.). Cambridge University Press. <https://doi.org/10.1017/CBO9781139165372>

Lloyd, S. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, *28*(2), 129–137. <https://doi.org/10.1109/TIT.1982.1056489>

Sherman, J., & Morrison, W. J. (1950). Adjustment of an inverse matrix corresponding to a change in one element of a given matrix. *The Annals of Mathematical Statistics*, *21*(1), 124–127. <https://doi.org/10.1214/aoms/1177729893>

Stone, M. (1974). Cross-validatory choice and assessment of statistical predictions. *Journal of the Royal Statistical Society: Series B (Methodological)*, *36*(2), 111–133. <https://doi.org/10.1111/j.2517-6161.1974.tb00994.x>

White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, *48*(4), 817–838. <https://doi.org/10.2307/1912934>

## License

MIT License. See `LICENSE`.
