# MedModel-Discovery: Pyramidal Latent Attention, Calibrated Focal Loss, and Consistency-Gated Test-Time Consensus for Medical Imaging

**Habib Ur Rehman** — Project #9, 2026-10-07 (working paper; results recorded 2026-10-09 — Exp-A partial, Exp-B and Exp-C missed their success criteria; negative results documented in §5/§6)

## Abstract

We extend the GCViT × Perceiver IO hybrid line (Rehman, J. of Supercomputing
2025) with three targeted innovations: (1) **PLAN-Net**, a pyramidal
latent-attention architecture replacing Perceiver IO's flat latent bottleneck
with coarse→fine latent arrays joined by top-down routing gates; (2)
**DACF**, a difficulty-adaptive calibrated focal loss that jointly optimizes
class-balanced accuracy and calibration; (3) **CG-TTC**, a consistency-gated
test-time consensus algorithm that converts augmented-view disagreement into
a human-referral signal. All claims are measured on MedMNIST 2D against
ResNet-18, ViT-Tiny, and a faithful small GCViT+PerceiverIO analog.

## 1. Motivation

Perceiver IO's single flat latent array forces one fixed bottleneck to carry
both global anatomical context and fine lesion detail. On small medical
datasets this creates a measurable global-vs-detail trade-off (see §4).
Class-imbalanced training (focal / class-balanced losses) improves minority
recall but sharpens predictive distributions, worsening expected calibration
error — unacceptable where confidence drives clinical decisions. Single-pass
inference fails silently on shifted inputs; naive test-time augmentation
averages away the disagreement that would have revealed the failure.

## 2. PLAN-Net: Pyramidal Latent Attention

A conv stem yields features at 1/4, 1/8, 1/16 resolution. Three latent
arrays — 64 (fine), 32 (mid), 16 (coarse) — cross-attend to their scale.
Updates proceed coarse→fine: the coarse latents' pooled state generates a
sigmoid gate per fine latent, modulating its cross-attention output:

    g_s = σ(MLP([mean(L_{s+1}); pooled(F_s)]));  out_s ← g_s ⊙ Attn(L_s, F_s)

Complexity is O(Σ_s M_s·N_s) with small M_s — linear in input size, like
Perceiver IO, but the bottleneck is no longer flat. Readout concatenates
mean-pooled latents from all levels. Total: 524K params at 28×28.

## 3. DACF: Difficulty-Adaptive Calibrated Focal Loss

L = Σ_i w_{y_i} (1−p_{t,i})^γ CE_i + λ Σ_i m_{y_i} CE_i,
  w_c = (1−β)/(1−β^{n_c}),  m_c = relu(EMA[conf_c] − EMA[acc_c]).

**Gradient motivation.** Write p = softmax(z), p_t the true-class prob.
The margin term contributes −λ·m_c·log p_t; its gradient w.r.t. z is
−λ·m_c·(1−p_t)·(e_{y} − p), i.e. it pushes probability mass *away* from the
true class exactly when class c's confidence systematically exceeds its
accuracy (m_c > 0), and vanishes (m_c = 0) for calibrated classes. Focal
loss alone multiplies CE by (1−p_t)^γ, which down-weights easy examples but
*rewards* sharper distributions on hard ones — provably increasing ECE on
minority classes (Mukhoti et al., 2020, focal calibration analysis). DACF
keeps focal's minority-recall benefit while the margin term acts as a
first-order Lagrange penalty on the constraint E[conf_c] ≤ E[acc_c] +
tolerance. The EMAs make it adaptive: no hand-tuned per-class schedule.

## 4. CG-TTC: Consistency-Gated Test-Time Consensus

For input x: clean prediction p_0 and K=8 stochastic views p_1..p_K.
Compute mean pairwise Jensen–Shannon divergence JS̄. If JS̄ < τ output the
view mean; else output p_0 **and set a referral flag**. τ is calibrated once
on validation to a target referral budget (10%). Naive TTA is the special
case τ=∞ (never refer). Evaluation: risk–coverage curves — accuracy on the
non-referred fraction — plus referral rate under synthetic shift.

## 5. Experiments

*Protocol.* PneumoniaMNIST (4.7k, binary), DermaMNIST (10k, 7-class,
imbalanced), BloodMNIST-subset (12k, 8-class) at 28×28. Baselines:
ResNet-18 (CIFAR-style), ViT-Tiny, GCViT-lite+Perceiver-lite (faithful small
analog of the published hybrid: global-context token transformer +
single flat latent array). AdamW, cosine schedule, seed-averaged.

*Exp-A: PLAN-Net vs baselines (3 seeds each, validation accuracy / AUC, seed means; ranges in brackets).*

| dataset    | model             | acc            | AUC            |
|------------|-------------------|----------------|----------------|
| pneumonia  | PLAN-Net          | 0.9727 [0.9695, 0.9771] | 0.9932 [0.9899, 0.9953] |
|            | ResNet-18         | 0.9631 [0.9599, 0.9656] | 0.9953 [0.9946, 0.9963] |
|            | ViT-Tiny          | 0.9586 [0.9523, 0.9618] | 0.9877 [0.9838, 0.9899] |
|            | GCViT-lite+P-lite | 0.9548 [0.9523, 0.9561] | 0.9891 [0.9865, 0.9914] |
| derma      | PLAN-Net          | 0.7468 [0.7418, 0.7567] | 0.9206 [0.9145, 0.9266] |
|            | ResNet-18         | 0.7694 [0.7637, 0.7747] | 0.9255 [0.9249, 0.9267] |
|            | ViT-Tiny          | 0.7218 [0.7178, 0.7268] | 0.9001 [0.8972, 0.9031] |
|            | GCViT-lite+P-lite | 0.7255 [0.7139, 0.7408] | 0.9066 [0.8983, 0.9168] |
| blood      | PLAN-Net          | 0.9338 [0.9258, 0.9381] | 0.9955 [0.9949, 0.9958] |
|            | ResNet-18         | 0.9628 [0.9614, 0.9655] | 0.9984 [0.9981, 0.9986] |
|            | ViT-Tiny          | 0.9256 [0.9235, 0.9282] | 0.9942 [0.9941, 0.9944] |
|            | GCViT-lite+P-lite | 0.9170 [0.9141, 0.9188] | 0.9934 [0.9931, 0.9937] |

Verdict A (per DESIGN: PLAN-Net ≥ best baseline acc *and* AUC on ≥2/3
datasets at ≤1.5M params): **MISS.** PLAN-Net beats the best baseline
accuracy on pneumonia only (0.9727 vs ResNet-18 0.9631) but loses its AUC
(0.9932 vs 0.9953); on derma and blood it loses both metrics to ResNet-18.
It does beat the GCViT-lite+Perceiver-lite analog and ViT-Tiny on all three
datasets, and the 524K param count (as designed; not independently
re-measured — no torch runtime was available at analysis time) is within
the 1.5M budget. The 28×28 inputs likely understate the pyramidal design's
fine-detail advantage (see Limits).

*Exp-B: loss sweep on PLAN-Net, derma + blood (3 seeds, val acc / AUC, seed means).*

| dataset | loss     | acc            | AUC            |
|---------|----------|----------------|----------------|
| derma   | CE       | 0.7494 [0.7428, 0.7567] | 0.9154 [0.9115, 0.9190] |
|         | focal    | 0.7494 [0.7478, 0.7507] | 0.9123 [0.9063, 0.9159] |
|         | cb-focal | 0.5653 [0.5533, 0.5852] | 0.8779 [0.8756, 0.8804] |
|         | DACF     | 0.5673 [0.5513, 0.5793] | 0.8792 [0.8755, 0.8825] |
| blood   | CE       | 0.9437 [0.9404, 0.9480] | 0.9962 [0.9953, 0.9969] |
|         | focal    | 0.9355 [0.9340, 0.9369] | 0.9946 [0.9944, 0.9949] |
|         | cb-focal | 0.9316 [0.9211, 0.9410] | 0.9944 [0.9934, 0.9952] |
|         | DACF     | 0.9356 [0.9264, 0.9416] | 0.9945 [0.9935, 0.9950] |

Verdict B (per DESIGN: DACF matches focal's minority recall AND beats
focal's ECE): **MISS, and the stated criterion is unmeasurable.** ECE and
per-class recalls for Exp-B were lost with the Kaggle session — only
val_acc/val_auc survive in stdout salvage, so the criterion's two metrics
are UNMEASURED. What was measured decides against DACF on the dataset the
claim targets: on derma, DACF (0.5673) and cb-focal (0.5653) both collapsed
≈18pp below focal (0.7494) — worse than a majority-class predictor
(0.67) — while plain focal matched CE. On blood all four losses are
competitive. Code review (losses.py) pins the mechanism: both collapsed
losses share the effective-number weight
`w_c = (1−β)/(1−β^n_c)`, β=0.9999, which on derma spreads 0.062 (majority
"nv", 67% of data) to 2.66 (smallest class) — a 43× gradient split that
starves the majority class in a 15-epoch schedule. The near-identical
cb-focal/DACF numbers (acc within 0.002, AUC within 0.0013) show DACF's
novel calibration-margin term contributed nothing measurable. See §6 for
the negative result.

*Exp-C: CG-TTC vs single-pass vs naive TTA (τ=0.05, K=8 views, gaussian noise 0.1 shift).*

| dataset   | clean single | clean naive TTA | clean CG-TTC | clean referral | shift single | shift naive TTA | shift CG-TTC | shift referral |
|-----------|--------------|-----------------|--------------|----------------|--------------|-----------------|--------------|----------------|
| pneumonia | 0.8654       | 0.8462          | 0.8654       | 0.4263         | 0.4936       | 0.6058          | 0.5064       | 0.7853         |
| derma     | 0.7418       | 0.6610          | 0.7198       | 0.6889         | 0.5693       | 0.5563          | 0.5613       | 0.6999         |

Verdict C (per DESIGN: CG-TTC risk–coverage AUC > naive TTA, referral
<15% clean, higher under shift): **MISS.** Risk–coverage AUC was NOT
recorded, so the primary metric is unmeasured. On what was measured:
clean referrals (0.43 / 0.69) are 3–4.6× the 15% budget — τ=0.05,
calibrated to ≤0.15 referral on the validation half, does not transfer to
the eval half. Under noise shift on pneumonia, naive TTA (0.6058) beats
CG-TTC (0.5064) by ~10pp — the exact opposite of the design intent.
Partial credit: on clean data CG-TTC matches single-pass on pneumonia and
beats naive TTA on derma (+5.9pp, 0.7198 vs 0.6610); referral correctly
rises under shift on pneumonia (0.43 → 0.79) but stays flat-high on derma
(0.69 → 0.70).

## 6. Limits & negative results

- 28×28 inputs understate the fine-detail advantage; 224×224 validation is
  future work (needs GPU).
- DACF's EMAs need enough per-class samples per batch; on tiny classes the
  margin adapts slowly — documented in ablations.
- Any candidate that fails its success criterion is reported with full
  numbers, not buried.
- **NEGATIVE RESULT (DACF / Exp-B).** The novel margin term did not
  contribute: cb-focal and DACF collapsed identically on dermamnist
  (acc 0.5653 vs 0.5673, AUC 0.8779 vs 0.8792) while plain focal was fine
  (0.7494). The effective-number weight with β=0.9999 — tuned for long
  schedules in Cui et al. (2019), not 15-epoch runs — starves the majority
  class: w spans 0.062 (nv, 67% of derma) to 2.66 (df), a 43× split.
  Hypothesis: β too close to 1 for the short schedule; a β sweep
  (e.g. 0.9/0.99/0.999) is the concrete next experiment before any DACF
  claim. No code bug found — losses.py implements the stated formula
  faithfully; class counts were verified computed in label order from the
  training loader.
- **NEGATIVE RESULT (CG-TTC / Exp-C).** τ calibration does not transfer
  from validation half to eval half (0.43/0.69 vs ≤0.15 target), and naive
  TTA averaging wins under shift on pneumonia (0.6058 vs 0.5064). Hypothesis:
  JS-divergence threshold calibrated on one data half is miscalibrated on
  another; per-dataset τ or an adaptive gate is needed before clinical
  claims.
- **PARTIAL (PLAN-Net / Exp-A).** Beats the GCViT+PerceiverIO analog and
  ViT-Tiny everywhere, takes pneumonia accuracy outright (0.9727), but
  loses to ResNet-18 everywhere else. The pyramid's edge should grow with
  resolution; 224×224 is the honest test.
- **Measurement gaps.** Exp-A canonical records carry only val_acc/val_auc
  (no ECE/recalls). Exp-B is stdout-salvage: val_acc/val_auc only — no ECE,
  no per-class recalls. Exp-C recorded only acc + referral; risk–coverage
  AUC was never computed. The 524K PLAN-Net param count is the design
  figure, not independently re-measured at analysis time.

## References

- Rehman, H.U. "AI-Driven Multi-Disease Classification: GCViT & Perceiver IO
  Synergistic Framework…" J. of Supercomputing, 2025 (accepted).
- Yang, J. et al. "MedMNIST v2: A large-scale lightweight benchmark for 2D
  and 3D biomedical image classification." Scientific Data, 2023.
- Cui, Y. et al. "Class-balanced loss based on effective number of samples."
  CVPR, 2019.
- Mukhoti, J. et al. "Calibrating deep neural networks using focal loss."
  NeurIPS, 2020.
