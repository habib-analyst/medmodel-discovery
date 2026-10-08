"""Smoke tests for medmodel-discovery (fast, CPU)."""
import torch

from models import build_model, count_params
from losses import FocalLoss, CBFocalLoss, DACFLoss
from tta import predict_single, predict_naive_tta, predict_cg_ttc


def _tiny_batch(n=8, c=3):
    return torch.rand(n, c, 28, 28)


def test_model_shapes():
    for name, ncls in [("plannet", 2), ("plannet", 7),
                       ("gcvit-perceiver-lite", 2), ("resnet18", 8),
                       ("vit-tiny", 2)]:
        m = build_model(name, 3, ncls)
        y = m(_tiny_batch(4))
        assert y.shape == (4, ncls), (name, y.shape)


def test_plannet_param_budget():
    m = build_model("plannet", 3, 7)
    assert count_params(m) <= 1_500_000, count_params(m)


def test_plannet_gating_changes_output():
    # top-down gates must actually modulate the fine-level output
    torch.manual_seed(0)
    m = build_model("plannet", 3, 2)
    x = _tiny_batch(4)
    with torch.no_grad():
        y1 = m(x)
    # zero the gate MLPs -> gates saturate at sigmoid(0)=0.5, output must differ
    with torch.no_grad():
        for mod in [m.lvl2.gate, m.lvl1.gate]:
            for p in mod.parameters():
                p.zero_()
        y2 = m(x)
    assert not torch.allclose(y1, y2), "gates have no effect!"


def test_losses_finite():
    logits = torch.randn(16, 7)
    targets = torch.randint(0, 7, (16,))
    counts = [100, 50, 30, 20, 10, 5, 2]
    for loss in [torch.nn.CrossEntropyLoss(), FocalLoss(),
                 CBFocalLoss(counts), DACFLoss(counts)]:
        v = loss(logits, targets)
        assert torch.isfinite(v), type(loss)


def test_dacf_margin_activates_on_miscalibration():
    counts = [1000, 10]
    loss = DACFLoss(counts, lam=1.0)
    # feed overconfident wrong predictions for class 1 repeatedly
    for _ in range(20):
        logits = torch.tensor([[5.0, -5.0]] * 8)  # confident class 0
        targets = torch.ones(8, dtype=torch.long)  # but true class is 1
        loss(logits, targets)
    margins = loss.margins()
    # margin activates on the TRUE class (1) whose samples are predicted
    # overconfidently wrong — that is the class whose loss gets the penalty
    assert margins[1] > 0, f"margin should activate for miscalibrated class 1: {margins}"


def test_tta_shapes_and_referral():
    m = build_model("resnet18", 3, 2)
    x = _tiny_batch(8)
    p1 = predict_single(m, x)
    assert p1.shape == (8, 2)
    pn = predict_naive_tta(m, x, k=4)
    assert pn.shape == (8, 2)
    po, ref, js = predict_cg_ttc(m, x, k=4, tau=0.05)
    assert po.shape == (8, 2) and ref.shape == (8,) and js.shape == (8,)
    assert ((ref == 0) | (ref == 1)).all()
    # tau=0 -> everything referred; tau=inf -> nothing referred
    _, ref_all, _ = predict_cg_ttc(m, x, k=4, tau=0.0)
    assert (ref_all == 1).all()
    _, ref_none, _ = predict_cg_ttc(m, x, k=4, tau=1e9)
    assert (ref_none == 0).all()


def test_transformers_do_not_collapse_to_constant():
    """Regression test for the 2026-10-08 collapse bug.

    gcvit-perceiver-lite and vit-tiny trained to CONSTANT majority-class
    predictions: val_acc identical across seeds (0.7424 pneumoniamnist),
    single predicted class, val AUC 0.61-0.75 -- while plannet/resnet18
    trained normally. The 6 pre-existing smoke tests did not catch it.

    Root cause (models.py): LayerNorm leaves the batch-mean (constant)
    component of the transformer residual stream unconstrained, so the head
    weights and that constant component form a runaway feedback loop
    (||c|| 10 -> 118 in one epoch) whose constant logit offset saturates the
    softmax; the per-block post-norms additionally attenuate the
    input-dependent signal geometrically (0.086 -> 1e-4 across 4 blocks).
    Fix: pre-norm transformer blocks + BatchNorm1d (not LayerNorm) before
    the classification head.

    A short 2-epoch run must leave a strong discriminative signal: collapsed
    models score AUC 0.61-0.75 here, fixed models 0.89+.
    """
    torch.set_num_threads(2)
    from train import train_one
    for name in ["vit-tiny", "gcvit-perceiver-lite"]:
        res, _ = train_one(name, "pneumoniamnist", "ce", epochs=2, seed=0,
                           subsample=1500, batch=128)
        auc = res["val"]["auc"]
        assert auc > 0.80, (
            f"{name} looks collapsed after short train: val_auc={auc:.4f} "
            f"(collapsed models score 0.61-0.75, healthy ones 0.89+)")
