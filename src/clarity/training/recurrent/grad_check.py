"""Numerical gradient validation.

For each module/loss, builds a tiny random forward and compares the hand-
written analytic gradient to a central-difference numerical gradient. To
avoid fp32 noise dominating the comparison, the test upcasts all layers
and inputs to fp64 (analytic and numerical both run in double precision).

Tolerance: |a - n| < atol + rtol * max(|a|, |n|).  Defaults atol=1e-6,
rtol=1e-4 are tight enough to catch real bugs (sign errors, transposes,
wrong gate ordering) without firing on legitimate fp64 rounding.

Run: python -m clarity.training.recurrent.grad_check
"""

from __future__ import annotations

import numpy as np

from .nn import Linear, GRU, GRUCell, relu_forward, relu_backward
from .losses import cross_entropy, ppo_update_grads
from .policy import RecurrentActorCritic


EPS = 1e-5            # central-difference step in fp64
ATOL = 1e-6
RTOL = 1e-4


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

def _upcast_module(mod, dtype=np.float64):
    """Recursively upcast all named parameters on a module to `dtype`."""
    for name in getattr(mod, "_param_names", ()):
        setattr(mod, name, getattr(mod, name).astype(dtype))
    # Recurse one level for composite modules (GRU has .cell;
    # RecurrentActorCritic has encoder/gru/actor/value)
    for attr in ("cell", "encoder", "gru", "actor", "value"):
        sub = getattr(mod, attr, None)
        if sub is not None and hasattr(sub, "_param_names") or hasattr(sub, "cell"):
            _upcast_module(sub, dtype)


# ---------------------------------------------------------------------------
# numerical gradient via central differences
# ---------------------------------------------------------------------------

def numerical_grad(forward_fn, params, n_check_per_param: int = 16):
    """Return {name: grad_array_with_NaN_at_unchecked_positions}."""
    rng = np.random.default_rng(0)
    out = {}
    for name, p in params.items():
        flat = p.ravel()
        grad = np.full(flat.shape, np.nan, dtype=np.float64)
        idx = rng.choice(len(flat), min(n_check_per_param, len(flat)),
                         replace=False)
        for i in idx:
            orig = float(flat[i])
            flat[i] = orig + EPS
            f_plus = float(forward_fn())
            flat[i] = orig - EPS
            f_minus = float(forward_fn())
            flat[i] = orig
            grad[i] = (f_plus - f_minus) / (2.0 * EPS)
        out[name] = grad.reshape(p.shape)
    return out


def compare(name, analytic, numerical, atol=ATOL, rtol=RTOL):
    a = analytic.ravel().astype(np.float64)
    n = numerical.ravel().astype(np.float64)
    mask = ~np.isnan(n)
    if not mask.any():
        print(f"  [ok ] {name:24s}  (no samples)")
        return 0.0, True
    a, n = a[mask], n[mask]
    diff = np.abs(a - n)
    threshold = atol + rtol * np.maximum(np.abs(a), np.abs(n))
    excess = diff / threshold
    max_excess = float(excess.max())
    ok = max_excess <= 1.0
    flag = "ok " if ok else "FAIL"
    worst = int(excess.argmax())
    print(f"  [{flag}] {name:24s}  max_excess={max_excess:.3e}  "
          f"a={a[worst]:+.4e}  n={n[worst]:+.4e}  diff={diff[worst]:.2e}")
    return max_excess, ok


# ============================================================================
# Tests
# ============================================================================

def test_linear():
    print("\n== Linear ==")
    rng = np.random.default_rng(0)
    layer = Linear(in_dim=5, out_dim=4, seed=1)
    _upcast_module(layer)
    x = rng.standard_normal((3, 5))                # fp64
    target = rng.standard_normal((3, 4))

    def fwd():
        out, _ = layer.forward(x)
        return ((out - target) ** 2).mean()

    out, cache = layer.forward(x)
    dout = 2.0 * (out - target) / out.size
    dx, grads = layer.backward(dout, cache)

    num = numerical_grad(fwd, {"W": layer.W, "b": layer.b})
    ok_all = True
    for k in ("W", "b"):
        _, ok = compare(f"Linear.{k}", grads[k], num[k]);   ok_all = ok_all and ok
    num_x = numerical_grad(fwd, {"x": x})
    _, ok = compare("Linear.dx", dx, num_x["x"]);   ok_all = ok_all and ok
    return ok_all


def test_gru_cell():
    print("\n== GRUCell ==")
    rng = np.random.default_rng(1)
    cell = GRUCell(input_dim=4, hidden_dim=6, seed=2)
    _upcast_module(cell)
    x = rng.standard_normal((2, 4))
    h0 = rng.standard_normal((2, 6))
    target = rng.standard_normal((2, 6))

    def fwd():
        h_new, _ = cell.forward(x, h0)
        return ((h_new - target) ** 2).mean()

    h_new, cache = cell.forward(x, h0)
    dh_new = 2.0 * (h_new - target) / h_new.size
    dx, dh_prev, grads = cell.backward(dh_new, cache)

    num = numerical_grad(fwd, {k: getattr(cell, k) for k in cell._param_names})
    ok_all = True
    for k in cell._param_names:
        _, ok = compare(f"GRUCell.{k}", grads[k], num[k]);  ok_all = ok_all and ok
    num_xh = numerical_grad(fwd, {"x": x, "h0": h0})
    _, ok1 = compare("GRUCell.dx",      dx,      num_xh["x"])
    _, ok2 = compare("GRUCell.dh_prev", dh_prev, num_xh["h0"])
    return ok_all and ok1 and ok2


def test_gru_sequence():
    print("\n== GRU.sequence ==")
    rng = np.random.default_rng(2)
    gru = GRU(input_dim=3, hidden_dim=4, seed=3)
    _upcast_module(gru)
    B, T = 2, 5
    x = rng.standard_normal((B, T, 3))
    h0 = rng.standard_normal((B, 4))
    target = rng.standard_normal((B, T, 4))

    def fwd():
        out, _, _ = gru.forward_sequence(x, h0)
        return ((out - target) ** 2).mean()

    out, _, caches = gru.forward_sequence(x, h0)
    dout = 2.0 * (out - target) / out.size
    dx, dh0, grads = gru.backward_sequence(dout, caches)

    num = numerical_grad(fwd, {k: getattr(gru.cell, k)
                                for k in gru.cell._param_names})
    ok_all = True
    for k in gru.cell._param_names:
        _, ok = compare(f"GRU.seq.{k}", grads[k], num[k]);  ok_all = ok_all and ok
    num_xh = numerical_grad(fwd, {"x": x, "h0": h0})
    _, ok1 = compare("GRU.seq.dx",  dx,  num_xh["x"])
    _, ok2 = compare("GRU.seq.dh0", dh0, num_xh["h0"])
    return ok_all and ok1 and ok2


def test_cross_entropy():
    print("\n== cross_entropy ==")
    rng = np.random.default_rng(3)
    logits = rng.standard_normal((5, 4))
    targets = rng.integers(0, 4, size=5)

    def fwd():
        l, _ = cross_entropy(logits, targets)
        return l

    _, dlogits = cross_entropy(logits, targets)
    num = numerical_grad(fwd, {"logits": logits})
    _, ok = compare("ce.dlogits", dlogits, num["logits"])
    return ok


def test_ppo_grads():
    print("\n== ppo_update_grads ==")
    rng = np.random.default_rng(4)
    B, T, C = 3, 4, 5
    logits = rng.standard_normal((B, T, C))
    values = rng.standard_normal((B, T))
    actions = rng.integers(0, C, size=(B, T))
    old_logp = rng.standard_normal((B, T))
    advantages = rng.standard_normal((B, T))
    returns = rng.standard_normal((B, T))
    mask = np.ones((B, T), dtype=np.float64)
    mask[0, -1] = 0
    mask[2, -2:] = 0
    cfg = dict(clip_eps=0.2, value_coeff=0.5, entropy_coeff=-1.0, bc_coeff=0.05)

    def fwd():
        l, _, _, _ = ppo_update_grads(logits, values, actions, old_logp,
                                       advantages, returns, mask, **cfg)
        return l

    _, _, dlogits, dvalues = ppo_update_grads(
        logits, values, actions, old_logp, advantages, returns, mask, **cfg)
    num = numerical_grad(fwd, {"logits": logits, "values": values})
    _, ok1 = compare("ppo.dlogits", dlogits, num["logits"])
    _, ok2 = compare("ppo.dvalues", dvalues, num["values"])
    return ok1 and ok2


def test_actor_critic_end_to_end():
    print("\n== RecurrentActorCritic end-to-end ==")
    rng = np.random.default_rng(5)
    obs_dim, hidden, n_actions = 4, 6, 3
    B, T = 2, 4
    model = RecurrentActorCritic(obs_dim, n_actions, hidden, seed=7)
    _upcast_module(model)
    # The encoder/actor/value Linear sub-modules need explicit upcast since
    # the recursion in _upcast_module bails after one level.
    for sub in (model.encoder, model.actor, model.value):
        _upcast_module(sub)
    _upcast_module(model.gru.cell)
    obs = rng.standard_normal((B, T, obs_dim))
    h0 = np.zeros((B, hidden), dtype=np.float64)
    targets = rng.integers(0, n_actions, size=(B * T,))

    def fwd():
        logits, values, _ = model.forward_sequence(obs, h0)
        flat = logits.reshape(B * T, n_actions)
        l_ce, _ = cross_entropy(flat, targets)
        v_loss = (values * values).mean()
        return l_ce + 0.1 * v_loss

    logits, values, cache = model.forward_sequence(obs, h0)
    flat = logits.reshape(B * T, n_actions)
    _, dflat = cross_entropy(flat, targets)
    dlogits = dflat.reshape(B, T, n_actions)
    dvalues = 0.1 * 2.0 * values / values.size
    flat_grads = model.backward_sequence(dlogits, dvalues, cache)

    num = numerical_grad(fwd, model.parameters(), n_check_per_param=6)
    ok_all = True
    for name in flat_grads:
        _, ok = compare(f"AC.{name}", flat_grads[name], num[name])
        ok_all = ok_all and ok
    return ok_all


# ============================================================================
# Driver
# ============================================================================

def main():
    np.random.seed(0)
    results = []
    for name, fn in [
        ("Linear",       test_linear),
        ("GRUCell",      test_gru_cell),
        ("GRU.sequence", test_gru_sequence),
        ("cross_entropy", test_cross_entropy),
        ("ppo_grads",    test_ppo_grads),
        ("ActorCritic",  test_actor_critic_end_to_end),
    ]:
        try:
            ok = fn()
        except Exception as exc:
            print(f"\n[{name}] EXCEPTION: {exc}")
            import traceback; traceback.print_exc()
            ok = False
        results.append((name, ok))

    print("\n========= summary =========")
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
