#!/usr/bin/env python3
"""
PPO training for a CONTINUOUS shielded SysML controller (plan A).

Self-contained continuous counterpart to train.py:
  - GaussianRecurrentActorCritic policy (mean head + global log_std)
  - ContinuousShield clamps the executed action to the safe interval
  - PPO trains on the executed (shield-corrected) action
  - GAE reused from ppo.py (distribution-agnostic)

No oracle/BC warm-start phase: the shield already guarantees safe (and, in
the SysML requirement) actions, so PPO is run from scratch.

Usage:
    python rl/train_continuous.py [model.sysml] [--episodes N] ...
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sysml-models"))

from ppo import compute_gae
from continuous_env import SysMLContinuousEnv
from continuous_model import (GaussianRecurrentActorCritic,
                              build_continuous_composite)
from runtime_settings import DEFAULT_DT, validate_dt


# =====================================================================
# Continuous episode buffer
# =====================================================================

class ContinuousEpisodeBuffer:
    def __init__(self, act_dim: int):
        self.act_dim = act_dim
        self.episodes = []

    def add_episode(self, obs, actions, rewards, values, log_probs, dones):
        self.episodes.append({
            "obs": obs, "actions": actions, "rewards": rewards,
            "values": values, "log_probs": log_probs, "dones": dones,
        })

    def clear(self):
        self.episodes = []

    def build_batch(self, gamma, lam, device):
        B = len(self.episodes)
        T = max(len(e["obs"]) for e in self.episodes)
        O = len(self.episodes[0]["obs"][0])
        A = self.act_dim

        obs = np.zeros((B, T, O), np.float32)
        actions = np.zeros((B, T, A), np.float32)
        old_logp = np.zeros((B, T), np.float32)
        adv = np.zeros((B, T), np.float32)
        ret = np.zeros((B, T), np.float32)
        mask = np.zeros((B, T), np.float32)

        for i, e in enumerate(self.episodes):
            L = len(e["obs"])
            a, r = compute_gae(e["rewards"], e["values"], e["dones"], gamma, lam)
            obs[i, :L] = np.array(e["obs"], np.float32)
            actions[i, :L] = np.array(e["actions"], np.float32).reshape(L, A)
            old_logp[i, :L] = np.array(e["log_probs"], np.float32)
            adv[i, :L] = np.array(a, np.float32)
            ret[i, :L] = np.array(r, np.float32)
            mask[i, :L] = 1.0

        valid = mask.astype(bool)
        av = adv[valid]
        if len(av) > 1:
            adv[valid] = (av - av.mean()) / (av.std() + 1e-8)

        t = lambda x: torch.tensor(x, device=device)
        return {"obs": t(obs), "actions": t(actions), "old_logp": t(old_logp),
                "advantages": t(adv), "returns": t(ret), "mask": t(mask)}


# =====================================================================
# Continuous PPO update (Normal distribution)
# =====================================================================

def ppo_update(policy, optimizer, buffer, device, *,
               clip_eps=0.2, value_coeff=0.5, entropy_coeff=0.01,
               n_epochs=4, max_grad_norm=0.5, bc_aux_coeff=0.0,
               gamma=0.99, lam=0.95, target_kl=0.03):
    """Continuous PPO update with the standard stabilizers (this continuous
    path only; the discrete ppo.py is untouched):

      - KL early-stopping: stop the epoch loop once the new policy has moved
        more than 1.5*target_kl from the data-collection policy, so a single
        update can't over-commit (the cause of the post-peak collapse).
      - entropy regularization (entropy_coeff > 0): keeps std from freezing /
        the policy from drifting into a deterministic over-aggressive attractor.

    These shape the optimization so the *final* policy stays good — they are
    not checkpoint selection.
    """
    b = buffer.build_batch(gamma, lam, device)
    obs, actions = b["obs"], b["actions"]
    old_logp, adv, ret, mask = (b["old_logp"], b["advantages"],
                                b["returns"], b["mask"])
    ms = mask.sum().clamp_min(1.0)

    m = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
         "std": 0.0, "kl": 0.0, "epochs": 0.0}
    steps = 0
    for _ in range(n_epochs):
        h0 = policy.initial_hidden(obs.shape[0]).to(device)
        mean, values = policy.forward_sequence(obs, h0, mask)
        values = values.squeeze(-1)
        std = policy.std().expand_as(mean)
        dist = Normal(mean, std)

        new_logp = dist.log_prob(actions).sum(-1)   # (B, T)
        entropy = dist.entropy().sum(-1)            # (B, T)

        logratio = new_logp - old_logp
        ratio = torch.exp(logratio)

        # Schulman approx-KL (>=0), masked to real steps.
        with torch.no_grad():
            approx_kl = ((((ratio - 1) - logratio) * mask).sum() / ms).item()
        m["kl"] = approx_kl
        # Stop before applying an over-large update (always do >=1 epoch:
        # epoch 0 KL ~0 since the policy still equals the collection policy).
        if steps > 0 and approx_kl > 1.5 * target_kl:
            break

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2)
        value_loss = (values - ret).pow(2)
        bc_loss = -new_logp

        loss = (policy_loss
                + value_coeff * value_loss
                + entropy_coeff * (-entropy)
                + bc_aux_coeff * bc_loss)
        loss = (loss * mask).sum() / ms

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            m["policy_loss"] += (policy_loss * mask).sum().item() / ms.item()
            m["value_loss"] += (value_loss * mask).sum().item() / ms.item()
            m["entropy"] += (entropy * mask).sum().item() / ms.item()
            m["std"] += float(policy.std().mean().item())
        steps += 1

    denom = max(steps, 1)
    for k in ("policy_loss", "value_loss", "entropy", "std"):
        m[k] /= denom
    m["epochs"] = steps
    return m


# =====================================================================
# Rollout + eval
# =====================================================================

def collect_episode(env, composite, device, greedy=False, override_penalty=0.0):
    obs = env.reset()
    hidden = composite.initial_hidden(1).to(device)
    ep = {"obs": [], "actions": [], "rewards": [],
          "values": [], "log_probs": [], "dones": []}
    true_rewards = []
    overrides = 0
    done = False
    reward = 0.0
    info = {}
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        raw = env._twin._model_inputs
        obs_dict = {n: float(raw.get(n, 0)) for n in env._obs_keys}
        safe, z_exec, dist, value, hidden, overridden, clamp_dist = composite.act(
            obs_t, hidden, obs_dict, greedy=greedy)
        # log_prob/storage in the policy's normalized space (z_exec), not force.
        z_t = torch.tensor([[z_exec]], dtype=torch.float32, device=device)
        logp = dist.log_prob(z_t).sum(-1)
        nobs, reward, done, info = env.step(safe)
        true_rewards.append(reward)
        # Intervention penalty (training only): reward proposing actions the
        # shield accepts, so the policy keeps its own output in the safe set
        # rather than relying on the shield to mask a drifting mean.
        shaped = reward - override_penalty * clamp_dist

        ep["obs"].append(obs)
        ep["actions"].append([float(z_exec)])
        ep["rewards"].append(shaped)
        ep["values"].append(value.item())
        ep["log_probs"].append(logp.item())
        ep["dones"].append(done)
        overrides += int(overridden)
        obs = nobs

    outcome = ("SUCCESS" if reward > 0 else
               ("VIOLATION" if reward < 0 else "TRUNCATED"))
    stats = {"reward": float(sum(true_rewards)), "steps": len(ep["obs"]),
             "outcome": outcome, "overrides": overrides}
    if "statuses" in info:
        stats["violations"] = [n for n, s in info["statuses"].items()
                               if not s["status"]]
    return ep, stats


def evaluate(env, composite, device, n_episodes=30):
    composite.policy.eval()
    res = []
    for _ in range(n_episodes):
        _, stats = collect_episode(env, composite, device, greedy=True)
        res.append(stats)
    composite.policy.train()
    return res


def summarize(res, label):
    n = len(res)
    succ = sum(1 for r in res if r["outcome"] == "SUCCESS")
    viol = sum(1 for r in res if r["outcome"] == "VIOLATION")
    trunc = sum(1 for r in res if r["outcome"] == "TRUNCATED")
    avg_r = np.mean([r["reward"] for r in res])
    avg_s = np.mean([r["steps"] for r in res])
    tot_steps = sum(r["steps"] for r in res)
    tot_ovr = sum(r["overrides"] for r in res)
    ovr_pct = 100 * tot_ovr / max(tot_steps, 1)
    print(f"  [{label}] {n} eps: success={succ}/{n} ({100*succ/n:.0f}%) | "
          f"violation={viol} | truncated={trunc} | "
          f"avg_r={avg_r:+.2f} | avg_steps={avg_s:.0f} | "
          f"override={ovr_pct:.1f}%")
    if viol:
        allv = set()
        for r in res:
            allv.update(r.get("violations", []))
        if allv:
            print(f"           violations: {', '.join(sorted(allv))}")
    return succ / n


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="Continuous shielded PPO training")
    ap.add_argument("model", help="SysML file path")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--episodes-per-update", type=int, default=20)
    ap.add_argument("--eval-interval", type=int, default=200)   # = checkpoint interval
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--dt", type=validate_dt, default=DEFAULT_DT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--anneal-lr", action="store_true",
                    help="linearly decay lr from --lr to 0 over training "
                         "(standard PPO schedule). Shrinks the policy-mean "
                         "random-walk step over time so it commits in the good "
                         "basin instead of eventually escaping into creep. "
                         "Leaves sigma (exploration) untouched.")
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--init-log-std", type=float, default=-0.7)
    ap.add_argument("--final-log-std", type=float, default=-0.7,
                    help="log-std annealed linearly from --init-log-std to this "
                         "over training. DEFAULT = --init-log-std (NO anneal): "
                         "empirically, annealing exploration DOWN destabilizes "
                         "this shielded task (high exploration is the escape "
                         "from the shield's duplicate bad basin; lowering it "
                         "locks the mean into whatever basin it is in). Set "
                         "e.g. -2.3 to re-enable annealing for experimentation.")
    ap.add_argument("--action-scale", type=float, default=100.0)
    ap.add_argument("--anneal-episodes", type=int, default=0,
                    help="episodes over which log_std goes init->final, then "
                         "holds. 0 = full training length. Smaller = faster decay.")
    ap.add_argument("--entropy-coeff", type=float, default=0.01)
    ap.add_argument("--bc-aux-coeff", type=float, default=0.0)
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--override-penalty", type=float, default=0.2,
                    help="reward penalty per unit normalized shield clamp "
                         "distance (training only); keeps the policy's own "
                         "output in the safe set")
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 66)
    print("Continuous shielded PPO")
    print("=" * 66)
    print(f"  Model:  {args.model}")
    print(f"  Device: {device}")

    env = SysMLContinuousEnv(args.model, dt=args.dt, max_steps=args.max_steps,
                             phase=2, rng_seed=args.seed + 2)
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}  "
          f"out_params={env._out_names}")
    print(f"  in_params={env._obs_keys}")

    policy = GaussianRecurrentActorCritic(
        obs_dim=obs_dim, act_dim=act_dim, hidden_dim=args.hidden_dim,
        init_log_std=args.init_log_std).to(device)
    composite = build_continuous_composite(args.model, policy,
                                           action_scale=args.action_scale)
    print(f"  Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  Shield interval (eval): see override% below")
    print("=" * 66)

    # log_std is annealed on a fixed schedule (not learned), so exclude it from
    # the optimizer; it is a scheduled exploration hyperparameter, not a weight.
    opt_params = [p for n, p in policy.named_parameters() if n != "log_std"]
    optimizer = torch.optim.Adam(opt_params, lr=args.lr)
    buffer = ContinuousEpisodeBuffer(act_dim)
    total_updates = max(args.episodes // args.episodes_per_update, 1)
    anneal_eps = args.anneal_episodes if args.anneal_episodes > 0 else args.episodes
    anneal_updates = max(anneal_eps // args.episodes_per_update, 1)

    def scheduled_log_std(ep):
        upd = (ep - 1) // args.episodes_per_update
        frac = min(1.0, upd / max(anneal_updates - 1, 1))
        return args.init_log_std + frac * (args.final_log_std - args.init_log_std)

    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    checkpoints = []                 # one entry per eval checkpoint, for selection
    eval_seed = args.seed + 777      # fixed held-out eval set, identical per checkpoint

    all_stats = []
    t0 = time.time()
    init_std = float(np.exp(args.init_log_std))
    for ep in range(1, args.episodes + 1):
        # Anneal exploration: set the (scheduled) std for this batch so the
        # collection policy and its update share the same std.
        sched = scheduled_log_std(ep)
        policy.log_std.data.fill_(sched)
        # Compensate the learning rate: the Gaussian mean-gradient scales as
        # 1/sigma^2, so as std anneals down the effective step grows and
        # overshoots. Scaling lr by (sigma/sigma_init)^2 keeps the trust-region
        # step constant and makes low-std = genuine commitment (tiny updates).
        cur_std = float(np.exp(sched))
        # Linear lr anneal to 0 (standard PPO), composed with the sigma^2
        # gradient compensation (the latter is a no-op when sigma is fixed).
        lr_frac = (1.0 - (ep - 1) / max(args.episodes - 1, 1)) if args.anneal_lr else 1.0
        lr_now = args.lr * lr_frac * (cur_std / init_std) ** 2
        for g in optimizer.param_groups:
            g["lr"] = lr_now
        ep_data, stats = collect_episode(env, composite, device, greedy=False,
                                         override_penalty=args.override_penalty)
        buffer.add_episode(**ep_data)
        all_stats.append(stats)

        if ep % args.episodes_per_update == 0:
            m = ppo_update(policy, optimizer, buffer, device,
                           entropy_coeff=args.entropy_coeff,
                           bc_aux_coeff=args.bc_aux_coeff,
                           target_kl=args.target_kl)
            buffer.clear()
            recent = all_stats[-args.episodes_per_update:]
            avg_r = float(np.mean([s["reward"] for s in recent]))
            avg_s = float(np.mean([s["steps"] for s in recent]))
            succ = float(np.mean([s["outcome"] == "SUCCESS" for s in recent]))
            tot_steps = sum(s["steps"] for s in recent)
            ovr = 100 * sum(s["overrides"] for s in recent) / max(tot_steps, 1)
            print(f"  ep {ep:5d}/{args.episodes} | avg_r={avg_r:+.3f} | "
                  f"steps={avg_s:5.0f} | success={succ:.0%} | "
                  f"override={ovr:4.1f}% | std={m['std']:.2f} | lr={lr_now:.1e} | "
                  f"kl={m['kl']:.3f}({m['epochs']:.0f}ep) | "
                  f"v_loss={m['value_loss']:.3f} | {time.time()-t0:.0f}s")

        if ep % args.eval_interval == 0:
            eval_env = SysMLContinuousEnv(
                args.model, dt=args.dt, max_steps=args.max_steps,
                phase=2, rng_seed=eval_seed)   # same held-out set every checkpoint
            try:
                res = evaluate(eval_env, composite, device,
                               n_episodes=args.eval_episodes)
            finally:
                eval_env.close()
            n = len(res)
            n_viol = sum(1 for r in res if r["outcome"] == "VIOLATION")
            n_succ = sum(1 for r in res if r["outcome"] == "SUCCESS")
            acc = n_succ / n
            tot_steps = sum(r["steps"] for r in res)
            ovr = sum(r["overrides"] for r in res) / max(tot_steps, 1)
            sd = {k: v.detach().cpu().clone()
                  for k, v in policy.state_dict().items()}
            checkpoints.append({"ep": ep, "sd": sd, "viol": n_viol,
                                "acc": acc, "ovr": ovr})
            print(f"  [ckpt ep {ep}] {n} eps: safety_violations={n_viol} | "
                  f"accuracy={acc:.0%} | override={ovr*100:.1f}%")
            if save_dir:
                torch.save(policy.state_dict(),
                           os.path.join(save_dir, f"ep_{ep:05d}.pt"))

    env.close()

    # ---- Checkpoint selection: lexicographic safety > accuracy > override ----
    # Never select an unsafe model. Among safe (0 safety-violation) checkpoints,
    # pick the most accurate; break ties by lowest shield-override.
    print("\n" + "=" * 66)
    print(f"CHECKPOINT SELECTION  ({args.eval_episodes}-ep held-out eval;  "
          f"safety > accuracy > override)")
    print("=" * 66)
    print(f"  {'episode':>7} | {'safety_viol':>11} | {'accuracy':>8} | {'override':>9}")
    for c in checkpoints:
        print(f"  {c['ep']:>7} | {c['viol']:>11} | {c['acc']:>7.0%}  | "
              f"{c['ovr']*100:>7.1f}%")

    safe = [c for c in checkpoints if c["viol"] == 0]
    if not safe:
        print("\n" + "!" * 66)
        print("!!  UNSAFE TRAINING: no checkpoint reached zero safety violations.")
        print("!!  NO MODEL SELECTED.")
        print("!" * 66)
        return

    best = sorted(safe, key=lambda c: (-c["acc"], c["ovr"]))[0]
    print(f"\n  SELECTED: ep {best['ep']}  |  safety_violations=0  |  "
          f"accuracy={best['acc']:.0%}  |  override={best['ovr']*100:.1f}%")
    if save_dir:
        policy.load_state_dict({k: v.to(device) for k, v in best["sd"].items()})
        torch.save(policy.state_dict(), os.path.join(save_dir, "best.pt"))
        print(f"  Saved selected model -> {os.path.join(save_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
