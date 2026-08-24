#!/usr/bin/env python3
"""
Train a non-recurrent MLP with the observation/action buffer read from a spec.

Reuses the existing continuous PPO machinery verbatim (ppo_update,
collect_episode, evaluate, ContinuousEpisodeBuffer, the shield, the
lexicographic safety>accuracy>override selection) from rl/train_continuous.py.
Only the policy class (GaussianMLPActorCritic) and the env (BufferedContinuousEnv)
are swapped in. Same methodology/args as the GRU run for a fair comparison.
"""
import os, sys, argparse, random, time, json
from pathlib import Path
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_RL = os.path.join(_HERE, "..", "rl")
_MODELS = os.path.join(_HERE, "..", "sysml-models")
sys.path.insert(0, _RL)
sys.path.insert(0, _HERE)
sys.path.insert(0, _MODELS)

from train_continuous import (ContinuousEpisodeBuffer, ppo_update,
                              collect_episode, evaluate)
from continuous_model import build_continuous_composite
from mlp_buffer import GaussianMLPActorCritic, BufferedContinuousEnv
from certification.reduced_mdp_spec import (
    check_reduced_mdp_spec,
    load_reduced_mdp_spec,
)
from runtime_settings import DEFAULT_DT, validate_dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--episodes-per-update", type=int, default=20)
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--dt", type=validate_dt, default=DEFAULT_DT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--anneal-lr", action="store_true")
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--init-log-std", type=float, default=-0.7)
    ap.add_argument("--action-scale", type=float, default=None)
    ap.add_argument("--override-penalty", type=float, default=0.2)
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--entropy-coeff", type=float, default=0.01)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--summary-json", default=None)
    ap.add_argument(
        "--reduced-mdp-spec",
        required=True,
        help="Use a reduced-MDP architecture spec generated earlier in this run.",
    )
    args = ap.parse_args()

    reduced_spec = load_reduced_mdp_spec(args.reduced_mdp_spec)
    spec_errors = check_reduced_mdp_spec(reduced_spec)
    if spec_errors:
        raise RuntimeError(
            "reduced-MDP spec checker failed:\n"
            + "\n".join(f"  - {err}" for err in spec_errors)
        )
    if os.path.abspath(reduced_spec["model"]["path"]) != os.path.abspath(args.model):
        raise RuntimeError("reduced-MDP spec is for a different model")
    spec_dt = validate_dt(reduced_spec["model"].get("dt"))
    if args.dt != spec_dt:
        raise RuntimeError(
            f"training dt does not match reduced-MDP spec: {args.dt} != {spec_dt}"
        )
    action_space = reduced_spec["action_space"]
    if action_space["type"] != "continuous_single_real_output":
        raise RuntimeError("continuous trainer received a non-continuous reduced-MDP spec")
    args.n_act = int(reduced_spec["certified_buffer"]["b_act"])
    args.n_obs = int(reduced_spec["certified_buffer"]["b_obs"])
    derived_architecture = reduced_spec.get("feedforward_architecture")
    if not isinstance(derived_architecture, dict):
        raise RuntimeError("reduced-MDP spec has no feedforward architecture")
    derived_hidden_dim = int(derived_architecture["hidden_dim"])
    if args.hidden_dim is not None and args.hidden_dim != derived_hidden_dim:
        raise RuntimeError(
            "hidden size does not match the architecture extracted from SysML"
        )
    args.hidden_dim = derived_hidden_dim
    extracted_action_scale = float(action_space["action_scale"])
    if args.action_scale is not None and args.action_scale != extracted_action_scale:
        raise RuntimeError("action scale does not match the reduced-MDP spec")
    args.action_scale = extracted_action_scale

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cpu")

    def make_env(seed):
        return BufferedContinuousEnv(args.model, dt=args.dt,
                                     max_steps=args.max_steps, phase=2,
                                     rng_seed=seed, n_act=args.n_act,
                                     n_obs=args.n_obs, action_scale=args.action_scale)

    print("=" * 66)
    print("NON-RECURRENT MLP + BUFFER")
    print("=" * 66)
    env = make_env(args.seed + 2)
    obs_dim, act_dim = env.obs_dim, env.act_dim
    print(f"  base obs={env._base_obs_dim}  buffer: last {args.n_act} actions + "
          f"{args.n_obs} past obs  ->  augmented obs_dim={obs_dim}")
    if reduced_spec is not None:
        expected_obs_dim = int(reduced_spec["policy_input"]["input_dim"])
        expected_act_dim = int(reduced_spec["action_space"]["act_dim"])
        if obs_dim != expected_obs_dim:
            raise RuntimeError(
                f"buffered obs dim mismatch: env={obs_dim}, spec={expected_obs_dim}"
            )
        if act_dim != expected_act_dim:
            raise RuntimeError(
                f"action dim mismatch: env={act_dim}, spec={expected_act_dim}"
            )

    policy = GaussianMLPActorCritic(obs_dim=obs_dim, act_dim=act_dim,
                                    hidden_dim=args.hidden_dim,
                                    init_log_std=args.init_log_std).to(device)
    composite = build_continuous_composite(args.model, policy,
                                           action_scale=args.action_scale)
    npar = sum(p.numel() for p in policy.parameters())
    if npar != int(derived_architecture["parameter_count"]):
        raise RuntimeError(
            "derived feedforward parameter count does not match runtime policy"
        )
    print(f"  Policy params: {npar:,}  (GRU baseline was 29,571)")
    print("=" * 66)

    opt_params = [p for n, p in policy.named_parameters() if n != "log_std"]
    optimizer = torch.optim.Adam(opt_params, lr=args.lr)
    buffer = ContinuousEpisodeBuffer(act_dim)

    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    checkpoints = []
    eval_seed = args.seed + 777
    all_stats = []
    t0 = time.time()

    for ep in range(1, args.episodes + 1):
        policy.log_std.data.fill_(args.init_log_std)
        lr_frac = (1.0 - (ep - 1) / max(args.episodes - 1, 1)) if args.anneal_lr else 1.0
        for g in optimizer.param_groups:
            g["lr"] = args.lr * lr_frac
        ep_data, stats = collect_episode(env, composite, device, greedy=False,
                                         override_penalty=args.override_penalty)
        buffer.add_episode(**ep_data)
        all_stats.append(stats)

        if ep % args.episodes_per_update == 0:
            m = ppo_update(policy, optimizer, buffer, device,
                           entropy_coeff=args.entropy_coeff,
                           target_kl=args.target_kl)
            buffer.clear()
            recent = all_stats[-args.episodes_per_update:]
            avg_r = float(np.mean([s["reward"] for s in recent]))
            avg_s = float(np.mean([s["steps"] for s in recent]))
            succ = float(np.mean([s["outcome"] == "SUCCESS" for s in recent]))
            tot = sum(s["steps"] for s in recent)
            ovr = 100 * sum(s["overrides"] for s in recent) / max(tot, 1)
            print(f"  ep {ep:5d}/{args.episodes} | avg_r={avg_r:+.3f} | "
                  f"steps={avg_s:5.0f} | success={succ:.0%} | override={ovr:4.1f}% | "
                  f"std={m['std']:.2f} | lr={args.lr*lr_frac:.1e} | "
                  f"kl={m['kl']:.3f}({m['epochs']:.0f}ep) | {time.time()-t0:.0f}s")

        if ep % args.eval_interval == 0:
            eval_env = make_env(eval_seed)
            try:
                res = evaluate(eval_env, composite, device,
                               n_episodes=args.eval_episodes)
            finally:
                eval_env.close()
            n = len(res)
            n_viol = sum(1 for r in res if r["outcome"] == "VIOLATION")
            n_succ = sum(1 for r in res if r["outcome"] == "SUCCESS")
            acc = n_succ / n
            tot = sum(r["steps"] for r in res)
            ovr = sum(r["overrides"] for r in res) / max(tot, 1)
            sd = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            checkpoints.append({"ep": ep, "sd": sd, "viol": n_viol,
                                "acc": acc, "ovr": ovr})
            print(f"  [ckpt ep {ep}] {n} eps: safety_violations={n_viol} | "
                  f"accuracy={acc:.0%} | override={ovr*100:.1f}%")
            if save_dir:
                torch.save(policy.state_dict(),
                           os.path.join(save_dir, f"ep_{ep:05d}.pt"))
    env.close()

    print("\n" + "=" * 66)
    print(f"CHECKPOINT SELECTION ({args.eval_episodes}-ep held-out; safety>accuracy>override)")
    print("=" * 66)
    print(f"  {'episode':>7} | {'safety_viol':>11} | {'accuracy':>8} | {'override':>9}")
    for c in checkpoints:
        print(f"  {c['ep']:>7} | {c['viol']:>11} | {c['acc']:>7.0%}  | {c['ovr']*100:>7.1f}%")
    safe = [c for c in checkpoints if c["viol"] == 0]
    if not safe:
        print("\n!!  UNSAFE TRAINING: no checkpoint reached zero safety violations. NO MODEL SELECTED.")
        return
    best = sorted(safe, key=lambda c: (-c["acc"], c["ovr"]))[0]
    print(f"\n  SELECTED: ep {best['ep']}  |  safety_violations=0  |  "
          f"accuracy={best['acc']:.0%}  |  override={best['ovr']*100:.1f}%")
    if save_dir:
        policy.load_state_dict({k: v.to(device) for k, v in best["sd"].items()})
        torch.save(policy.state_dict(), os.path.join(save_dir, "best.pt"))
        print(f"  Saved -> {os.path.join(save_dir, 'best.pt')}")

    if args.summary_json:
        test_env = make_env(args.seed + 20_000)
        try:
            test_res = evaluate(test_env, composite, device,
                                n_episodes=args.eval_episodes)
        finally:
            test_env.close()
        n = len(test_res)
        n_viol = sum(1 for r in test_res if r["outcome"] == "VIOLATION")
        n_succ = sum(1 for r in test_res if r["outcome"] == "SUCCESS")
        tot_steps = sum(r["steps"] for r in test_res)
        tot_overrides = sum(r["overrides"] for r in test_res)
        summary = {
            "model_path": os.path.abspath(args.model),
            "dt": args.dt,
            "seed": args.seed,
            "training_stack": "torch_continuous_reduced_mdp_mlp",
            "device": "cpu",
            "shield_type": "program_ast_continuous_interval_shield",
            "policy_type": "memoryless_gaussian_mlp",
            "hidden_dim": args.hidden_dim,
            "parameter_count": int(npar),
            "reduced_mdp_spec_path": (
                None if args.reduced_mdp_spec is None
                else os.path.abspath(args.reduced_mdp_spec)
            ),
            "certified_buffer": {"b_obs": args.n_obs, "b_act": args.n_act},
            "obs_dim": int(obs_dim),
            "act_dim": int(act_dim),
            "selected_checkpoint": {
                "episode": int(best["ep"]),
                "safety_violation_rate": float(best["viol"] / max(args.eval_episodes, 1)),
                "success_rate": float(best["acc"]),
                "override_rate": float(best["ovr"]),
            },
            "test": {
                "episodes": n,
                "safety_violation_rate": float(n_viol / max(n, 1)),
                "success_rate": float(n_succ / max(n, 1)),
                "pooled_override_rate": float(tot_overrides / max(tot_steps, 1)),
                "mean_episode_steps": float(tot_steps / max(n, 1)),
            },
            "config": vars(args),
        }
        out = Path(args.summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(f"WROTE {out}")


if __name__ == "__main__":
    main()
