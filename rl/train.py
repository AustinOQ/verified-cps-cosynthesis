#!/usr/bin/env python3
"""
Full-shield-only training for SysML-extracted CPS controllers.

Runs oracle cloning (Phase 1) then PPO (Phase 2) with the full
#NeuralRequirement shield. No fallback to prohibition-only or no-shield.

Usage:
    python rl/train.py [path/to/model.sysml] [--csv-dir DIR]
"""

import argparse
import os
import random
import sys
import time
import torch
import torch.nn as nn
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from model import RecurrentActorCritic
from env import SysMLEnv
from ppo import RecurrentPPO, EpisodeBuffer
from oracle import extract_interface, brute_force_oracle
from composite_model import (build_full_shield_model,
                             build_no_shield_model)


# =====================================================================
# Per-mode sample size configuration
# =====================================================================

MODE_CONFIG = {
    "full": {
        "oracle_samples": 5_000,
        "oracle_epochs": 30,
        "ppo_episodes": 5_000,
    },
    "prohibition": {
        "oracle_samples": 20_000,
        "oracle_epochs": 30,
        "ppo_episodes": 20_000,
    },
    "none": {
        "oracle_samples": 200_000,
        "oracle_epochs": 30,
        "ppo_episodes": 20_000,
    },
}

# Pass criteria
PASS_SUCCESS_RATE = 0.95    # ≥95% success
PASS_SAFETY_VIOLATIONS = 0  # 0 safety violations


# =====================================================================
# Oracle data generation
# =====================================================================

def generate_oracle_data(iface, env, n_samples, rng, max_steps=1000):
    sync_engine = iface["sync_engine"]
    obs_names = iface["obs_names"]
    action_names = iface["action_names"]
    goal_distance = iface["goal_distance"]
    is_done_fn = iface["is_done"]
    prop_delay = iface["propagation_delay"]

    obs_all, act_all = [], []
    while len(obs_all) < n_samples:
        norm_obs = env.reset()
        for step in range(max_steps):
            label = brute_force_oracle(
                sync_engine, env._twin._engine,
                obs_names, action_names,
                goal_distance, is_done_fn,
                n_steps=prop_delay)
            discrete = _oracle_to_discrete(label)
            obs_all.append(norm_obs.copy())
            act_all.append(discrete)
            if len(obs_all) >= n_samples:
                break
            norm_obs, reward, done, info = env.step(discrete)
            if done:
                break
        if len(obs_all) % 5000 < max_steps:
            print(f"    ... {len(obs_all)}/{n_samples} samples collected")

    return (np.array(obs_all[:n_samples], dtype=np.float32),
            np.array(act_all[:n_samples], dtype=np.int64))


def _oracle_to_discrete(label):
    action_id = 0
    for i, val in enumerate(label):
        if bool(val):
            action_id |= (1 << i)
    return action_id


# =====================================================================
# Phase 1: Oracle training
# =====================================================================

def train_oracle(composite, env, iface, device,
                 n_samples=20_000, batch_size=1024, n_epochs=30, lr=1e-3):
    rng = np.random.default_rng(42)
    print(f"  Generating oracle training data (brute-force)...")
    obs_data, action_data = generate_oracle_data(iface, env, n_samples, rng)

    unique, counts = np.unique(action_data, return_counts=True)
    print(f"  Oracle data: {n_samples} samples, "
          f"class dist: {dict(zip(unique.tolist(), counts.tolist()))}")

    obs_t = torch.tensor(obs_data, device=device)
    action_t = torch.tensor(action_data, device=device)

    optimizer = torch.optim.Adam(composite.policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    composite.policy.train()
    for epoch in range(1, n_epochs + 1):
        perm = torch.randperm(n_samples, device=device)
        obs_shuffled = obs_t[perm]
        act_shuffled = action_t[perm]
        total_loss = 0.0
        correct = 0
        n_batches = 0
        for i in range(0, n_samples, batch_size):
            batch_obs = obs_shuffled[i:i + batch_size]
            batch_act = act_shuffled[i:i + batch_size]
            bs = batch_obs.shape[0]
            h0 = composite.initial_hidden(bs).to(device)
            dist, _, _ = composite.forward_policy(batch_obs, h0)
            loss = criterion(dist.logits, batch_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (dist.logits.argmax(dim=-1) == batch_act).sum().item()
            n_batches += 1
        print(f"  [Oracle] epoch {epoch:2d}/{n_epochs} | "
              f"loss={total_loss/n_batches:.4f} | acc={correct/n_samples:.1%}")


# =====================================================================
# Evaluation
# =====================================================================

def evaluate(env, composite, device, n_episodes=50):
    composite.policy.eval()
    results = []
    for _ in range(n_episodes):
        obs = env.reset()
        hidden = composite.initial_hidden(1).to(device)
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            raw_obs = env._twin._model_inputs
            final_action, dist, value, hidden, overridden = composite.act(
                obs_t, hidden, raw_obs, greedy=True)
            obs, reward, done, info = env.step(final_action)
            total_reward += reward
            steps += 1
        statuses = info.get("statuses", {})
        failed = [n for n, s in statuses.items() if not s["status"]]
        results.append({
            "reward": total_reward, "steps": steps,
            "outcome": "SUCCESS" if reward > 0 else (
                "VIOLATION" if reward < 0 else "TRUNCATED"),
            "violations": failed,
        })
    composite.policy.train()
    return results


def print_eval(results, label):
    n = len(results)
    succ = sum(1 for r in results if r["outcome"] == "SUCCESS")
    viol = sum(1 for r in results if r["outcome"] == "VIOLATION")
    trunc = sum(1 for r in results if r["outcome"] == "TRUNCATED")
    avg_steps = np.mean([r["steps"] for r in results])
    avg_reward = np.mean([r["reward"] for r in results])
    print(f"  [{label}] Eval {n} eps: "
          f"success={succ}/{n} ({100*succ/n:.0f}%) | "
          f"violation={viol}/{n} | truncated={trunc}/{n} | "
          f"avg_r={avg_reward:+.2f} | avg_steps={avg_steps:.0f}")
    if viol > 0:
        all_v = set()
        for r in results:
            all_v.update(r.get("violations", []))
        if all_v:
            print(f"           violations: {', '.join(sorted(all_v))}")


def detailed_eval(model_path, composite, device, dt=0.1, max_steps=1200,
                  n_episodes=100):
    from oracle import extract_interface
    eval_env = SysMLEnv(model_path, dt=dt, max_steps=max_steps, phase=2)
    iface = extract_interface(model_path, dt=dt)
    goal_distance = iface["goal_distance"]
    obs_names = iface["obs_names"]
    SAFETY_NAMES = {"No Dry Running", "No Dead Heading"}

    composite.policy.eval()
    all_steps, all_goal_dist = [], []
    safety_violation_eps = 0
    accuracy_violation_eps = 0
    completed_eps = 0
    truncated_eps = 0
    per_req_violations = {}
    total_overrides = 0
    total_steps = 0

    for ep in range(n_episodes):
        obs = eval_env.reset()
        hidden = composite.initial_hidden(1).to(device)
        done = False
        steps = 0
        ep_safety, ep_accuracy = set(), set()
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            raw_obs = eval_env._twin._model_inputs
            final_action, dist, value, hidden, overridden = composite.act(
                obs_t, hidden, raw_obs, greedy=True)
            obs, reward, done, info = eval_env.step(final_action)
            steps += 1
            total_steps += 1
            if overridden:
                total_overrides += 1
            statuses = eval_env._twin._engine.requirement_statuses()
            for name, entry in statuses.items():
                if not entry["status"]:
                    if name in SAFETY_NAMES:
                        ep_safety.add(name)
                    else:
                        ep_accuracy.add(name)

        state = eval_env._twin._engine.state
        obs_dict = {}
        for short_name, qual_name in zip(obs_names, eval_env._obs_keys):
            obs_dict[short_name] = float(state.get(qual_name, 0))
        gdist = goal_distance(obs_dict)

        all_steps.append(steps)
        all_goal_dist.append(gdist)
        if ep_safety: safety_violation_eps += 1
        if ep_accuracy: accuracy_violation_eps += 1
        if reward > 0: completed_eps += 1
        elif reward == 0: truncated_eps += 1
        for name in ep_safety | ep_accuracy:
            per_req_violations[name] = per_req_violations.get(name, 0) + 1

    eval_env.close()
    composite.policy.train()

    print(f"\n  Episodes: {n_episodes}")
    print(f"  Completed (done=True): {completed_eps}/{n_episodes} "
          f"({100*completed_eps/n_episodes:.0f}%)")
    print(f"  Truncated (max steps): {truncated_eps}/{n_episodes}")
    print()
    print(f"  SAFETY violations: {safety_violation_eps}/{n_episodes} episodes")
    print(f"  ACCURACY violations: {accuracy_violation_eps}/{n_episodes} episodes")
    print()
    print(f"  Shield overrides: {total_overrides}/{total_steps} steps "
          f"({100*total_overrides/max(total_steps,1):.1f}%)")
    print()
    if per_req_violations:
        print("  Per-requirement violation counts:")
        for name, count in sorted(per_req_violations.items()):
            kind = "SAFETY" if name in SAFETY_NAMES else "ACCURACY"
            print(f"    {name}: {count}/{n_episodes} ({kind})")
        print()
    print(f"  Avg steps: {np.mean(all_steps):.0f} (std={np.std(all_steps):.0f})")
    print(f"  Goal distance: mean={np.mean(all_goal_dist):.2f}, "
          f"std={np.std(all_goal_dist):.2f}, max={np.max(all_goal_dist):.2f}")
    within_5 = sum(1 for d in all_goal_dist if d <= 5)
    within_10 = sum(1 for d in all_goal_dist if d <= 10)
    print(f"  Within 5: {within_5}/{n_episodes} ({100*within_5/n_episodes:.0f}%)")
    print(f"  Within 10: {within_10}/{n_episodes} ({100*within_10/n_episodes:.0f}%)")

    return {
        "success_rate": completed_eps / n_episodes,
        "safety_violations": safety_violation_eps,
        "accuracy_violations": accuracy_violation_eps,
    }


# =====================================================================
# Phase 2: PPO
# =====================================================================

def collect_episode(env, composite, device):
    obs = env.reset()
    hidden = composite.initial_hidden(1).to(device)
    ep_obs, ep_actions, ep_rewards = [], [], []
    ep_values, ep_log_probs, ep_dones = [], [], []
    done = False
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        raw_obs = env._twin._model_inputs
        final_action, dist, value, hidden, overridden = composite.act(
            obs_t, hidden, raw_obs, greedy=False)
        safe_action_t = torch.tensor([final_action], device=device)
        log_prob = dist.log_prob(safe_action_t)
        next_obs, reward, done, info = env.step(final_action)
        ep_obs.append(obs)
        ep_actions.append(final_action)
        ep_rewards.append(reward)
        ep_values.append(value.item())
        ep_log_probs.append(log_prob.item())
        ep_dones.append(done)
        obs = next_obs

    outcome = "SUCCESS" if reward > 0 else ("VIOLATION" if reward < 0 else "TRUNCATED")
    stats = {"reward": sum(ep_rewards), "steps": len(ep_obs), "outcome": outcome}
    if "statuses" in info:
        stats["violations"] = [n for n, s in info["statuses"].items()
                               if not s["status"]]
    return {
        "obs": ep_obs, "actions": ep_actions, "rewards": ep_rewards,
        "values": ep_values, "log_probs": ep_log_probs, "dones": ep_dones,
    }, stats


def train_phase(env, composite, ppo, n_episodes, episodes_per_update,
                eval_interval, phase_name, device, save_dir=None):
    buffer = EpisodeBuffer()
    all_stats = []
    best_success = -1
    t_start = time.time()

    for ep in range(1, n_episodes + 1):
        ep_data, stats = collect_episode(env, composite, device)
        buffer.add_episode(**ep_data)
        all_stats.append(stats)

        if ep % episodes_per_update == 0:
            metrics = ppo.update(buffer)
            buffer.clear()
            recent = all_stats[-episodes_per_update:]
            avg_reward = np.mean([s["reward"] for s in recent])
            avg_steps = np.mean([s["steps"] for s in recent])
            succ_rate = np.mean([s["outcome"] == "SUCCESS" for s in recent])
            elapsed = time.time() - t_start
            print(f"  [{phase_name}] ep {ep:5d}/{n_episodes} | "
                  f"avg_r={avg_reward:+.3f} | avg_steps={avg_steps:5.0f} | "
                  f"success={succ_rate:.0%} | "
                  f"ent={metrics['entropy']:.3f} | "
                  f"p_loss={metrics['policy_loss']:.4f} | "
                  f"{elapsed:.0f}s")

        if ep % eval_interval == 0:
            eval_env = SysMLEnv(env._model_path, phase=2,
                                max_steps=env._max_steps, rng_seed=ep)
            try:
                results = evaluate(eval_env, composite, device, n_episodes=100)
                print_eval(results, f"{phase_name} ep {ep}")
                succ = sum(1 for r in results if r["outcome"] == "SUCCESS")
                if save_dir and succ > best_success:
                    best_success = succ
                    torch.save(composite.policy.state_dict(),
                               os.path.join(save_dir, "best.pt"))
                    print(f"    >> New best ({succ}/{len(results)})")
            finally:
                eval_env.close()

    total = n_episodes
    succ = sum(1 for s in all_stats if s["outcome"] == "SUCCESS")
    viol = sum(1 for s in all_stats if s["outcome"] == "VIOLATION")
    trunc = sum(1 for s in all_stats if s["outcome"] == "TRUNCATED")
    print(f"\n  {phase_name} done: {succ}/{total} success, "
          f"{viol}/{total} violation, {trunc}/{total} truncated\n")
    return all_stats


# =====================================================================
# Check pass criteria
# =====================================================================

def check_pass(results_100):
    """Check if detailed eval passes: 0 safety violations, ≥95% success."""
    sr = results_100["success_rate"]
    sv = results_100["safety_violations"]
    passed = (sv == PASS_SAFETY_VIOLATIONS and sr >= PASS_SUCCESS_RATE)
    return passed, sr, sv


# =====================================================================
# Run one training mode
# =====================================================================

def run_training_mode(mode_name, build_fn, model_path, iface,
                      obs_dim, n_actions, device, save_dir, config):
    """Run full oracle + PPO training for one shield mode.

    Returns (composite, detailed_results, passed).
    """
    DT = 0.1
    MAX_STEPS = 1200
    SEED = 42
    HIDDEN_DIM = 64
    ORACLE_BATCH_SIZE = 1024
    ORACLE_LR = 1e-3
    EPISODES_PER_UPDATE = 80
    EVAL_INTERVAL = 500
    RL_LR = 3e-4
    ENTROPY_START = 0.01
    ENTROPY_END = 0.001

    oracle_samples = config["oracle_samples"]
    oracle_epochs = config["oracle_epochs"]
    ppo_episodes = config["ppo_episodes"]

    # Reset seeds for reproducibility within each mode
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Build fresh policy + shield
    policy = RecurrentActorCritic(
        obs_dim=obs_dim, n_actions=n_actions, hidden_dim=HIDDEN_DIM).to(device)
    composite = build_fn(model_path, policy).to(device)

    mode_save = os.path.join(save_dir, mode_name)
    os.makedirs(mode_save, exist_ok=True)

    print(f"  Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  Oracle: {oracle_samples} samples, {oracle_epochs} epochs")
    print(f"  PPO: {ppo_episodes} eps")
    print()

    # Phase 1: Oracle
    print(f"  Phase 1: Oracle cloning...")
    probe_env = SysMLEnv(model_path, dt=DT, max_steps=MAX_STEPS,
                         phase=1, rng_seed=SEED)
    train_oracle(composite, probe_env, iface, device,
                 n_samples=oracle_samples, batch_size=ORACLE_BATCH_SIZE,
                 n_epochs=oracle_epochs, lr=ORACLE_LR)

    print(f"\n  Post-oracle evaluation...")
    eval_env = SysMLEnv(model_path, dt=DT, max_steps=MAX_STEPS,
                        phase=2, rng_seed=SEED + 1)
    try:
        results = evaluate(eval_env, composite, device, n_episodes=100)
        print_eval(results, f"{mode_name}/Post-Oracle")
    finally:
        eval_env.close()

    torch.save(composite.policy.state_dict(),
               os.path.join(mode_save, "oracle.pt"))
    probe_env.close()

    # Phase 2: PPO
    print(f"\n  Phase 2: PPO ({ppo_episodes} episodes)...")
    total_updates = ppo_episodes // EPISODES_PER_UPDATE
    ppo = RecurrentPPO(composite.policy, lr=RL_LR, device=str(device),
                       entropy_coeff=ENTROPY_START,
                       entropy_coeff_end=ENTROPY_END,
                       total_updates=total_updates)
    env = SysMLEnv(model_path, dt=DT, max_steps=MAX_STEPS,
                   phase=2, rng_seed=SEED + 2)
    try:
        train_phase(env, composite, ppo,
                    n_episodes=ppo_episodes,
                    episodes_per_update=EPISODES_PER_UPDATE,
                    eval_interval=EVAL_INTERVAL,
                    phase_name=f"{mode_name}/P2", device=device,
                    save_dir=mode_save)
    finally:
        env.close()

    torch.save(composite.policy.state_dict(),
               os.path.join(mode_save, "final.pt"))

    # Load best checkpoint for final eval
    best_path = os.path.join(mode_save, "best.pt")
    if os.path.exists(best_path):
        composite.policy.load_state_dict(
            torch.load(best_path, map_location=device))
        print(f"  Loaded best.pt for final evaluation")

    # Final quick eval
    print(f"\n  Final evaluation (100 episodes)...")
    eval_env = SysMLEnv(model_path, dt=DT, max_steps=MAX_STEPS, phase=2)
    try:
        results = evaluate(eval_env, composite, device, n_episodes=100)
        print_eval(results, f"{mode_name}/FINAL")
    finally:
        eval_env.close()

    # Detailed eval (100 episodes) — the one that determines pass/fail
    print(f"\n  Detailed evaluation (100 episodes)...")
    detail = detailed_eval(model_path, composite, device,
                           dt=DT, max_steps=MAX_STEPS, n_episodes=100)

    passed, sr, sv = check_pass(detail)
    print(f"\n  >>> {mode_name} result: success={sr:.0%}, "
          f"safety_violations={sv}, "
          f"{'PASS' if passed else 'FAIL'}")

    return composite, detail, passed


# =====================================================================
# Main: three-way sequential training
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full-shield training for SysML CPS controllers")
    parser.add_argument("model", nargs="?",
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "sysml-models",
                                             "mixing-sysml-model", "model.sysml"))
    parser.add_argument("--csv-dir", default=None,
                        help="Directory for training metrics CSV (used by pipeline)")
    args = parser.parse_args()

    MODEL_PATH = args.model
    SAVE_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(SAVE_DIR, exist_ok=True)

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Probe env for dimensions
    probe_env = SysMLEnv(MODEL_PATH, dt=0.1, max_steps=1200, phase=1, rng_seed=SEED)
    obs_dim = probe_env.obs_dim
    n_actions = probe_env.n_actions
    probe_env.close()

    print("Extracting SysML interface for oracle...")
    iface = extract_interface(MODEL_PATH, dt=0.1)

    print()
    print("=" * 70)
    print("Full Shield Training")
    print("=" * 70)
    print(f"  Model: {MODEL_PATH}")
    print(f"  Obs: {iface['obs_names']}")
    print(f"  Actions: {iface['action_names']}")
    print(f"  Propagation delay: {iface['propagation_delay']} steps")
    print(f"  Pass criteria: ≥{PASS_SUCCESS_RATE:.0%} success, "
          f"{PASS_SAFETY_VIOLATIONS} safety violations")
    print(f"  Device: {device}")
    print(f"  Sequence: full shield only")
    print("=" * 70)

    # === Full Shield Only ===
    print()
    print("╔" + "═" * 60 + "╗")
    print("║  FULL SPECIFICATION SHIELD                              ║")
    print("╚" + "═" * 60 + "╝")
    print()

    composite, detail, passed = run_training_mode(
        "full", build_full_shield_model, MODEL_PATH, iface,
        obs_dim, n_actions, device, SAVE_DIR, MODE_CONFIG["full"])

    if passed:
        print("\n  ★ FULL SHIELD PASSED — training complete.")
    else:
        print("\n  ✗ Full shield did not pass. Using best available result.")

    best_src = os.path.join(SAVE_DIR, "full", "best.pt")
    if os.path.exists(best_src):
        torch.save(composite.policy.state_dict(),
                   os.path.join(SAVE_DIR, "best.pt"))
    print(f"\nDone. Best model: {SAVE_DIR}/full/best.pt")


if __name__ == "__main__":
    main()
