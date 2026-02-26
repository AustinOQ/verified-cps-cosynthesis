#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:
    yaml = None


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required (pip install PyYAML).")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping (key: value).")
    return data


@dataclass
class SimConfig:
    timestep: float = 10.0
    duration: float = 300.0
    output_interval: float = 10.0

    S: int = 1
    T_set: float = 22.0
    T_room: float = 18.0

    outside_temp: float = 10.0
    heating_rate: float = 0.06
    cooling_rate: float = 0.01


def parse_config(cfg: Dict[str, Any]) -> SimConfig:
    sc = SimConfig()
    for k, v in cfg.items():
        if not hasattr(sc, k):
            continue
        if k in {"S"}:
            setattr(sc, k, int(v))
        elif k in {"timestep", "duration", "output_interval", "T_set", "T_room",
                   "outside_temp", "heating_rate", "cooling_rate"}:
            setattr(sc, k, float(v))
        else:
            setattr(sc, k, v)
    if sc.timestep <= 0:
        raise ValueError("timestep must be > 0")
    if sc.duration < 0:
        raise ValueError("duration must be >= 0")
    if sc.output_interval <= 0:
        raise ValueError("output_interval must be > 0")
    sc.S = 1 if sc.S else 0
    return sc


class ThermostatSim:
    def __init__(self, cfg: SimConfig):
        self.dt = cfg.timestep
        self.duration = cfg.duration
        self.out_every = cfg.output_interval

        self.S = cfg.S
        self.T_set = cfg.T_set
        self.T_room = cfg.T_room

        self.outside = cfg.outside_temp
        self.heat_rate = cfg.heating_rate
        self.cool_rate = cfg.cooling_rate

        self.time = 0.0
        self.state = "OFF"
        self.H = 0

    def _controller(self) -> None:
        if self.S == 0:
            self.state = "OFF"
            self.H = 0
            return

        if self.T_room < self.T_set:
            self.state = "ON"
            self.H = 1
        else:
            self.state = "OFF"
            self.H = 0

    def _plant(self) -> None:
        heat = self.heat_rate * self.H
        cool = self.cool_rate * (self.outside - self.T_room)
        self.T_room = self.T_room + self.dt * (heat + cool)

    def step(self) -> None:
        self._controller()
        self._plant()
        self.time += self.dt

    def run(self) -> None:
        steps = int(self.duration / self.dt + 1e-9)
        out_steps = max(1, int(self.out_every / self.dt + 1e-9))

        print("time\tS\tT_set\tT_room\tH\tstate")
        for k in range(steps + 1):
            if k % out_steps == 0:
                print(f"{self.time:.1f}\t{self.S}\t{self.T_set:.2f}\t{self.T_room:.2f}\t{self.H}\t{self.state}")
            if k < steps:
                self.step()


def main() -> int:
    ap = argparse.ArgumentParser(description="Simple digital thermostat simulator")
    ap.add_argument("--config", default="config.yaml", help="YAML config file (default: config.yaml)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    cfg = parse_config(load_yaml(cfg_path))
    sim = ThermostatSim(cfg)
    sim.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
