#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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
    # simulation
    timestep: float = 0.1
    duration: float = 60.0
    output_interval: float = 1.0
    initial_state: Optional[str] = None  # "OFF" or "ON" (optional)

    # controller inputs
    S: bool = True
    T_set: float = 22.0

    # room
    T_room: float = 18.0
    T_ambient: float = 10.0
    thermal_mass: float = 50.0
    leak_rate: float = 0.01

    # heater
    max_heat_flow: float = 30.0


def _get(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    # simple helper
    return cfg[key] if key in cfg else default


def parse_config(cfg: Dict[str, Any]) -> SimConfig:
    sc = SimConfig()

    # simulation params
    sc.timestep = float(_get(cfg, "timestep", sc.timestep))
    sc.duration = float(_get(cfg, "duration", sc.duration))
    sc.output_interval = float(_get(cfg, "output_interval", sc.output_interval))
    if "initial_state" in cfg:
        sc.initial_state = str(cfg["initial_state"]).strip().upper()

    # model params using your YAML names
    sc.S = bool(_get(cfg, "controller-S", sc.S))
    sc.T_set = float(_get(cfg, "controller-T-set", sc.T_set))

    sc.T_room = float(_get(cfg, "room-T-room", sc.T_room))
    sc.T_ambient = float(_get(cfg, "room-T-ambient", sc.T_ambient))
    sc.thermal_mass = float(_get(cfg, "room-thermal-mass", sc.thermal_mass))
    sc.leak_rate = float(_get(cfg, "room-leak-rate", sc.leak_rate))

    sc.max_heat_flow = float(_get(cfg, "heater-max-heat-flow", sc.max_heat_flow))

    # basic validation
    if sc.timestep <= 0:
        raise ValueError("timestep must be > 0")
    if sc.duration < 0:
        raise ValueError("duration must be >= 0")
    if sc.output_interval <= 0:
        raise ValueError("output_interval must be > 0")
    if sc.thermal_mass <= 0:
        raise ValueError("room-thermal-mass must be > 0")
    if sc.leak_rate < 0:
        raise ValueError("room-leak-rate must be >= 0")
    if sc.max_heat_flow < 0:
        raise ValueError("heater-max-heat-flow must be >= 0")
    if sc.initial_state is not None and sc.initial_state not in {"OFF", "ON"}:
        raise ValueError("initial_state must be OFF or ON (if provided)")

    return sc


class ThermostatSystemSim:
    """
    Small "system-level" sim:
    - Controller state machine: OFF/ON
    - Heater heat flow constraint
    - Room dynamics constraint (Euler integration)
    """

    def __init__(self, cfg: SimConfig):
        self.dt = cfg.timestep
        self.duration = cfg.duration
        self.out_every = cfg.output_interval

        # controller inputs
        self.S = cfg.S
        self.T_set = cfg.T_set

        # room state + params
        self.T_room = cfg.T_room
        self.T_ambient = cfg.T_ambient
        self.thermal_mass = cfg.thermal_mass
        self.leak_rate = cfg.leak_rate

        # heater param
        self.max_heat_flow = cfg.max_heat_flow

        # system state
        self.time = 0.0
        self.state = cfg.initial_state if cfg.initial_state else "OFF"
        self.H = 1 if self.state == "ON" else 0

    # Controller state machine (ThermostatBehavior) 
    def _update_controller_state(self) -> None:
        # OFF -> ON if (S and T_room < T_set)
        if self.state == "OFF":
            if self.S and (self.T_room < self.T_set):
                self.state = "ON"

        # ON -> OFF if ((not S) or (T_room >= T_set))
        elif self.state == "ON":
            if (not self.S) or (self.T_room >= self.T_set):
                self.state = "OFF"

        else:
            # fallback 
            self.state = "OFF"

        # output constraint from controller state
        self.H = 1 if self.state == "ON" else 0

    # Heater + Room constraints/dynamics 
    def _heater_heat_flow(self) -> float:
        # thermal.heatFlow == (H ? maxHeatFlow : 0)
        return self.max_heat_flow if self.H == 1 else 0.0

    def _room_derivative(self, heat_flow: float) -> float:
        # der(T_room) == (heatFlow / thermalMass) - leakRate*(T_room - T_ambient)
        return (heat_flow / self.thermal_mass) - self.leak_rate * (self.T_room - self.T_ambient)

    def step(self) -> None:
        # 1) controller reads current T_room
        self._update_controller_state()

        # 2) heater produces heat flow based on H
        heat_flow = self._heater_heat_flow()

        # 3) room updates based on constraints (simple Euler)
        dT = self._room_derivative(heat_flow)
        self.T_room = self.T_room + self.dt * dT

        self.time += self.dt

    def run(self) -> None:
        steps = int(self.duration / self.dt + 1e-9)
        out_steps = max(1, int(self.out_every / self.dt + 1e-9))

        print("time\tS\tT_set\tT_room\tH\tstate\theatFlow")
        for k in range(steps + 1):
            if k % out_steps == 0:
                heat_flow = self._heater_heat_flow()
                s_int = 1 if self.S else 0
                print(f"{self.time:.1f}\t{s_int}\t{self.T_set:.2f}\t{self.T_room:.2f}\t{self.H}\t{self.state}\t{heat_flow:.2f}")
            if k < steps:
                self.step()


def main() -> int:
    ap = argparse.ArgumentParser(description="Digital thermostat system simulator (matches SysML constraints)")
    ap.add_argument("--config", default="config.yaml", help="YAML config file (default: config.yaml)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    cfg = parse_config(load_yaml(cfg_path))
    sim = ThermostatSystemSim(cfg)
    sim.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
