verified-cps-cosynthesis

End-to-end pipeline for co-synthesizing verified defense-in-depth cyber-physical systems from SysML v2 specifications. The pipeline produces formally verified controllers, fault-tolerant sensor filters, and anomaly detectors, automating tasks that typically require multiple specialists.

Features

SysML v2 → LTL Extraction
Automatically translates safety and liveness requirements from SysML v2 into Linear Temporal Logic properties.

SysML v2 → Digital Twin
Generates an executable simulation environment including plant dynamics, sensor configuration, and redundancy topology.

LTL → Adversarial RL Objectives
Converts formal safety properties into constraints for adversarial reinforcement learning.

RL + Digital Twin → Controller Synthesis
Trains controllers in simulation environments with formally derived objectives.

Model Verification
Formally verifies trained controllers against extracted LTL properties using neural network verification and model checking.

Defense-in-Depth Components
Produces three integrated components from a single specification:

Verified Controller

Fault-Tolerant Sensor Filter
