# DQN Applied to Optimal Trade Execution

This repository explores **optimal trade execution** using **Reinforcement Learning (Deep Q-Network)** in an **Almgren–Chriss-style market impact environment**, and compares the learned policy to standard execution benchmarks (TWAP, VWAP proxy, and Almgren–Chriss).

> Note: The project structure will change, the current code is kept minimal to provide an overview and a working baseline.

---

## Overview

The goal is to liquidate (or acquire) a large inventory over a fixed horizon while balancing:
- **Temporary market impact** (execution price worsens with aggressive trading)
- **Permanent impact** (trading pressure moves the mid-price)
- **Stochastic price movement** (volatility)

The environment is calibrated using historical MSFT data via simple regressions:
- **Permanent impact (γ)** estimated by regressing next-day returns on a trade-sign proxy
- **Temporary impact (η)** estimated by regressing intraday slippage on the same proxy
- **Volatility (σ)** estimated from daily log returns

A **DQN agent** is trained to select discrete trading rates that determine how aggressively to trade at each step.

---

## Files

- `Project.py`  
  - Downloads MSFT data using `yfinance`
  - Builds features + estimates parameters (γ, η, σ)
  - Implements a simplified Almgren–Chriss execution environment:
    - `init_env`, `reset_env`, `step_env`
  - Implements benchmark strategies:
    - `run_twap`
    - `run_vwap` (with a U-shaped synthetic volume profile)
    - `run_almgren_chriss`
  - Produces comparison plots for inventory decay and cumulative cash

- `DQN.py`
  - Implements a Deep Q-Network (policy + target network)
  - Uses **Prioritized Experience Replay (PER)**
  - Trains over a grid of hyperparameters (PER alpha, epsilon decay, batch size)
  - Evaluates the best model and compares against benchmarks
  - Computes simple TCA metrics (implementation shortfall)

---

## Environment (Execution Model)

State:
- `S`: mid-price
- `q`: remaining inventory
- `t`: time step

Action:
- Discrete choice among **11 execution aggressiveness levels** (0%, 10%, ..., 100%)
- Converted into a trading rate `v_t` and executed during the step

Price dynamics (simplified):
- Execution price includes **temporary impact**
- Mid-price evolves with **volatility** and **permanent impact**

Reward shaping:
- Revenue per step (scaled by `S0`)
- Penalty for impact (scaled)
- Optional terminal penalty for leftover inventory

