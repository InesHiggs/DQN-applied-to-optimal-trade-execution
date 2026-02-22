# DQN Applied to Optimal Trade Execution

This repository investigates **optimal trade execution** using **Reinforcement Learning (Deep Q-Network)** within an **Almgren–Chriss-style market impact environment**, and compares the learned policy to standard execution benchmarks (TWAP, VWAP proxy, and Almgren–Chriss).

> ⚠️ The project structure will evolve. The current implementation is intentionally compact to provide a clear and reproducible baseline before modular refactoring.

---

## Overview

The objective is to liquidate (or acquire) a large position over a fixed horizon while balancing three competing effects:

* **Temporary Market Impact** — aggressive trading deteriorates execution prices.
* **Permanent Market Impact** — trading pressure shifts the underlying mid-price.
* **Price Uncertainty** — stochastic volatility introduces timing risk.

This setting follows the classical **Almgren–Chriss optimal execution framework**, extended here with **data-driven calibration** and a **reinforcement learning agent** trained to learn execution policies directly from the simulated environment.

---

## Data

Historical daily data for Microsoft (MSFT) from 2019–2024 is downloaded using `yfinance` to calibrate the environment.

![MSFT Price](figures/msft_price.png)

The dataset is used to estimate key structural parameters:

* **Permanent Impact (γ)**
  Estimated by regressing next-day returns on a trade-sign proxy.

* **Temporary Impact (η)**
  Estimated by regressing intraday slippage (`Close − Open`) on the same proxy.

* **Volatility (σ)**
  Computed from daily log-returns to model stochastic price evolution.

These estimates parameterize the simulated execution model.

---

## Repository Structure (Current)

| File         | Description                                                                           |
| ------------ | ------------------------------------------------------------------------------------- |
| `Project.py` | Data download, parameter estimation, environment definition, and benchmark strategies |
| `DQN.py`     | Deep Q-Network implementation with Prioritized Experience Replay                      |
| `figures/`   | Saved plots used in documentation and analysis                                        |

---

## Execution Environment

A discrete-time execution simulator inspired by Almgren–Chriss dynamics is implemented.

### State Representation

Each timestep is defined by:

* `S` — Current mid-price
* `q` — Remaining inventory
* `t` — Time index within the execution horizon

### Action Space

The agent selects one of **11 discrete trading intensities**:

```
{0%, 10%, 20%, ..., 100% of remaining inventory}
```

Each action is mapped to a trading rate `v_t`.

### Price Dynamics

* **Execution Price** incorporates temporary impact:

  ```
  S_exec = S_t − η · v_t
  ```

* **Mid-Price Evolution** includes volatility and permanent impact:

  ```
  S_{t+1} = S_t + σ√Δt · ε − γ · v_t Δt
  ```

  where `ε ~ N(0,1)` represents market noise.

---

## Reward Design

The reward function promotes efficient execution while discouraging excessive market impact:

* Positive contribution from execution revenue (scaled to return space).
* Quadratic penalty on trading intensity to avoid overly aggressive liquidation.
* Terminal penalty if inventory remains unexecuted.

This structure reflects the trade-off between:

> Execution speed, price quality, and risk exposure.

---

## Benchmarks Implemented

Performance is evaluated against standard execution strategies:

| Strategy                       | Description                                                |
| ------------------------------ | ---------------------------------------------------------- |
| **TWAP**                       | Linear liquidation over time                               |
| **VWAP (Proxy)**               | U-shaped synthetic volume curve (high open/close activity) |
| **Almgren–Chriss Closed Form** | Analytical optimal trajectory given risk aversion          |

---

## Deep Reinforcement Learning Approach

The execution policy is learned using a **Deep Q-Network (DQN)** featuring:

* Two hidden layers with ReLU activations
* Discrete execution action space
* **Prioritized Experience Replay (PER)** for improved sample efficiency
* Target network stabilization
* ε-greedy exploration with decay
* Hyperparameter grid search across replay prioritization and exploration schedules

The trained model learns a mapping from market state to execution aggressiveness.

---

## Academic Context

This work is motivated by the optimal execution literature, particularly:

> Almgren, R., & Chriss, N. (2000).
> *Optimal Execution of Portfolio Transactions.*
