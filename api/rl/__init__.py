"""
Reinforcement learning: a PPO agent that decides, every few minutes, whether to
be long or flat in a stock, trained on one-minute bars under the platform's own
fills, stops, costs and session rules.

    rl/features.py   the observation, shared by training and live trading
    rl/dataset.py    sessions of historical bars -> precomputed features
    rl/env.py        a vectorised trading environment (torch, GPU or CPU)
    rl/ppo.py        actor-critic network and the PPO update
    rl/evaluate.py   walk-forward evaluation against baselines
    rl/policy.py     numpy inference for the live engine (no torch needed)
    rl/train.py      python -m rl.train: train, evaluate, gate, save
    rl/live.py       observations from the live minute bars; decision log
"""
