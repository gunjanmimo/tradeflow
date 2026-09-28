"""
Few-shot adaptation: learn a short-horizon forecast that re-fits to the latest
market in minutes, and trade it only where the forecast beats the costs.

    fewshot/data.py      decision-bar dataset: RL features -> next-H-minute return
    fewshot/models.py    a tiny forecaster (numpy-exportable for the live engine)
    fewshot/meta.py      fine-tuning and first-order MAML meta-training
    fewshot/chronos.py   Chronos-Bolt (a small pretrained forecaster) + few-shot calibration
    fewshot/evaluate.py  walk-forward: adapt on the days before, trade the day

    python -m fewshot                 # every adapter, walk-forward, vs baselines
"""
