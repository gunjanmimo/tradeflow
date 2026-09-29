"""
Research tools: historical bars, candidate signals, and an honest measure of
whether a signal has an edge after costs -- before anything is allowed to trade.

    python -m research download --days 730          # cache 1-minute bars
    python -m research edge                         # test every signal
    python -m research edge --signals reversal_z --horizons 15,30
"""
