"""
Scout: our own stock discovery. Ranks the US stocks worth watching today.

Every hour (SCOUT_INTERVAL_SECONDS) it:

  1. gathers a pool from sources we query ourselves (scout/sources.py): Alpaca's
     most active stocks and top gainers, every symbol in the last day of Alpaca
     news, the stocks Reddit discusses most (ApeWisdom), StockTwits' trending
     list, the curated universe and the current watchlist
  2. keeps real, liquid US common stocks: listed on Alpaca, not an ETF, warrant,
     unit or leveraged product, price and daily dollar volume above the floors,
     60+ sessions of history
  3. scores four components (scout/ranker.py), each 0..1:
       performance   past-window returns, risk-adjusted momentum, trend and
                     volume surge from daily bars
       today         today's move and volume against the previous session
       news          headline count in the last day and their tone (Jev/Laya)
       discussion    Reddit mentions and their growth, StockTwits trending
  4. puts the top SCOUT_TOP_N on the watchlist and logs the ranking
     (data/scout/rankings.jsonl) so `python -m scout scorecard` can judge it

The Watcher agent (scout/watcher.py) then follows each pick every few seconds and
the scout strategy (engine/strategies/scout.py) trades a pick once the watcher
has been confident long enough. Nothing here places an order.
"""
