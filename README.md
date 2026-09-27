# TradeFlow: Sub-Second Quant & Laya Trading Terminal

An autonomous, continuous quantitative trading system combining **Convai Innovations' Laya decision model** for sub-second sentiment scoring, a high-frequency **Quant Matrix ($t_1 \dots t_N$)**, deterministic risk management, and the **Alpaca Trading API**.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│               4-PILLAR MARKET INTELLIGENCE CONSENSUS                   │
│  1. SEC Form 4 (Insider Purchases via SEC EDGAR)       [Weight: 35%]   │
│  2. eToro (Top Sharpe Popular Investor Portfolios)     [Weight: 25%]   │
│  3. Dub.app / Public.com (Thematic Creator Pies)       [Weight: 20%]   │
│  4. StockTwits (Sentiment Leaders & Bull/Bear Ratios)  [Weight: 20%]   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                         (Consensus Score & Thesis)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. Laya Decision Model (Convai 421M ModernBERT)                        │
│    Fast in-process probability scoring: { pos_prob, neg_prob, open }   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. In-Memory Quant Matrix (t1...tN)                                    │
│    Sub-millisecond indicators: EMA 9/21, RSI 14, ATR, Spread, Volume  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. Deterministic Risk Guard & Alpaca Paper Execution                   │
│    2% equity risk sizing, automated Bracket Orders (SL + TP)           │
└────────────────────────────────────────────────────────────────────────┘
```

### The Sub-Second Execution Path
1. **In-Process Laya Model**: Evaluates news headlines and catalyst statements into calibrated probabilities (`pos_prob`, `neg_prob`) in **15–35ms** without external LLM API roundtrips.
2. **Asynchronous Memory Cache**: Sentiment scores are pre-cached in RAM (`0.001ms` lookup).
3. **C-Accelerated Quant Matrix**: RSI, EMA, and volatility are computed on rolling circular buffers in **< 0.05ms**.
4. **Alpaca Execution**: Automated bracket orders (Entry + Stop Loss + Take Profit) sent via `alpaca-py`.

---

## 🚀 Quick Start

### 1. Configure Alpaca Paper Trading Keys
Create or edit `api/.env`:
```bash
cp api/.env.example api/.env
```
Fill in your keys from your [Alpaca Paper Account](https://alpaca.markets):
```env
ALPACA_API_KEY=PK_YOUR_PAPER_KEY
ALPACA_SECRET_KEY=YOUR_PAPER_SECRET
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_URL=https://data.alpaca.markets
IS_PAPER=True
```
*(Note: If left as placeholders, TradeFlow automatically runs in **simulated paper trading mode** so you can test the entire pipeline safely without an account).*

### 2. Launch with Docker Compose
Run both backend and frontend in containers with a single command from the root directory:
```bash
docker compose up --build
```
- **Web Dashboard**: [http://localhost:5173](http://localhost:5173)
- **FastAPI Backend & Telemetry**: [http://localhost:8000](http://localhost:8000) (WebSocket at `ws://localhost:8000/ws`)

To run in detached/background mode:
```bash
docker compose up -d
```
To stop the containers:
```bash
docker compose down
```

---

## 📊 Features & Asset Classes

- **Dual Asset Support (US Equities + 24/7 Crypto)**:
  - **US Equities**: `NVDA`, `AAPL`, `MSFT`, `PLTR`, etc. (NYSE / NASDAQ).
  - **24/7 Cryptocurrencies**: `BTC/USD`, `ETH/USD`, `SOL/USD` (Trades continuously 24/7/365, including weekends and evenings!).
  - **Fractional Sizing for Crypto**: Automatically sizes positions in decimal quantities (e.g. `0.024 BTC`).
- **4-Pillar Consensus Engine**: Fuses **SEC Form 4 / On-Chain Whale Flows** (35%), **eToro Copy Portfolios** (25%), **Dub/Public Thematic Pies** (20%), and **StockTwits Sentiment** (20%).
- **In-Process Laya Model**: Classifies financial catalysts and news headlines into calibrated probabilities in **15–35ms**.
- **C-Speed Quant Matrix ($t_1 \dots t_N$)**: Evaluates EMA 9/21, RSI 14, ATR volatility, and bid-ask spreads in **< 0.05ms**.
- **Interactive Dashboard**: Filter by `ALL`, `STOCKS`, or `CRYPTO (24/7)`, inject live news to test Laya, and manage positions.
- **Emergency Kill Switch**: Panic button to liquidate open positions and halt trading immediately.
