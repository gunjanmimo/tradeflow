import React, { useState, useEffect, useMemo } from 'react';

const money = (v, decimals = 2) => {
  if (v === null || v === undefined || isNaN(v)) return '$0.00';
  const num = Number(v);
  const sign = num < 0 ? '-' : '';
  return `${sign}$${Math.abs(num).toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })}`;
};

const pct = (v, d = 1) => (v === null || v === undefined || isNaN(v) ? '0.0%' : `${Number(v).toFixed(d)}%`);

export function CalculatorIcon({ className = 'w-4 h-4' }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden="true">
      <rect width="16" height="20" x="4" y="2" rx="2" />
      <line x1="8" x2="16" y1="6" y2="6" />
      <line x1="16" x2="16" y1="14" y2="18" />
      <path d="M16 10h.01" />
      <path d="M12 10h.01" />
      <path d="M8 10h.01" />
      <path d="M12 14h.01" />
      <path d="M8 14h.01" />
      <path d="M12 18h.01" />
      <path d="M8 18h.01" />
    </svg>
  );
}

export function LockIcon({ className = 'w-3 h-3' }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden="true">
      <rect width="18" height="11" x="3" y="11" rx="2" ry="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
  );
}

export function DailyPnlChip({ dailyPnl, onClick }) {
  const net = Number(dailyPnl?.net_pnl || 0);
  const positive = net >= 0;
  return (
    <button
      onClick={onClick}
      className="group flex items-center gap-2 rounded-xl px-3 py-1.5 bg-white/[0.03] ring-1 ring-white/10 hover:ring-emerald-400/50 hover:bg-emerald-400/5 transition-all text-left"
      title="Open Daily Profit & Loss Calculator & Risk Simulator"
    >
      <CalculatorIcon className="w-4 h-4 text-emerald-400 group-hover:scale-110 transition-transform" />
      <div className="flex flex-col min-w-[70px]">
        <span className="text-[10px] uppercase tracking-wider text-slate-500 leading-none flex items-center gap-1">
          Daily PnL <span className="text-[9px] text-emerald-400/80 font-normal">Calc</span>
        </span>
        <span className={`text-sm font-mono font-semibold leading-tight ${positive ? 'text-emerald-400' : 'text-rose-400'}`}>
          {positive ? '+' : ''}${net.toFixed(2)}
        </span>
      </div>
    </button>
  );
}

export function DailyPnlCalculatorModal({ isOpen, onClose, telemetry, apiBase }) {
  const [tab, setTab] = useState('overview'); // 'overview' | 'calculator' | 'history'
  const [historyData, setHistoryData] = useState([]);
  const [scenariosData, setScenariosData] = useState(null);

  // Interactive Calculator State
  const hardCap = telemetry?.budget?.hard_cap ?? telemetry?.budget?.allocated_capital ?? 10000;
  const lockedBrokerEquity = telemetry?.budget?.locked_broker_equity ?? Math.max(0, (telemetry?.account?.equity || 100000) - hardCap);
  const liveNetPnl = Number(telemetry?.daily_pnl?.net_pnl || 0);

  const [targetProfit, setTargetProfit] = useState(100);
  const [maxLoss, setMaxLoss] = useState(50);
  const [plannedTrades, setPlannedTrades] = useState(5);
  const [winRate, setWinRate] = useState(60);
  const [rewardRisk, setRewardRisk] = useState(2.0);
  const [riskPerTrade, setRiskPerTrade] = useState(20);

  // Fetch full daily PnL data & history
  useEffect(() => {
    if (!isOpen) return;
    let active = true;
    async function loadData() {
      try {
        const res = await fetch(`${apiBase}/api/daily-pnl`);
        if (res.ok) {
          const data = await res.json();
          if (active) {
            setHistoryData(data.history || []);
            setScenariosData(data.scenarios || null);
          }
        }
      } catch (err) {
        console.error('Failed to load daily PnL history:', err);
      }
    }
    loadData();
    return () => { active = false; };
  }, [isOpen, apiBase]);

  // Real-time Calculator Math
  const calc = useMemo(() => {
    const wr = Math.max(0, Math.min(100, Number(winRate))) / 100;
    const rr = Math.max(0.1, Number(rewardRisk));
    const risk = Math.max(1, Number(riskPerTrade));
    const trades = Math.max(1, parseInt(plannedTrades, 10) || 1);
    const target = Number(targetProfit) || 0;
    const lossLimit = Math.max(1, Number(maxLoss) || 1);

    const winAmount = risk * rr;
    const lossAmount = risk;
    const evPerTrade = (wr * winAmount) - ((1 - wr) * lossAmount);
    const projectedPnl = evPerTrade * trades;
    const projectedReturnPct = hardCap > 0 ? (projectedPnl / hardCap) * 100 : 0;
    const breakevenWinRatePct = (1 / (1 + rr)) * 100;

    const targetGap = Math.max(0, target - liveNetPnl);
    const targetProgressPct = target > 0 ? Math.min(100, Math.max(0, (liveNetPnl / target) * 100)) : 0;
    const currentDrawdown = Math.abs(Math.min(0, liveNetPnl));
    const lossLimitUsedPct = Math.min(100, Math.max(0, (currentDrawdown / lossLimit) * 100));
    const remainingLossCushion = Math.max(0, lossLimit - currentDrawdown);

    const tradesToTarget = evPerTrade > 0 && targetGap > 0 ? Math.ceil(targetGap / evPerTrade) : 0;
    const tradesBeforeLossLimit = lossAmount > 0 ? Math.floor(remainingLossCushion / lossAmount) : 0;

    let status = 'ON_TRACK';
    let statusLabel = 'On Track';
    let statusTone = 'text-emerald-400 bg-emerald-500/10 ring-emerald-500/30';

    if (liveNetPnl >= target && target > 0) {
      status = 'TARGET_REACHED';
      statusLabel = 'Target Reached 🎯';
      statusTone = 'text-emerald-300 bg-emerald-500/20 ring-emerald-400/50';
    } else if (lossLimitUsedPct >= 100) {
      status = 'LOSS_LIMIT_REACHED';
      statusLabel = 'Daily Loss Limit Hit 🛑';
      statusTone = 'text-rose-400 bg-rose-500/20 ring-rose-500/50';
    } else if (lossLimitUsedPct >= 80) {
      status = 'NEAR_LOSS_LIMIT';
      statusLabel = 'Near Loss Limit ⚠️';
      statusTone = 'text-amber-400 bg-amber-500/20 ring-amber-500/40';
    } else if (evPerTrade < 0) {
      status = 'NEGATIVE_EXPECTANCY';
      statusLabel = 'Negative Expectancy ⚠️';
      statusTone = 'text-amber-400 bg-amber-500/10 ring-amber-500/30';
    }

    return {
      winAmount,
      lossAmount,
      evPerTrade,
      projectedPnl,
      projectedReturnPct,
      breakevenWinRatePct,
      targetGap,
      targetProgressPct,
      lossLimitUsedPct,
      remainingLossCushion,
      tradesToTarget,
      tradesBeforeLossLimit,
      status,
      statusLabel,
      statusTone,
    };
  }, [winRate, rewardRisk, riskPerTrade, plannedTrades, targetProfit, maxLoss, hardCap, liveNetPnl]);

  if (!isOpen) return null;

  const dp = telemetry?.daily_pnl || {};
  const realized = Number(dp.realized_pnl || 0);
  const unrealized = Number(dp.unrealized_pnl || 0);
  const net = Number(dp.net_pnl || (realized + unrealized));
  const closedTrades = Number(dp.closed_trades || 0);
  const winningTrades = Number(dp.winning_trades || 0);
  const losingTrades = Number(dp.losing_trades || 0);
  const winRateActual = closedTrades > 0 ? (winningTrades / closedTrades) * 100 : 0;
  const profitFactor = dp.profit_factor !== undefined && dp.profit_factor !== null ? Number(dp.profit_factor) : (Number(dp.gross_loss || 0) === 0 && Number(dp.gross_profit || 0) > 0 ? '∞' : 0);
  const returnOnBudget = hardCap > 0 ? (net / hardCap) * 100 : 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fadeIn">
      <div className="relative w-full max-w-2xl bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl p-6 overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-800/80">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-emerald-500/10 ring-1 ring-emerald-500/30 text-emerald-400">
              <CalculatorIcon className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                Daily Profit & Loss Calculator
                <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 ring-1 ring-white/10">
                  {dp.trading_day || 'Today'}
                </span>
              </h2>
              <p className="text-xs text-slate-400">Real-time daily accounting, scenario simulation, and expectancy model</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/5 transition-all text-sm font-semibold"
          >
            ✕
          </button>
        </div>

        {/* Locked Broker Equity & Hard Cap Guarantee Banner */}
        <div className="mt-3 p-3 rounded-xl bg-amber-500/[0.05] ring-1 ring-amber-500/20 text-xs text-slate-300 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <LockIcon className="w-4 h-4 text-amber-400 shrink-0" />
            <span>
              <strong className="text-amber-300">Main Broker Equity:</strong> {money(lockedBrokerEquity, 0)} is locked and untouched.
              Bot trades strictly within hard cap <strong className="text-white">{money(hardCap, 0)}</strong>.
            </span>
          </div>
          <span className="px-2 py-0.5 rounded-md bg-amber-400/10 text-amber-300 font-mono text-[10px] font-semibold tracking-wider uppercase shrink-0">
            Locked & Isolated
          </span>
        </div>

        {/* Tab Navigation */}
        <div className="grid grid-cols-3 gap-1 mt-4 p-1 rounded-xl bg-white/[0.03] ring-1 ring-white/10 text-xs font-semibold">
          {[
            ['overview', 'Today Overview'],
            ['calculator', 'P&L Calculator'],
            ['history', '30-Day Ledger'],
          ].map(([k, label]) => (
            <button
              key={k}
              onClick={() => setTab(k)}
              className={`py-2 rounded-lg transition-all ${
                tab === k
                  ? 'bg-emerald-500/20 text-emerald-300 ring-1 ring-emerald-400/40 shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Tab Content Container */}
        <div className="flex-1 overflow-y-auto mt-4 pr-1 space-y-4 text-xs">
          {/* TAB 1: OVERVIEW */}
          {tab === 'overview' && (
            <div className="space-y-4">
              {/* Hero Stat Tile */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                <div className="p-3 rounded-xl bg-black/30 ring-1 ring-white/10 text-center">
                  <div className="text-[10px] uppercase text-slate-500">Net Daily PnL</div>
                  <div className={`text-base font-mono font-bold ${net >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {net >= 0 ? '+' : ''}{money(net)}
                  </div>
                  <div className="text-[10px] text-slate-400">{pct(returnOnBudget, 2)} on bot cap</div>
                </div>

                <div className="p-3 rounded-xl bg-black/30 ring-1 ring-white/10 text-center">
                  <div className="text-[10px] uppercase text-slate-500">Realised PnL</div>
                  <div className={`text-base font-mono font-semibold ${realized >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {realized >= 0 ? '+' : ''}{money(realized)}
                  </div>
                  <div className="text-[10px] text-slate-400">{closedTrades} closed trades</div>
                </div>

                <div className="p-3 rounded-xl bg-black/30 ring-1 ring-white/10 text-center">
                  <div className="text-[10px] uppercase text-slate-500">Open Unrealised</div>
                  <div className={`text-base font-mono font-semibold ${unrealized >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {unrealized >= 0 ? '+' : ''}{money(unrealized)}
                  </div>
                  <div className="text-[10px] text-slate-400">Live mark-to-market</div>
                </div>

                <div className="p-3 rounded-xl bg-black/30 ring-1 ring-white/10 text-center">
                  <div className="text-[10px] uppercase text-slate-500">Win Rate / PF</div>
                  <div className="text-base font-mono font-semibold text-white">
                    {pct(winRateActual, 1)}
                  </div>
                  <div className="text-[10px] text-slate-400">PF: {profitFactor}</div>
                </div>
              </div>

              {/* Trade Stats Breakdown */}
              <div className="p-4 rounded-xl bg-white/[0.02] ring-1 ring-white/10 space-y-3">
                <div className="flex items-center justify-between text-[11px] text-slate-400">
                  <span>Trading Execution ({closedTrades} Total Trades)</span>
                  <span>{winningTrades} Won · {losingTrades} Lost</span>
                </div>
                {/* Progress bar of win/loss */}
                <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden flex">
                  <div style={{ width: `${winRateActual}%` }} className="bg-emerald-400 h-full transition-all" />
                  <div style={{ width: `${100 - winRateActual}%` }} className="bg-rose-500 h-full transition-all" />
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 text-center font-mono">
                  <div>
                    <span className="text-[9px] uppercase text-slate-500 block">Gross Profit</span>
                    <span className="text-emerald-400 font-semibold">{money(dp.gross_profit || 0)}</span>
                  </div>
                  <div>
                    <span className="text-[9px] uppercase text-slate-500 block">Gross Loss</span>
                    <span className="text-rose-400 font-semibold">-{money(dp.gross_loss || 0)}</span>
                  </div>
                  <div>
                    <span className="text-[9px] uppercase text-slate-500 block">Avg Win</span>
                    <span className="text-slate-200">{money(dp.avg_win || 0)}</span>
                  </div>
                  <div>
                    <span className="text-[9px] uppercase text-slate-500 block">Avg Loss</span>
                    <span className="text-slate-200">{money(dp.avg_loss || 0)}</span>
                  </div>
                </div>
              </div>

              {/* Open Positions Scenario Projections */}
              <div className="p-4 rounded-xl bg-cyan-400/[0.03] ring-1 ring-cyan-400/20 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] uppercase tracking-wider text-cyan-300 font-semibold">
                    Position Scenarios ({scenariosData?.open_positions_count || 0} Open)
                  </span>
                  <span className="text-[10px] text-slate-400">Projected Daily Net PnL</span>
                </div>
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div className="p-2 rounded-lg bg-emerald-500/10 ring-1 ring-emerald-500/30">
                    <span className="text-[9px] uppercase text-emerald-400 block font-semibold">Best Case (All TP)</span>
                    <span className="text-sm font-mono font-bold text-emerald-300">
                      {Number(scenariosData?.projected_best_net || net) >= 0 ? '+' : ''}
                      {money(scenariosData?.projected_best_net || net)}
                    </span>
                  </div>
                  <div className="p-2 rounded-lg bg-white/5 ring-1 ring-white/10">
                    <span className="text-[9px] uppercase text-slate-400 block">Current Mark</span>
                    <span className="text-sm font-mono font-bold text-white">
                      {net >= 0 ? '+' : ''}{money(net)}
                    </span>
                  </div>
                  <div className="p-2 rounded-lg bg-rose-500/10 ring-1 ring-rose-500/30">
                    <span className="text-[9px] uppercase text-rose-400 block font-semibold">Worst Case (All SL)</span>
                    <span className="text-sm font-mono font-bold text-rose-300">
                      {Number(scenariosData?.projected_worst_net || net) >= 0 ? '+' : ''}
                      {money(scenariosData?.projected_worst_net || net)}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: CALCULATOR & SIMULATOR */}
          {tab === 'calculator' && (
            <div className="space-y-4">
              {/* Goal & Expectancy Summary Banner */}
              <div className="p-3 rounded-xl bg-black/40 ring-1 ring-white/10 flex items-center justify-between">
                <div>
                  <div className="text-[10px] uppercase text-slate-400 font-medium">Daily Expectancy Status</div>
                  <div className="text-sm font-semibold text-white mt-0.5">
                    Projected Daily PnL: <strong className={calc.projectedPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}>{calc.projectedPnl >= 0 ? '+' : ''}{money(calc.projectedPnl)}</strong>
                    <span className="text-slate-400 text-xs font-normal ml-1">({pct(calc.projectedReturnPct, 2)} on cap)</span>
                  </div>
                </div>
                <span className={`px-2.5 py-1 rounded-full text-xs font-semibold ring-1 ${calc.statusTone}`}>
                  {calc.statusLabel}
                </span>
              </div>

              {/* Interactive Input Sliders & Fields */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 p-4 rounded-xl bg-white/[0.02] ring-1 ring-white/10">
                <div>
                  <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                    <span>Daily Profit Target ($)</span>
                    <span className="font-mono text-emerald-300">{money(targetProfit)}</span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="1000"
                    step="10"
                    value={targetProfit}
                    onChange={(e) => setTargetProfit(Number(e.target.value))}
                    className="w-full accent-emerald-400"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                    <span>Max Daily Loss Limit ($)</span>
                    <span className="font-mono text-rose-300">{money(maxLoss)}</span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="500"
                    step="5"
                    value={maxLoss}
                    onChange={(e) => setMaxLoss(Number(e.target.value))}
                    className="w-full accent-rose-400"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                    <span>Win Rate Expected (%)</span>
                    <span className="font-mono text-cyan-300">{winRate}%</span>
                  </div>
                  <input
                    type="range"
                    min="20"
                    max="90"
                    step="1"
                    value={winRate}
                    onChange={(e) => setWinRate(Number(e.target.value))}
                    className="w-full accent-cyan-400"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                    <span>Reward / Risk Ratio</span>
                    <span className="font-mono text-amber-300">{Number(rewardRisk).toFixed(1)}x</span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="4.0"
                    step="0.1"
                    value={rewardRisk}
                    onChange={(e) => setRewardRisk(Number(e.target.value))}
                    className="w-full accent-amber-400"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                    <span>Risk Per Trade ($)</span>
                    <span className="font-mono text-slate-200">{money(riskPerTrade)}</span>
                  </div>
                  <input
                    type="range"
                    min="5"
                    max="100"
                    step="5"
                    value={riskPerTrade}
                    onChange={(e) => setRiskPerTrade(Number(e.target.value))}
                    className="w-full accent-slate-300"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                    <span>Planned Trades / Day</span>
                    <span className="font-mono text-slate-200">{plannedTrades}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="25"
                    step="1"
                    value={plannedTrades}
                    onChange={(e) => setPlannedTrades(Number(e.target.value))}
                    className="w-full accent-slate-300"
                  />
                </div>
              </div>

              {/* Calculated Expectancy Details */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center">
                <div className="p-2.5 rounded-xl bg-black/20 ring-1 ring-white/10">
                  <span className="text-[9px] uppercase text-slate-500 block">EV Per Trade</span>
                  <span className={`text-sm font-mono font-bold ${calc.evPerTrade >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {calc.evPerTrade >= 0 ? '+' : ''}{money(calc.evPerTrade)}
                  </span>
                </div>
                <div className="p-2.5 rounded-xl bg-black/20 ring-1 ring-white/10">
                  <span className="text-[9px] uppercase text-slate-500 block">Breakeven Win Rate</span>
                  <span className="text-sm font-mono font-bold text-cyan-300">{pct(calc.breakevenWinRatePct, 1)}</span>
                </div>
                <div className="p-2.5 rounded-xl bg-black/20 ring-1 ring-white/10">
                  <span className="text-[9px] uppercase text-slate-500 block">Trades to Target</span>
                  <span className="text-sm font-mono font-bold text-white">{calc.tradesToTarget || '—'}</span>
                </div>
                <div className="p-2.5 rounded-xl bg-black/20 ring-1 ring-white/10">
                  <span className="text-[9px] uppercase text-slate-500 block">Loss Stop Cushion</span>
                  <span className="text-sm font-mono font-bold text-amber-300">{money(calc.remainingLossCushion)}</span>
                </div>
              </div>

              {/* Target & Loss Limit Progress Bars */}
              <div className="space-y-2 p-3 rounded-xl bg-white/[0.02] ring-1 ring-white/10">
                <div>
                  <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                    <span>Daily Profit Target Progress ({pct(calc.targetProgressPct, 0)})</span>
                    <span>Gap: {money(calc.targetGap)}</span>
                  </div>
                  <div className="h-1.5 w-full bg-white/10 rounded-full overflow-hidden">
                    <div style={{ width: `${calc.targetProgressPct}%` }} className="h-full bg-emerald-400 transition-all duration-300" />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                    <span>Daily Loss Limit Used ({pct(calc.lossLimitUsedPct, 0)})</span>
                    <span>Max {money(maxLoss)}</span>
                  </div>
                  <div className="h-1.5 w-full bg-white/10 rounded-full overflow-hidden">
                    <div style={{ width: `${calc.lossLimitUsedPct}%` }} className="h-full bg-rose-400 transition-all duration-300" />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: 30-DAY LEDGER */}
          {tab === 'history' && (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-slate-400 text-xs">
                <span>Recent Daily History (Last 31 Trading Days)</span>
                <span className="font-mono text-[11px]">{historyData.length} recorded days</span>
              </div>
              {historyData.length === 0 ? (
                <div className="p-8 text-center text-slate-500 bg-white/[0.02] rounded-xl ring-1 ring-white/10">
                  No historical daily records yet. Trades booked today will persist in this ledger.
                </div>
              ) : (
                <div className="overflow-x-auto rounded-xl ring-1 ring-white/10">
                  <table className="w-full text-left text-xs font-mono">
                    <thead className="bg-slate-950/80 text-slate-400 border-b border-white/10 text-[10px] uppercase">
                      <tr>
                        <th className="py-2.5 px-3">Date</th>
                        <th className="py-2.5 px-3 text-right">Realised PnL</th>
                        <th className="py-2.5 px-3 text-right">Win Rate</th>
                        <th className="py-2.5 px-3 text-right">Profit Factor</th>
                        <th className="py-2.5 px-3 text-right">Gross Profit</th>
                        <th className="py-2.5 px-3 text-right">Gross Loss</th>
                        <th className="py-2.5 px-3 text-right">Trades</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5 bg-slate-900/40">
                      {historyData.map((d) => {
                        const dayPnl = Number(d.realized_pnl || 0);
                        const isWin = dayPnl >= 0;
                        return (
                          <tr key={d.date} className="hover:bg-white/[0.02] transition-colors">
                            <td className="py-2 px-3 font-semibold text-white">{d.date}</td>
                            <td className={`py-2 px-3 text-right font-bold ${isWin ? 'text-emerald-400' : 'text-rose-400'}`}>
                              {isWin ? '+' : ''}{money(dayPnl)}
                            </td>
                            <td className="py-2 px-3 text-right text-slate-300">{pct(d.win_rate_pct, 1)}</td>
                            <td className="py-2 px-3 text-right text-slate-300">{d.profit_factor ?? '—'}</td>
                            <td className="py-2 px-3 text-right text-emerald-300">{money(d.gross_profit)}</td>
                            <td className="py-2 px-3 text-right text-rose-300">-{money(d.gross_loss)}</td>
                            <td className="py-2 px-3 text-right text-slate-400">{d.closed_trades}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer info */}
        <div className="pt-3 mt-3 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-500">
          <span>Main broker equity is locked. Bot trades only within hard cap.</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-white/10 hover:bg-white/15 text-white font-semibold transition-all text-xs"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
