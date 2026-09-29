import React, { useState, useEffect, useRef } from 'react';
import { LatencyChip, LatencyPanel, QuantDeskCard } from './QuantPanels';
import { CapitalModeChip, CapitalPlanPanel } from './CapitalPlan';
import { DiscoveryChip, DiscoveryPanel } from './DiscoveryPanel';
import { ManagerChip, ManagerPanel } from './ManagerPanel';
import { DailyPnlChip, DailyPnlCalculatorModal } from './DailyPnlCalculator';
import { RLChip, RLPanel } from './RLPanel';
import { SmartMoneyPanel } from './SmartMoneyPanel';
import { ScoutPanel } from './ScoutPanel';
import { DeskPanel } from './DeskPanel';
import { NewsIngestPanel } from './NewsIngestPanel';
import { setDisplayTz, clockTime } from './timefmt';
import { MarketsModal } from './MarketsModal';
import { HugeiconsIcon } from '@hugeicons/react';
import { 
  Activity01Icon, 
  OctagonAlertIcon, 
  TrendingUpIcon, 
  TrendingDownIcon, 
  CoinsDollarIcon, 
  PlayIcon, 
  PauseIcon, 
  PlusSignIcon, 
  TrashIcon, 
  FlashIcon, 
  AiBrain01Icon, 
  LayerIcon, 
  UserGroupIcon, 
  ShieldAlertIcon,
  CheckmarkCircle01Icon,
  ClockIcon,
  Compass01Icon,
  Refresh01Icon,
  GlobeIcon,
  Wallet01Icon,
  Edit01Icon,
  Cancel01Icon,
  SlidersHorizontalIcon,
  Robot01Icon,
  Shield01Icon,
  Building01Icon,
  SmartPhone01Icon,
  Comment01Icon,
  LockIcon,
  CircleDotIcon
} from '@hugeicons/core-free-icons';

// Official Hugeicons components
const Activity = (props) => <HugeiconsIcon icon={Activity01Icon} size="1em" {...props} />;
const AlertOctagon = (props) => <HugeiconsIcon icon={OctagonAlertIcon} size="1em" {...props} />;
const TrendingUp = (props) => <HugeiconsIcon icon={TrendingUpIcon} size="1em" {...props} />;
const TrendingDown = (props) => <HugeiconsIcon icon={TrendingDownIcon} size="1em" {...props} />;
const DollarSign = (props) => <HugeiconsIcon icon={CoinsDollarIcon} size="1em" {...props} />;
const Play = (props) => <HugeiconsIcon icon={PlayIcon} size="1em" {...props} />;
const Pause = (props) => <HugeiconsIcon icon={PauseIcon} size="1em" {...props} />;
const Plus = (props) => <HugeiconsIcon icon={PlusSignIcon} size="1em" {...props} />;
const Trash2 = (props) => <HugeiconsIcon icon={TrashIcon} size="1em" {...props} />;
const Zap = (props) => <HugeiconsIcon icon={FlashIcon} size="1em" {...props} />;
const BrainCircuit = (props) => <HugeiconsIcon icon={AiBrain01Icon} size="1em" {...props} />;
const Layers = (props) => <HugeiconsIcon icon={LayerIcon} size="1em" {...props} />;
const Users = (props) => <HugeiconsIcon icon={UserGroupIcon} size="1em" {...props} />;
const ShieldAlert = (props) => <HugeiconsIcon icon={ShieldAlertIcon} size="1em" {...props} />;
const CheckCircle = (props) => <HugeiconsIcon icon={CheckmarkCircle01Icon} size="1em" {...props} />;
const Clock = (props) => <HugeiconsIcon icon={ClockIcon} size="1em" {...props} />;
const Compass = (props) => <HugeiconsIcon icon={Compass01Icon} size="1em" {...props} />;
const RefreshCw = (props) => <HugeiconsIcon icon={Refresh01Icon} size="1em" {...props} />;
const Globe = (props) => <HugeiconsIcon icon={GlobeIcon} size="1em" {...props} />;
const Wallet = (props) => <HugeiconsIcon icon={Wallet01Icon} size="1em" {...props} />;
const Edit3 = (props) => <HugeiconsIcon icon={Edit01Icon} size="1em" {...props} />;
const X = (props) => <HugeiconsIcon icon={Cancel01Icon} size="1em" {...props} />;
const Sliders = (props) => <HugeiconsIcon icon={SlidersHorizontalIcon} size="1em" {...props} />;
const Bot = (props) => <HugeiconsIcon icon={Robot01Icon} size="1em" {...props} />;
const Shield = (props) => <HugeiconsIcon icon={Shield01Icon} size="1em" {...props} />;
const Building = (props) => <HugeiconsIcon icon={Building01Icon} size="1em" {...props} />;
const SmartPhone = (props) => <HugeiconsIcon icon={SmartPhone01Icon} size="1em" {...props} />;
const Comment = (props) => <HugeiconsIcon icon={Comment01Icon} size="1em" {...props} />;
const Lock = (props) => <HugeiconsIcon icon={LockIcon} size="1em" {...props} />;
const CircleDot = (props) => <HugeiconsIcon icon={CircleDotIcon} size="1em" {...props} />;

const riskTone = (f) => (f <= 3 ? 'text-emerald-400' : f <= 6 ? 'text-amber-400' : 'text-rose-400');

// Centered dialog with backdrop; closes on Escape or backdrop click
function Modal({ title, icon, onClose, children, wide = false, xl = false }) {
  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm" onMouseDown={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        className={`w-full ${xl ? 'max-w-6xl max-h-[92vh] overflow-y-auto' : wide ? 'max-w-4xl' : 'max-w-md'} rounded-2xl bg-[#0f1624] ring-1 ring-white/10 shadow-2xl shadow-black/50 p-6`}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            {icon}
            <h2 className="text-base font-semibold text-white">{title}</h2>
          </div>
          <button onClick={onClose} className="w-8 h-8 flex items-center justify-center rounded-lg text-slate-400 hover:text-white hover:bg-white/5" aria-label="Close">
            <X className="w-4 h-4" />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

// Override with VITE_API_BASE (e.g. http://localhost:8001) to point the dashboard at another backend.
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const WS_URL = API_BASE.replace(/^http/, "ws") + "/ws";

export default function App() {
  const [connected, setConnected] = useState(false);
  const [telemetry, setTelemetry] = useState({
    is_trading_active: false,
    account: { equity: 100000, cash: 100000, buying_power: 200000 },
    budget: { allocated_capital: 10000, hard_cap: 10000, committed_capital: 0, total_position_exposure: 0, remaining_budget: 10000, utilization_pct: 0 },
    daily_pnl: { realized_pnl: 0, unrealized_pnl: 0, net_pnl: 0, return_pct: 0, closed_trades: 0 },
    positions: {},
    watchlist: ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN"],
    latest_prices: {},
    quant_metrics: {},
    sentiment: {},
    recent_decisions: [],
    aggregated_trends: {},
    market_clock: { is_us_market_open: false, is_eu_market_open: false, next_us_open: "09:30 AM EST" },
    recent_trades: [],
    logs: [],
    risk: {
      risk_factor: 4, risk_label: "Balanced (default)",
      open_positions: 0, max_positions: 5,
      total_notional: 0, total_risk_to_stops: 0, total_risk_pct_of_budget: 0,
      realized_pnl_today: 0, daily_loss_pct: 0, daily_loss_limit_pct: 3,
      drawdown_pct: 0, drawdown_limit_pct: 10, halt_reason: null,
      positions: [], next_trade_preview: null
    }
  });

  const [newSymbol, setNewSymbol] = useState("");
  const [isSyncingTrends, setIsSyncingTrends] = useState(false);
  const [experts, setExperts] = useState([]);
  
  // Interactive Trading Budget Controls
  const [isEditingBudget, setIsEditingBudget] = useState(false);
  const [budgetInput, setBudgetInput] = useState("10000");
  const [isSavingBudget, setIsSavingBudget] = useState(false);

  // Portfolio risk dial (1-10). Held locally while dragging so the slider stays
  // responsive, then reconciled from telemetry once the backend confirms.
  const [riskFactor, setRiskFactor] = useState(4);
  const [riskLevels, setRiskLevels] = useState({});
  const [isSavingRisk, setIsSavingRisk] = useState(false);
  const [riskDirty, setRiskDirty] = useState(false);
  const [isRiskOpen, setIsRiskOpen] = useState(false);
  const [riskDraft, setRiskDraft] = useState(4);

  const wsRef = useRef(null);

  // --- Client-side latency: what the operator actually experiences ---
  // Frame timing lives in refs (no re-render per frame); the visible numbers
  // are pushed into state once per ping, every 2s.
  const [isLatencyOpen, setIsLatencyOpen] = useState(false);
  const [isCapitalOpen, setIsCapitalOpen] = useState(false);
  const [isDiscoveryOpen, setIsDiscoveryOpen] = useState(false);
  const [isManagerOpen, setIsManagerOpen] = useState(false);
  const [isRLOpen, setIsRLOpen] = useState(false);
  const [isPnlCalcOpen, setIsPnlCalcOpen] = useState(false);
  const [isMarketsOpen, setIsMarketsOpen] = useState(false);
  const [clientLatency, setClientLatency] = useState({ rttMs: null, rttP95: null, frameMs: null, ageMs: null });
  const pingSentRef = useRef(null);
  const rttSamplesRef = useRef([]);
  const lastFrameRef = useRef(null);
  const frameGapRef = useRef(null);
  const frameAgeRef = useRef(null);

  // Live portfolio risk from telemetry, and the limits for the dial position the
  // user is currently looking at (which may lead telemetry by one round-trip).
  const rp = telemetry.risk || {};
  const lvl = riskLevels[riskFactor] || riskLevels[String(riskFactor)] || {};

  const saveBudget = async (newVal) => {
    const val = Number(newVal || budgetInput);
    if (!val || val < 100) return;
    setIsSavingBudget(true);
    try {
      const res = await fetch(`${API_BASE}/api/budget`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ allocated_capital: val })
      });
      if (res.ok) {
        setIsEditingBudget(false);
      }
    } catch (e) {
      console.error("Failed to update budget:", e);
    } finally {
      setIsSavingBudget(false);
    }
  };

  const saveRiskFactor = async (val) => {
    const f = Math.max(1, Math.min(10, Number(val)));
    setRiskFactor(f);
    setRiskDirty(true);
    setIsSavingRisk(true);
    try {
      const res = await fetch(`${API_BASE}/api/risk-factor`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ risk_factor: f })
      });
      if (!res.ok) throw new Error(await res.text());
    } catch (e) {
      console.error("Failed to update risk factor:", e);
    } finally {
      setIsSavingRisk(false);
      // Let telemetry take over again shortly after the write lands
      setTimeout(() => setRiskDirty(false), 800);
    }
  };

  // Load the level table once so the UI can describe each notch
  useEffect(() => {
    fetch(`${API_BASE}/api/risk-factor`)
      .then(r => r.json())
      .then(d => { setRiskLevels(d.levels || {}); setRiskFactor(d.risk_factor || 4); })
      .catch(() => {});
  }, []);

  // Follow the backend unless the user is mid-adjustment
  useEffect(() => {
    if (!riskDirty && telemetry.risk?.risk_factor) {
      setRiskFactor(telemetry.risk.risk_factor);
    }
  }, [telemetry.risk?.risk_factor, riskDirty]);

  // Connect WebSocket
  useEffect(() => {
    let reconnectTimer;
    function connect() {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
      };

      ws.onmessage = (event) => {
        if (event.data === 'pong') {
          if (pingSentRef.current !== null) {
            const rtt = performance.now() - pingSentRef.current;
            pingSentRef.current = null;
            const samples = rttSamplesRef.current;
            samples.push(rtt);
            if (samples.length > 60) samples.shift();
            const sorted = [...samples].sort((a, b) => a - b);
            setClientLatency({
              rttMs: rtt,
              rttP95: sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))],
              frameMs: frameGapRef.current,
              ageMs: frameAgeRef.current,
            });
          }
          return;
        }
        try {
          const data = JSON.parse(event.data);
          const now = performance.now();
          if (lastFrameRef.current !== null) {
            // Exponentially smoothed frame gap; a rising value means the UI is starved
            const gap = now - lastFrameRef.current;
            frameGapRef.current = frameGapRef.current === null ? gap : frameGapRef.current * 0.8 + gap * 0.2;
          }
          lastFrameRef.current = now;
          if (data.server_time) frameAgeRef.current = Math.max(0, Date.now() - data.server_time * 1000);
          setDisplayTz(data.market_clock?.display_tz);
          setTelemetry(data);
        } catch (e) {
          console.error("WS Parse error", e);
        }
      };

      ws.onclose = () => {
        setConnected(false);
        reconnectTimer = setTimeout(connect, 1500);
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connect();

    // Round-trip probe over the telemetry socket (server answers 'ping' with 'pong')
    const pingTimer = setInterval(() => {
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        pingSentRef.current = performance.now();
        ws.send('ping');
      }
    }, 2000);

    // Fetch experts
    fetch(`${API_BASE}/api/experts`)
      .then(res => res.json())
      .then(data => setExperts(data.experts || []))
      .catch(() => {});

    return () => {
      clearTimeout(reconnectTimer);
      clearInterval(pingTimer);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const toggleTrading = async () => {
    try {
      await fetch(`${API_BASE}/api/toggle-trading`, { method: "POST" });
    } catch (e) {
      console.error(e);
    }
  };

  const triggerKillSwitch = async () => {
    if (window.confirm("EMERGENCY KILL SWITCH: Close all positions and stop trading immediately?")) {
      try {
        await fetch(`${API_BASE}/api/kill-switch`, { method: "POST" });
      } catch (e) {
        console.error(e);
      }
    }
  };

  const triggerMorningSync = async () => {
    setIsSyncingTrends(true);
    try {
      await fetch(`${API_BASE}/api/schedule/run-now?market=US`, { method: "POST" });
    } catch (e) {
      console.error(e);
    } finally {
      setIsSyncingTrends(false);
    }
  };

  const addWatchlist = async (e) => {
    e.preventDefault();
    if (!newSymbol) return;
    try {
      await fetch(`${API_BASE}/api/watchlist`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol: newSymbol })
      });
      setNewSymbol("");
    } catch (e) {
      console.error(e);
    }
  };

  const removeWatchlist = async (sym) => {
    try {
      await fetch(`${API_BASE}/api/watchlist/${sym}`, { method: "DELETE" });
    } catch (e) {
      console.error(e);
    }
  };

  const positionsList = Object.values(telemetry.positions || {}).map((p) => {
    const livePrice = telemetry.latest_prices?.[p.symbol]?.price || telemetry.prices?.[p.symbol]?.price || p.current_price || p.avg_entry_price;
    const qty = p.qty || 0;
    const avg = p.avg_entry_price || livePrice;
    const pnl = avg && qty ? (livePrice - avg) * qty : (p.unrealized_pl || 0);
    const pnlpc = avg ? (livePrice - avg) / avg : (p.unrealized_plpc || 0);
    const sentinel = telemetry.sentinel_bots?.[p.symbol];
    return {
      ...p,
      current_price: livePrice,
      unrealized_pl: pnl,
      unrealized_plpc: pnlpc,
      sentinel: sentinel,
      action: sentinel?.action || p.action || "HOLD",
      action_space: sentinel?.action_space || p.action_space || ["BUY", "HOLD", "SELL", "CLOSE"],
      buy_prob: sentinel?.buy_prob !== undefined ? sentinel.buy_prob : (p.buy_prob !== undefined ? p.buy_prob : 0.15),
      hold_prob: sentinel?.hold_prob !== undefined ? sentinel.hold_prob : (p.hold_prob !== undefined ? p.hold_prob : 0.70),
      sell_prob: sentinel?.sell_prob !== undefined ? sentinel.sell_prob : (p.sell_prob !== undefined ? p.sell_prob : 0.10),
      close_prob: sentinel?.close_prob !== undefined ? sentinel.close_prob : (p.close_prob !== undefined ? p.close_prob : 0.05),
      stop_loss: sentinel?.stop_loss || p.stop_loss,
      take_profit: sentinel?.take_profit || p.take_profit,
      bot_thesis: sentinel?.thesis || p.bot_thesis
    };
  });
  const totalPnL = positionsList.reduce((acc, p) => acc + (p.unrealized_pl || 0), 0);
  const hardCap = telemetry.budget?.hard_cap ?? telemetry.budget?.allocated_capital ?? 0;
  const dailyPnl = telemetry.daily_pnl || {};
  const dailyNetPnl = Number(dailyPnl.net_pnl ?? ((rp.realized_pnl_today || 0) + totalPnL));
  const clock = telemetry.market_clock || {};

  return (
    <div className="min-h-screen bg-[#090d16] text-slate-100 flex flex-col font-sans">
      {/* Top Header */}
      <header className="sticky top-0 z-50 border-b border-white/5 bg-[#0b111c]/80 backdrop-blur-xl">
        <div className="px-6 h-16 flex items-center justify-between gap-4">
          {/* Brand */}
          <div className="flex items-center gap-3 shrink-0">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-emerald-500 to-cyan-500 flex items-center justify-center shadow-lg shadow-emerald-500/20 shrink-0">
              <Zap className="w-5 h-5 text-slate-950 fill-current" />
            </div>
            <div className="min-w-0">
              <h1 className="font-semibold text-lg leading-tight tracking-tight text-white">TradeFlow</h1>
              <p className="hidden md:block text-[11px] text-slate-500 truncate">{['US stocks', 'RL policy (PPO)', ...(telemetry.source_health?.real_sources_available || []), 'Alpaca Paper'].join(' · ')}</p>
            </div>
          </div>

          {/* Market status: one pill that says whether stocks are trading */}
          {(() => {
            const stocksLive = clock.is_us_market_open || clock.is_eu_market_open;
            const venues = [clock.is_us_market_open && 'US', clock.is_eu_market_open && 'EU'].filter(Boolean).join(' + ');
            const stocksOn = telemetry.markets?.markets?.stocks !== false;
            return (
              <div
                className="hidden min-[2100px]:flex items-center rounded-full bg-white/[0.03] ring-1 ring-white/10 text-xs font-medium"
                title={stocksLive ? `Stock markets open: ${venues}` : `Stock markets closed · next US open: ${clock.next_us_open_local || clock.next_us_open || '—'}`}
              >
                <button type="button" onClick={() => setIsMarketsOpen(true)} className="flex items-center gap-2 pl-3 pr-3 py-1.5 rounded-full hover:bg-white/[0.05] transition-colors">
                  <span className={`w-2 h-2 rounded-full ${!stocksOn ? 'bg-rose-500' : stocksLive ? 'bg-emerald-400 animate-pulse shadow-[0_0_8px] shadow-emerald-400' : 'bg-slate-500'}`} />
                  <span className="text-slate-400">Stocks</span>
                  <span className={stocksLive ? 'text-emerald-300 font-semibold' : 'text-slate-300'}>
                    {stocksLive ? `LIVE${venues ? ` · ${venues}` : ''}` : 'CLOSED'}
                  </span>
                  {!stocksOn && <span className="text-rose-300 font-semibold">· OFF</span>}
                </button>
              </div>
            );
          })()}

          {/* Your clock, and when the US session opens or closes in your timezone */}
          <div className="hidden sm:flex flex-col items-end leading-tight shrink-0" title={`Times shown in ${clock.display_tz || 'Europe/Paris'}; New York: ${clock.current_time_ny || ''}`}>
            <span className="font-mono text-sm text-slate-200">{clock.current_time_local || ''}</span>
            <span className="text-[10px] text-slate-500">
              {clock.us_session === 'regular'
                ? `US open · closes ${clock.next_us_close_local || ''}`
                : clock.us_session === 'pre' ? `US pre-market · opens ${clock.next_us_open_local || ''}`
                : `US opens ${clock.next_us_open_local || '—'}`}
            </span>
          </div>

          {/* Right cluster: stat chips + actions */}
          <div className="flex items-center gap-2">
            {/* Main Broker Equity (Locked outside bot budget) */}
            <div className="hidden min-[2100px]:flex flex-col items-end px-3 py-1 rounded-xl bg-white/[0.02] ring-1 ring-white/5" title="Main broker equity is locked outside bot budget. Bot cannot touch outside the cap.">
              <span className="text-[10px] uppercase tracking-wider text-slate-500 flex items-center gap-1">
                <Lock className="w-2.5 h-2.5 text-amber-400" />
                Broker equity
              </span>
              <span className="text-sm font-mono text-slate-300">
                ${(telemetry.budget?.locked_broker_equity ?? Math.max(0, (telemetry.account?.equity || 100000) - hardCap)).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                <span className="text-[10px] text-amber-400/90 ml-1 font-sans">Locked</span>
              </span>
            </div>

            {/* Bot Equity */}
            <div className="hidden 2xl:flex flex-col items-end px-3 py-1 rounded-xl bg-cyan-400/[0.03] ring-1 ring-cyan-400/20" title="Bot account equity: allocated budget + realized & unrealized bot PnL.">
              <span className="text-[10px] uppercase tracking-wider text-cyan-400/90 font-medium">Bot Equity</span>
              <span className="text-sm font-mono font-semibold text-white">
                ${Number(telemetry.budget?.bot_equity ?? hardCap).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
            </div>

            {/* Latency chip -> opens latency breakdown */}
            <LatencyChip latency={telemetry.latency} client={clientLatency} onClick={() => setIsLatencyOpen(true)} />

            {/* Capital mode chip (classic / stair) -> opens capital plan */}
            <CapitalModeChip plan={telemetry.capital_plan} onClick={() => setIsCapitalOpen(true)} />

            {/* Discovery chip -> candidates, diversification and portfolio risk */}
            <DiscoveryChip brief={telemetry.diversification} onClick={() => setIsDiscoveryOpen(true)} />

            {/* RL chip -> the PPO policy: mode, promotion gate, results vs baselines */}
            <RLChip apiBase={API_BASE} onClick={() => setIsRLOpen(true)} />

            {/* Manager chip -> what the portfolio manager is deploying, and why budget is idle */}
            <ManagerChip manager={telemetry.manager} onClick={() => setIsManagerOpen(true)} />

            {/* Budget chip -> opens modal */}
            <button
              onClick={() => {
                // In stair mode the ladder owns the budget; send the user to it instead
                if (telemetry.capital_plan?.mode === 'stair') { setIsCapitalOpen(true); return; }
                setBudgetInput(String(telemetry.budget?.allocated_capital || 10000)); setIsEditingBudget(true);
              }}
              className="group flex items-center gap-2.5 rounded-xl px-3 py-1.5 bg-white/[0.03] ring-1 ring-white/10 hover:ring-cyan-400/50 hover:bg-cyan-400/5 transition-all text-left"
              title="Edit the capital the bot is allowed to trade"
            >
              <Wallet className="w-4 h-4 text-cyan-400" />
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-slate-500 leading-none">Hard cap</span>
                <span className="text-sm font-mono font-semibold text-white leading-tight">
                  ${hardCap.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </span>
                <div className="w-full h-0.5 bg-white/10 rounded-full overflow-hidden">
                  <div style={{ width: `${Math.min(100, telemetry.budget?.utilization_pct || 0)}%` }} className="h-full bg-cyan-400 transition-all duration-300" />
                </div>
              </div>
              <Edit3 className="w-3 h-3 text-slate-500 group-hover:text-cyan-300" />
            </button>

            {/* Risk chip -> opens modal */}
            <button
              onClick={() => { setRiskDraft(riskFactor); setIsRiskOpen(true); }}
              className="group flex items-center gap-2.5 rounded-xl px-3 py-1.5 bg-white/[0.03] ring-1 ring-white/10 hover:ring-amber-400/50 hover:bg-amber-400/5 transition-all text-left"
              title="Adjust portfolio-wide risk factor"
            >
              <Shield className={`w-4 h-4 ${rp.halt_reason ? 'text-rose-400' : 'text-amber-400'}`} />
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-slate-500 leading-none">Risk</span>
                <span className="text-sm font-semibold leading-tight">
                  <span className={`font-mono ${riskTone(riskFactor)}`}>{riskFactor}</span>
                  <span className="text-slate-500 font-mono">/10</span>
                  <span className="hidden 2xl:inline text-[11px] text-slate-400 font-normal ml-1.5">{rp.risk_label || ''}</span>
                </span>
              </div>
              {rp.halt_reason && <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse" title={rp.halt_reason} />}
              <Sliders className="w-3 h-3 text-slate-500 group-hover:text-amber-300" />
            </button>

            {/* Daily PnL chip -> opens Daily Profit and Loss Calculator */}
            <DailyPnlChip dailyPnl={dailyPnl} onClick={() => setIsPnlCalcOpen(true)} />

            <div className="hidden 2xl:flex flex-col items-end px-3">
              <span className="text-[10px] uppercase tracking-wider text-slate-500">Open PnL</span>
              <span className={`text-sm font-mono font-semibold ${totalPnL >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {totalPnL >= 0 ? '+' : ''}${totalPnL.toFixed(2)}
              </span>
            </div>

            <span className="w-px h-8 bg-white/10 mx-1" />

            {/* Actions */}
            <button
              onClick={triggerMorningSync}
              disabled={isSyncingTrends}
              className="w-9 h-9 flex items-center justify-center rounded-xl text-cyan-400 bg-white/[0.03] ring-1 ring-white/10 hover:ring-cyan-400/50 hover:bg-cyan-400/10 transition-all"
              title="Morning Sync: SEC Form 4 + eToro + StockTwits (when reachable), news scored with Laya"
            >
              <RefreshCw className={`w-4 h-4 ${isSyncingTrends ? 'animate-spin' : ''}`} />
            </button>

            <button
              onClick={toggleTrading}
              className={`flex items-center gap-2 h-9 px-3 rounded-xl text-xs font-semibold ring-1 transition-all ${
                telemetry.is_trading_active
                  ? 'bg-emerald-500/10 text-emerald-300 ring-emerald-500/30 hover:bg-emerald-500/20'
                  : 'bg-amber-500/10 text-amber-300 ring-amber-500/30 hover:bg-amber-500/20'
              }`}
              title={telemetry.is_trading_active ? 'Pause the bot' : 'Resume the bot'}
            >
              {telemetry.is_trading_active ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              <span>{telemetry.is_trading_active ? 'Active' : 'Paused'}</span>
            </button>

            <button
              onClick={triggerKillSwitch}
              className="flex items-center gap-1.5 h-9 px-3 rounded-xl text-xs font-bold bg-rose-600/15 text-rose-300 ring-1 ring-rose-500/40 hover:bg-rose-600 hover:text-white transition-all"
              title="Close all positions and halt trading immediately"
            >
              <AlertOctagon className="w-4 h-4" />
              <span className="hidden md:inline">Kill</span>
            </button>

            <span
              className={`ml-1 w-2 h-2 rounded-full ${connected ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'}`}
              title={connected ? 'Live feed connected (4Hz)' : 'Live feed disconnected'}
            />
          </div>
        </div>
      </header>

      {/* Latency modal */}
      {isLatencyOpen && (
        <Modal
          wide
          title="Engine latency"
          icon={<Zap className="w-4 h-4 text-violet-400" />}
          onClose={() => setIsLatencyOpen(false)}
        >
          <LatencyPanel latency={telemetry.latency} client={clientLatency} analysisStatus={telemetry.analysis_status} />
        </Modal>
      )}

      {/* Capital plan modal */}
      {isCapitalOpen && (
        <Modal wide title="Capital mode" icon={<Wallet className="w-4 h-4 text-emerald-400" />} onClose={() => setIsCapitalOpen(false)}>
          <CapitalPlanPanel plan={telemetry.capital_plan} apiBase={API_BASE} cash={telemetry.account?.cash} onDone={() => setIsCapitalOpen(false)} />
        </Modal>
      )}

      {/* Discovery & diversification modal */}
      {isDiscoveryOpen && (
        <Modal xl title="Discovery & diversification" icon={<Globe className="w-4 h-4 text-cyan-400" />} onClose={() => setIsDiscoveryOpen(false)}>
          <DiscoveryPanel apiBase={API_BASE} positions={telemetry.positions} />
        </Modal>
      )}

      {/* Portfolio manager modal */}
      {isManagerOpen && (
        <Modal xl title="Agent fleet" icon={<Bot className="w-4 h-4 text-cyan-400" />} onClose={() => setIsManagerOpen(false)}>
          <ManagerPanel manager={telemetry.manager} positions={telemetry.positions} fleet={telemetry.fleet} />
        </Modal>
      )}

      {/* RL policy modal */}
      {isRLOpen && (
        <Modal xl title="RL policy (PPO)" icon={<BrainCircuit className="w-4 h-4 text-cyan-400" />} onClose={() => setIsRLOpen(false)}>
          <RLPanel apiBase={API_BASE} />
        </Modal>
      )}

      {/* Daily Profit and Loss Calculator Modal */}
      <MarketsModal
        isOpen={isMarketsOpen}
        onClose={() => setIsMarketsOpen(false)}
        markets={telemetry.markets}
        positions={telemetry.positions}
        apiBase={API_BASE}
        onChange={(markets) => setTelemetry((t) => ({ ...t, markets }))}
      />
      <DailyPnlCalculatorModal
        isOpen={isPnlCalcOpen}
        onClose={() => setIsPnlCalcOpen(false)}
        telemetry={telemetry}
        apiBase={API_BASE}
      />

      {/* Budget modal */}
      {isEditingBudget && (
        <Modal onClose={() => setIsEditingBudget(false)} title="Bot budget cap" icon={<Wallet className="w-4 h-4 text-cyan-400" />}>
          <p className="text-xs text-slate-400 mb-3">The hard maximum capital the bot may commit across all positions and pending buys.</p>

          {/* Locked Main Broker Equity Guarantee Banner */}
          <div className="flex items-center justify-between p-2.5 mb-4 rounded-xl bg-amber-500/[0.05] ring-1 ring-amber-500/20 text-xs">
            <span className="flex items-center gap-1.5 text-slate-300">
              <Lock className="w-3.5 h-3.5 text-amber-400 shrink-0" />
              Main broker equity locked: <strong className="text-white">${(telemetry.budget?.locked_broker_equity ?? Math.max(0, (telemetry.account?.equity || 100000) - hardCap)).toLocaleString(undefined, { maximumFractionDigits: 0 })}</strong>
            </span>
            <span className="text-[10px] text-amber-300 font-mono px-2 py-0.5 rounded bg-amber-400/10">Protected</span>
          </div>

          <div className="grid grid-cols-3 gap-2 mb-5 text-center">
            {[
              ['Hard cap', `$${hardCap.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, 'text-white'],
              ['Committed', `$${(telemetry.budget?.committed_capital || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`, 'text-cyan-300'],
              ['Utilization', `${telemetry.budget?.utilization_pct || 0}%`, 'text-slate-200'],
            ].map(([k, v, c]) => (
              <div key={k} className="rounded-xl bg-white/[0.03] ring-1 ring-white/10 py-2">
                <div className="text-[10px] uppercase tracking-wider text-slate-500">{k}</div>
                <div className={`font-mono text-sm font-semibold ${c}`}>{v}</div>
              </div>
            ))}
          </div>

          <div className="rounded-xl bg-white/[0.03] ring-1 ring-white/10 p-3 mb-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] uppercase tracking-wider text-slate-500">Today’s bot P&amp;L</span>
              <div className="flex items-center gap-2">
                <span className={`font-mono text-sm font-semibold ${dailyNetPnl >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
                  {dailyNetPnl >= 0 ? '+' : ''}${dailyNetPnl.toFixed(2)}
                </span>
                <button
                  type="button"
                  onClick={() => { setIsEditingBudget(false); setIsPnlCalcOpen(true); }}
                  className="text-[10px] text-emerald-400 hover:text-emerald-300 underline font-medium ml-1"
                >
                  Calculator 🧮
                </button>
              </div>
            </div>
            <div className="grid grid-cols-3 gap-2 text-center">
              {[
                ['Realised', dailyPnl.realized_pnl || 0],
                ['Open', dailyPnl.unrealized_pnl || 0],
                ['Closed', `${dailyPnl.closed_trades || 0} trades`],
              ].map(([label, value]) => {
                const numeric = typeof value === 'number';
                return (
                  <div key={label}>
                    <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
                    <div className={`font-mono text-xs ${numeric ? (value >= 0 ? 'text-emerald-300' : 'text-rose-300') : 'text-slate-200'}`}>
                      {numeric ? `${value >= 0 ? '+' : ''}$${value.toFixed(2)}` : value}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <form onSubmit={(e) => { e.preventDefault(); saveBudget(budgetInput); }}>
            <label className="text-[11px] uppercase tracking-wider text-slate-500">New cap</label>
            <div className="mt-1.5 flex items-center rounded-xl bg-slate-950 ring-1 ring-white/10 focus-within:ring-cyan-400/60 px-3">
              <span className="text-slate-500 font-mono">$</span>
              <input
                autoFocus
                type="number"
                min="100"
                value={budgetInput}
                onChange={(e) => setBudgetInput(e.target.value)}
                className="flex-1 bg-transparent px-2 py-2.5 font-mono text-lg text-white focus:outline-none"
                placeholder="10000"
              />
            </div>
            <div className="flex gap-2 mt-2">
              {[2000, 5000, 10000, 25000].map(amt => (
                <button
                  type="button"
                  key={amt}
                  onClick={() => setBudgetInput(String(amt))}
                  className={`flex-1 text-xs font-mono py-1.5 rounded-lg ring-1 transition-all ${
                    Number(budgetInput) === amt ? 'bg-cyan-400/15 text-cyan-200 ring-cyan-400/50' : 'bg-white/[0.03] text-slate-300 ring-white/10 hover:ring-white/20'
                  }`}
                >
                  ${amt / 1000}k
                </button>
              ))}
            </div>
            {Number(budgetInput) > 0 && Number(budgetInput) < 100 && (
              <p className="text-[11px] text-rose-300 mt-2">Minimum budget is $100.</p>
            )}

            <div className="flex justify-end gap-2 mt-6">
              <button type="button" onClick={() => setIsEditingBudget(false)} className="px-4 py-2 rounded-xl text-sm text-slate-300 hover:bg-white/5">
                Cancel
              </button>
              <button
                type="submit"
                disabled={isSavingBudget || !(Number(budgetInput) >= 100)}
                className="px-4 py-2 rounded-xl text-sm font-semibold bg-cyan-400 text-slate-950 hover:bg-cyan-300 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {isSavingBudget ? 'Saving…' : 'Save budget'}
              </button>
            </div>
          </form>
        </Modal>
      )}

      {/* Risk modal */}
      {isRiskOpen && (() => {
        const d = riskLevels[riskDraft] || riskLevels[String(riskDraft)] || {};
        const changed = riskDraft !== riskFactor;
        return (
          <Modal onClose={() => setIsRiskOpen(false)} title="Portfolio risk" icon={<Shield className="w-4 h-4 text-amber-400" />}>
            <p className="text-xs text-slate-400 mb-4">Scales position size, position count, loss limits and entry strictness across the whole portfolio.</p>

            <div className="flex items-end justify-between mb-2">
              <div>
                <div className={`text-4xl font-mono font-bold ${riskTone(riskDraft)}`}>
                  {riskDraft}<span className="text-lg text-slate-600">/10</span>
                </div>
                <div className="text-sm text-slate-300">{d.label || ''}</div>
              </div>
              {changed && <span className="text-[11px] text-amber-300 font-mono">was {riskFactor}</span>}
            </div>

            <input
              type="range"
              min="1" max="10" step="1"
              value={riskDraft}
              onChange={(e) => setRiskDraft(Number(e.target.value))}
              className="w-full accent-amber-400 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-1">
              <span>1 · Preserve</span>
              <span className="text-cyan-400">4 · Default</span>
              <span>10 · Aggressive</span>
            </div>

            <div className="grid grid-cols-2 gap-2 mt-5">
              {[
                ['Risk / trade', `${d.risk_per_trade_pct ?? '-'}% of budget`],
                ['Max position', `${d.max_position_notional_pct ?? '-'}%`],
                ['Positions', `${rp.open_positions ?? 0} / ${d.max_concurrent_positions ?? '-'}`],
                ['Daily halt', `${Math.max(rp.daily_loss_pct || 0, rp.broker_day_loss_pct || 0).toFixed(2)}% / ${d.max_daily_loss_pct ?? '-'}%`],
                ['Broker account today', `${(rp.broker_day_loss_pct || 0) > 0 ? '-' : ''}${(rp.broker_day_loss_pct || 0).toFixed(2)}% of budget`],
                ['Entry bar', `buy_prob ≥ ${d.min_buy_prob ?? '-'}`],
                ['At risk now', `$${(rp.total_risk_to_stops || 0).toFixed(2)} (${(rp.total_risk_pct_of_budget || 0).toFixed(2)}%)`,
                  (rp.total_risk_pct_of_budget || 0) > (d.max_daily_loss_pct || 3) ? 'text-rose-300' : 'text-emerald-300'],
              ].map(([k, v, c]) => (
                <div key={k} className="rounded-xl bg-white/[0.03] ring-1 ring-white/10 px-3 py-2">
                  <div className="text-[10px] uppercase tracking-wider text-slate-500">{k}</div>
                  <div className={`font-mono text-sm ${c || 'text-slate-200'}`}>{v}</div>
                </div>
              ))}
            </div>

            {rp.next_trade_preview && !changed && (
              <div className="mt-3 text-[11px] font-mono text-slate-400 leading-relaxed">
                Next trade <span className="text-cyan-300">${(rp.next_trade_preview.position_notional || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}</span> notional,
                risking <span className="text-rose-300">${(rp.next_trade_preview.actual_risk_dollars || 0).toFixed(2)}</span> @ {(rp.next_trade_preview.stop_pct || 0).toFixed(2)}% stop · R:R <span className="text-emerald-300">{rp.next_trade_preview.reward_risk_ratio}</span>
              </div>
            )}

            {rp.halt_reason && (
              <div className="mt-3 rounded-xl bg-rose-500/10 ring-1 ring-rose-500/30 px-3 py-2 text-xs text-rose-200">⚠ {rp.halt_reason}</div>
            )}

            <div className="flex justify-end gap-2 mt-6">
              <button onClick={() => setIsRiskOpen(false)} className="px-4 py-2 rounded-xl text-sm text-slate-300 hover:bg-white/5">
                Cancel
              </button>
              <button
                onClick={async () => { await saveRiskFactor(riskDraft); setIsRiskOpen(false); }}
                disabled={!changed || isSavingRisk}
                className="px-4 py-2 rounded-xl text-sm font-semibold bg-amber-400 text-slate-950 hover:bg-amber-300 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {isSavingRisk ? 'Applying…' : 'Apply'}
              </button>
            </div>
          </Modal>
        );
      })()}

      {/* Main Grid Body */}
      <main className="flex-1 p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Watchlist & Smart-Money Intelligence (7 Cols) */}
        <section className="lg:col-span-7 flex flex-col space-y-6">

          {/* Scout: our own hourly ranking of what to watch today, and the watcher's live read */}
          <ScoutPanel apiBase={API_BASE} />

          {/* Smart-money consensus: only the sources that actually answered this cycle */}
          <div className="bg-[#0f172a]/70 border border-cyan-900/40 rounded-2xl p-5 shadow-xl relative overflow-hidden">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Compass className="w-5 h-5 text-cyan-400" />
                <h2 className="font-semibold text-base text-slate-100">
                  Smart money
                  <span className="ml-2 text-xs font-normal text-slate-500">
                    {(telemetry.source_health?.real_sources_available || []).join(' + ') || 'no source reachable'}
                    {Object.entries(telemetry.source_health?.sources || {})
                      .filter(([, v]) => !v.available)
                      .map(([k]) => ` · ${k} unavailable`).join('')}
                  </span>
                </h2>
              </div>
              <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/30 font-mono">
                {(telemetry.smart_money?.rows || []).filter((r) => r.verdict === 'BUY' && r.tradable).length} TRADING
              </span>
            </div>

            <SmartMoneyPanel data={telemetry.smart_money}
              sources={(telemetry.source_health?.real_sources_available || []).join(' + ')} />
          </div>
          
          {/* Card: Watchlist & Sub-Second Evaluation */}
          <div className="bg-[#0f172a]/70 border border-slate-800 rounded-2xl p-5 shadow-xl">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <div className="flex items-center space-x-2">
                <BrainCircuit className="w-5 h-5 text-cyan-400" />
                <h2 className="font-semibold text-base text-slate-100">Dynamic Watchlist & Quant Matrix</h2>
              </div>

              {/* Add ticker form */}
              <form onSubmit={addWatchlist} className="flex items-center space-x-2">
                <input
                  type="text"
                  placeholder="ADD TICKER (e.g. AMD)"
                  value={newSymbol}
                  onChange={(e) => setNewSymbol(e.target.value)}
                  className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1 text-xs uppercase font-mono text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                />
                <button
                  type="submit"
                  className="bg-cyan-500 hover:bg-cyan-400 text-slate-950 p-1.5 rounded-lg transition-colors"
                >
                  <Plus className="w-3.5 h-3.5" />
                </button>
              </form>
            </div>

            {/* Watchlist Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 font-medium">
                    <th className="pb-2.5">SYMBOL</th>
                    <th className="pb-2.5">PRICE</th>
                    <th className="pb-2.5">LAYA SENTIMENT</th>
                    <th className="pb-2.5">QUANT (RSI / EMA)</th>
                    <th className="pb-2.5">TRIGGER</th>
                    <th className="pb-2.5 text-right">ACTION</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {(telemetry.watchlist || [])
                    .map((sym) => {
                    const tick = telemetry.latest_prices?.[sym];
                    const quant = telemetry.quant_metrics?.[sym];
                    const sentiment = telemetry.sentiment?.[sym];
                    const isHolding = !!telemetry.positions?.[sym];

                    const posProb = sentiment?.pos_prob ?? 0.5;
                    const negProb = sentiment?.neg_prob ?? 0.5;

                    return (
                      <tr key={sym} className="hover:bg-slate-800/20 transition-colors">
                        <td className="py-3 font-bold text-slate-200 flex items-center space-x-1.5">
                          <span>{sym}</span>
                          {isHolding && (
                            <span className="text-[9px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-1 rounded">
                              HELD
                            </span>
                          )}
                        </td>
                        <td className="py-3 font-semibold text-slate-100">
                          {tick ? `$${tick.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '--'}
                        </td>
                        <td className="py-3">
                          <div className="flex flex-col space-y-1 w-28">
                            <div className="flex justify-between text-[10px] text-slate-400">
                              <span className="text-emerald-400 font-bold">{(posProb * 100).toFixed(0)}% Bull</span>
                              <span className="text-rose-400 font-bold">{(negProb * 100).toFixed(0)}% Bear</span>
                            </div>
                            <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden flex">
                              <div style={{ width: `${posProb * 100}%` }} className="bg-emerald-500" />
                              <div style={{ width: `${negProb * 100}%` }} className="bg-rose-500" />
                            </div>
                          </div>
                        </td>
                        <td className="py-3">
                          <div className="flex items-center space-x-2">
                            <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                              quant?.rsi < 35 ? 'bg-emerald-500/20 text-emerald-400 font-bold' :
                              quant?.rsi > 70 ? 'bg-rose-500/20 text-rose-400 font-bold' :
                              'bg-slate-800 text-slate-300'
                            }`}>
                              RSI {quant?.rsi ? quant.rsi.toFixed(1) : '--'}
                            </span>
                            <span className="text-[10px] text-slate-400">
                              {quant?.ema_fast && quant?.ema_slow ? (
                                quant.ema_fast > quant.ema_slow ? (
                                  <span className="inline-flex items-center space-x-1 text-emerald-400 font-semibold">
                                    <CircleDot className="w-2.5 h-2.5 text-emerald-400" />
                                    <span>Bull EMA</span>
                                  </span>
                                ) : (
                                  <span className="inline-flex items-center space-x-1 text-rose-400 font-semibold">
                                    <CircleDot className="w-2.5 h-2.5 text-rose-400" />
                                    <span>Bear EMA</span>
                                  </span>
                                )
                              ) : '--'}
                            </span>
                          </div>
                        </td>
                        <td className="py-3">
                          {isHolding ? (
                            <span className="px-2 py-0.5 rounded text-[10px] bg-amber-500/10 text-amber-400 border border-amber-500/20 font-bold">
                              HOLDING
                            </span>
                          ) : posProb >= 0.65 ? (
                            <span className="px-2 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-bold animate-pulse">
                              READY BUY
                            </span>
                          ) : (
                            <span className="text-slate-500 text-[10px]">SCANNING</span>
                          )}
                        </td>
                        <td className="py-3 text-right">
                          <button
                            onClick={() => removeWatchlist(sym)}
                            className="text-slate-500 hover:text-rose-400 transition-colors p-1"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Quant Desk: off-process analytics (regime, council, Monte Carlo, pairs, portfolio risk) */}
          <QuantDeskCard
            analysis={telemetry.analysis}
            portfolio={telemetry.portfolio_analytics}
            status={telemetry.analysis_status}
            routing={telemetry.strategy_routing}
          />

        </section>

        {/* Right Column: Positions, Real-time Trades & Live Logs (5 Cols) */}
        <section className="lg:col-span-5 flex flex-col space-y-6">

          {/* Trade desk: observer, analyst and critic agents argue every entry before it opens */}
          <DeskPanel desk={telemetry.desk} apiBase={API_BASE} />
          
          {/* Active Positions Card */}
          <div className="bg-[#0f172a]/70 border border-slate-800 rounded-2xl p-5 shadow-xl">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Layers className="w-5 h-5 text-emerald-400" />
                <h3 className="font-semibold text-base text-slate-100">
                  Active Positions ({positionsList.length})
                </h3>
              </div>
              <span className="text-xs text-slate-400 font-mono">Max: {rp.max_positions ?? lvl.max_concurrent_positions ?? 5}</span>
            </div>

            {/* Capital Budget Utilization Bar */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-2.5 mb-3 text-xs font-mono">
              <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                <span>Hard-cap committed: <strong className="text-cyan-300">${(telemetry.budget?.committed_capital || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</strong> / ${hardCap.toLocaleString()}</span>
                <span className="text-emerald-400 font-bold">${(telemetry.budget?.remaining_budget || 10000).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} Avail</span>
              </div>
              <div className="w-full h-1.5 bg-slate-950 rounded-full overflow-hidden flex">
                <div 
                  style={{ width: `${Math.min(100, telemetry.budget?.utilization_pct || 0)}%` }} 
                  className={`transition-all duration-300 ${
                    (telemetry.budget?.utilization_pct || 0) > 85 ? 'bg-amber-400' : 'bg-cyan-400'
                  }`}
                />
              </div>
            </div>

            {positionsList.length === 0 ? (
              <div className="py-8 text-center text-slate-500 text-xs">
                No open positions. Quant bot is scanning market ticks for breakout triggers.
              </div>
            ) : (
              <div className="space-y-3 font-mono">
                {positionsList.map((pos) => {
                  const pnl = pos.unrealized_pl || 0;
                  const pnlColor = pnl >= 0 ? 'text-emerald-400' : 'text-rose-400';
                  
                  // Real-time model probabilities (Normalized 100% distribution across 4 actions)
                  const rawBP = pos.buy_prob !== undefined ? pos.buy_prob : 0.15;
                  const rawHP = pos.hold_prob !== undefined ? pos.hold_prob : 0.70;
                  const rawSP = pos.sell_prob !== undefined ? pos.sell_prob : 0.10;
                  const rawCP = pos.close_prob !== undefined ? pos.close_prob : 0.05;
                  const sumP = (rawBP + rawHP + rawSP + rawCP) || 1.0;
                  // CLOSE reads 100% only when the backend has fired the exit (close_prob = 1);
                  // the rounding residue goes to the largest of BUY/HOLD/SELL, never to CLOSE.
                  const closePct = rawCP >= 1 ? 100 : Math.min(99, Math.round((rawCP / sumP) * 100));
                  const pcts = [rawBP, rawHP, rawSP].map((p) => Math.round((p / sumP) * 100));
                  const topIdx = pcts.indexOf(Math.max(...pcts));
                  pcts[topIdx] = Math.max(0, pcts[topIdx] + 100 - closePct - (pcts[0] + pcts[1] + pcts[2]));
                  const [buyPct, holdPct, sellPct] = pcts;

                  return (
                    <div key={pos.symbol} className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 flex flex-col space-y-3 shadow-md">
                      {/* Micro-Agent Header: Dedicated Sentinel Bot for this position */}
                      <div className="flex items-center justify-between bg-slate-950/80 border border-emerald-500/25 rounded-lg px-2.5 py-1.5 font-mono text-[11px]">
                        <div className="flex items-center space-x-2">
                          <div className="relative flex items-center justify-center">
                            <Bot className="w-3.5 h-3.5 text-emerald-400" />
                            <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-emerald-400 rounded-full animate-ping" />
                          </div>
                          <span className="font-bold text-emerald-300">
                            {pos.sentinel?.bot_id || pos.bot_id || `BOT-${pos.symbol.replace('/', '')}`}
                          </span>
                          <span className="text-[9px] px-1.5 py-0.2 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded font-sans font-semibold">
                            DEDICATED SENTINEL
                          </span>
                          <span className={`text-[9px] px-1.5 py-0.2 rounded font-sans font-semibold border ${
                            pos.action === 'BUY' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' :
                            pos.action === 'CLOSE' ? 'bg-rose-500/10 text-rose-400 border-rose-500/20' :
                            pos.action === 'SELL' ? 'bg-amber-500/10 text-amber-400 border-amber-500/20' :
                            'bg-cyan-500/10 text-cyan-400 border-cyan-500/20'
                          }`}>
                            ACTION: {pos.action || 'HOLD'}
                          </span>
                        </div>
                        <div className="flex items-center space-x-2 text-[10px] text-slate-400">
                          <span className="flex items-center space-x-1">
                            <Activity className="w-3 h-3 text-cyan-400 animate-spin" />
                            <span>{pos.sentinel?.evaluations_count || 1} ticks</span>
                          </span>
                          {pos.sentinel?.highest_price && pos.sentinel.highest_price > pos.sentinel.entry_price * 1.008 && (
                            <span className="text-[9px] px-1.5 py-0.5 bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 rounded font-bold inline-flex items-center space-x-1">
                              <Lock className="w-2.5 h-2.5 text-cyan-300" />
                              <span>TRAILING SL</span>
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Top Row: Symbol, shares, entry price, live price, and real-time Open PnL */}
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="flex items-center space-x-2">
                            <span className="font-bold text-sm text-slate-100">{pos.symbol}</span>
                            <span className="text-xs text-slate-400 font-mono">{pos.qty} shares</span>
                            {pos.mode && (
                              <span className={`text-[9px] px-1.5 py-0.5 rounded font-sans font-semibold ${
                                pos.mode.includes('ALPACA') 
                                  ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' 
                                  : 'bg-purple-500/10 text-purple-400 border border-purple-500/20'
                              }`}>
                                {pos.mode.replace('_', ' ')}
                              </span>
                            )}
                          </div>
                          <div className="text-[11px] text-slate-400 mt-1 font-mono">
                            Avg: ${pos.avg_entry_price?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: (pos.avg_entry_price < 1 ? 6 : 2) })} → Now: <strong className="text-cyan-300 font-bold">${pos.current_price?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: (pos.current_price < 1 ? 6 : 2) })}</strong>
                            {pos.price_age_s >= 60 && (
                              <span
                                className="ml-2 text-[9px] px-1.5 py-0.5 rounded font-sans font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20"
                                title="The engine is watching this position every second; the market has printed no new price. Probabilities only move when price or news does."
                              >
                                no new price {pos.price_age_s >= 3600 ? `${(pos.price_age_s / 3600).toFixed(1)}h` : `${Math.floor(pos.price_age_s / 60)}m`}
                              </span>
                            )}
                          </div>
                          {pos.stop_loss && (
                            <div className="text-[10px] text-slate-400 mt-0.5 font-mono flex items-center space-x-2">
                              <span>SL: <strong className="text-rose-400">${pos.stop_loss}</strong></span>
                              <span>|</span>
                              <span>TP: <strong className="text-emerald-400">${pos.take_profit}</strong></span>
                              {pos.sentinel?.highest_price && (
                                <>
                                  <span>|</span>
                                  <span className="text-slate-500">High: ${pos.sentinel.highest_price.toFixed(2)}</span>
                                </>
                              )}
                            </div>
                          )}
                        </div>

                        <div className="text-right font-mono">
                          <div className="flex items-center justify-end space-x-1">
                            <span className={`text-base font-bold ${pnlColor}`}>
                              {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}
                            </span>
                          </div>
                          <div className="text-[11px]">
                            <span className={pnlColor}>
                              {pos.unrealized_plpc ? `${pos.unrealized_plpc >= 0 ? '+' : ''}${(pos.unrealized_plpc * 100).toFixed(2)}%` : '0.00%'}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Four Bars: BUY, HOLD, SELL, CLOSE Real-time Probabilities summing to 100% */}
                      <div className="pt-2 border-t border-slate-800/80">
                        <div className="text-[10px] text-slate-400 mb-1.5 flex items-center justify-between font-sans">
                          <span className="flex items-center space-x-1.5 font-semibold text-slate-300">
                            <BrainCircuit className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
                            <span>Action Policy Distribution</span>
                          </span>
                          <span className="text-[9px] text-slate-500 font-mono">100% Normalized [BUY, HOLD, SELL, CLOSE]</span>
                        </div>

                        {/* 100% Unified Composite Distribution Bar */}
                        <div className="w-full h-1.5 bg-slate-950 rounded-full overflow-hidden flex mb-2 border border-slate-800/60">
                          <div style={{ width: `${buyPct}%` }} className="h-full bg-emerald-400 transition-all duration-300" title={`BUY: ${buyPct}%`} />
                          <div style={{ width: `${holdPct}%` }} className="h-full bg-cyan-400 transition-all duration-300" title={`HOLD: ${holdPct}%`} />
                          <div style={{ width: `${sellPct}%` }} className="h-full bg-amber-400 transition-all duration-300" title={`SELL: ${sellPct}%`} />
                          <div style={{ width: `${closePct}%` }} className="h-full bg-rose-400 transition-all duration-300" title={`CLOSE: ${closePct}%`} />
                        </div>

                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px] font-mono">
                          {/* BUY Probability Bar */}
                          <div className={`bg-slate-950/70 border ${pos.action === 'BUY' ? 'border-emerald-500/50 ring-1 ring-emerald-500/30' : 'border-slate-800/80'} rounded-lg p-2 flex flex-col justify-between`}>
                            <div className="flex justify-between items-center mb-1.5">
                              <span className="text-emerald-400 font-bold flex items-center space-x-1">
                                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                                <span>BUY</span>
                              </span>
                              <span className="font-bold text-slate-200">{buyPct}%</span>
                            </div>
                            <div className="w-full h-1.5 bg-slate-800/80 rounded-full overflow-hidden">
                              <div 
                                style={{ width: `${buyPct}%` }}
                                className="h-full bg-emerald-400 transition-all duration-300 rounded-full shadow-[0_0_8px_rgba(52,211,153,0.5)]"
                              />
                            </div>
                            <span className="text-[9px] text-slate-500 mt-1 font-sans">Momentum Add</span>
                          </div>

                          {/* HOLD Probability Bar */}
                          <div className={`bg-slate-950/70 border ${pos.action === 'HOLD' ? 'border-cyan-500/50 ring-1 ring-cyan-500/30' : 'border-slate-800/80'} rounded-lg p-2 flex flex-col justify-between`}>
                            <div className="flex justify-between items-center mb-1.5">
                              <span className="text-cyan-400 font-bold flex items-center space-x-1">
                                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                                <span>HOLD</span>
                              </span>
                              <span className="font-bold text-slate-200">{holdPct}%</span>
                            </div>
                            <div className="w-full h-1.5 bg-slate-800/80 rounded-full overflow-hidden">
                              <div 
                                style={{ width: `${holdPct}%` }}
                                className="h-full bg-cyan-400 transition-all duration-300 rounded-full shadow-[0_0_8px_rgba(34,211,238,0.5)]"
                              />
                            </div>
                            <span className="text-[9px] text-slate-500 mt-1 font-sans">Maintain Position</span>
                          </div>

                          {/* SELL Probability Bar */}
                          <div className={`bg-slate-950/70 border ${pos.action === 'SELL' ? 'border-amber-500/50 ring-1 ring-amber-500/30' : 'border-slate-800/80'} rounded-lg p-2 flex flex-col justify-between`}>
                            <div className="flex justify-between items-center mb-1.5">
                              <span className="text-amber-400 font-bold flex items-center space-x-1">
                                <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
                                <span>SELL</span>
                              </span>
                              <span className="font-bold text-slate-200">{sellPct}%</span>
                            </div>
                            <div className="w-full h-1.5 bg-slate-800/80 rounded-full overflow-hidden">
                              <div 
                                style={{ width: `${sellPct}%` }}
                                className="h-full bg-amber-400 transition-all duration-300 rounded-full shadow-[0_0_8px_rgba(251,191,36,0.5)]"
                              />
                            </div>
                            <span className="text-[9px] text-slate-500 mt-1 font-sans">Not used (full exits only)</span>
                          </div>

                          {/* CLOSE Probability Bar */}
                          <div className={`bg-slate-950/70 border ${pos.action === 'CLOSE' ? 'border-rose-500/50 ring-1 ring-rose-500/30' : 'border-slate-800/80'} rounded-lg p-2 flex flex-col justify-between`}>
                            <div className="flex justify-between items-center mb-1.5">
                              <span className="text-rose-400 font-bold flex items-center space-x-1">
                                <span className="w-1.5 h-1.5 rounded-full bg-rose-400"></span>
                                <span>CLOSE</span>
                              </span>
                              <span className="font-bold text-slate-200">{closePct}%</span>
                            </div>
                            <div className="w-full h-1.5 bg-slate-800/80 rounded-full overflow-hidden">
                              <div 
                                style={{ width: `${closePct}%` }}
                                className="h-full bg-rose-400 transition-all duration-300 rounded-full shadow-[0_0_8px_rgba(244,63,94,0.5)]"
                              />
                            </div>
                            <span className="text-[9px] text-slate-500 mt-1 font-sans">Full Exit (SL/TP)</span>
                          </div>
                        </div>
                      </div>

                      {/* Bot Sizing Intelligence: HOLD Position Sizing & SELL Divestment Strategy */}
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[10px] font-mono">
                        {/* HOLD Sizing */}
                        <div className="bg-slate-950/70 border border-cyan-900/40 rounded-lg p-2 flex flex-col justify-between">
                          <div className="flex items-center justify-between text-slate-400 mb-1">
                            <span className="text-cyan-400 font-bold flex items-center space-x-1.5">
                              <DollarSign className="w-3.5 h-3.5 text-cyan-400" />
                              <span>HOLD SIZING</span>
                            </span>
                            <span className="text-cyan-300 font-bold">
                              ${(pos.invested_dollars || (pos.qty * pos.avg_entry_price) || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                            </span>
                          </div>
                          <div className="flex items-center justify-between text-[9px] text-slate-400 border-t border-slate-800/60 pt-1">
                            <span>Budget Allocation:</span>
                            <span className="text-cyan-300 font-semibold">{pos.allocated_pct || ((pos.invested_dollars / Math.max(1, hardCap)) * 100).toFixed(1)}%</span>
                          </div>
                          <div className="flex items-center justify-between text-[9px] text-slate-400 mt-0.5">
                            <span>Risk / Reward:</span>
                            <span>
                              <strong className="text-rose-400">-${(pos.dollar_risk || 0).toFixed(2)}</strong> / <strong className="text-emerald-400">+${(pos.dollar_reward || 0).toFixed(2)}</strong>
                            </span>
                          </div>
                        </div>

                        {/* SELL Sizing */}
                        <div className="bg-slate-950/70 border border-amber-900/40 rounded-lg p-2 flex flex-col justify-between">
                          <div className="flex items-center justify-between text-slate-400 mb-1">
                            <span className="text-amber-400 font-bold flex items-center space-x-1.5">
                              <TrendingDown className="w-3.5 h-3.5 text-amber-400" />
                              <span>SELL SIZING</span>
                            </span>
                            <span className="text-amber-300 font-bold">
                              {pos.sell_pct || 100}% ({pos.sell_qty || pos.qty}x)
                            </span>
                          </div>
                          <div className="text-[9px] text-slate-300 border-t border-slate-800/60 pt-1 leading-tight font-sans">
                            <span className="text-slate-400 font-mono">Plan: </span>
                            <span className="text-amber-200/90 font-medium">
                              {pos.sell_plan || `Liquidate 100% on SL $${pos.stop_loss} or TP $${pos.take_profit}`}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Sentinel Bot Directive & Live Thesis */}
                      <div className="bg-slate-950/60 border border-slate-800/70 rounded-lg p-2 text-[10px] text-slate-300 flex items-start space-x-2">
                        <span className="text-cyan-400 font-bold shrink-0 mt-0.5">DIRECTIVE:</span>
                        <span className="font-sans leading-relaxed text-slate-300">
                          {pos.bot_thesis || pos.sentinel?.thesis || `Assigned dedicated bot continuously supervising ${pos.symbol} order flow, Laya sentiment, and trailing stops for autonomous exit booking.`}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Recent Automated Orders Card */}
          <div className="bg-[#0f172a]/70 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Clock className="w-4 h-4 text-cyan-400" />
                <h3 className="font-semibold text-sm text-slate-200">Recent Automated Fills & Signals</h3>
              </div>
              <span className="text-[11px] text-slate-500 font-mono">
                {(telemetry.recent_trades || []).length} recorded
              </span>
            </div>

            <div className="space-y-2 max-h-[380px] overflow-y-auto pr-1 font-mono text-xs">
              {(telemetry.recent_trades || []).length === 0 ? (
                <div className="text-slate-500 text-xs py-10 text-center">No trades placed yet.</div>
              ) : (
                [...telemetry.recent_trades].reverse().map((tr, idx) => (
                  <div key={idx} className="bg-slate-900/90 border border-slate-800 rounded-lg p-2.5 flex items-center justify-between hover:border-slate-700/80 transition-colors">
                    <div className="flex items-center space-x-2">
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                        tr.side === 'BUY' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'
                      }`}>
                        {tr.side}
                      </span>
                      <span className="font-bold text-slate-200">{tr.symbol}</span>
                      <span className="text-slate-400 text-[11px]">{tr.qty}x @ ${tr.price?.toFixed(2)}</span>
                    </div>
                    {(() => {
                      const amount = Number(tr.qty || 0) * Number(tr.price || 0);
                      const buy = tr.side === 'BUY';
                      const cost = Number(tr.entry_price || 0) * Number(tr.qty || 0);
                      const pnlPct = tr.pnl !== undefined && cost > 0 ? (tr.pnl / cost) * 100 : null;
                      const usd = (v) => `$${Math.abs(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
                      return (
                        <div className="flex flex-col items-end text-right leading-tight shrink-0 pl-2" title={tr.exit_reason || ''}>
                          <span className={`text-[12px] font-bold ${buy ? 'text-emerald-300' : 'text-rose-300'}`}>
                            {usd(amount)} <span className="font-normal text-[10px] text-slate-400">{buy ? 'bought' : 'sold'}</span>
                            {tr.partial && <span className="ml-1 text-[9px] px-1 rounded bg-amber-500/15 text-amber-300">PARTIAL</span>}
                          </span>
                          <span className="text-[10px] text-slate-500">
                            {tr.time ? clockTime(tr.time) : ''}
                            {tr.pnl !== undefined ? (
                              <span className={`ml-1.5 font-bold ${tr.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                {tr.pnl >= 0 ? '+' : '-'}{usd(tr.pnl)}{pnlPct !== null ? ` (${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%)` : ''}
                              </span>
                            ) : (
                              <span className="ml-1.5">{tr.mode || 'ALPACA_PAPER'}</span>
                            )}
                          </span>
                        </div>
                      );
                    })()}
                  </div>
                ))
              )}
            </div>
          </div>

          {/* News ingestion: every headline, its entities and events, and how it was scored */}
          <NewsIngestPanel news={telemetry.news_log} />

          {/* Real-time Engine Event Logs */}
          <div className="bg-[#0f172a]/70 border border-slate-800 rounded-2xl p-4 shadow-xl">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Engine Micro-Logs</span>
              <span className="text-[10px] text-slate-500 font-mono">Live WebSocket</span>
            </div>
            <div className="h-44 overflow-y-auto font-mono text-[10px] space-y-1.5 bg-slate-950 p-2.5 rounded-lg border border-slate-900">
              {(telemetry.logs || []).slice(-20).reverse().map((log, i) => (
                <div key={i} className="flex space-x-2 leading-relaxed items-start">
                  <span className="text-slate-600 shrink-0">
                    {clockTime(log.timestamp)}
                  </span>
                  <span className={`shrink-0 font-bold ${
                    log.level === 'ORDER_FILLED' || log.level === 'SIGNAL' ? 'text-emerald-400' :
                    log.level === 'SENTIMENT' || log.level === 'AGGREGATOR' ? 'text-cyan-400' :
                    log.level === 'KILL_SWITCH' || log.level === 'ORDER_ERROR' ? 'text-rose-400' :
                    'text-slate-400'
                  }`}>
                    [{log.level}]
                  </span>
                  <span className="text-slate-300 break-words">{log.message}</span>
                </div>
              ))}
            </div>
          </div>

        </section>

      </main>
    </div>
  );
}
