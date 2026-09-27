import React from 'react';
import { HugeiconsIcon } from '@hugeicons/react';
import { FlashIcon, AiBrain01Icon } from '@hugeicons/core-free-icons';

const Zap = (props) => <HugeiconsIcon icon={FlashIcon} size="1em" {...props} />;
const Brain = (props) => <HugeiconsIcon icon={AiBrain01Icon} size="1em" {...props} />;

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

export const fmtUs = (us) => {
  if (us === null || us === undefined || Number.isNaN(us)) return '—';
  if (us < 1000) return `${us.toFixed(us < 10 ? 1 : 0)}µs`;
  if (us < 1e6) return `${(us / 1000).toFixed(us < 1e4 ? 2 : 1)}ms`;
  return `${(us / 1e6).toFixed(2)}s`;
};

// p95 against the stage's budget: comfortably inside, close, or over.
const budgetTone = (p95, budget) => {
  if (!budget) return 'text-slate-300';
  const r = p95 / budget;
  return r < 0.5 ? 'text-emerald-400' : r < 1 ? 'text-amber-400' : 'text-rose-400';
};
const budgetBar = (p95, budget) => {
  if (!budget) return 'bg-slate-500';
  const r = p95 / budget;
  return r < 0.5 ? 'bg-emerald-400' : r < 1 ? 'bg-amber-400' : 'bg-rose-400';
};

const GROUPS = [
  { key: 'hot', label: 'Hot path', hint: 'Tick arrival → decision. This is what the engine controls.' },
  { key: 'broker', label: 'Broker round-trip', hint: 'Order submit / close network time to Alpaca.' },
  { key: 'feed', label: 'Market data age', hint: 'How old prices are when they reach the engine (poll / bar delay).' },
  { key: 'loop', label: 'Event loop', hint: 'A ~0.5–1ms baseline is uvloop timer resolution; sustained lag above 2ms means something blocked the loop that handles ticks.' },
  { key: 'worker', label: 'Background worker', hint: 'Council, regime, Monte Carlo, pairs, risk — separate process, off the hot path.' },
];

const STAGE_LABELS = {
  tick_total: 'Tick → decision (total)',
  quant_matrix: 'Indicator matrix',
  decision: 'Strategy decision',
  sentinel_tick: 'Trade bot per tick',
  order_submit: 'Order submit',
  order_close: 'Position close',
  feed_crypto_poll: 'Crypto price poll',
  feed_stock_bar_age: 'Stock bar delivery',
  event_loop_lag: 'Event-loop lag',
  telemetry_build: 'Dashboard frame build',
  worker_snapshot: 'Snapshot copy (chunked)',
  worker_cycle: 'Analysis cycle',
};

// ---------------------------------------------------------------------------
// Header chip
// ---------------------------------------------------------------------------

export function LatencyChip({ latency, client, onClick }) {
  const tick = latency?.stages?.tick_total;
  const tone = tick ? budgetTone(tick.p95_us, tick.budget_us) : 'text-slate-500';
  return (
    <button
      onClick={onClick}
      className="group flex items-center gap-2.5 rounded-xl px-3 py-1.5 bg-white/[0.03] ring-1 ring-white/10 hover:ring-violet-400/50 hover:bg-violet-400/5 transition-all text-left"
      title="Engine latency: tick-to-decision p50 / p95, and dashboard round-trip"
    >
      <Zap className={`w-4 h-4 ${tone}`} />
      <div className="flex flex-col">
        <span className="text-[10px] uppercase tracking-wider text-slate-500 leading-none">Latency</span>
        <span className="text-sm font-mono font-semibold leading-tight">
          <span className={tone}>{tick ? fmtUs(tick.p50_us) : '—'}</span>
          <span className="text-slate-500 text-[11px]"> / p95 {tick ? fmtUs(tick.p95_us) : '—'}</span>
        </span>
      </div>
      {client?.rttMs !== null && client?.rttMs !== undefined && (
        <span className="hidden 2xl:inline text-[11px] font-mono text-slate-400">UI {client.rttMs.toFixed(0)}ms</span>
      )}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Latency modal body
// ---------------------------------------------------------------------------

export function LatencyPanel({ latency, client, analysisStatus }) {
  const stages = latency?.stages || {};
  return (
    <div className="space-y-4 max-h-[70vh] overflow-y-auto p-1">
      <p className="text-xs text-slate-400">
        Rolling window of the last ~2,000 samples per stage. Colours compare p95 with the stage budget.
        Recording a sample costs well under 1µs, so tracking adds no measurable latency.
      </p>

      {/* Client-side view: what the operator actually experiences */}
      <div className="grid grid-cols-3 gap-2">
        <Stat label="Dashboard RTT" value={client?.rttMs != null ? `${client.rttMs.toFixed(1)}ms` : '—'} sub={client?.rttP95 != null ? `p95 ${client.rttP95.toFixed(1)}ms` : 'ws ping/pong'} />
        <Stat label="Frame interval" value={client?.frameMs != null ? `${client.frameMs.toFixed(0)}ms` : '—'} sub="target 250ms" />
        <Stat label="Frame age" value={client?.ageMs != null ? `${client.ageMs.toFixed(0)}ms` : '—'} sub="server→browser (clock-dependent)" />
      </div>

      {GROUPS.map((g) => {
        const rows = Object.entries(stages).filter(([, v]) => v.group === g.key);
        if (!rows.length) return null;
        return (
          <div key={g.key}>
            <div className="flex items-baseline justify-between mb-1">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">{g.label}</h3>
              <span className="text-[10px] text-slate-500">{g.hint}</span>
            </div>
            <div className="rounded-xl ring-1 ring-white/10 overflow-hidden">
              <table className="w-full text-xs font-mono">
                <thead className="bg-white/[0.03] text-slate-500">
                  <tr>
                    <th className="text-left font-normal px-3 py-1.5">Stage</th>
                    <th className="text-right font-normal px-2">p50</th>
                    <th className="text-right font-normal px-2">p95</th>
                    <th className="text-right font-normal px-2">p99</th>
                    <th className="text-right font-normal px-2">max</th>
                    <th className="text-left font-normal px-3 w-28">vs budget</th>
                    <th className="text-right font-normal px-3">n</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map(([k, v]) => (
                    <tr key={k} className="border-t border-white/5">
                      <td className="px-3 py-1.5 text-slate-300 font-sans">{STAGE_LABELS[k] || k}</td>
                      <td className="text-right px-2 text-slate-300">{fmtUs(v.p50_us)}</td>
                      <td className={`text-right px-2 font-semibold ${budgetTone(v.p95_us, v.budget_us)}`}>{fmtUs(v.p95_us)}</td>
                      <td className="text-right px-2 text-slate-400">{fmtUs(v.p99_us)}</td>
                      <td className="text-right px-2 text-slate-500">{fmtUs(v.max_us)}</td>
                      <td className="px-3">
                        {v.budget_us ? (
                          <div className="flex items-center gap-1.5" title={`budget ${fmtUs(v.budget_us)}`}>
                            <div className="flex-1 h-1 bg-white/10 rounded-full overflow-hidden">
                              <div className={`h-full ${budgetBar(v.p95_us, v.budget_us)}`} style={{ width: `${Math.min(100, (v.p95_us / v.budget_us) * 100)}%` }} />
                            </div>
                            <span className="text-[10px] text-slate-500">{fmtUs(v.budget_us)}</span>
                          </div>
                        ) : <span className="text-slate-600">—</span>}
                      </td>
                      <td className="text-right px-3 text-slate-500">{v.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      })}

      {analysisStatus && (
        <p className="text-[11px] text-slate-500">
          Worker: {analysisStatus.enabled ? 'running' : 'disabled'} · {analysisStatus.cycles} cycles ·
          last compute {analysisStatus.last_compute_ms ?? '—'}ms · results {analysisStatus.age_s ?? '—'}s old ·
          {' '}{analysisStatus.errors} errors{analysisStatus.last_error ? ` (${analysisStatus.last_error})` : ''}
        </p>
      )}
    </div>
  );
}

function Stat({ label, value, sub }) {
  return (
    <div className="rounded-xl bg-white/[0.03] ring-1 ring-white/10 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className="text-base font-mono font-semibold text-white">{value}</div>
      <div className="text-[10px] text-slate-500">{sub}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Quant desk card: worker results per symbol + portfolio analytics
// ---------------------------------------------------------------------------

const REGIME_STYLE = {
  trending_up: 'text-emerald-300 bg-emerald-400/10',
  trending_down: 'text-rose-300 bg-rose-400/10',
  ranging: 'text-sky-300 bg-sky-400/10',
  volatile: 'text-amber-300 bg-amber-400/10',
  mixed: 'text-slate-300 bg-white/5',
  unknown: 'text-slate-500 bg-white/5',
};
const VERDICT_STYLE = {
  support: 'text-emerald-400',
  oppose: 'text-rose-400',
  neutral: 'text-slate-300',
  insufficient: 'text-slate-500',
};
const pct = (v) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(0)}%`);
const money = (v) => (v === null || v === undefined ? '—' : `$${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`);
const num = (v, d = 2) => (v === null || v === undefined ? '—' : Number(v).toFixed(d));

export function QuantDeskCard({ analysis, portfolio, status, routing }) {
  const rows = Object.entries(analysis || {}).sort(([a], [b]) => a.localeCompare(b));
  const ledger = portfolio?.ledger || {};
  const risk = portfolio?.open_risk || {};
  return (
    <div className="rounded-2xl bg-[#0f1624] ring-1 ring-white/10 p-5">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-violet-400" />
          <h2 className="text-sm font-semibold text-white">Quant Desk</h2>
          <span className="text-[10px] text-slate-500">regime · council · Monte Carlo · pairs</span>
        </div>
        <span className="text-[10px] font-mono text-slate-500" title="Analysis runs in a separate process, off the tick path">
          {status?.enabled ? `worker ${status.last_compute_ms ?? '—'}ms · ${status.age_s ?? '—'}s old` : 'worker off'}
        </span>
      </div>

      {/* Portfolio analytics */}
      <div className="grid grid-cols-4 gap-2 mb-3">
        <Mini label="VaR95 1m" value={money(risk.var95_1m)} title={risk.method} />
        <Mini label="VaR95 1h" value={money(risk.var95_1h)} title={risk.method} />
        <Mini label="CVaR95 1h" value={money(risk.cvar95_1h)} title="Expected loss beyond VaR95 (normal approx.)" />
        <Mini label="Diversif." value={risk.diversification_ratio ? `${num(risk.diversification_ratio)}x` : '—'} title="Undiversified risk / portfolio risk" />
        <Mini label="Expectancy" value={ledger.expectancy_r != null ? `${num(ledger.expectancy_r)}R` : '—'} />
        <Mini label="Sharpe/trade" value={num(ledger.sharpe_per_trade)} title="Mean R / stdev R, per trade (not annualised)" />
        <Mini label="Profit factor" value={num(ledger.profit_factor)} />
        <Mini label="Max DD" value={money(ledger.max_drawdown)} title={`${ledger.n_trades || 0} closed trades`} />
      </div>

      <div className="max-h-80 overflow-y-auto rounded-xl ring-1 ring-white/10">
        <table className="w-full text-xs">
          <thead className="bg-white/[0.03] text-slate-500 sticky top-0">
            <tr>
              <th className="text-left font-normal px-3 py-1.5">Symbol</th>
              <th className="text-left font-normal px-2">Regime</th>
              <th className="text-left font-normal px-2">Council</th>
              <th className="text-left font-normal px-2">Best fit</th>
              <th className="text-right font-normal px-2" title="Monte Carlo: probability the take-profit is hit before the stop">P(TP)</th>
              <th className="text-right font-normal px-3" title="Cointegrated partner and spread z-score">Pair</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr><td colSpan={6} className="px-3 py-4 text-center text-slate-500">Warming up — analysis appears after the first worker cycle.</td></tr>
            )}
            {rows.map(([sym, a]) => (
              <tr key={sym} className="border-t border-white/5">
                <td className="px-3 py-1.5 font-mono text-slate-200">
                  {sym}
                  {routing?.[sym] === 'adaptive' && <span className="ml-1 text-[9px] text-violet-400" title="Routed to the adaptive selector">AD</span>}
                </td>
                <td className="px-2">
                  <span className={`px-1.5 py-0.5 rounded text-[10px] ${REGIME_STYLE[a.regime] || REGIME_STYLE.unknown}`}>{(a.regime || 'unknown').replace('_', ' ')}</span>
                </td>
                <td className={`px-2 font-mono ${VERDICT_STYLE[a.verdict] || 'text-slate-500'}`} title={`${a.n_bullish ?? 0} bull / ${a.n_bearish ?? 0} bear of ${a.n_voters ?? 0}`}>
                  {a.verdict || '—'} {a.consensus != null && <span className="text-slate-500">{a.consensus >= 0 ? '+' : ''}{num(a.consensus)}</span>}
                </td>
                <td className="px-2 text-slate-300" title={(a.top_candidates || []).join(', ')}>
                  {a.recommended || (a.top_candidates || [])[0] || '—'}
                </td>
                <td className={`text-right px-2 font-mono ${a.mc_p_tp_first == null ? 'text-slate-500' : a.mc_p_tp_first >= 0.4 ? 'text-emerald-400' : a.mc_p_tp_first >= 0.3 ? 'text-slate-300' : 'text-rose-400'}`}
                    title={a.mc_drift_edge != null ? `drift edge ${a.mc_drift_edge >= 0 ? '+' : ''}${num(a.mc_drift_edge)}` : ''}>
                  {pct(a.mc_p_tp_first)}
                </td>
                <td className="text-right px-3 font-mono text-slate-400">
                  {a.pair_partner ? <span title={`vs ${a.pair_partner}`}>{a.pair_partner.replace('/USD', '')} <span className={a.pair_z <= -2 ? 'text-emerald-400' : a.pair_z >= 2 ? 'text-rose-400' : ''}>{num(a.pair_z, 1)}</span></span> : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Mini({ label, value, title }) {
  return (
    <div className="rounded-lg bg-white/[0.03] ring-1 ring-white/5 px-2 py-1.5" title={title}>
      <div className="text-[9px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className="text-xs font-mono font-semibold text-slate-200">{value}</div>
    </div>
  );
}
