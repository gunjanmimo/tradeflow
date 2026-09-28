import React from 'react';
import { HugeiconsIcon } from '@hugeicons/react';
import { Robot01Icon } from '@hugeicons/core-free-icons';

const money = (v) => (v === null || v === undefined ? '—'
  : `$${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`);

const STATE_TONE = {
  active: ['text-emerald-300', 'bg-emerald-400'],
  watching: ['text-cyan-300', 'bg-cyan-400'],
  full: ['text-cyan-300', 'bg-cyan-400'],
  paused: ['text-amber-300', 'bg-amber-400'],
  stalled: ['text-rose-300', 'bg-rose-500'],
  disabled: ['text-slate-400', 'bg-slate-500'],
  starting: ['text-slate-400', 'bg-slate-500'],
};

// Header chip: what the manager is doing and how much budget it has left to deploy.
export function ManagerChip({ manager, onClick }) {
  const st = manager?.state || 'starting';
  const [text, dot] = STATE_TONE[st] || STATE_TONE.starting;
  return (
    <button
      onClick={onClick}
      className="group flex items-center gap-2.5 rounded-xl px-3 py-1.5 bg-white/[0.03] ring-1 ring-white/10 hover:ring-cyan-400/50 hover:bg-cyan-400/5 transition-all text-left"
      title={manager?.message || 'Portfolio manager'}
    >
      <HugeiconsIcon icon={Robot01Icon} size="1em" className={`w-4 h-4 ${text}`} />
      <div className="flex flex-col min-w-[74px]">
        <span className="text-[10px] uppercase tracking-wider text-slate-500 leading-none">Manager</span>
        <span className={`text-sm font-semibold leading-tight capitalize ${text}`}>
          {st}
          <span className="text-[10px] text-slate-500 font-normal ml-1">{money(manager?.budget?.idle)} idle</span>
        </span>
      </div>
      <span className={`w-1.5 h-1.5 rounded-full ${dot} ${st === 'active' ? 'animate-pulse' : ''}`} />
    </button>
  );
}

const fmtMin = (m) => (m === null || m === undefined ? '—'
  : m >= 60 ? `${Math.floor(m / 60)}h ${Math.round(m % 60)}m` : `${m.toFixed(0)}m`);

const TREND_TONE = { uptrend: 'text-emerald-300', downtrend: 'text-rose-300', range: 'text-slate-300', unknown: 'text-slate-500' };
const ACTION_TONE = { BUY: 'text-emerald-300', SELL: 'text-amber-300', CLOSE: 'text-rose-300', HOLD: 'text-slate-300' };

function TrendCell({ t }) {
  if (!t) return <span className="text-slate-600">—</span>;
  if (!t.ready) return <span className="text-slate-500" title={`${t.bars} one-minute bars so far`}>learning</span>;
  return (
    <span className={TREND_TONE[t.label] || 'text-slate-300'} title={`confidence ${t.confidence.toFixed(2)}`}>
      {t.label} {t.direction > 0 ? '+' : ''}{t.direction.toFixed(2)}
      {t.reversal_down ? ' ↘' : t.reversal_up ? ' ↗' : ''}
    </span>
  );
}

function FleetCards({ fleet }) {
  const agents = fleet?.agents || [];
  if (!agents.length) return null;
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1.5">Agent fleet</div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {agents.map((a) => {
          const bad = a.error || a.state === 'stalled' || a.state === 'error';
          const ago = a.last_at ? Math.max(0, Math.round(Date.now() / 1000 - a.last_at)) : null;
          return (
            <div key={a.name} className="rounded-xl bg-white/[0.03] ring-1 ring-white/10 p-2.5" title={a.role}>
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-white">{a.name}</span>
                <span className={`text-[10px] font-mono ${bad ? 'text-rose-300' : 'text-emerald-300'}`}>
                  {bad ? (a.state === 'stalled' ? 'stalled' : 'error') : a.state}
                  {a.interval_s ? ` · ${a.interval_s}s` : ' · per tick'}
                  {ago !== null ? ` · ${ago}s ago` : ''}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-0.5 leading-snug line-clamp-2">{a.error || a.summary}</p>
              {a.last_action && (
                <p className="text-[10px] text-slate-500 mt-0.5 truncate">
                  last: <span className="text-slate-300">{a.last_action.action} {a.last_action.symbol}</span> · {a.last_action.reason}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function ManagerPanel({ manager, positions, fleet }) {
  const m = manager || {};
  const b = m.budget || {};
  const [text] = STATE_TONE[m.state] || STATE_TONE.starting;
  const held = Object.values(positions || {});
  const blocked = Object.entries(m.blocked_by || {});

  return (
    <div className="space-y-4 text-sm">
      <FleetCards fleet={fleet} />
      <div className="rounded-xl bg-white/[0.03] ring-1 ring-white/10 p-3">
        <div className={`text-xs font-semibold uppercase tracking-wider ${text}`}>{m.state || 'starting'}</div>
        <p className="text-slate-300 text-xs mt-1 leading-relaxed">{m.message}</p>
        {m.error && <p className="text-rose-300 text-xs mt-1 font-mono">{m.error}</p>}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center">
        {[
          ['Hard cap', money(b.hard_cap), 'text-white'],
          ['Committed', money(b.committed), 'text-cyan-300'],
          ['Idle', money(b.idle), b.idle > 0 ? 'text-amber-300' : 'text-slate-300'],
          ['Slots free', `${m.slots_free ?? '—'} / ${m.max_positions ?? '—'}`, 'text-slate-200'],
        ].map(([k, v, c]) => (
          <div key={k} className="rounded-xl bg-white/[0.03] ring-1 ring-white/10 py-2">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">{k}</div>
            <div className={`font-mono text-sm font-semibold ${c}`}>{v}</div>
          </div>
        ))}
      </div>
      {b.dial_deployable_pct !== undefined && b.dial_deployable_pct < 100 && (
        <p className="text-[11px] text-slate-500">
          At this risk dial the position limits cap deployment at {b.dial_deployable_pct}% of the budget
          ({m.max_positions} positions); the rest stays idle by design until the dial is raised.
        </p>
      )}

      <div>
        <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1.5">
          Watchlist · {m.watching ?? 0} symbols watched every {m.interval_s ?? '—'}s (open positions excluded)
        </div>
        {(m.watch || []).length === 0 ? (
          <div className="text-xs text-slate-500 py-2">Nothing on the watchlist to watch.</div>
        ) : (
          <div className="max-h-56 overflow-y-auto rounded-lg ring-1 ring-white/5">
            <table className="w-full text-xs font-mono">
              <thead className="sticky top-0 bg-[#0f1624] text-[10px] uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="text-left font-normal px-2 py-1">Symbol</th>
                  <th className="text-right font-normal px-2 py-1">Price</th>
                  <th className="text-right font-normal px-2 py-1">1m</th>
                  <th className="text-right font-normal px-2 py-1">5m</th>
                  <th className="text-right font-normal px-2 py-1" title="Seconds since the price last changed">Age</th>
                  <th className="text-left font-normal px-2 py-1" title="Composite of 1-minute micro and session trends and the daily backdrop">Trend</th>
                  <th className="text-left font-normal px-2 py-1">Verdict</th>
                </tr>
              </thead>
              <tbody>
                {m.watch.map((w) => (
                  <tr key={w.symbol} className="border-t border-white/5">
                    <td className="px-2 py-1 font-semibold text-white">{w.symbol}</td>
                    <td className="px-2 py-1 text-right text-slate-300">{w.price === null ? '—' : Number(w.price).toPrecision(6).replace(/\.?0+$/, '')}</td>
                    {[w.chg_1m_pct, w.chg_5m_pct].map((c, i) => (
                      <td key={i} className={`px-2 py-1 text-right ${c === null || c === undefined ? 'text-slate-600' : c > 0 ? 'text-emerald-300' : c < 0 ? 'text-rose-300' : 'text-slate-400'}`}>
                        {c === null || c === undefined ? '—' : `${c > 0 ? '+' : ''}${c.toFixed(2)}%`}
                      </td>
                    ))}
                    <td className={`px-2 py-1 text-right ${(w.price_age_s ?? 0) >= 60 ? 'text-amber-300' : 'text-slate-400'}`}>
                      {w.price_age_s === null || w.price_age_s === undefined ? '—' : `${Math.round(w.price_age_s)}s`}
                    </td>
                    <td className="px-2 py-1 whitespace-nowrap"><TrendCell t={w.trend} /></td>
                    <td className={`px-2 py-1 ${w.status === 'buy signal' ? 'text-emerald-300 font-semibold' : 'text-slate-400'}`}>
                      {w.status === 'no entry' ? `no entry: ${w.verdict}` : w.status}
                      {w.buy_prob !== undefined && w.status === 'buy signal' ? ` (${w.buy_prob.toFixed(2)})` : ''}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div>
        <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1.5">
          Next entries, ranked by conviction and diversification fit
        </div>
        {(m.picks || []).length === 0 ? (
          <div className="text-xs text-slate-500 py-2">No symbol qualifies right now.</div>
        ) : (
          <div className="space-y-1.5">
            {m.picks.map((p, i) => (
              <div key={p.symbol} className="flex items-center justify-between rounded-lg bg-white/[0.03] ring-1 ring-white/5 px-3 py-1.5 text-xs">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-slate-500 font-mono w-4">{i + 1}</span>
                  <span className="font-semibold text-white">{p.symbol}</span>
                  <span className="text-slate-500 truncate">{p.sector} · {p.region}</span>
                </div>
                <div className="flex items-center gap-3 font-mono shrink-0" title={(p.why || []).join('; ')}>
                  {p.plan && (
                    <span className="text-cyan-300" title={`stop ${p.plan.stop} · target ${p.plan.target} · ${p.plan.tier}`}>
                      {money(p.plan.dollars)} · {p.plan.qty}x · risk {money(p.plan.risk)}
                    </span>
                  )}
                  <TrendCell t={p.trend} />
                  <span className="text-slate-400">conv {p.conviction.toFixed(2)}</span>
                  <span className="text-slate-400">fit {p.fit.toFixed(2)}</span>
                  <span className="text-emerald-300 font-semibold">{p.score.toFixed(2)}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {blocked.length > 0 && (
        <div>
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1.5">Why symbols were skipped</div>
          <div className="flex flex-wrap gap-1.5">
            {blocked.map(([k, v]) => (
              <span key={k} className="text-[11px] text-slate-300 bg-white/[0.04] ring-1 ring-white/10 rounded-md px-2 py-0.5">
                {k} <span className="text-slate-500">×{v}</span>
              </span>
            ))}
          </div>
        </div>
      )}

      <div>
        <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1.5">
          Day-trade clock · stocks close before the market does; a position with no new price for 2 min is closed
        </div>
        {held.length === 0 ? (
          <div className="text-xs text-slate-500 py-2">No open positions.</div>
        ) : (
          <div className="space-y-1.5">
            {held.map((p) => (
              <div key={p.symbol} className="rounded-lg bg-white/[0.03] ring-1 ring-white/5 px-3 py-1.5 text-xs">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-white">
                  {p.symbol}
                  {p.fleet_action && <span className={`ml-2 font-mono ${ACTION_TONE[p.fleet_action] || ''}`}>{p.fleet_action}</span>}
                  {p.trend && <span className="ml-2 font-mono font-normal"><TrendCell t={p.trend} /></span>}
                </span>
                <div className="flex items-center gap-4 font-mono text-slate-400">
                  <span>
                    price age <span className={(p.price_age_s ?? 0) >= 90 ? 'text-amber-300' : 'text-slate-200'}>
                      {p.price_age_s === undefined ? '—' : `${Math.round(p.price_age_s)}s`}
                    </span>
                  </span>
                  <span title="Minutes until the market this stock trades on closes">
                    closes in <span className="text-slate-200">{p.minutes_to_close === null || p.minutes_to_close === undefined ? '—' : fmtMin(p.minutes_to_close)}</span>
                  </span>
                  <span title="Minutes until it is force-closed for the day">
                    flatten in <span className={p.flatten_in_min !== null && p.flatten_in_min !== undefined && p.flatten_in_min <= 5 ? 'text-rose-300' : 'text-slate-200'}>
                      {p.flatten_in_min === null || p.flatten_in_min === undefined ? '—' : fmtMin(p.flatten_in_min)}
                    </span>
                  </span>
                </div>
              </div>
              {p.fleet_reason && <p className="text-[10px] text-slate-500 mt-0.5 truncate">{p.fleet_reason}</p>}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="text-[10px] text-slate-600 font-mono">
        {m.cycles ?? 0} cycles · {m.cycle_ms ?? 0}ms · every {m.interval_s ?? '—'}s · {m.entries_total ?? 0} entries placed
      </div>
    </div>
  );
}
