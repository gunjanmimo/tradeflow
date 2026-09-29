import React, { useState, useEffect, useCallback } from 'react';
import { HugeiconsIcon } from '@hugeicons/react';
import { Radar01Icon, Refresh01Icon } from '@hugeicons/core-free-icons';

const num = (v, d = 2) => (v === null || v === undefined ? '—' : Number(v).toFixed(d));
const signed = (v, d = 1) => (v === null || v === undefined ? '—' : `${v >= 0 ? '+' : ''}${Number(v).toFixed(d)}%`);
const ago = (t) => {
  if (!t) return 'never';
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 90 ? `${Math.round(s)}s ago` : s < 5400 ? `${Math.round(s / 60)} min ago` : `${(s / 3600).toFixed(1)} h ago`;
};
const until = (t) => (!t ? '—' : `in ${Math.max(0, Math.round((t - Date.now() / 1000) / 60))} min`);

const STATUS_TONE = (s = '') => (s === 'ready' ? 'text-emerald-300'
  : s.startsWith('confirming') ? 'text-cyan-300'
  : s.startsWith('blocked') ? 'text-amber-300/90' : 'text-slate-400');

function Meter({ value, entry, exit }) {
  const v = Math.max(0, Math.min(1, value || 0));
  const tone = v >= entry ? 'bg-emerald-400' : v <= exit ? 'bg-rose-400' : 'bg-cyan-400';
  return (
    <div className="relative h-1.5 w-full bg-white/5 rounded-full overflow-hidden" title={`confidence ${num(v)}`}>
      <div style={{ width: `${v * 100}%` }} className={`h-full ${tone} transition-all`} />
      <div style={{ left: `${entry * 100}%` }} className="absolute top-0 h-full w-px bg-white/70" title={`entry ${entry}`} />
    </div>
  );
}

const COMPONENTS = [['performance', 'perf'], ['today', 'today'], ['home', 'home'], ['news', 'news'], ['discussion', 'talk']];

const Country = ({ c }) => (c && c !== 'US'
  ? <span className="text-[9px] px-1 rounded bg-sky-500/10 text-sky-300 ring-1 ring-sky-500/30">{c}</span> : null);

const money = (v, ccy) => {
  if (v === null || v === undefined) return '';
  const a = Math.abs(v);
  const f = a >= 1e9 ? `${(v / 1e9).toFixed(1)}B` : a >= 1e6 ? `${(v / 1e6).toFixed(0)}M` : `${Math.round(v / 1e3)}k`;
  return `${f} ${ccy || ''}`;
};

// One exchange: what it traded most today, and what each stock is (or is not) on our side.
function ExchangeBoard({ b }) {
  const [all, setAll] = useState(false);
  const rows = all ? b.rows : b.rows.slice(0, 6);
  return (
    <div className="rounded-lg bg-slate-950/60 ring-1 ring-white/5 p-2 min-w-0">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[11px] font-semibold text-slate-200">
          <span className={`inline-block w-1.5 h-1.5 rounded-full mr-1 ${b.ok ? 'bg-emerald-400' : 'bg-rose-500'}`} />
          {b.label} <span className="text-slate-500 font-normal">{b.country}</span>
        </span>
        <span className="text-[9px] text-slate-500" title="of its most-traded stocks, how many have a US line we can trade">
          {b.ok ? `${b.tradable} tradable` : 'down'}
        </span>
      </div>
      {!b.ok && <div className="text-[10px] text-rose-300/80">{b.detail}</div>}
      <ul>
        {rows.map((r) => (
          <li key={r.symbol} className="flex items-center justify-between gap-2 text-[10px] leading-5">
            <span className="min-w-0 truncate text-slate-400" title={`${r.name} · ${money(r.value_traded, r.currency)} traded`}>
              <span className="font-mono text-slate-300">{r.symbol}</span> {r.name}
            </span>
            <span className="shrink-0 flex items-center gap-1.5">
              <span className={(r.change_pct ?? 0) >= 0 ? 'text-emerald-300' : 'text-rose-300'}>{signed(r.change_pct, 1)}</span>
              {r.us_symbol && r.eligible ? (
                <span className={`font-mono ${r.picked ? 'text-emerald-300 font-bold' : 'text-cyan-300'}`} title={`score ${num(r.score)}, #${r.rank} overall`}>
                  → {r.us_symbol}{r.picked ? ' ★' : ''}
                </span>
              ) : (
                <span className="text-slate-600 max-w-[110px] truncate" title={r.why_not || ''}>
                  {r.us_symbol ? `${r.us_symbol}: ` : ''}{r.why_not || '—'}
                </span>
              )}
            </span>
          </li>
        ))}
      </ul>
      {b.rows.length > 6 && (
        <button onClick={() => setAll((o) => !o)} className="text-[9px] text-slate-500 hover:text-slate-300 mt-0.5">
          {all ? 'fewer' : `all ${b.rows.length}`}
        </button>
      )}
    </div>
  );
}

// The scout's picks for today and the watcher's live read on each.
export function ScoutPanel({ apiBase }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const [showWorld, setShowWorld] = useState(true);

  const load = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/api/scout`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
      setError(null);
    } catch (e) {
      setError(String(e.message || e));
    }
  }, [apiBase]);

  useEffect(() => {
    load();
    const id = setInterval(load, 5000);    // the watcher re-scores every 5 s
    return () => clearInterval(id);
  }, [load]);

  const refresh = async () => {
    setBusy(true);
    try {
      await fetch(`${apiBase}/api/scout/refresh`, { method: 'POST' });
      await load();
    } finally {
      setBusy(false);
    }
  };

  const reads = data?.watcher?.reads || {};
  const rules = data?.watcher?.rules || { entry: 0.65, exit: 0.4 };
  // Best to worst by the watcher's live confidence (the chance of a trade), re-sorted
  // on every refresh; a pick without a live read yet goes last, by its scout rank.
  const picks = [...(data?.picks || [])].sort((a, b) => {
    const ca = reads[a.symbol]?.confidence, cb = reads[b.symbol]?.confidence;
    if ((ca ?? -1) !== (cb ?? -1)) return (cb ?? -1) - (ca ?? -1);
    return (a.rank ?? 999) - (b.rank ?? 999);
  });
  const byRank = Object.fromEntries((data?.ranking || []).map((r) => [r.symbol, r]));
  const sources = Object.entries(data?.sources || {});

  return (
    <div className="bg-[#0f172a]/70 border border-emerald-900/40 rounded-2xl p-5 shadow-xl">
      <div className="flex items-center justify-between mb-3 gap-3">
        <div className="flex items-center space-x-2 min-w-0">
          <HugeiconsIcon icon={Radar01Icon} size="1em" className="w-5 h-5 text-emerald-400" />
          <h2 className="font-semibold text-base text-slate-100 truncate">
            Today&apos;s watch
            <span className="ml-2 text-xs font-normal text-slate-500">
              ranked {ago(data?.ranked_at)} · next {until(data?.next_at)} · {data?.eligible ?? 0} of {data?.pool ?? 0} eligible
            </span>
          </h2>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className={`text-[10px] px-2 py-0.5 rounded border font-mono ${data?.trading
            ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30' : 'bg-slate-500/10 text-slate-400 border-slate-500/30'}`}>
            {data?.trading ? 'TRADING' : 'WATCH ONLY'}
          </span>
          <button onClick={refresh} disabled={busy || !data?.enabled}
            className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-slate-300 disabled:opacity-40" title="Re-rank now">
            <HugeiconsIcon icon={Refresh01Icon} size="1em" className={`w-3.5 h-3.5 ${busy ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      <div className="flex flex-wrap gap-x-3 gap-y-1 mb-3 text-[10px] text-slate-500">
        {sources.map(([name, h]) => (
          <span key={name} title={h.detail || ''}>
            <span className={`inline-block w-1.5 h-1.5 rounded-full mr-1 ${h.ok ? 'bg-emerald-400' : 'bg-rose-500'}`} />
            {name}{h.ok ? ` ${h.count}` : ' down'}
          </span>
        ))}
      </div>

      {error && <div className="text-xs text-rose-300 mb-2">Scout unavailable: {error}</div>}
      {data && !data.enabled && <div className="text-xs text-slate-500">The scout is switched off (SCOUT_ENABLED).</div>}
      {data?.enabled && picks.length === 0 && (
        <div className="text-xs text-slate-500">{data.ranked_at ? 'No stock scored high enough to watch this hour.' : 'First ranking in progress…'}</div>
      )}

      {picks.length > 0 && (
        <ul className="divide-y divide-white/5">
          {picks.map((p) => {
            const r = reads[p.symbol] || {};
            const row = byRank[p.symbol];
            return (
              <li key={p.symbol} className="py-2">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-baseline gap-2 min-w-0">
                    <span className="text-[10px] text-slate-500 w-5 text-right">{p.rank ? `#${p.rank}` : '—'}</span>
                    <span className="font-mono font-bold text-sm text-slate-100">{p.symbol}</span>
                    <Country c={p.country} />
                    {p.slot && p.slot !== 'overall' && <span className="text-[9px] text-sky-300/80">{p.slot}</span>}
                    {p.held && <span className="text-[10px] font-semibold text-emerald-300">HELD</span>}
                    <span className="text-[10px] text-slate-500">score {num(p.score)}</span>
                    <span className={`text-[11px] ${(r.chg_since_pick_pct ?? 0) >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}
                      title="since it was picked">{signed(r.chg_since_pick_pct, 2)}</span>
                  </div>
                  <span className={`text-[11px] shrink-0 ${STATUS_TONE(r.status)}`}>{r.status || 'waiting for a price'}</span>
                </div>
                <div className="flex items-center gap-3 mt-1 pl-7">
                  <div className="w-28 shrink-0"><Meter value={r.confidence} entry={rules.entry} exit={rules.exit} /></div>
                  <span className="text-[11px] font-mono text-slate-300 w-8">{num(r.confidence)}</span>
                  <span className="text-[11px] text-slate-500 truncate">
                    {(r.reasons || []).join(' · ') || (row?.reasons || []).slice(0, 2).join(' · ')}
                  </span>
                </div>
              </li>
            );
          })}
        </ul>
      )}

      {(data?.exchanges || []).length > 0 && (
        <div className="mt-3">
          <button onClick={() => setShowWorld((o) => !o)} className="text-[11px] uppercase tracking-wider text-slate-400 hover:text-slate-200">
            World exchanges ({data.exchanges.filter((b) => b.ok).length}/{data.exchanges.length} answering) {showWorld ? '▾' : '▸'}
          </button>
          {showWorld && (
            <>
              <p className="text-[10px] text-slate-500 mt-1 mb-2">
                Each exchange&apos;s most-traded stocks today. → is the US line Alpaca can trade (★ = picked). Stocks with no US
                line, or only an OTC one (no prices on this data plan), are shown but cannot be traded. Best of each exchange gets
                a watch slot ({data.rules?.exchange_slots ?? 1} each, {data.rules?.intl_max_picks ?? 6} max, score ≥ {data.rules?.exchange_min_score ?? 0.45}).
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {data.exchanges.map((b) => <ExchangeBoard key={b.market} b={b} />)}
              </div>
            </>
          )}
        </div>
      )}

      {(data?.ranking || []).length > 0 && (
        <div className="mt-3">
          <button onClick={() => setShowAll((o) => !o)} className="text-[11px] uppercase tracking-wider text-slate-400 hover:text-slate-200">
            Full ranking ({data.ranking.length}) {showAll ? '▾' : '▸'}
          </button>
          {showAll && (
            <div className="mt-2 overflow-x-auto">
              <table className="w-full text-[11px]">
                <thead className="text-slate-500">
                  <tr>
                    <th className="text-left font-normal pr-2">#</th>
                    <th className="text-left font-normal pr-2">symbol</th>
                    <th className="text-right font-normal pr-2">score</th>
                    {COMPONENTS.map(([, l]) => <th key={l} className="text-right font-normal pr-2">{l}</th>)}
                    <th className="text-left font-normal">why</th>
                  </tr>
                </thead>
                <tbody>
                  {data.ranking.map((r) => (
                    <tr key={r.symbol} className="border-t border-white/5">
                      <td className="pr-2 text-slate-500">{r.rank}</td>
                      <td className="pr-2 font-mono text-slate-200">{r.symbol} <Country c={r.country} /></td>
                      <td className="pr-2 text-right font-mono text-slate-200">{num(r.score)}</td>
                      {COMPONENTS.map(([k, l]) => (
                        <td key={l} className="pr-2 text-right font-mono text-slate-400">{num(r.components?.[k])}</td>
                      ))}
                      <td className="text-slate-500 truncate max-w-[280px]" title={(r.reasons || []).join('; ')}>
                        {(r.reasons || []).slice(0, 2).join(' · ')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
