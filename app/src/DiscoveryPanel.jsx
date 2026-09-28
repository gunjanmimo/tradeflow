import React, { useState, useEffect, useCallback } from 'react';
import { HugeiconsIcon } from '@hugeicons/react';
import { GlobeIcon } from '@hugeicons/core-free-icons';

const money = (v) => (v === null || v === undefined ? '—'
  : `$${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`);
const pct = (v, d = 1) => (v === null || v === undefined ? '—' : `${Number(v).toFixed(d)}%`);
const num = (v, d = 2) => (v === null || v === undefined ? '—' : Number(v).toFixed(d));

const scoreTone = (s) => (s === null || s === undefined ? 'bg-slate-600'
  : s >= 0.7 ? 'bg-emerald-400' : s >= 0.5 ? 'bg-cyan-400' : s >= 0.35 ? 'bg-amber-400' : 'bg-rose-400');

// Header chip: book spread and any diversification warnings, from telemetry.
export function DiscoveryChip({ brief, onClick }) {
  const warn = brief?.warnings?.length || 0;
  return (
    <button
      onClick={onClick}
      className="group flex items-center gap-2.5 rounded-xl px-3 py-1.5 bg-white/[0.03] ring-1 ring-white/10 hover:ring-cyan-400/50 hover:bg-cyan-400/5 transition-all text-left"
      title="Stock discovery, diversification and portfolio risk"
    >
      <HugeiconsIcon icon={GlobeIcon} size="1em" className={`w-4 h-4 ${warn ? 'text-amber-400' : 'text-cyan-400'}`} />
      <div className="flex flex-col min-w-[74px]">
        <span className="text-[10px] uppercase tracking-wider text-slate-500 leading-none">Discovery</span>
        <span className="text-sm font-semibold leading-tight text-white">
          {brief?.candidates ?? 0}<span className="text-[10px] text-slate-500 font-normal ml-1">candidates</span>
        </span>
      </div>
      {warn > 0 && <span className="text-[10px] font-semibold text-amber-300 bg-amber-500/10 ring-1 ring-amber-500/30 rounded-md px-1.5">{warn}</span>}
    </button>
  );
}

function Bar({ value, cap, target, tone = 'bg-cyan-400' }) {
  // value/cap/target are % of budget; the track spans to the larger of cap or value.
  const span = Math.max(cap || 0, value || 0, target || 0, 1);
  const over = cap !== null && cap !== undefined && value > cap + 0.01;
  return (
    <div className="relative h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
      <div style={{ width: `${Math.min(100, (value / span) * 100)}%` }} className={`h-full ${over ? 'bg-rose-400' : tone} transition-all`} />
      {cap !== null && cap !== undefined && <div style={{ left: `${(cap / span) * 100}%` }} className="absolute top-0 h-full w-px bg-white/60" title={`cap ${cap}%`} />}
      {target ? <div style={{ left: `${(target / span) * 100}%` }} className="absolute top-0 h-full w-px bg-emerald-300" title={`target ${target}%`} /> : null}
    </div>
  );
}

const TABS = [['candidates', 'Candidates'], ['diversification', 'Diversification'], ['risk', 'Risk & sleeves']];

export function DiscoveryPanel({ apiBase, positions: telemetryPositions }) {
  const [tab, setTab] = useState('candidates');
  const [data, setData] = useState(null);
  const [div, setDiv] = useState(null);
  const [status, setStatus] = useState('active');
  const [sector, setSector] = useState('');
  const [region, setRegion] = useState('');
  const [open, setOpen] = useState(null);
  const [whatIf, setWhatIf] = useState({});
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      const q = new URLSearchParams({ limit: '150' });
      if (status !== 'all') q.set('status', status);
      if (sector) q.set('sector', sector);
      if (region) q.set('region', region);
      const [a, b] = await Promise.all([
        fetch(`${apiBase}/api/discovery?${q}`).then((r) => r.json()),
        fetch(`${apiBase}/api/diversification`).then((r) => r.json()),
      ]);
      setData(a); setDiv(b); setError(null);
    } catch (e) {
      setError(`Could not load discovery data: ${e.message}`);
    }
  }, [apiBase, status, sector, region]);

  useEffect(() => {
    load();
    const id = setInterval(load, 20000);
    return () => clearInterval(id);
  }, [load]);

  const post = async (path, symbol) => {
    setBusy(symbol); setError(null);
    try {
      const r = await fetch(`${apiBase}${path}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ symbol }),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail || r.statusText);
      await load();
    } catch (e) {
      setError(`${symbol}: ${e.message}`);
    } finally {
      setBusy(null);
    }
  };

  const unwatch = async (symbol) => {
    setBusy(symbol); setError(null);
    try {
      const r = await fetch(`${apiBase}/api/watchlist/${encodeURIComponent(symbol)}`, { method: 'DELETE' });
      if (!r.ok) throw new Error(r.statusText);
      await load();
    } catch (e) {
      setError(`${symbol}: ${e.message}`);
    } finally {
      setBusy(null);
    }
  };

  const toggle = async (sym) => {
    if (open === sym) { setOpen(null); return; }
    setOpen(sym);
    if (!whatIf[sym]) {
      try {
        const r = await fetch(`${apiBase}/api/discovery/what-if/${encodeURIComponent(sym)}`);
        const body = await r.json();
        setWhatIf((w) => ({ ...w, [sym]: body }));
      } catch { /* the row still shows its reasons */ }
    }
  };

  const refresh = async () => {
    setBusy('__refresh');
    try { await fetch(`${apiBase}/api/discovery/refresh`, { method: 'POST' }); await load(); }
    finally { setBusy(null); }
  };

  const st = data?.status;
  const sectors = div?.sleeves?.sectors?.map((s) => s.sleeve) || [];
  const regions = div?.sleeves?.regions?.map((s) => s.sleeve) || [];

  return (
    <div className="text-sm">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div className="flex gap-1 bg-white/[0.03] ring-1 ring-white/10 rounded-lg p-0.5">
          {TABS.map(([k, label]) => (
            <button key={k} onClick={() => setTab(k)}
              className={`px-3 py-1 rounded-md text-xs font-medium ${tab === k ? 'bg-cyan-500/15 text-cyan-200' : 'text-slate-400 hover:text-white'}`}>
              {label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-3 text-[11px] text-slate-500">
          {st && <span>{st.scored}/{st.candidates} scored · {st.daily_bars?.symbols ?? 0} with history · risk dial {div?.risk_factor ?? '—'}</span>}
          <button onClick={refresh} disabled={busy === '__refresh'} className="px-2 py-1 rounded-md ring-1 ring-white/10 hover:ring-cyan-400/50 text-slate-300 disabled:opacity-50">
            {busy === '__refresh' ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      {error && <div className="mb-3 text-xs text-rose-300 bg-rose-500/10 ring-1 ring-rose-500/30 rounded-lg px-3 py-2">{error}</div>}
      {st && !st.daily_bars?.available && (
        <div className="mb-3 text-xs text-amber-300 bg-amber-500/10 ring-1 ring-amber-500/30 rounded-lg px-3 py-2">
          No Alpaca data keys: momentum, correlation and VaR are unavailable. Classification and sleeve caps still apply.
        </div>
      )}

      {tab === 'candidates' && (
        <>
          <div className="flex flex-wrap gap-2 mb-3">
            {[['status', status, setStatus, [['active', 'Candidates'], ['watching', 'Watching'], ['dismissed', 'Hidden'], ['all', 'All']]],
              ['sector', sector, setSector, [['', 'All sectors'], ...sectors.map((s) => [s, s])]],
              ['region', region, setRegion, [['', 'All regions'], ...regions.map((s) => [s, s])]]]
              .map(([k, v, set, opts]) => (
                <select key={k} value={v} onChange={(e) => set(e.target.value)}
                  className="bg-[#0b111c] ring-1 ring-white/10 rounded-md px-2 py-1 text-xs text-slate-200">
                  {opts.map(([ov, ol]) => <option key={ov} value={ov}>{ol}</option>)}
                </select>
              ))}
          </div>
          <div className="max-h-[60vh] overflow-auto rounded-lg ring-1 ring-white/5">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-[#0f1624] text-slate-500 text-[10px] uppercase tracking-wider">
                <tr>
                  <th className="text-left px-2 py-2">Symbol</th>
                  <th className="text-left px-2">Sector · theme</th>
                  <th className="text-left px-2">Region</th>
                  <th className="text-left px-2 w-28">Score</th>
                  <th className="text-right px-2" title="Smart money (insiders + copy traders)">Smart</th>
                  <th className="text-right px-2" title="Public sentiment (news via Laya + StockTwits)">Public</th>
                  <th className="text-right px-2">Mom.</th>
                  <th className="text-right px-2" title="Diversification fit under the current dial">Fit</th>
                  <th className="text-right px-2">Room</th>
                  <th className="px-2" />
                </tr>
              </thead>
              <tbody>
                {(data?.candidates || []).map((c) => {
                  const m = c.meta; const k = c.components || {};
                  const cell = (v) => <td className="text-right px-2 font-mono text-slate-300">{v === undefined ? <span className="text-slate-600">—</span> : num(v)}</td>;
                  return (
                    <React.Fragment key={c.symbol}>
                      <tr onClick={() => toggle(c.symbol)} className="border-t border-white/5 hover:bg-white/[0.03] cursor-pointer">
                        <td className="px-2 py-1.5">
                          <div className="font-semibold text-white">{c.symbol}</div>
                          <div className="text-[10px] text-slate-500 truncate max-w-[140px]">{m.name}</div>
                        </td>
                        <td className="px-2">
                          <div className={m.is_defensive ? 'text-emerald-300' : 'text-slate-300'}>{m.sector}</div>
                          <div className="text-[10px] text-slate-500">{m.theme}</div>
                        </td>
                        <td className="px-2 text-slate-300">{m.region}<div className="text-[10px] text-slate-500">{m.country}{m.currency !== 'USD' ? ` · ${m.currency} risk` : ''}</div></td>
                        <td className="px-2">
                          <div className="flex items-center gap-2">
                            <div className="h-1.5 w-14 bg-white/5 rounded-full overflow-hidden"><div style={{ width: `${(c.score || 0) * 100}%` }} className={`h-full ${scoreTone(c.score)}`} /></div>
                            <span className="font-mono text-slate-200">{c.score === null ? '—' : c.score.toFixed(2)}</span>
                          </div>
                        </td>
                        {cell(k.smart_money)}{cell(k.public_sentiment)}{cell(k.momentum)}{cell(k.diversification)}
                        <td className="text-right px-2 font-mono text-slate-400">{c.fit ? money(c.fit.max_dollars) : <span className="text-slate-600">n/a</span>}</td>
                        <td className="px-2 text-right whitespace-nowrap" onClick={(e) => e.stopPropagation()}>
                          {c.status === 'candidate' && m.tradable_on_alpaca && (
                            <button disabled={busy === c.symbol} onClick={() => post('/api/discovery/promote', c.symbol)}
                              className="px-2 py-0.5 rounded-md text-[11px] bg-emerald-500/10 text-emerald-300 ring-1 ring-emerald-500/30 hover:bg-emerald-500/20 disabled:opacity-50">Watch</button>
                          )}
                          {c.status === 'candidate' && (
                            <button disabled={busy === c.symbol} onClick={() => post('/api/discovery/dismiss', c.symbol)}
                              className="ml-1 px-2 py-0.5 rounded-md text-[11px] text-slate-400 ring-1 ring-white/10 hover:text-white">Hide</button>
                          )}
                          {c.status === 'dismissed' && (
                            <button onClick={() => post('/api/discovery/restore', c.symbol)} className="px-2 py-0.5 rounded-md text-[11px] text-slate-300 ring-1 ring-white/10">Restore</button>
                          )}
                          {c.status === 'watching' && (
                            <button
                              disabled={busy === c.symbol || !!telemetryPositions?.[c.symbol]}
                              onClick={() => unwatch(c.symbol)}
                              title={telemetryPositions?.[c.symbol]
                                ? 'Watching · position open, it stays on the watchlist until it closes'
                                : `${c.auto ? 'Auto-picked by discovery' : 'Watching'} · click to stop watching`}
                              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] bg-cyan-500/10 text-cyan-300 ring-1 ring-cyan-500/30 hover:bg-rose-500/10 hover:text-rose-300 hover:ring-rose-500/30 disabled:hover:bg-cyan-500/10 disabled:hover:text-cyan-300 disabled:hover:ring-cyan-500/30">
                              <svg viewBox="0 0 24 24" className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5" /></svg>
                              {c.auto ? 'Auto' : 'Watching'}
                            </button>
                          )}
                        </td>
                      </tr>
                      {open === c.symbol && <CandidateDetail c={c} w={whatIf[c.symbol]} />}
                    </React.Fragment>
                  );
                })}
                {data && !data.candidates?.length && (
                  <tr><td colSpan={10} className="text-center text-slate-500 py-6">No candidates match these filters.</td></tr>
                )}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-[10px] text-slate-500">
            Score weights: smart money {pct(data?.weights?.smart_money * 100, 0)}, public sentiment {pct(data?.weights?.public_sentiment * 100, 0)},
            momentum {pct(data?.weights?.momentum * 100, 0)}, diversification fit {pct(data?.weights?.diversification * 100, 0)}. Missing components do not vote.
            {data?.status?.auto_promote
              ? `The top ${data.status.auto_top_n} stocks scoring ${data.status.auto_min_score}+ are put on the watchlist automatically (✓ Auto). `
              : ''}
            "Watch" adds one by hand. Every entry still passes the strategy, risk guard and diversification caps.
          </p>
        </>
      )}

      {tab === 'diversification' && div && <DiversificationView div={div} />}
      {tab === 'risk' && div && <RiskView div={div} sleeves={data?.sleeve_momentum || []} />}
    </div>
  );
}

function CandidateDetail({ c, w }) {
  const news = c.public_sentiment?.news; const stw = c.public_sentiment?.stocktwits;
  return (
    <tr className="bg-white/[0.02]">
      <td colSpan={10} className="px-3 py-3">
        <div className="grid md:grid-cols-3 gap-4 text-xs">
          <div>
            <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Why</div>
            <ul className="space-y-0.5 text-slate-300 list-disc list-inside">
              {(c.reasons || []).map((r, i) => <li key={i}>{r}</li>)}
              {!c.reasons?.length && <li className="text-slate-500">No evidence yet.</li>}
            </ul>
            <div className="mt-1 text-[10px] text-slate-500">Found via: {c.origins.join(', ')}</div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Public sentiment</div>
            {news ? (
              <div className="text-slate-300">
                News: <span className="text-emerald-300">{pct(news.pos * 100, 0)}</span> / <span className="text-rose-300">{pct(news.neg * 100, 0)}</span> over {news.n_headlines} headlines ({pct(news.agreement * 100, 0)} agree)
                <div className="text-[10px] text-slate-500 truncate" title={news.headline}>“{news.headline}”</div>
              </div>
            ) : <div className="text-slate-500">No recent scored news.</div>}
            <div className="mt-1 text-slate-300">{stw ? `StockTwits: ${stw.details}` : <span className="text-slate-500">No StockTwits reading.</span>}</div>
            {c.momentum && (
              <div className="mt-2 text-slate-400">
                3m {pct(c.momentum.ret_3m_pct)} · 12-1m {pct(c.momentum.mom_12_1_pct)} · vs {c.momentum.sector_etf || 'sector'} {pct(c.momentum.rel_vs_sector_3m_pct)} · vol {pct(c.momentum.vol_3m_ann_pct, 0)}
              </div>
            )}
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Portfolio impact (standard position)</div>
            {!w ? <div className="text-slate-500">Measuring…</div> : (
              <div className="text-slate-300 space-y-0.5">
                <div className="font-medium text-white">{w.verdict}</div>
                <div>Size {money(w.position_size)} ({pct(w.position_pct)} of budget){w.limited_by ? ` · limited by ${w.limited_by}` : ''}</div>
                {w.before && w.after && (
                  <table className="mt-1 text-[11px] font-mono">
                    <tbody>
                      {[['Vol (ann.)', 'vol_ann_pct', (v) => pct(v)], ['VaR95 1d', 'var95_1d', money], ['CVaR95 1d', 'cvar95_1d', money],
                        ['Div. ratio', 'diversification_ratio', (v) => num(v)], ['Effective bets', 'effective_bets', (v) => num(v, 1)]]
                        .map(([l, k, f]) => (
                          <tr key={k}><td className="pr-3 text-slate-500 font-sans">{l}</td><td className="pr-2">{f(w.before[k])}</td><td className="text-slate-500 pr-2">→</td><td>{f(w.after[k])}</td></tr>
                        ))}
                    </tbody>
                  </table>
                )}
                {(w.assessment?.notes || []).map((n, i) => <div key={i} className="text-amber-300">{n}</div>)}
              </div>
            )}
          </div>
        </div>
      </td>
    </tr>
  );
}

function SleeveRow({ s }) {
  return (
    <div className="grid grid-cols-[150px_1fr_110px] items-center gap-3 py-1">
      <div className={`truncate ${s.defensive ? 'text-emerald-300' : 'text-slate-300'}`}>{s.sleeve}</div>
      <Bar value={s.pct} cap={s.cap_pct} target={s.target_pct} tone={s.defensive ? 'bg-emerald-400' : 'bg-cyan-400'} />
      <div className="text-right font-mono text-[11px] text-slate-400">
        {pct(s.pct)}{s.cap_pct !== null && s.cap_pct !== undefined ? <span className="text-slate-600"> / {s.cap_pct}%</span> : ''}
      </div>
    </div>
  );
}

function DiversificationView({ div }) {
  const sl = div.sleeves || {}; const L = div.limits || {};
  return (
    <div className="grid md:grid-cols-2 gap-5 text-xs">
      <div>
        <div className="flex items-baseline justify-between mb-1">
          <h3 className="text-[10px] uppercase tracking-wider text-slate-500">Sectors (% of {money(sl.base)} budget)</h3>
          <span className="text-[10px] text-slate-500">white tick = cap</span>
        </div>
        {(sl.sectors || []).map((s) => <SleeveRow key={s.sleeve} s={s} />)}
      </div>
      <div className="space-y-4">
        <div>
          <h3 className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Regions <span className="normal-case">(green tick = target)</span></h3>
          {(sl.regions || []).map((s) => <SleeveRow key={s.sleeve} s={s} />)}
        </div>
        <div>
          <h3 className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Defensive vs cyclical</h3>
          <SleeveRow s={{ sleeve: 'Defensive share', pct: sl.defensive_pct, target_pct: sl.min_defensive_pct, defensive: true }} />
          <div className="text-[11px] text-slate-500">Invested {pct(sl.invested_pct)} of budget · cyclical room {money(sl.cyclical_headroom)}</div>
        </div>
        <div className="rounded-lg ring-1 ring-white/10 p-3 bg-white/[0.02]">
          <h3 className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Limits at risk dial {div.risk_factor} ({div.risk_label})</h3>
          <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-slate-300">
            <span>Any sector ≤ {L.max_sector_pct}%</span>
            <span>US ≤ {L.max_us_pct}%</span><span>Europe / Asia ≤ {L.max_intl_region_pct}% each</span>
            <span>Defensive ≥ {L.min_defensive_pct}%</span><span>Single position ≤ {L.max_position_pct}%</span>
            <span className="col-span-2">Correlation ≥ {L.max_pair_correlation} with a holding halves size</span>
          </div>
        </div>
        <Warnings list={div.warnings} />
      </div>
    </div>
  );
}

function Warnings({ list }) {
  if (!list?.length) return <div className="text-[11px] text-emerald-300">No diversification warnings.</div>;
  return (
    <ul className="space-y-1">
      {list.map((w, i) => <li key={i} className="text-[11px] text-amber-300 bg-amber-500/10 ring-1 ring-amber-500/20 rounded-md px-2 py-1">{w}</li>)}
    </ul>
  );
}

function RiskView({ div, sleeves }) {
  const r = div.risk;
  const tiles = r ? [
    ['Volatility (ann.)', pct(r.vol_ann_pct)], ['1-day VaR 95%', money(r.var95_1d)], ['1-day CVaR 95%', money(r.cvar95_1d)],
    ['Historical VaR 95%', money(r.hist_var95_1d)], ['VaR / budget', pct(r.var95_pct_of_budget, 2)], ['Worst day (window)', money(r.worst_day)],
    ['Diversification ratio', num(r.diversification_ratio)], ['Effective bets', num(r.effective_bets, 1)], ['History coverage', pct(r.coverage_pct, 0)],
  ] : [];
  const corr = r?.correlation;
  const heat = (v) => `rgba(${v >= 0 ? '244,63,94' : '56,189,248'},${Math.min(1, Math.abs(v)) * 0.8})`;
  return (
    <div className="grid md:grid-cols-2 gap-5 text-xs">
      <div className="space-y-4">
        {!r ? (
          <div className="text-slate-500">{div.positions ? 'No daily history for holdings yet.' : 'No open positions: portfolio risk appears once something is held. Use a candidate’s portfolio impact to preview.'}</div>
        ) : (
          <>
            <div className="grid grid-cols-3 gap-2">
              {tiles.map(([k, v]) => (
                <div key={k} className="rounded-lg bg-white/[0.03] ring-1 ring-white/5 px-2.5 py-2">
                  <div className="text-[10px] text-slate-500">{k}</div>
                  <div className="font-mono text-slate-100">{v}</div>
                </div>
              ))}
            </div>
            <div>
              <h3 className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Share of portfolio risk by sector</h3>
              {Object.entries(r.risk_contrib_by_sector || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                <div key={k} className="grid grid-cols-[150px_1fr_50px] items-center gap-3 py-0.5">
                  <span className="text-slate-300 truncate">{k}</span><Bar value={Math.max(0, v)} tone="bg-violet-400" />
                  <span className="text-right font-mono text-slate-400">{pct(v, 0)}</span>
                </div>
              ))}
            </div>
            {corr?.symbols?.length > 1 && (
              <div>
                <h3 className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Holdings correlation ({r.days} days)</h3>
                <table className="text-[10px] font-mono">
                  <thead><tr><th />{corr.symbols.map((s) => <th key={s} className="px-1 text-slate-500 font-normal">{s}</th>)}</tr></thead>
                  <tbody>
                    {corr.matrix.map((row, i) => (
                      <tr key={i}><td className="pr-1 text-slate-500">{corr.symbols[i]}</td>
                        {row.map((v, j) => <td key={j} style={{ background: i === j ? 'transparent' : heat(v) }} className="px-1.5 py-0.5 text-center text-slate-100">{i === j ? '·' : v.toFixed(2)}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
        <Warnings list={div.warnings} />
        <p className="text-[10px] text-slate-500">Method: {div.method}.</p>
      </div>
      <div>
        <h3 className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Sleeve leadership (sector & country ETFs)</h3>
        <table className="w-full text-xs">
          <thead className="text-[10px] text-slate-500 uppercase tracking-wider"><tr><th className="text-left">#</th><th className="text-left">Sleeve</th><th className="text-right">12-1m</th><th className="text-right">3m</th><th className="text-right">200d</th></tr></thead>
          <tbody>
            {sleeves.map((s) => (
              <tr key={s.etf} className="border-t border-white/5">
                <td className="py-1 text-slate-500">{s.rank}</td>
                <td><span className="text-slate-200">{s.sleeve}</span> <span className="text-slate-500">{s.etf}</span></td>
                <td className={`text-right font-mono ${s.mom_12_1_pct >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>{pct(s.mom_12_1_pct)}</td>
                <td className={`text-right font-mono ${s.ret_3m_pct >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>{pct(s.ret_3m_pct)}</td>
                <td className="text-right">{s.above_200d === null || s.above_200d === undefined ? '—' : s.above_200d ? <span className="text-emerald-300">above</span> : <span className="text-rose-300">below</span>}</td>
              </tr>
            ))}
            {!sleeves.length && <tr><td colSpan={5} className="text-slate-500 py-3">No ETF history yet.</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
}
