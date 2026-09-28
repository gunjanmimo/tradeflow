import React, { useEffect, useMemo, useRef, useState } from 'react';
import { HugeiconsIcon } from '@hugeicons/react';
import { AiBrain01Icon } from '@hugeicons/core-free-icons';

// Chart colours: categorical slots 1-2 (dark steps), validated against the
// dashboard surface #0f1624 (scripts/validate_palette.js: all checks pass).
const SERIES = { train: '#3987e5', val: '#d95926' };

const MODES = [
  { id: 'auto', label: 'Auto', hint: 'Trades only a policy that passed its promotion gate; otherwise shadow.' },
  { id: 'shadow', label: 'Shadow', hint: 'Decides and logs every decision, never places an order.' },
  { id: 'live', label: 'Live', hint: 'Trades whatever policy is deployed, even one that failed the gate.' },
  { id: 'off', label: 'Off', hint: 'The policy is not consulted.' },
];

const POLICY_LABEL = { ppo: 'PPO policy', flat: 'Flat (never trade)', always_long: 'Always long', random: 'Random, same turnover' };
const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d));
const signed = (v, d = 2) => (v === null || v === undefined ? '—' : `${v > 0 ? '+' : ''}${Number(v).toFixed(d)}`);
const barTime = (bar) => {
  const m = 9 * 60 + 30 + bar + 1;
  return `${Math.floor(m / 60)}:${String(m % 60).padStart(2, '0')}`;
};

// Header chip: RL mode, and whether the deployed policy may trade.
export function RLChip({ apiBase, onClick }) {
  const [s, setS] = useState(null);
  useEffect(() => {
    let alive = true;
    const load = () => fetch(`${apiBase}/api/rl`).then((r) => r.json()).then((d) => alive && setS(d)).catch(() => {});
    load();
    const id = setInterval(load, 15000);
    return () => { alive = false; clearInterval(id); };
  }, []);
  const trading = s?.may_trade;
  const tone = !s?.loaded ? 'text-slate-400' : trading ? 'text-emerald-300' : 'text-amber-300';
  const label = !s ? '…' : !s.loaded ? 'no policy' : trading ? 'trading' : s.mode === 'off' ? 'off' : 'shadow';
  return (
    <button
      onClick={onClick}
      className="group flex items-center gap-2.5 rounded-xl px-3 py-1.5 bg-white/[0.03] ring-1 ring-white/10 hover:ring-cyan-400/50 hover:bg-cyan-400/5 transition-all text-left"
      title={s?.approved ? 'Deployed policy passed its promotion gate' : 'Deployed policy has not passed its promotion gate'}
    >
      <HugeiconsIcon icon={AiBrain01Icon} size="1em" className={`w-4 h-4 ${tone}`} />
      <div className="flex flex-col min-w-[70px]">
        <span className="text-[10px] uppercase tracking-wider text-slate-500 leading-none">RL policy</span>
        <span className={`text-sm font-semibold leading-tight ${tone}`}>
          {label}<span className="text-[10px] text-slate-500 font-normal ml-1">{s?.mode || ''}</span>
        </span>
      </div>
    </button>
  );
}

function Check({ ok, label }) {
  return (
    <li className="flex items-center gap-2 text-xs">
      <span className={`w-4 text-center font-bold ${ok ? 'text-emerald-400' : 'text-rose-400'}`} aria-hidden>{ok ? '✓' : '✗'}</span>
      <span className="text-slate-300">{label}</span>
      <span className={`ml-auto text-[10px] uppercase tracking-wider ${ok ? 'text-emerald-400' : 'text-rose-400'}`}>{ok ? 'pass' : 'fail'}</span>
    </li>
  );
}

const CHECK_LABEL = {
  test_positive: 'Makes money on the unseen test days',
  test_significant: 'Test result is statistically significant (t ≥ min)',
  beats_always_long: 'Beats simply holding (Sharpe vs always long)',
  val_positive: 'Makes money on the validation days too',
  trades: 'Actually trades',
};

function ResultsTable({ title, rows }) {
  if (!rows) return null;
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">{title}</div>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-slate-500 border-b border-white/5">
            <th className="text-left font-normal py-1">Policy</th>
            <th className="text-right font-normal">bps/day</th>
            <th className="text-right font-normal">t</th>
            <th className="text-right font-normal">Sharpe</th>
            <th className="text-right font-normal">trades/session</th>
            <th className="text-right font-normal">time in market</th>
          </tr>
        </thead>
        <tbody className="font-mono">
          {['ppo', 'flat', 'always_long', 'random'].filter((k) => rows[k]).map((k) => {
            const r = rows[k];
            return (
              <tr key={k} className={`border-b border-white/5 ${k === 'ppo' ? 'text-slate-100' : 'text-slate-400'}`}>
                <td className="py-1 font-sans">{POLICY_LABEL[k]}</td>
                <td className="text-right">{signed(r.mean_daily_bps)}</td>
                <td className="text-right">{signed(r.t_daily)}</td>
                <td className="text-right">{signed(r.sharpe)}</td>
                <td className="text-right">{num(r.trades_per_session)}</td>
                <td className="text-right">{(r.exposure * 100).toFixed(1)}%</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// Learning curve: train return per session and validation return per day, both in bps.
function LearningCurve({ curves, best }) {
  const [hover, setHover] = useState(null);
  const [asTable, setAsTable] = useState(false);
  const ref = useRef(null);
  const wrap = useRef(null);
  const [W, setW] = useState(640);
  // Drawn at the container's real pixel width, so text stays 10-11px instead
  // of scaling with a viewBox.
  useEffect(() => {
    if (!wrap.current) return undefined;
    const ro = new ResizeObserver(([e]) => setW(Math.max(320, Math.round(e.contentRect.width))));
    ro.observe(wrap.current);
    return () => ro.disconnect();
  }, [asTable]);
  const H = 200, P = { l: 44, r: 72, t: 14, b: 24 };
  const pts = useMemo(() => (curves || []).filter((c) => c.iteration), [curves]);
  if (!pts.length) return <div className="text-xs text-slate-500">No training curve recorded.</div>;
  const train = pts.map((c) => [c.iteration, c.train_ret_bps]);
  const val = pts.filter((c) => c.val_daily_bps !== undefined).map((c) => [c.iteration, c.val_daily_bps]);
  const xs = pts.map((c) => c.iteration);
  const ys = [...train, ...val].map((p) => p[1]).concat([0]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
  const pad = (Math.max(...ys) - Math.min(...ys)) * 0.08 || 1;
  const y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad;
  const sx = (x) => P.l + ((x - x0) / (x1 - x0)) * (W - P.l - P.r);
  const sy = (y) => H - P.b - ((y - y0) / (y1 - y0)) * (H - P.t - P.b);
  const path = (s) => s.map((p, i) => `${i ? 'L' : 'M'}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join('');
  // Gridlines at round-ish values; none so close to the zero line that labels collide.
  const ticks = [y0 + (y1 - y0) * 0.1, (y0 + y1) / 2, y1 - (y1 - y0) * 0.1]
    .filter((t) => Math.abs(sy(t) - sy(0)) > 14);

  const onMove = (e) => {
    const r = ref.current.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * W;
    let bestI = 0;
    pts.forEach((c, i) => { if (Math.abs(sx(c.iteration) - x) < Math.abs(sx(pts[bestI].iteration) - x)) bestI = i; });
    setHover(pts[bestI]);
  };
  const lastT = train[train.length - 1], lastV = val[val.length - 1];

  return (
    <div>
      <div className="flex items-center gap-4 mb-1 text-[11px] text-slate-400">
        <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 rounded" style={{ background: SERIES.train }} />Training days (bps per session)</span>
        <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full" style={{ background: SERIES.val }} />Validation days (bps per day)</span>
        <button className="ml-auto text-slate-500 hover:text-slate-200 underline decoration-dotted" onClick={() => setAsTable((v) => !v)}>
          {asTable ? 'Show chart' : 'Show table'}
        </button>
      </div>
      {asTable ? (
        <div className="max-h-48 overflow-y-auto">
          <table className="w-full text-xs font-mono">
            <thead><tr className="text-slate-500"><th className="text-left font-normal">iteration</th><th className="text-right font-normal">train bps/session</th><th className="text-right font-normal">val bps/day</th></tr></thead>
            <tbody>{pts.map((c) => (
              <tr key={c.iteration} className="text-slate-300"><td>{c.iteration}</td><td className="text-right">{signed(c.train_ret_bps)}</td><td className="text-right">{c.val_daily_bps === undefined ? '' : signed(c.val_daily_bps, 3)}</td></tr>
            ))}</tbody>
          </table>
        </div>
      ) : (
        <div className="relative" ref={wrap}>
          <svg ref={ref} width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block" onPointerMove={onMove} onPointerLeave={() => setHover(null)} role="img"
               aria-label="Learning curve: training and validation return by iteration">
            {ticks.map((t) => (
              <g key={t}>
                <line x1={P.l} x2={W - P.r} y1={sy(t)} y2={sy(t)} stroke="rgba(255,255,255,0.06)" />
                <text x={P.l - 6} y={sy(t) + 3} textAnchor="end" fontSize="10" fill="#64748b">{t.toFixed(1)}</text>
              </g>
            ))}
            <line x1={P.l} x2={W - P.r} y1={sy(0)} y2={sy(0)} stroke="rgba(255,255,255,0.25)" strokeDasharray="3 3" />
            <text x={P.l - 6} y={sy(0) + 3} textAnchor="end" fontSize="10" fill="#94a3b8">0</text>
            {best ? <line x1={sx(best)} x2={sx(best)} y1={P.t} y2={H - P.b} stroke="rgba(255,255,255,0.18)" /> : null}
            {best ? <text x={sx(best) - 4} y={P.t + 9} fontSize="10" fill="#94a3b8" textAnchor="end">picked</text> : null}
            <path d={path(train)} fill="none" stroke={SERIES.train} strokeWidth="2" strokeLinejoin="round" />
            {val.length > 1 && <path d={path(val)} fill="none" stroke={SERIES.val} strokeWidth="2" strokeLinejoin="round" />}
            {val.map((p) => <circle key={p[0]} cx={sx(p[0])} cy={sy(p[1])} r="4" fill={SERIES.val} stroke="#0f1624" strokeWidth="2" />)}
            <text x={sx(lastT[0]) + 6} y={sy(lastT[1]) + 3} fontSize="10" fill="#cbd5e1">train</text>
            {lastV && <text x={sx(lastV[0]) + 6} y={sy(lastV[1]) + 3} fontSize="10" fill="#cbd5e1">validation</text>}
            <text x={P.l} y={H - 8} fontSize="10" fill="#64748b">iteration {x0}</text>
            <text x={W - P.r} y={H - 8} fontSize="10" fill="#64748b" textAnchor="end">{x1}</text>
            {hover && <line x1={sx(hover.iteration)} x2={sx(hover.iteration)} y1={P.t} y2={H - P.b} stroke="rgba(255,255,255,0.35)" />}
          </svg>
          {hover && (
            <div className="pointer-events-none absolute top-6 rounded-lg bg-slate-900/95 ring-1 ring-white/10 px-2.5 py-1.5 text-[11px] text-slate-300 whitespace-nowrap"
                 style={sx(hover.iteration) < W / 2 ? { left: sx(hover.iteration) + 12 } : { right: W - sx(hover.iteration) + 12 }}>
              <div className="text-slate-500">iteration {hover.iteration}</div>
              <div><span className="font-mono font-semibold text-slate-100">{signed(hover.train_ret_bps)}</span> train bps/session</div>
              {hover.val_daily_bps !== undefined && <div><span className="font-mono font-semibold text-slate-100">{signed(hover.val_daily_bps, 3)}</span> validation bps/day</div>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function RLPanel({ apiBase }) {
  const [s, setS] = useState(null);
  const [err, setErr] = useState(null);
  const [armLive, setArmLive] = useState(false);
  const [busy, setBusy] = useState(false);

  const load = () => fetch(`${apiBase}/api/rl`).then((r) => r.json()).then(setS).catch((e) => setErr(String(e)));
  useEffect(() => { load(); const id = setInterval(load, 5000); return () => clearInterval(id); }, []);

  const setMode = async (mode) => {
    if (mode === 'live' && !s?.approved && !armLive) { setArmLive(true); return; }
    setArmLive(false);
    setBusy(true);
    try {
      const r = await fetch(`${apiBase}/api/rl/mode`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode }) });
      if (!r.ok) throw new Error((await r.json()).detail);
      await load();
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };
  const retrain = async () => {
    setBusy(true);
    try { await fetch(`${apiBase}/api/rl/retrain`, { method: 'POST' }); await load(); } finally { setBusy(false); }
  };

  if (!s) return <div className="text-sm text-slate-400">{err || 'Loading…'}</div>;
  const g = s.gate || {};
  const learner = s.learner || {};

  return (
    <div className="space-y-5">
      <p className="text-xs text-slate-400 leading-relaxed">
        A PPO agent decides every 5 minutes whether to be long or flat, trained on {s.n_sessions ? s.n_sessions.toLocaleString() : '—'} historical
        sessions of {s.n_symbols ?? '—'} stocks under the same fills, stops, costs and close as live trading.
        It is judged on days it never trained on, against doing nothing, holding, and trading at random.
      </p>

      {!s.loaded ? (
        <div className="rounded-xl bg-amber-500/10 ring-1 ring-amber-500/30 px-4 py-3 text-sm text-amber-200">
          No trained policy yet. Run <code className="font-mono">python -m research download</code> then <code className="font-mono">python -m rl.train</code> in <code className="font-mono">api/</code>.
        </div>
      ) : (
        <div className={`rounded-xl px-4 py-3 ring-1 text-sm ${s.may_trade ? 'bg-emerald-500/10 ring-emerald-500/30 text-emerald-200' : 'bg-amber-500/10 ring-amber-500/30 text-amber-200'}`}>
          <span className="font-semibold">{s.may_trade ? 'Trading (paper).' : 'Shadow mode: the policy decides and logs, it places no orders.'}</span>{' '}
          {s.approved ? 'The deployed policy passed its promotion gate.' : 'The deployed policy did not pass its promotion gate.'}
          {s.mode === 'live' && !s.approved && ' Live mode overrides the gate: it trades a policy that lost money on unseen days.'}
          <span className="block text-[11px] text-slate-400 mt-1">Trained {s.trained_at || '—'} · {s.deploy_reason || ''}</span>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[10px] uppercase tracking-wider text-slate-500 mr-1">Mode</span>
        {MODES.map((m) => (
          <button key={m.id} disabled={busy} onClick={() => setMode(m.id)} title={m.hint}
            className={`px-3 py-1 rounded-lg text-xs ring-1 transition-all ${s.mode === m.id ? 'bg-cyan-500 text-slate-950 ring-cyan-400 font-semibold' : 'text-slate-300 ring-white/10 hover:ring-cyan-400/50'}`}>
            {m.label}
          </button>
        ))}
        {armLive && <span className="text-xs text-rose-300">This policy failed its gate. Click Live again to trade it anyway.</span>}
      </div>

      <div className="grid md:grid-cols-2 gap-5">
        <div>
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Promotion gate {g.min_t ? `(min t ${g.min_t})` : ''}</div>
          <ul className="space-y-1">
            {Object.entries(g.checks || {}).map(([k, v]) => <Check key={k} ok={v} label={CHECK_LABEL[k] || k} />)}
          </ul>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Learner</div>
          <div className="text-xs text-slate-300">{learner.summary || '—'}</div>
          {!s.torch_available && <div className="text-[11px] text-slate-500 mt-1">torch is not installed in this environment: retrain on the host with <code className="font-mono">python -m rl.train --warm-start</code>.</div>}
          <button disabled={busy || learner.retraining || !s.torch_available} onClick={retrain}
            className="mt-2 px-3 py-1 rounded-lg text-xs ring-1 ring-white/10 text-slate-300 hover:ring-cyan-400/50 disabled:opacity-40">
            {learner.retraining ? 'Retraining…' : 'Retrain now'}
          </button>
          {(learner.log_tail || []).length > 0 && (
            <pre className="mt-2 max-h-24 overflow-y-auto text-[10px] text-slate-500 whitespace-pre-wrap">{learner.log_tail.join('\n')}</pre>
          )}
        </div>
      </div>

      <div>
        <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Learning curve {s.best_iteration ? `· checkpoint picked on validation at iteration ${s.best_iteration}` : ''}</div>
        <LearningCurve curves={s.curves} best={s.best_iteration} />
      </div>

      <div className="grid lg:grid-cols-2 gap-5">
        <ResultsTable title={`Test: ${s.splits?.test?.[2] ?? '—'} newest days, never trained on`} rows={s.test} />
        <ResultsTable title={`Validation: ${s.splits?.val?.[2] ?? '—'} days that picked the checkpoint`} rows={s.val} />
      </div>

      <div>
        <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Recent decisions</div>
        {(s.recent_decisions || []).length === 0 ? (
          <div className="text-xs text-slate-500">None yet today: the policy decides at the close of 09:34, 09:39, … on watchlist stocks.</div>
        ) : (
          <table className="w-full text-xs">
            <tbody className="font-mono">
              {[...s.recent_decisions].reverse().slice(0, 20).map((d, i) => (
                <tr key={i} className="border-b border-white/5 text-slate-300">
                  <td className="py-1">{d.symbol}</td>
                  <td>{d.day} {barTime(d.bar)}</td>
                  <td className={d.action === 1 ? 'text-slate-100 font-semibold' : 'text-slate-400'}>{d.action === 1 ? 'LONG' : 'FLAT'}</td>
                  <td className="text-right">p(long) {num(d.p_long)}</td>
                  <td className="text-right text-slate-500 font-sans">{d.in_position ? 'held' : ''} {d.may_trade ? 'trading' : 'shadow'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
