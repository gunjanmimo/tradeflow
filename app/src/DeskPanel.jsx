import React, { useState, useEffect, useRef } from 'react';
import { clockTime } from './timefmt';

const num = (v, d = 2) => (v === null || v === undefined ? '—' : Number(v).toFixed(d));
const signed = (v, d = 2) => (v === null || v === undefined ? '—' : `${v >= 0 ? '+' : ''}${Number(v).toFixed(d)}%`);
const clock = (t) => clockTime(t);

const STEPS = [['observing', 'Observe'], ['analyst', 'Analyst'], ['critic', 'Critic'], ['decision', 'Decision']];
const STEP_INDEX = { observing: 0, queued: 0, analyst: 1, critic: 2 };

const VERDICT_TONE = {
  approved: 'bg-emerald-500/15 text-emerald-300 ring-emerald-500/40',
  executed: 'bg-emerald-500/25 text-emerald-200 ring-emerald-400/60',
  rejected: 'bg-rose-500/15 text-rose-300 ring-rose-500/40',
  faded: 'bg-slate-500/15 text-slate-300 ring-slate-500/40',
  expired: 'bg-amber-500/15 text-amber-300 ring-amber-500/40',
  error: 'bg-rose-500/15 text-rose-300 ring-rose-500/40',
};

const AGENT_TONE = (s) => (['thinking', 'answering', 'watching', 'working'].includes(s) ? 'bg-cyan-400 animate-pulse'
  : s === 'offline' ? 'bg-rose-500' : 'bg-slate-600');

function Agents({ agents }) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
      {agents.map((a) => (
        <div key={a.name} className="rounded-lg bg-slate-950/70 ring-1 ring-white/5 px-2.5 py-2 min-w-0">
          <div className="flex items-center gap-1.5">
            <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${AGENT_TONE(a.state)}`} />
            <span className="text-[11px] font-semibold text-slate-200">{a.name}</span>
            {a.model && <span className="text-[9px] text-slate-500 truncate">{a.model}</span>}
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5 truncate" title={a.summary}>{a.summary}</div>
        </div>
      ))}
    </div>
  );
}

function Stepper({ stage }) {
  const at = STEP_INDEX[stage] ?? 3;
  return (
    <div className="flex items-center gap-1 text-[10px]">
      {STEPS.map(([key, label], i) => (
        <React.Fragment key={key}>
          <span className={`px-1.5 py-0.5 rounded ${i < at ? 'text-emerald-300/80' : i === at ? 'bg-cyan-500/15 text-cyan-200 ring-1 ring-cyan-500/40' : 'text-slate-600'}`}>
            {i < at ? '✓ ' : ''}{label}
          </span>
          {i < STEPS.length - 1 && <span className="text-slate-700">›</span>}
        </React.Fragment>
      ))}
    </div>
  );
}

function Spark({ path }) {
  if (!path || path.length < 2) return null;
  const lo = Math.min(...path, 0), hi = Math.max(...path, 0), span = hi - lo || 1;
  const pts = path.map((v, i) => `${(i / (path.length - 1)) * 100},${28 - ((v - lo) / span) * 26 - 1}`).join(' ');
  const zero = 28 - ((0 - lo) / span) * 26 - 1;
  const up = path[path.length - 1] >= 0;
  return (
    <svg viewBox="0 0 100 28" preserveAspectRatio="none" className="w-24 h-7 shrink-0">
      <line x1="0" x2="100" y1={zero} y2={zero} stroke="currentColor" className="text-slate-700" strokeWidth="0.6" strokeDasharray="2 2" />
      <polyline points={pts} fill="none" strokeWidth="1.4" className={up ? 'stroke-emerald-400' : 'stroke-rose-400'} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

// Probabilities on one 0..1 track: the bar a trade must clear, and where each agent put it.
function Odds({ c, rules = {} }) {
  const d = c.decision || {};
  const pa = c.analyst?.answer?.p_target_first ?? d.p_analyst;
  const pc = c.critic?.answer?.p_target_first ?? d.p_critic;
  const marks = [['signal', c.signal_prob, 'bg-slate-400'], ['analyst', pa, 'bg-cyan-400'], ['critic', pc, 'bg-violet-400'],
    ['final', d.p_final, 'bg-white']].filter(([, v]) => v !== undefined && v !== null);
  const minP = rules.min_prob ?? 0.55, margin = rules.edge_margin ?? 0.05;
  const need = d.need ?? (c.breakeven_p != null ? Math.max(minP, c.breakeven_p + margin) : null);
  return (
    <div className="mt-2">
      <div className="relative h-2 rounded-full bg-white/5">
        {need != null && <div style={{ left: `${need * 100}%` }} className="absolute -top-0.5 h-3 w-px bg-amber-300" title={`bar ${num(need)}`} />}
        {c.breakeven_p != null && <div style={{ width: `${c.breakeven_p * 100}%` }} className="absolute h-full rounded-l-full bg-rose-500/20" title={`breakeven ${num(c.breakeven_p)}`} />}
        {marks.map(([k, v, tone]) => (
          <div key={k} style={{ left: `calc(${v * 100}% - 4px)` }} className={`absolute top-0 w-2 h-2 rounded-full ${tone} ring-2 ring-slate-950`} title={`${k} ${num(v)}`} />
        ))}
      </div>
      <div className="flex flex-wrap gap-x-3 mt-1 text-[10px] text-slate-400">
        <span>signal <b className="text-slate-200">{num(c.signal_prob)}</b></span>
        <span className="text-cyan-300">analyst <b>{num(pa)}</b></span>
        <span className="text-violet-300">critic <b>{num(pc)}</b></span>
        <span>mean <b className="text-white">{num(d.p_final)}</b></span>
        <span className="text-amber-300">bar {num(need)}</span>
        <span className="text-rose-300/80">breakeven {num(c.breakeven_p)}</span>
      </div>
    </div>
  );
}

function Thinking({ slot, label }) {
  const ref = useRef(null);
  useEffect(() => { if (ref.current) ref.current.scrollTop = ref.current.scrollHeight; }, [slot?.thinking_tail]);
  if (!slot || !slot.thinking_tail) return null;
  return (
    <div className="mt-2">
      <div className="text-[10px] text-slate-500 mb-0.5">
        {label} {slot.status === 'thinking' ? 'reasoning…' : slot.status}
        {slot.elapsed_s != null && ` · ${slot.elapsed_s}s`} · {slot.thinking_chars} chars
      </div>
      <pre ref={ref} className="max-h-28 overflow-y-auto whitespace-pre-wrap text-[10px] leading-snug text-slate-400 bg-slate-950 rounded-md p-2 ring-1 ring-white/5">
        {slot.thinking_tail}
      </pre>
    </div>
  );
}

function Timeline({ items, n = 6 }) {
  return (
    <ul className="mt-2 space-y-0.5">
      {(items || []).slice(-n).map((e, i) => (
        <li key={i} className="text-[10px] leading-snug flex gap-1.5">
          <span className="text-slate-600 font-mono shrink-0">{clock(e.t)}</span>
          <span className={`shrink-0 font-semibold ${e.agent === 'Analyst' ? 'text-cyan-300' : e.agent === 'Critic' ? 'text-violet-300' : e.agent === 'Decision' ? 'text-amber-200' : 'text-slate-300'}`}>{e.agent}</span>
          <span className="text-slate-400">{e.text}</span>
        </li>
      ))}
    </ul>
  );
}

function ActiveCase({ c, rules }) {
  const obs = c.observation || {};
  const pct = Math.min(100, ((obs.seconds || 0) / (c.observe_s || rules.observe_s || 1)) * 100);
  return (
    <div className="rounded-xl bg-slate-900/80 ring-1 ring-cyan-500/20 p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-baseline gap-2">
          <span className="font-mono font-bold text-slate-100">{c.symbol}</span>
          <span className="text-[10px] text-slate-500">{c.strategy}</span>
        </div>
        <Stepper stage={c.stage} />
      </div>
      {c.stage === 'observing' || c.stage === 'queued' ? (
        <div className="flex items-center gap-3 mt-2">
          <div className="flex-1">
            <div className="h-1.5 rounded-full bg-white/5 overflow-hidden"><div style={{ width: `${pct}%` }} className="h-full bg-cyan-400/80 transition-all" /></div>
            <div className="text-[10px] text-slate-400 mt-1">
              {c.stage === 'queued' ? 'waiting for the analyst' : `observing ${Math.round(obs.seconds || 0)}/${Math.round(c.observe_s)}s`}
              {' · '}{signed(obs.move_pct)} since signal · signal {Math.round((obs.persistence || 0) * 100)}%
              {obs.above_vwap_share != null && ` · above VWAP ${Math.round(obs.above_vwap_share * 100)}%`}
            </div>
          </div>
          <Spark path={obs.path} />
        </div>
      ) : null}
      <Odds c={c} rules={rules} />
      <Thinking slot={c.stage === 'critic' ? c.critic : c.analyst} label={c.stage === 'critic' ? 'Critic' : 'Analyst'} />
      <Timeline items={c.timeline} n={4} />
    </div>
  );
}

function RecentCase({ c, apiBase, rules }) {
  const [open, setOpen] = useState(false);
  const [full, setFull] = useState(null);
  const verdict = c.stage;
  useEffect(() => {
    if (!open || full) return;
    fetch(`${apiBase}/api/desk/case/${encodeURIComponent(c.id)}`).then((r) => (r.ok ? r.json() : null)).then(setFull).catch(() => {});
  }, [open, full, apiBase, c.id]);
  const a = c.analyst?.answer, cr = c.critic?.answer;
  return (
    <li className="py-1.5 border-b border-white/5 last:border-0">
      <button onClick={() => setOpen((o) => !o)} className="w-full text-left flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className={`text-[9px] font-bold uppercase px-1.5 py-0.5 rounded ring-1 ${VERDICT_TONE[verdict] || VERDICT_TONE.faded}`}>{verdict}</span>
          <span className="font-mono font-bold text-xs text-slate-200">{c.symbol}</span>
          <span className="text-[10px] text-slate-500 truncate">{c.decision?.reason || a?.thesis || ''}</span>
        </div>
        <span className="text-[10px] text-slate-500 shrink-0">
          {c.decision?.p_final != null ? `P ${num(c.decision.p_final)} · ` : ''}{clock(c.closed_at || c.stage_at)} {open ? '▾' : '▸'}
        </span>
      </button>
      {open && (
        <div className="pl-2 mt-1">
          <Odds c={c} rules={rules} />
          {a && <div className="text-[10px] text-cyan-200/90 mt-2"><b>Analyst P {num(a.p_target_first)}, {(a.lean || "").toLowerCase()}</b> ({a.confidence}): {a.thesis}</div>}
          {cr && (
            <div className="text-[10px] text-violet-200/90 mt-1">
              <b>Critic {cr.verdict}</b>: {cr.summary}
              {(cr.objections || []).length > 0 && <ul className="list-disc pl-4 text-slate-400">{cr.objections.map((o, i) => <li key={i}>{o}</li>)}</ul>}
            </div>
          )}
          <Timeline items={c.timeline} n={12} />
          {full?.analyst?.thinking && (
            <details className="mt-1">
              <summary className="text-[10px] text-slate-500 cursor-pointer">Analyst's full reasoning ({full.analyst.thinking.length} chars)</summary>
              <pre className="max-h-48 overflow-y-auto whitespace-pre-wrap text-[10px] text-slate-400 bg-slate-950 rounded-md p-2 mt-1">{full.analyst.thinking}</pre>
            </details>
          )}
          {full?.brief_text && (
            <details className="mt-1">
              <summary className="text-[10px] text-slate-500 cursor-pointer">Case file the agents read</summary>
              <pre className="max-h-48 overflow-y-auto whitespace-pre-wrap text-[10px] text-slate-400 bg-slate-950 rounded-md p-2 mt-1">{full.brief_text}</pre>
            </details>
          )}
        </div>
      )}
    </li>
  );
}

// The trade desk, live over the telemetry websocket: every entry is observed, then argued by LLM agents.
export function DeskPanel({ desk, apiBase }) {
  const [sym, setSym] = useState('');
  const [msg, setMsg] = useState(null);
  if (!desk) return null;
  const rules = desk.rules || {};
  const counts = desk.counts || {};
  const ask = async (e) => {
    e.preventDefault();
    const s = sym.trim().toUpperCase();
    if (!s) return;
    setMsg(null);
    const r = await fetch(`${apiBase}/api/desk/review/${encodeURIComponent(s)}`, { method: 'POST' });
    const body = await r.json().catch(() => ({}));
    setMsg(r.ok ? `Desk is reviewing ${s}` : body.detail || `HTTP ${r.status}`);
    if (r.ok) setSym('');
  };
  return (
    <div className="bg-[#0f172a]/70 border border-cyan-900/40 rounded-2xl p-4 shadow-xl">
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-baseline gap-2">
          <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">Trade desk</span>
          <span className="text-[10px] text-slate-500">
            observe {rules.observe_s}s → analyst → critic → decision
          </span>
        </div>
        <span className={`text-[10px] px-2 py-0.5 rounded font-mono ring-1 ${desk.llm?.ok ? 'bg-emerald-500/10 text-emerald-300 ring-emerald-500/30' : 'bg-rose-500/10 text-rose-300 ring-rose-500/30'}`}
          title={desk.llm?.detail}>
          {!desk.enabled ? 'OFF' : desk.llm?.ok ? 'LLM READY' : 'LLM OFFLINE'}
        </span>
      </div>
      {desk.enabled && !desk.llm?.ok && (
        <div className="text-[11px] text-rose-300/90 mb-2">
          {desk.llm?.detail}. {desk.required ? 'No trade opens until the desk can review it.' : ''}
        </div>
      )}
      <Agents agents={desk.agents || []} />

      {(desk.active || []).length > 0 ? (
        <div className="space-y-2">{desk.active.map((c) => <ActiveCase key={c.id} c={c} rules={rules} />)}</div>
      ) : (
        <div className="text-[11px] text-slate-500">No signal under review. The desk opens a case when the manager ranks a buy signal.</div>
      )}

      {(desk.cleared || []).length > 0 && (
        <div className="mt-2 text-[11px] text-emerald-300">
          Cleared to buy: {desk.cleared.map((c) => `${c.symbol} (P ${num(c.decision?.p_final)}, until ${clock(c.cleared_until)})`).join(', ')}
        </div>
      )}

      <div className="flex items-center justify-between mt-3 mb-1">
        <span className="text-[10px] uppercase tracking-wider text-slate-500">
          Verdicts · {counts.approved || 0} approved · {counts.rejected || 0} rejected · {counts.faded || 0} faded · {counts.executed || 0} traded
        </span>
        <form onSubmit={ask} className="flex items-center gap-1">
          <input value={sym} onChange={(e) => setSym(e.target.value)} placeholder="ASK THE DESK"
            className="w-24 bg-slate-900 border border-slate-700 rounded px-1.5 py-0.5 text-[10px] uppercase font-mono text-slate-200 placeholder-slate-600 focus:outline-none focus:border-cyan-500" />
          <button type="submit" className="text-[10px] px-2 py-0.5 rounded bg-cyan-500/80 hover:bg-cyan-400 text-slate-950 font-semibold">Review</button>
        </form>
      </div>
      {msg && <div className="text-[10px] text-slate-400 mb-1">{msg}</div>}
      <ul className="max-h-64 overflow-y-auto">
        {(desk.recent || []).map((c) => <RecentCase key={c.id + c.stage} c={c} apiBase={apiBase} rules={rules} />)}
        {(desk.recent || []).length === 0 && <li className="text-[11px] text-slate-600">No verdicts yet.</li>}
      </ul>
      <div className="text-[9px] text-slate-600 mt-2">
        Probabilities are the models' own estimates of P(target before stop), not calibrated odds. Every case and the P&amp;L of what it cleared are logged to data/desk/cases.jsonl.
      </div>
    </div>
  );
}
