import React, { useState } from 'react';

const ago = (t) => {
  if (!t) return '';
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 60 ? `${Math.round(s)}s` : s < 3600 ? `${Math.round(s / 60)}m` : `${Math.round(s / 3600)}h`;
};

const KIND_TONE = {
  ticker: 'text-slate-200 bg-slate-700/40', company: 'text-sky-200 bg-sky-500/10', org: 'text-amber-200 bg-amber-500/10',
  broker: 'text-fuchsia-200 bg-fuchsia-500/10', person: 'text-teal-200 bg-teal-500/10', place: 'text-lime-200 bg-lime-500/10',
  money: 'text-emerald-200 bg-emerald-500/10', percent: 'text-emerald-200 bg-emerald-500/10',
};
const KIND_ORDER = ['company', 'person', 'org', 'broker', 'place', 'money', 'percent'];

function Chip({ children, tone, title }) {
  return <span title={title} className={`inline-block px-1.5 py-px rounded text-[9px] leading-4 ${tone}`}>{children}</span>;
}

function Score({ sym, s }) {
  if (s.error) return <span className="text-[9px] text-rose-300" title={s.error}>{sym} error</span>;
  const pos = Math.round(s.pos * 100), neg = Math.round(s.neg * 100);
  return (
    <span className="inline-flex items-center gap-1 text-[9px] text-slate-400" title={`${s.backend}${s.ms != null ? `, ${s.ms} ms` : ''}`}>
      <span className="font-mono text-slate-300">{sym}</span>
      <span className="inline-flex h-1.5 w-12 rounded-full overflow-hidden bg-white/5">
        <span style={{ width: `${pos}%` }} className="bg-emerald-400" />
        <span style={{ width: `${Math.max(0, 100 - pos - neg)}%` }} className="bg-slate-600" />
        <span style={{ width: `${neg}%` }} className="bg-rose-400" />
      </span>
      <span className="text-emerald-300">{pos}</span>/<span className="text-rose-300">{neg}</span>
      <span className="text-slate-600">{s.backend}</span>
    </span>
  );
}

function Item({ r }) {
  const tracked = new Set(r.tracked || []);
  return (
    <li className="py-1.5 border-b border-white/5 last:border-0">
      <div className="flex items-start gap-2">
        <span className="text-[9px] text-slate-600 font-mono w-7 shrink-0 pt-0.5">{ago(r.at)}</span>
        <div className="min-w-0 flex-1">
          <div className="text-[11px] text-slate-200 leading-snug">
            <span className={`mr-1.5 text-[8px] uppercase font-bold px-1 rounded ${r.origin === 'scout' ? 'bg-emerald-500/15 text-emerald-300' : 'bg-cyan-500/15 text-cyan-300'}`}>{r.origin}</span>
            {r.headline}
            {r.source && <span className="text-[9px] text-slate-600"> · {r.source}</span>}
          </div>
          <div className="flex flex-wrap gap-1 mt-1">
            {(r.entities?.ticker || []).map((t) => (
              <Chip key={`t-${t}`} tone={tracked.has(t) ? 'text-cyan-200 bg-cyan-500/20 ring-1 ring-cyan-500/40' : KIND_TONE.ticker}
                title={tracked.has(t) ? 'tracked: scored for this symbol' : 'mentioned, not tracked'}>${t}</Chip>
            ))}
            {KIND_ORDER.flatMap((k) => (r.entities?.[k] || []).map((v) => <Chip key={`${k}-${v}`} tone={KIND_TONE[k]} title={k}>{v}</Chip>))}
            {(r.events || []).map((e) => (
              <Chip key={`e-${e.event}`} title="event keyword"
                tone={e.direction > 0 ? 'text-emerald-300 ring-1 ring-emerald-500/40' : e.direction < 0 ? 'text-rose-300 ring-1 ring-rose-500/40' : 'text-slate-300 ring-1 ring-slate-500/40'}>
                {e.direction > 0 ? '▲ ' : e.direction < 0 ? '▼ ' : '• '}{e.event}
              </Chip>
            ))}
          </div>
          {Object.keys(r.scores || {}).length > 0 ? (
            <div className="flex flex-wrap gap-x-3 gap-y-0.5 mt-1">
              {Object.entries(r.scores).map(([sym, s]) => <Score key={sym} sym={sym} s={s} />)}
            </div>
          ) : (
            <div className="text-[9px] text-slate-600 mt-0.5">{r.status === 'extracted' ? 'scoring…' : r.status}</div>
          )}
        </div>
      </div>
    </li>
  );
}

// Every headline the engine ingests: the entities and events found in it, and how it was scored.
export function NewsIngestPanel({ news }) {
  const [origin, setOrigin] = useState('all');
  if (!news) return null;
  const items = (news.items || []).filter((r) => origin === 'all' || r.origin === origin);
  const lh = news.last_hour || {};
  const feed = news.feed || {};
  return (
    <div className="bg-[#0f172a]/70 border border-slate-800 rounded-2xl p-4 shadow-xl">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">News ingestion</span>
        <span className="text-[10px] text-slate-500 font-mono">
          {feed.is_live ? `polled ${feed.seconds_since_poll ?? '—'}s ago` : 'feed offline'} · scorer {news.backend?.active || '—'}
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-slate-400 mb-2">
        <span>last hour: <b className="text-slate-200">{lh.ingested ?? 0}</b> ingested, <b className="text-slate-200">{lh.scored ?? 0}</b> scored</span>
        <span>total {news.counts?.ingested ?? 0} / scored {news.counts?.scored ?? 0}{news.counts?.errors ? ` / ${news.counts.errors} errors` : ''}</span>
        <span className="ml-auto flex gap-1">
          {['all', 'feed', 'scout'].map((o) => (
            <button key={o} onClick={() => setOrigin(o)}
              className={`px-1.5 rounded ${origin === o ? 'bg-white/10 text-slate-100' : 'text-slate-500 hover:text-slate-300'}`}>{o}</button>
          ))}
        </span>
      </div>
      {((lh.top_entities || []).length > 0 || (lh.top_events || []).length > 0) && (
        <div className="flex flex-wrap gap-1 mb-2">
          {(lh.top_events || []).map(([e, n]) => <Chip key={`ev-${e}`} tone="text-slate-300 ring-1 ring-white/10">{e} ×{n}</Chip>)}
          {(lh.top_entities || []).map(([e, n]) => <Chip key={`en-${e}`} tone="text-sky-200 bg-sky-500/10">{e} ×{n}</Chip>)}
        </div>
      )}
      <ul className="h-64 overflow-y-auto pr-1">
        {items.map((r) => <Item key={`${r.origin}-${r.id}`} r={r} />)}
        {items.length === 0 && <li className="text-[11px] text-slate-600">No headlines ingested yet.</li>}
      </ul>
    </div>
  );
}
