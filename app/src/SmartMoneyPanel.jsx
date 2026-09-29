import React, { useState } from 'react';

const money = (v) => (v >= 1e6 ? `$${(v / 1e6).toFixed(1)}M` : `$${Math.round(v / 1e3)}k`);

function Row({ r, right }) {
  return (
    <li className="flex items-start justify-between gap-3 py-1.5 border-b border-white/5 last:border-0">
      <div className="min-w-0">
        <span className="font-mono font-bold text-sm text-slate-100">{r.symbol}</span>
        <span className="ml-2 text-[11px] text-slate-400">{(r.reasons || []).join(' · ') || '—'}</span>
      </div>
      <div className="shrink-0 text-[11px] text-right">{right}</div>
    </li>
  );
}

function Section({ title, hint, rows, render, tone, collapsed = false }) {
  const [open, setOpen] = useState(!collapsed);
  if (!rows.length) return null;
  return (
    <section>
      <button onClick={() => setOpen((o) => !o)} className="w-full flex items-center justify-between text-left mb-1">
        <span className={`text-[11px] uppercase tracking-wider font-semibold ${tone}`}>
          {title} <span className="text-slate-500 font-normal">({rows.length})</span>
        </span>
        <span className="text-[10px] text-slate-500">{hint} {open ? '▾' : '▸'}</span>
      </button>
      {open && <ul>{rows.map((r) => <Row key={r.symbol} r={r} right={render(r)} />)}</ul>}
    </section>
  );
}

// The smart-money book: what the bots trade from SEC Form 4 and eToro, and why the rest is not traded.
export function SmartMoneyPanel({ data, sources }) {
  const rows = data?.rows || [];
  const rules = data?.rules || {};
  const trading = rows.filter((r) => r.verdict === 'BUY' && r.tradable);
  const notTradable = rows.filter((r) => r.verdict === 'BUY' && !r.tradable);
  const hold = rows.filter((r) => r.verdict === 'HOLD');
  const avoid = rows.filter((r) => r.verdict === 'AVOID');
  return (
    <div className="space-y-4">
      <p className="text-[11px] text-slate-400 leading-relaxed">
        A <span className="text-slate-200">buy</span> is an insider open-market purchase of at least {money(rules.min_buy_usd || 100000)} in
        the last {rules.lookback_days ?? 5} days, or ownership by at least {Math.round((rules.min_breadth || 0.2) * 100)}% of the top eToro investors.
        Buys that are liquid US stocks (≥ ${rules.min_price ?? 5}, ≥ {money(rules.min_dollar_volume || 1e7)}/day, no leveraged ETFs or crypto) are
        {data?.trading_enabled === false ? ' listed but trading is switched off.' : ' put on the watchlist and day-traded by the smart-money strategy.'}
      </p>
      {rows.length === 0 && (
        <div className="text-xs text-slate-500">No smart-money signals yet {sources ? `(${sources})` : ''}.</div>
      )}
      <Section title="Trading" hint="day trade, flat by the close" tone="text-emerald-300" rows={trading}
        render={(r) => (r.held ? <span className="text-emerald-300 font-semibold">HELD</span>
          : <span className="text-slate-400">{r.on_watchlist ? 'watching for entry' : 'adding…'}</span>)} />
      <Section title="Buy signal, not tradable" hint="why not" tone="text-amber-300" rows={notTradable}
        render={(r) => <span className="text-amber-300/90 max-w-[220px] inline-block">{r.why_not}</span>} />
      <Section title="Avoid" hint="insiders selling" tone="text-rose-300" rows={avoid} collapsed
        render={(r) => <span className="text-slate-400">sold {money(r.insider_sold)}</span>} />
      <Section title="No signal" hint="mentioned, not a buy" tone="text-slate-400" rows={hold} collapsed
        render={() => <span className="text-slate-500">—</span>} />
    </div>
  );
}
