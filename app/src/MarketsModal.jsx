import React, { useState } from 'react';

const MARKET_LABELS = {
  stocks: { title: 'Stocks', sub: 'US / EU equities, market hours', accent: 'emerald' },
  crypto: { title: 'Crypto', sub: 'Trades 24/7', accent: 'cyan' },
};

function Toggle({ on, onChange, disabled, label }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!on)}
      className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors disabled:opacity-40 ${
        on ? 'bg-emerald-500' : 'bg-slate-700'
      }`}
    >
      <span className={`inline-block h-4 w-4 rounded-full bg-white shadow transition-transform ${on ? 'translate-x-4' : 'translate-x-0.5'}`} />
    </button>
  );
}

export function MarketsModal({ isOpen, onClose, markets, apiBase, positions, onChange }) {
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);
  if (!isOpen) return null;

  const marketOn = markets?.markets || { stocks: true, crypto: true };
  const symbols = markets?.symbols || [];

  async function post(path, body, key) {
    setBusy(key);
    setError(null);
    try {
      const res = await fetch(`${apiBase}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      onChange?.(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm" onClick={onClose}>
      <div
        className="w-full max-w-lg max-h-[85vh] flex flex-col rounded-2xl bg-slate-950 ring-1 ring-white/10 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between px-5 pt-5 pb-3 border-b border-white/5">
          <div>
            <h2 className="text-base font-semibold text-slate-100">Markets</h2>
            <p className="text-[11px] text-slate-500 mt-0.5">
              Switching a market or symbol off stops new entries. Open positions still exit on their stops, targets and strategy exits.
            </p>
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-white text-lg leading-none px-1" aria-label="Close">×</button>
        </div>

        <div className="overflow-y-auto px-5 py-4 space-y-5">
          {error && <div className="text-xs text-rose-300 bg-rose-500/10 ring-1 ring-rose-500/30 rounded-lg px-3 py-2">{error}</div>}

          {Object.entries(MARKET_LABELS).map(([market, meta]) => {
            const on = marketOn[market] !== false;
            const rows = symbols.filter((s) => s.market === market);
            return (
              <section key={market} className="space-y-2">
                <div className={`flex items-center justify-between rounded-xl px-3 py-2.5 ring-1 ${on ? 'bg-white/[0.03] ring-white/10' : 'bg-rose-500/5 ring-rose-500/20'}`}>
                  <div>
                    <div className="text-sm font-semibold text-slate-100">{meta.title}</div>
                    <div className="text-[11px] text-slate-500">{on ? meta.sub : 'Off: no new entries'}</div>
                  </div>
                  <Toggle
                    on={on}
                    label={`${meta.title} trading`}
                    disabled={busy !== null}
                    onChange={(v) => post('/api/markets', { market, enabled: v }, market)}
                  />
                </div>

                {rows.length > 0 && (
                  <ul className={`divide-y divide-white/5 rounded-xl ring-1 ring-white/5 ${on ? '' : 'opacity-50'}`}>
                    {rows.map((s) => (
                      <li key={s.symbol} className="flex items-center justify-between px-3 py-1.5">
                        <span className="font-mono text-xs text-slate-200 flex items-center gap-2">
                          {s.symbol}
                          {positions?.[s.symbol] && (
                            <span className="text-[9px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-1 rounded">HELD</span>
                          )}
                        </span>
                        <Toggle
                          on={s.enabled}
                          label={`${s.symbol} trading`}
                          disabled={busy !== null || !on}
                          onChange={(v) => post('/api/markets/symbol', { symbol: s.symbol, enabled: v }, s.symbol)}
                        />
                      </li>
                    ))}
                  </ul>
                )}
              </section>
            );
          })}
        </div>
      </div>
    </div>
  );
}
