import React, { useState } from 'react';

const money = (v) => `$${Number(v || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

// Header chip: shows the active capital mode, and stair progress when on.
export function CapitalModeChip({ plan, onClick }) {
  const stair = plan?.mode === 'stair';
  return (
    <button
      onClick={onClick}
      className="group flex items-center gap-2.5 rounded-xl px-3 py-1.5 bg-white/[0.03] ring-1 ring-white/10 hover:ring-emerald-400/50 hover:bg-emerald-400/5 transition-all text-left"
      title="Capital mode: classic budget or stair (profit ratchet)"
    >
      <StairIcon className={`w-4 h-4 ${stair ? 'text-emerald-400' : 'text-slate-500'}`} />
      <div className="flex flex-col min-w-[74px]">
        <span className="text-[10px] uppercase tracking-wider text-slate-500 leading-none">
          {stair ? `Stair · step ${plan.stage}` : 'Capital mode'}
        </span>
        <span className="text-sm font-semibold leading-tight text-white">
          {stair ? <span className="font-mono text-emerald-300">{money(plan.banked_income)}</span> : 'Classic'}
          {stair && <span className="text-[10px] text-slate-500 font-normal ml-1">banked</span>}
        </span>
        {stair && (
          <div className="w-full h-0.5 bg-white/10 rounded-full overflow-hidden">
            <div style={{ width: `${(plan.progress || 0) * 100}%` }} className="h-full bg-emerald-400 transition-all duration-300" />
          </div>
        )}
      </div>
      {stair && plan.halted && <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse" title="Stair halted: trading capital too small" />}
    </button>
  );
}

function StairIcon({ className }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden="true">
      <path d="M3 20h5v-5h5v-5h5V5h3" />
    </svg>
  );
}

// Modal body: mode selector, stair settings, ladder progress and history.
// The tabs only SELECT a mode; the button below applies it and closes the modal.
const AMOUNT_RE = /^\d*(\.\d{0,2})?$/;   // digits, optionally one "." and up to 2 decimals

export function CapitalPlanPanel({ plan, apiBase, cash, onDone }) {
  const running = plan?.mode === 'stair';
  const [selected, setSelected] = useState(plan?.mode || 'classic');
  const [deposit, setDeposit] = useState(plan?.deposit ? String(plan.deposit) : '400');
  const [depositNote, setDepositNote] = useState(null);
  const [deployPct, setDeployPct] = useState(Math.round((plan?.deploy_pct ?? 0.5) * 100));
  const [targetMult, setTargetMult] = useState(plan?.target_multiple ?? 2);
  const [harvestPct, setHarvestPct] = useState(Math.round((plan?.harvest_pct ?? 0.5) * 100));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const stairSelected = selected === 'stair';

  // Reject anything that is not a plain number as it is typed or pasted,
  // rather than accepting it and failing later.
  const onDepositChange = (e) => {
    const v = e.target.value.trim();
    if (AMOUNT_RE.test(v)) {
      setDeposit(v);
      setDepositNote(null);
    } else {
      setDepositNote('Numbers only: digits and an optional decimal point (e.g. 400 or 400.50).');
    }
  };
  const depositValue = parseFloat(deposit);
  const depositValid = deposit !== '' && deposit !== '.' && Number.isFinite(depositValue) && depositValue > 0;

  const submit = async () => {
    if (stairSelected && !depositValid) {
      setError('Enter a deposit amount greater than 0.');
      return;
    }
    setBusy(true); setError(null);
    try {
      const body = stairSelected
        ? { mode: 'stair', deposit: depositValue, deploy_pct: deployPct / 100, target_multiple: parseFloat(targetMult), harvest_pct: harvestPct / 100 }
        : { mode: 'classic' };
      const r = await fetch(`${apiBase}/api/capital-plan`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      });
      if (!r.ok) {
        setError((await r.json()).detail || 'Request failed');
        return;
      }
      onDone && onDone();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  // Preview of the first few steps with the current inputs (assumes each stage hits its target).
  const preview = [];
  let cap = (depositValid ? depositValue : 0) * (deployPct / 100);
  let banked = 0;
  for (let i = 1; i <= 4 && cap > 0; i++) {
    const reached = cap * targetMult;
    const bank = (reached - cap) * (harvestPct / 100);
    banked += bank;
    preview.push({ stage: i, start: cap, target: reached, bank, banked });
    cap = reached - bank;
  }

  return (
    // p-1: rings are drawn outside each box, and the scroll container would
    // otherwise clip them at its edges.
    <div className="space-y-4 max-h-[75vh] overflow-y-auto p-1">
      {/* Mode selector */}
      <div className="grid grid-cols-2 gap-1 p-1 rounded-xl bg-white/[0.03] ring-1 ring-white/10" role="tablist">
        {[['classic', 'Classic'], ['stair', 'Stair']].map(([key, label]) => (
          <button key={key} role="tab" aria-selected={selected === key} disabled={busy} onClick={() => { setSelected(key); setError(null); }}
            className={`py-2 rounded-lg text-sm font-semibold transition-all ${selected === key
              ? (key === 'stair' ? 'bg-emerald-500/20 text-emerald-300 ring-1 ring-emerald-400/40' : 'bg-white/10 text-white')
              : 'text-slate-400 hover:text-white'}`}>
            {label}{plan?.mode === key && <span className="ml-1.5 text-[10px] font-normal opacity-70">(active)</span>}
          </button>
        ))}
      </div>

      {stairSelected ? (
        <p className="text-xs text-slate-400 leading-relaxed">
          <b className="text-slate-200">Stair</b> trades only part of your deposit and keeps the rest as a reserve it never touches.
          Each time the trading capital reaches its target on <i>closed</i> trades, part of that step's profit is banked as income
          and the rest keeps compounding. <b className="text-amber-300">Main broker equity outside your deposit is locked and untouched.</b> Banked income and the reserve stay as cash at the broker.
        </p>
      ) : (
        <p className="text-xs text-slate-400 leading-relaxed">
          <b className="text-slate-200">Classic</b> trades only your fixed budget cap. <b className="text-amber-300">Main broker equity outside the budget cap is locked and untouched.</b> Profit stays in broker cash and never increases the hard cap.
          {running && ' Switching stops the stair ladder; banked income stays recorded and the previous budget is restored.'}
        </p>
      )}

      {running && (
        <div className="rounded-xl bg-emerald-400/[0.04] ring-1 ring-emerald-400/20 p-3 space-y-3">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            <Cell label="Trading" value={money(plan.trading_capital)} />
            <Cell label="Reserve" value={money(plan.reserve)} sub="untouched" />
            <Cell label="Banked income" value={money(plan.banked_income)} tone="text-emerald-300" />
            <Cell label="Total" value={money(plan.total_value)} sub={`deposit ${money(plan.deposit)}`} />
          </div>
          <div>
            <div className="flex justify-between text-[11px] text-slate-400 mb-1">
              <span>Step {plan.stage}: {money(plan.stage_start_capital)} → {money(plan.target)}</span>
              <span className="font-mono">{((plan.progress || 0) * 100).toFixed(0)}%</span>
            </div>
            <div className="h-2 bg-white/10 rounded-full overflow-hidden">
              <div className="h-full bg-emerald-400 transition-all duration-500" style={{ width: `${(plan.progress || 0) * 100}%` }} />
            </div>
            <div className="text-[10px] text-slate-500 mt-1">Progress counts realised PnL only; open positions don't move it until they close.</div>
          </div>
          {plan.halted && (
            <div className="text-xs text-rose-300">Halted: trading capital is too small to place an order. Reserve and income were not used. Restart the ladder to continue.</div>
          )}
          {plan.history?.length > 0 && (
            <table className="w-full text-[11px] font-mono">
              <thead className="text-slate-500"><tr><th className="text-left font-normal">Step</th><th className="text-right font-normal">Start</th><th className="text-right font-normal">Reached</th><th className="text-right font-normal">Banked</th><th className="text-right font-normal">Next</th></tr></thead>
              <tbody>
                {plan.history.map((h) => (
                  <tr key={`${h.ladder}-${h.stage}`} className="border-t border-white/5 text-slate-300">
                    <td>{h.ladder > 1 ? `L${h.ladder}·` : ''}{h.stage}</td><td className="text-right">{money(h.start_capital)}</td><td className="text-right">{money(h.reached_capital)}</td>
                    <td className="text-right text-emerald-300">{money(h.banked)}</td><td className="text-right">{money(h.next_capital)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {stairSelected && (
        <>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Deposit ($)" hint={cash != null ? `broker cash ${money(cash)}` : ''}>
              <input
                type="text" inputMode="decimal" autoComplete="off" spellCheck={false}
                value={deposit} onChange={onDepositChange}
                aria-invalid={!depositValid}
                className={`w-full bg-black/30 ring-1 rounded-lg px-3 py-2 font-mono text-sm text-white outline-none ${depositValid ? 'ring-white/10 focus:ring-emerald-400/50' : 'ring-rose-400/60'}`}
              />
              {depositNote && <div className="text-[10px] text-rose-300 mt-1">{depositNote}</div>}
            </Field>
            <Field label={`Trade ${deployPct}% · reserve ${100 - deployPct}%`}>
              <input type="range" min="10" max="100" step="5" value={deployPct} onChange={(e) => setDeployPct(+e.target.value)} className="w-full accent-emerald-400" />
            </Field>
            <Field label={`Step target ${Number(targetMult).toFixed(1)}x`}>
              <input type="range" min="1.2" max="4" step="0.1" value={targetMult} onChange={(e) => setTargetMult(+e.target.value)} className="w-full accent-emerald-400" />
            </Field>
            <Field label={`Bank ${harvestPct}% of each step's profit`}>
              <input type="range" min="0" max="100" step="5" value={harvestPct} onChange={(e) => setHarvestPct(+e.target.value)} className="w-full accent-emerald-400" />
            </Field>
          </div>

          {preview.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">If every step reaches its target</div>
              <table className="w-full text-[11px] font-mono">
                <thead className="text-slate-500"><tr><th className="text-left font-normal">Step</th><th className="text-right font-normal">Trade with</th><th className="text-right font-normal">Target</th><th className="text-right font-normal">Bank</th><th className="text-right font-normal">Total banked</th></tr></thead>
                <tbody>
                  {preview.map((r) => (
                    <tr key={r.stage} className="border-t border-white/5 text-slate-300">
                      <td>{r.stage}</td><td className="text-right">{money(r.start)}</td><td className="text-right">{money(r.target)}</td>
                      <td className="text-right text-emerald-300">{money(r.bank)}</td><td className="text-right">{money(r.banked)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="text-[10px] text-slate-500 mt-1">Illustration only. Steps are not guaranteed to complete, and a step can lose money.</div>
            </div>
          )}
        </>
      )}

      {error && <div className="text-xs text-rose-300 bg-rose-500/10 ring-1 ring-rose-500/30 rounded-lg p-2">{error}</div>}

      <button disabled={busy || (stairSelected && !depositValid)} onClick={submit}
        className={`w-full py-2.5 rounded-xl text-sm font-semibold ring-1 disabled:opacity-50 ${stairSelected
          ? 'bg-emerald-500/20 text-emerald-300 ring-emerald-400/40 hover:bg-emerald-500/30'
          : 'bg-white/10 text-white ring-white/15 hover:bg-white/15'}`}>
        {busy ? 'Applying…' : stairSelected ? 'Start stair mode' : 'Start classic mode'}
      </button>
      {stairSelected && running && (
        <div className="text-[10px] text-slate-500 text-center -mt-2">Stair is already running: starting it again begins a new ladder at step 1. Banked income and past steps are kept.</div>
      )}
    </div>
  );
}

// One figure per tile, with its own border, so values never sit against a clipped edge.
function Cell({ label, value, sub, tone = 'text-white' }) {
  return (
    <div className="min-w-0 rounded-lg bg-black/20 ring-1 ring-white/10 px-3 py-2 text-center">
      <div className="text-[9px] uppercase tracking-wider text-slate-500 truncate">{label}</div>
      <div className={`text-sm font-mono font-semibold tabular-nums truncate ${tone}`} title={value}>{value}</div>
      {sub && <div className="text-[9px] text-slate-500 truncate">{sub}</div>}
    </div>
  );
}

function Field({ label, hint, children }) {
  return (
    <label className="block">
      <div className="flex justify-between text-[11px] text-slate-400 mb-1"><span>{label}</span><span className="text-slate-500">{hint}</span></div>
      {children}
    </label>
  );
}
