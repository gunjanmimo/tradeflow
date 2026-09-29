// Every clock time on the dashboard is shown in the operator's timezone
// (settings.DISPLAY_TIMEZONE on the backend, sent as market_clock.display_tz),
// whatever the viewing device's own timezone. Market logic never uses it.
let displayTz = 'Europe/Paris';
const cache = {};

export function setDisplayTz(tz) {
  if (tz && tz !== displayTz) {
    try {
      new Intl.DateTimeFormat('en-GB', { timeZone: tz });
      displayTz = tz;
    } catch {
      /* unknown zone: keep the current one */
    }
  }
}

function fmt(seconds) {
  const key = `${displayTz}|${seconds}`;
  if (!cache[key]) {
    cache[key] = new Intl.DateTimeFormat('en-GB', {
      timeZone: displayTz, hour: '2-digit', minute: '2-digit', ...(seconds ? { second: '2-digit' } : {}), hour12: false,
    });
  }
  return cache[key];
}

// Epoch seconds -> "14:05:09" (or "14:05") in the display timezone.
export function clockTime(epochSeconds, seconds = true) {
  if (!epochSeconds) return '';
  return fmt(seconds).format(new Date(epochSeconds * 1000));
}
