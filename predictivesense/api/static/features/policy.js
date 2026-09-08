/* Recognition-policy overlay controls (Phase 2.5).
 *
 * The authoritative policy runs in the backend analysis loop from
 * `config.policy` (shown read-only). These per-viewer toggles change only how
 * the overlay PRESENTS each detection - they re-derive from the additive
 * Detection fields (`raw_class_name`, `policy_state`) already on the snapshot,
 * so no API route is added (same pattern as analysis-prefs.js):
 *
 *   policy     off -> overlay shows the raw model label, ignoring policy_state
 *   domain     off -> a detection rejected ONLY as out-of-domain shows its raw label
 *   margin     off -> a detection sent to `unknown` ONLY by the margin rule shows raw
 *
 * `effectiveDetection(d)` returns { label, kind } for the overlay to draw.
 */
"use strict";

import { runtime } from "/static/features/runtime.js";

const KEY = "ps.policyview";
const listeners = new Set();
let state = null;

function defaults() {
  const p = runtime.config?.policy || {};
  return {
    policy: p.enabled !== false,
    domain: p.domain_restriction !== false,
    margin: p.margin_rule !== false,
  };
}

function load() {
  const base = defaults();
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      for (const k of ["policy", "domain", "margin"]) {
        if (typeof parsed[k] === "boolean") base[k] = parsed[k];
      }
    }
  } catch {
    /* private mode - config defaults */
  }
  return base;
}

function ensure() {
  if (state === null) state = load();
  return state;
}

export function isPolicyView(id) {
  return !!ensure()[id];
}

export function setPolicyView(id, on) {
  const s = ensure();
  if (s[id] === !!on) return;
  s[id] = !!on;
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* ignore */
  }
  for (const fn of listeners) {
    try {
      fn(s);
    } catch {
      /* one bad listener must not break the rest */
    }
  }
}

export function onPolicyViewChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** Read-only effective per-class thresholds from config. */
export function activeThresholds() {
  const p = runtime.config?.policy || {};
  const out = {};
  const dflt = p.default_threshold ?? 0.35;
  for (const c of p.domain_classes || []) {
    out[c] = (p.per_class_thresholds && p.per_class_thresholds[c]) ?? dflt;
  }
  return out;
}

/** How the overlay should draw one detection given the viewer's toggles.
 *  kind: "accepted" | "unknown" | "rejected". */
export function effectiveDetection(d) {
  const s = ensure();
  const raw = d.raw_class_name || d.class_name;
  const st = d.policy_state || "accepted";

  if (!s.policy) return { label: raw, kind: "accepted" };
  if (st === "accepted") return { label: d.class_name, kind: "accepted" };
  if (st === "rejected_out_of_domain" && !s.domain) return { label: raw, kind: "accepted" };
  if (st === "unknown_margin" && !s.margin) return { label: raw, kind: "accepted" };

  const kind = st.startsWith("rejected") ? "rejected" : "unknown";
  return { label: "Unknown", kind, reason: st, raw };
}
