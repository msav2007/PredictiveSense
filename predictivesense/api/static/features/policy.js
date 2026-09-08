/* Recognition-policy overlay presentation (Phase 2.5, redesigned Phase 5).
 *
 * The authoritative policy runs in the backend analysis loop from
 * `config.policy` (shown read-only). These per-viewer toggles change only how
 * the overlay PRESENTS each detection - they re-derive from the additive
 * Detection fields (`raw_class_name`, `policy_state`, `tier`, `runner_up`)
 * already on the snapshot, so no API route is added (same pattern as
 * analysis-prefs.js):
 *
 *   policy      off -> overlay shows the raw model label, ignoring policy_state
 *   domain      off -> a detection suppressed ONLY as implausible shows its raw label
 *   margin      off -> a detection sent to `unknown` ONLY by the margin rule shows raw
 *   suppressed  on  -> reveal `suppressed_implausible` boxes (Diagnostics only,
 *                      BLOCK 3.8) - muted, labelled with the raw class + "suppressed"
 *
 * `effectiveDetection(d)` returns { label, kind, reason, raw } for the overlay.
 * kind: "accepted" | "secondary" | "unknown" | "suppressed" | "rejected".
 */
"use strict";

import { runtime } from "/static/features/runtime.js";

const KEY = "ps.policyview";
const listeners = new Set();
let state = null;

const UNKNOWN_STATES = new Set(["unknown_low_confidence", "unknown_margin"]);

function defaults() {
  const p = runtime.config?.policy || {};
  return {
    policy: p.enabled !== false,
    domain: p.domain_restriction !== false,
    margin: p.margin_rule !== false,
    // Diagnostics-only reveal; default from config but off unless config opts in.
    suppressed: false,
  };
}

function load() {
  const base = defaults();
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      for (const k of ["policy", "domain", "margin", "suppressed"]) {
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

/** Whether thresholds have been fitted on a labelled val split (BLOCK 3.5). */
export function thresholdsFitted() {
  return runtime.config?.policy?.thresholds_fitted === true;
}

/** Read-only effective per-class thresholds from config (primary + secondary). */
export function activeThresholds() {
  const p = runtime.config?.policy || {};
  const v = p.vocabulary || {};
  const dflt = p.default_threshold ?? 0.35;
  const out = {};
  for (const c of [...(v.primary || []), ...(v.secondary || [])].sort()) {
    out[c] = (p.per_class_thresholds && p.per_class_thresholds[c]) ?? dflt;
  }
  return out;
}

/** Effective threshold used for one detection's raw class. */
export function thresholdFor(d) {
  const p = runtime.config?.policy || {};
  const raw = d.raw_class_name || d.class_name;
  return (p.per_class_thresholds && p.per_class_thresholds[raw]) ?? (p.default_threshold ?? 0.35);
}

/** Human-readable name of the rule that produced a policy_state. */
export function ruleLabel(policyState) {
  return (
    {
      accepted: "—",
      accepted_secondary: "secondary tier (shown, de-emphasised)",
      unknown_low_confidence: "below per-class confidence threshold",
      unknown_margin: "top-1 / top-2 margin below margin_min",
      suppressed_implausible: "implausible tier (suppressed)",
      rejected_size: "box too small / aspect implausible",
    }[policyState] || policyState || "—"
  );
}

/** How the overlay should draw one detection given the viewer's toggles. */
export function effectiveDetection(d) {
  const s = ensure();
  const raw = d.raw_class_name || d.class_name;
  const st = d.policy_state || "accepted";
  const base = { raw, reason: st, tier: d.tier || "primary" };

  if (!s.policy) return { ...base, label: raw, kind: "accepted" };
  if (st === "accepted") return { ...base, label: d.class_name || raw, kind: "accepted" };
  if (st === "accepted_secondary") return { ...base, label: d.class_name || raw, kind: "secondary" };
  if (st === "unknown_margin" && !s.margin) return { ...base, label: raw, kind: "accepted" };
  if (st === "suppressed_implausible" && !s.domain) return { ...base, label: raw, kind: "accepted" };
  if (UNKNOWN_STATES.has(st)) return { ...base, label: "Unknown", kind: "unknown" };
  if (st === "suppressed_implausible") return { ...base, label: "Unknown", kind: "suppressed" };
  if (st === "rejected_size") return { ...base, label: "Unknown", kind: "rejected" };
  return { ...base, label: d.class_name || raw, kind: "accepted" };
}
