/* Research group (Phase 2.5 fills the slot reserved in Phase 1.6).
 *
 * Links to the labelling tool, a labelling-progress summary (frames labelled,
 * per class, per session, seeded vs unseeded), and the latest evaluation summary
 * read from results/. It registers into the existing registry - the shell, the
 * registry and the group taxonomy are unchanged (constants.js just moved
 * `research` out of the still-reserved list, exactly as Phase 2 did for
 * `analysis`). A missing results file shows "not yet run" (BLOCK 10).
 */
"use strict";

import { el, actionButton } from "/static/ui/controls.js";
import { GROUP_ORDER } from "/static/groups/constants.js";

export const id = "research";
export const title = "Research";
export const order = GROUP_ORDER.research;
export const modes = ["realtime", "recorded"];
export const view = "normal";

export function summary() {
  // Pure of store state (no research data is in the store); the live counts
  // live in the body and refresh on mount / on the Refresh button.
  return "Labelling & evaluation";
}

export function render(body) {
  body.append(
    el("p", {}, [
      el("a", { href: "/label", target: "_blank", text: "Open the labelling tool →" }),
    ]),
    el("p", { class: "subhead", text: "Labelling progress" }),
    el("div", { id: "research-progress", class: "kv-list" }, [
      el("div", {}, [el("dt", { text: "status" }), el("dd", { text: "loading…" })]),
    ]),
    el("p", { class: "subhead", text: "Per class" }),
    el("div", { id: "research-per-class", class: "kv-list" }),
    el("p", { class: "subhead", text: "Per session" }),
    el("div", { id: "research-per-session", class: "kv-list" }),
    el("p", { class: "subhead", text: "Latest evaluation" }),
    el("div", { id: "research-eval", class: "kv-list" }, [
      el("div", {}, [el("dt", { text: "status" }), el("dd", { text: "loading…" })]),
    ]),
    actionButton("Refresh", { variant: "secondary", id: "research-refresh", onClick: refresh }),
  );
  refresh();
}

function kvRows(target, pairs) {
  const node = document.getElementById(target);
  if (!node) return;
  node.replaceChildren(
    ...pairs.map(([k, v]) =>
      el("div", {}, [el("dt", { text: String(k) }), el("dd", { text: String(v) })]),
    ),
  );
}

async function refresh() {
  try {
    const p = await fetch("/api/labels/progress").then((r) => (r.ok ? r.json() : Promise.reject(r.status)));
    kvRows("research-progress", [
      ["frames sampled", p.images],
      ["labelled", `${p.labelled} / ${p.images}`],
      ["seeded / unseeded", `${p.seeded} / ${p.unseeded}`],
      ["unseeded fraction", `${(p.unseeded_fraction * 100).toFixed(0)}% (target ≥ ${(p.min_unseeded_fraction * 100).toFixed(0)}%)`],
      ["meets unseeded target", p.meets_unseeded_target ? "yes" : "no"],
      ["sessions", p.sessions],
      ["boxes", p.annotations],
      ["target frames", p.target_frames],
    ]);
    kvRows(
      "research-per-class",
      Object.entries(p.per_class || {}).map(([k, v]) => [k, v]),
    );
    kvRows(
      "research-per-session",
      Object.entries(p.per_session || {}).map(([k, v]) => [
        k.slice(0, 10),
        `${v.labelled}/${v.frames} labelled · ${v.unseeded} unseeded`,
      ]),
    );
  } catch (err) {
    kvRows("research-progress", [["status", `not available (${err}) — run scripts/build_eval_frames.py`]]);
    kvRows("research-per-class", []);
    kvRows("research-per-session", []);
  }

  try {
    const e = await fetch("/api/labels/eval-summary").then((r) => r.json());
    if (!e.available) {
      kvRows("research-eval", [["status", e.reason || "not yet run"]]);
      return;
    }
    const sc = e.sample_counts || {};
    kvRows("research-eval", [
      ["file", e.file],
      ["model", e.model],
      ["split", e.split],
      ["policy", e.policy_enabled ? "on" : "off"],
      ["false-class rate", e.false_class_rate ?? "—"],
      ["macro F1", e.macro_f1 ?? "—"],
      ["mAP@0.5", e.map50 ?? "—"],
      ["samples", `${sc.images ?? "?"} img · ${sc.ground_truth_boxes ?? "?"} gt · ${sc.matched_pairs ?? "?"} matched`],
    ]);
  } catch (err) {
    kvRows("research-eval", [["status", `not yet run (${err})`]]);
  }
}
