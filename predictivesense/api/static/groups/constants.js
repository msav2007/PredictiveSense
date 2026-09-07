/* Group taxonomy - the stable order and the ids reserved for later phases.
 *
 * Every phase that adds a control registers a group whose `order` comes from
 * here, so the panel layout never has to be renegotiated. `analysis`, `alerts`
 * and `research` have no content yet and are NOT registered in Phase 1.6 - their
 * slots are reserved so a later phase drops in without moving anything.
 */
"use strict";

export const GROUP_ORDER = {
  input: 10,
  camera: 20,
  video: 30,
  dataset: 40,
  analysis: 50, // reserved
  alerts: 60, // reserved
  research: 70, // reserved
  diagnostics: 100, // always last
};

/** Declared but never passed to registerGroup() in this phase. */
export const RESERVED_GROUP_IDS = ["analysis", "alerts", "research"];

export const RESERVED_GROUPS = [
  { id: "analysis", title: "Analysis", order: GROUP_ORDER.analysis },
  { id: "alerts", title: "Alerts", order: GROUP_ORDER.alerts },
  { id: "research", title: "Research", order: GROUP_ORDER.research },
];
