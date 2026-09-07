/* Group taxonomy - the stable order and the ids reserved for later phases.
 *
 * Every phase that adds a control registers a group whose `order` comes from
 * here, so the panel layout never has to be renegotiated. Reserved ids have
 * their slot held until the phase that owns them fills it - dropping in without
 * moving anything else. Phase 2 fills `analysis` (Detection + Pose sub-modules);
 * `alerts` and `research` are still reserved.
 */
"use strict";

export const GROUP_ORDER = {
  input: 10,
  camera: 20,
  video: 30,
  dataset: 40,
  analysis: 50, // Phase 2: Detection + Pose
  alerts: 60, // reserved
  research: 70, // reserved
  diagnostics: 100, // always last
};

/** Still declared but never passed to registerGroup(). `analysis` was reserved
 *  in Phase 1.6 and is filled by Phase 2 (groups/analysis.js). */
export const RESERVED_GROUP_IDS = ["alerts", "research"];

export const RESERVED_GROUPS = [
  { id: "alerts", title: "Alerts", order: GROUP_ORDER.alerts },
  { id: "research", title: "Research", order: GROUP_ORDER.research },
];
