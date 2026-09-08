/* Group taxonomy - the stable order and the ids reserved for later phases.
 *
 * Every phase that adds a control registers a group whose `order` comes from
 * here, so the panel layout never has to be renegotiated. Reserved ids have
 * their slot held until the phase that owns them fills it - dropping in without
 * moving anything else. Phase 2 fills `analysis` (Detection + Pose sub-modules);
 * Phase 2.5 fills `research` (labelling + evaluation); `alerts` is still reserved.
 */
"use strict";

export const GROUP_ORDER = {
  input: 10,
  camera: 20,
  video: 30,
  dataset: 40,
  analysis: 50, // Phase 2: Detection + Pose
  alerts: 60, // reserved
  research: 70, // Phase 2.5: labelling + evaluation
  diagnostics: 100, // always last
};

/** Still declared but never passed to registerGroup(). `analysis` was filled by
 *  Phase 2, `research` by Phase 2.5; only `alerts` stays reserved. */
export const RESERVED_GROUP_IDS = ["alerts"];

export const RESERVED_GROUPS = [
  { id: "alerts", title: "Alerts", order: GROUP_ORDER.alerts },
];
