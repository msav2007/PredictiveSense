/* Recorded video group (Recorded mode only): local file review, server file
 * list + Analyse, replay speed, analysis output. Logic lives in
 * features/videos.js.
 */
"use strict";

import { el, settingRow } from "/static/ui/controls.js";
import { REPLAY_MODES } from "/static/ui/format.js";
import { GROUP_ORDER } from "/static/groups/constants.js";
import { initVideos } from "/static/features/videos.js";

export const id = "video";
export const title = "Recorded video";
export const order = GROUP_ORDER.video;
export const modes = ["recorded"];
export const view = "normal";

export function summary(state) {
  const v = state.videoInfo;
  if (!v) return "No video selected";
  return v.duration ? `${v.name} · ${v.duration}` : v.name;
}

export function render(body) {
  const localInput = el("input", { type: "file", id: "local-video", accept: "video/*" });
  const selected = el("div", { id: "video-selected", class: "readout ellipsis", text: "No file loaded" });

  const replay = el("select", { id: "replay-mode" });
  for (const m of REPLAY_MODES) replay.append(el("option", { value: m.value, text: m.label }));

  const list = el("ul", { id: "video-list", class: "line-list" });
  const out = el("pre", { id: "analyze-out", class: "analyze-out", hidden: true });

  body.append(
    settingRow("Review a local file", localInput, "Plays in the viewport; not uploaded"),
    selected,
    settingRow("Replay speed", replay),
    el("p", { class: "subhead", text: "Files in data/videos/" }),
    list,
    out,
  );

  initVideos();
}
