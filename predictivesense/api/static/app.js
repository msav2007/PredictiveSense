/* PredictiveSense dashboard - composition root.
 *
 * This file wires the application together and does nothing else: fetch config,
 * register the panel groups, mount the shell, start the feature modules, and
 * connect the store to the registry. All camera / worker / recording / video /
 * metrics logic lives under features/; all panel controls live under groups/.
 *
 * Preview independence, the newest-wins mailbox and the worker backpressure
 * behaviour are unchanged - see features/camera-capture.js and
 * static/analysis-worker.js.
 */
"use strict";

import { store } from "/static/ui/store.js";
import { runtime } from "/static/features/runtime.js";
import { mountShell } from "/static/ui/shell.js";
import { note } from "/static/ui/log.js";
import { registerGroup, mount as mountRegistry, applyFilters, refresh } from "/static/ui/registry.js";

import * as inputGroup from "/static/groups/input.js";
import * as cameraGroup from "/static/groups/camera.js";
import * as videoGroup from "/static/groups/video.js";
import * as datasetGroup from "/static/groups/dataset.js";
import * as analysisGroup from "/static/groups/analysis.js";
import * as researchGroup from "/static/groups/research.js";
import * as diagnosticsGroup from "/static/groups/diagnostics.js";

import { initAnalysisClient } from "/static/features/analysis-client.js";
import {
  initCameraCapture,
  releaseCapture,
  resumeCapture,
  assertPreviewUncomposited,
} from "/static/features/camera-capture.js";
import { showRecordedViewport, clearRecordedViewport } from "/static/features/videos.js";

const GROUPS = [
  inputGroup,
  cameraGroup,
  videoGroup,
  datasetGroup,
  analysisGroup,
  researchGroup,
  diagnosticsGroup,
];

async function main() {
  runtime.config = await fetch("/api/config").then((r) => r.json());
  runtime.owner = runtime.config.capture?.owner ?? "browser";

  for (const g of GROUPS) registerGroup(g);

  mountShell(document.getElementById("app-shell"), { panel: runtime.config.ui?.panel });
  mountRegistry(document.getElementById("panel-body"), { runtime });

  initAnalysisClient();
  assertPreviewUncomposited();
  await initCameraCapture();

  if (store.get().mode === "recorded") showRecordedViewport();

  let lastMode = store.get().mode;
  store.subscribe((state) => {
    applyFilters();
    refresh(state);
    if (state.mode !== lastMode) {
      lastMode = state.mode;
      if (state.mode === "recorded") {
        releaseCapture();
        showRecordedViewport();
      } else {
        clearRecordedViewport();
        resumeCapture();
      }
    }
  });

  applyFilters();
  refresh(store.get());
}

main().catch((err) => {
  store.setStatus("error");
  note(`init failed: ${err}`);
});
