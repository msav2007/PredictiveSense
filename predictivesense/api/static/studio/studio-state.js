/* Object Learning Studio - explicit state machine (Phase 5).
 *
 * The Phase 4 Studio guessed at implicit state (loose flags on a `state`
 * object), which produced the "Back to camera loses the box", "re-selecting an
 * object leaves stale inspect controls" and "Save & Return silently drops a
 * staged upload" bugs. This module makes the state explicit and testable
 * without a browser:
 *
 *   browsing         no object selected (empty state)
 *   object_selected  object chosen; live camera + box editor at rest
 *   capturing        a candidate image is staged in the box editor (upload flow)
 *   reviewing        inspecting a previously-saved sample (retag / re-box)
 *
 * Transitions: select, capture, review, back_to_camera, save_and_return, discard.
 * One owner of the state; the DOM is a pure function of (state, context).
 *
 * Phase 6: `save_and_return` is reachable from every state (including a clean
 * `browsing` return - Phase 5 omitted it and `dispatch` threw an uncaught error
 * from `browsing`, which the async click handler swallowed and the "Save and
 * return" button went dead). A refused transition is recorded in
 * `lastRefused` and surfaced (Diagnostics readout + a visible note) rather than
 * failing silently.
 */
"use strict";

export const STUDIO_STATES = ["browsing", "object_selected", "capturing", "reviewing"];
export const STUDIO_TRANSITIONS = [
  "select",
  "capture",
  "review",
  "back_to_camera",
  "save_and_return",
  "discard",
];

function freshContext() {
  return {
    objectId: null,
    stage: "camera", // "camera" | "upload" | "inspect"
    editingSampleId: null,
    pendingUploads: 0, // staged, not yet saved
    dirtyInspect: false, // inspect open with unsaved box/tag edits
  };
}

// from-state -> allowed transitions
const TABLE = {
  browsing: { select: "object_selected", save_and_return: "browsing" },
  object_selected: {
    select: "object_selected",
    capture: "capturing",
    review: "reviewing",
    save_and_return: "browsing",
  },
  capturing: {
    select: "object_selected",
    capture: "capturing",
    review: "reviewing",
    back_to_camera: "object_selected",
    discard: "object_selected",
    save_and_return: "browsing",
  },
  reviewing: {
    select: "object_selected",
    review: "reviewing",
    back_to_camera: "object_selected",
    discard: "object_selected",
    save_and_return: "browsing",
  },
};

export function createStudioState() {
  let state = "browsing";
  let ctx = freshContext();
  // The last transition the UI asked for and could not have (missing from the
  // table, or blocked by `hasPending`). Observable in the Diagnostics readout;
  // cleared by the next successful transition, so it can never latch.
  let lastRefused = null;
  const listeners = new Set();

  function notify() {
    for (const fn of listeners) {
      try {
        fn(snapshot());
      } catch {
        /* one bad listener must not break the rest */
      }
    }
  }

  function snapshot() {
    return { state, context: { ...ctx }, lastRefused };
  }

  /** Anything unsaved that a navigation-away would discard (BLOCK 3.19). */
  function hasPending() {
    return ctx.pendingUploads > 0 || ctx.dirtyInspect;
  }

  function can(transition) {
    if (!(transition in (TABLE[state] || {}))) return false;
    if (transition === "save_and_return") return !hasPending();
    return true;
  }

  /** Record a transition the UI asked for but could not have. Does not throw;
   *  the caller is expected to show the user why (BLOCK 5.13 / 14). */
  function refuse(transition) {
    lastRefused = transition;
    notify();
    return { ...snapshot(), blocked: true };
  }

  function dispatch(transition, payload = {}) {
    const target = (TABLE[state] || {})[transition];
    if (target === undefined) {
      throw new Error(`invalid transition ${transition} from ${state}`);
    }
    if (transition === "save_and_return" && hasPending()) {
      // caller must resolve the pending work (or force via discard) first
      lastRefused = transition;
      notify();
      return { ...snapshot(), blocked: "pending" };
    }

    switch (transition) {
      case "select": {
        const objectId = payload.objectId;
        if (!objectId) throw new Error("select requires payload.objectId");
        ctx = freshContext();
        ctx.objectId = objectId;
        break;
      }
      case "capture": {
        ctx.stage = "upload";
        ctx.editingSampleId = null;
        ctx.pendingUploads = Math.max(1, Number(payload.count) || 1);
        ctx.dirtyInspect = false;
        break;
      }
      case "review": {
        if (!payload.sampleId) throw new Error("review requires payload.sampleId");
        ctx.stage = "inspect";
        ctx.editingSampleId = payload.sampleId;
        ctx.pendingUploads = 0;
        ctx.dirtyInspect = false;
        break;
      }
      case "back_to_camera":
      case "discard": {
        // preserve the selected object; drop only the transient capture/inspect
        const keepObject = ctx.objectId;
        ctx = freshContext();
        ctx.objectId = keepObject;
        break;
      }
      case "save_and_return": {
        ctx = freshContext();
        break;
      }
      default:
        break;
    }
    state = target;
    lastRefused = null; // a successful transition clears any prior refusal
    notify();
    return snapshot();
  }

  // -- fine-grained pending tracking (not transitions) -----------------
  function setPending({ uploads, dirtyInspect } = {}) {
    if (typeof uploads === "number") ctx.pendingUploads = Math.max(0, uploads);
    if (typeof dirtyInspect === "boolean") ctx.dirtyInspect = dirtyInspect;
    notify();
  }

  return {
    get state() {
      return state;
    },
    get context() {
      return { ...ctx };
    },
    get lastRefused() {
      return lastRefused;
    },
    snapshot,
    can,
    dispatch,
    refuse,
    hasPending,
    setPending,
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}
