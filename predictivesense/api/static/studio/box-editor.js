/* Shared bounding-box editor (Phase 7).
 *
 * One instance per (stage, box-element, media) triple: drag the rectangle to
 * move it, drag a corner handle to resize, arrow keys to nudge (Shift = larger
 * step). The Studio capture path (studio.js) and the bulk-upload reviewer
 * (batch.js) both use this - there is exactly one box editor, not two.
 *
 * `media()` returns `{ w, h }`, the pixel size of whatever is currently shown
 * in the stage (the <video> for the live camera, the <img> for an upload). The
 * editor keeps the box in *image* pixels and maps to/from screen coordinates
 * using the same object-fit: contain letterbox maths the stage uses.
 */
"use strict";

export function createBoxEditor({ stage, box, media, onChange }) {
  const b = { x: 0, y: 0, w: 0, h: 0 };

  function dims() {
    const m = (typeof media === "function" ? media() : media) || {};
    return { w: Number(m.w) || 0, h: Number(m.h) || 0 };
  }

  function contentRect() {
    const s = stage.getBoundingClientRect();
    const { w: mw, h: mh } = dims();
    if (!mw || !mh) return { x: 0, y: 0, w: s.width, h: s.height, scale: 1 };
    const scale = Math.min(s.width / mw, s.height / mh);
    return {
      x: (s.width - mw * scale) / 2,
      y: (s.height - mh * scale) / 2,
      w: mw * scale,
      h: mh * scale,
      scale,
    };
  }

  function clamp() {
    const { w, h } = dims();
    if (!w || !h) return;
    b.w = Math.max(8, Math.min(b.w, w));
    b.h = Math.max(8, Math.min(b.h, h));
    b.x = Math.max(0, Math.min(b.x, w - b.w));
    b.y = Math.max(0, Math.min(b.y, h - b.h));
  }

  function layout() {
    clamp();
    const c = contentRect();
    if (!box) return;
    box.style.left = `${c.x + b.x * c.scale}px`;
    box.style.top = `${c.y + b.y * c.scale}px`;
    box.style.width = `${b.w * c.scale}px`;
    box.style.height = `${b.h * c.scale}px`;
  }

  function centre() {
    const { w, h } = dims();
    if (!w || !h) {
      b.x = 0;
      b.y = 0;
      b.w = 0;
      b.h = 0;
      layout();
      return;
    }
    b.x = Math.round(w * 0.3);
    b.y = Math.round(h * 0.3);
    b.w = Math.round(w * 0.4);
    b.h = Math.round(h * 0.4);
    layout();
  }

  function setBox(arr) {
    if (arr && arr.length === 4) {
      b.x = Number(arr[0]);
      b.y = Number(arr[1]);
      b.w = Number(arr[2]);
      b.h = Number(arr[3]);
    }
    layout();
  }

  function getBox() {
    clamp();
    return [Math.round(b.x), Math.round(b.y), Math.round(b.w), Math.round(b.h)];
  }

  let wired = false;
  function wire() {
    if (wired || !box) return;
    wired = true;
    let mode = null;
    let startX = 0;
    let startY = 0;
    let orig = null;
    const toImgDelta = (dx, dy) => {
      const s = contentRect().scale || 1;
      return { dx: dx / s, dy: dy / s };
    };
    box.addEventListener("pointerdown", (ev) => {
      mode = ev.target.classList.contains("bh")
        ? [...ev.target.classList].find((c) => ["nw", "ne", "sw", "se"].includes(c))
        : "move";
      startX = ev.clientX;
      startY = ev.clientY;
      orig = { ...b };
      box.setPointerCapture?.(ev.pointerId);
      ev.preventDefault();
    });
    window.addEventListener("pointermove", (ev) => {
      if (!mode) return;
      const { dx, dy } = toImgDelta(ev.clientX - startX, ev.clientY - startY);
      if (mode === "move") {
        b.x = orig.x + dx;
        b.y = orig.y + dy;
      } else {
        if (mode.includes("w")) {
          b.x = orig.x + dx;
          b.w = orig.w - dx;
        }
        if (mode.includes("e")) b.w = orig.w + dx;
        if (mode.includes("n")) {
          b.y = orig.y + dy;
          b.h = orig.h - dy;
        }
        if (mode.includes("s")) b.h = orig.h + dy;
      }
      layout();
    });
    window.addEventListener("pointerup", () => {
      if (mode) onChange?.();
      mode = null;
    });
    box.addEventListener("keydown", (ev) => {
      const step = ev.shiftKey ? 10 : 2;
      if (ev.key === "ArrowLeft") b.x -= step;
      else if (ev.key === "ArrowRight") b.x += step;
      else if (ev.key === "ArrowUp") b.y -= step;
      else if (ev.key === "ArrowDown") b.y += step;
      else return;
      ev.preventDefault();
      onChange?.();
      layout();
    });
    window.addEventListener("resize", layout);
  }

  return { box: b, contentRect, dims, clamp, layout, centre, setBox, getBox, wire };
}
