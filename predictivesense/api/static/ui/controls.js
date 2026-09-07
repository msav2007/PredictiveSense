/* Shared UI primitives. Every group builds its body from these so spacing,
 * button hierarchy and metric rows look identical everywhere.
 */
"use strict";

/** Minimal hyperscript. `props.class`, `props.text`, `props.html`, everything
 * else is set as an attribute or (for on*) an event listener. */
export function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === undefined || v === null) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (v === true) node.setAttribute(k, "");
    else if (v === false) {
      /* skip */
    } else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c == null) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return node;
}

/** label + control (+ optional hint) on one row. */
export function settingRow(labelText, control, hint) {
  const id = control.id || `ctl-${Math.random().toString(36).slice(2, 8)}`;
  control.id = id;
  const label = el("label", { class: "setting-label", for: id, text: labelText });
  const row = el("div", { class: "setting-row" }, [label, control]);
  if (hint) row.append(el("span", { class: "setting-hint", text: hint }));
  return row;
}

const VARIANTS = new Set(["primary", "secondary", "danger"]);

/** Action button with a visible hierarchy variant. */
export function actionButton(text, { variant = "secondary", onClick, id, disabled } = {}) {
  const v = VARIANTS.has(variant) ? variant : "secondary";
  const btn = el("button", {
    class: `btn btn-${v}`,
    type: "button",
    text,
    id,
    disabled: disabled ? true : false,
  });
  if (onClick) btn.addEventListener("click", onClick);
  return btn;
}

/** Coloured dot + text; returns the element with a `.set(level, text)` method. */
export function statusIndicator(id) {
  const dot = el("span", { class: "status-ind-dot" });
  const txt = el("span", { class: "status-ind-txt", text: "—" });
  const wrap = el("span", { class: "status-ind", id }, [dot, txt]);
  wrap.set = (level, text) => {
    wrap.dataset.level = level || "";
    txt.textContent = text ?? "—";
  };
  return wrap;
}

/** dt/dd metric row; the dd gets `id` so a feature can update it in place. */
export function metricRow(label, valueId, rawKey) {
  const dt = el("dt", { text: label });
  if (rawKey && rawKey !== label) dt.append(el("span", { class: "raw-key", text: ` ${rawKey}` }));
  const dd = el("dd", { id: valueId, text: "—" });
  return el("div", { class: "metric-row" }, [dt, dd]);
}

/** One-line compact summary paragraph for a group body. */
export function summaryLine(text) {
  return el("p", { class: "summary-line", text });
}
