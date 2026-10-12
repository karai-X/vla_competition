(() => {
  "use strict";

  /* ---------- tabs (hash routing) ---------- */
  // Derive the tab list from the sections in the page so it cannot drift from the HTML.
  const views = [...document.querySelectorAll("main > section[id^='view-']")].map((el) => el.id.slice(5));
  const nav = document.getElementById("nav");

  function show(name) {
    if (!views.includes(name)) name = "overview";
    for (const v of views) {
      document.getElementById("view-" + v).hidden = v !== name;
    }
    for (const a of nav.querySelectorAll("a")) {
      if (a.getAttribute("href") === "#" + name) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    }
    const active = nav.querySelector('[aria-current="page"]');
    if (active) active.scrollIntoView({ block: "nearest", inline: "nearest" });
  }

  window.addEventListener("hashchange", () => {
    show(location.hash.slice(1));
    window.scrollTo(0, 0);
  });
  show(location.hash.slice(1));

  /* ---------- theme toggle: auto -> light -> dark ---------- */
  const themeButton = document.getElementById("theme");
  const labels = { auto: "テーマ: 自動", light: "テーマ: ライト", dark: "テーマ: ダーク" };
  function applyTheme(mode) {
    if (mode === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", mode);
    themeButton.textContent = labels[mode];
  }
  let theme = "auto";
  try { theme = localStorage.getItem("theme") || "auto"; } catch (_) { /* storage unavailable */ }
  if (!labels[theme]) theme = "auto";
  applyTheme(theme);
  themeButton.addEventListener("click", () => {
    theme = { auto: "light", light: "dark", dark: "auto" }[theme];
    applyTheme(theme);
    try { localStorage.setItem("theme", theme); } catch (_) { /* ignore */ }
  });

  /* ---------- comparison charts (grouped horizontal bars) ---------- */
  // Rates (%) on identical scenarios: without vs with fine-tuning.
  const charts = {
    task: {
      metric: "タスク成功率",
      rows: [
        { label: "背景・照明", base: 90, ours: 99 },
        { label: "視点・配置など", base: 80, ours: 84 },
      ],
    },
    cf: {
      metric: "衝突なし成功率",
      rows: [
        { label: "背景・照明", base: 64, ours: 70 },
        { label: "視点・配置など", base: 52, ours: 60 },
      ],
    },
  };
  const series = [
    ["base", "bar-base", "ファインチューニングなし"],
    ["ours", "bar-ours", "ファインチューニング後"],
  ];
  const hosts = [...document.querySelectorAll(".chart[data-chart]")];
  const tip = document.getElementById("tooltip");
  const NS = "http://www.w3.org/2000/svg";

  function el(name, attrs, text) {
    const node = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // Bar with a rounded data-end and a square baseline end.
  function barPath(x, y, w, h, r) {
    if (w <= 0) return "";
    const rr = Math.min(r, w, h / 2);
    return `M${x},${y} H${x + w - rr} Q${x + w},${y} ${x + w},${y + rr} V${y + h - rr} Q${x + w},${y + h} ${x + w - rr},${y + h} H${x} Z`;
  }

  function render(host) {
    const { metric, rows } = charts[host.dataset.chart];
    host.innerHTML = "";
    const width = Math.max(host.clientWidth, 260);
    const left = 104, right = 44, top = 6, rowH = 58, barH = 18, gap = 2;
    const height = top + rowH * rows.length + 22;
    const plotW = width - left - right;
    const x = (v) => left + (v / 100) * plotW;
    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width, height, "aria-hidden": "true" });

    for (const t of [0, 25, 50, 75, 100]) {
      svg.appendChild(el("line", { class: "axis", x1: x(t), x2: x(t), y1: top, y2: height - 20, "stroke-dasharray": t ? "2 4" : "" }));
      svg.appendChild(el("text", { class: "lbl", x: x(t), y: height - 4, "text-anchor": "middle" }, `${t}%`));
    }

    rows.forEach((row, i) => {
      const y0 = top + i * rowH + (rowH - (barH * 2 + gap)) / 2;
      svg.appendChild(el("text", { class: "lbl", x: 0, y: y0 + barH + 5 }, row.label));
      series.forEach(([key, cls, name], j) => {
        const y = y0 + j * (barH + gap);
        const v = row[key];
        const d = barPath(x(0), y, x(v) - x(0), barH, 4);
        if (d) svg.appendChild(el("path", { class: cls, d }));
        svg.appendChild(el("text", { class: "val", x: x(v) + 6, y: y + barH - 4 }, `${v}%`));
        const hit = el("rect", { class: "hit", x: left, y: y - 1, width: plotW + right, height: barH + 2 });
        hit.addEventListener("pointerenter", () => {
          host.appendChild(tip);
          tip.textContent = `${row.label} · ${name}: ${metric} ${v}%`;
          tip.classList.add("on");
        });
        hit.addEventListener("pointermove", (e) => {
          const box = host.getBoundingClientRect();
          tip.style.left = `${e.clientX - box.left}px`;
          tip.style.top = `${e.clientY - box.top}px`;
        });
        hit.addEventListener("pointerleave", () => tip.classList.remove("on"));
        svg.appendChild(hit);
      });
    });
    host.appendChild(svg);
  }

  const renderAll = () => hosts.forEach(render);
  renderAll();
  let pending = 0;
  window.addEventListener("resize", () => {
    cancelAnimationFrame(pending);
    pending = requestAnimationFrame(renderAll);
  });
  // The charts live in the first view; re-measure when it becomes visible again.
  window.addEventListener("hashchange", () => {
    if (hosts.length && !hosts[0].closest("section").hidden) renderAll();
  });
})();
