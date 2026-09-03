/**
 * Radar overlay viewer.
 * Draws the radar PNG plus the normalized [0,1] layer coordinates served by
 * /api/radar/... onto a single 1024x1024 canvas with a zoom/pan view
 * transform, per-player tracing, distinct utility glyphs, and hover
 * inspection (plan Task 9).
 */

(function () {
  "use strict";

  const CANVAS_PX = 1024;
  const MIN_ZOOM = 1;
  const MAX_ZOOM = 12;
  const ZOOM_STEP = 1.2;

  const SIDE_COLORS = { T: "#f0883e", CT: "#58a6ff" };
  // Utility palette is deliberately disjoint from side and kill colors, and
  // every kind also gets its own glyph shape so color is never the only cue.
  const UTIL_COLORS = {
    smoke: "#9ecbff",
    flash: "#ffd24d",
    he: "#b07cff",
    molotov: "#ff6a3d",
    decoy: "#7ee2b8",
  };
  const KILL_FOR = "#3fb950";
  const KILL_AGAINST = "#f85149";
  const NEUTRAL = "#8b949e";
  // Okabe-Ito colorblind-safe palette for per-player traces.
  const PLAYER_COLORS = [
    "#e69f00",
    "#56b4e9",
    "#009e73",
    "#f0e442",
    "#0072b2",
    "#d55e00",
    "#cc79a7",
    "#999999",
    "#ffffff",
    "#94d82d",
  ];

  const state = {
    teamKey: null,
    mapName: null,
    displayName: null,
    info: null,
    payload: null,
    loading: false,
    level: "default",
    visible: { heatmap: true, trails: true, utility: true, duels: true, bombs: true },
    view: { k: 1, tx: 0, ty: 0 },
    roster: [], // [{sid, name}] discovered from an unfiltered payload
    selected: null, // Set of sid strings, or null = everyone (no server filter)
    drag: null,
  };

  const el = {};

  function cacheElements() {
    el.tabChat = document.getElementById("tab-chat");
    el.tabRadar = document.getElementById("tab-radar");
    el.radarView = document.getElementById("radar-view");
    el.messages = document.getElementById("messages-container");
    el.inputBar = document.getElementById("chat-input-bar");
    el.image = document.getElementById("radar-image");
    el.canvas = document.getElementById("radar-canvas");
    el.status = document.getElementById("radar-status");
    el.side = document.getElementById("radar-side");
    el.round = document.getElementById("radar-round");
    el.trailRounds = document.getElementById("radar-trail-rounds");
    el.levelToggle = document.getElementById("radar-level-toggle");
    el.levelDefault = document.getElementById("radar-level-default");
    el.levelLower = document.getElementById("radar-level-lower");
    el.refresh = document.getElementById("radar-refresh");
    el.resetView = document.getElementById("radar-reset-view");
    el.players = document.getElementById("radar-players");
    el.tooltip = document.getElementById("radar-tooltip");
    el.checks = {
      heatmap: document.getElementById("layer-heatmap"),
      trails: document.getElementById("layer-trails"),
      utility: document.getElementById("layer-utility"),
      duels: document.getElementById("layer-duels"),
      bombs: document.getElementById("layer-bombs"),
    };
  }

  function setStatus(text, kind) {
    el.status.textContent = text;
    el.status.className = kind === "error" ? "radar-status is-error" : "radar-status";
  }

  function readJson(res) {
    if (!res.ok) {
      return res.json().then(
        function (body) {
          throw new Error((body && body.detail) || `Server returned ${res.status}`);
        },
        function () {
          throw new Error(`Server returned ${res.status}`);
        }
      );
    }
    return res.json();
  }

  // ---------------------------------------------------------------- fetching

  function selectionFilterActive() {
    return (
      state.selected !== null &&
      state.roster.length > 0 &&
      state.selected.size > 0 &&
      state.selected.size < state.roster.length
    );
  }

  function layersUrl() {
    const params = new URLSearchParams();
    if (el.side.value) params.set("side", el.side.value);
    if (el.round.value) params.set("rounds", el.round.value);
    params.set("trail_rounds", el.trailRounds.value);
    if (selectionFilterActive()) {
      params.set("players", Array.from(state.selected).join(","));
    }
    const multiLevel = state.info && state.info.levels.length > 1;
    params.set("level", multiLevel ? state.level : "all");
    return (
      `/api/radar/${encodeURIComponent(state.teamKey)}` +
      `/${encodeURIComponent(state.mapName)}/layers?${params.toString()}`
    );
  }

  function load() {
    if (!state.teamKey || !state.mapName || state.loading) return;
    state.loading = true;
    setStatus("Loading radar...", null);

    fetch(`/api/radar/${encodeURIComponent(state.mapName)}/info`)
      .then(readJson)
      .then(function (info) {
        state.info = info;
        el.levelToggle.classList.toggle("hidden", info.levels.length < 2);
        const useLower = state.level === "lower" && info.lower_image_url;
        const src = useLower ? info.lower_image_url : info.image_url;
        if (el.image.src !== new URL(src, window.location.href).href) {
          el.image.src = src; // draw() re-runs on its load event
        }
        return fetch(layersUrl()).then(readJson);
      })
      .then(function (payload) {
        state.payload = payload;
        populateRounds(payload.rounds);
        if (!selectionFilterActive()) updateRoster(payload);
        renderPlayerPanel();
        setStatus(summarize(payload), null);
        draw();
      })
      .catch(function (err) {
        state.payload = null;
        draw();
        setStatus(err.message, "error");
      })
      .finally(function () {
        state.loading = false;
      });
  }

  function populateRounds(rounds) {
    const previous = el.round.value;
    el.round.innerHTML = '<option value="">All rounds</option>';
    (rounds || []).forEach(function (r) {
      const opt = document.createElement("option");
      opt.value = String(r.round_num);
      opt.textContent = `Round ${r.round_num} (${r.side})`;
      el.round.appendChild(opt);
    });
    el.round.value = previous;
  }

  function summarize(payload) {
    const L = payload.layers;
    const solo = selectionFilterActive() ? ` | tracing ${state.selected.size} player(s)` : "";
    return (
      `${payload.rounds.length} rounds | ` +
      `${L.heatmap.samples} position samples | ` +
      `${L.trails.length} trails | ` +
      `${L.utility.length} nades | ` +
      `${L.duels.length} duels | ` +
      `${L.bombs.length} bomb events${solo}`
    );
  }

  // ---------------------------------------------------------------- roster

  function updateRoster(payload) {
    const seen = new Map();
    (payload.layers.trails || []).forEach(function (t) {
      if (t.steamid && !seen.has(t.steamid)) seen.set(t.steamid, t.name || t.steamid);
    });
    (payload.layers.utility || []).forEach(function (u) {
      if (u.steamid && !seen.has(u.steamid)) seen.set(u.steamid, u.thrower || u.steamid);
    });
    state.roster = Array.from(seen, function (pair) {
      return { sid: pair[0], name: pair[1] };
    }).sort(function (a, b) {
      return a.name.localeCompare(b.name);
    });
  }

  function playerColor(sid) {
    const i = state.roster.findIndex(function (p) {
      return p.sid === sid;
    });
    return i === -1 ? NEUTRAL : PLAYER_COLORS[i % PLAYER_COLORS.length];
  }

  function renderPlayerPanel() {
    if (!el.players) return;
    el.players.classList.toggle("hidden", state.roster.length === 0);
    el.players.innerHTML = "";
    if (!state.roster.length) return;

    const all = document.createElement("button");
    all.type = "button";
    all.className = "btn btn-sm btn-outline radar-player-all" +
      (state.selected === null ? " active" : "");
    all.textContent = "All";
    all.title = "Show every player";
    all.addEventListener("click", function () {
      state.selected = null;
      load();
    });
    el.players.appendChild(all);

    state.roster.forEach(function (p) {
      const chip = document.createElement("button");
      chip.type = "button";
      const active = state.selected !== null && state.selected.has(p.sid);
      chip.className = "radar-player-chip" + (active ? " active" : "");
      chip.title = active
        ? `Tracing ${p.name} - click to show everyone`
        : `Click to trace ${p.name} alone (shift-click to add)`;
      chip.innerHTML =
        '<span class="radar-player-swatch" style="background:' +
        playerColor(p.sid) +
        '"></span>' +
        p.name;
      chip.addEventListener("click", function (ev) {
        if (ev.shiftKey && state.selected !== null) {
          if (state.selected.has(p.sid)) state.selected.delete(p.sid);
          else state.selected.add(p.sid);
          if (state.selected.size === 0) state.selected = null;
        } else if (state.selected !== null && state.selected.size === 1 && active) {
          state.selected = null; // clicking the solo player again restores all
        } else {
          state.selected = new Set([p.sid]);
        }
        load();
      });
      el.players.appendChild(chip);
    });
  }

  // ---------------------------------------------------------------- drawing

  function px(n) {
    return n * CANVAS_PX;
  }

  function lw(base) {
    return base / state.view.k;
  }

  function dot(ctx, x, y, radius, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  function cross(ctx, x, y, size, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = lw(2.5);
    ctx.beginPath();
    ctx.moveTo(x - size, y - size);
    ctx.lineTo(x + size, y + size);
    ctx.moveTo(x + size, y - size);
    ctx.lineTo(x - size, y + size);
    ctx.stroke();
  }

  function heatColor(t) {
    // cold blue (#3c78f0-ish) -> warm orange, matching the app accents
    const r = Math.round(60 + 195 * t);
    const g = Math.round(120 + 16 * t);
    const b = Math.round(240 - 178 * t);
    return `rgb(${r}, ${g}, ${b})`;
  }

  function drawHeatmap(ctx, hm) {
    if (!hm || !hm.cells.length || !hm.max) return;
    const size = CANVAS_PX / hm.grid;
    hm.cells.forEach(function (cell) {
      const t = cell[2] / hm.max;
      ctx.fillStyle = heatColor(t);
      ctx.globalAlpha = 0.2 + 0.6 * t;
      ctx.fillRect(cell[0] * size, cell[1] * size, size, size);
    });
    ctx.globalAlpha = 1;
  }

  function trailColor(trail) {
    if (selectionFilterActive()) return playerColor(trail.steamid);
    return SIDE_COLORS[trail.side] || NEUTRAL;
  }

  function drawTrails(ctx, trails) {
    ctx.lineJoin = "round";
    (trails || []).forEach(function (trail) {
      if (!trail.points || !trail.points.length) return;
      const color = trailColor(trail);
      ctx.strokeStyle = color;
      ctx.lineWidth = lw(2);
      ctx.globalAlpha = 0.75;
      ctx.beginPath();
      trail.points.forEach(function (p, i) {
        if (i === 0) ctx.moveTo(px(p[0]), px(p[1]));
        else ctx.lineTo(px(p[0]), px(p[1]));
      });
      ctx.stroke();
      ctx.globalAlpha = 1;
      const first = trail.points[0];
      const last = trail.points[trail.points.length - 1];
      dot(ctx, px(first[0]), px(first[1]), lw(3), color);
      dot(ctx, px(last[0]), px(last[1]), lw(5), color);
      if (state.view.k >= 2 && trail.name) {
        ctx.fillStyle = color;
        ctx.font = `${12 / state.view.k}px sans-serif`;
        ctx.fillText(trail.name, px(last[0]) + lw(7), px(last[1]) + lw(4));
      }
    });
  }

  function drawGlyph(ctx, kind, x, y, r, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = lw(2);
    ctx.globalAlpha = 0.95;
    ctx.beginPath();
    if (kind === "smoke") {
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.stroke();
      dot(ctx, x, y, r / 3.2, color);
    } else if (kind === "flash") {
      // 4-point star: vertical, horizontal and both diagonals at half length.
      ctx.moveTo(x, y - r);
      ctx.lineTo(x, y + r);
      ctx.moveTo(x - r, y);
      ctx.lineTo(x + r, y);
      const d = r * 0.55;
      ctx.moveTo(x - d, y - d);
      ctx.lineTo(x + d, y + d);
      ctx.moveTo(x + d, y - d);
      ctx.lineTo(x - d, y + d);
      ctx.stroke();
    } else if (kind === "he") {
      ctx.moveTo(x, y - r);
      ctx.lineTo(x + r, y);
      ctx.lineTo(x, y + r);
      ctx.lineTo(x - r, y);
      ctx.closePath();
      ctx.stroke();
    } else if (kind === "molotov") {
      ctx.moveTo(x, y - r);
      ctx.lineTo(x + r * 0.9, y + r * 0.75);
      ctx.lineTo(x - r * 0.9, y + r * 0.75);
      ctx.closePath();
      ctx.stroke();
    } else {
      // decoy and anything unknown: hollow square
      ctx.strokeRect(x - r * 0.8, y - r * 0.8, r * 1.6, r * 1.6);
    }
    ctx.globalAlpha = 1;
  }

  function drawUtility(ctx, util) {
    (util || []).forEach(function (u) {
      if (u.u === null || u.v === null) return;
      const color = UTIL_COLORS[u.kind] || NEUTRAL;
      if (u.from_u !== null && u.from_v !== null) {
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.3;
        ctx.lineWidth = lw(1.5);
        ctx.setLineDash([lw(6), lw(6)]);
        ctx.beginPath();
        ctx.moveTo(px(u.from_u), px(u.from_v));
        ctx.lineTo(px(u.u), px(u.v));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }
      drawGlyph(ctx, u.kind, px(u.u), px(u.v), lw(9), color);
    });
  }

  function drawDuels(ctx, duels) {
    (duels || []).forEach(function (d) {
      if (!d.victim || d.victim.u === null) return;
      const color = d.by_team ? KILL_FOR : KILL_AGAINST;
      if (d.attacker && d.attacker.u !== null) {
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.45;
        ctx.lineWidth = lw(1.5);
        ctx.beginPath();
        ctx.moveTo(px(d.attacker.u), px(d.attacker.v));
        ctx.lineTo(px(d.victim.u), px(d.victim.v));
        ctx.stroke();
        ctx.globalAlpha = 1;
        dot(ctx, px(d.attacker.u), px(d.attacker.v), lw(3), color);
      }
      cross(ctx, px(d.victim.u), px(d.victim.v), lw(6), color);
    });
  }

  function drawBombs(ctx, bombs) {
    (bombs || []).forEach(function (b) {
      if (b.u === null || b.v === null) return;
      const plant = b.event === "plant";
      const w = lw(22);
      const h = lw(14);
      const x = px(b.u) - w / 2;
      const y = px(b.v) - h / 2;
      ctx.fillStyle = "#0d1117";
      ctx.globalAlpha = 0.9;
      ctx.fillRect(x, y, w, h);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = plant ? "#f0883e" : "#58a6ff";
      ctx.lineWidth = lw(1.5);
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = plant ? "#f0883e" : "#58a6ff";
      ctx.font = `bold ${10 / state.view.k}px sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(plant ? "C4" : "DEF", px(b.u), px(b.v));
      ctx.textAlign = "start";
      ctx.textBaseline = "alphabetic";
    });
  }

  function draw() {
    const ctx = el.canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, CANVAS_PX, CANVAS_PX);
    const v = state.view;
    ctx.setTransform(v.k, 0, 0, v.k, v.tx, v.ty);
    if (el.image.complete && el.image.naturalWidth > 0) {
      ctx.drawImage(el.image, 0, 0, CANVAS_PX, CANVAS_PX);
    }
    if (!state.payload) return;
    const L = state.payload.layers;
    if (state.visible.heatmap) drawHeatmap(ctx, L.heatmap);
    if (state.visible.trails) drawTrails(ctx, L.trails);
    if (state.visible.utility) drawUtility(ctx, L.utility);
    if (state.visible.duels) drawDuels(ctx, L.duels);
    if (state.visible.bombs) drawBombs(ctx, L.bombs);
  }

  // ---------------------------------------------------------------- view transform

  function clampView() {
    const v = state.view;
    v.k = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, v.k));
    const min = CANVAS_PX * (1 - v.k);
    v.tx = Math.min(0, Math.max(min, v.tx));
    v.ty = Math.min(0, Math.max(min, v.ty));
    if (v.k === 1) {
      v.tx = 0;
      v.ty = 0;
    }
  }

  function canvasPoint(ev) {
    const rect = el.canvas.getBoundingClientRect();
    return {
      x: ((ev.clientX - rect.left) * CANVAS_PX) / rect.width,
      y: ((ev.clientY - rect.top) * CANVAS_PX) / rect.height,
    };
  }

  function resetView() {
    state.view = { k: 1, tx: 0, ty: 0 };
    draw();
  }

  function onWheel(ev) {
    ev.preventDefault();
    const v = state.view;
    const c = canvasPoint(ev);
    const factor = ev.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
    const k2 = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, v.k * factor));
    // Keep the world point under the cursor fixed while scaling.
    v.tx = c.x - ((c.x - v.tx) * k2) / v.k;
    v.ty = c.y - ((c.y - v.ty) * k2) / v.k;
    v.k = k2;
    clampView();
    hideTooltip();
    draw();
  }

  function onPointerDown(ev) {
    if (state.view.k === 1) return;
    const c = canvasPoint(ev);
    state.drag = { x: c.x, y: c.y, tx: state.view.tx, ty: state.view.ty, moved: false };
    el.canvas.setPointerCapture(ev.pointerId);
  }

  function onPointerMove(ev) {
    const c = canvasPoint(ev);
    if (state.drag) {
      state.view.tx = state.drag.tx + (c.x - state.drag.x);
      state.view.ty = state.drag.ty + (c.y - state.drag.y);
      state.drag.moved = true;
      clampView();
      hideTooltip();
      draw();
      return;
    }
    updateTooltip(ev, c);
  }

  function onPointerUp(ev) {
    if (state.drag) {
      el.canvas.releasePointerCapture(ev.pointerId);
      state.drag = null;
    }
  }

  // ---------------------------------------------------------------- tooltip

  function hideTooltip() {
    if (el.tooltip) el.tooltip.classList.add("hidden");
  }

  function fmtClock(seconds) {
    if (seconds === null || seconds === undefined) return "";
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return ` @ ${m}:${String(s).padStart(2, "0")}`;
  }

  function hitTest(c) {
    if (!state.payload) return null;
    const v = state.view;
    const wx = (c.x - v.tx) / v.k / CANVAS_PX;
    const wy = (c.y - v.ty) / v.k / CANVAS_PX;
    const r = 12 / v.k / CANVAS_PX;
    const L = state.payload.layers;

    let best = null;
    function consider(u, vv, text) {
      if (u === null || vv === null) return;
      const d = Math.hypot(u - wx, vv - wy);
      if (d <= r && (best === null || d < best.d)) best = { d: d, text: text };
    }
    if (state.visible.utility) {
      (L.utility || []).forEach(function (u) {
        consider(u.u, u.v, `${u.kind} - ${u.thrower || "?"}, R${u.round_num}`);
      });
    }
    if (state.visible.duels) {
      (L.duels || []).forEach(function (d) {
        const a = d.attacker ? d.attacker.name : "?";
        consider(
          d.victim.u,
          d.victim.v,
          `${a} > ${d.victim.name} (${d.weapon || "?"}${d.headshot ? ", HS" : ""}), R${d.round_num}`
        );
      });
    }
    if (state.visible.bombs) {
      (L.bombs || []).forEach(function (b) {
        consider(b.u, b.v, `${b.event} ${b.bombsite || ""} - ${b.name || "?"}, R${b.round_num}`);
      });
    }
    if (state.visible.trails && !best) {
      (L.trails || []).forEach(function (t) {
        const last = t.points && t.points[t.points.length - 1];
        if (last) consider(last[0], last[1], `${t.name} (R${t.round_num})${fmtClock(last[2])}`);
      });
    }
    return best;
  }

  function updateTooltip(ev, c) {
    if (!el.tooltip) return;
    const hit = hitTest(c);
    if (!hit) {
      hideTooltip();
      return;
    }
    const frameRect = el.canvas.parentElement.getBoundingClientRect();
    el.tooltip.textContent = hit.text;
    el.tooltip.style.left = `${ev.clientX - frameRect.left + 12}px`;
    el.tooltip.style.top = `${ev.clientY - frameRect.top + 12}px`;
    el.tooltip.classList.remove("hidden");
  }

  // ---------------------------------------------------------------- wiring

  function showView(view) {
    const radar = view === "radar";
    el.radarView.classList.toggle("hidden", !radar);
    el.messages.classList.toggle("hidden", radar);
    el.inputBar.classList.toggle("hidden", radar);
    el.tabChat.classList.toggle("active", !radar);
    el.tabRadar.classList.toggle("active", radar);
    if (radar && !state.payload) load();
  }

  function setLevel(level) {
    if (state.level === level) return;
    state.level = level;
    el.levelDefault.classList.toggle("active", level === "default");
    el.levelLower.classList.toggle("active", level === "lower");
    load();
  }

  function onTargetSelected(teamKey, mapName, displayName) {
    state.teamKey = teamKey;
    state.mapName = mapName;
    state.displayName = displayName;
    state.payload = null;
    state.info = null;
    state.level = "default";
    state.roster = [];
    state.selected = null;
    state.view = { k: 1, tx: 0, ty: 0 };
    el.tabRadar.disabled = false;
    el.tabRadar.title = `Radar overlay for ${displayName} on ${mapName}`;
    el.round.value = "";
    renderPlayerPanel();
    draw();
    if (!el.radarView.classList.contains("hidden")) load();
    else setStatus("Open the Radar tab to plot this selection.", null);
  }

  function initEvents() {
    el.tabChat.addEventListener("click", function () {
      showView("chat");
    });
    el.tabRadar.addEventListener("click", function () {
      showView("radar");
    });
    Object.keys(el.checks).forEach(function (key) {
      el.checks[key].addEventListener("change", function () {
        state.visible[key] = el.checks[key].checked;
        draw();
      });
    });
    [el.side, el.round, el.trailRounds].forEach(function (control) {
      control.addEventListener("change", load);
    });
    el.levelDefault.addEventListener("click", function () {
      setLevel("default");
    });
    el.levelLower.addEventListener("click", function () {
      setLevel("lower");
    });
    el.refresh.addEventListener("click", load);
    if (el.resetView) el.resetView.addEventListener("click", resetView);

    el.canvas.addEventListener("wheel", onWheel, { passive: false });
    el.canvas.addEventListener("pointerdown", onPointerDown);
    el.canvas.addEventListener("pointermove", onPointerMove);
    el.canvas.addEventListener("pointerup", onPointerUp);
    el.canvas.addEventListener("pointerleave", function (ev) {
      hideTooltip();
      onPointerUp(ev);
    });
    el.canvas.addEventListener("dblclick", resetView);
    el.image.addEventListener("load", draw);
  }

  window.CounterStratRadar = {
    onTargetSelected: function (teamKey, mapName, displayName) {
      if (!el.radarView) return;
      onTargetSelected(teamKey, mapName, displayName);
    },
  };

  document.addEventListener("DOMContentLoaded", function () {
    cacheElements();
    if (!el.radarView) return;
    initEvents();
  });
})();
