/**
 * Radar overlay viewer.
 * Draws the normalized [0,1] layer coordinates served by /api/radar/... onto a
 * 1024x1024 canvas stacked over the CS2 overhead radar PNG.
 */

(function () {
  "use strict";

  const CANVAS_PX = 1024;

  const SIDE_COLORS = { T: "#f0883e", CT: "#58a6ff" };
  const UTIL_COLORS = {
    smoke: "#c9d1d9",
    flash: "#e3b341",
    he: "#f85149",
    molotov: "#db6d28",
    decoy: "#8b949e",
  };
  const KILL_FOR = "#3fb950";
  const KILL_AGAINST = "#f85149";
  const NEUTRAL = "#8b949e";

  const state = {
    teamKey: null,
    mapName: null,
    displayName: null,
    info: null,
    payload: null,
    loading: false,
    level: "default",
    visible: { heatmap: true, trails: true, utility: true, duels: true, bombs: true },
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

  function layersUrl() {
    const params = new URLSearchParams();
    if (el.side.value) params.set("side", el.side.value);
    if (el.round.value) params.set("rounds", el.round.value);
    params.set("trail_rounds", el.trailRounds.value);
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
        el.image.src = useLower ? info.lower_image_url : info.image_url;
        return fetch(layersUrl()).then(readJson);
      })
      .then(function (payload) {
        state.payload = payload;
        populateRounds(payload.rounds);
        setStatus(summarize(payload), null);
        draw();
      })
      .catch(function (err) {
        state.payload = null;
        clearCanvas();
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
    return (
      `${payload.rounds.length} rounds | ` +
      `${L.heatmap.samples} position samples | ` +
      `${L.trails.length} trails | ` +
      `${L.utility.length} nades | ` +
      `${L.duels.length} duels | ` +
      `${L.bombs.length} bomb events`
    );
  }

  // ---------------------------------------------------------------- drawing

  function px(n) {
    return n * CANVAS_PX;
  }

  function clearCanvas() {
    el.canvas.getContext("2d").clearRect(0, 0, CANVAS_PX, CANVAS_PX);
  }

  function dot(ctx, x, y, radius, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  function cross(ctx, x, y, size, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
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

  function drawTrails(ctx, trails) {
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    (trails || []).forEach(function (trail) {
      if (!trail.points || !trail.points.length) return;
      const color = SIDE_COLORS[trail.side] || NEUTRAL;
      ctx.strokeStyle = color;
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
      dot(ctx, px(first[0]), px(first[1]), 3, color);
      dot(ctx, px(last[0]), px(last[1]), 5, color);
    });
  }

  function drawUtility(ctx, util) {
    (util || []).forEach(function (u) {
      if (u.u === null || u.v === null) return;
      const color = UTIL_COLORS[u.kind] || NEUTRAL;
      if (u.from_u !== null && u.from_v !== null) {
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.3;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 6]);
        ctx.beginPath();
        ctx.moveTo(px(u.from_u), px(u.from_v));
        ctx.lineTo(px(u.u), px(u.v));
        ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.9;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px(u.u), px(u.v), 9, 0, Math.PI * 2);
      ctx.stroke();
      ctx.globalAlpha = 1;
    });
  }

  function drawDuels(ctx, duels) {
    (duels || []).forEach(function (d) {
      if (!d.victim || d.victim.u === null) return;
      const color = d.by_team ? KILL_FOR : KILL_AGAINST;
      if (d.attacker && d.attacker.u !== null) {
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.45;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(px(d.attacker.u), px(d.attacker.v));
        ctx.lineTo(px(d.victim.u), px(d.victim.v));
        ctx.stroke();
        ctx.globalAlpha = 1;
        dot(ctx, px(d.attacker.u), px(d.attacker.v), 3, color);
      }
      cross(ctx, px(d.victim.u), px(d.victim.v), 6, color);
    });
  }

  function drawBombs(ctx, bombs) {
    (bombs || []).forEach(function (b) {
      if (b.u === null || b.v === null) return;
      const color = b.event === "plant" ? "#f0883e" : "#58a6ff";
      const size = 12;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(px(b.u) - size / 2, px(b.v) - size / 2, size, size);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = "#0d1117";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(px(b.u) - size / 2, px(b.v) - size / 2, size, size);
    });
  }

  function draw() {
    const ctx = el.canvas.getContext("2d");
    ctx.clearRect(0, 0, CANVAS_PX, CANVAS_PX);
    if (!state.payload) return;
    const L = state.payload.layers;
    if (state.visible.heatmap) drawHeatmap(ctx, L.heatmap);
    if (state.visible.trails) drawTrails(ctx, L.trails);
    if (state.visible.utility) drawUtility(ctx, L.utility);
    if (state.visible.duels) drawDuels(ctx, L.duels);
    if (state.visible.bombs) drawBombs(ctx, L.bombs);
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
    el.tabRadar.disabled = false;
    el.tabRadar.title = `Radar overlay for ${displayName} on ${mapName}`;
    el.round.value = "";
    clearCanvas();
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

