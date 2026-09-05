/**
 * Callout editor: rename map zones; the analyst then speaks YOUR names.
 * Aliases are stored per map and applied as the single vocabulary at the LLM
 * boundary - blank alias means the game's default name stays in use.
 */

(function () {
  "use strict";

  const state = {
    mapName: null,
    zones: [], // [{name, alias, u, v, level, custom, half_x/y, half_u/v, bounds}]
    levels: ["default"],
    level: "default", // which radar level is on screen (nuke upper/lower)
    view: { k: 1, tx: 0, ty: 0 }, // zoom/pan transform of the map frame
    pan: null,
    showAreas: false, // dashed occupancy boxes of the game's own zones
    dirty: false,
    placing: false, // add-callout mode: next map click names a new zone
  };

  const el = {};

  function cacheElements() {
    el.tabChat = document.getElementById("tab-chat");
    el.tabCallouts = document.getElementById("tab-callouts");
    el.view = document.getElementById("callouts-view");
    el.mapSelect = document.getElementById("callouts-map");
    el.resetAll = document.getElementById("callouts-reset-all");
    el.addBtn = document.getElementById("callouts-add");
    el.frame = document.querySelector(".callouts-frame");
    el.status = document.getElementById("callouts-status");
    el.image = document.getElementById("callouts-image");
    el.labels = document.getElementById("callouts-labels");
    el.levelToggle = document.getElementById("callouts-level-toggle");
    el.levelDefault = document.getElementById("callouts-level-default");
    el.levelLower = document.getElementById("callouts-level-lower");
    el.tableBody = document.querySelector("#callouts-table tbody");
    el.messages = document.getElementById("messages-container");
    el.inputBar = document.getElementById("chat-input-bar");
    el.zoomBox = document.getElementById("callouts-zoom");
    el.areasBtn = document.getElementById("callouts-areas");
  }

  // ------------------------------------------------------------ zoom & pan

  function applyView() {
    const view = state.view;
    el.zoomBox.style.transform = `translate(${view.tx}px, ${view.ty}px) scale(${view.k})`;
  }

  function resetView() {
    state.view = { k: 1, tx: 0, ty: 0 };
    applyView();
  }

  function onWheel(ev) {
    ev.preventDefault();
    const rect = el.frame.getBoundingClientRect();
    const px = ev.clientX - rect.left;
    const py = ev.clientY - rect.top;
    const view = state.view;
    const k2 = Math.min(8, Math.max(1, view.k * Math.exp(-ev.deltaY * 0.0015)));
    // Keep the point under the cursor fixed while scaling.
    view.tx = px - ((px - view.tx) * k2) / view.k;
    view.ty = py - ((py - view.ty) * k2) / view.k;
    view.k = k2;
    if (view.k === 1) { view.tx = 0; view.ty = 0; }
    applyView();
  }

  function panStart(ev) {
    if (state.placing || state.view.k === 1) return;
    state.pan = { x: ev.clientX, y: ev.clientY, moved: false };
    el.frame.setPointerCapture(ev.pointerId);
  }

  function panMove(ev) {
    if (!state.pan) return;
    const dx = ev.clientX - state.pan.x;
    const dy = ev.clientY - state.pan.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) state.pan.moved = true;
    state.view.tx += dx;
    state.view.ty += dy;
    state.pan.x = ev.clientX;
    state.pan.y = ev.clientY;
    applyView();
  }

  function panEnd(ev) {
    if (!state.pan) return;
    const moved = state.pan.moved;
    state.pan = null;
    try { el.frame.releasePointerCapture(ev.pointerId); } catch { /* released */ }
    if (moved) state.suppressClick = true; // a pan is not a label click
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function setStatus(text, isError) {
    el.status.textContent = text;
    el.status.className = "callouts-status" + (isError ? " is-error" : "");
  }

  // ---------------------------------------------------------------- data

  function loadMaps() {
    fetch("/api/maps")
      .then(function (res) { return res.json(); })
      .then(function (maps) {
        el.mapSelect.innerHTML = "";
        maps.forEach(function (m) {
          const opt = document.createElement("option");
          opt.value = m;
          opt.textContent = m;
          el.mapSelect.appendChild(opt);
        });
        if (!maps.length) {
          setStatus("No map cards yet - ingest a demo first.", true);
          return;
        }
        if (!state.mapName || maps.indexOf(state.mapName) === -1) {
          state.mapName = maps[0];
        }
        el.mapSelect.value = state.mapName;
        loadCallouts();
      })
      .catch(function () { setStatus("Failed to load maps.", true); });
  }

  function loadCallouts() {
    if (!state.mapName) return;
    fetch(`/api/maps/${encodeURIComponent(state.mapName)}/callouts`)
      .then(function (res) {
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        return res.json();
      })
      .then(function (data) {
        state.zones = data.zones || [];
        state.levels = data.levels || ["default"];
        if (state.levels.indexOf(state.level) === -1) state.level = "default";
        el.levelToggle.classList.toggle("hidden", state.levels.length < 2);
        setImage();
        render();
      })
      .catch(function (err) { setStatus(`Failed to load callouts: ${err.message}`, true); });
  }

  function saveAll() {
    const aliases = {};
    state.zones.forEach(function (z) {
      aliases[z.name] = z.alias || "";
    });
    fetch(`/api/maps/${encodeURIComponent(state.mapName)}/aliases`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ aliases: aliases }),
    })
      .then(function (res) {
        if (!res.ok) {
          return res.json().then(function (body) {
            throw new Error((body && body.detail) || `Server returned ${res.status}`);
          });
        }
        return res.json();
      })
      .then(function (data) {
        state.dirty = false;
        const n = Object.keys(data.aliases || {}).length;
        setStatus(
          `Saved - ${n} custom callout${n === 1 ? "" : "s"}. The analyst uses them from the ` +
            "next chat session or regenerated read.",
          false
        );
        loadCallouts();
      })
      .catch(function (err) { setStatus(`Not saved: ${err.message}`, true); });
  }

  function setAlias(zoneName, value) {
    const zone = state.zones.find(function (z) { return z.name === zoneName; });
    if (!zone) return;
    zone.alias = (value || "").trim() || null;
    state.dirty = true;
    saveAll();
  }

  function resetAll() {
    if (!state.mapName) return;
    if (!window.confirm(`Remove ALL custom callouts for ${state.mapName}? The game names come back.`)) {
      return;
    }
    state.zones.forEach(function (z) { z.alias = null; });
    state.dirty = true;
    saveAll();
  }

  // ------------------------------------------------------- custom zones

  function zonesPayload() {
    // Round-trip each rect as its two corners around the stored center.
    return state.zones
      .filter(function (z) { return z.custom; })
      .map(function (z) {
        return {
          name: z.name,
          u: z.u - z.half_u,
          v: z.v - z.half_v,
          u2: z.u + z.half_u,
          v2: z.v + z.half_v,
          level: z.level || "default",
        };
      });
  }

  function putZones(payload, verb) {
    setStatus(`${verb}...`, false);
    fetch(`/api/maps/${encodeURIComponent(state.mapName)}/zones`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ zones: payload }),
    })
      .then(function (res) {
        if (!res.ok) {
          return res.json().then(function (body) {
            throw new Error((body && body.detail) || `Server returned ${res.status}`);
          });
        }
        return res.json();
      })
      .then(function (data) { pollJob(data.job_id); })
      .catch(function (err) { setStatus(`Not saved: ${err.message}`, true); });
  }

  function pollJob(jobId) {
    setStatus("Rebuilding map data - tendencies, scripts, and reads pick up your callouts...", false);
    const timer = window.setInterval(function () {
      fetch(`/api/jobs/${encodeURIComponent(jobId)}`)
        .then(function (res) { return res.json(); })
        .then(function (job) {
          if (job.stage === "done") {
            window.clearInterval(timer);
            setStatus(
              "Callouts rebuilt - the analyst speaks them from the next chat session or regenerated read.",
              false
            );
            loadCallouts();
          } else if (job.stage === "error") {
            window.clearInterval(timer);
            setStatus(`Rebuild failed: ${job.detail}`, true);
          }
        })
        .catch(function () { /* transient poll errors: keep polling */ });
    }, 1500);
  }

  function setPlacing(on) {
    state.placing = on;
    el.addBtn.classList.toggle("active", on);
    el.frame.classList.toggle("is-placing", on);
    if (!on) clearGhost();
    if (on) setStatus("Drag a rectangle over the area the callout covers. Esc cancels.", false);
  }

  function clearGhost() {
    if (state.dragGhost) {
      state.dragGhost.remove();
      state.dragGhost = null;
    }
    state.dragStart = null;
  }

  function pointerUV(ev) {
    const rect = el.labels.getBoundingClientRect();
    return {
      u: Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width)),
      v: Math.min(1, Math.max(0, (ev.clientY - rect.top) / rect.height)),
    };
  }

  function dragStart(ev) {
    if (!state.placing) return;
    ev.stopPropagation();
    ev.preventDefault();
    state.dragStart = pointerUV(ev);
    const ghost = document.createElement("div");
    ghost.className = "zone-ghost";
    el.labels.appendChild(ghost);
    state.dragGhost = ghost;
    dragMove(ev);
  }

  function dragMove(ev) {
    if (!state.placing || !state.dragStart || !state.dragGhost) return;
    const cur = pointerUV(ev);
    const left = Math.min(state.dragStart.u, cur.u);
    const top = Math.min(state.dragStart.v, cur.v);
    state.dragGhost.style.left = `${left * 100}%`;
    state.dragGhost.style.top = `${top * 100}%`;
    state.dragGhost.style.width = `${Math.abs(cur.u - state.dragStart.u) * 100}%`;
    state.dragGhost.style.height = `${Math.abs(cur.v - state.dragStart.v) * 100}%`;
  }

  function dragEnd(ev) {
    if (!state.placing || !state.dragStart) return;
    ev.stopPropagation();
    const start = state.dragStart;
    const cur = pointerUV(ev);
    const frame = el.labels.getBoundingClientRect();
    const draggedPx = Math.max(
      Math.abs(cur.u - start.u) * frame.width,
      Math.abs(cur.v - start.v) * frame.height
    );
    if (draggedPx < 8) {
      clearGhost();
      setStatus("Drag a rectangle (press and move) to size the zone. Esc cancels.", false);
      return;
    }
    const corners = {
      u: Math.min(start.u, cur.u),
      v: Math.min(start.v, cur.v),
      u2: Math.max(start.u, cur.u),
      v2: Math.max(start.v, cur.v),
    };
    // The drawn rectangle stays on screen while the zone is named; the next
    // render (cancel, or the post-save reload) replaces it with the real
    // footprint.
    state.dragGhost = null;
    state.dragStart = null;
    setPlacing(false);
    nameNewZone(corners);
  }

  function nameNewZone(corners) {
    const cu = (corners.u + corners.u2) / 2;
    const cv = (corners.v + corners.v2) / 2;
    const input = document.createElement("input");
    input.type = "text";
    input.className = "callout-edit-input";
    input.placeholder = "callout name";
    input.style.position = "absolute";
    input.style.left = `${cu * 100}%`;
    input.style.top = `${cv * 100}%`;
    el.labels.appendChild(input);
    input.focus();
    let done = false;
    function commit(save) {
      if (done) return;
      done = true;
      const name = (input.value || "").trim();
      input.remove();
      if (!save || !name) { render(); return; }
      const payload = zonesPayload();
      payload.push({
        name: name,
        u: corners.u,
        v: corners.v,
        u2: corners.u2,
        v2: corners.v2,
        level: state.level,
      });
      putZones(payload, `Placing ${name}`);
    }
    input.addEventListener("keydown", function (kev) {
      if (kev.key === "Enter") commit(true);
      if (kev.key === "Escape") commit(false);
    });
    input.addEventListener("blur", function () { commit(true); });
  }

  function removeZone(name) {
    if (!window.confirm(`Remove the callout ${name}? Its ground goes back to the game's zone.`)) {
      return;
    }
    putZones(
      zonesPayload().filter(function (z) { return z.name !== name; }),
      `Removing ${name}`
    );
  }

  // ---------------------------------------------------------------- render

  function setImage() {
    const level = state.levels.length > 1 ? state.level : "default";
    el.image.src =
      `/api/radar/${encodeURIComponent(state.mapName)}/image?level=${level}`;
  }

  function setLevel(level) {
    if (state.level === level) return;
    state.level = level;
    el.levelDefault.classList.toggle("active", level === "default");
    el.levelLower.classList.toggle("active", level === "lower");
    setImage();
    render();
  }

  function labelFor(zone) {
    return zone.alias || zone.name;
  }

  function startEdit(zone, anchorEl) {
    const input = document.createElement("input");
    input.type = "text";
    input.className = "callout-edit-input";
    input.value = zone.alias || "";
    input.placeholder = zone.name;
    anchorEl.replaceChildren(input);
    input.focus();
    input.select();
    let done = false;
    function commit(save) {
      if (done) return;
      done = true;
      if (save) setAlias(zone.name, input.value);
      else render();
    }
    input.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter") commit(true);
      if (ev.key === "Escape") commit(false);
    });
    input.addEventListener("blur", function () { commit(true); });
  }

  function render() {
    // Map labels for zones with a known anchor.
    el.labels.innerHTML = "";
    const multiLevel = state.levels.length > 1;
    // Occupancy boxes of the game's own zones (toggle): where the engine's
    // vocabulary actually lives, from the shipped calibration.
    if (state.showAreas) {
      state.zones.forEach(function (zone) {
        if (zone.custom || !zone.bounds) return;
        if (multiLevel && zone.level && zone.level !== state.level) return;
        const box = document.createElement("div");
        box.className = "zone-area";
        box.style.left = `${zone.bounds[0] * 100}%`;
        box.style.top = `${zone.bounds[1] * 100}%`;
        box.style.width = `${(zone.bounds[2] - zone.bounds[0]) * 100}%`;
        box.style.height = `${(zone.bounds[3] - zone.bounds[1]) * 100}%`;
        el.labels.appendChild(box);
      });
    }
    // The user's zone footprints render under the labels - placement is
    // never blind.
    state.zones.forEach(function (zone) {
      if (!zone.custom || zone.u === null || zone.u === undefined) return;
      if (multiLevel && zone.level && zone.level !== state.level) return;
      if (!zone.half_u || !zone.half_v) return;
      const fp = document.createElement("div");
      fp.className = "zone-footprint";
      fp.style.left = `${(zone.u - zone.half_u) * 100}%`;
      fp.style.top = `${(zone.v - zone.half_v) * 100}%`;
      fp.style.width = `${zone.half_u * 2 * 100}%`;
      fp.style.height = `${zone.half_v * 2 * 100}%`;
      el.labels.appendChild(fp);
    });
    state.zones.forEach(function (zone) {
      if (zone.u === null || zone.u === undefined) return;
      // On multi-level maps only the selected level's zones are shown, so the
      // underground site never clutters the upper radar (and vice versa).
      if (multiLevel && zone.level && zone.level !== state.level) return;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className =
        "callout-label" +
        (zone.alias ? " is-custom" : "") +
        (zone.custom ? " is-user-zone" : "");
      btn.style.left = `${zone.u * 100}%`;
      btn.style.top = `${zone.v * 100}%`;
      btn.textContent = labelFor(zone);
      btn.title = zone.custom
        ? `${zone.name} (your callout, ${Math.round((zone.half_x || 0) * 2)}\u00d7${Math.round((zone.half_y || 0) * 2)} units)`
        : zone.alias
          ? `${zone.alias} (game name: ${zone.name}) - click to edit`
          : `${zone.name} - click to rename`;
      btn.addEventListener("click", function () { startEdit(zone, btn); });
      el.labels.appendChild(btn);
    });

    // Full table, covering zones without a map anchor too.
    el.tableBody.innerHTML = "";
    state.zones.forEach(function (zone) {
      const tr = document.createElement("tr");
      const nameTd = document.createElement("td");
      nameTd.innerHTML = zone.custom
        ? `<code>${escapeHtml(zone.name)}</code> <span class="callout-user-tag">yours</span>`
        : `<code>${escapeHtml(zone.name)}</code>`;
      const aliasTd = document.createElement("td");
      if (zone.custom) {
        // A user-created zone: its position/size are the identity - delete it
        // to redraw (its ground folds back to the game zone).
        const dims = document.createElement("span");
        dims.className = "callout-zone-dims";
        dims.textContent =
          `${Math.round((zone.half_x || 0) * 2)}\u00d7${Math.round((zone.half_y || 0) * 2)} units`;
        dims.title = "Rectangle size - delete and redraw to resize";
        aliasTd.appendChild(dims);
        const del = document.createElement("button");
        del.type = "button";
        del.className = "callout-reset-btn";
        del.textContent = "\u00d7";
        del.title = `Remove ${zone.name}`;
        del.addEventListener("click", function () { removeZone(zone.name); });
        aliasTd.appendChild(del);
      } else {
        const input = document.createElement("input");
        input.type = "text";
        input.className = "form-control form-control-sm callout-table-input";
        input.value = zone.alias || "";
        input.placeholder = "(game name)";
        input.addEventListener("change", function () { setAlias(zone.name, input.value); });
        aliasTd.appendChild(input);
        if (zone.alias) {
          const reset = document.createElement("button");
          reset.type = "button";
          reset.className = "callout-reset-btn";
          reset.textContent = "\u00d7";
          reset.title = `Reset to ${zone.name}`;
          reset.addEventListener("click", function () { setAlias(zone.name, ""); });
          aliasTd.appendChild(reset);
        }
      }
      tr.appendChild(nameTd);
      tr.appendChild(aliasTd);
      el.tableBody.appendChild(tr);
    });
  }

  // ---------------------------------------------------------------- tabs

  function showCallouts() {
    el.view.classList.remove("hidden");
    el.messages.classList.add("hidden");
    el.inputBar.classList.add("hidden");
    el.tabCallouts.classList.add("active");
    el.tabChat.classList.remove("active");
    loadMaps();
  }

  function showChat() {
    el.view.classList.add("hidden");
    el.messages.classList.remove("hidden");
    el.inputBar.classList.remove("hidden");
    el.tabCallouts.classList.remove("active");
    el.tabChat.classList.add("active");
  }

  document.addEventListener("DOMContentLoaded", function () {
    cacheElements();
    if (!el.view) return;
    el.tabCallouts.addEventListener("click", showCallouts);
    el.tabChat.addEventListener("click", showChat);
    el.mapSelect.addEventListener("change", function () {
      state.mapName = el.mapSelect.value;
      state.level = "default";
      resetView();
      loadCallouts();
    });
    el.levelDefault.addEventListener("click", function () { setLevel("default"); });
    el.levelLower.addEventListener("click", function () { setLevel("lower"); });
    el.resetAll.addEventListener("click", resetAll);
    el.addBtn.addEventListener("click", function () { setPlacing(!state.placing); });
    el.areasBtn.addEventListener("click", function () {
      state.showAreas = !state.showAreas;
      el.areasBtn.classList.toggle("active", state.showAreas);
      render();
    });
    // Capture phase: while placing, the drag wins over label buttons.
    el.labels.addEventListener("pointerdown", dragStart, true);
    el.labels.addEventListener("pointermove", dragMove, true);
    el.labels.addEventListener("pointerup", dragEnd, true);
    // Zoom for precision placement; pan only when zoomed and not placing.
    el.frame.addEventListener("wheel", onWheel, { passive: false });
    el.frame.addEventListener("pointerdown", panStart);
    el.frame.addEventListener("pointermove", panMove);
    el.frame.addEventListener("pointerup", panEnd);
    el.frame.addEventListener("dblclick", resetView);
    el.frame.addEventListener(
      "click",
      function (ev) {
        if (state.suppressClick) {
          state.suppressClick = false;
          ev.stopPropagation();
          ev.preventDefault();
        }
      },
      true
    );
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" && state.placing) setPlacing(false);
    });
  });
})();
