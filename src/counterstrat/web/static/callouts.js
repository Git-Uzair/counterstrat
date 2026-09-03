/**
 * Callout editor: rename map zones; the analyst then speaks YOUR names.
 * Aliases are stored per map and applied as the single vocabulary at the LLM
 * boundary - blank alias means the game's default name stays in use.
 */

(function () {
  "use strict";

  const state = {
    mapName: null,
    zones: [], // [{name, alias, u, v, level}]
    levels: ["default"],
    level: "default", // which radar level is on screen (nuke upper/lower)
    dirty: false,
  };

  const el = {};

  function cacheElements() {
    el.tabChat = document.getElementById("tab-chat");
    el.tabRadar = document.getElementById("tab-radar");
    el.tabCallouts = document.getElementById("tab-callouts");
    el.view = document.getElementById("callouts-view");
    el.mapSelect = document.getElementById("callouts-map");
    el.status = document.getElementById("callouts-status");
    el.image = document.getElementById("callouts-image");
    el.labels = document.getElementById("callouts-labels");
    el.levelToggle = document.getElementById("callouts-level-toggle");
    el.levelDefault = document.getElementById("callouts-level-default");
    el.levelLower = document.getElementById("callouts-level-lower");
    el.tableBody = document.querySelector("#callouts-table tbody");
    el.messages = document.getElementById("messages-container");
    el.inputBar = document.getElementById("chat-input-bar");
    el.radarView = document.getElementById("radar-view");
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
    state.zones.forEach(function (zone) {
      if (zone.u === null || zone.u === undefined) return;
      // On multi-level maps only the selected level's zones are shown, so the
      // underground site never clutters the upper radar (and vice versa).
      if (multiLevel && zone.level && zone.level !== state.level) return;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "callout-label" + (zone.alias ? " is-custom" : "");
      btn.style.left = `${zone.u * 100}%`;
      btn.style.top = `${zone.v * 100}%`;
      btn.textContent = labelFor(zone);
      btn.title = zone.alias
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
      nameTd.innerHTML = `<code>${escapeHtml(zone.name)}</code>`;
      const aliasTd = document.createElement("td");
      const input = document.createElement("input");
      input.type = "text";
      input.className = "form-control form-control-sm callout-table-input";
      input.value = zone.alias || "";
      input.placeholder = "(game name)";
      input.addEventListener("change", function () { setAlias(zone.name, input.value); });
      aliasTd.appendChild(input);
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
    if (el.radarView) el.radarView.classList.add("hidden");
    el.tabCallouts.classList.add("active");
    el.tabChat.classList.remove("active");
    el.tabRadar.classList.remove("active");
    loadMaps();
  }

  function hideCallouts() {
    el.view.classList.add("hidden");
    el.tabCallouts.classList.remove("active");
  }

  document.addEventListener("DOMContentLoaded", function () {
    cacheElements();
    if (!el.view) return;
    el.tabCallouts.addEventListener("click", showCallouts);
    // radar.js owns chat/radar switching; we only retract our own view.
    el.tabChat.addEventListener("click", hideCallouts);
    el.tabRadar.addEventListener("click", hideCallouts);
    el.mapSelect.addEventListener("change", function () {
      state.mapName = el.mapSelect.value;
      state.level = "default";
      loadCallouts();
    });
    el.levelDefault.addEventListener("click", function () { setLevel("default"); });
    el.levelLower.addEventListener("click", function () { setLevel("lower"); });
  });
})();
