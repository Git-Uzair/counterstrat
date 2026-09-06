/**
 * CS2 Counter-Strat Analyst Frontend Application
 * Vanilla JS client connecting to Counter-Strat FastAPI endpoints.
 */

(function () {
  "use strict";

  // Application State
  const state = {
    currentSessionId: null,
    currentTeamKey: null,
    currentMapName: null,
    currentTeamData: null,
    currentMatchIds: null, // null = full corpus scope; array = subset
    readGate: null, // token: chat stays gated while the CURRENT First Read runs
    setupReady: false, // ingest stays gated until GET /api/readiness says ready
    readinessTimer: null,
    isProcessing: false,
    pollTimers: {}, // jobId -> interval handle (one per queued ingest)
    settings: null,
  };

  // DOM Elements
  const el = {
    // Top Bar
    settingsBtn: document.getElementById("settings-btn"),
    cs2Banner: document.getElementById("cs2-banner"),
    cs2BannerBtn: document.getElementById("cs2-banner-btn"),
    
    // Ingestion
    dropZone: document.getElementById("drop-zone"),
    fileInput: document.getElementById("file-input"),
    browseBtn: document.getElementById("browse-btn"),
    setupChecklist: document.getElementById("setup-checklist"),
    setupChecklistItems: document.getElementById("setup-checklist-items"),
    ingestQueue: document.getElementById("ingest-queue"),

    // Catalog
    teamsList: document.getElementById("teams-list"),
    teamsEmpty: document.getElementById("teams-empty"),
    catalogCount: document.getElementById("catalog-count"),

    // Chat
    targetTitle: document.getElementById("target-title"),
    targetMeta: document.getElementById("target-meta"),
    deleteChatBtn: document.getElementById("delete-chat-btn"),

    messagesContainer: document.getElementById("messages-container"),
    firstReadPanel: document.getElementById("first-read-panel"),
    firstReadWait: document.getElementById("first-read-wait"),
    firstReadWaitLabel: document.getElementById("first-read-wait-label"),
    chatWelcome: document.getElementById("chat-welcome"),
    chatForm: document.getElementById("chat-form"),
    chatInput: document.getElementById("chat-input"),
    chatInputBar: document.getElementById("chat-input-bar"),
    sendBtn: document.getElementById("send-btn"),
    chatSpinner: document.getElementById("chat-spinner"),

    // Settings Modal
    settingsModal: document.getElementById("settings-modal"),
    closeSettingsBtn: document.getElementById("close-settings-btn"),
    cancelSettingsBtn: document.getElementById("cancel-settings-btn"),
    settingsForm: document.getElementById("settings-form"),
    settingProvider: document.getElementById("setting-provider"),
    settingModel: document.getElementById("setting-model"),
    settingApiKey: document.getElementById("setting-api-key"),
    settingCs2Path: document.getElementById("setting-cs2-path"),
    cs2PathStatus: document.getElementById("cs2-path-status"),
    keyStatusIndicator: document.getElementById("key-status-indicator"),
    settingsFeedback: document.getElementById("settings-feedback"),
    saveSettingsBtn: document.getElementById("save-settings-btn"),
  };

  // Helper: Escape HTML
  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Helper: Format message text with markdown-like code fences and inline backticks
  function formatMessageText(rawText) {
    if (!rawText) return "";
    
    // Extract fenced code blocks first
    const codeBlocks = [];
    let text = rawText.replace(/```([\w-]*)\n?([\s\S]*?)```/g, function (_, lang, code) {
      const idx = codeBlocks.length;
      codeBlocks.push({ lang, code });
      return `@@CODE_BLOCK_${idx}@@`;
    });

    // Escape HTML of the remaining text
    text = escapeHtml(text);

    // Replace bold text **bold**
    text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

    // Replace inline backticks `zone`
    text = text.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Replace paragraphs / line breaks
    text = text.split("\n\n").map(function (para) {
      return "<p>" + para.replace(/\n/g, "<br>") + "</p>";
    }).join("");

    // Reinsert code blocks
    text = text.replace(/@@CODE_BLOCK_(\d+)@@/g, function (_, idx) {
      const block = codeBlocks[Number(idx)];
      return `<pre><code>${escapeHtml(block.code)}</code></pre>`;
    });

    return text;
  }

  // Scroll messages pane to bottom
  function scrollToBottom() {
    el.messagesContainer.scrollTop = el.messagesContainer.scrollHeight;
  }

  // =========================================================================
  // 1. Ingestion & Job Polling
  // =========================================================================

  function initDropZone() {
    ["dragenter", "dragover"].forEach(function (eventName) {
      el.dropZone.addEventListener(eventName, function (e) {
        e.preventDefault();
        e.stopPropagation();
        el.dropZone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach(function (eventName) {
      el.dropZone.addEventListener(eventName, function (e) {
        e.preventDefault();
        e.stopPropagation();
        el.dropZone.classList.remove("dragover");
      });
    });

    el.dropZone.addEventListener("drop", function (e) {
      uploadDemoFiles(e.dataTransfer.files);
    });

    el.browseBtn.addEventListener("click", function () {
      el.fileInput.click();
    });

    el.fileInput.addEventListener("change", function () {
      uploadDemoFiles(el.fileInput.files);
      el.fileInput.value = "";
    });
  }

  // =========================================================================
  // Setup readiness gate: no ingest until the app is fully prepared. The
  // server enforces it too (POST /demos -> 409); this keeps the UI honest,
  // shows what's left, and spins while the bootstrap works.
  // =========================================================================

  function renderChecklistItem(item) {
    const li = document.createElement("li");
    li.className = "setup-item " + (item.ok ? "setup-item-ok" : "setup-item-missing");
    let icon = item.ok ? "\u2713" : "\u2717";
    let spin = "";
    const busy =
      !item.ok && item.id === "decompiler" &&
      (item.status === "downloading" || item.status === "warming");
    if (busy) {
      icon = "";
      spin = '<span class="setup-spinner"></span>';
    }
    let fix = "";
    if (!item.ok && (item.id === "cs2_path" || item.id === "api_key")) {
      fix = '<button type="button" class="btn btn-sm btn-outline setup-fix-btn">Open Settings</button>';
    }
    li.innerHTML =
      `<span class="setup-item-icon">${icon}</span>${spin}` +
      `<span class="setup-item-label">${escapeHtml(item.label)}</span>` +
      `<span class="setup-item-detail">${escapeHtml(item.detail || "")}</span>` +
      fix;
    const btn = li.querySelector(".setup-fix-btn");
    if (btn) btn.addEventListener("click", openSettings);
    return li;
  }

  function applyReadiness(data) {
    state.setupReady = !!data.ready;
    el.setupChecklist.classList.toggle("hidden", state.setupReady);
    el.dropZone.classList.toggle("gated", !state.setupReady);
    el.browseBtn.disabled = !state.setupReady;
    if (!state.setupReady) {
      el.setupChecklistItems.innerHTML = "";
      (data.items || []).forEach(function (item) {
        el.setupChecklistItems.appendChild(renderChecklistItem(item));
      });
    }
  }

  function pollReadiness() {
    fetch("/api/readiness")
      .then(function (res) {
        if (!res.ok) throw new Error("readiness unavailable");
        return res.json();
      })
      .then(function (data) {
        applyReadiness(data);
        if (!state.setupReady) {
          clearTimeout(state.readinessTimer);
          state.readinessTimer = setTimeout(pollReadiness, 2500);
        }
      })
      .catch(function (err) {
        console.error("Readiness poll failed:", err);
        clearTimeout(state.readinessTimer);
        state.readinessTimer = setTimeout(pollReadiness, 5000);
      });
  }

  // Multi-file ingestion: one queue row per file, uploads run one at a time,
  // each accepted job polls independently. Server serializes the pipelines.
  function uploadDemoFiles(fileList) {
    if (!state.setupReady) {
      el.setupChecklist.classList.remove("hidden");
      return; // the checklist says what's left; the server would 409 anyway
    }
    const files = Array.from(fileList || []);
    if (files.length === 0) return;
    el.ingestQueue.classList.remove("hidden");
    let chain = Promise.resolve();
    files.forEach(function (file) {
      const row = addQueueRow(file.name);
      chain = chain.then(function () {
        return uploadOneDemo(file, row);
      });
    });
  }

  function addQueueRow(name) {
    const row = document.createElement("div");
    row.className = "queue-item";
    row.innerHTML =
      '<div class="queue-item-head">' +
      `<span class="queue-name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>` +
      '<span class="badge badge-stage">waiting</span>' +
      "</div>" +
      '<div class="queue-detail"></div>';
    el.ingestQueue.appendChild(row);
    return row;
  }

  function setQueueRow(row, stage, detail, statusClass) {
    row.querySelector(".badge").textContent = stage;
    row.querySelector(".queue-detail").textContent = detail || "";
    row.classList.remove("status-error", "status-done", "status-duplicate", "status-running");
    if (statusClass) row.classList.add(statusClass);
  }

  // Finished rows clean themselves up; errors stay until the page reloads.
  function dismissQueueRow(row, delayMs) {
    setTimeout(function () {
      row.remove();
      if (el.ingestQueue.children.length === 0) {
        el.ingestQueue.classList.add("hidden");
      }
    }, delayMs);
  }

  function uploadOneDemo(file, row) {
    setQueueRow(row, "uploading", `Uploading ${file.name}...`, "status-running");
    const formData = new FormData();
    formData.append("demo", file);
    return fetch("/api/demos", { method: "POST", body: formData })
      .then(function (res) {
        if (!res.ok) {
          return res.json().then(function (err) {
            throw new Error(err.detail || "Failed to upload demo");
          });
        }
        return res.json();
      })
      .then(function (data) {
        setQueueRow(row, "queued", "Waiting for the ingest pipeline...", "status-running");
        pollJobStatus(data.job_id, row);
      })
      .catch(function (err) {
        setQueueRow(row, "error", err.message, "status-error");
      });
  }

  function pollJobStatus(jobId, row) {
    state.pollTimers[jobId] = setInterval(function () {
      fetch(`/api/jobs/${jobId}`)
        .then(function (res) {
          if (!res.ok) {
            throw new Error(`Status check returned ${res.status}`);
          }
          return res.json();
        })
        .then(function (job) {
          if (job.stage === "done") {
            clearInterval(state.pollTimers[jobId]);
            delete state.pollTimers[jobId];
            setQueueRow(row, "done", `Ingested match ${job.match_id || ""}`, "status-done");
            loadTeams();
            dismissQueueRow(row, 1200);
          } else if (job.stage === "duplicate") {
            clearInterval(state.pollTimers[jobId]);
            delete state.pollTimers[jobId];
            setQueueRow(row, "duplicate", job.detail || "Already ingested", "status-duplicate");
            dismissQueueRow(row, 6000);
          } else if (job.stage === "error") {
            clearInterval(state.pollTimers[jobId]);
            delete state.pollTimers[jobId];
            setQueueRow(
              row, "error", job.detail || "Ingestion pipeline encountered an error", "status-error"
            );
          } else {
            setQueueRow(row, job.stage, job.detail || `Running stage: ${job.stage}...`, "status-running");
          }
        })
        .catch(function (err) {
          clearInterval(state.pollTimers[jobId]);
          delete state.pollTimers[jobId];
          setQueueRow(row, "error", err.message, "status-error");
        });
    }, 2000);
  }

  // =========================================================================
  // 2. Catalog & Selection
  // =========================================================================

  function loadTeams() {
    fetch("/api/teams")
      .then(function (res) {
        if (!res.ok) throw new Error("Failed to load teams");
        return res.json();
      })
      .then(function (teams) {
        renderTeams(teams);
      })
      .catch(function (err) {
        console.error("Error loading teams:", err);
      });
  }

  function teamDisplay(team) {
    return (team.names && team.names.length > 0) ? team.names.join(" / ") : team.team_key;
  }

  function fmtAdded(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    if (isNaN(d.getTime())) return "";
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  }

  function matchLine(m) {
    const opp = m.opponent_name ? `vs ${m.opponent_name}` : "vs unknown";
    const score = (m.score_won !== null && m.score_won !== undefined)
      ? `${m.score_won}\u2013${m.score_lost}`
      : "";
    return { opp: opp, score: score, added: fmtAdded(m.added_at) };
  }

  function renderTeams(teams) {
    el.teamsList.innerHTML = "";
    if (!teams || teams.length === 0) {
      el.teamsEmpty.classList.remove("hidden");
      el.catalogCount.textContent = "0 teams";
      return;
    }

    el.teamsEmpty.classList.add("hidden");
    let totalMatches = 0;

    teams.forEach(function (team) {
      const displayName = teamDisplay(team);
      const maps = (team.maps && team.maps.length > 0) ? team.maps : [];
      const group = document.createElement("div");
      group.className = "team-group";

      const headRow = document.createElement("div");
      headRow.className = "team-group-head-row";

      const header = document.createElement("button");
      header.type = "button";
      header.className = "team-group-header";
      header.innerHTML =
        `<span class="team-group-chevron">\u25be</span>` +
        `<span class="team-group-name">${escapeHtml(displayName)}</span>` +
        `<span class="team-group-meta">${maps.length} map${maps.length === 1 ? "" : "s"} \u00b7 ` +
        `${team.demos} match${team.demos === 1 ? "" : "es"}</span>`;
      header.addEventListener("click", function () {
        group.classList.toggle("collapsed");
      });
      headRow.appendChild(header);

      const teamDelete = document.createElement("button");
      teamDelete.type = "button";
      teamDelete.className = "team-delete";
      teamDelete.textContent = "\u00d7";
      teamDelete.title = `Delete team ${displayName} and every match under it`;
      teamDelete.addEventListener("click", function (ev) {
        ev.stopPropagation();
        const allIds = [];
        maps.forEach(function (mapName) {
          const mStats = (team.map_stats && team.map_stats[mapName]) || {};
          (mStats.matches || []).forEach(function (m) {
            allIds.push(m.match_id);
          });
        });
        deleteMatches(
          allIds,
          `Delete team ${displayName} entirely ` +
            `(${allIds.length} match${allIds.length === 1 ? "" : "es"} across ` +
            `${maps.length} map${maps.length === 1 ? "" : "s"})?`
        );
      });
      headRow.appendChild(teamDelete);
      group.appendChild(headRow);

      const body = document.createElement("div");
      body.className = "team-group-body";
      maps.forEach(function (mapName) {
        const mStats = (team.map_stats && team.map_stats[mapName]) ||
          { rounds: team.rounds, demos: team.demos, matches: [] };
        totalMatches += (mStats.matches || []).length;
        body.appendChild(buildMapCard(team, displayName, mapName, mStats));
      });
      group.appendChild(body);
      el.teamsList.appendChild(group);
    });

    // Every match is listed under both teams: show the team count instead.
    el.catalogCount.textContent = `${teams.length} team${teams.length === 1 ? "" : "s"}`;
  }

  function buildMapCard(team, displayName, mapName, mStats) {
    const matches = mStats.matches || [];
    const card = document.createElement("div");
    card.className = "team-card map-card";
    card.dataset.teamKey = team.team_key;
    card.dataset.mapName = mapName;

    const head = document.createElement("div");
    head.className = "team-card-title";
    head.innerHTML =
      `<span class="team-card-map">${escapeHtml(mapName)}</span>` +
      `<span class="map-card-meta"><strong class="stat-n">n=${mStats.rounds}</strong> \u00b7 ` +
      `${matches.length} match${matches.length === 1 ? "" : "es"}</span>`;
    const delAll = document.createElement("button");
    delAll.type = "button";
    delAll.className = "map-delete";
    delAll.textContent = "\u00d7";
    delAll.title = `Delete all ${matches.length} ${mapName} match(es) of ${displayName}`;
    delAll.addEventListener("click", function (ev) {
      ev.stopPropagation();
      deleteMatches(
        matches.map(function (m) { return m.match_id; }),
        `Delete all ${matches.length} ${mapName} match(es) of ${displayName}?`
      );
    });
    head.appendChild(delAll);
    card.appendChild(head);

    const list = document.createElement("div");
    list.className = "match-list";
    matches.forEach(function (m) {
      list.appendChild(buildMatchRow(team, displayName, mapName, card, m));
    });
    card.appendChild(list);

    const footer = document.createElement("div");
    footer.className = "match-actions";
    const analyze = document.createElement("button");
    analyze.type = "button";
    analyze.className = "btn btn-sm btn-primary analyze-btn";
    footer.appendChild(analyze);
    card.appendChild(footer);

    function selection() {
      const checks = Array.from(card.querySelectorAll(".match-check"));
      const picked = checks.filter(function (c) { return c.checked; });
      return {
        ids: picked.map(function (c) { return c.dataset.matchId; }),
        rounds: picked.reduce(function (acc, c) { return acc + Number(c.dataset.rounds || 0); }, 0),
        all: picked.length === checks.length && checks.length > 0,
      };
    }

    function refreshAnalyzeLabel() {
      const sel = selection();
      analyze.disabled = sel.ids.length === 0;
      if (sel.all) {
        analyze.textContent = `Analyze all ${sel.ids.length} \u00b7 n=${mStats.rounds}`;
      } else {
        analyze.textContent =
          `Analyze ${sel.ids.length} of ${matches.length} \u00b7 n=${sel.rounds}`;
      }
    }
    card.addEventListener("change", function (ev) {
      if (ev.target.classList.contains("match-check")) refreshAnalyzeLabel();
    });
    refreshAnalyzeLabel();

    analyze.addEventListener("click", function (ev) {
      ev.stopPropagation();
      const sel = selection();
      if (sel.ids.length === 0) return;
      // Analyze is the explicit ask: generate the First Read for this scope.
      selectTarget(team, displayName, mapName, card, sel.all ? null : sel.ids, true);
    });

    return card;
  }

  function buildMatchRow(team, displayName, mapName, card, m) {
    const line = matchLine(m);
    const row = document.createElement("div");
    row.className = "match-row";
    row.title = `Match ${m.match_id} \u00b7 ${m.rounds} rounds`;

    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = true;
    check.className = "match-check";
    check.dataset.matchId = m.match_id;
    check.dataset.rounds = m.rounds;
    row.appendChild(check);

    const label = document.createElement("button");
    label.type = "button";
    label.className = "match-label";
    label.title = `Analyze only this match (${m.match_id})`;
    label.innerHTML =
      `<span class="match-opp">${escapeHtml(line.opp)}</span>` +
      (line.score ? `<span class="match-score">${escapeHtml(line.score)}</span>` : "") +
      `<span class="match-rounds">${m.rounds}r</span>` +
      (line.added ? `<span class="match-added">added ${escapeHtml(line.added)}</span>` : "");
    label.addEventListener("click", function (ev) {
      ev.stopPropagation();
      selectTarget(team, displayName, mapName, card, [m.match_id]);
    });
    row.appendChild(label);

    const del = document.createElement("button");
    del.type = "button";
    del.className = "match-delete";
    del.textContent = "\u00d7";
    del.title = `Delete this match (${line.opp} ${line.score}) and everything mined from it`;
    del.addEventListener("click", function (ev) {
      ev.stopPropagation();
      deleteMatches(
        [m.match_id],
        `Delete this ${mapName} match of ${displayName} (${line.opp}${line.score ? " " + line.score : ""})?`
      );
    });
    row.appendChild(del);

    // Clicking the row background toggles the checkbox.
    row.addEventListener("click", function (ev) {
      if (ev.target === row) {
        check.checked = !check.checked;
        check.dispatchEvent(new Event("change", { bubbles: true }));
      }
    });

    return row;
  }

  function deleteMatches(matchIds, question) {
    if (!matchIds || matchIds.length === 0) return;
    const ok = window.confirm(
      question +
        "\n\nEach demo file, its parsed data, and everything mined from it are removed - " +
        "including from the opposing team's card (a demo belongs to both teams). " +
        "Team profiles are rebuilt from the remaining demos."
    );
    if (!ok) return;
    fetch(`/api/demos?matches=${encodeURIComponent(matchIds.join(","))}`, { method: "DELETE" })
      .then(function (res) {
        if (!res.ok) {
          return res.json().then(function (body) {
            throw new Error((body && body.detail) || `Server returned ${res.status}`);
          });
        }
        return res.json();
      })
      .then(function () {
        // Deletion invalidates scopes server-side; mirror it in the view.
        state.currentSessionId = null;
        state.currentMatchIds = null;
        clearFirstRead();
        el.messagesContainer.innerHTML = "";
        el.targetTitle.textContent = "No Team Selected";
        el.targetMeta.textContent = "Pick a target from the catalog on the left to start";
        el.chatInput.disabled = true;
        el.sendBtn.disabled = true;
        el.deleteChatBtn.classList.add("hidden");
        loadTeams();
      })
      .catch(function (err) {
        alert("Delete failed: " + err.message);
      });
  }

  function selectTarget(team, displayName, mapName, cardEl, matchIds, autoGenerate) {
    // Update active highlight
    document.querySelectorAll(".team-card").forEach(function (c) {
      c.classList.remove("selected");
    });
    if (cardEl) cardEl.classList.add("selected");

    state.currentTeamKey = team.team_key;
    state.currentMapName = mapName;
    state.currentTeamData = team;
    state.currentMatchIds = matchIds || null;
    state.currentDisplayName = displayName;

    const mStats = (team.map_stats && team.map_stats[mapName]) || { rounds: team.rounds, demos: team.demos };
    const matches = mStats.matches || [];

    // Update Header
    el.targetTitle.textContent = `${displayName} - ${mapName}`;
    if (matchIds && matchIds.length > 0) {
      const picked = matches.filter(function (m) { return matchIds.indexOf(m.match_id) !== -1; });
      const rounds = picked.reduce(function (acc, m) { return acc + m.rounds; }, 0);
      const names = picked.map(function (m) {
        const line = matchLine(m);
        return `${line.opp}${line.score ? " " + line.score : ""}`;
      });
      el.targetMeta.innerHTML =
        `Scope: <strong class="stat-n">${matchIds.length} of ${matches.length}</strong> matches ` +
        `(n = ${rounds} rounds) \u2014 ${escapeHtml(names.join(", "))}`;
    } else {
      el.targetMeta.innerHTML = `Sample size: <strong class="stat-n">n = ${mStats.rounds} rounds</strong> across ${mStats.demos} demo${mStats.demos === 1 ? "" : "s"} (${escapeHtml(team.team_key)})`;
    }

    // Create session
    createChatSession(team.team_key, mapName, displayName, matchIds || null, autoGenerate);
  }

  // =========================================================================
  // 3. Chat Sessions & Messaging
  // =========================================================================

  // Session creation mines the whole scope server-side (tendencies, utility,
  // gaps, economy) and can take seconds on a big corpus: the right pane shows
  // a working bar and the chat input stays hidden until the session exists -
  // nobody types into a session that is not there yet.
  function showSessionLoading(displayName, mapName, matchIds) {
    removeSessionLoading();
    const strip = document.createElement("div");
    strip.className = "session-loading";
    strip.id = "session-loading";
    const scope =
      matchIds && matchIds.length
        ? `${matchIds.length} match${matchIds.length === 1 ? "" : "es"}`
        : "all matches";
    strip.innerHTML =
      '<div class="session-loading-label">Preparing analyst session for ' +
      `<strong>${escapeHtml(displayName)}</strong> on <code>${escapeHtml(mapName)}</code>` +
      ` \u2014 mining tendencies, utility, gaps and economy over ${scope}\u2026</div>` +
      '<div class="progress-bar-container">' +
      '<div class="progress-bar-fill progress-indeterminate"></div></div>';
    el.messagesContainer.appendChild(strip);
    strip.scrollIntoView({ block: "end" });
  }

  function removeSessionLoading() {
    const strip = document.getElementById("session-loading");
    if (strip) strip.remove();
  }

  // While the AI First Read generates (Analyze click), the chat form is
  // replaced by a progress bar INSIDE the input bar - the transcript replay
  // can't wipe it and the Callouts tab toggle hides it with the bar.
  function gateChatBehindFirstRead(displayName, mapName) {
    el.firstReadWaitLabel.innerHTML =
      "AI First Read in progress for " +
      `<strong>${escapeHtml(displayName)}</strong> on <code>${escapeHtml(mapName)}</code>` +
      " \u2014 the analyst is reading the selected games. Chat opens when the report is ready\u2026";
    el.chatForm.classList.add("hidden");
    el.firstReadWait.classList.remove("hidden");
  }

  function ungateChat() {
    el.firstReadWait.classList.add("hidden");
    el.chatForm.classList.remove("hidden");
  }

  function createChatSession(teamKey, mapName, displayName, matchIds, autoGenerate) {
    el.chatInput.disabled = true;
    el.sendBtn.disabled = true;
    el.chatInputBar.classList.add("hidden");
    showSessionLoading(displayName, mapName, matchIds);

    fetch("/api/chat/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ team_key: teamKey, map_name: mapName, match_ids: matchIds || null }),
    })
      .then(function (res) {
        if (!res.ok) {
          return res.json().then(
            function (errData) {
              throw new Error((errData && errData.detail) || "Failed to create chat session");
            },
            function () {
              throw new Error("Failed to create chat session");
            }
          );
        }
        return res.json();
      })
      .then(function (data) {
        // The session id IS the scope hash: same selection -> same session.
        const sameScope = state.currentSessionId === data.session_id;
        state.currentSessionId = data.session_id;

        // The First Read panel renders for EVERY scope, but only an explicit
        // Analyze click spends an LLM call; picking a match title just probes
        // the cache and otherwise waits for Generate/Regenerate.
        const readPromise = renderFirstRead(
          teamKey, mapName, matchIds, autoGenerate ? "generate" : "probe"
        );

        removeSessionLoading();
        if (!sameScope) {
          el.messagesContainer.innerHTML = "";
          replayTranscript(data.session_id, displayName, mapName, matchIds);
        }
        el.deleteChatBtn.classList.remove("hidden");
        el.chatInputBar.classList.remove("hidden");

        if (autoGenerate) {
          // Analyze: no chat until the First Read lands - the input form is
          // swapped for a progress bar until the report (or its error card)
          // is on screen. The gate token keeps a stale finally from ungating
          // a newer Analyze still in flight.
          const gate = {};
          state.readGate = gate;
          gateChatBehindFirstRead(displayName, mapName);
          readPromise.finally(function () {
            if (state.readGate !== gate) return;
            state.readGate = null;
            ungateChat();
            el.chatInput.disabled = false;
            el.sendBtn.disabled = false;
            el.chatInput.focus();
          });
        } else {
          state.readGate = null;
          ungateChat();
          el.chatInput.disabled = false;
          el.sendBtn.disabled = false;
          el.chatInput.focus();
        }
      })
      .catch(function (err) {
        removeSessionLoading();
        el.chatInputBar.classList.remove("hidden");
        alert("Failed to initialize session: " + err.message);
      });
  }

  // Delete the current conversation's history; the scope (and its First Read)
  // survives, so a fresh chat starts immediately for the same selection.
  function deleteCurrentChat() {
    if (!state.currentSessionId || state.isProcessing) return;
    const ok = window.confirm(
      "Delete this conversation's history?\n\nThe AI First Read and mined data are kept; " +
        "a fresh chat starts for the same selection."
    );
    if (!ok) return;
    const sid = state.currentSessionId;
    fetch(`/api/chat/sessions/${encodeURIComponent(sid)}`, { method: "DELETE" })
      .then(function (res) {
        if (!res.ok && res.status !== 404) {
          throw new Error(`Server returned ${res.status}`);
        }
        // Same scope, fresh conversation: null the id so the welcome replays.
        state.currentSessionId = null;
        el.messagesContainer.innerHTML = "";
        createChatSession(
          state.currentTeamKey,
          state.currentMapName,
          state.currentDisplayName || "team",
          state.currentMatchIds
        );
      })
      .catch(function (err) {
        alert("Delete chat failed: " + err.message);
      });
  }

  // Prior conversation for this exact selection, or a fresh welcome.
  function replayTranscript(sessionId, displayName, mapName, matchIds) {
    fetch(`/api/chat/sessions/${encodeURIComponent(sessionId)}`)
      .then(function (res) { return res.ok ? res.json() : { messages: [] }; })
      .then(function (data) {
        el.messagesContainer.innerHTML = "";
        const messages = data.messages || [];
        if (messages.length === 0) {
          const scopeNote = (matchIds && matchIds.length > 0)
            ? (matchIds.length === 1
              ? `\n\n**Scope: 1 match only** - every answer describes this one game.`
              : `\n\n**Scope: ${matchIds.length} selected matches** - answers describe these games only.`)
            : "";
          appendAssistantMessage({
            text: `Active session started for **${displayName}** on \`${mapName}\`.${scopeNote}\n\nYou can ask about buy-round tendencies, utility setups, opening duels, or cite specific rounds.`,
            tool_trace: [],
            warnings: [],
          });
          return;
        }
        messages.forEach(function (m) {
          if (m.role === "user") appendUserMessage(m.text);
          else appendAssistantMessage({ text: m.text, tool_trace: [], warnings: [] });
        });
      })
      .catch(function () { /* a fresh pane is fine */ });
  }

  // AI First Read: generated for the exact selection, rendered as cards in
  // the pinned panel above the chat. Cached scopes are instant. mode:
  //   "probe"    - serve the cached read if one exists; otherwise show a
  //                Generate button and make NO LLM call (match title click).
  //   "generate" - serve the cache or generate the read (Analyze / Generate).
  //   "force"    - skip the cache and re-run the read (Regenerate).
  function renderFirstRead(teamKey, mapName, matchIds, mode) {
    mode = mode || "probe";
    el.firstReadPanel.classList.remove("hidden", "collapsed");
    el.firstReadPanel.innerHTML =
      '<div class="first-read-header">' +
      '<span class="first-read-title"><span class="insights-chevron">\u25be</span> AI First Read</span>' +
      '<span class="first-read-actions">' +
      '<button type="button" class="first-read-regen" ' +
      'title="Re-run the AI First Read for this selection (one LLM call)">' +
      "\u21bb Regenerate</button>" +
      '<span class="first-read-meta"></span>' +
      "</span>" +
      "</div>" +
      '<div class="first-read-cards">' +
      '<div class="first-read-card is-loading">' +
      (mode === "force"
        ? "Re-reading the selected games\u2026"
        : mode === "generate"
          ? "Reading the selected games\u2026"
          : "Checking for a saved read\u2026") +
      "</div>" +
      "</div>";
    el.firstReadPanel
      .querySelector(".first-read-title")
      .addEventListener("click", function () {
        el.firstReadPanel.classList.toggle("collapsed");
      });
    const regenBtn = el.firstReadPanel.querySelector(".first-read-regen");
    let nextMode = "force"; // Regenerate - unless the probe finds nothing yet
    regenBtn.disabled = true; // no double-generate while a fetch is in flight
    regenBtn.addEventListener("click", function (ev) {
      ev.stopPropagation();
      renderFirstRead(teamKey, mapName, matchIds, nextMode);
    });

    const isMock = window.location.search.includes("mock=1");
    const params = new URLSearchParams();
    if (mode !== "probe") params.set("generate", "1");
    if (mode === "force") params.set("force", "1");
    if (isMock) params.set("mock", "1");
    if (matchIds && matchIds.length) params.set("matches", matchIds.join(","));
    // Returned so Analyze can hold the chat gate until the read settles.
    return fetch(
      `/api/teams/${encodeURIComponent(teamKey)}/${encodeURIComponent(mapName)}/insights?` +
        params.toString()
    )
      .then(function (res) {
        if (res.ok) return res.json();
        return res.json().then(function (body) {
          throw { status: res.status, detail: (body && body.detail) || "" };
        });
      })
      .then(function (data) { renderFirstReadCards(data); })
      .catch(function (err) {
        const cards = el.firstReadPanel.querySelector(".first-read-cards");
        if (mode === "probe" && err && err.status === 404) {
          // Nothing cached for this scope: wait for an explicit click.
          nextMode = "generate";
          regenBtn.textContent = "Generate";
          regenBtn.title =
            "Generate the AI First Read for this selection (one LLM call)";
          cards.innerHTML =
            '<div class="first-read-card is-hint">No AI First Read for this selection yet - ' +
            "click Generate to run it (one LLM call).</div>";
        } else if (err && err.status === 503) {
          cards.innerHTML =
            '<div class="first-read-card is-hint">No API key configured - set one in ' +
            "Settings to generate the AI First Read for this selection.</div>";
        } else {
          cards.innerHTML =
            '<div class="first-read-card is-hint">First Read failed: ' +
            escapeHtml((err && err.detail) || "unknown error") +
            "</div>";
        }
      })
      .finally(function () {
        regenBtn.disabled = false;
      });
  }

  function renderFirstReadCards(data) {
    const meta = el.firstReadPanel.querySelector(".first-read-meta");
    const games = (data.games || []).map(function (g) { return g.label; });
    meta.textContent =
      `${games.join(" \u00b7 ")}${games.length ? " \u00b7 " : ""}${data.model || ""}`;

    // The prompt mandates six "## " sections: each becomes a card.
    const chunks = String(data.text || "").split(/\n(?=## )/);
    let html = "";
    chunks.forEach(function (chunk) {
      const trimmed = chunk.trim();
      if (!trimmed) return;
      const lines = trimmed.split("\n");
      let title = "";
      let body = trimmed;
      if (lines[0].startsWith("## ")) {
        title = lines[0].slice(3).trim();
        body = lines.slice(1).join("\n").trim();
      }
      html +=
        '<div class="first-read-card">' +
        (title ? `<div class="first-read-card-title">${escapeHtml(title)}</div>` : "") +
        `<div class="first-read-card-body">${formatMessageText(body)}</div>` +
        "</div>";
    });
    if (data.warnings && data.warnings.length) {
      html +=
        '<div class="first-read-card is-warnings"><div class="first-read-card-title">' +
        "Verification warnings</div><div class=\"first-read-card-body\">" +
        escapeHtml(data.warnings.join("; ")) +
        "</div></div>";
    }
    el.firstReadPanel.querySelector(".first-read-cards").innerHTML =
      html || '<div class="first-read-card is-hint">The model returned nothing readable.</div>';
  }

  function clearFirstRead() {
    el.firstReadPanel.classList.add("hidden");
    el.firstReadPanel.innerHTML = "";
  }

  function sendMessage() {
    const text = el.chatInput.value.trim();
    if (!text || state.isProcessing || !state.currentSessionId) return;

    // Append user message bubble
    appendUserMessage(text);
    el.chatInput.value = "";

    // Set processing state
    state.isProcessing = true;
    el.chatInput.disabled = true;
    el.sendBtn.disabled = true;
    el.chatSpinner.classList.remove("hidden");

    const isMock = window.location.search.includes("mock=1");
    const msgUrl = `/api/chat/sessions/${state.currentSessionId}/messages${isMock ? "?mock=1" : ""}`;

    fetch(msgUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    })
      .then(function (res) {
        if (!res.ok) {
          return res.json().then(function (err) {
            throw new Error(err.detail || `Server returned ${res.status}`);
          });
        }
        return res.json();
      })
      .then(function (reply) {
        appendAssistantMessage(reply);
      })
      .catch(function (err) {
        appendErrorMessage(err.message);
      })
      .finally(function () {
        state.isProcessing = false;
        el.chatInput.disabled = false;
        el.sendBtn.disabled = false;
        el.chatSpinner.classList.add("hidden");
        el.chatInput.focus();
        scrollToBottom();
      });
  }

  function appendUserMessage(text) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message user";
    msgDiv.innerHTML = `
      <div class="message-bubble">
        <div class="message-content">${escapeHtml(text).replace(/\n/g, "<br>")}</div>
      </div>
    `;
    el.messagesContainer.appendChild(msgDiv);
    scrollToBottom();
  }

  function appendAssistantMessage(reply) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message assistant";

    let toolsHtml = "";
    if (reply.tool_trace && reply.tool_trace.length > 0) {
      const items = reply.tool_trace
        .map(function (t) {
          const args = t.arguments ? JSON.stringify(t.arguments) : "{}";
          const preview = t.result_preview
            ? `<div class="tool-preview">${escapeHtml(t.result_preview)}</div>`
            : "";
          return `
            <div class="tool-call-item">
              <span class="tool-name">${escapeHtml(t.name)}</span>(<span class="tool-args">${escapeHtml(args)}</span>)
              ${preview}
            </div>
          `;
        })
        .join("");

      toolsHtml = `
        <details class="tool-trace">
          <summary>Tools used (${reply.tool_trace.length})</summary>
          <div class="tool-trace-body">${items}</div>
        </details>
      `;
    }

    let warningsHtml = "";
    if (reply.warnings && reply.warnings.length > 0) {
      const warningItems = reply.warnings
        .map(function (w) {
          return `<li>${escapeHtml(w)}</li>`;
        })
        .join("");

      warningsHtml = `
        <div class="warnings-container">
          <div class="warning-header">[Warning] Soft Lint Findings (${reply.warnings.length}):</div>
          <ul class="warning-list">${warningItems}</ul>
        </div>
      `;
    }

    const formattedBody = formatMessageText(reply.text);

    msgDiv.innerHTML = `
      <div class="message-bubble">
        ${toolsHtml}
        ${warningsHtml}
        <div class="message-content">${formattedBody}</div>
      </div>
    `;

    el.messagesContainer.appendChild(msgDiv);
    scrollToBottom();
  }

  function appendErrorMessage(errText) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message assistant error";
    msgDiv.innerHTML = `
      <div class="message-bubble">
        <strong>Error:</strong> ${escapeHtml(errText)}
      </div>
    `;
    el.messagesContainer.appendChild(msgDiv);
    scrollToBottom();
  }

  // =========================================================================
  // 5. Settings Modal
  // =========================================================================

  function openSettings() {
    el.settingsFeedback.classList.add("hidden");
    el.settingApiKey.value = "";
    el.settingsModal.classList.remove("hidden");
    loadSettings();
  }

  function closeSettings() {
    el.settingsModal.classList.add("hidden");
  }

  function loadSettings() {
    fetch("/api/settings")
      .then(function (res) {
        if (!res.ok) throw new Error("Failed to load settings");
        return res.json();
      })
      .then(function (cfg) {
        state.settings = cfg;
        el.settingProvider.value = cfg.provider;
        updateKeyIndicator(cfg.provider);
        el.settingCs2Path.value = cfg.cs2_install_path || "";
        updateCs2Status(cfg);
        loadModels(cfg.provider, cfg.model);
      })
      .catch(function (err) {
        console.error("Settings load error:", err);
      });
  }

  // The CS2 install is required kit (radar art, callout maps and new-map
  // ingest all read the game VPK): a banner stays up and the settings badge
  // stays red until the configured folder actually looks like an install.
  function updateCs2Status(cfg) {
    el.cs2Banner.classList.toggle("hidden", !!cfg.cs2_path_valid);
    if (cfg.cs2_path_valid) {
      el.cs2PathStatus.textContent = "Folder found";
      el.cs2PathStatus.className = "key-status badge-configured";
    } else if (cfg.cs2_install_path) {
      el.cs2PathStatus.textContent = "Folder invalid \u2014 required";
      el.cs2PathStatus.className = "key-status badge-unconfigured";
    } else {
      el.cs2PathStatus.textContent = "Not set \u2014 required";
      el.cs2PathStatus.className = "key-status badge-unconfigured";
    }
  }

  function updateKeyIndicator(provider) {
    const isSet = state.settings && state.settings.keys_present && state.settings.keys_present[provider];
    if (isSet) {
      el.keyStatusIndicator.textContent = "Key set (configured)";
      el.keyStatusIndicator.className = "key-status badge-configured";
      el.settingApiKey.placeholder = "(Unchanged - key already stored)";
    } else {
      el.keyStatusIndicator.textContent = "No key set";
      el.keyStatusIndicator.className = "key-status badge-unconfigured";
      el.settingApiKey.placeholder = "Enter API key";
    }
  }

  // Model lists load per provider and requests can resolve out of order (a
  // slow Gemini listing once landed AFTER a later Anthropic one, leaving
  // Sonnet under a Google provider - field bug, 2026-09-06). Last request
  // wins: stale responses are dropped, and the dropdown says so while loading.
  let modelsRequestSeq = 0;

  function loadModels(provider, selectedModel) {
    const seq = ++modelsRequestSeq;
    el.settingModel.innerHTML = '<option value="">Loading models\u2026</option>';
    el.settingModel.disabled = true;
    fetch(`/api/models?provider=${encodeURIComponent(provider)}`)
      .then(function (res) {
        if (!res.ok) return [];
        return res.json();
      })
      .then(function (models) {
        if (seq !== modelsRequestSeq) return; // a newer request owns the dropdown
        el.settingModel.innerHTML = "";
        el.settingModel.disabled = false;
        const modelList = (models && models.length > 0)
          ? models
          : (provider === "gemini" ? ["gemini-2.5-pro", "gemini-2.5-flash"] : ["claude-sonnet-5"]);

        modelList.forEach(function (m) {
          const opt = document.createElement("option");
          opt.value = m;
          opt.textContent = m;
          if (m === selectedModel) {
            opt.selected = true;
          }
          el.settingModel.appendChild(opt);
        });
      })
      .catch(function () {
        if (seq !== modelsRequestSeq) return;
        el.settingModel.disabled = false;
        el.settingModel.innerHTML = `<option value="${selectedModel || ''}">${selectedModel || 'default'}</option>`;
      });
  }

  function saveSettings(e) {
    e.preventDefault();
    const provider = el.settingProvider.value;
    const model = el.settingModel.value;
    const apiKey = el.settingApiKey.value.trim();

    const payload = {
      provider: provider,
      model: model,
      // Always sent: the field shows the stored value, so an emptied input is
      // an explicit clear and a bogus path comes back as a 400 with the reason.
      cs2_install_path: el.settingCs2Path.value.trim(),
    };
    if (apiKey) {
      payload.api_key = apiKey;
    }

    el.saveSettingsBtn.disabled = true;
    el.settingsFeedback.classList.add("hidden");

    fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then(function (res) {
        if (!res.ok) {
          return res.json().then(function (err) {
            throw new Error(err.detail || "Failed to update settings");
          });
        }
        return res.json();
      })
      .then(function (updated) {
        state.settings = updated;
        updateKeyIndicator(updated.provider);
        el.settingCs2Path.value = updated.cs2_install_path || "";
        updateCs2Status(updated);
        // A saved key/path may have unlocked setup work: reflect it now.
        pollReadiness();
        el.settingApiKey.value = "";
        el.settingsFeedback.className = "settings-feedback success";
        el.settingsFeedback.textContent = "Settings saved successfully!";
        el.settingsFeedback.classList.remove("hidden");
        setTimeout(function () {
          closeSettings();
        }, 800);
      })
      .catch(function (err) {
        el.settingsFeedback.className = "settings-feedback error";
        el.settingsFeedback.textContent = err.message;
        el.settingsFeedback.classList.remove("hidden");
      })
      .finally(function () {
        el.saveSettingsBtn.disabled = false;
      });
  }

  // =========================================================================
  // 6. Event Wiring & Startup
  // =========================================================================

  function initEvents() {
    initDropZone();

    // Chat events
    el.chatForm.addEventListener("submit", function (e) {
      e.preventDefault();
      sendMessage();
    });

    el.chatInput.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    el.deleteChatBtn.addEventListener("click", deleteCurrentChat);

    // Settings Modal
    el.settingsBtn.addEventListener("click", openSettings);
    el.cs2BannerBtn.addEventListener("click", openSettings);
    el.closeSettingsBtn.addEventListener("click", closeSettings);
    el.cancelSettingsBtn.addEventListener("click", closeSettings);
    el.settingsForm.addEventListener("submit", saveSettings);

    el.settingProvider.addEventListener("change", function () {
      const p = el.settingProvider.value;
      updateKeyIndicator(p);
      loadModels(p, null);
    });

    el.settingsModal.addEventListener("click", function (e) {
      if (e.target === el.settingsModal) {
        closeSettings();
      }
    });

    window.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && !el.settingsModal.classList.contains("hidden")) {
        closeSettings();
      }
    });
  }

  // Before anything else can happen, the user learns the CS2 install folder
  // is still required: the warning banner comes up and Settings opens itself
  // on the field that fixes it.
  function checkCs2Gate() {
    fetch("/api/settings")
      .then(function (res) {
        if (!res.ok) throw new Error("Failed to load settings");
        return res.json();
      })
      .then(function (cfg) {
        state.settings = cfg;
        updateCs2Status(cfg);
        if (!cfg.cs2_path_valid) {
          openSettings();
          el.settingCs2Path.focus();
        }
      })
      .catch(function (err) {
        console.error("CS2 gate check failed:", err);
      });
  }

  // On page load
  document.addEventListener("DOMContentLoaded", function () {
    initEvents();
    loadTeams();
    checkCs2Gate();
    pollReadiness();
  });
})();
