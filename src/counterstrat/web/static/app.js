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
    isProcessing: false,
    pollTimer: null,
    settings: null,
  };

  // DOM Elements
  const el = {
    // Top Bar
    settingsBtn: document.getElementById("settings-btn"),
    
    // Ingestion
    dropZone: document.getElementById("drop-zone"),
    fileInput: document.getElementById("file-input"),
    browseBtn: document.getElementById("browse-btn"),
    jobStatus: document.getElementById("job-status"),
    jobStage: document.getElementById("job-stage"),
    jobDetail: document.getElementById("job-detail"),
    jobProgressBar: document.getElementById("job-progress-bar"),

    // Catalog
    teamsList: document.getElementById("teams-list"),
    teamsEmpty: document.getElementById("teams-empty"),
    catalogCount: document.getElementById("catalog-count"),

    // Chat
    targetTitle: document.getElementById("target-title"),
    targetMeta: document.getElementById("target-meta"),
    dossierBtn: document.getElementById("dossier-btn"),
    messagesContainer: document.getElementById("messages-container"),
    chatWelcome: document.getElementById("chat-welcome"),
    chatForm: document.getElementById("chat-form"),
    chatInput: document.getElementById("chat-input"),
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
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        uploadDemoFile(files[0]);
      }
    });

    el.browseBtn.addEventListener("click", function () {
      el.fileInput.click();
    });

    el.fileInput.addEventListener("change", function () {
      if (el.fileInput.files.length > 0) {
        uploadDemoFile(el.fileInput.files[0]);
        el.fileInput.value = "";
      }
    });
  }

  function uploadDemoFile(file) {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }

    el.jobStatus.classList.remove("hidden", "status-error", "status-done");
    el.jobStage.textContent = "uploading";
    el.jobDetail.textContent = `Uploading ${file.name}...`;
    el.jobProgressBar.classList.add("progress-indeterminate");

    const formData = new FormData();
    formData.append("demo", file);

    fetch("/api/demos", {
      method: "POST",
      body: formData,
    })
      .then(function (res) {
        if (!res.ok) {
          return res.json().then(function (err) {
            throw new Error(err.detail || "Failed to upload demo");
          });
        }
        return res.json();
      })
      .then(function (data) {
        el.jobStage.textContent = "queued";
        el.jobDetail.textContent = `Job ID: ${data.job_id}. Ingestion pipeline starting...`;
        pollJobStatus(data.job_id);
      })
      .catch(function (err) {
        el.jobStatus.classList.add("status-error");
        el.jobStage.textContent = "error";
        el.jobDetail.textContent = err.message;
        el.jobProgressBar.classList.remove("progress-indeterminate");
      });
  }

  function pollJobStatus(jobId) {
    state.pollTimer = setInterval(function () {
      fetch(`/api/jobs/${jobId}`)
        .then(function (res) {
          if (!res.ok) {
            throw new Error(`Status check returned ${res.status}`);
          }
          return res.json();
        })
        .then(function (job) {
          el.jobStage.textContent = job.stage;
          if (job.stage === "done") {
            clearInterval(state.pollTimer);
            state.pollTimer = null;
            el.jobStatus.classList.add("status-done");
            el.jobDetail.textContent = `Ingested match ${job.match_id || ""} successfully!`;
            el.jobProgressBar.classList.remove("progress-indeterminate");
            loadTeams();
          } else if (job.stage === "error") {
            clearInterval(state.pollTimer);
            state.pollTimer = null;
            el.jobStatus.classList.add("status-error");
            el.jobDetail.textContent = job.detail || "Ingestion pipeline encountered an error";
            el.jobProgressBar.classList.remove("progress-indeterminate");
          } else {
            el.jobDetail.textContent = job.detail || `Running stage: ${job.stage}...`;
          }
        })
        .catch(function (err) {
          clearInterval(state.pollTimer);
          state.pollTimer = null;
          el.jobStatus.classList.add("status-error");
          el.jobStage.textContent = "error";
          el.jobDetail.textContent = err.message;
          el.jobProgressBar.classList.remove("progress-indeterminate");
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

  function renderTeams(teams) {
    el.teamsList.innerHTML = "";
    if (!teams || teams.length === 0) {
      el.teamsEmpty.classList.remove("hidden");
      el.catalogCount.textContent = "0 targets";
      return;
    }

    el.teamsEmpty.classList.add("hidden");
    let totalPairs = 0;

    teams.forEach(function (team) {
      const teamDisplayName = (team.names && team.names.length > 0)
        ? team.names.join(" / ")
        : team.team_key;

      const maps = (team.maps && team.maps.length > 0) ? team.maps : ["unknown_map"];

      maps.forEach(function (mapName) {
        totalPairs++;
        const mStats = (team.map_stats && team.map_stats[mapName]) || { rounds: team.rounds, demos: team.demos };
        const card = document.createElement("div");
        card.className = "team-card";
        card.dataset.teamKey = team.team_key;
        card.dataset.mapName = mapName;

        card.innerHTML = `
          <div class="team-card-title">
            <span>${escapeHtml(teamDisplayName)}</span>
            <span class="team-card-map">${escapeHtml(mapName)}</span>
          </div>
          <div class="team-card-stats">
            <span class="stat-tag"><strong class="stat-n">n=${mStats.rounds}</strong> rounds</span>
            <span class="stat-tag">${mStats.demos} demo${mStats.demos === 1 ? "" : "s"}</span>
          </div>
        `;

        card.addEventListener("click", function () {
          selectTarget(team, teamDisplayName, mapName, card);
        });

        el.teamsList.appendChild(card);
      });
    });

    el.catalogCount.textContent = `${totalPairs} target${totalPairs === 1 ? "" : "s"}`;
  }

  function selectTarget(team, displayName, mapName, cardEl) {
    // Update active highlight
    document.querySelectorAll(".team-card").forEach(function (c) {
      c.classList.remove("selected");
    });
    if (cardEl) cardEl.classList.add("selected");

    state.currentTeamKey = team.team_key;
    state.currentMapName = mapName;
    state.currentTeamData = team;

    const mStats = (team.map_stats && team.map_stats[mapName]) || { rounds: team.rounds, demos: team.demos };

    // Update Header
    el.targetTitle.textContent = `${displayName} - ${mapName}`;
    el.targetMeta.innerHTML = `Sample size: <strong class="stat-n">n = ${mStats.rounds} rounds</strong> across ${mStats.demos} demo${mStats.demos === 1 ? "" : "s"} (${escapeHtml(team.team_key)})`;

    // Enable Dossier button
    el.dossierBtn.disabled = false;
    el.dossierBtn.title = "Download strategic dossier markdown report";

    // Create session
    createChatSession(team.team_key, mapName, displayName);
  }

  // =========================================================================
  // 3. Chat Sessions & Messaging
  // =========================================================================

  function createChatSession(teamKey, mapName, displayName) {
    el.chatInput.disabled = true;
    el.sendBtn.disabled = true;

    fetch("/api/chat/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ team_key: teamKey, map_name: mapName }),
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
        state.currentSessionId = data.session_id;

        // Reset and show initial welcome in message stream
        el.messagesContainer.innerHTML = "";
        appendAssistantMessage({
          text: `Active session started for **${displayName}** on \`${mapName}\`.\n\nYou can ask about buy-round tendencies, utility setups, opening duels, or cite specific rounds.`,
          tool_trace: [],
          warnings: [],
        });

        el.chatInput.disabled = false;
        el.sendBtn.disabled = false;
        el.chatInput.focus();
      })
      .catch(function (err) {
        alert("Failed to initialize session: " + err.message);
      });
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
  // 4. Dossier Download
  // =========================================================================

  function downloadDossier() {
    if (!state.currentTeamKey || !state.currentMapName) return;

    const isMock = window.location.search.includes("mock=1");
    const url = `/api/reports/${state.currentTeamKey}/${state.currentMapName}${isMock ? "?mock=1" : ""}`;
    el.dossierBtn.disabled = true;
    el.dossierBtn.textContent = "Generating...";

    fetch(url)
      .then(function (res) {
        if (!res.ok) {
          return res.json().then(function (err) {
            throw new Error(err.detail || `Server returned ${res.status}`);
          });
        }
        return res.text();
      })
      .then(function (markdownText) {
        const blob = new Blob([markdownText], { type: "text/markdown;charset=utf-8" });
        const downloadUrl = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = downloadUrl;
        a.download = `${state.currentTeamKey}_${state.currentMapName}_dossier.md`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(downloadUrl);
      })
      .catch(function (err) {
        alert(`Could not download dossier:\n${err.message}`);
      })
      .finally(function () {
        el.dossierBtn.disabled = false;
        el.dossierBtn.textContent = "Download Dossier (.md)";
      });
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
        loadModels(cfg.provider, cfg.model);
      })
      .catch(function (err) {
        console.error("Settings load error:", err);
      });
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

  function loadModels(provider, selectedModel) {
    fetch(`/api/models?provider=${encodeURIComponent(provider)}`)
      .then(function (res) {
        if (!res.ok) return [];
        return res.json();
      })
      .then(function (models) {
        el.settingModel.innerHTML = "";
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
        // Fallback
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

    // Dossier Download
    el.dossierBtn.addEventListener("click", downloadDossier);

    // Settings Modal
    el.settingsBtn.addEventListener("click", openSettings);
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

  // On page load
  document.addEventListener("DOMContentLoaded", function () {
    initEvents();
    loadTeams();
  });
})();
