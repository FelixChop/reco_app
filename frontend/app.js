const API_BASE = "http://localhost:8000";

// -----------------------------------------------------------------------------
// State
// -----------------------------------------------------------------------------
let sessionUserId = null;
let availableModes = [];
let currentMode = "colors";
let sampledItems = [];
let currentIndex = 0;
let currentItem = null;
let currentRating = 0;
let hoverRating = 0;
let ratingsGivenCount = 0;
let predictionsData = [];
let isModeMenuOpen = false;

// -----------------------------------------------------------------------------
// DOM elements
// -----------------------------------------------------------------------------
const ratingSection = document.getElementById("rating-section");
const predictionsSection = document.getElementById("predictions-section");
const diagnosticsSection = document.getElementById("diagnostics-section");

const colorSwatch = document.getElementById("color-swatch");
const colorName = document.getElementById("color-name");
const starContainer = document.getElementById("star-container");
const ratingHint = document.getElementById("ratingHint");
const skipButton = document.getElementById("skipButton");
const getRecsButton = document.getElementById("getRecsButton");

const modeSelector = document.getElementById("modeSelector");
const modeToggleButton = document.getElementById("modeToggle");
const modeTabs = document.getElementById("modeTabs");
const modeTabsWrapper = document.getElementById("modeTabsWrapper");
const modeSelect = document.getElementById("modeSelect");
const sortSelect = document.getElementById("sort-order");
const predictionsList = document.getElementById("predictions-list");
const toDiagnosticsBtn = document.getElementById("to-diagnostics");
const restartBtn = document.getElementById("restart");

const leaderboardBody = document.getElementById("leaderboard-body");
const bestModelSummary = document.getElementById("best-model-summary");
const confusionBody = document.getElementById("confusion-body");

const trainingOverlay = document.getElementById("trainingOverlay");
const trainingStats = document.getElementById("trainingStats");
const algoList = document.getElementById("algoList");
const recsGrid = document.getElementById("recsGrid");
const refreshRecsButton = document.getElementById("refreshRecsButton");
const showModelsButton = document.getElementById("showModelsButton");
const modelsPanel = document.getElementById("modelsPanel");
const modelsTableBody = document.querySelector("#modelsTable tbody");

// -----------------------------------------------------------------------------
// Utils
// -----------------------------------------------------------------------------
function showSection(section) {
    [ratingSection, predictionsSection, diagnosticsSection].forEach((sec) => {
        if (sec === section) {
            sec.classList.add("active");
            sec.classList.remove("hidden");
        } else {
            sec.classList.remove("active");
            sec.classList.add("hidden");
        }
    });
}

function getStoredUserId() {
    return localStorage.getItem("color_recommender_user_id");
}

function setStoredUserId(id) {
    localStorage.setItem("color_recommender_user_id", id);
}

function getImageUrl(item) {
    if (currentMode === "colors") return null;
    if (item.image_url) return item.image_url;
    const kw = item.image_keyword || item.name;
    return "https://source.unsplash.com/featured/?" + encodeURIComponent(kw);
}

function setButtonLoading(button, isLoading) {
    button.disabled = isLoading;
    if (isLoading) {
        button.dataset.originalText = button.textContent;
        button.textContent = "Chargement…";
    } else if (button.dataset.originalText) {
        button.textContent = button.dataset.originalText;
        delete button.dataset.originalText;
    }
}

function focusActiveModeTab() {
    if (!modeTabs) return;
    const active = modeTabs.querySelector(".mode-tab-active") || modeTabs.querySelector(".mode-tab");
    if (active) {
        active.focus();
    }
}

function setModeMenuState(open) {
    if (!modeSelector || !modeToggleButton) return;
    isModeMenuOpen = open;
    modeSelector.classList.toggle("menu-open", open);
    modeToggleButton.setAttribute("aria-expanded", String(open));
    if (open) {
        requestAnimationFrame(focusActiveModeTab);
    }
}

function toggleModeMenu(forcedState) {
    const nextState = typeof forcedState === "boolean" ? forcedState : !isModeMenuOpen;
    setModeMenuState(nextState);
}

function closeModeMenu() {
    setModeMenuState(false);
}

function handleModeTabKeydown(event) {
    if (!modeTabs) return;
    const keys = ["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "Home", "End"];
    if (!keys.includes(event.key)) return;

    const tabs = Array.from(modeTabs.querySelectorAll(".mode-tab"));
    const currentIndex = tabs.indexOf(event.currentTarget);
    if (currentIndex === -1) return;

    event.preventDefault();

    if (event.key === "Home") {
        tabs[0]?.focus();
        return;
    }
    if (event.key === "End") {
        tabs[tabs.length - 1]?.focus();
        return;
    }

    const delta = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : -1;
    const nextIndex = (currentIndex + delta + tabs.length) % tabs.length;
    tabs[nextIndex]?.focus();
}

// -----------------------------------------------------------------------------
// API calls
// -----------------------------------------------------------------------------
async function initSession() {
    const existing = getStoredUserId();
    if (existing) {
        sessionUserId = existing;
        return;
    }
    const res = await fetch(`${API_BASE}/new-session`);
    sessionUserId = await res.json();
    setStoredUserId(sessionUserId);
}

async function fetchModes() {
    const res = await fetch(`${API_BASE}/modes`);
    availableModes = await res.json();
    currentMode = availableModes[0]?.id || "colors";
    renderModeSelector();
}

async function fetchSampleItems() {
    const res = await fetch(`${API_BASE}/sample-items?mode=${currentMode}`);
    sampledItems = await res.json();
    currentIndex = 0;
    currentItem = sampledItems[0];
    currentRating = 0;
    hoverRating = 0;
    ratingsGivenCount = 0;
    getRecsButton.disabled = true;
    renderCurrentItem();
    showSection(ratingSection);
}

async function sendRating(itemId, ratingValue) {
    const payload = {
        user_id: sessionUserId,
        mode: currentMode,
        item_id: itemId,
        rating: ratingValue,
    };
    await fetch(`${API_BASE}/rate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
}

async function trainAndPredict() {
    setButtonLoading(getRecsButton, true);
    trainingOverlay.classList.remove("hidden");
    trainingStats.textContent = "Préparation des données…";
    algoList.innerHTML = "";

    const res = await fetch(
        `${API_BASE}/train-and-predict?user_id=${sessionUserId}&mode=${currentMode}`,
        { method: "POST" }
    );
    const data = await res.json();

    predictionsData = data.predictions;
    renderPredictions();
    renderDiagnostics(data);
    renderModelsTable(data.leaderboard);

    trainingOverlay.classList.add("hidden");
    showSection(predictionsSection);
    setButtonLoading(getRecsButton, false);
}

// -----------------------------------------------------------------------------
// Rendering helpers
// -----------------------------------------------------------------------------
function renderModeSelector() {
    if (modeTabs) {
        modeTabs.innerHTML = "";
        availableModes.forEach((mode) => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "mode-tab" + (mode.id === currentMode ? " mode-tab-active" : "");
            btn.textContent = mode.label;
            btn.addEventListener("click", async () => {
                if (mode.id === currentMode) {
                    closeModeMenu();
                    return;
                }
                currentMode = mode.id;
                renderModeSelector();
                await fetchSampleItems();
                closeModeMenu();
            });
            btn.addEventListener("keydown", handleModeTabKeydown);
            modeTabs.appendChild(btn);
        });
    }

    if (modeSelect) {
        modeSelect.innerHTML = "";
        availableModes.forEach((mode) => {
            const option = document.createElement("option");
            option.value = mode.id;
            option.textContent = mode.label;
            modeSelect.appendChild(option);
        });

        modeSelect.value = currentMode;
        modeSelect.onchange = async (event) => {
            const newMode = event.target.value;
            if (newMode === currentMode) return;
            currentMode = newMode;
            renderModeSelector();
            await fetchSampleItems();
        };
    }
}

function renderStars() {
    starContainer.innerHTML = "";
    const effectiveRating = hoverRating || currentRating;

    for (let value = 1; value <= 5; value++) {
        const star = document.createElement("button");
        star.type = "button";
        star.className = "star" + (value <= effectiveRating ? " star-active" : "");
        star.dataset.value = value;
        star.textContent = "★";

        star.addEventListener("click", () => handleStarClick(value));
        star.addEventListener("mouseenter", () => {
            hoverRating = value;
            renderStars();
        });
    
        starContainer.appendChild(star);
    }

    starContainer.addEventListener("mouseleave", () => {
        hoverRating = 0;
        renderStars();
    }, { once: true });
}

function renderCurrentItem() {
    currentItem = sampledItems[currentIndex];
    if (!currentItem) return;

    if (currentMode === "colors") {
        colorSwatch.style.backgroundColor = currentItem.hex || "#ffffff";
        colorSwatch.innerHTML = "";
    } else {
        colorSwatch.style.backgroundColor = "#ffffff";
        colorSwatch.innerHTML = "";
        const url = getImageUrl(currentItem);
        if (url) {
            const img = document.createElement("img");
            img.className = "item-image";
            img.src = url;
            img.alt = currentItem.name;
            colorSwatch.appendChild(img);
        } else {
            const label = document.createElement("div");
            label.className = "item-label";
            label.textContent = currentItem.name;
            colorSwatch.appendChild(label);
        }
    }

    let title = currentItem.name;
    if (currentItem.subtitle) {
        title += " · " + currentItem.subtitle;
    }
    colorName.textContent = title;

    renderStars();
}

function renderPredictions() {
    recsGrid.innerHTML = "";
    predictionsList.innerHTML = "";

    predictionsData.forEach((item) => {
        // Grid cards with inline rating
        const card = document.createElement("div");
        card.className = "rec-card";

        const imgUrl = getImageUrl(item);
        if (imgUrl) {
            const img = document.createElement("img");
            img.src = imgUrl;
            img.alt = item.name;
            img.className = "item-image";
            card.appendChild(img);
        } else if (currentMode === "colors") {
            const swatch = document.createElement("div");
            swatch.className = "prediction-thumb";
            swatch.style.backgroundColor = item.hex || "#ffffff";
            card.appendChild(swatch);
        }

        const title = document.createElement("div");
        title.className = "rec-title";
        title.textContent = item.subtitle ? `${item.name} · ${item.subtitle}` : item.name;
        card.appendChild(title);

        const score = document.createElement("div");
        score.className = "prediction-score";
        score.textContent = item.predicted_rating.toFixed(2);
        card.appendChild(score);

        const starsContainer = document.createElement("div");
        starsContainer.className = "rec-stars";
        card.appendChild(starsContainer);
        renderRecStars(starsContainer, item);

        recsGrid.appendChild(card);

        // List view
        const row = document.createElement("div");
        row.className = "prediction-item";

        const thumb = document.createElement(currentMode === "colors" ? "div" : "img");
        thumb.className = "prediction-thumb";
        if (currentMode === "colors") {
            thumb.style.backgroundColor = item.hex || "#ffffff";
        } else {
            thumb.src = imgUrl;
            thumb.alt = item.name;
        }
        row.appendChild(thumb);

        const nameDiv = document.createElement("div");
        nameDiv.className = "prediction-name";
        nameDiv.textContent = item.subtitle ? `${item.name} · ${item.subtitle}` : item.name;
        row.appendChild(nameDiv);

        const scoreDiv = document.createElement("div");
        scoreDiv.className = "prediction-score";
        scoreDiv.textContent = item.predicted_rating.toFixed(2);
        row.appendChild(scoreDiv);

        predictionsList.appendChild(row);
    });
}

function renderRecStars(container, item) {
    container.innerHTML = "";
    let hover = 0;
    const effectiveBase = Math.round(item.predicted_rating);

    for (let value = 1; value <= 5; value++) {
        const star = document.createElement("button");
        star.type = "button";
        star.className = "star";
        star.textContent = "★";

        const effective = hover || item.user_rating || effectiveBase;
        if (value <= effective) {
            star.classList.add("star-active");
        }

        star.addEventListener("mouseenter", () => {
            hover = value;
            renderRecStars(container, item);
        });

        star.addEventListener("mouseleave", () => {
            hover = 0;
            renderRecStars(container, item);
        });

        star.addEventListener("click", async () => {
            await sendRating(item.item_id, value);
            item.user_rating = value;
            refreshRecsButton.classList.remove("hidden");
            renderRecStars(container, item);
        });

        container.appendChild(star);
    }
}

function sortPredictions(order) {
    predictionsData.sort((a, b) => {
        if (order === "asc") {
            return a.predicted_rating - b.predicted_rating;
        }
        return b.predicted_rating - a.predicted_rating;
    });
    renderPredictions();
}

function renderDiagnostics(data) {
    leaderboardBody.innerHTML = "";
    data.leaderboard.forEach((entry) => {
        const tr = document.createElement("tr");
        if (entry.model_name === data.best_model_name) {
            tr.classList.add("best-row");
        }
        const tdName = document.createElement("td");
        const tdRmse = document.createElement("td");
        const tdRank = document.createElement("td");
        tdName.textContent = entry.model_name;
        tdRmse.textContent = entry.rmse.toFixed(3);
        tdRank.textContent = entry.rank;
        tr.appendChild(tdName);
        tr.appendChild(tdRmse);
        tr.appendChild(tdRank);
        leaderboardBody.appendChild(tr);
    });

    bestModelSummary.textContent = `Le meilleur modèle est ${data.best_model_name} avec un RMSE de ${data.best_model_rmse.toFixed(3)} pour le mode ${currentMode}.`;

    confusionBody.innerHTML = "";
    data.confusion_matrix.forEach((row, i) => {
        const tr = document.createElement("tr");
        const labelCell = document.createElement("td");
        labelCell.textContent = i + 1;
        tr.appendChild(labelCell);
        row.forEach((val) => {
            const td = document.createElement("td");
            td.textContent = val;
            tr.appendChild(td);
        });
        confusionBody.appendChild(tr);
    });
}

function renderModelsTable(leaderboard) {
    modelsTableBody.innerHTML = "";
    leaderboard.forEach((entry) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${entry.model_name}</td>
            <td>${entry.rmse.toFixed(3)}</td>
            <td>${entry.rmse.toFixed(3)}</td>
            <td>${entry.rank}</td>
        `;
        modelsTableBody.appendChild(tr);
    });
}

// -----------------------------------------------------------------------------
// Navigation helpers
// -----------------------------------------------------------------------------
function goToNextItem() {
    if (sampledItems.length === 0) return;
    currentIndex = (currentIndex + 1) % sampledItems.length;
    currentRating = 0;
    hoverRating = 0;
    renderCurrentItem();
}

function handleStarClick(value) {
    currentRating = value;
    hoverRating = 0;
    renderStars();
    submitCurrentRating();
}

async function submitCurrentRating() {
    if (!currentRating || !currentItem) return;
    await sendRating(currentItem.id, currentRating);
    ratingsGivenCount += 1;
    ratingHint.textContent = `Notes données : ${ratingsGivenCount} / 3`;
    currentRating = 0;
    hoverRating = 0;
    renderStars();

    if (ratingsGivenCount >= 3) {
        getRecsButton.disabled = false;
    }

    goToNextItem();
}

// -----------------------------------------------------------------------------
// Event bindings
// -----------------------------------------------------------------------------
if (modeToggleButton) {
    setModeMenuState(false);
    modeToggleButton.addEventListener("click", () => toggleModeMenu());
}

document.addEventListener("click", (event) => {
    if (!isModeMenuOpen || !modeSelector) return;
    if (!modeSelector.contains(event.target)) {
        closeModeMenu();
    }
});

document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && isModeMenuOpen) {
        closeModeMenu();
        modeToggleButton?.focus();
    }
});

sortSelect.addEventListener("change", (e) => {
    sortPredictions(e.target.value);
});

toDiagnosticsBtn.addEventListener("click", () => {
    showSection(diagnosticsSection);
});

restartBtn.addEventListener("click", async () => {
    await fetchSampleItems();
});

skipButton.addEventListener("click", () => {
    currentRating = 0;
    hoverRating = 0;
    renderStars();
    goToNextItem();
});

getRecsButton.addEventListener("click", async () => {
    await trainAndPredict();
});

refreshRecsButton.addEventListener("click", async () => {
    await trainAndPredict();
    refreshRecsButton.classList.add("hidden");
});

showModelsButton.addEventListener("click", () => {
    modelsPanel.classList.toggle("hidden");
});

// -----------------------------------------------------------------------------
// Init
// -----------------------------------------------------------------------------
(async function initApp() {
    try {
        await initSession();
        await fetchModes();
        await fetchSampleItems();
        showSection(ratingSection);
    } catch (e) {
        console.error("Init failed", e);
        alert("Erreur lors du chargement de l'app : " + e.message);
    }
})();
