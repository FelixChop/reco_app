const API_BASE = "http://localhost:8000";

let sessionUserId = null;
let availableModes = [];
let currentMode = "colors";
let sampledItems = [];
let currentIndex = 0;
let predictionsData = [];
let currentRating = 0;
let hoverRating = 0;


// DOM elements (comme avant)
const ratingSection = document.getElementById("rating-section");
const loadingSection = document.getElementById("loading-section");
const predictionsSection = document.getElementById("predictions-section");
const diagnosticsSection = document.getElementById("diagnostics-section");

const modeSelect = document.getElementById("mode-select");

const currentIndexSpan = document.getElementById("current-index");
const colorSwatch = document.getElementById("color-swatch");
const colorName = document.getElementById("color-name");
const starContainer = document.getElementById("star-container");

const sortSelect = document.getElementById("sort-order");
const predictionsList = document.getElementById("predictions-list");
const toDiagnosticsBtn = document.getElementById("to-diagnostics");

const leaderboardBody = document.getElementById("leaderboard-body");
const bestModelSummary = document.getElementById("best-model-summary");
const confusionBody = document.getElementById("confusion-body");
const restartBtn = document.getElementById("restart");

// ------------------------------------------------------------------
// Utils
// ------------------------------------------------------------------

function showSection(section) {
    [ratingSection, loadingSection, predictionsSection, diagnosticsSection].forEach(
        (sec) => {
            if (sec === section) {
                sec.classList.add("active");
                sec.classList.remove("hidden");
            } else {
                sec.classList.remove("active");
                sec.classList.add("hidden");
            }
        }
    );
}

function getStoredUserId() {
    return localStorage.getItem("color_recommender_user_id");
}

function setStoredUserId(id) {
    localStorage.setItem("color_recommender_user_id", id);
}

function getImageUrl(item) {
    if (currentMode === "colors") return null;

    // 1) On privilégie l’URL fournie par le backend (Wikipedia, TMDb, etc.)
    if (item.image_url) {
        return item.image_url;
    }

    // 2) Sinon on retombe sur un placeholder dynamique (Unsplash)
    const kw = item.image_keyword || item.name;
    return "https://source.unsplash.com/featured/?" + encodeURIComponent(kw);
}


// ------------------------------------------------------------------
// API calls
// ------------------------------------------------------------------

async function initSession() {
    let existing = getStoredUserId();
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
    modeSelect.innerHTML = "";
    availableModes.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = m.label;
        modeSelect.appendChild(opt);
    });
    currentMode = availableModes[0]?.id || "colors";
    modeSelect.value = currentMode;
}

async function fetchSampleItems() {
    const res = await fetch(`${API_BASE}/sample-items?mode=${currentMode}`);
    sampledItems = await res.json();
    currentIndex = 0;
    renderCurrentItem();
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
    showSection(loadingSection);
    await new Promise((resolve) => setTimeout(resolve, 600));
    const res = await fetch(
        `${API_BASE}/train-and-predict?user_id=${sessionUserId}&mode=${currentMode}`,
        { method: "POST" }
    );
    const data = await res.json();
    predictionsData = data.predictions;
    renderPredictions();
    renderDiagnostics(data);
    showSection(predictionsSection);
}

// ------------------------------------------------------------------
// Rendering
// ------------------------------------------------------------------

function renderStars() {
    ratingStarsContainer.innerHTML = "";

    for (let value = 1; value <= 5; value++) {
        const star = document.createElement("button");
        star.type = "button";
        star.className = "star";
        star.dataset.value = value;
        star.innerHTML = "★";

        const effectiveRating = hoverRating || currentRating;
        if (value <= effectiveRating) {
            star.classList.add("star-active");   // jaune
        }

        // clic = on fixe la note
        star.addEventListener("click", () => {
            handleStarClick(value);
        });

        // survol = on met à jour hoverRating
        star.addEventListener("mouseenter", () => {
            hoverRating = value;
            renderStars();
        });

        ratingStarsContainer.appendChild(star);
    }

    // en sortant du conteneur, on enlève le hover
    ratingStarsContainer.addEventListener("mouseleave", () => {
        hoverRating = 0;
        renderStars();
    }, { once: true });
}

function handleStarClick(value) {
    currentRating = value;
    hoverRating = 0;
    renderStars();
    submitCurrentRating();
}


function renderCurrentItem() {
    const item = sampledItems[currentIndex];
    currentIndexSpan.textContent = currentIndex + 1;

    // Affichage différent selon le mode
    if (currentMode === "colors") {
        colorSwatch.style.backgroundColor = item.hex || "#ffffff";
        colorSwatch.innerHTML = "";
    } else {
        colorSwatch.style.backgroundColor = "#ffffff";
        colorSwatch.innerHTML = "";

        const url = getImageUrl(item);

        if (url) {
            const img = document.createElement("img");
            img.className = "item-image";
            img.src = url;
            img.alt = item.name;
            colorSwatch.appendChild(img);
        } else {
            const label = document.createElement("div");
            label.className = "item-label";
            label.textContent = item.name;
            colorSwatch.appendChild(label);
        }
    }

    let title = item.name;
    if (item.subtitle) {
        title += " · " + item.subtitle;
    }
    colorName.textContent = title;

    renderStars();
}

function renderPredictions() {
    predictionsList.innerHTML = "";
    predictionsData.forEach((p) => {
        const item = p; // déjà les champs nécessaires

        const card = document.createElement("div");
        card.className = "prediction-item";

        let thumb;
        if (currentMode === "colors") {
            thumb = document.createElement("div");
            thumb.className = "prediction-thumb";
            thumb.style.backgroundColor = item.hex || "#ffffff";
        } else {
            thumb = document.createElement("img");
            thumb.className = "prediction-thumb";
            thumb.src = getImageUrl(item);
            thumb.alt = item.name;
        }

        const nameDiv = document.createElement("div");
        nameDiv.className = "prediction-name";
        nameDiv.textContent = item.subtitle
            ? `${item.name} · ${item.subtitle}`
            : item.name;

        const scoreDiv = document.createElement("div");
        scoreDiv.className = "prediction-score";
        scoreDiv.textContent = item.predicted_rating.toFixed(2);

        card.appendChild(thumb);
        card.appendChild(nameDiv);
        card.appendChild(scoreDiv);
        predictionsList.appendChild(card);
    });
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

    bestModelSummary.textContent = `Le meilleur modèle est ${data.best_model_name} avec un RMSE de ${data.best_model_rmse.toFixed(
        3
    )} pour le mode ${currentMode}.`;

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

// ------------------------------------------------------------------
// Events
// ------------------------------------------------------------------

sortSelect.addEventListener("change", (e) => {
    sortPredictions(e.target.value);
});

toDiagnosticsBtn.addEventListener("click", () => {
    showSection(diagnosticsSection);
});

restartBtn.addEventListener("click", async () => {
    await fetchSampleItems();
    showSection(ratingSection);
});

modeSelect.addEventListener("change", async (e) => {
    currentMode = e.target.value;
    await fetchSampleItems();
    showSection(ratingSection);
});

// ------------------------------------------------------------------
// Init
// ------------------------------------------------------------------

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

const skipButton = document.getElementById("skipButton");
skipButton.addEventListener("click", handleSkip);

function handleSkip() {
    // on ne compte pas comme une note donnée
    currentRating = 0;
    hoverRating = 0;
    renderStars();
    goToNextItem();      // tu as sûrement déjà une fonction qui passe à l’item suivant
}

const modeTabs = document.getElementById("modeTabs");

async function fetchModes() {
    const res = await fetch(`${API_BASE}/modes`);
    availableModes = await res.json(); // {colors: "Couleurs", ...}
    renderModeTabs();
    // on peut démarrer une session pour currentMode ici
    startNewSession();
}

function renderModeTabs() {
    modeTabs.innerHTML = "";
    Object.entries(availableModes).forEach(([value, label]) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "mode-tab" + (value === currentMode ? " mode-tab-active" : "");
        btn.textContent = label;
        btn.addEventListener("click", () => {
            if (value === currentMode) return;
            currentMode = value;
            renderModeTabs();
            startNewSession(); // nouvelle session API pour ce mode
        });
        modeTabs.appendChild(btn);
    });
}

let ratingsGivenCount = 0;
let hasRecommendations = false;
let hasNewRatingsSinceLastRecommend = false;
async function submitCurrentRating() {
    if (!currentRating || !currentItem) return;

    await sendRating(currentItem.id, currentRating);

    ratingsGivenCount += 1;
    currentRating = 0;
    hoverRating = 0;
    renderStars();

    // activer le bouton après au moins 3 ratings
    if (ratingsGivenCount >= 3) {
        getRecsButton.disabled = false;
    }

    goToNextItem();
}

const getRecsButton = document.getElementById("getRecsButton");
getRecsButton.addEventListener("click", handleGetRecs);

async function handleGetRecs() {
    showTrainingOverlay();  // animation "les modèles tournent"
    const res = await fetch(`${API_BASE}/train-and-predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            session_id: sessionId,
            mode: currentMode
        }),
    });
    const data = await res.json();
    hideTrainingOverlay();

    trainingStats.textContent =
        `${data.total_ratings} votes au total · ${data.total_users} utilisateurs`;

    algoList.innerHTML = "";
    data.models.forEach(m => {
        const li = document.createElement("li");
        li.textContent = `${m.name} – RMSE ${m.rmse.toFixed(2)} (${m.rank}e)`;
        algoList.appendChild(li);
    });

    hideTrainingOverlay();
    displayRecommendations(data); // ton affichage actuel + petites modifs pour les notes
    hasRecommendations = true;
    hasNewRatingsSinceLastRecommend = false;
    refreshRecsButton.classList.add("hidden");
}

const trainingOverlay = document.getElementById("trainingOverlay");
const trainingStats = document.getElementById("trainingStats");
const algoList = document.getElementById("algoList");

function showTrainingOverlay() {
    trainingStats.textContent = "Préparation des données…";
    algoList.innerHTML = "";
    trainingOverlay.classList.remove("hidden");
}

function hideTrainingOverlay() {
    trainingOverlay.classList.add("hidden");
}

const refreshRecsButton = document.getElementById("refreshRecsButton");

function displayRecommendations(data) {
    const items = data.predictions;
    recsGrid.innerHTML = "";

    items.forEach(item => {
        const card = document.createElement("div");
        card.className = "rec-card";

        // image
        const imgUrl = getImageUrl(item);
        if (imgUrl) {
            const img = document.createElement("img");
            img.src = imgUrl;
            img.alt = item.name;
            img.className = "item-image";
            card.appendChild(img);
        }

        const title = document.createElement("div");
        title.className = "rec-title";
        title.textContent = item.subtitle ? `${item.name} · ${item.subtitle}` : item.name;
        card.appendChild(title);

        // bloc étoiles interactif pour cette carte
        const starsContainer = document.createElement("div");
        starsContainer.className = "rec-stars";
        card.appendChild(starsContainer);

        renderRecStars(starsContainer, item);

        recsGrid.appendChild(card);
    });
}

function renderRecStars(container, item) {
    container.innerHTML = "";
    const userRating = item.user_rating || 0; // si backend renvoie déjà une note existante pour cet utilisateur
    let hover = 0;

    for (let value = 1; value <= 5; value++) {
        const star = document.createElement("button");
        star.type = "button";
        star.className = "star";
        star.innerHTML = "★";

        const effective = hover || userRating || Math.round(item.predicted_rating);
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
            hasNewRatingsSinceLastRecommend = true;
            refreshRecsButton.classList.remove("hidden");
            renderRecStars(container, item);
        });

        container.appendChild(star);
    }
}

refreshRecsButton.addEventListener("click", async () => {
    if (!hasNewRatingsSinceLastRecommend) return;
    await handleGetRecs(); // on réutilise la même fonction que pour le premier calcul
    hasNewRatingsSinceLastRecommend = false;
    refreshRecsButton.classList.add("hidden");
});

async function sendRating(itemId, rating) {
    await fetch(`${API_BASE}/rate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            session_id: sessionId,
            mode: currentMode,
            item_id: itemId,
            rating,
        }),
    });
}

const showModelsButton = document.getElementById("showModelsButton");
const modelsPanel = document.getElementById("modelsPanel");
const modelsTableBody = document.querySelector("#modelsTable tbody");

showModelsButton.addEventListener("click", () => {
    modelsPanel.classList.toggle("hidden");
});
