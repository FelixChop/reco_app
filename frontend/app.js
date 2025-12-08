// If we are on localhost, use port 8000. Otherwise (production), use relative path (same origin)
const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://localhost:8000"
    : "";

// -----------------------------------------------------------------------------
// State
// -----------------------------------------------------------------------------
let sessionUserId = null;
let availableModes = [];
let currentMode = "politicians";
let sampledItems = [];
let currentIndex = 0;
let currentItem = null;
let currentRating = 0;
let hoverRating = 0;
let ratingCountsByMode = {};
let predictionsData = [];
let seenItemIdsByMode = {};
let isModeMenuOpen = false;
let isComparisonMode = false;
let originalPredictionsData = [];
let neighborNavContainer = null;

let viewMode = "grid"; // 'grid' | 'list'
let sortOrder = "desc"; // 'desc' | 'asc'

// -----------------------------------------------------------------------------
// DOM elements
// -----------------------------------------------------------------------------
const ratingSection = document.getElementById("rating-section");
const predictionsSection = document.getElementById("predictions-section");
const diagnosticsModal = document.getElementById("diagnostics-modal");

const colorSwatch = document.getElementById("color-swatch");
const itemName = document.getElementById("itemName");
const itemSubtitle = document.getElementById("itemSubtitle");
const starContainer = document.getElementById("star-container");
const skipButton = document.getElementById("skipButton");
const getRecsButton = document.getElementById("getRecsButton");
const getRecsButtonLabel = document.querySelector("#getRecsButton .button-text");
const getRecsButtonStatus = document.getElementById("ctaStatus");

const modeSelector = document.getElementById("modeSelector"); // Obsolete?
const voterCountEl = document.getElementById("voterCount");

const sortBtn = document.getElementById("sortBtn");
const viewToggleBtn = document.getElementById("viewToggleBtn");
const mlInfoBtn = document.getElementById("mlInfoBtn");
const closeDiagnosticsBtn = document.getElementById("closeDiagnostics");

const leaderboardBody = document.getElementById("leaderboard-body");
const bestModelSummary = document.getElementById("best-model-summary");
const bestModelParams = document.getElementById("best-model-params");
const confusionBody = document.getElementById("confusion-body");

const trainingOverlay = document.getElementById("trainingOverlay");
const trainingSteps = document.getElementById("trainingSteps");
const recsGrid = document.getElementById("recsGrid");
const refreshRecsButton = document.getElementById("refreshRecsButton");

const similarUsersBtn = document.getElementById("similarUsersBtn");

let similarUsersData = [];
let currentNeighborIndex = 0;

const CTA_LABEL = "Calculer mes affinités";
const CTA_LOADING_LABEL = "Calcul des affinités…";

// -----------------------------------------------------------------------------
// Utils
// -----------------------------------------------------------------------------
const predictionControls = document.getElementById("predictionControls");

function showSection(section) {
    [ratingSection, predictionsSection].forEach((sec) => {
        if (sec === section) {
            sec.classList.add("active");
            sec.classList.remove("hidden");
        } else {
            sec.classList.remove("active");
            sec.classList.add("hidden");
        }
    });

    if (predictionControls) {
        if (section === predictionsSection) {
            predictionControls.classList.remove("hidden");
            if (voterCountEl) voterCountEl.classList.add("hidden"); // Hide stats in predictions
        } else {
            predictionControls.classList.add("hidden");
            if (voterCountEl) voterCountEl.classList.remove("hidden"); // Show stats in rating
        }
    }
}

// Removed localStorage persistence for User ID to ensure new session on refresh
function getStoredUserId() {
    return null; // Always return null to force new session
}

function setStoredUserId(id) {
    // We can still store it if we want to persist within the session, 
    // but the requirement is "refresh = new voter". 
    // So we might not even need to store it in localStorage if we don't want it to survive refresh.
    // But keeping it in memory (sessionUserId variable) is enough for the page life.
    // Let's just not write to localStorage for user_id.
}

function getStoredRatingCounts() {
    // We also shouldn't load previous rating counts if we are a new user.
    return {};
}

function loadRatingCountsForUser(userId) {
    return {};
}

function persistRatingCounts() {
    // No persistence needed if we want fresh start
}

function getModeRatingCount(mode = currentMode) {
    return ratingCountsByMode[mode] || 0;
}

function updateGetRecsState() {
    const count = getModeRatingCount();
    const isVisible = count >= 3;

    if (isVisible) {
        getRecsButton.classList.remove("hidden-magic");
        getRecsButton.classList.add("visible-magic");
        getRecsButton.disabled = false;
    } else {
        getRecsButton.classList.add("hidden-magic");
        getRecsButton.classList.remove("visible-magic");
        getRecsButton.disabled = true;
    }

    // Similar Users button state
    if (similarUsersBtn) {
        if (count >= 3) {
            similarUsersBtn.classList.remove("btn-disabled-visual");
            similarUsersBtn.setAttribute("aria-disabled", "false");
        } else {
            similarUsersBtn.classList.add("btn-disabled-visual");
            similarUsersBtn.setAttribute("aria-disabled", "true");
        }
    }
}

function getSeenIdsForMode(mode = currentMode) {
    if (!seenItemIdsByMode[mode]) {
        seenItemIdsByMode[mode] = new Set();
    }
    return seenItemIdsByMode[mode];
}

function resetSeenIdsForMode(mode = currentMode) {
    seenItemIdsByMode[mode] = new Set();
}

function getImageUrl(item) {
    if (currentMode === "colors") return null;
    if (item.image_url) {
        if (item.image_url.startsWith("http")) return item.image_url;
        // If relative, prepend API_BASE (which handles localhost:8000 vs prod)
        // If API_BASE is empty (prod same origin), it just returns the relative path, which is correct.
        return API_BASE + item.image_url;
    }
    // Fallback if no image URL
    return "https://placehold.co/400?text=" + encodeURIComponent(item.name);
}

function setButtonLoading(button, isLoading) {
    if (isLoading) {
        button.setAttribute("aria-busy", "true");
        updateCtaMessaging(CTA_LOADING_LABEL);
    } else {
        button.removeAttribute("aria-busy");
        updateCtaMessaging(CTA_LABEL);
    }
}

function updateCtaMessaging(text) {
    if (getRecsButtonLabel) {
        getRecsButtonLabel.textContent = text;
    }
    if (getRecsButtonStatus) {
        getRecsButtonStatus.textContent = text;
    }
    if (getRecsButton) {
        getRecsButton.setAttribute("aria-label", text);
    }
}

function setCtaDisabled(isDisabled) {
    getRecsButton.disabled = isDisabled;
    getRecsButton.setAttribute("aria-disabled", isDisabled ? "true" : "false");
}

function focusActiveModeTab() {
    if (!modeTabs) return;
    const active = modeTabs.querySelector(".mode-tab-active") || modeTabs.querySelector(".mode-tab");
    if (active) {
        active.focus();
    }
}

// Obsolete mode menu functions removed

// -----------------------------------------------------------------------------
// API calls
// -----------------------------------------------------------------------------
async function initSession() {
    // Always create a new session
    const res = await fetch(`${API_BASE}/new-session`);
    sessionUserId = await res.json();
    ratingCountsByMode = {};
    voteBuffer = []; // Reset buffer on new session
    console.log("New session started:", sessionUserId);
}



async function fetchVoterCount() {
    try {
        const res = await fetch(`${API_BASE}/voter-count?mode=${currentMode}`);
        if (res.ok) {
            const data = await res.json();
            if (voterCountEl) {
                const myVotes = getModeRatingCount(currentMode);
                voterCountEl.textContent = `Nombre votants : ${data.count} · Total items : ${data.total_items} · Mes votes : ${myVotes}`;
            }
        }
    } catch (e) {
        console.warn("Failed to fetch voter count", e);
    }
}

async function fetchRatingCounts() {
    // Since we start fresh, we assume 0 counts from server initially or just track locally.
    // But if we want to be sure, we can ask the server.
    if (!sessionUserId) return;
    try {
        const res = await fetch(`${API_BASE}/rating-counts?user_id=${sessionUserId}`);
        if (res.ok) {
            const serverCounts = await res.json();
            ratingCountsByMode = { ...ratingCountsByMode, ...serverCounts };
        }
    } catch (e) {
        console.warn("Failed to fetch rating counts", e);
    }
}

async function fetchSampleItems() {
    await fetchRatingCounts();
    fetchVoterCount(); // Update header with loaded counts
    const seenIds = Array.from(getSeenIdsForMode());
    const params = new URLSearchParams({ mode: currentMode });
    seenIds.forEach((id) => params.append("exclude_ids", id));

    const res = await fetch(`${API_BASE}/sample-items?${params.toString()}`);
    let items = [];

    if (res.ok) {
        items = await res.json();
    }

    if (items.length === 0 && seenIds.length > 0) {
        resetSeenIdsForMode();
        return fetchSampleItems();
    }

    const seenSet = getSeenIdsForMode();
    sampledItems = items.filter((item) => !seenSet.has(item.id));
    sampledItems.forEach((item) => seenSet.add(item.id));

    if (sampledItems.length === 0) {
        alert("Aucun item disponible pour le moment.");
        return;
    }

    currentIndex = 0;
    currentItem = sampledItems[0];

    // Reset stars for the new batch/item
    currentRating = 0;
    hoverRating = 0;

    updateGetRecsState();
    updateCtaMessaging(CTA_LABEL);
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

async function runTrainingAnimation() {
    const steps = [
        "Chargement des données...",
        "Entraînement kNN...",
        "Entraînement SVD...",
        "Entraînement CoClustering...",
        "Comparaison des modèles...",
        "Finalisation..."
    ];

    trainingSteps.innerHTML = "";

    for (const step of steps) {
        const div = document.createElement("div");
        div.className = "step-item";
        div.innerHTML = `<span>⏳</span> ${step}`;
        trainingSteps.appendChild(div);

        // Trigger reflow
        div.offsetHeight;
        div.classList.add("visible");

        // Simulate delay
        await new Promise(r => setTimeout(r, 600));
        div.innerHTML = `<span>✅</span> ${step}`;
    }
}

async function trainAndPredict() {
    setButtonLoading(getRecsButton, true);
    trainingOverlay.classList.remove("hidden");

    // Start animation and fetch in parallel
    const animationPromise = runTrainingAnimation();

    const fetchPromise = fetch(
        `${API_BASE}/train-and-predict?user_id=${sessionUserId}&mode=${currentMode}`,
        { method: "POST" }
    ).then(res => res.json());

    const [_, data] = await Promise.all([animationPromise, fetchPromise]);

    predictionsData = data.predictions;
    sortPredictions(sortOrder); // Initial sort
    renderDiagnostics(data);

    trainingOverlay.classList.add("hidden");
    showSection(predictionsSection);
    setButtonLoading(getRecsButton, false);

    // Check if similar users are available and disable button if not
    checkSimilarUsersAvailability();
}

async function checkSimilarUsersAvailability() {
    if (!sessionUserId) {
        similarUsersBtn.disabled = true;
        similarUsersBtn.title = "Vous devez d'abord voter pour des items";
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/similar-users`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                user_id: sessionUserId,
                mode: currentMode,
                limit: 1 // Just check if at least one exists
            })
        });
        const data = await res.json();

        if (data.neighbors && data.neighbors.length > 0) {
            similarUsersBtn.disabled = false;
            similarUsersBtn.title = "Voir les utilisateurs similaires";
        } else {
            similarUsersBtn.disabled = true;
            similarUsersBtn.title = "Aucun utilisateur similaire trouvé (besoin de plus de votes en commun)";
        }
    } catch (e) {
        console.error("Error checking similar users:", e);
        similarUsersBtn.disabled = true;
        similarUsersBtn.title = "Erreur lors de la vérification";
    }
}

// -----------------------------------------------------------------------------
// Rendering helpers
// -----------------------------------------------------------------------------
const mobileModeTrigger = document.getElementById("mobileModeTrigger");
const mobileModeLabel = document.getElementById("mobileModeLabel");
const mobileModeOverlay = document.getElementById("mobileModeOverlay");
const closeMobileOverlay = document.getElementById("closeMobileOverlay");
const mobileModeList = document.getElementById("mobileModeList");
const desktopModeTabs = document.getElementById("desktopModeTabs");

// ... (existing constants)

async function fetchModes() {
    // We define hardcoded structure ensuring the order and specific labels
    // We can verify against availableModes from backend if needed, but let's prioritize exact UI requirements
    // Backend returns list of IDs and Labels, we overlay our custom UI on top.

    // Custom Config
    const MODE_CONFIG = [
        { id: "politicians", label: "Politiques", icon: "🏛" },
        { id: "colors", label: "Couleurs", icon: "🎨" },
        { id: "destinations", label: "Vacances", icon: "✈️" },
        { id: "movies", label: "Films", icon: "🎬" },
        { id: "songs", label: "Musiques", icon: "🎵" },
        { id: "dishes", label: "Plats", icon: "🥘" },
        { id: "books", label: "Livres", icon: "📚" }
    ];

    availableModes = MODE_CONFIG;

    // Set initial
    currentMode = availableModes[0].id;
    document.body.className = "mode-" + currentMode;

    renderModeSelector();
    fetchVoterCount();
}

function renderModeSelector() {
    // 1. Desktop Tabs
    if (desktopModeTabs) {
        desktopModeTabs.innerHTML = "";
        availableModes.forEach(mode => {
            const btn = document.createElement("button");
            btn.className = `mode-pill ${mode.id === currentMode ? "active" : ""}`;
            btn.innerHTML = `${mode.icon} ${mode.label}`;
            btn.onclick = () => switchMode(mode.id);
            desktopModeTabs.appendChild(btn);
        });
    }

    // 2. Mobile Trigger & List
    const activeModeObj = availableModes.find(m => m.id === currentMode);
    if (mobileModeLabel && activeModeObj) {
        mobileModeLabel.textContent = `${activeModeObj.icon} ${activeModeObj.label}`;
    }

    if (mobileModeList) {
        mobileModeList.innerHTML = "";
        availableModes.forEach(mode => {
            const card = document.createElement("div");
            card.className = `mobile-mode-card ${mode.id === currentMode ? "active" : ""}`;
            card.innerHTML = `
                <span class="mobile-mode-icon">${mode.icon}</span>
                <span class="mobile-mode-label">${mode.label}</span>
            `;
            card.onclick = () => {
                switchMode(mode.id);
                setMobileOverlay(false);
            };
            mobileModeList.appendChild(card);
        });
    }
}

async function switchMode(newModeId) {
    if (newModeId === currentMode) return;
    currentMode = newModeId;
    document.body.className = "mode-" + currentMode;
    voteBuffer = [];

    // Exit comparison mode when switching themes
    if (isComparisonMode) {
        isComparisonMode = false;
        similarUsersData = [];
        originalPredictionsData = [];
        if (similarUsersBtn) {
            similarUsersBtn.classList.remove("btn-active");
        }
    }

    renderModeSelector();
    fetchVoterCount();
    await fetchSampleItems();
}

function setMobileOverlay(open) {
    if (mobileModeOverlay) {
        if (open) {
            mobileModeOverlay.classList.remove("hidden");
        } else {
            mobileModeOverlay.classList.add("hidden");
        }
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

        // Use mousedown for instant response
        star.addEventListener("mousedown", () => handleStarClick(value));
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

    // 1) Image / Color
    if (currentMode === "colors") {
        colorSwatch.style.backgroundColor = currentItem.hex || "#ffffff";
        colorSwatch.innerHTML = "";
        colorSwatch.classList.remove("hidden");
    } else if (currentMode === "songs") {
        // Pour les musiques, on cache l'image (le player prend la place)
        colorSwatch.classList.add("hidden");
    } else {
        colorSwatch.classList.remove("hidden");
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

    // 2) Textes
    if (currentMode === "songs") {
        itemName.classList.add("hidden");
        itemSubtitle.classList.add("hidden");
    } else {
        itemName.classList.remove("hidden");
        itemSubtitle.classList.remove("hidden");
        itemName.textContent = currentItem.name;
        itemSubtitle.textContent = currentItem.subtitle || "";
    }

    // 3) Spotify Player
    const spotifyPlayer = document.getElementById("spotifyPlayer");
    if (spotifyPlayer) {
        if (currentItem.spotify_id && currentMode === "songs") {
            spotifyPlayer.classList.remove("hidden");
            // Autoplay : allow="autoplay" + src="...?autoplay=1"
            spotifyPlayer.innerHTML = `
                <iframe style="border-radius:12px" 
                    src="https://open.spotify.com/embed/track/${currentItem.spotify_id}?utm_source=generator&autoplay=1" 
                    width="100%" 
                    height="352" 
                    frameBorder="0" 
                    allowfullscreen="" 
                    allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" 
                    loading="eager">
                </iframe>
            `;
        } else {
            spotifyPlayer.classList.add("hidden");
            spotifyPlayer.innerHTML = "";
        }
    }

    renderStars();
}

const BATCH_SIZE = 20;
let renderedCount = 0;
let observer = null;

function setupIntersectionObserver() {
    if (observer) observer.disconnect();

    const options = {
        root: null,
        rootMargin: '100px',
        threshold: 0.1
    };

    observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                renderNextBatch();
            }
        });
    }, options);
}

function renderPredictions() {
    recsGrid.innerHTML = "";
    recsGrid.className = `recs-grid ${viewMode === 'list' ? 'list-view' : ''} ${isComparisonMode ? 'comparison-mode' : ''}`;
    renderedCount = 0;

    if (isComparisonMode) {
        renderNeighborControls();
    }

    setupIntersectionObserver();
    renderNextBatch();
}

function renderNextBatch() {
    const nextBatch = predictionsData.slice(renderedCount, renderedCount + BATCH_SIZE);

    if (nextBatch.length === 0) return;

    // Remove sentinel if it exists
    const sentinel = document.getElementById('scroll-sentinel');
    if (sentinel) sentinel.remove();

    nextBatch.forEach((item) => {
        const card = createPredictionCard(item);
        recsGrid.appendChild(card);
    });

    renderedCount += nextBatch.length;

    // Add sentinel if there are more items
    if (renderedCount < predictionsData.length) {
        const newSentinel = document.createElement('div');
        newSentinel.id = 'scroll-sentinel';
        newSentinel.style.height = '20px';
        newSentinel.style.width = '100%';
        recsGrid.appendChild(newSentinel);
        observer.observe(newSentinel);
    }
}

function createPredictionCard(item) {
    const card = document.createElement("div");
    card.className = "rec-card";

    const imgUrl = getImageUrl(item);
    if (imgUrl) {
        const img = document.createElement("img");
        img.src = imgUrl;
        img.alt = item.name;
        img.className = "item-image";
        // Lazy load images
        img.loading = "lazy";
        card.appendChild(img);
    } else if (currentMode === "colors") {
        const swatch = document.createElement("div");
        swatch.className = "prediction-thumb";
        swatch.style.backgroundColor = item.hex || "#ffffff";
        card.appendChild(swatch);
    }

    const content = document.createElement("div");
    content.className = "rec-content";

    const title = document.createElement("div");
    title.className = "rec-title";
    title.textContent = item.subtitle ? `${item.name} · ${item.subtitle}` : item.name;
    content.appendChild(title);

    const score = document.createElement("div");
    score.className = "prediction-score";
    score.textContent = item.predicted_rating.toFixed(2);
    content.appendChild(score);

    const stats = document.createElement("div");
    stats.className = "prediction-stats";
    stats.style.fontSize = "0.8em";
    stats.style.color = "#666";
    stats.style.marginTop = "4px";

    const avgText = item.average_rating ? `Moy: ${item.average_rating}/5` : "Moy: -";
    const countText = item.vote_count ? `(${item.vote_count} votes)` : "(0 votes)";
    stats.textContent = `${avgText} ${countText}`;
    content.appendChild(stats);

    // --- Star Container for USER (Me) ---
    const userStarsLabel = document.createElement("div");
    userStarsLabel.className = "stars-label";
    userStarsLabel.textContent = isComparisonMode ? "Moi" : "";
    userStarsLabel.style.fontSize = "0.75em";
    userStarsLabel.style.color = "#888";
    userStarsLabel.style.marginBottom = "2px";
    if (isComparisonMode) content.appendChild(userStarsLabel);

    const starsContainer = document.createElement("div");
    starsContainer.className = "rec-stars";
    content.appendChild(starsContainer);
    renderRecStars(starsContainer, item);

    // --- Star Container for NEIGHBOR (Them) ---
    if (isComparisonMode && item.their_rating !== undefined) {
        const theirStarsLabel = document.createElement("div");
        theirStarsLabel.className = "stars-label";
        theirStarsLabel.textContent = "Similaire";
        theirStarsLabel.style.fontSize = "0.75em";
        theirStarsLabel.style.color = "#888";
        theirStarsLabel.style.marginTop = "8px";
        theirStarsLabel.style.marginBottom = "2px";
        content.appendChild(theirStarsLabel);

        const theirStarsContainer = document.createElement("div");
        theirStarsContainer.className = "rec-stars rec-stars-their";
        content.appendChild(theirStarsContainer);
        renderRecStarsStatic(theirStarsContainer, item.their_rating);
    }

    // Visual distinction for rated items
    if (item.user_rating) {
        card.classList.add("rated-item");
        card.style.border = "2px solid #ffd700"; // Gold border for ME
    } else if (item.their_rating) {
        card.classList.add("rated-item-neighbor"); // Blue border for THEM
        // style handled by CSS class or inline
        card.style.border = "2px solid #3b82f6";
    }

    card.appendChild(content);
    return card;
}

function renderRecStarsStatic(container, rating) {
    container.innerHTML = "";
    for (let value = 1; value <= 5; value++) {
        const star = document.createElement("span"); // span so not clickable
        star.className = "star-mini"; // Use mini style or similar
        star.textContent = "★";
        if (value <= rating) {
            star.classList.add("star-active-blue"); // Different color for neighbor
        }
        container.appendChild(star);
    }
}

function renderRecStars(container, item) {
    container.innerHTML = "";
    // We use a closure to keep track of hover state for this specific item
    // But since we re-render the whole grid often, we might need a better way if perf is bad.
    // For now, simple re-render of the star container is fine.

    // We need to store hover state on the item object or a separate map if we want it to persist across full re-renders
    // But here we just re-render the stars container on hover.

    // If user has rated it, use that. Otherwise 0 (empty stars) for prediction list
    // The user wants to see "notes que j'ai mises moi-même"
    const currentVal = item.user_rating || 0;

    for (let value = 1; value <= 5; value++) {
        const star = document.createElement("button");
        star.type = "button";
        star.className = "star";
        star.textContent = "★";

        if (value <= currentVal) {
            star.classList.add("star-active");
        }

        star.addEventListener("click", async () => {
            // Instant update UI
            item.user_rating = value;
            renderRecStars(container, item);

            // Send rating
            await sendRating(item.item_id, value);

            // If in comparison mode, refresh similarity scores
            if (isComparisonMode && similarUsersData && similarUsersData.length > 0) {
                try {
                    // Re-fetch similar users with updated ratings
                    const res = await fetch(`${API_BASE}/similar-users`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            user_id: sessionUserId,
                            mode: currentMode,
                            limit: 5
                        })
                    });
                    const data = await res.json();
                    similarUsersData = data.neighbors;
                    // Update the display with new similarity scores
                    updateComparisonData();
                } catch (e) {
                    console.error("Error refreshing similarity:", e);
                }
            } else {
                // Trigger re-training immediately as requested (normal mode)
                await trainAndPredict();
            }
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
    data.leaderboard.forEach((entry, index) => {
        const tr = document.createElement("tr");
        if (entry.model_name === data.best_model_name) {
            tr.classList.add("best-row");
        }

        // Medal icons for top 3
        const medals = ["🥇", "🥈", "🥉"];
        const medal = index < 3 ? medals[index] : "";

        const tdRank = document.createElement("td");
        tdRank.textContent = entry.rank;

        const tdName = document.createElement("td");
        tdName.innerHTML = medal ? `${medal} <strong>${entry.model_name}</strong>` : entry.model_name;

        const tdRmse = document.createElement("td");
        tdRmse.textContent = entry.rmse.toFixed(3);

        const tdMse = document.createElement("td");
        tdMse.textContent = entry.mse ? entry.mse.toFixed(3) : "N/A";

        const tdMae = document.createElement("td");
        tdMae.textContent = entry.mae ? entry.mae.toFixed(3) : "N/A";

        const tdFcp = document.createElement("td");
        tdFcp.textContent = entry.fcp !== null && entry.fcp !== undefined ? entry.fcp.toFixed(3) : "N/A";

        tr.appendChild(tdRank);
        tr.appendChild(tdName);
        tr.appendChild(tdRmse);
        tr.appendChild(tdMse);
        tr.appendChild(tdMae);
        tr.appendChild(tdFcp);
        leaderboardBody.appendChild(tr);
    });

    // Summary with docs and baseline comparison
    const algoDocs = {
        "SVD": "https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD",
        "SVDpp": "https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp",
        "NMF": "https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF",
        "KNNBasic": "https://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic",
        "KNNWithMeans": "https://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans",
        "KNNBaseline": "https://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline",
        "CoClustering": "https://surprise.readthedocs.io/en/stable/co_clustering.html#surprise.prediction_algorithms.co_clustering.CoClustering",
        "SlopeOne": "https://surprise.readthedocs.io/en/stable/slope_one.html#surprise.prediction_algorithms.slope_one.SlopeOne",
        "BaselineOnly": "https://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly",
        "NormalPredictor": "https://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.random_pred.NormalPredictor"
    };

    const bestAlgoUrl = algoDocs[data.best_model_name] || "https://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html";

    // Find baseline for comparison
    const baselineEntry = data.leaderboard.find(e => e.model_name === "BaselineOnly") || data.leaderboard.find(e => e.model_name === "NormalPredictor");
    let comparisonText = "";
    if (baselineEntry) {
        const diff = baselineEntry.rmse - data.best_model_rmse;
        const percent = (diff / baselineEntry.rmse) * 100;
        comparisonText = `<br>Comparé à la baseline (${baselineEntry.model_name}, RMSE=${baselineEntry.rmse.toFixed(3)}), ce modèle améliore la précision de <strong>${percent.toFixed(1)}%</strong>.`;
    }

    // Get best model metrics
    const bestModel = data.leaderboard[0];
    const metricsText = `
        <br><strong>Métriques :</strong>
        <ul style="margin: 8px 0; padding-left: 20px;">
            <li><strong>RMSE</strong>: ${data.best_model_rmse.toFixed(3)} - Erreur quadratique moyenne</li>
            <li><strong>MSE</strong>: ${bestModel.mse ? bestModel.mse.toFixed(3) : 'N/A'} - Erreur au carré moyenne</li>
            <li><strong>MAE</strong>: ${bestModel.mae ? bestModel.mae.toFixed(3) : 'N/A'} - Erreur absolue moyenne</li>
            <li><strong>FCP</strong>: ${bestModel.fcp !== null && bestModel.fcp !== undefined ? bestModel.fcp.toFixed(3) : 'N/A'} - Fraction de paires concordantes</li>
        </ul>
    `;

    bestModelSummary.innerHTML = `
        🥇 Le meilleur modèle est <strong><a href="${bestAlgoUrl}" target="_blank">${data.best_model_name}</a></strong>.
        <br><small>Plus les métriques sont basses (sauf FCP), plus le modèle est précis.</small>
        ${metricsText}
        ${comparisonText}
    `;

    // Show params as bullet points
    if (data.best_model_params) {
        let html = "<strong>Paramètres :</strong><ul>";
        for (const [key, val] of Object.entries(data.best_model_params)) {
            html += `<li>${key}: ${val}</li>`;
        }
        html += "</ul>";
        bestModelParams.innerHTML = html;
    }

    confusionBody.innerHTML = "";
    data.confusion_matrix.forEach((row, i) => {
        const tr = document.createElement("tr");
        const labelCell = document.createElement("td");
        labelCell.textContent = i + 1;
        labelCell.style.fontWeight = "bold";
        tr.appendChild(labelCell);

        row.forEach((val, j) => {
            const td = document.createElement("td");
            td.textContent = val;

            // Color coding
            // i is actual index (0-4 for ratings 1-5)
            // j is predicted index (0-4 for ratings 1-5)
            const actual = i + 1;
            const predicted = j + 1;
            const diff = Math.abs(actual - predicted);

            // Classes defined in CSS: conf-0 (perfect), conf-1 (good), conf-2 (ok), conf-3 (bad), conf-4 (worst)
            td.className = `conf-${diff}`;

            tr.appendChild(td);
        });
        confusionBody.appendChild(tr);
    });
}

// -----------------------------------------------------------------------------
// Navigation helpers
// -----------------------------------------------------------------------------
async function goToNextItem() {
    if (sampledItems.length === 0) return;

    if (currentIndex < sampledItems.length - 1) {
        currentIndex += 1;
        currentRating = 0;
        hoverRating = 0;
        renderCurrentItem();
        return;
    }

    await fetchSampleItems();
}

function handleStarClick(value) {
    currentRating = value;
    hoverRating = 0;
    renderStars();
    submitCurrentRating();
}

// Buffer for votes to ensure we only save when user has >= 3 votes
let voteBuffer = [];

async function submitCurrentRating() {
    if (!currentRating || !currentItem) return;

    // Add to buffer
    const vote = {
        itemId: currentItem.id,
        rating: currentRating
    };
    voteBuffer.push(vote);

    // Update local count
    const currentCount = (ratingCountsByMode[currentMode] || 0) + 1;
    ratingCountsByMode[currentMode] = currentCount;

    // Logic:
    // If count < 3: just keep in buffer (don't send to backend yet)
    // If count == 3: send the 2 buffered votes + the current one (which is in buffer)
    // If count > 3: send the current one immediately

    if (currentCount === 3) {
        // Flush buffer
        console.log("Threshold reached, flushing buffer...", voteBuffer);
        for (const v of voteBuffer) {
            await sendRating(v.itemId, v.rating);
        }
        voteBuffer = []; // Clear buffer
    } else if (currentCount > 3) {
        // Send immediately
        await sendRating(vote.itemId, vote.rating);
        voteBuffer = []; // Ensure buffer is empty
    } else {
        console.log("Vote buffered. Count:", currentCount);
    }

    persistRatingCounts();
    currentRating = 0;
    hoverRating = 0;
    renderStars();

    updateGetRecsState();

    const myVotes = getModeRatingCount(currentMode);

    // Update header
    if (voterCountEl) {
        // "Nombre votants : X · Total items : Y · Mes votes : Z"
        // Note: fetchVoterCount is async and updates voterCountEl
        // We will do it inside fetchVoterCount to keep it centralized
        fetchVoterCount();
    }

    await goToNextItem();
}

// -----------------------------------------------------------------------------
// API calls
// -----------------------------------------------------------------------------
// ... (existing code)

// We need to override fetchVoterCount to incorporate local "myVotes" state
// Or we just update fetchVoterCount definition earlier in file.
// Let's go to fetchVoterCount definition and update it there.

// -----------------------------------------------------------------------------
// Event bindings
// -----------------------------------------------------------------------------
if (mobileModeTrigger) {
    mobileModeTrigger.addEventListener("click", () => setMobileOverlay(true));
}

if (closeMobileOverlay) {
    closeMobileOverlay.addEventListener("click", () => setMobileOverlay(false));
}

document.addEventListener("click", (event) => {
    if (!isModeMenuOpen || !modeSelector) return;
    if (!modeSelector.contains(event.target)) {
        closeModeMenu();
    }
});

if (sortBtn) {
    sortBtn.addEventListener("click", () => {
        sortOrder = sortOrder === "desc" ? "asc" : "desc";
        // Rotate icon for ascending order
        const img = sortBtn.querySelector("img");
        if (img) {
            img.style.transform = sortOrder === "asc" ? "rotate(180deg)" : "rotate(0deg)";
            img.style.transition = "transform 0.2s ease";
        }
        sortPredictions(sortOrder);
    });
}

if (viewToggleBtn) {
    viewToggleBtn.addEventListener("click", () => {
        viewMode = viewMode === "grid" ? "list" : "grid";
        // Optional: Change opacity or style to indicate state if needed
        // For now just toggle the view
        renderPredictions();
    });
}

if (mlInfoBtn) {
    mlInfoBtn.addEventListener("click", () => {
        diagnosticsModal.classList.remove("hidden");
    });
}

if (closeDiagnosticsBtn) {
    closeDiagnosticsBtn.addEventListener("click", () => {
        diagnosticsModal.classList.add("hidden");
    });
}

skipButton.addEventListener("click", async () => {
    currentRating = 0;
    hoverRating = 0;
    updateGetRecsState();

    // Refresh header stats
    fetchVoterCount();

    await goToNextItem();
});

getRecsButton.addEventListener("click", async () => {
    await trainAndPredict();
});

if (refreshRecsButton) {
    refreshRecsButton.addEventListener("click", async () => {
        await trainAndPredict();
        refreshRecsButton.classList.add("hidden");
    });
}

if (similarUsersBtn) {
    // Toggle Comparison Mode
    similarUsersBtn.addEventListener("click", () => {
        const myVotes = getModeRatingCount(currentMode);
        if (myVotes < 3) {
            alert("Vous devez avoir plus de 3 votes pour découvrir des utilisateurs similaires et explorer ce que les autres ont pensé de vos choix.");
            return;
        }
        toggleComparisonMode();
    });
}

// if (closeSimilarUsersBtn) {
//     closeSimilarUsersBtn.addEventListener("click", () => {
//         similarUsersModal.classList.add("hidden");
//     });
// }


async function toggleComparisonMode() {
    if (isComparisonMode) {
        // Turn OFF
        isComparisonMode = false;
        predictionsData = [...originalPredictionsData];
        similarUsersBtn.classList.remove("btn-active"); // Add active style class

        // Remove nav controls if any
        if (neighborNavContainer) neighborNavContainer.remove();

        renderPredictions();
    } else {
        // Turn ON
        if (!sessionUserId) return;

        // Save current predictions to restore later
        if (originalPredictionsData.length === 0 && predictionsData.length > 0) {
            originalPredictionsData = [...predictionsData];
        }

        setButtonLoading(similarUsersBtn, true);

        try {
            const res = await fetch(`${API_BASE}/similar-users`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    user_id: sessionUserId,
                    mode: currentMode,
                    limit: 5
                })
            });
            const data = await res.json();
            similarUsersData = data.neighbors;
            currentNeighborIndex = 0;

            if (similarUsersData.length === 0) {
                alert("Aucun utilisateur similaire trouvé.");
                setButtonLoading(similarUsersBtn, false);
                return;
            }

            isComparisonMode = true;
            similarUsersBtn.classList.add("btn-active");

            showSection(predictionsSection); // Ensure we are on predictions view
            updateComparisonData();

        } catch (e) {
            console.error(e);
            alert("Erreur lors de la récupération des voisins.");
        } finally {
            setButtonLoading(similarUsersBtn, false);
        }
    }
}

function updateComparisonData() {
    if (!similarUsersData || similarUsersData.length === 0) return;

    const neighbor = similarUsersData[currentNeighborIndex];

    // Combine items: common, user_items (me only), other_items (them only)
    // We map them to the structure of prediction items, adding 'their_rating' and 'user_rating'

    // Helper to normalize
    const normalize = (items, type) => items.map(i => ({
        ...i,
        // Now using real values from backend
        // user_rating might be in i.my_rating (common) or i.rating (me_only)
        user_rating: type === 'common' ? i.my_rating : (type === 'me_only' ? i.rating : 0),
        their_rating: type === 'common' ? i.their_rating : (type === 'they_only' ? i.rating : 0),
        // other fields matched by spread ...i
    }));

    const common = normalize(neighbor.common_items, 'common');
    const meOnly = normalize(neighbor.user_items, 'me_only');
    const theyOnly = normalize(neighbor.other_items, 'they_only');

    predictionsData = [...common, ...meOnly, ...theyOnly];

    // Sort? Maybe common first, then by meaningfulness?
    // Let's sort by presence of double ratings, then my rating?
    // User asked: "afficher les étoiles de l'utilisateur similaire en dessous"
    // implies comparison is key.

    renderPredictions();
}

function renderNeighborControls() {
    // Inject controls at top of recsGrid or distinct container
    // Let's put it as the first element of recsGrid or before it?
    // renderPredictions clears recsGrid, so we can append it first.

    const neighbor = similarUsersData[currentNeighborIndex];
    const similarityPercent = Math.round(neighbor.similarity_score * 100);

    const div = document.createElement("div");
    div.className = "neighbor-controls";
    div.style.gridColumn = "1 / -1";
    div.style.display = "flex";
    div.style.alignItems = "center";
    div.style.justifyContent = "center";
    div.style.background = "#f0f9ff";
    div.style.padding = "12px";
    div.style.borderRadius = "12px";
    div.style.marginBottom = "16px";
    div.style.border = "1px solid #bae6fd";

    div.innerHTML = `
        <div style="display:flex; align-items:center; gap:12px;">
            ${currentNeighborIndex > 0 ? `<button class="btn-secondary btn-sm" onclick="changeNeighborProxy(-1)">←</button>` : ''}
            <div style="text-align:center">
                <strong>Utilisateur similaire #${currentNeighborIndex + 1}</strong>
                <div style="font-size:0.85em; color:#0284c7;">${similarityPercent}% de similarité</div>
            </div>
            ${currentNeighborIndex < similarUsersData.length - 1 ? `<button class="btn-secondary btn-sm" onclick="changeNeighborProxy(1)">→</button>` : ''}
        </div>
    `;

    recsGrid.prepend(div);
}

window.changeNeighborProxy = (delta) => {
    currentNeighborIndex += delta;
    updateComparisonData();
};

window.exitComparisonProxy = () => {
    toggleComparisonMode();
};

// -----------------------------------------------------------------------------
// Init
// -----------------------------------------------------------------------------
// Add lightning icons to title
const titleEl = document.querySelector(".app-header h1");
if (titleEl) {
    titleEl.innerHTML = "⚡ " + titleEl.textContent + " ⚡";
}


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
