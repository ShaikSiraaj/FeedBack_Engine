"use strict";

const translationSource = document.querySelector("#translation-source");
const translationTarget = document.querySelector("#translation-target");
const translationStatus = document.querySelector("#translation-status");
const translationButtons = document.querySelectorAll(".translate-button");
const translationPairs = [];
let translationLanguages = [];
const uploadInput = document.querySelector("#feedback-upload");
const uploadPreviewButton = document.querySelector("#preview-upload");
const uploadProcessButton = document.querySelector("#process-upload");

function showTranslationStatus(message, kind = "") {
    if (!translationStatus) return;
    translationStatus.textContent = message;
    translationStatus.className = `translation-status ${kind}`.trim();
}

function updateTranslationTargets(preferredTarget = "") {
    if (!translationSource || !translationTarget) return;
    const source = translationSource.value;
    const supportedTargets = translationPairs
        .filter((pair) => pair.source === source)
        .map((pair) => pair.target);

    translationTarget.replaceChildren();
    for (const language of translationLanguages) {
        if (supportedTargets.includes(language.code)) {
            const option = document.createElement("option");
            option.value = language.code;
            option.textContent = language.name;
            translationTarget.append(option);
        }
    }

    if (supportedTargets.includes(preferredTarget)) {
        translationTarget.value = preferredTarget;
    }
}

async function loadTranslationLanguages() {
    if (!translationSource || !translationTarget) return;
    for (const button of translationButtons) button.disabled = true;
    try {
        const response = await fetch("/api/translation/languages");
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Could not load local languages.");
        if (!payload.languages.length || !payload.pairs.length) {
            throw new Error("No local translation models are installed. Install an Argos language package to enable translation.");
        }

        translationPairs.push(...payload.pairs);
        translationLanguages = payload.languages;
        const sources = [...new Set(payload.pairs.map((pair) => pair.source))];
        translationSource.replaceChildren();
        for (const code of sources) {
            const language = payload.languages.find((item) => item.code === code);
            if (!language) continue;
            const option = document.createElement("option");
            option.value = language.code;
            option.textContent = language.name;
            translationSource.append(option);
        }
        translationSource.disabled = false;
        translationTarget.disabled = false;
        updateTranslationTargets();
        translationSource.addEventListener("change", () => updateTranslationTargets());
        showTranslationStatus(
            `${payload.engine} · ${payload.languages.length} offline languages available. ` +
            "Choose a supported source and target above.",
        );
        for (const button of translationButtons) button.disabled = false;
    } catch (error) {
        showTranslationStatus(error.message, "error");
        translationSource.replaceChildren(new Option("Unavailable", ""));
        translationTarget.replaceChildren(new Option("Unavailable", ""));
    }
}

document.addEventListener("click", async (event) => {
    const button = event.target.closest(".translate-button");
    if (!button) return;

    const row = button.closest("[data-feedback-text]");
    const result = row.querySelector(".translation-result");
    const text = row.dataset.feedbackText;
    if (!text.trim()) {
        result.textContent = "There is no feedback text to translate.";
        return;
    }
    if (!translationSource.value || !translationTarget.value) {
        result.textContent = "Local translation models are still loading or unavailable. Check the language status above.";
        return;
    }

    button.disabled = true;
    result.textContent = "Translating locally…";
    try {
        const response = await fetch("/api/translate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text,
                source_language: translationSource.value,
                target_language: translationTarget.value,
            }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Translation failed.");
        result.textContent = payload.translation;
    } catch (error) {
        result.textContent = `Translation unavailable: ${error.message}`;
    } finally {
        button.disabled = false;
    }
});

loadTranslationLanguages();

if (uploadInput) {
    const fileName = document.querySelector("#upload-file-name");
    const previewPanel = document.querySelector("#upload-preview");
    const uploadStatus = document.querySelector("#upload-status");
    const previewRows = document.querySelector("#upload-preview-rows");
    const previewSummary = document.querySelector(".upload-preview-summary");
    const dropzone = document.querySelector("#upload-dropzone");
    let previewIsCurrent = false;

    function setUploadStatus(message, kind = "") {
        uploadStatus.textContent = message;
        uploadStatus.className = kind;
    }

    function selectedFile() {
        return uploadInput.files && uploadInput.files[0];
    }

    function refreshSelectedFile() {
        const file = selectedFile();
        fileName.textContent = file ? file.name : "No file selected";
        uploadPreviewButton.disabled = !file;
        previewIsCurrent = false;
        uploadProcessButton.disabled = true;
        previewPanel.hidden = true;
        setUploadStatus("");
    }

    uploadInput.addEventListener("change", refreshSelectedFile);

    for (const eventName of ["dragenter", "dragover"]) {
        dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropzone.classList.add("dragover");
        });
    }
    for (const eventName of ["dragleave", "drop"]) {
        dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropzone.classList.remove("dragover");
        });
    }
    dropzone.addEventListener("drop", (event) => {
        if (event.dataTransfer.files.length) {
            uploadInput.files = event.dataTransfer.files;
            refreshSelectedFile();
        }
    });

    uploadPreviewButton.addEventListener("click", async () => {
        const file = selectedFile();
        if (!file) return;
        if (file.size > 16 * 1024 * 1024) {
            setUploadStatus("Choose a CSV smaller than 16 MB.", "error");
            return;
        }

        uploadPreviewButton.disabled = true;
        setUploadStatus("Validating the CSV and checking duplicate IDs…");
        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/api/data/preview", {
                method: "POST",
                body: formData,
            });
            const payload = await response.json();
            if (!response.ok) throw new Error(payload.error || "CSV validation failed.");

            previewRows.replaceChildren();
            for (const record of payload.sample) {
                const row = document.createElement("tr");
                for (const value of [
                    record.feedback_id,
                    record.created_at,
                    record.store_id,
                    record.channel,
                    record.text,
                ]) {
                    const cell = document.createElement("td");
                    cell.textContent = value || "—";
                    row.append(cell);
                }
                previewRows.append(row);
            }
            previewSummary.textContent =
                `${payload.filename_records.toLocaleString()} rows found · ` +
                `${payload.records_to_add.toLocaleString()} new · ` +
                `${payload.duplicates_skipped.toLocaleString()} duplicate IDs will be skipped. ` +
                "Previewing the first five rows.";
            previewPanel.hidden = false;
            previewIsCurrent = true;
            uploadProcessButton.disabled = payload.records_to_add === 0;
            setUploadStatus(
                payload.records_to_add
                    ? "Validation passed. Review the preview before importing."
                    : "Every feedback ID is already in the active data.",
                payload.records_to_add ? "success" : "",
            );
        } catch (error) {
            previewIsCurrent = false;
            setUploadStatus(error.message, "error");
        } finally {
            uploadPreviewButton.disabled = false;
        }
    });

    uploadProcessButton.addEventListener("click", async () => {
        const file = selectedFile();
        if (!file || !previewIsCurrent) {
            setUploadStatus("Validate and preview the current file before importing.", "error");
            return;
        }

        uploadProcessButton.disabled = true;
        uploadPreviewButton.disabled = true;
        setUploadStatus("Running the local classifier and appending new records…");
        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/api/data/upload", {
                method: "POST",
                body: formData,
            });
            const payload = await response.json();
            if (!response.ok) throw new Error(payload.error || "The feedback import failed.");
            previewIsCurrent = false;
            setUploadStatus(
                `${payload.message} Added ${payload.records_added.toLocaleString()}; ` +
                `skipped ${payload.duplicates_skipped.toLocaleString()} duplicates.`,
                "success",
            );
            window.setTimeout(() => window.location.reload(), 1600);
        } catch (error) {
            uploadProcessButton.disabled = false;
            uploadPreviewButton.disabled = false;
            setUploadStatus(error.message, "error");
        }
    });
}

for (const select of document.querySelectorAll(".issue-status-select")) {
    select.addEventListener("change", async () => {
        const card = select.closest(".managed-issue-card");
        const message = card.querySelector(".issue-status-message");
        const badge = card.querySelector(".issue-status-badge");
        select.disabled = true;
        message.className = "issue-status-message";
        message.textContent = "Saving status…";
        try {
            const response = await fetch("/api/emerging/status", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    issue_key: card.dataset.issueKey,
                    status: select.value,
                    week: Number(new URLSearchParams(window.location.search).get("week")) ||
                        Number(card.querySelector(".issue-source").textContent.match(/Week (\d+)/)?.[1]),
                    region: document.querySelector("#workspace-region").value,
                }),
            });
            const payload = await response.json();
            if (!response.ok) throw new Error(payload.error || "Status could not be saved.");
            badge.textContent = payload.status[0].toUpperCase() + payload.status.slice(1);
            badge.className = `issue-status-badge ${payload.status}`;
            card.dataset.currentStatus = payload.status;
            message.textContent = "Status saved.";
            const selectedFilter = new URLSearchParams(window.location.search).get("status") || "all";
            if (selectedFilter !== "all" && selectedFilter !== payload.status) {
                card.hidden = true;
            }
        } catch (error) {
            select.value = card.dataset.currentStatus;
            message.className = "issue-status-message error";
            message.textContent = error.message;
        } finally {
            select.disabled = false;
        }
    });
}

const translationModelStatus = document.querySelector("#translation-model-status");
if (translationModelStatus) {
    fetch("/api/translation/languages")
        .then(async (response) => {
            const payload = await response.json();
            if (!response.ok) throw new Error(payload.error || "Local models are unavailable.");
            const pairs = payload.pairs.map((pair) => `${pair.source} → ${pair.target}`);
            translationModelStatus.textContent = pairs.length
                ? `${payload.engine} · Installed pairs: ${pairs.join(", ")}`
                : "No local translation pairs are installed.";
        })
        .catch((error) => {
            translationModelStatus.textContent = error.message;
            translationModelStatus.classList.add("error");
        });
}