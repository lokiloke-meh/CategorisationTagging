// Client-side interactions for Report Filter Form

document.addEventListener('DOMContentLoaded', () => {
    const topicCheckboxes = document.querySelectorAll('.topic-checkbox');
    const tagsContainer = document.getElementById('tags-container');

    const btnSelectAllTopics = document.getElementById('btn-select-all-topics');
    const btnClearTopics = document.getElementById('btn-clear-topics');
    const btnSelectAllTags = document.getElementById('btn-select-all-tags');
    const btnClearTags = document.getElementById('btn-clear-tags');

    // Topic Selection helpers
    if (btnSelectAllTopics) {
        btnSelectAllTopics.addEventListener('click', () => {
            topicCheckboxes.forEach(cb => cb.checked = true);
            refreshTags();
        });
    }

    if (btnClearTopics) {
        btnClearTopics.addEventListener('click', () => {
            topicCheckboxes.forEach(cb => cb.checked = false);
            refreshTags();
        });
    }

    // Tag Selection helpers
    if (btnSelectAllTags) {
        btnSelectAllTags.addEventListener('click', () => {
            const tagCheckboxes = document.querySelectorAll('.tag-checkbox');
            tagCheckboxes.forEach(cb => cb.checked = true);
        });
    }

    if (btnClearTags) {
        btnClearTags.addEventListener('click', () => {
            const tagCheckboxes = document.querySelectorAll('.tag-checkbox');
            tagCheckboxes.forEach(cb => cb.checked = false);
        });
    }

    // Refresh tags when topic selection changes
    topicCheckboxes.forEach(cb => {
        cb.addEventListener('change', () => {
            refreshTags();
        });
    });

    function getSelectedTopics() {
        const selected = [];
        document.querySelectorAll('.topic-checkbox:checked').forEach(cb => {
            selected.push(cb.value);
        });
        return selected;
    }

    function getSelectedTags() {
        const selected = new Set();
        document.querySelectorAll('.tag-checkbox:checked').forEach(cb => {
            selected.add(cb.value);
        });
        return selected;
    }

    function refreshTags() {
        if (!tagsContainer) return;

        const selectedTopics = getSelectedTopics();
        if (selectedTopics.length === 0) {
            tagsContainer.innerHTML = '<p class="text-muted small text-center my-3">Select at least one topic to see available tags.</p>';
            return;
        }

        const currentSelectedTags = getSelectedTags();
        const params = new URLSearchParams();
        selectedTopics.forEach(t => params.append('topics', t));

        tagsContainer.innerHTML = '<div class="text-center py-3 text-muted small"><span class="spinner-border spinner-border-sm me-2"></span>Loading tags...</div>';

        fetch(`/api/tags?${params.toString()}`)
            .then(res => res.json())
            .then(data => {
                const tags = data.tags || [];
                if (tags.length === 0) {
                    tagsContainer.innerHTML = '<p class="text-muted small text-center my-3">No tags found for the selected topic(s).</p>';
                    return;
                }

                let html = '';
                tags.forEach((tag, idx) => {
                    // Check if tag was previously checked or default to checked
                    const isChecked = currentSelectedTags.size === 0 || currentSelectedTags.has(tag);
                    html += `
                        <div class="form-check py-1">
                            <input class="form-check-input tag-checkbox" type="checkbox" name="tags" value="${escapeHtml(tag)}" id="tag-${idx + 1}" ${isChecked ? 'checked' : ''}>
                            <label class="form-check-label text-truncate small" for="tag-${idx + 1}" title="${escapeHtml(tag)}">
                                ${escapeHtml(tag)}
                            </label>
                        </div>
                    `;
                });
                tagsContainer.innerHTML = html;
            })
            .catch(err => {
                tagsContainer.innerHTML = `<p class="text-danger small text-center my-3">Failed to load tags: ${escapeHtml(err.message)}</p>`;
            });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Shared resize state
    let isResizing = false;

    // Generic resize handler for all handles
    document.querySelectorAll('.resize-handle').forEach(handle => {
        handle.addEventListener('mousedown', (e) => {
            isResizing = true;
            handle.classList.add('active');
            document.body.style.cursor = 'row-resize';
            document.body.style.userSelect = 'none';
            e.preventDefault();
        });
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;

        const activeHandle = document.querySelector('.resize-handle.active');
        if (!activeHandle) return;

        const pair = activeHandle.dataset.resizePair;
        const sidebar = document.querySelector('.filter-card .card-body');
        if (!sidebar) return;

        const sidebarRect = sidebar.getBoundingClientRect();
        const offset = e.clientY - sidebarRect.top;

        if (pair === 'topics-tags') {
            const topicsPanel = document.querySelector('[data-resize-target="topics"]');
            const tagsPanel = document.querySelector('[data-resize-target="tags"]');
            if (!topicsPanel || !tagsPanel) return;

            const topicsHeight = Math.max(60, Math.min(offset - 10, 350));
            const remaining = sidebarRect.height - topicsHeight - 40; // handle + margins
            const tagsHeight = Math.max(60, Math.min(remaining, 350));

            topicsPanel.style.height = `${topicsHeight}px`;
            if (tagsPanel) tagsPanel.style.height = `${tagsHeight}px`;
        } else if (pair === 'tags-date') {
            const tagsPanel = document.querySelector('[data-resize-target="tags"]');
            if (!tagsPanel) return;

            // Calculate top offset of the tags section
            const topicsPanel = document.querySelector('[data-resize-target="topics"]');
            const topicsHeight = topicsPanel ? topicsPanel.offsetHeight : 180;
            const tagsTopOffset = topicsHeight + 50; // margins + handle

            const tagsHeight = Math.max(60, Math.min(offset - tagsTopOffset, 350));
            tagsPanel.style.height = `${tagsHeight}px`;
        }
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            document.querySelectorAll('.resize-handle.active').forEach(h => h.classList.remove('active'));
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }
    });
});
