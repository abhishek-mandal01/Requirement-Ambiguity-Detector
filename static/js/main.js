document.addEventListener('DOMContentLoaded', () => {
    
    // --- SRS Analysis Workflow ---
    const srsBtn = document.getElementById('analyze-srs-btn');
    const srsInput = document.getElementById('srs-input');
    const resultsContainer = document.getElementById('results-container');
    const loadingIndicator = document.getElementById('loading-indicator');
    const loadingMessage = document.getElementById('loading-message');
    const loadingProgress = document.getElementById('loading-progress');
    const loadingMessages = [
        'Teaching GPT-6 Astra the secrets of this dataset...',
        "Establishing a secure connection to NASA's computational servers...",
        'Consulting 14,000 GPUs in the AI dimension...',
        'Sending your data to the quantum supercomputer beneath Area 51...',
        'Recalibrating the neural network after accidentally discovering AGI...'
    ];
    
    if (srsBtn) {
        srsBtn.addEventListener('click', async () => {
            const fileInput = document.getElementById('srs-file');
            const file = fileInput && fileInput.files.length > 0 ? fileInput.files[0] : null;
            const srsText = srsInput.value.trim();
            
            if (!file && !srsText) {
                alert('Please upload a file or paste requirement text to analyze.');
                return;
            }
            
            srsBtn.disabled = true;
            loadingIndicator.style.display = 'inline-flex';
            loadingIndicator.setAttribute('aria-hidden', 'false');
            resultsContainer.style.display = 'none';

            let messageIndex = Math.floor(Math.random() * loadingMessages.length);
            let progress = 0;
            const startedAt = performance.now();
            const messageTimer = setInterval(() => {
                messageIndex = (messageIndex + 1) % loadingMessages.length;
                loadingMessage.textContent = loadingMessages[messageIndex];
            }, 2200);
            const progressTimer = setInterval(() => {
                const elapsed = performance.now() - startedAt;
                progress = Math.min(92, Math.round((1 - Math.exp(-elapsed / 9000)) * 92));
                loadingProgress.parentElement.style.setProperty('--progress', `${progress}%`);
            }, 120);
            loadingMessage.textContent = loadingMessages[messageIndex];
            loadingProgress.parentElement.style.setProperty('--progress', '4%');
            
            try {
                let fetchOptions = { method: 'POST' };
                
                if (file) {
                    const formData = new FormData();
                    formData.append('file', file);
                    fetchOptions.body = formData;
                } else {
                    fetchOptions.headers = { 'Content-Type': 'application/json' };
                    fetchOptions.body = JSON.stringify({ srs: srsText });
                }
                
                const response = await fetch('/api/analyze/srs', fetchOptions);
                
                const data = await response.json();
                if (data.error) throw new Error(data.error);
                
                populateDashboard(data);
                resultsContainer.style.display = 'block';
                
            } catch (error) {
                alert('Error analyzing SRS: ' + error.message);
            } finally {
                clearInterval(messageTimer);
                clearInterval(progressTimer);
                loadingProgress.parentElement.style.setProperty('--progress', '100%');
                srsBtn.disabled = false;
                loadingIndicator.style.display = 'none';
                loadingIndicator.setAttribute('aria-hidden', 'true');
            }
        });
    }

    function populateDashboard(data) {
        // 1. KPI Summary
        document.getElementById('sum-total').textContent = data.summary.total_detected;
        document.getElementById('sum-exact').textContent = data.summary.exact_duplicates;
        document.getElementById('sum-near').textContent = data.summary.near_duplicates;
        document.getElementById('sum-ambig').textContent = data.summary.potentially_ambiguous;

        // 2. Requirements Table
        const reqTbody = document.querySelector('#req-table tbody');
        reqTbody.innerHTML = '';
        
        let demoReq = null;
        const preferredDemoId = data.demo_requirement_id;

        data.requirements.forEach(req => {
            const tr = document.createElement('tr');
            
            // Indicators are shown separately from Potentially Ambiguous classification
            const feats = req.features;
            const indicators = (req.ambiguity_indicators || []);
            const indicatorStr = indicators.length ? indicators.join(', ') : 'None detected';
            const featStr = `WC: ${feats.word_count} | VW: ${feats.vague_word_count} | PV: ${feats.passive_voice ? 'Y' : 'N'}`;
            
            const isAmbig = req.prediction === 'Potentially Ambiguous';
            const isMixed = req.prediction === 'Mixed/Non-English';
            let pillClass = 'status-clear';
            if (isAmbig || isMixed) pillClass = 'status-ambiguous';
            const statHtml = req.is_exact_duplicate ? `<span style="color:var(--text-muted)">Skipped</span>` :
                             `<span class="status-pill ${pillClass}" style="padding: 4px 10px; font-size: 0.75rem;">${req.prediction}</span>`;
            const indicatorHtml = req.is_exact_duplicate
                ? `<span style="color:var(--text-muted)">-</span>`
                : `<span title="${req.explanation || ''}" style="font-size: 0.85rem; color: ${indicators.length ? 'var(--text-primary)' : 'var(--text-muted)'};">${indicatorStr}</span>`;

            tr.innerHTML = `
                <td style="font-weight: 600;">${req.id}</td>
                <td class="text-ellipsis" title="${req.original}">${req.original}</td>
                <td class="text-ellipsis" style="color: var(--accent-blue);" title="${req.cleaned || '-'}">${req.cleaned || '-'}</td>
                <td>${req.duplicate_status}</td>
                <td>${indicatorHtml}</td>
                <td>${statHtml}</td>
                <td style="font-size: 0.8rem; color: var(--text-secondary);">${featStr}</td>
            `;
            reqTbody.appendChild(tr);

            if (preferredDemoId && req.id === preferredDemoId) {
                demoReq = req;
            }
        });

        // Fallback: most preprocessing operations / changes
        if (!demoReq) {
            const ranked = data.requirements
                .filter(r => !r.is_exact_duplicate && r.pipeline_steps && r.pipeline_steps.length > 0)
                .sort((a, b) => (b.ops_count || 0) - (a.ops_count || 0));
            demoReq = ranked[0] || null;
        }

        // 3. Pipeline Demo
        if (demoReq) {
            document.getElementById('pl-original').textContent = demoReq.original;
            document.getElementById('pl-final').textContent = demoReq.cleaned;
            
            const ul = document.getElementById('pl-steps');
            ul.innerHTML = '';
            demoReq.pipeline_steps.forEach(step => {
                const li = document.createElement('li');
                li.textContent = step;
                ul.appendChild(li);
            });
        }

        // 4. Similarity Table
        const simTbody = document.querySelector('#sim-table tbody');
        const simMsg = document.getElementById('no-sim-msg');
        simTbody.innerHTML = '';
        
        if (data.similarities.length === 0) {
            simMsg.style.display = 'block';
        } else {
            simMsg.style.display = 'none';
            data.similarities.forEach(sim => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="text-ellipsis" title="${sim.req_a_text}"><strong>${sim.req_a_id}</strong>: ${sim.req_a_text}</td>
                    <td class="text-ellipsis" title="${sim.req_b_text}"><strong>${sim.req_b_id}</strong>: ${sim.req_b_text}</td>
                    <td style="font-weight: 700; color: var(--accent); font-size: 1.1rem;">${sim.score.toFixed(3)}</td>
                    <td>${sim.classification}</td>
                `;
                simTbody.appendChild(tr);
            });
        }
    }

    // --- Model Evaluation Workflow ---
    const evalBtn = document.getElementById('eval-btn');
    if (evalBtn) {
        evalBtn.addEventListener('click', async () => {
            evalBtn.disabled = true;
            evalBtn.textContent = "Validating...";
            
            try {
                const response = await fetch('/api/model/evaluation');
                const data = await response.json();
                
                if (data.error) throw new Error(data.error);
                
                document.getElementById('eval-results').style.display = 'grid';
                
                document.getElementById('met-acc-a').textContent = (data.mode_a.accuracy * 100).toFixed(1) + "%";
                document.getElementById('met-pre-a').textContent = (data.mode_a.precision * 100).toFixed(1) + "%";
                document.getElementById('met-rec-a').textContent = (data.mode_a.recall * 100).toFixed(1) + "%";
                document.getElementById('met-f1-a').textContent = (data.mode_a.f1_score * 100).toFixed(1) + "%";
                
                document.getElementById('met-acc-b').textContent = (data.mode_b.accuracy * 100).toFixed(1) + "%";
                document.getElementById('met-pre-b').textContent = (data.mode_b.precision * 100).toFixed(1) + "%";
                document.getElementById('met-rec-b').textContent = (data.mode_b.recall * 100).toFixed(1) + "%";
                document.getElementById('met-f1-b').textContent = (data.mode_b.f1_score * 100).toFixed(1) + "%";
                
                document.getElementById('bar-chart-img').src = "data:image/png;base64," + data.charts.bar_chart_base64;
                document.getElementById('cm-chart-img').src = "data:image/png;base64," + data.charts.confusion_matrix_base64;

                const noteEl = document.getElementById('eval-methodology-note');
                if (noteEl && data.methodology) {
                    const limits = (data.methodology.known_limitations || []).map(l => `• ${l}`).join('\n');
                    noteEl.style.display = 'block';
                    noteEl.textContent = (data.methodology.notes || '') + (limits ? '\n\n' + limits : '');
                }
                
            } catch (error) {
                alert("Evaluation error: " + error.message);
            } finally {
                evalBtn.textContent = "Run Model Validation";
                evalBtn.disabled = false;
            }
        });
    }

    // --- Active Link Scrollspy ---
    const sections = document.querySelectorAll('.dashboard-section');
    const navLinks = document.querySelectorAll('.nav-link');

    const observerOptions = {
        root: null,
        rootMargin: '-20% 0px -60% 0px',
        threshold: 0
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === '#' + entry.target.id) {
                        link.classList.add('active');
                    }
                });
            }
        });
    }, observerOptions);

    sections.forEach(section => observer.observe(section));
});
