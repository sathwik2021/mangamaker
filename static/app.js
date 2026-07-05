/* === Global State === */

let currentJobId = null;
let statusCheckInterval = null;
let startTime = null;
let timerInterval = null;

const STEPS = ['extraction', 'layout', 'generation', 'composition'];
const STEP_LABELS = {
    'extraction': 'Extract beats from text...',
    'layout': 'Generate layout...',
    'generation': 'Generate images with Stable Diffusion...',
    'composition': 'Compositing manga page...'
};

/* === DOM Elements === */

function getElements() {
    return {
        // Sections
        inputSection: document.getElementById('inputSection'),
        statusSection: document.getElementById('statusSection'),
        resultsSection: document.getElementById('resultsSection'),
        errorSection: document.getElementById('errorSection'),
        
        // Input
        textInput: document.getElementById('textInput'),
        charCount: document.getElementById('charCount'),
        wordCount: document.getElementById('wordCount'),
        maxBeats: document.getElementById('maxBeats'),
        sdSteps: document.getElementById('sdSteps'),
        guidanceScale: document.getElementById('guidanceScale'),
        generateBtn: document.getElementById('generateBtn'),
        exampleBtn: document.getElementById('exampleBtn'),
        
        // Status
        statusText: document.getElementById('statusText'),
        statusTimer: document.getElementById('statusTimer'),
        statusMessage: document.getElementById('statusMessage'),
        progressFill: document.getElementById('progressFill'),
        progressPercent: document.getElementById('progressPercent'),
        cancelBtn: document.getElementById('cancelBtn'),
        
        // Results
        resultImage: document.getElementById('resultImage'),
        imageDisplay: document.getElementById('imageDisplay'),
        generationTime: document.getElementById('generationTime'),
        beatsCount: document.getElementById('beatsCount'),
        panelsCount: document.getElementById('panelsCount'),
        filesList: document.getElementById('filesList'),
        downloadBtn: document.getElementById('downloadBtn'),
        
        // Error
        errorMessage: document.getElementById('errorMessage'),
        errorCode: document.getElementById('errorCode'),
        errorDetails: document.getElementById('errorDetails')
    };
}

/* === Input Stats === */

document.addEventListener('DOMContentLoaded', () => {
    const el = getElements();
    
    el.textInput.addEventListener('input', updateInputStats);
    el.textInput.addEventListener('keydown', handleTextInputKeydown);

    // Display loading state on connection errors
    checkPipelineHealth();
});

function updateInputStats() {
    const el = getElements();
    const text = el.textInput.value;
    
    // Character count
    el.charCount.textContent = text.length;
    
    // Word count
    const words = text.trim().split(/\s+/).filter(w => w.length > 0).length;
    el.wordCount.textContent = words;
    
    // Enable/disable generate button
    el.generateBtn.disabled = text.length < 50;
}

function handleTextInputKeydown(event) {
    if (event.ctrlKey && event.key === 'Enter') {
        generateManga();
    }
}

/* === Pipeline Health Check === */

async function checkPipelineHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (!data.pipeline_available) {
            showError('Pipeline components not loaded. Check server logs.');
        }
    } catch (error) {
        console.warn('Could not check pipeline health:', error);
    }
}

/* === Generate Manga === */

async function generateManga() {
    const el = getElements();
    
    const textInput = el.textInput.value.trim();
    
    if (textInput.length < 50) {
        alert('Please enter at least 50 characters');
        return;
    }
    
    // Disable button and show status
    el.generateBtn.disabled = true;
    el.exampleBtn.disabled = true;
    
    // Hide input, show status
    el.inputSection.style.display = 'none';
    el.statusSection.style.display = 'block';
    el.resultsSection.style.display = 'none';
    el.errorSection.style.display = 'none';
    
    // Reset status
    startTime = Date.now();
    startStatusTimer();
    updateProgress(0, 'initializing', 'Initializing pipeline...');
    
    try {
        // Send generation request
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: textInput,
                options: {
                    max_beats: el.maxBeats.value,
                    sd_steps: el.sdSteps.value,
                    guidance_scale: el.guidanceScale.value
                }
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to start generation');
        }
        
        const data = await response.json();
        currentJobId = data.job_id;
        
        // Start checking status
        pollJobStatus();
        statusCheckInterval = setInterval(pollJobStatus, 1000);
        
    } catch (error) {
        showError(`Failed to start generation: ${error.message}`);
        el.generateBtn.disabled = false;
        el.exampleBtn.disabled = false;
    }
}

/* === Poll Job Status === */

async function pollJobStatus() {
    if (!currentJobId) return;
    
    try {
        const response = await fetch(`/api/status/${currentJobId}`);
        const data = await response.json();

        if (!response.ok) {
            clearInterval(statusCheckInterval);
            clearInterval(timerInterval);
            showError(data.error || `Server error: ${response.status}`);
            return;
        }
        
        const { status, progress, message } = data;
        
        // Update progress bar and message
        updateProgress(progress, status, message);
        
        // Update step indicators
        updateStepIndicators(status);
        
        // Handle completion
        if (status === 'completed') {
            clearInterval(statusCheckInterval);
            clearInterval(timerInterval);
            await showResults();
        }
        
        // Handle error
        if (status === 'error') {
            clearInterval(statusCheckInterval);
            clearInterval(timerInterval);
            showError(data.error || 'Generation failed');
        }
        
    } catch (error) {
        console.error('Error polling status:', error);
        clearInterval(statusCheckInterval);
        clearInterval(timerInterval);
        showError('Network error while polling status: ' + error.message);
    }
}

/* === Update Progress === */

function updateProgress(percent, status, message) {
    const el = getElements();
    
    el.progressFill.style.width = percent + '%';
    el.progressPercent.textContent = percent + '%';
    el.statusText.textContent = message;
    el.statusMessage.textContent = message;
}

function updateStepIndicators(status) {
    STEPS.forEach(step => {
        const el = document.getElementById(`step-${step}`);
        el.classList.remove('active', 'completed');
        
        if (status === step) {
            el.classList.add('active');
        } else if (STEPS.indexOf(step) < STEPS.indexOf(status)) {
            el.classList.add('completed');
        }
    });
}

/* === Status Timer === */

function startStatusTimer() {
    clearInterval(timerInterval);
    
    timerInterval = setInterval(() => {
        const el = getElements();
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        el.statusTimer.textContent = formatSeconds(elapsed) + 's';
    }, 1000);
}

function formatSeconds(seconds) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return m > 0 ? `${m}:${String(s).padStart(2, '0')}` : s;
}

/* === Show Results === */

async function showResults() {
    const el = getElements();
    
    try {
        // Fetch results
        const response = await fetch(`/api/results/${currentJobId}`);
        const data = await response.json();
        
        if (!data || data.error) {
            throw new Error(data.error || 'Failed to fetch results');
        }
        
        const { results, files } = data;
        
        // Display main image
        if (results.final_page) {
            const imagePath = `/results/${results.final_page}`;
            el.resultImage.src = imagePath;
            el.downloadBtn.href = imagePath;
        }
        
        // Display info
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        el.generationTime.textContent = formatSeconds(elapsed);
        el.beatsCount.textContent = results.beats_count || '-';
        el.panelsCount.textContent = results.panels_count || '-';
        
        // Display generated files
        displayGeneratedFiles(files);
        
        // Show results section
        el.statusSection.style.display = 'none';
        el.resultsSection.style.display = 'block';
        
    } catch (error) {
        showError(`Failed to load results: ${error.message}`);
    }
}

/* === Display Generated Files === */

function displayGeneratedFiles(files) {
    const el = getElements();
    el.filesList.innerHTML = '';
    
    if (!files || files.length === 0) {
        el.filesList.innerHTML = '<p style="color: var(--neutral-500);">No files generated</p>';
        return;
    }
    
    files.forEach(file => {
        const isImage = file.type === 'image';
        const icon = isImage ? '🖼️' : '📄';
        const fileName = file.path.split('/').pop();
        
        const link = document.createElement('a');
        link.className = 'file-item';
        link.href = `/results/${file.path}`;
        link.target = '_blank';
        link.innerHTML = `
            <span class="file-icon">${icon}</span>
            <span class="file-name">${fileName}</span>
        `;
        
        el.filesList.appendChild(link);
    });
}

/* === Error Handling === */

function showError(message) {
    const el = getElements();
    
    clearInterval(statusCheckInterval);
    clearInterval(timerInterval);
    
    el.inputSection.style.display = 'none';
    el.statusSection.style.display = 'none';
    el.resultsSection.style.display = 'none';
    el.errorSection.style.display = 'block';
    
    el.errorMessage.textContent = message;
    
    el.generateBtn.disabled = false;
    el.exampleBtn.disabled = false;
}

/* === Cancel Generation === */

function cancelGeneration() {
    clearInterval(statusCheckInterval);
    clearInterval(timerInterval);
    
    currentJobId = null;
    generateNew();
}

/* === Generate New === */

function generateNew() {
    const el = getElements();
    
    // Reset state
    currentJobId = null;
    clearInterval(statusCheckInterval);
    clearInterval(timerInterval);
    
    // Show input section
    el.inputSection.style.display = 'block';
    el.statusSection.style.display = 'none';
    el.resultsSection.style.display = 'none';
    el.errorSection.style.display = 'none';
    
    // Re-enable buttons
    el.generateBtn.disabled = false;
    el.exampleBtn.disabled = false;
    
    // Focus input
    el.textInput.focus();
    updateInputStats();
}

/* === Load Example === */

function loadExample() {
    const el = getElements();
    
    const exampleText = `The rain fell heavily on the narrow streets of the old town, creating puddles that reflected the flickering neon signs of closed shops. Akira ran with every ounce of strength in his body, his lungs burning, his heart pounding in his chest like a war drum. Behind him, the sound of footsteps echoed against the wet pavement, growing closer with each passing second.

"Stop!" a voice shouted from the darkness, but Akira didn't dare look back. He knew what they wanted, and he would never hand it over. His hand gripped the small data drive in his jacket pocket, the proof they had been searching for all these years.

He turned sharply at the corner, nearly slipping on the wet surface. His boots found purchase just in time, and he pushed himself forward with renewed determination. The headquarters building was just three blocks away. If he could make it there, if he could expose what they had done, everything would change. For the first time since this nightmare began, a flicker of hope ignited within him.`;
    
    el.textInput.value = exampleText;
    updateInputStats();
    el.textInput.focus();
}

/* === Initial Stats Update === */

updateInputStats();
