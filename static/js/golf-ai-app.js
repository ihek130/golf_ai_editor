// Golf AI Video Editor Pro - Complete Ultra-Fast JavaScript Application
// ====================================================================

console.log('🚀 Golf AI Video Editor Pro - Ultra-Fast Processing initialized');

// Application state
const AppState = {
    uploadedFiles: {
        tee: null,
        green: null,
        putt: null
    },
    currentJob: null,
    progressInterval: null,
    startTime: null,
    isProcessing: false
};

// Configuration presets for ultra-fast processing
const ConfigPresets = {
    ultra_fast_standard: {
        processing_mode: "ultra_fast",
        players_per_group: 4,
        output_quality: "high",
        description: "Standard (4 Players, 4 minutes)"
    },
    ultra_fast_small_group: {
        processing_mode: "ultra_fast", 
        players_per_group: 2,
        output_quality: "high",
        description: "Small Group (2 Players, 3 minutes)"
    },
    ultra_fast_preview: {
        processing_mode: "ultra_fast",
        players_per_group: 4,
        output_quality: "low",
        description: "Lightning Preview (Under 3 minutes)"
    }
};

// DOM Elements
const Elements = {};

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('🔧 Initializing Golf AI Video Editor Pro...');
    
    initializeElements();
    setupEventListeners();
    setupFileUploads();
    setupConfiguration();
    
    console.log('✅ Golf AI Video Editor Pro ready!');
});

// Initialize DOM element references
function initializeElements() {
    Elements.uploadForm = document.getElementById('uploadForm');
    Elements.processButton = document.getElementById('processButton');
    Elements.progressContainer = document.getElementById('progressContainer');
    Elements.resultsSection = document.getElementById('resultsSection');
    Elements.alertContainer = document.getElementById('alertContainer');
    Elements.loadingOverlay = document.getElementById('loadingOverlay');
    
    // Upload cards
    Elements.teeCard = document.getElementById('teeCard');
    Elements.greenCard = document.getElementById('greenCard');
    Elements.puttCard = document.getElementById('puttCard');
    
    // File inputs
    Elements.teeInput = document.getElementById('tee_shot');
    Elements.greenInput = document.getElementById('green_shot');
    Elements.puttInput = document.getElementById('putt_shot');
    
    // Progress elements
    Elements.progressCircle = document.getElementById('progressCircle');
    Elements.progressPercentage = document.getElementById('progressPercentage');
    Elements.progressBar = document.getElementById('progressBar');
    Elements.statusMessage = document.getElementById('statusMessage');
    Elements.timeRemaining = document.getElementById('timeRemaining');
    Elements.progressTime = document.getElementById('progressTime');
    Elements.progressSteps = document.getElementById('progressSteps');
    
    // Configuration elements
    Elements.configToggle = document.getElementById('toggleConfig');
    Elements.configContent = document.getElementById('configContent');
    Elements.processingMode = document.getElementById('processing_mode');
    Elements.playersPerGroup = document.getElementById('players_per_group');
    Elements.outputQuality = document.getElementById('output_quality');
    
    // Results elements
    Elements.videoGrid = document.getElementById('videoGrid');
    Elements.resultsSummary = document.getElementById('resultsSummary');
}

// Setup main event listeners
function setupEventListeners() {
    // Form submission
    if (Elements.uploadForm) {
        Elements.uploadForm.addEventListener('submit', handleFormSubmission);
    }
    
    // Configuration toggle
    if (Elements.configToggle) {
        Elements.configToggle.addEventListener('click', toggleConfiguration);
    }
    
    // Preset buttons
    document.querySelectorAll('.preset-btn').forEach(button => {
        button.addEventListener('click', handlePresetSelection);
    });
    
    // Upload card clicks
    [Elements.teeCard, Elements.greenCard, Elements.puttCard].forEach(card => {
        if (card) {
            card.addEventListener('click', handleCardClick);
        }
    });
}

// Setup file upload handling
function setupFileUploads() {
    const fileInputs = [
        { input: Elements.teeInput, card: Elements.teeCard, type: 'tee' },
        { input: Elements.greenInput, card: Elements.greenCard, type: 'green' },
        { input: Elements.puttInput, card: Elements.puttCard, type: 'putt' }
    ];
    
    fileInputs.forEach(({ input, card, type }) => {
        if (input) {
            input.addEventListener('change', (e) => handleFileSelection(e, card, type));
        }
    });
}

// Handle file selection
function handleFileSelection(event, card, type) {
    const file = event.target.files[0];
    
    if (file) {
        // Validate file
        if (!validateVideoFile(file)) {
            showAlert('error', `Invalid video file: ${file.name}. Please select a valid video file.`);
            return;
        }
        
        // Store file
        AppState.uploadedFiles[type] = file;
        
        // Update UI
        updateCardUI(card, file.name, true);
        
        console.log(`✅ ${type} file selected: ${file.name}`);
        
        // Enable process button if tee shot is uploaded
        if (AppState.uploadedFiles.tee) {
            enableProcessButton();
        }
    }
}

// Validate video file
function validateVideoFile(file) {
    const validTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv', 'video/wmv', 'video/quicktime'];
    const validExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv'];
    
    // Check MIME type
    if (!validTypes.includes(file.type)) {
        // Fallback to extension check
        const extension = file.name.toLowerCase().substr(file.name.lastIndexOf('.'));
        if (!validExtensions.includes(extension)) {
            return false;
        }
    }
    
    // Check file size (500MB limit)
    const maxSize = 500 * 1024 * 1024;
    if (file.size > maxSize) {
        showAlert('error', `File too large: ${(file.size / (1024 * 1024)).toFixed(1)}MB. Maximum size is 500MB.`);
        return false;
    }
    
    return true;
}

// Update card UI after file selection
function updateCardUI(card, fileName, isUploaded) {
    if (!card) return;
    
    const uploadButton = card.querySelector('.upload-button span');
    const uploadStatus = card.querySelector('.upload-status');
    
    if (isUploaded) {
        card.classList.add('uploaded');
        if (uploadButton) {
            uploadButton.textContent = fileName.length > 20 ? fileName.substring(0, 20) + '...' : fileName;
        }
        if (uploadStatus) {
            uploadStatus.style.display = 'flex';
        }
    }
}

// Enable process button
function enableProcessButton() {
    if (Elements.processButton) {
        Elements.processButton.disabled = false;
        Elements.processButton.innerHTML = '<i class="fas fa-rocket"></i><span>Process Videos Ultra-Fast (3-5 Minutes)</span>';
        Elements.processButton.style.opacity = '1';
    }
}

// Disable process button
function disableProcessButton() {
    if (Elements.processButton) {
        Elements.processButton.disabled = true;
        Elements.processButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Ultra-Fast Processing...</span>';
        Elements.processButton.style.opacity = '0.7';
    }
}

// Handle card clicks
function handleCardClick(event) {
    const card = event.currentTarget;
    const cardId = card.id;
    
    let inputId;
    switch (cardId) {
        case 'teeCard':
            inputId = 'tee_shot';
            break;
        case 'greenCard':
            inputId = 'green_shot';
            break;
        case 'puttCard':
            inputId = 'putt_shot';
            break;
    }
    
    if (inputId) {
        const input = document.getElementById(inputId);
        if (input) {
            input.click();
        }
    }
}

// Handle form submission
async function handleFormSubmission(event) {
    event.preventDefault();
    event.stopPropagation();
    
    console.log('🚀 Starting ultra-fast video processing...');
    
    // Validate required tee shot
    if (!AppState.uploadedFiles.tee) {
        showAlert('error', 'Please select a tee shot video before processing');
        return;
    }
    
    // Disable form and show loading
    disableProcessButton();
    showLoadingOverlay('Preparing your videos for ultra-fast processing...');
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('tee_shot', AppState.uploadedFiles.tee);
        
        if (AppState.uploadedFiles.green) {
            formData.append('green_shot', AppState.uploadedFiles.green);
        }
        
        if (AppState.uploadedFiles.putt) {
            formData.append('putt_shot', AppState.uploadedFiles.putt);
        }
        
        // Add configuration
        const config = getFormConfiguration();
        Object.keys(config).forEach(key => {
            formData.append(key, config[key]);
        });
        
        console.log('📤 Uploading files...', {
            tee: AppState.uploadedFiles.tee?.name,
            green: AppState.uploadedFiles.green?.name,
            putt: AppState.uploadedFiles.putt?.name,
            config: config
        });
        
        // Submit form
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        hideLoadingOverlay();
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }
        
        const result = await response.json();
        console.log('🚀 Ultra-fast upload successful:', result);
        
        // Start progress tracking
        AppState.currentJob = result.job_id;
        AppState.startTime = Date.now();
        AppState.isProcessing = true;
        
        showAlert('success', `🚀 Upload successful! Ultra-fast processing started (3-5 minutes) - Job ID: ${result.job_id}`);
        startProgressTracking(result.job_id);
        
    } catch (error) {
        console.error('❌ Upload failed:', error);
        hideLoadingOverlay();
        showAlert('error', `❌ Upload failed: ${error.message}`);
        enableProcessButton();
    }
}

// Get form configuration for ultra-fast processing
function getFormConfiguration() {
    return {
        processing_mode: Elements.processingMode?.value || 'ultra_fast',
        players_per_group: Elements.playersPerGroup?.value || '4',
        output_quality: Elements.outputQuality?.value || 'medium'
    };
}

// Start progress tracking for ultra-fast processing
function startProgressTracking(jobId) {
    console.log('🚀 Starting ultra-fast progress tracking for job:', jobId);
    
    // Show progress container
    if (Elements.progressContainer) {
        Elements.progressContainer.classList.add('active');
        Elements.progressContainer.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Initialize progress
    updateProgress(0, 'Starting ultra-fast processing...', 'upload');
    
    // Start polling - check more frequently for ultra-fast processing
    AppState.progressInterval = setInterval(() => {
        checkProgress(jobId);
    }, 1500); // Check every 1.5 seconds for faster updates
    
    // Also check immediately
    checkProgress(jobId);
}

// Check progress
async function checkProgress(jobId) {
    try {
        const response = await fetch(`/status/${jobId}`);
        if (!response.ok) {
            throw new Error('Status check failed');
        }
        
        const job = await response.json();
        console.log(`Progress: ${job.progress}% - ${job.message}`);
        
        // Update progress UI
        updateProgressUI(job);
        
        // Check if completed
        if (job.status === 'completed') {
            clearInterval(AppState.progressInterval);
            handleProcessingComplete(job);
        } else if (job.status === 'failed') {
            clearInterval(AppState.progressInterval);
            handleProcessingFailed(job);
        }
        
    } catch (error) {
        console.error('Progress check error:', error);
    }
}

// Update progress UI for ultra-fast processing
function updateProgressUI(job) {
    const progress = job.progress || 0;
    const message = job.message || 'Ultra-fast processing...';
    
    // Determine current step for ultra-fast processing
    let currentStep = 'upload';
    if (progress >= 10 && progress < 25) currentStep = 'audio';
    else if (progress >= 25 && progress < 50) currentStep = 'correlation';
    else if (progress >= 50 && progress < 100) currentStep = 'processing';
    else if (progress >= 100) currentStep = 'complete';
    
    updateProgress(progress, message, currentStep);
}

// Update progress display
function updateProgress(percentage, message, currentStep) {
    // Update percentage
    if (Elements.progressPercentage) {
        Elements.progressPercentage.textContent = `${Math.round(percentage)}%`;
    }
    
    // Update progress circle
    if (Elements.progressCircle) {
        const degrees = (percentage / 100) * 360;
        Elements.progressCircle.style.background = `conic-gradient(var(--primary-green) ${degrees}deg, var(--light-gray) ${degrees}deg)`;
    }
    
    // Update progress bar
    if (Elements.progressBar) {
        Elements.progressBar.style.width = `${percentage}%`;
    }
    
    // Update status message
    if (Elements.statusMessage) {
        Elements.statusMessage.textContent = message;
    }
    
    // Update time remaining with ultra-fast estimates
    updateTimeRemainingUltraFast(percentage);
    
    // Update elapsed time
    updateElapsedTime();
    
    // Update steps for ultra-fast processing
    updateProgressStepsUltraFast(currentStep);
}

// Update time remaining estimation for ultra-fast processing
function updateTimeRemainingUltraFast(percentage) {
    if (!AppState.startTime || !Elements.timeRemaining) return;
    
    const elapsedTime = Date.now() - AppState.startTime;
    
    if (percentage > 5) {
        // Ultra-fast processing should complete in 3-5 minutes
        const maxExpectedTime = 5 * 60 * 1000; // 5 minutes in milliseconds
        const estimatedTotalTime = Math.min(
            (elapsedTime / percentage) * 100,
            maxExpectedTime
        );
        const remainingTime = estimatedTotalTime - elapsedTime;
        
        if (remainingTime > 0) {
            const minutes = Math.floor(remainingTime / 60000);
            const seconds = Math.floor((remainingTime % 60000) / 1000);
            
            if (minutes > 0) {
                Elements.timeRemaining.textContent = `About ${minutes}m ${seconds}s remaining (Ultra-Fast!)`;
            } else {
                Elements.timeRemaining.textContent = `About ${seconds}s remaining (Ultra-Fast!)`;
            }
        } else {
            Elements.timeRemaining.textContent = 'Almost complete...';
        }
    } else {
        Elements.timeRemaining.textContent = 'Ultra-fast processing starting...';
    }
}

// Update elapsed time
function updateElapsedTime() {
    if (!AppState.startTime || !Elements.progressTime) return;
    
    const elapsedTime = Date.now() - AppState.startTime;
    const minutes = Math.floor(elapsedTime / 60000);
    const seconds = Math.floor((elapsedTime % 60000) / 1000);
    
    if (minutes > 0) {
        Elements.progressTime.textContent = `${minutes}m ${seconds}s elapsed (Target: 3-5 min)`;
    } else {
        Elements.progressTime.textContent = `${seconds}s elapsed (Target: 3-5 min)`;
    }
}

// Update progress steps for ultra-fast processing
function updateProgressStepsUltraFast(currentStep) {
    if (!Elements.progressSteps) return;
    
    const steps = Elements.progressSteps.querySelectorAll('.step');
    const stepOrder = ['upload', 'audio', 'correlation', 'processing', 'complete'];
    const currentIndex = stepOrder.indexOf(currentStep);
    
    steps.forEach((step, index) => {
        step.classList.remove('active', 'completed');
        
        if (index < currentIndex) {
            step.classList.add('completed');
        } else if (index === currentIndex) {
            step.classList.add('active');
        }
    });
}

// Handle processing completion for ultra-fast processing
function handleProcessingComplete(job) {
    console.log('🎉 Ultra-fast processing completed:', job);
    
    AppState.isProcessing = false;
    
    // Update final progress
    updateProgress(100, 'Ultra-fast processing complete!', 'complete');
    
    // Hide progress after delay
    setTimeout(() => {
        if (Elements.progressContainer) {
            Elements.progressContainer.classList.remove('active');
        }
    }, 3000);
    
    // Show success message with speed info
    const videoCount = job.results ? job.results.length : 0;
    const processingTime = AppState.startTime ? Math.round((Date.now() - AppState.startTime) / 1000) : 0;
    const minutes = Math.floor(processingTime / 60);
    const seconds = processingTime % 60;
    
    let timeText = '';
    if (minutes > 0) {
        timeText = ` in ${minutes}m ${seconds}s`;
    } else {
        timeText = ` in ${seconds}s`;
    }
    
    showAlert('success', `🚀 Ultra-fast processing complete${timeText}! Created ${videoCount} highlight videos.`);
    
    // Enable button
    enableProcessButton();
    
    // Show results
    if (job.results && job.results.length > 0) {
        displayResults(job.results);
    } else {
        showAlert('warning', '⚠️ No highlight videos were created. This may happen if no golf activity was detected in the videos.');
    }
}

// Handle processing failure
function handleProcessingFailed(job) {
    console.error('❌ Processing failed:', job);
    
    AppState.isProcessing = false;
    
    // Hide progress
    if (Elements.progressContainer) {
        Elements.progressContainer.classList.remove('active');
    }
    
    // Show error message
    const errorMsg = job.error || 'Unknown processing error occurred';
    showAlert('error', `❌ Processing failed: ${errorMsg}`);
    
    // Enable button
    enableProcessButton();
}

// Display results
function displayResults(videos) {
    console.log('📺 Displaying results:', videos);
    
    if (!Elements.resultsSection || !Elements.videoGrid) return;
    
    // Show results section
    Elements.resultsSection.classList.add('active');
    
    // Update summary
    if (Elements.resultsSummary) {
        Elements.resultsSummary.textContent = `${videos.length} highlight videos created with ultra-fast processing`;
    }
    
    // Clear previous results
    Elements.videoGrid.innerHTML = '';
    
    // Create video cards
    videos.forEach((videoUrl, index) => {
        const videoCard = createVideoCard(videoUrl, index + 1);
        Elements.videoGrid.appendChild(videoCard);
    });
    
    // Scroll to results
    Elements.resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Create video card
function createVideoCard(videoUrl, playerNumber) {
    const card = document.createElement('div');
    card.className = 'video-card';
    
    card.innerHTML = `
        <div class="video-header">
            <h3><i class="fas fa-user"></i> Player ${playerNumber}</h3>
            <div class="video-badge">Ultra-Fast</div>
        </div>
        <div class="video-content">
            <video width="100%" height="250" controls preload="metadata">
                <source src="${videoUrl}" type="video/mp4" />
                Your browser does not support the video tag.
            </video>
            <div class="video-actions">
                <a href="${videoUrl}" download class="download-btn">
                    <i class="fas fa-download"></i>
                    <span>Download</span>
                </a>
            </div>
        </div>
    `;
    
    return card;
}

// Configuration management
function setupConfiguration() {
    // Initialize configuration state
    updateConfigurationVisibility();
}

// Toggle configuration visibility
function toggleConfiguration() {
    if (!Elements.configContent || !Elements.configToggle) return;
    
    const isExpanded = Elements.configContent.classList.contains('expanded');
    const toggleIcon = Elements.configToggle.querySelector('i');
    const toggleText = Elements.configToggle.querySelector('span');
    
    if (isExpanded) {
        Elements.configContent.classList.remove('expanded');
        if (toggleIcon) toggleIcon.className = 'fas fa-cog';
        if (toggleText) toggleText.textContent = 'Show Settings';
    } else {
        Elements.configContent.classList.add('expanded');
        if (toggleIcon) toggleIcon.className = 'fas fa-eye-slash';
        if (toggleText) toggleText.textContent = 'Hide Settings';
        
        // Smooth scroll to configuration
        setTimeout(() => {
            Elements.configContent.scrollIntoView({ behavior: 'smooth' });
        }, 100);
    }
}

// Update configuration visibility
function updateConfigurationVisibility() {
    // Configuration is initially hidden
    if (Elements.configContent) {
        Elements.configContent.classList.remove('expanded');
    }
}

// Handle preset selection
async function handlePresetSelection(event) {
    const presetName = event.currentTarget.dataset.preset;
    
    if (!presetName || !ConfigPresets[presetName]) {
        showAlert('error', 'Invalid preset selected');
        return;
    }
    
    const preset = ConfigPresets[presetName];
    
    try {
        // Apply preset values
        applyPreset(preset);
        
        // Visual feedback
        const button = event.currentTarget;
        const originalContent = button.innerHTML;
        const originalBackground = button.style.background;
        
        button.innerHTML = '<i class="fas fa-check"></i><span>Applied!</span>';
        button.style.background = 'linear-gradient(135deg, #27ae60 0%, #229954 100%)';
        
        setTimeout(() => {
            button.innerHTML = originalContent;
            button.style.background = originalBackground;
        }, 2000);
        
        showAlert('success', `🚀 Applied ultra-fast preset: ${preset.description}`);
        
    } catch (error) {
        console.error('Failed to apply preset:', error);
        showAlert('error', 'Failed to apply preset configuration');
    }
}

// Apply preset configuration
function applyPreset(preset) {
    Object.keys(preset).forEach(key => {
        const element = document.getElementById(key);
        if (element && key !== 'description') {
            element.value = preset[key];
        }
    });
    
    console.log('Applied ultra-fast preset:', preset);
}

// Alert system
function showAlert(type, message) {
    if (!Elements.alertContainer) return;
    
    const alert = document.createElement('div');
    alert.className = `alert ${type}`;
    alert.style.display = 'flex';
    
    // Add icon based on type
    let icon = 'fas fa-info-circle';
    if (type === 'success') icon = 'fas fa-check-circle';
    else if (type === 'error') icon = 'fas fa-exclamation-circle';
    else if (type === 'warning') icon = 'fas fa-exclamation-triangle';
    
    alert.innerHTML = `
        <i class="${icon}"></i>
        <span>${message}</span>
    `;
    
    Elements.alertContainer.appendChild(alert);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alert.parentNode) {
            alert.style.opacity = '0';
            alert.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (alert.parentNode) {
                    alert.parentNode.removeChild(alert);
                }
            }, 300);
        }
    }, 5000);
    
    console.log(`Alert [${type}]: ${message}`);
}

// Loading overlay
function showLoadingOverlay(message) {
    if (Elements.loadingOverlay) {
        const loadingText = Elements.loadingOverlay.querySelector('.loading-text');
        if (loadingText) {
            loadingText.textContent = message;
        }
        Elements.loadingOverlay.classList.add('active');
    }
}

function hideLoadingOverlay() {
    if (Elements.loadingOverlay) {
        Elements.loadingOverlay.classList.remove('active');
    }
}

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Error handling
window.addEventListener('error', function(event) {
    console.error('JavaScript error:', event.error);
    showAlert('error', 'An unexpected error occurred. Please refresh the page and try again.');
});

window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    showAlert('error', 'An unexpected error occurred. Please refresh the page and try again.');
});

// Cleanup on page unload
window.addEventListener('beforeunload', function(event) {
    if (AppState.isProcessing) {
        event.preventDefault();
        event.returnValue = 'Ultra-fast video processing is in progress. Are you sure you want to leave?';
        return event.returnValue;
    }
});

// Expose global functions for debugging
window.GolfAI = {
    AppState,
    showAlert,
    updateProgress,
    checkProgress: () => AppState.currentJob ? checkProgress(AppState.currentJob) : console.log('No active job'),
    version: 'Ultra-Fast 3.0'
};

console.log('🚀 Golf AI Video Editor Pro - Ultra-Fast Processing fully loaded!');
console.log('⚡ Target processing time: 3-5 minutes (12x speed improvement)');