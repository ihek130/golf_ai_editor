# config.py - Centralized Configuration Management for Golf AI Video Editor
# ============================================================================

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# Environment Detection
# ============================================================================

class Environment(Enum):
    """Application environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

# Detect current environment
ENVIRONMENT = Environment(os.getenv("GOLF_AI_ENV", "development"))
IS_DEVELOPMENT = ENVIRONMENT == Environment.DEVELOPMENT
IS_PRODUCTION = ENVIRONMENT == Environment.PRODUCTION

# ============================================================================
# Base Directories
# ============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
LOGS_DIR = PROJECT_ROOT / "logs"

# Processing directories
UPLOAD_DIR = STATIC_DIR / "uploads"
PROCESSED_DIR = STATIC_DIR / "processed"
TEMP_DIR = STATIC_DIR / "temp"
MODELS_DIR = DATA_DIR / "models"

# Ensure directories exist
for directory in [DATA_DIR, STATIC_DIR, TEMPLATES_DIR, LOGS_DIR, 
                  UPLOAD_DIR, PROCESSED_DIR, TEMP_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Server Configuration
# ============================================================================

@dataclass
class ServerConfig:
    """Server and API configuration"""
    host: str = "0.0.0.0"
    port: int = int(os.getenv("PORT", 8000))
    debug: bool = IS_DEVELOPMENT
    reload: bool = IS_DEVELOPMENT
    workers: int = 1 if IS_DEVELOPMENT else 4
    
    # API settings
    api_title: str = "Golf AI Video Editor"
    api_version: str = "2.0.0"
    api_description: str = "AI-powered golf video processing and highlight generation"
    
    # CORS settings
    cors_origins: List[str] = ["*"] if IS_DEVELOPMENT else []
    cors_credentials: bool = True
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]
    
    # Request limits
    max_request_size: int = 1024 * 1024 * 1024  # 1GB for large videos
    request_timeout: int = 3600  # 1 hour for video processing

SERVER = ServerConfig()

# ============================================================================
# Video Processing Configuration
# ============================================================================

@dataclass
class VideoConfig:
    """Video processing and validation settings"""
    
    # File validation
    max_file_size: int = 500 * 1024 * 1024  # 500MB default
    max_file_size_premium: int = 2 * 1024 * 1024 * 1024  # 2GB for premium users
    
    allowed_extensions: tuple = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv')
    allowed_mime_types: tuple = (
        'video/mp4', 'video/avi', 'video/quicktime', 
        'video/x-msvideo', 'video/x-matroska'
    )
    
    # Processing constraints
    max_duration: int = 3600  # 1 hour max video duration
    min_duration: int = 5     # 5 seconds minimum
    max_resolution: tuple = (4096, 2160)  # 4K max resolution
    min_resolution: tuple = (640, 480)    # 480p minimum
    
    # Output settings
    output_format: str = "mp4"
    output_codec: str = "libx264"
    output_audio_codec: str = "aac"
    
    # Quality presets
    quality_presets: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.quality_presets is None:
            self.quality_presets = {
                "low": {
                    "bitrate": "2000k",
                    "preset": "fast",
                    "crf": 28,
                    "resolution": (1280, 720)
                },
                "medium": {
                    "bitrate": "5000k", 
                    "preset": "medium",
                    "crf": 23,
                    "resolution": (1920, 1080)
                },
                "high": {
                    "bitrate": "8000k",
                    "preset": "slower", 
                    "crf": 20,
                    "resolution": (1920, 1080)
                },
                "ultra": {
                    "bitrate": "15000k",
                    "preset": "slower",
                    "crf": 18,
                    "resolution": (3840, 2160)
                }
            }

VIDEO = VideoConfig()

# ============================================================================
# AI Model Configuration  
# ============================================================================

@dataclass
class AIConfig:
    """AI and machine learning model settings"""
    
    # YOLO Detection Model
    yolo_model_path: str = "yolov8n.pt"  # Default lightweight model
    yolo_model_custom: Optional[str] = None  # Path to custom golf-trained model
    yolo_confidence_threshold: float = 0.5
    yolo_iou_threshold: float = 0.45
    yolo_max_detections: int = 10
    
    # Player Detection Settings
    max_players: int = 4
    min_detections_per_player: int = 5  # Minimum detections to consider a valid player
    player_similarity_threshold: float = 0.7  # Face/body similarity threshold
    
    # Shot Detection Parameters
    min_shot_duration: float = 1.0   # Minimum shot duration in seconds
    max_shot_duration: float = 15.0  # Maximum shot duration in seconds
    swing_motion_threshold: float = 0.3  # Motion threshold for golf swing detection
    shot_boundary_gap: float = 5.0   # Time gap to separate shots
    
    # Feature Extraction
    face_feature_size: int = 256     # LBP face feature vector size
    body_feature_size: int = 96      # Color histogram feature size
    temporal_window_size: int = 10   # Frames to analyze for motion
    
    # Processing Optimization
    frame_skip_interval: int = 2     # Process every Nth frame for speed
    batch_processing_size: int = 32  # Batch size for model inference
    use_gpu: bool = True             # Use GPU acceleration if available
    
    # Advanced Settings
    tracking_max_age: int = 30       # Max frames to keep lost tracks
    tracking_min_hits: int = 3       # Min detections before confirming track
    optical_flow_quality: float = 0.01  # Optical flow corner detection quality

AI = AIConfig()

# ============================================================================
# Job Management Configuration
# ============================================================================

@dataclass  
class JobConfig:
    """Background job processing settings"""
    
    # Job lifecycle
    max_concurrent_jobs: int = 2 if IS_DEVELOPMENT else 4
    job_timeout: int = 7200  # 2 hours max processing time
    job_cleanup_interval: int = 1800  # Clean up every 30 minutes
    job_retention_hours: int = 24  # Keep completed jobs for 24 hours
    
    # Progress tracking
    progress_update_interval: float = 2.0  # Update progress every 2 seconds
    status_poll_timeout: int = 300  # 5 minute timeout for status polling
    
    # Job queue settings
    max_queue_size: int = 50
    priority_processing: bool = True  # Premium users get priority
    
    # Retry logic
    max_retries: int = 3
    retry_delay: int = 60  # 1 minute between retries
    
    # Notification settings
    enable_email_notifications: bool = False
    enable_webhook_notifications: bool = False

JOBS = JobConfig()

# ============================================================================
# Storage Configuration
# ============================================================================

@dataclass
class StorageConfig:
    """File storage and cleanup settings"""
    
    # Local storage paths
    upload_path: Path = UPLOAD_DIR
    processed_path: Path = PROCESSED_DIR
    temp_path: Path = TEMP_DIR
    
    # File lifecycle
    temp_file_lifetime: int = 3600    # 1 hour for temp files
    upload_file_lifetime: int = 21600  # 6 hours for uploaded files  
    processed_file_lifetime: int = 86400  # 24 hours for processed files
    
    # Cleanup settings
    auto_cleanup_enabled: bool = True
    cleanup_check_interval: int = 1800  # Check every 30 minutes
    storage_quota_gb: int = 100  # 100GB storage limit
    
    # Cloud storage (optional)
    use_cloud_storage: bool = False
    cloud_provider: Optional[str] = None  # "aws", "gcp", "azure"
    cloud_bucket: Optional[str] = None
    cloud_credentials_path: Optional[str] = None

STORAGE = StorageConfig()

# ============================================================================
# Logging Configuration
# ============================================================================

@dataclass
class LoggingConfig:
    """Logging and monitoring settings"""
    
    # Log levels
    level: str = "INFO" if IS_PRODUCTION else "DEBUG"
    log_to_file: bool = True
    log_to_console: bool = True
    
    # File settings
    log_dir: Path = LOGS_DIR
    log_filename: str = "golf_ai_editor.log"
    max_log_size: int = 10 * 1024 * 1024  # 10MB per log file
    backup_count: int = 5  # Keep 5 backup files
    
    # Format settings
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Performance logging
    log_processing_times: bool = True
    log_memory_usage: bool = IS_DEVELOPMENT
    log_api_requests: bool = True
    
    # Error tracking
    enable_error_tracking: bool = IS_PRODUCTION
    sentry_dsn: Optional[str] = os.getenv("SENTRY_DSN")

LOGGING = LoggingConfig()

# ============================================================================
# Security Configuration
# ============================================================================

@dataclass
class SecurityConfig:
    """Security and authentication settings"""
    
    # API Security
    enable_api_key_auth: bool = IS_PRODUCTION
    api_key_header: str = "X-API-Key"
    rate_limit_requests: int = 100  # Requests per minute
    rate_limit_window: int = 60     # Window in seconds
    
    # File upload security
    scan_uploaded_files: bool = IS_PRODUCTION
    allowed_file_signatures: Dict[str, bytes] = None
    max_filename_length: int = 255
    sanitize_filenames: bool = True
    
    # CORS and headers
    secure_headers: bool = IS_PRODUCTION
    content_security_policy: bool = IS_PRODUCTION
    
    # User limits
    max_uploads_per_user_per_hour: int = 10
    max_processing_jobs_per_user: int = 3
    
    def __post_init__(self):
        if self.allowed_file_signatures is None:
            self.allowed_file_signatures = {
                'mp4': b'\x00\x00\x00\x20\x66\x74\x79\x70',
                'avi': b'RIFF',
                'mov': b'\x00\x00\x00\x14\x66\x74\x79\x70\x71\x74'
            }

SECURITY = SecurityConfig()

# ============================================================================
# Feature Flags
# ============================================================================

@dataclass
class FeatureFlags:
    """Feature toggle configuration"""
    
    # AI Features
    enable_face_recognition: bool = True
    enable_advanced_tracking: bool = True
    enable_shot_quality_analysis: bool = False  # Coming soon
    enable_ball_tracking: bool = False          # Experimental
    
    # Video Features  
    enable_video_stabilization: bool = False    # Requires additional deps
    enable_slow_motion_detection: bool = False  # Experimental
    enable_audio_analysis: bool = False         # Future feature
    
    # UI Features
    enable_real_time_preview: bool = False      # Resource intensive
    enable_batch_processing: bool = True
    enable_custom_branding: bool = False
    
    # Analytics
    enable_usage_analytics: bool = IS_PRODUCTION
    enable_performance_monitoring: bool = True
    enable_error_reporting: bool = IS_PRODUCTION

FEATURES = FeatureFlags()

# ============================================================================
# Database Configuration (Optional)
# ============================================================================

@dataclass
class DatabaseConfig:
    """Database settings for job tracking and user management"""
    
    # Database type
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///golf_ai.db")
    
    # Connection settings
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # Features
    enable_job_persistence: bool = IS_PRODUCTION
    enable_user_management: bool = False
    enable_usage_tracking: bool = IS_PRODUCTION

DATABASE = DatabaseConfig()

# ============================================================================
# Performance Configuration
# ============================================================================

@dataclass
class PerformanceConfig:
    """Performance optimization settings"""
    
    # Memory management
    max_memory_usage_gb: int = 8     # Max RAM usage
    memory_cleanup_threshold: float = 0.8  # Cleanup at 80% usage
    
    # Processing optimization
    max_concurrent_video_processing: int = 2
    frame_processing_batch_size: int = 50
    use_multiprocessing: bool = True
    max_worker_processes: int = os.cpu_count() or 4
    
    # Caching
    enable_result_caching: bool = True
    cache_duration_hours: int = 24
    max_cache_size_gb: int = 5
    
    # GPU settings
    gpu_memory_fraction: float = 0.8  # Use 80% of GPU memory
    allow_gpu_growth: bool = True

PERFORMANCE = PerformanceConfig()

# ============================================================================
# Development Configuration
# ============================================================================

@dataclass
class DevelopmentConfig:
    """Development and debugging settings"""
    
    # Debug features
    enable_debug_mode: bool = IS_DEVELOPMENT
    enable_verbose_logging: bool = IS_DEVELOPMENT
    save_intermediate_results: bool = IS_DEVELOPMENT
    
    # Testing
    enable_mock_processing: bool = False  # For UI testing
    mock_processing_delay: int = 5        # Seconds
    
    # Development tools
    enable_hot_reload: bool = IS_DEVELOPMENT
    enable_api_docs: bool = True
    enable_profiling: bool = IS_DEVELOPMENT

DEVELOPMENT = DevelopmentConfig()

# ============================================================================
# Configuration Validation
# ============================================================================

def validate_config() -> List[str]:
    """Validate configuration settings and return any errors"""
    errors = []
    
    # Check required directories
    if not MODELS_DIR.exists():
        errors.append(f"Models directory does not exist: {MODELS_DIR}")
    
    # Validate file size limits
    if VIDEO.max_file_size <= 0:
        errors.append("Video max_file_size must be positive")
    
    # Check AI model availability
    if not Path(AI.yolo_model_path).exists() and not AI.yolo_model_path.startswith('yolo'):
        errors.append(f"YOLO model not found: {AI.yolo_model_path}")
    
    # Validate thresholds
    if not 0 <= AI.yolo_confidence_threshold <= 1:
        errors.append("YOLO confidence threshold must be between 0 and 1")
    
    if not 0 <= AI.player_similarity_threshold <= 1:
        errors.append("Player similarity threshold must be between 0 and 1")
    
    # Check processing limits
    if AI.min_shot_duration >= AI.max_shot_duration:
        errors.append("min_shot_duration must be less than max_shot_duration")
    
    return errors

# ============================================================================
# Configuration Export
# ============================================================================

def get_config_summary() -> Dict:
    """Get a summary of current configuration"""
    return {
        "environment": ENVIRONMENT.value,
        "server": {
            "host": SERVER.host,
            "port": SERVER.port,
            "debug": SERVER.debug
        },
        "video": {
            "max_file_size_mb": VIDEO.max_file_size // (1024 * 1024),
            "allowed_extensions": VIDEO.allowed_extensions,
            "max_players": AI.max_players
        },
        "features": {
            "face_recognition": FEATURES.enable_face_recognition,
            "advanced_tracking": FEATURES.enable_advanced_tracking,
            "batch_processing": FEATURES.enable_batch_processing
        },
        "performance": {
            "max_concurrent_jobs": JOBS.max_concurrent_jobs,
            "use_gpu": AI.use_gpu,
            "max_workers": PERFORMANCE.max_worker_processes
        }
    }

# ============================================================================
# Environment-Specific Overrides
# ============================================================================

if ENVIRONMENT == Environment.PRODUCTION:
    # Production optimizations
    AI.frame_skip_interval = 1  # Process all frames for quality
    VIDEO.output_codec = "libx264"  # Ensure compatibility
    JOBS.max_concurrent_jobs = 4  # Allow more concurrent processing
    LOGGING.level = "INFO"
    
elif ENVIRONMENT == Environment.TESTING:
    # Testing configurations
    VIDEO.max_file_size = 50 * 1024 * 1024  # 50MB for tests
    AI.max_players = 2  # Faster testing
    JOBS.job_timeout = 300  # 5 minute timeout for tests
    DEVELOPMENT.enable_mock_processing = True

# ============================================================================
# Configuration Initialization
# ============================================================================

def initialize_config():
    """Initialize and validate configuration on startup"""
    # Validate configuration
    config_errors = validate_config()
    if config_errors:
        for error in config_errors:
            logging.error(f"Configuration error: {error}")
        raise ValueError(f"Configuration validation failed: {config_errors}")
    
    # Log configuration summary
    logging.info(f"🏌️ Golf AI Video Editor initialized")
    logging.info(f"Environment: {ENVIRONMENT.value}")
    logging.info(f"Server: {SERVER.host}:{SERVER.port}")
    logging.info(f"Max file size: {VIDEO.max_file_size // (1024*1024)}MB")
    logging.info(f"Max players: {AI.max_players}")
    logging.info(f"GPU enabled: {AI.use_gpu}")
    
    return True

# Auto-initialize when imported
if __name__ != "__main__":
    try:
        initialize_config()
    except Exception as e:
        logging.error(f"Failed to initialize configuration: {e}")

# ============================================================================
# Export all configuration objects
# ============================================================================

__all__ = [
    'ENVIRONMENT', 'IS_DEVELOPMENT', 'IS_PRODUCTION',
    'PROJECT_ROOT', 'UPLOAD_DIR', 'PROCESSED_DIR', 'MODELS_DIR',
    'SERVER', 'VIDEO', 'AI', 'JOBS', 'STORAGE', 'LOGGING', 
    'SECURITY', 'FEATURES', 'DATABASE', 'PERFORMANCE', 'DEVELOPMENT',
    'validate_config', 'get_config_summary', 'initialize_config'
]