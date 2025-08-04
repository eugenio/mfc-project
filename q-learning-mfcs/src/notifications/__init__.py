"""
Comprehensive Notification System for MFC Monitoring
===================================================

This module provides a complete notification system with:
- Cross-platform desktop notifications
- Audio playback with ding sounds  
- Text-to-Speech (TTS) synthesis
- Priority-based queue management
- Platform-specific optimizations
- Comprehensive fallback mechanisms

Main Components:
- NotificationManager: Main interface for all notifications
- AudioManager: Handles sound playback and audio themes
- QueueManager: Manages notification queues with priorities
- Platform handlers: Linux, macOS, Windows specific implementations

Usage:
    from notifications import NotificationManager, NotificationLevel
    
    # Initialize notification system
    manager = NotificationManager()
    
    # Send notifications
    manager.notify("Alert", "System status critical", 
                   NotificationLevel.CRITICAL, 
                   sound_enabled=True, tts_enabled=True)
    
    # Or use convenience methods
    manager.critical("System Failure", "Database connection lost")
    manager.success("Task Complete", "Data processing finished")

Created: 2025-08-03  
Author: Agent Delta - Audio Integration Specialist
"""

from .base import NotificationConfig, NotificationHandler, NotificationLevel

# Core notification system
__all__ = [
    "NotificationHandler",
    "NotificationLevel",
    "NotificationConfig"
]

# Audio integration components
try:
    from .audio_manager import AudioConfig, AudioEvent, AudioManager
    __all__.extend(["AudioManager", "AudioConfig", "AudioEvent"])
except ImportError:
    pass

try:
    from .queue_manager import (
        NotificationQueueManager,
        QueueConfig,
        QueuePriority,
        TTSManager,
    )
    __all__.extend(["NotificationQueueManager", "QueueConfig", "QueuePriority", "TTSManager"])
except ImportError:
    pass

# Main notification manager
try:
    from .manager import (
        NotificationManager,
        NotificationManagerConfig,
        get_notification_manager,
        initialize_notifications,
        shutdown_notifications,
    )
    __all__.extend([
        "NotificationManager", "NotificationManagerConfig",
        "get_notification_manager", "initialize_notifications", "shutdown_notifications"
    ])
except ImportError:
    pass

# Platform detection and handlers
try:
    from .platform_detection import (
        PlatformInfo,
        get_platform_handler,
        validate_platform_capabilities,
    )
    __all__.extend(["get_platform_handler", "PlatformInfo", "validate_platform_capabilities"])
except ImportError:
    pass

# Platform-specific handlers (optional)
try:
    from .linux_handler import LinuxNotificationHandler
    __all__.append("LinuxNotificationHandler")
except ImportError:
    pass

try:
    from .windows_handler import WindowsNotificationHandler
    __all__.append("WindowsNotificationHandler")
except ImportError:
    pass

try:
    from .macos_handler import MacOSNotificationHandler
    __all__.append("MacOSNotificationHandler")
except ImportError:
    pass

# TTS components
try:
    from .tts_handler import (
        Pyttsx3Engine,
        TTSEngineType,
        TTSMode,
        TTSNotificationHandler,
    )
    __all__.extend([
        "TTSNotificationHandler",
        "TTSMode",
        "TTSEngineType",
        "Pyttsx3Engine"
    ])
except ImportError as e:
    # TTS components are optional
    import logging
    logging.getLogger(__name__).debug(f"TTS components not available: {e}")

# Advanced TTS components (Coqui TTS)
try:
    from .coqui_tts_manager import CoquiTTSConfig, CoquiTTSManager, HybridTTSManager
    __all__.extend(["CoquiTTSManager", "CoquiTTSConfig", "HybridTTSManager"])
except ImportError as e:
    # Coqui TTS is optional
    import logging
    logging.getLogger(__name__).debug(f"Coqui TTS components not available: {e}")

# Convenience aliases for common usage patterns
try:
    # Create default notification manager instance for simple usage
    _default_manager = None

    def notify(title: str, message: str = "", level: NotificationLevel = NotificationLevel.INFO, **kwargs):
        """
        Send a notification using the default manager.
        
        Args:
            title: Notification title
            message: Notification message  
            level: Notification level
            **kwargs: Additional arguments passed to NotificationManager.notify()
        """
        global _default_manager
        if _default_manager is None:
            _default_manager = get_notification_manager()
        return _default_manager.notify(title, message, level, **kwargs)

    def info(title: str, message: str = "", **kwargs):
        """Send info notification."""
        return notify(title, message, NotificationLevel.INFO, **kwargs)

    def warning(title: str, message: str = "", **kwargs):
        """Send warning notification."""
        return notify(title, message, NotificationLevel.WARNING, **kwargs)

    def critical(title: str, message: str = "", **kwargs):
        """Send critical notification."""
        return notify(title, message, NotificationLevel.CRITICAL, **kwargs)

    def success(title: str, message: str = "", **kwargs):
        """Send success notification."""
        return notify(title, message, NotificationLevel.SUCCESS, **kwargs)

    def play_ding(level: NotificationLevel = NotificationLevel.INFO, **kwargs):
        """Play a ding sound."""
        global _default_manager
        if _default_manager is None:
            _default_manager = get_notification_manager()
        return _default_manager.play_ding(level, **kwargs)

    def speak(text: str, level: NotificationLevel = NotificationLevel.INFO, **kwargs):
        """Speak text using TTS."""
        global _default_manager
        if _default_manager is None:
            _default_manager = get_notification_manager()
        return _default_manager.speak(text, level, **kwargs)

    __all__.extend([
        "notify", "info", "warning", "critical", "success",
        "play_ding", "speak"
    ])

except ImportError:
    # If manager components not available, skip convenience functions
    pass
