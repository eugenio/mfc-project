"""Tests for notifications/audio_manager.py - coverage target 98%+."""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from notifications.audio_manager import (
    AudioConfig,
    AudioDevice,
    AudioEvent,
    AudioManager,
    AudioPlayer,
)
from notifications.base import NotificationLevel


class TestAudioConfig:
    def test_default_config(self):
        cfg = AudioConfig()
        assert cfg.enabled is True
        assert cfg.volume == 0.7
        assert cfg.max_concurrent_sounds == 3
        assert cfg.audio_device is None
        assert cfg.fallback_to_system is True
        assert ".wav" in cfg.supported_formats

    def test_custom_config(self):
        cfg = AudioConfig(enabled=False, volume=0.5)
        assert cfg.enabled is False
        assert cfg.volume == 0.5

    def test_post_init_custom_formats(self):
        cfg = AudioConfig(supported_formats=[".wav"])
        assert cfg.supported_formats == [".wav"]


class TestAudioEvent:
    def test_default_event(self):
        evt = AudioEvent(sound_path=None, level=NotificationLevel.INFO)
        assert evt.volume is None
        assert evt.priority == 1

    def test_custom_event(self):
        evt = AudioEvent(
            sound_path=Path("/tmp/t.wav"),
            level=NotificationLevel.CRITICAL,
            volume=0.8,
            priority=4,
        )
        assert evt.priority == 4


class TestAudioDevice:
    def test_str_default(self):
        dev = AudioDevice("d", "Dev", is_default=True)
        assert "(default)" in str(dev)

    def test_str_not_default(self):
        dev = AudioDevice("d", "Dev")
        assert "(default)" not in str(dev)


class TestAudioPlayer:
    def test_play_sound_disabled(self):
        p = AudioPlayer(AudioConfig(enabled=False))
        assert p.play_sound(None) is False
        p.cleanup()

    def test_play_sound_file_blocking(self, tmp_path):
        p = AudioPlayer(AudioConfig())
        f = tmp_path / "t.wav"
        f.write_text("x")
        with patch.object(p, "_play_file_sync", return_value=True):
            assert p.play_sound(f, blocking=True) is True
        p.cleanup()

    def test_play_sound_file_nonblocking(self, tmp_path):
        p = AudioPlayer(AudioConfig())
        f = tmp_path / "t.wav"
        f.write_text("x")
        with patch.object(p, "_executor") as me:
            assert p.play_sound(f, blocking=False) is True
            me.submit.assert_called_once()
        p.cleanup()

    def test_play_sound_system_blocking(self):
        p = AudioPlayer(AudioConfig(fallback_to_system=True))
        with patch.object(p, "_play_system_sound_sync", return_value=True):
            assert p.play_sound(None, blocking=True) is True
        p.cleanup()

    def test_play_sound_system_nonblocking(self):
        p = AudioPlayer(AudioConfig(fallback_to_system=True))
        with patch.object(p, "_executor") as me:
            assert p.play_sound(None, blocking=False) is True
        p.cleanup()

    def test_play_sound_no_fallback(self):
        p = AudioPlayer(AudioConfig(fallback_to_system=False))
        assert p.play_sound(None) is False
        p.cleanup()

    def test_play_sound_volume_clamp(self, tmp_path):
        p = AudioPlayer(AudioConfig())
        f = tmp_path / "t.wav"
        f.write_text("x")
        with patch.object(p, "_play_file_sync", return_value=True):
            assert p.play_sound(f, volume=5.0, blocking=True) is True
            assert p.play_sound(f, volume=-1.0, blocking=True) is True
        p.cleanup()

    def test_play_file_sync_max_concurrent(self):
        p = AudioPlayer(AudioConfig(max_concurrent_sounds=1))
        p._active_sounds.add("x")
        assert p._play_file_sync(Path("/tmp/t.wav"), 0.5) is False
        p.cleanup()

    def test_play_file_sync_success(self, tmp_path):
        p = AudioPlayer(AudioConfig())
        f = tmp_path / "t.wav"
        f.write_text("x")
        with patch.object(p, "_execute_playback", return_value=True):
            assert p._play_file_sync(f, 0.5) is True
        assert str(f) not in p._active_sounds
        p.cleanup()

    def test_play_file_sync_exception(self):
        p = AudioPlayer(AudioConfig())
        with patch.object(p, "_execute_playback", side_effect=Exception):
            assert p._play_file_sync(Path("/tmp/t.wav"), 0.5) is False
        p.cleanup()

    def test_play_system_sound_sync_max(self):
        p = AudioPlayer(AudioConfig(max_concurrent_sounds=1))
        p._active_sounds.add("x")
        assert p._play_system_sound_sync(NotificationLevel.INFO, 0.5) is False
        p.cleanup()

    def test_play_system_sound_sync_ok(self):
        p = AudioPlayer(AudioConfig())
        with patch.object(p, "_execute_system_sound", return_value=True):
            assert p._play_system_sound_sync(NotificationLevel.INFO, 0.5) is True
        p.cleanup()

    def test_play_system_sound_sync_err(self):
        p = AudioPlayer(AudioConfig())
        with patch.object(p, "_execute_system_sound", side_effect=Exception):
            assert p._play_system_sound_sync(NotificationLevel.INFO, 0.5) is False
        p.cleanup()

    def test_execute_playback_linux(self):
        p = AudioPlayer(AudioConfig())
        p.system = "Linux"
        with patch.object(p, "_play_linux", return_value=True):
            assert p._execute_playback(Path("/t.wav"), 0.5) is True
        p.cleanup()

    def test_execute_playback_darwin(self):
        p = AudioPlayer(AudioConfig())
        p.system = "Darwin"
        with patch.object(p, "_play_macos", return_value=True):
            assert p._execute_playback(Path("/t.wav"), 0.5) is True
        p.cleanup()

    def test_execute_playback_windows(self):
        p = AudioPlayer(AudioConfig())
        p.system = "Windows"
        with patch.object(p, "_play_windows", return_value=True):
            assert p._execute_playback(Path("/t.wav"), 0.5) is True
        p.cleanup()

    def test_execute_playback_unsupported(self):
        p = AudioPlayer(AudioConfig())
        p.system = "FreeBSD"
        assert p._execute_playback(Path("/t.wav"), 0.5) is False
        p.cleanup()

    @patch("subprocess.run")
    def test_play_linux_paplay(self, mock_run):
        p = AudioPlayer(AudioConfig())
        assert p._play_linux(Path("/t.wav"), 0.7) is True
        p.cleanup()

    @patch("subprocess.run", side_effect=[FileNotFoundError("no paplay"), MagicMock()])
    def test_play_linux_aplay_fallback(self, mock_run):
        p = AudioPlayer(AudioConfig())
        assert p._play_linux(Path("/t.wav"), 0.7) is True
        p.cleanup()

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_play_linux_fail(self, mock_run):
        p = AudioPlayer(AudioConfig())
        assert p._play_linux(Path("/t.wav"), 0.7) is False
        p.cleanup()

    @patch("subprocess.run")
    def test_play_linux_with_device(self, mock_run):
        p = AudioPlayer(AudioConfig(audio_device="sink0"))
        assert p._play_linux(Path("/t.wav"), 0.7) is True
        p.cleanup()

    @patch("subprocess.run")
    def test_play_macos_ok(self, mock_run):
        p = AudioPlayer(AudioConfig())
        assert p._play_macos(Path("/t.wav"), 0.7) is True
        p.cleanup()

    @patch("subprocess.run", side_effect=Exception)
    def test_play_macos_fail(self, mock_run):
        p = AudioPlayer(AudioConfig())
        assert p._play_macos(Path("/t.wav"), 0.7) is False
        p.cleanup()

    def test_play_windows_winsound(self):
        p = AudioPlayer(AudioConfig())
        ws = MagicMock()
        ws.SND_FILENAME = 1
        ws.SND_NOWAIT = 2
        with patch.dict("sys.modules", {"winsound": ws}):
            assert p._play_windows(Path("/t.wav"), 0.7) is True
        p.cleanup()

    def test_execute_system_sound_routing(self):
        p = AudioPlayer(AudioConfig())
        for sys_name, method in [
            ("Linux", "_play_system_linux"),
            ("Darwin", "_play_system_macos"),
            ("Windows", "_play_system_windows"),
        ]:
            p.system = sys_name
            with patch.object(p, method, return_value=True):
                assert p._execute_system_sound(NotificationLevel.INFO, 0.5) is True
        p.system = "Other"
        assert p._execute_system_sound(NotificationLevel.INFO, 0.5) is False
        p.cleanup()

    def test_play_system_linux_found(self):
        p = AudioPlayer(AudioConfig())
        p.system = "Linux"
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(p, "_play_linux", return_value=True):
                assert p._play_system_linux(NotificationLevel.INFO, 0.5) is True
        p.cleanup()

    def test_play_system_linux_not_found(self):
        p = AudioPlayer(AudioConfig())
        p.system = "Linux"
        with patch("pathlib.Path.exists", return_value=False):
            assert p._play_system_linux(NotificationLevel.INFO, 0.5) is False
        p.cleanup()

    def test_play_system_macos_found(self):
        p = AudioPlayer(AudioConfig())
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(p, "_play_macos", return_value=True):
                assert p._play_system_macos(NotificationLevel.WARNING, 0.5) is True
        p.cleanup()

    def test_play_system_macos_not_found(self):
        p = AudioPlayer(AudioConfig())
        with patch("pathlib.Path.exists", return_value=False):
            assert p._play_system_macos(NotificationLevel.INFO, 0.5) is False
        p.cleanup()

    def test_play_system_windows_ok(self):
        p = AudioPlayer(AudioConfig())
        ws = MagicMock()
        ws.MB_OK = 0
        ws.MB_ICONEXCLAMATION = 1
        ws.MB_ICONHAND = 2
        with patch.dict("sys.modules", {"winsound": ws}):
            assert p._play_system_windows(NotificationLevel.CRITICAL, 0.5) is True
        p.cleanup()

    @patch("subprocess.run")
    def test_get_linux_devices_ok(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="0\tsink0\tmod\n1\tsink1\tmod\n"
        )
        p = AudioPlayer(AudioConfig())
        p.system = "Linux"
        devs = p._get_linux_devices()
        assert len(devs) >= 1
        assert devs[0].is_default is True
        p.cleanup()

    @patch("subprocess.run", side_effect=Exception)
    def test_get_linux_devices_err(self, mock_run):
        p = AudioPlayer(AudioConfig())
        p.system = "Linux"
        assert p._get_linux_devices() == []
        p.cleanup()

    def test_get_macos_devices(self):
        p = AudioPlayer(AudioConfig())
        devs = p._get_macos_devices()
        assert len(devs) == 1
        p.cleanup()

    def test_get_windows_devices(self):
        p = AudioPlayer(AudioConfig())
        devs = p._get_windows_devices()
        assert len(devs) == 1
        p.cleanup()

    def test_get_available_devices_default(self):
        p = AudioPlayer(AudioConfig())
        p.system = "UnknownOS"
        devs = p.get_available_devices()
        assert len(devs) >= 1
        p.cleanup()

    def test_test_audio_ok(self):
        p = AudioPlayer(AudioConfig())
        with patch.object(p, "_play_system_sound_sync", return_value=True):
            assert p.test_audio_playback() is True
        p.cleanup()

    def test_test_audio_fail(self):
        p = AudioPlayer(AudioConfig())
        with patch.object(p, "_play_system_sound_sync", side_effect=Exception):
            assert p.test_audio_playback() is False
        p.cleanup()

    def test_stop_all_sounds(self):
        p = AudioPlayer(AudioConfig())
        p._active_sounds.add("x")
        p.stop_all_sounds()
        assert len(p._active_sounds) == 0
        p.cleanup()


class TestAudioManager:
    def test_init(self):
        m = AudioManager()
        assert m.config.enabled is True
        m.cleanup()

    def test_play_notification_custom(self, tmp_path):
        m = AudioManager()
        f = tmp_path / "c.wav"
        f.write_text("x")
        with patch.object(m.player, "play_sound", return_value=True):
            assert m.play_notification_sound(NotificationLevel.INFO, custom_sound=f) is True
        m.cleanup()

    def test_play_notification_default(self, tmp_path):
        m = AudioManager()
        f = tmp_path / "d.wav"
        f.write_text("x")
        m.default_sounds[NotificationLevel.INFO] = f
        with patch.object(m.player, "play_sound", return_value=True):
            assert m.play_notification_sound(NotificationLevel.INFO) is True
        m.cleanup()

    def test_play_notification_system(self):
        m = AudioManager()
        with patch.object(m.player, "play_sound", return_value=True):
            assert m.play_notification_sound(NotificationLevel.WARNING) is True
        m.cleanup()

    def test_add_and_apply_theme(self):
        m = AudioManager()
        theme = {NotificationLevel.INFO: Path("/t/i.wav")}
        m.add_sound_theme("th", theme)
        assert m.apply_sound_theme("th") is True
        assert m.apply_sound_theme("bad") is False
        m.cleanup()

    def test_get_audio_info(self):
        m = AudioManager()
        with patch.object(m.player, "get_available_devices", return_value=[]):
            info = m.get_audio_info()
            assert "enabled" in info
        m.cleanup()

    def test_test_audio(self):
        m = AudioManager()
        with patch.object(m.player, "test_audio_playback", return_value=True):
            with patch.object(m.player, "play_sound", return_value=True):
                res = m.test_audio()
                assert "system_sound" in res
        m.cleanup()

    def test_reload_config(self):
        m = AudioManager()
        nc = AudioConfig(volume=0.3)
        m.reload_config(nc)
        assert m.config.volume == 0.3
        m.cleanup()
