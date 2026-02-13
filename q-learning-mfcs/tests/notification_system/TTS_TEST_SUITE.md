# TTS Notification Test Suite Documentation
## Overview

This comprehensive Test-Driven Development (TDD) test suite provides complete coverage for the Text-to-Speech (TTS) notification functionality in the MFC monitoring system. The suite follows TDD principles by implementing tests before the actual TTS functionality.

## Test Structure

```
tests/notification_system/
├── unit/
│   └── handlers/
│       └── test_tts_handler.py          # Comprehensive TTS unit tests
├── integration/
│   └── test_tts_integration.py          # Integration & performance tests
├── run_tts_tests.py                     # TTS-specific test runner
├── pytest-tts.ini                      # TTS pytest configuration
└── TTS_TEST_SUITE.md                   # This documentation
```
## Features Tested

### 1. TTS Engine Interface & Mocking Infrastructure

- **TTSConfig**: Configuration management with sensible defaults
- **TTSEngine**: Mock-able TTS engine interface
- **TTSNotificationHandler**: Main handler implementing NotificationInterface
- **Platform Detection**: Linux, Windows, macOS compatibility

**Test Classes:**
- `TestTTSConfig`: Configuration validation and defaults
- `TestTTSEngine`: Engine initialization and voice management
- `TestTTSNotificationHandler`: Core handler functionality

### 2. Voice Synthesis with Mocked pyttsx3

- **Engine Initialization**: Success/failure scenarios
- **Voice Management**: Getting available voices, setting voice parameters
- **Speech Synthesis**: Text-to-speech conversion with timing controls
- **File Output**: Saving speech to audio files

**Key Tests:**
- `test_tts_engine_initialization_success/failure`
- `test_tts_engine_voice_management`
- `test_tts_engine_speech_synthesis`
- `test_tts_engine_save_to_file`

### 3. Audio Toggle Functionality (TTS vs Ding)

- **TTS Enabled**: Uses speech synthesis when available
- **TTS Disabled**: Falls back to ding sounds
- **Automatic Fallback**: Switches to ding when TTS fails
- **Configuration Control**: Toggle behavior via config

**Test Classes:**
- `TestTTSAudioToggle`: Complete toggle functionality testing

### 4. Platform-Specific Voice Selection

- **Linux**: espeak and speech-dispatcher integration
- **Windows**: SAPI voice selection and configuration
- **macOS**: NSSpeechSynthesizer voice management
- **Cross-Platform**: Unified interface across platforms

**Test Classes:**
- `TestPlatformVoiceSelection`: Platform-specific voice handling

### 5. Queue Management for Concurrent Notifications

- **FIFO Processing**: First-in-first-out notification order
- **Thread Safety**: Concurrent access protection
- **Queue Limits**: Size management and overflow handling
- **Worker Threads**: Background processing management

**Test Classes:**
- `TestTTSQueueManagement`: Queue operations and concurrency
- `TestTTSThreadSafety`: Thread safety verification

### 6. Error Handling and Fallback Scenarios

- **Engine Crashes**: Recovery from TTS engine failures
- **Import Errors**: Graceful handling of missing pyttsx3
- **Timeout Handling**: Speech synthesis timeout management
- **Memory Pressure**: Behavior under resource constraints

**Test Classes:**
- `TestTTSErrorHandling`: Comprehensive error scenarios
- `TestTTSErrorRecovery`: System resilience testing

### 7. Text Processing and Sanitization

- **Length Limits**: Message truncation for long text
- **Special Characters**: Conversion for speech synthesis
- **Number Verbalization**: Converting digits to spoken words
- **HTML Removal**: Cleaning markup from messages

**Test Classes:**
- `TestTTSTextProcessing`: Text sanitization and preparation

### 8. Performance and Integration Tests

- **Latency Benchmarks**: Synthesis timing requirements
- **Memory Usage**: Resource consumption monitoring
- **Concurrent Load**: Multi-threaded performance testing
- **End-to-End**: Complete notification pipeline testing

**Test Classes:**
- `TestTTSPerformance`: Performance benchmarking
- `TestTTSEndToEndIntegration`: Complete system integration
## Performance Requirements

The test suite enforces the following performance benchmarks:

| Metric | Target | Test Coverage |
|--------|--------|---------------|
| Synthesis Latency (95th percentile) | < 500ms | ✅ |
| Memory Usage (Peak) | < 100MB | ✅ |
| Concurrent Throughput | > 10 notifications/sec | ✅ |
| Queue Processing Rate | > 20 items/sec | ✅ |
| Thread Safety | 100% safe | ✅ |
| Error Recovery | < 1s fallback | ✅ |
## Running the Tests

### Prerequisites

```bash
# Ensure test environment is set up
cd /home/uge/mfc-project/q-learning-mfcs/tests/notification_system

# Install test dependencies (if not already installed)
pixi install
```

### Test Execution Options

#### 1. Run All TTS Tests
```bash
python run_tts_tests.py --coverage --verbose
```

#### 2. Unit Tests Only
```bash
python run_tts_tests.py --unit-only --verbose
```

#### 3. Integration Tests Only
```bash
python run_tts_tests.py --integration --verbose
```

#### 4. Performance Benchmarks
```bash
python run_tts_tests.py --performance --verbose
```

#### 5. CI/CD Mode
```bash
python run_tts_tests.py --ci --coverage --lint
```

#### 6. Quick Development Testing
```bash
python run_tts_tests.py --unit-only --fail-fast
```

### Using Pytest Directly

```bash
# Run with TTS-specific configuration
pytest -c pytest-tts.ini -v -m tts

# Run specific test categories
pytest -m "tts and unit" -v
pytest -m "tts and performance" -v
pytest -m "tts and error_handling" -v

# Run with coverage
pytest -m tts --cov=../../../src/notifications --cov-report=html
```
## Test Markers

The test suite uses the following pytest markers for categorization:

- `@pytest.mark.tts`: All TTS-related tests
- `@pytest.mark.unit`: Unit tests for individual components
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.performance`: Performance and benchmark tests
- `@pytest.mark.audio`: Audio-related tests (requires mocking)
- `@pytest.mark.mock_required`: Tests requiring extensive mocking
- `@pytest.mark.cross_platform`: Cross-platform compatibility tests
- `@pytest.mark.pyttsx3`: pyttsx3-specific integration tests
- `@pytest.mark.queue`: Queue management tests
- `@pytest.mark.fallback`: Fallback mechanism tests
- `@pytest.mark.voice_selection`: Voice selection tests
- `@pytest.mark.error_handling`: Error handling tests
- `@pytest.mark.text_processing`: Text processing tests
## Coverage Requirements

The test suite aims for 100% code coverage with the following targets:

- **Line Coverage**: 100%
- **Branch Coverage**: 100%
- **Function Coverage**: 100%
- **Class Coverage**: 100%

### Coverage Reports

Coverage reports are generated in multiple formats:

1. **HTML Report**: `../../../htmlcov/tts/index.html`
2. **XML Report**: `../../../coverage-tts.xml`
3. **Terminal Report**: Displayed during test execution
## CI/CD Integration

### GitHub Actions / GitLab CI

```yaml
# Example CI configuration
tts_tests:
  script:
    - cd q-learning-mfcs/tests/notification_system
    - python run_tts_tests.py --ci --coverage --lint
  artifacts:
    reports:
      junit: q-learning-mfcs/tests/notification_system/tts-test-results.xml
      coverage_report:
        coverage_format: cobertura
        path: q-learning-mfcs/coverage-tts.xml
    paths:
      - q-learning-mfcs/htmlcov/tts/
  coverage: '/TOTAL.*\s+(\d+%)/'
```

### Pixi Integration

Add to `pixi.toml`:

```toml
[tasks]
test-tts = "cd q-learning-mfcs/tests/notification_system && python run_tts_tests.py"
test-tts-unit = "cd q-learning-mfcs/tests/notification_system && python run_tts_tests.py --unit-only"
test-tts-coverage = "cd q-learning-mfcs/tests/notification_system && python run_tts_tests.py --coverage"
test-tts-ci = "cd q-learning-mfcs/tests/notification_system && python run_tts_tests.py --ci --coverage --lint"
```
## Mock Infrastructure

The test suite provides comprehensive mocking for:

### 1. pyttsx3 Engine
```python
@pytest.fixture
def mock_pyttsx3():
    with patch('pyttsx3.init') as mock_init:
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        yield mock_engine
```

### 2. Platform Detection
```python
@pytest.fixture
def mock_platform_system():
    def platform_mocker(platform_name):
        with patch('platform.system', return_value=platform_name):
            yield platform_name
    return platform_mocker
```

### 3. Audio System
```python
@pytest.fixture
def mock_audio_system():
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        yield mock_run
```

### 4. File System
```python
@pytest.fixture
def mock_sound_files(tmp_path):
    sounds_dir = tmp_path / "sounds"
    sounds_dir.mkdir()
    # Create mock sound files
    return sounds_dir
```
## Test Data and Fixtures

### Sample Notifications
```python
@pytest.fixture
def sample_notification():
    return NotificationData(
        event_type="task_completion",
        agent_id="TDD-Agent-Beta",
        task_id="test-task-123",
        status="success",
        message="TTS test completed successfully",
        timestamp=datetime.now()
    )
```

### Voice Data
```python
@pytest.fixture
def sample_voices():
    return [
        {"id": "voice1", "name": "English (US) Female", "lang": "en-US"},
        {"id": "voice2", "name": "English (US) Male", "lang": "en-US"},
        {"id": "voice3", "name": "Spanish (ES) Female", "lang": "es-ES"},
    ]
```
## Debugging and Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Platform Issues**: Tests use mocking to avoid platform dependencies
3. **Audio Conflicts**: CI mode disables actual audio output
4. **Timeout Errors**: Adjust timeout values in configuration

### Debug Mode

```bash
# Enable verbose debugging
python run_tts_tests.py --verbose --fail-fast

# Run specific test with debugging
pytest unit/handlers/test_tts_handler.py::TestTTSConfig::test_tts_config_default_values -v -s

# Enable pytest debugging
pytest --pdb --pdbcls=IPython.terminal.debugger:Pdb
```

### Performance Profiling

```bash
# Profile test execution
python -m cProfile -s cumtime run_tts_tests.py --performance

# Memory profiling
python -m memory_profiler run_tts_tests.py --integration
```
## Contributing

### Adding New Tests

1. **Follow TDD Principles**: Write tests before implementation
2. **Use Appropriate Markers**: Tag tests with relevant markers
3. **Mock External Dependencies**: Avoid real TTS engines in tests
4. **Update Documentation**: Keep this README current
5. **Maintain Coverage**: Ensure 100% coverage is maintained

### Test Naming Conventions

- Test files: `test_tts_*.py`
- Test classes: `TestTTS*` or `*TTSTest`
- Test methods: `test_tts_*` or `test_*_tts`
- Fixtures: `*_tts_*` or `tts_*`

### Code Quality

- **Linting**: Run `python run_tts_tests.py --lint`
- **Type Checking**: Ensure mypy passes
- **Documentation**: Include comprehensive docstrings
- **Error Messages**: Provide clear, actionable error messages
## Implementation Notes

This test suite is designed using Test-Driven Development principles, meaning:

1. **Tests First**: All tests are written before implementation
2. **Red-Green-Refactor**: Standard TDD cycle
3. **Comprehensive Coverage**: Every code path is tested
4. **Mock Everything**: No external dependencies in tests
5. **Performance Focus**: Benchmarks built into tests

The actual TTS implementation should be built to satisfy these tests, ensuring:

- Robust error handling
- Cross-platform compatibility
- High performance
- Thread safety
- Comprehensive configuration options
## Future Enhancements

Potential additions to the test suite:

1. **Voice Quality Tests**: Synthesis quality benchmarks
2. **Accessibility Tests**: Screen reader compatibility
3. **Internationalization**: Multi-language support tests
4. **Network Tests**: Remote TTS service integration
5. **Security Tests**: Input sanitization and validation
6. **Stress Tests**: Extended load testing scenarios

## References

- [pytest Documentation](https://docs.pytest.org/)
- [pyttsx3 Documentation](https://pyttsx3.readthedocs.io/)
- [TDD Best Practices](https://testdriven.io/blog/modern-tdd/)
- [Python Testing Best Practices](https://realpython.com/python-testing/)

---

**Last Updated**: August 2025  
**Test Suite Version**: 1.0.0  
**Coverage Target**: 100%  
**Python Version**: 3.12+