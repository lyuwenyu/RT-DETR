"""Source classification for the capture layer.

Pure logic, so it is testable without a camera or a network. The distinction
matters: a live source is read on a thread that discards stale frames, while a
file is read in lockstep. Getting a phone stream classified as a file would
make guidance fall progressively further behind real time.
"""

import pytest

from src.assist.io.camera import classify_source


@pytest.mark.parametrize('source', [0, 1, '0', '2'])
def test_webcam_indices_are_devices(source):
    src, is_device, is_stream = classify_source(source)
    assert is_device and not is_stream
    assert isinstance(src, int)


@pytest.mark.parametrize('source', [
    'walk.mp4', 'C:/videos/a.mov', './clip.avi', 'some_file',
])
def test_paths_are_files(source):
    src, is_device, is_stream = classify_source(source)
    assert not is_device and not is_stream
    assert src == source


@pytest.mark.parametrize('source', [
    'http://192.168.1.5:8080/video',
    'https://192.168.1.5:8080/video',
    'rtsp://10.0.0.7:8554/live',
    'rtmp://host/app/key',
    'udp://239.0.0.1:1234',
    'tcp://10.0.0.2:5000',
])
def test_network_urls_are_live_streams(source):
    src, is_device, is_stream = classify_source(source)
    assert is_stream and not is_device
    assert src == source          # must stay a string, never int()


def test_scheme_match_is_case_insensitive():
    _, _, is_stream = classify_source('HTTP://192.168.1.5:8080/video')
    assert is_stream


def test_a_stream_url_is_never_read_in_lockstep():
    """Regression: a URL is not a digit, so it used to be treated as a file."""
    _, is_device, is_stream = classify_source('http://192.168.1.5:8080/video')
    assert (is_device or is_stream), 'stream must be live, not lockstep'
