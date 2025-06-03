"""Helper functions for test assertions in pipecat-flows tests."""


def assert_tts_speak_frames_queued(mock_task, expected_texts):
    """Assert that TTSSpeakFrames with expected texts were queued."""
    from pipecat.frames.frames import TTSSpeakFrame

    tts_calls = [
        call
        for call in mock_task.queue_frame.call_args_list
        if isinstance(call[0][0], TTSSpeakFrame)
    ]
    assert len(tts_calls) == len(expected_texts), (
        f"Expected {len(expected_texts)} TTS calls, got {len(tts_calls)}"
    )
    for text in expected_texts:
        assert any(text in getattr(call[0][0], "text", "") for call in tts_calls), (
            f"{text} TTS call not found"
        )


def assert_end_frame_queued(mock_task):
    """Assert that an EndFrame was queued."""
    from pipecat.frames.frames import EndFrame

    end_calls = [
        call for call in mock_task.queue_frame.call_args_list if isinstance(call[0][0], EndFrame)
    ]
    assert len(end_calls) == 1, "EndFrame not queued"
