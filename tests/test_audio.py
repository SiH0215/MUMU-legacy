from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mumu.errors import ServiceError
from mumu.stt import WhisperTranscriber, record_audio
from mumu.tts import SupertoneSynthesizer, play_audio


def response(content_type, body=b"audio", data=None):
    result = Mock()
    result.headers = {"Content-Type": content_type}
    result.content = body
    result.json.return_value = data
    return result


class AudioTests(unittest.TestCase):
    def test_audio_body_and_single_post(self):
        session = Mock()
        session.post.return_value = response("audio/wav")
        audio = SupertoneSynthesizer("dummy", "voice", session=session).synthesize("안녕")
        self.assertEqual(audio, b"audio")
        session.post.assert_called_once()
        self.assertEqual(session.post.call_args.kwargs["timeout"], 30)
        session.get.assert_not_called()

    def test_audio_url_download_has_no_api_key_header(self):
        session = Mock()
        session.post.return_value = response("application/json", data={"audio_url": "https://audio.example/test.wav"})
        session.get.return_value = response("audio/wav", body=b"downloaded")
        self.assertEqual(SupertoneSynthesizer("dummy", "voice", session=session).synthesize("안녕"), b"downloaded")
        session.get.assert_called_once_with("https://audio.example/test.wav", timeout=30)

    def test_rejects_error_response_missing_url_and_empty_audio(self):
        invalid = [response("text/html"), response("application/json", data={}), response("audio/wav", body=b"")]
        http_error = response("audio/wav")
        http_error.raise_for_status.side_effect = RuntimeError("HTTP failure")
        invalid.append(http_error)
        for value in invalid:
            with self.subTest(value=value):
                session = Mock()
                session.post.return_value = value
                with self.assertRaises(ServiceError):
                    SupertoneSynthesizer("dummy", "voice", session=session).synthesize("안녕")

    def test_recording_resources_close_on_read_failure(self):
        stream = Mock()
        stream.read.side_effect = RuntimeError("microphone disconnected")
        device = Mock()
        device.get_sample_size.return_value = 2
        device.open.return_value = stream
        fake = SimpleNamespace(paInt16=8, PyAudio=lambda: device)
        with patch.dict("sys.modules", {"pyaudio": fake}):
            with self.assertRaises(ServiceError):
                record_audio()
        stream.stop_stream.assert_called_once()
        stream.close.assert_called_once()
        device.terminate.assert_called_once()

    def test_recording_wav_is_removed_on_transcription_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "recorded.wav"
            path.write_bytes(b"audio")
            transcriber = WhisperTranscriber()
            with patch("mumu.stt.record_audio", return_value=path), patch.object(
                transcriber, "transcribe_file", side_effect=ServiceError("STT failed")
            ):
                with self.assertRaises(ServiceError):
                    transcriber.record_and_transcribe()
            self.assertFalse(path.exists())

    def test_playback_temp_file_is_removed_on_failure(self):
        seen = []
        music = Mock()

        def fail_load(path):
            seen.append(Path(path))
            self.assertTrue(Path(path).exists())
            raise RuntimeError("playback failed")

        music.load.side_effect = fail_load
        mixer = Mock(music=music)
        with patch.dict("sys.modules", {"pygame": SimpleNamespace(mixer=mixer)}):
            with self.assertRaises(ServiceError):
                play_audio(b"audio")
        mixer.quit.assert_called_once()
        self.assertFalse(seen[0].exists())


if __name__ == "__main__":
    unittest.main()
