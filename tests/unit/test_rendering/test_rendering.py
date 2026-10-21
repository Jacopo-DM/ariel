"""Test: renderers and the video recorder, with a stand-in for OpenGL."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self

import cv2
import mujoco
import numpy as np
import pytest
from PIL import Image

from ariel.rendering import VideoRecorder, renderers

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

WIDTH = 32
HEIGHT = 24
FPS = 30
DURATION = 0.1
MODEL_XML = """
<mujoco>
    <worldbody>
    <camera name="ortho-cam" pos="0 -2 1" fovy="45"/>
    <body name="robot-core" pos="0 0 1"><freejoint/>
        <geom size=".1"/></body>
    </worldbody>
</mujoco>"""
NO_CAMERA_XML = """
<mujoco>
    <worldbody>
    <body name="{name}" pos="0 0 1"><freejoint/><geom size=".1"/></body>
    </worldbody>
</mujoco>"""
FOVY = 60.0
TWO_FRAMES = 2


pytestmark = pytest.mark.usefixtures("fake_opengl")


class _FakeRenderer:
    """Draws black frames instead of using OpenGL."""

    # Every camera passed to update_scene, across instances (reset per test)
    cameras: ClassVar[list[object]] = []

    def __init__(self, model: mujoco.MjModel, width: int, height: int) -> None:
        del model
        self.shape = (height, width, 3)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def update_scene(
        self,
        data: mujoco.MjData,
        scene_option: object = None,
        camera: object = None,
    ) -> None:
        del data, scene_option
        self.cameras.append(camera)  # the shared class list

    def render(self) -> NDArray[np.uint8]:
        return np.zeros(self.shape, dtype=np.uint8)


@pytest.fixture
def fake_opengl(monkeypatch: pytest.MonkeyPatch) -> None:
    # Replace the OpenGL renderer; no backend works on every machine.
    monkeypatch.setattr(mujoco, "Renderer", _FakeRenderer)
    monkeypatch.setattr(_FakeRenderer, "cameras", [])


def _scene(xml: str = MODEL_XML) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(xml)
    return model, mujoco.MjData(model)


def _recorder(folder: Path) -> VideoRecorder:
    return VideoRecorder(
        file_name="clip", output_folder=folder, width=WIDTH, height=HEIGHT
    )


# --- video recorder --------------------------------------------------------- #


def test_recorder_writes_a_video_file(tmp_path: Path) -> None:
    """Frames written to the recorder end up in an mp4 file."""
    recorder = _recorder(tmp_path)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    recorder.write(frame)
    recorder.write(frame)
    recorder.release()
    assert recorder.frame_count == TWO_FRAMES
    (video,) = list(tmp_path.glob("clip_*.mp4"))
    assert video.stat().st_size > 0


def test_recorder_defaults_to_the_current_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an output folder the video goes to the working folder."""
    monkeypatch.chdir(tmp_path)
    recorder = VideoRecorder(width=WIDTH, height=HEIGHT)
    recorder.release()
    assert list(tmp_path.glob("video_*.mp4"))


class _FakeWriter:
    """A cv2.VideoWriter that opens only for chosen file endings."""

    openable: tuple[str, ...] = ()

    def __init__(
        self, path: object, fourcc: int, fps: int, size: tuple[int, int]
    ) -> None:
        del fourcc, fps, size
        self.path = str(path)

    def isOpened(self) -> bool:  # ruff: ignore[invalid-function-name]  # mirrors cv2.VideoWriter
        return self.path.endswith(self.openable)


def test_recorder_falls_back_to_another_codec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When mp4v fails the first fallback that opens is used."""
    monkeypatch.setattr(_FakeWriter, "openable", (".avi",))
    monkeypatch.setattr(cv2, "VideoWriter", _FakeWriter)
    recorder = _recorder(tmp_path)
    assert isinstance(recorder.video_writer, _FakeWriter)
    assert recorder.video_writer.path.endswith(".avi")


def test_recorder_reports_when_no_codec_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no usable codec the recorder says so."""
    monkeypatch.setattr(_FakeWriter, "openable", ("nothing",))
    monkeypatch.setattr(cv2, "VideoWriter", _FakeWriter)
    with pytest.raises(RuntimeError, match="Could not initialize"):
        _recorder(tmp_path)


# --- single frame ----------------------------------------------------------- #


def test_single_frame_without_a_camera() -> None:
    """MuJoCo's free camera (-1) is used if the model has no ortho-cam.

    ``mujoco.Renderer.update_scene`` takes -1 for the free camera and
    raises ``TypeError`` for ``None`` (MuJoCo 3.16).
    """
    model, data = _scene(NO_CAMERA_XML.format(name="b"))
    image = renderers.single_frame_renderer(
        model, data, width=WIDTH, height=HEIGHT
    )
    assert image.size == (WIDTH, HEIGHT)
    assert _FakeRenderer.cameras == [-1]


def test_single_frame_with_a_camera_override() -> None:
    """An ortho-cam gets the requested field of view and sensor size."""
    model, data = _scene()
    renderers.single_frame_renderer(
        model, data, width=WIDTH, height=HEIGHT, cam_fovy=FOVY
    )
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ortho-cam")
    assert model.cam_fovy[camera] == pytest.approx(FOVY)
    assert list(model.cam_sensorsize[camera]) == [WIDTH, HEIGHT]


def test_single_frame_saves_and_shows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """save=True writes a png; show=True opens the viewer."""
    shown: list[bool] = []
    monkeypatch.setattr(Image.Image, "show", lambda _self: shown.append(True))
    monkeypatch.chdir(tmp_path)
    model, data = _scene()
    explicit = tmp_path / "explicit.png"
    renderers.single_frame_renderer(
        model, data, width=WIDTH, height=HEIGHT, save=True, save_path=explicit
    )
    renderers.single_frame_renderer(
        model, data, width=WIDTH, height=HEIGHT, save=True, show=True
    )
    assert explicit.exists()
    assert list(tmp_path.glob("img_*.png"))
    assert shown == [True]


# --- videos ----------------------------------------------------------------- #


def test_video_renderer_writes_frames(tmp_path: Path) -> None:
    """The video renderer steps the model and writes each frame."""
    model, data = _scene()
    recorder = _recorder(tmp_path)
    renderers.video_renderer(
        model, data, duration=DURATION, video_recorder=recorder, cam_fovy=FOVY
    )
    assert recorder.frame_count > 0


def test_video_renderer_with_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a recorder one is created in the working folder."""
    monkeypatch.chdir(tmp_path)
    model, data = _scene(NO_CAMERA_XML.format(name="b"))
    renderers.video_renderer(model, data, duration=DURATION)
    assert list(tmp_path.glob("video_*.mp4"))


@pytest.mark.parametrize("body", ["robot-core", "my-core-body", "plain"])
def test_tracking_renderer_finds_the_core(tmp_path: Path, body: str) -> None:
    """The camera tracks the named body, a core-like body, or nothing."""
    model, data = _scene(NO_CAMERA_XML.format(name=body))
    recorder = _recorder(tmp_path)
    renderers.tracking_video_renderer(
        model, data, duration=DURATION, video_recorder=recorder
    )
    assert recorder.frame_count > 0


def test_tracking_renderer_with_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a recorder one is created in the working folder."""
    monkeypatch.chdir(tmp_path)
    model, data = _scene()
    renderers.tracking_video_renderer(model, data, duration=DURATION)
    assert list(tmp_path.glob("video_*.mp4"))


def test_recorder_accepts_a_string_folder(tmp_path: Path) -> None:
    """The output folder may be given as a string."""
    recorder = VideoRecorder(
        output_folder=str(tmp_path / "videos"), width=WIDTH, height=HEIGHT
    )
    recorder.release()
    assert list((tmp_path / "videos").glob("video_*.mp4"))


def test_recorder_without_a_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the timestamp switched off the file name is exact."""
    monkeypatch.setattr(VideoRecorder, "_add_timestamp_to_file_name", False)
    recorder = _recorder(tmp_path)
    recorder.release()
    assert (tmp_path / "clip.mp4").exists()
