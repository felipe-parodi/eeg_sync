from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

import portal.app as portal_app
from portal.app import build_submission_metadata, submission_error_page
from portal.app import index as render_index
from portal.jobs import (
    CompressionSettings,
    ExclusionWindow,
    PortalJobConfig,
    SessionBlock,
    assemble_chunked_upload,
    build_ffmpeg_compression_command,
    build_processing_steps,
    chunk_path_for,
    final_upload_path_for,
    make_result_zip,
    mark_chunk_received,
    parse_timecode,
    split_blocks_for_exclusions,
    validate_direct_chunked_upload,
    write_status,
)


def test_upload_page_exposes_drag_and_drop_targets() -> None:
    response = render_index(None)
    text = response.body.decode("utf-8")

    assert response.status_code == 200
    assert "Video Processing Portal" in text
    assert 'data-file-input="video_a"' in text
    assert 'data-file-input="video_b"' in text
    assert (
        'id="video_a" class="file-input-native" name="video_a" type="file" accept='
        in text
    )
    assert (
        'id="video_b" class="file-input-native" name="video_b" type="file" accept='
        in text
    )
    assert "Drop GoPro Video A here" in text
    assert "Drop GoPro Video B here" in text
    assert "Upload at least one GoPro video" in text
    assert "Video A Analysis Blocks" in text
    assert "Video B Analysis Blocks" in text
    assert 'class="info-icon"' in text
    assert "Proximity" in text
    assert "Movement synchrony" in text
    assert "Gaze estimation" in text


def test_portal_password_and_secret_must_be_configured(monkeypatch) -> None:
    monkeypatch.delenv("PORTAL_PASSWORD", raising=False)
    monkeypatch.delenv("PORTAL_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="PORTAL_PASSWORD must be set"):
        portal_app.portal_password()
    with pytest.raises(RuntimeError, match="PORTAL_SECRET_KEY must be set"):
        portal_app.portal_secret_key()


def test_login_sets_expiring_secure_session_cookie(monkeypatch) -> None:
    monkeypatch.setenv("PORTAL_PASSWORD", "test-password")
    monkeypatch.setenv("PORTAL_SECRET_KEY", "test-secret")
    monkeypatch.setenv("PORTAL_SESSION_TTL_SECONDS", "60")
    portal_app.SESSION_TOKENS.clear()
    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))

    response = portal_app.login(request, "test-password")

    cookie = response.headers["set-cookie"].lower()
    assert response.status_code == 303
    assert "httponly" in cookie
    assert "secure" in cookie
    assert "samesite=strict" in cookie
    assert "max-age=60" in cookie
    assert len(portal_app.SESSION_TOKENS) == 1


def test_submission_error_page_escapes_user_controlled_text() -> None:
    response = submission_error_page(ValueError("<script>alert(1)</script>.mov"))
    text = response.body.decode("utf-8")

    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;.mov" in text


def test_job_detail_escapes_status_message(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(portal_app, "JOB_ROOT", tmp_path)
    job_dir = tmp_path / "job-001"
    write_status(
        job_dir,
        {"state": "<b>failed</b>", "message": "<script>alert(1)</script>.mov"},
    )

    response = portal_app.job_detail("job-001", None)
    text = response.body.decode("utf-8")

    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;.mov" in text
    assert "&lt;b&gt;failed&lt;/b&gt;" in text


def test_build_submission_metadata_preserves_raw_and_parsed_inputs() -> None:
    metadata = build_submission_metadata(
        job_id="job-001",
        created_at="2026-05-14T13:00:00",
        safe_session_id="P001c",
        email="researcher@example.edu",
        blocks_text="free_play,1:00,2:00,green",
        exclusions_text="1:20,1:30,experimenter entered",
        video_a_name="gopro_a.mov",
        video_b_name="gopro_b.mp4",
        video_a_size=1024,
        video_b_size=2048,
        include_gaze=True,
    )

    assert metadata["portal_name"] == "Video Processing Portal"
    assert metadata["session_id"] == "P001c"
    assert metadata["email"] == "researcher@example.edu"
    assert metadata["raw_inputs"]["analysis_blocks_text"] == "free_play,1:00,2:00,green"
    assert metadata["raw_inputs"]["analysis_blocks_text_a"] == (
        "free_play,1:00,2:00,green"
    )
    assert metadata["raw_inputs"]["analysis_blocks_text_b"] == (
        "free_play,1:00,2:00,green"
    )
    assert metadata["raw_inputs"]["extra_person_windows_text"] == (
        "1:20,1:30,experimenter entered"
    )
    assert metadata["input_analysis_blocks"][0]["name"] == "free_play"
    assert metadata["extra_person_windows"][0]["reason"] == "experimenter entered"
    assert [block["name"] for block in metadata["analysis_blocks"]] == [
        "free_play_part1",
        "free_play_part2",
    ]
    assert metadata["videos"]["camera_a"]["original_filename"] == "gopro_a.mov"
    assert metadata["videos"]["camera_b"]["size_bytes"] == 2048
    assert metadata["analysis_blocks_by_camera"]["camera_a"][0]["name"] == (
        "free_play_part1"
    )
    assert metadata["analysis_blocks_by_camera"]["camera_b"][0]["name"] == (
        "free_play_part1"
    )
    assert metadata["processing"]["include_gaze"] is True
    assert metadata["processing"]["analysis_window_only"] is True
    assert metadata["processing"]["compression_encoders"] == ["h264_nvenc"]


def test_build_submission_metadata_allows_video_b_to_be_omitted() -> None:
    metadata = build_submission_metadata(
        job_id="job-001",
        created_at="2026-05-14T13:00:00",
        safe_session_id="P001c",
        email="researcher@example.edu",
        blocks_text="free_play,1:00,2:00,green",
        exclusions_text="",
        video_a_name="gopro_a.mov",
        video_b_name="",
        video_a_size=1024,
        video_b_size=0,
        include_gaze=False,
    )

    assert set(metadata["videos"]) == {"camera_a"}
    assert set(metadata["analysis_blocks_by_camera"]) == {"camera_a"}


def test_build_submission_metadata_requires_one_video() -> None:
    try:
        build_submission_metadata(
            job_id="job-001",
            created_at="2026-05-14T13:00:00",
            safe_session_id="P001c",
            email="researcher@example.edu",
            blocks_text="free_play,1:00,2:00,green",
            exclusions_text="",
            video_a_name="",
            video_b_name="",
            video_a_size=0,
            video_b_size=0,
            include_gaze=False,
        )
    except ValueError as error:
        assert "At least one GoPro video" in str(error)
    else:
        raise AssertionError("Expected missing videos to fail")


def test_build_submission_metadata_rejects_oversize_upload(monkeypatch) -> None:
    monkeypatch.setenv("PORTAL_MAX_JOB_UPLOAD_BYTES", "100")

    with pytest.raises(ValueError, match="per-job limit"):
        build_submission_metadata(
            job_id="job-001",
            created_at="2026-05-14T13:00:00",
            safe_session_id="P001c",
            email="researcher@example.edu",
            blocks_text="free_play,1:00,2:00,green",
            exclusions_text="",
            video_a_name="gopro_a.mov",
            video_b_name="",
            video_a_size=101,
            video_b_size=0,
            include_gaze=False,
        )


def test_parse_timecode_accepts_researcher_friendly_formats() -> None:
    assert parse_timecode("83") == 83.0
    assert parse_timecode("13:26") == 806.0
    assert parse_timecode("1:02:03.5") == 3723.5


def test_split_blocks_for_exclusions_removes_extra_person_windows() -> None:
    blocks = [
        SessionBlock(
            name="free_play",
            start_s=100.0,
            end_s=220.0,
            color="green",
        )
    ]
    exclusions = [
        ExclusionWindow(
            start_s=130.0,
            end_s=150.0,
            reason="experimenter entered",
        ),
        ExclusionWindow(
            start_s=190.0,
            end_s=260.0,
            reason="extra person returned",
        ),
    ]

    split = split_blocks_for_exclusions(blocks, exclusions)

    assert [(block.name, block.start_s, block.end_s) for block in split] == [
        ("free_play_part1", 100.0, 130.0),
        ("free_play_part2", 150.0, 190.0),
    ]


def test_build_ffmpeg_command_uses_hardware_encoder_profile() -> None:
    command = build_ffmpeg_compression_command(
        input_path=Path("raw.mov"),
        output_path=Path("compressed.mp4"),
        settings=CompressionSettings(target_fps=8.0, max_width=854, quality=34),
        encoder="h264_nvenc",
    )

    assert command[:5] == ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    assert "-hwaccel" in command
    assert command[command.index("-c:v") + 1] == "h264_nvenc"
    assert "fps=8.0,scale='min(854,iw)':-2" in command
    assert command[-1] == "compressed.mp4"


def test_portal_compression_requires_nvenc_by_default() -> None:
    assert CompressionSettings().encoder_order == ("h264_nvenc",)


def test_build_processing_steps_passes_explicit_pipeline_inputs(tmp_path: Path) -> None:
    job_config = PortalJobConfig(
        job_id="job-001",
        session_id="P001c",
        email="researcher@example.edu",
        video_a=tmp_path / "camera_a_rapid.mp4",
        video_b=tmp_path / "camera_b_rapid.mp4",
        job_dir=tmp_path / "job-001",
        output_root=tmp_path / "output",
        session_config_path=tmp_path / "job-001" / "session_config.json",
        blocks=[
            SessionBlock(
                name="free_play",
                start_s=100.0,
                end_s=220.0,
                color="green",
            )
        ],
    )

    steps = build_processing_steps(job_config)
    names = [step.name for step in steps]
    joined = "\n".join(" ".join(step.argv) for step in steps)

    assert names[:4] == [
        "pose inference",
        "filter tracks camera_a",
        "smooth tracks camera_a",
        "pose metrics camera_a",
    ]
    assert "--pose-input pose_3d_filtered.csv" in joined
    assert "--tracks-input tracks_2d_filtered.csv" in joined
    assert "--pose-input pose_3d_smooth.csv" in joined
    assert "--tracks-csv tracks_2d_smooth.csv" in joined
    assert "--analysis-windows-camera-a free_play,1:40,3:40" in joined
    assert "--analysis-windows-camera-b free_play,1:40,3:40" in joined


def test_build_processing_steps_supports_single_camera_a(tmp_path: Path) -> None:
    job_config = PortalJobConfig(
        job_id="job-001",
        session_id="P001c",
        email="researcher@example.edu",
        video_a=tmp_path / "camera_a_rapid.mp4",
        video_b=None,
        job_dir=tmp_path / "job-001",
        output_root=tmp_path / "output",
        session_config_path=tmp_path / "job-001" / "session_config.json",
        blocks=[
            SessionBlock(
                name="free_play",
                start_s=100.0,
                end_s=220.0,
                color="green",
            )
        ],
    )

    steps = build_processing_steps(job_config)
    names = [step.name for step in steps]
    joined = "\n".join(" ".join(step.argv) for step in steps)

    assert "--video-a" in joined
    assert "--video-b" not in joined
    assert "filter tracks camera_a" in names
    assert "filter tracks camera_b" not in names


def test_build_processing_steps_supports_single_camera_b_with_own_blocks(
    tmp_path: Path,
) -> None:
    job_config = PortalJobConfig(
        job_id="job-001",
        session_id="P001c",
        email="researcher@example.edu",
        video_a=None,
        video_b=tmp_path / "camera_b_rapid.mp4",
        job_dir=tmp_path / "job-001",
        output_root=tmp_path / "output",
        session_config_path=tmp_path / "job-001" / "session_config.json",
        blocks=[],
        blocks_by_camera={
            "camera_b": [
                SessionBlock(
                    name="storybook",
                    start_s=200.0,
                    end_s=260.0,
                    color="blue",
                )
            ]
        },
    )

    steps = build_processing_steps(job_config)
    names = [step.name for step in steps]
    joined = "\n".join(" ".join(step.argv) for step in steps)

    assert "--video-a" not in joined
    assert "--video-b" in joined
    assert "--analysis-windows-camera-b storybook,3:20,4:20" in joined
    assert "filter tracks camera_a" not in names
    assert "filter tracks camera_b" in names


def test_chunk_path_rejects_invalid_camera_id(tmp_path: Path) -> None:
    try:
        chunk_path_for(tmp_path, "../bad", 0)
    except ValueError as error:
        assert "camera_id" in str(error)
    else:
        raise AssertionError("Expected invalid camera_id to fail")


def test_assemble_chunked_upload_writes_final_video(tmp_path: Path) -> None:
    upload_dir = tmp_path / "upload"
    for index, data in enumerate([b"abc", b"def", b"ghi"]):
        path = chunk_path_for(upload_dir, "camera_a", index)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    final_path = assemble_chunked_upload(
        upload_dir=upload_dir,
        camera_id="camera_a",
        original_filename="session.mov",
        total_chunks=3,
    )

    assert final_path == upload_dir / "camera_a.mov"
    assert final_path.read_bytes() == b"abcdefghi"


def test_validate_direct_chunked_upload_uses_final_video_and_receipts(
    tmp_path: Path,
) -> None:
    upload_dir = tmp_path / "upload"
    final_path = final_upload_path_for(upload_dir, "camera_b", "session.mp4")
    final_path.parent.mkdir(parents=True)
    final_path.write_bytes(b"abcdef")
    mark_chunk_received(upload_dir, "camera_b", 0, 3)
    mark_chunk_received(upload_dir, "camera_b", 1, 3)

    validated = validate_direct_chunked_upload(
        upload_dir=upload_dir,
        camera_id="camera_b",
        original_filename="session.mp4",
        total_chunks=2,
        expected_size_bytes=6,
    )

    assert validated == final_path


def test_make_result_zip_includes_portal_metadata(tmp_path: Path) -> None:
    session_dir = tmp_path / "output" / "P001c"
    camera_dir = session_dir / "camera_a"
    camera_dir.mkdir(parents=True)
    (camera_dir / "tracks_2d.csv").write_text("frame_idx,track_id\n", encoding="utf-8")
    metadata_path = tmp_path / "job" / "submission_metadata.json"
    metadata_path.parent.mkdir()
    metadata_path.write_text('{"session_id": "P001c"}\n', encoding="utf-8")
    zip_path = tmp_path / "job" / "P001c_results.zip"

    make_result_zip(
        session_output_dir=session_dir,
        zip_path=zip_path,
        extra_files=[
            (metadata_path, Path("P001c/portal_metadata/submission_metadata.json"))
        ],
    )

    with ZipFile(zip_path) as archive:
        assert "P001c/camera_a/tracks_2d.csv" in archive.namelist()
        assert "P001c/portal_metadata/submission_metadata.json" in archive.namelist()
