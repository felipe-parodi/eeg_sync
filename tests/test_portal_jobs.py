import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pandas as pd
import pytest

import portal.app as portal_app
from portal.app import build_submission_metadata
from portal.app import new_session as render_upload_page
from portal.jobs import (
    PORTAL_DEFAULT_FRAME_RATE,
    PORTAL_DEFAULT_MAX_PERSONS,
    PORTAL_DEFAULT_MODEL_PATH,
    CompressionSettings,
    ExclusionWindow,
    PortalJobConfig,
    SessionBlock,
    build_ffmpeg_compression_command,
    build_processing_steps,
    build_results_readme,
    final_upload_path_for,
    make_result_zip,
    mark_chunk_received,
    parse_timecode,
    split_blocks_for_exclusions,
    validate_direct_chunked_upload,
    write_status,
)


def test_upload_page_exposes_drag_and_drop_targets(monkeypatch) -> None:
    monkeypatch.setenv("PORTAL_PASSWORD", "test-password")
    monkeypatch.setenv("PORTAL_SECRET_KEY", "test-secret")
    portal_app.SESSION_TOKENS.clear()
    response = render_upload_page(portal_app.create_session_token())
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
    assert "Choose at least one GoPro video" in text
    assert "Video A tasks" in text
    assert "Video B tasks" in text


def test_portal_password_and_secret_must_be_configured(monkeypatch) -> None:
    monkeypatch.delenv("PORTAL_PASSWORD", raising=False)
    monkeypatch.delenv("PORTAL_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="PORTAL_PASSWORD must be set"):
        portal_app.portal_password()
    with pytest.raises(RuntimeError, match="PORTAL_SECRET_KEY must be set"):
        portal_app.portal_secret_key()


def test_run_chunked_portal_job_records_failure_and_does_not_raise(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    job_dir = tmp_path / "job"
    upload_dir = tmp_path / "upload"
    job_dir.mkdir()
    upload_dir.mkdir()
    safe_job_id = "20260101_000000_PXXX_test"

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated subprocess failure")

    monkeypatch.setattr(portal_app, "_read_upload_meta", boom)

    portal_app.run_chunked_portal_job(safe_job_id, upload_dir, job_dir)

    status = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    # The raw error goes to the technical details, not the headline.
    assert "simulated subprocess failure" in status["error"]
    assert "simulated subprocess failure" not in status["message"]
    assert status["message"].startswith("The job stopped while")
    captured = capsys.readouterr()
    assert safe_job_id in captured.out
    assert "failed" in captured.out.lower()


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


def test_index_embeds_session_csrf_token(monkeypatch) -> None:
    monkeypatch.setenv("PORTAL_PASSWORD", "test-password")
    monkeypatch.setenv("PORTAL_SECRET_KEY", "test-secret")
    portal_app.SESSION_TOKENS.clear()
    session_token = portal_app.create_session_token()
    csrf_token = portal_app.csrf_token_for_session(session_token)

    response = render_upload_page(session_token)
    text = response.body.decode("utf-8")

    assert f'const CSRF_TOKEN = "{csrf_token}";' in text
    assert 'id="logout-button"' in text


def test_require_csrf_rejects_missing_or_wrong_token(monkeypatch) -> None:
    monkeypatch.setenv("PORTAL_PASSWORD", "test-password")
    monkeypatch.setenv("PORTAL_SECRET_KEY", "test-secret")
    portal_app.SESSION_TOKENS.clear()
    session_token = portal_app.create_session_token()
    csrf_token = portal_app.csrf_token_for_session(session_token)
    request = SimpleNamespace(
        cookies={"portal_session": session_token},
        headers={"x-csrf-token": csrf_token},
        url=SimpleNamespace(path="/uploads/init"),
    )

    assert portal_app.require_csrf(request) == session_token

    request.headers = {}
    with pytest.raises(portal_app.HTTPException) as error:
        portal_app.require_csrf(request)
    assert error.value.status_code == 403

    request.headers = {"x-csrf-token": "wrong"}
    with pytest.raises(portal_app.HTTPException) as error:
        portal_app.require_csrf(request)
    assert error.value.status_code == 403


def test_legacy_submit_route_is_not_registered() -> None:
    routes = {
        (route.path, tuple(sorted(getattr(route, "methods", []))))
        for route in portal_app.app.routes
    }

    assert not any(path == "/submit" for path, _methods in routes)


def test_job_lock_runs_action_under_lock(tmp_path: Path) -> None:
    seen_locked: list[bool] = []

    portal_app._run_with_job_lock(
        tmp_path,
        lambda: seen_locked.append(portal_app.JOB_RUN_LOCK.locked()),
    )

    assert seen_locked == [True]
    assert not portal_app.JOB_RUN_LOCK.locked()


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


def test_job_detail_links_to_annotation_when_awaiting(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(portal_app, "JOB_ROOT", tmp_path)
    job_dir = tmp_path / "job-001"
    write_status(
        job_dir,
        {
            "state": "awaiting_annotation",
            "message": "Label which detection is the parent on each anchor frame.",
            "annotate_url": "/jobs/job-001/annotate",
        },
    )

    response = portal_app.job_detail("job-001", None)
    text = response.body.decode("utf-8")

    assert 'href="/jobs/job-001/annotate"' in text


def test_job_detail_no_annotation_link_when_not_awaiting(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(portal_app, "JOB_ROOT", tmp_path)
    job_dir = tmp_path / "job-001"
    write_status(job_dir, {"state": "running", "message": "Tracking poses."})

    response = portal_app.job_detail("job-001", None)
    text = response.body.decode("utf-8")

    assert "/annotate" not in text


def test_build_submission_metadata_preserves_raw_and_parsed_inputs() -> None:
    metadata = build_submission_metadata(
        job_id="job-001",
        created_at="2026-05-14T13:00:00",
        safe_session_id="P001c",
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
    assert "email" not in metadata
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
    assert metadata["processing"]["frame_rate"] == PORTAL_DEFAULT_FRAME_RATE
    assert metadata["processing"]["max_persons"] == PORTAL_DEFAULT_MAX_PERSONS
    assert metadata["processing"]["pose_model"] == PORTAL_DEFAULT_MODEL_PATH


def test_portal_caps_detections_at_two_persons() -> None:
    # Parent + child only. Extra-person windows remove the experimenter from
    # the analysis blocks before inference, so a third detection is noise.
    assert PORTAL_DEFAULT_MAX_PERSONS == 2


def test_build_submission_metadata_allows_video_b_to_be_omitted() -> None:
    metadata = build_submission_metadata(
        job_id="job-001",
        created_at="2026-05-14T13:00:00",
        safe_session_id="P001c",
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

    assert names[:5] == [
        "pose inference",
        "filter tracks camera_a",
        "interpolate tracks camera_a",
        "smooth tracks camera_a",
        "pose metrics camera_a",
    ]
    # The interpolate step reads the filtered CSVs (gap-fill happens here).
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


def test_make_result_zip_excludes_frames_and_intermediate(tmp_path: Path) -> None:
    """Intermediate artifacts (extracted JPEGs, raw inference JSON) must not
    be bundled into the colleague-facing zip."""
    session_dir = tmp_path / "output" / "P001c"
    camera_dir = session_dir / "camera_a"
    frames_dir = camera_dir / "frames"
    intermediate_dir = camera_dir / "intermediate"
    frames_dir.mkdir(parents=True)
    intermediate_dir.mkdir()

    (camera_dir / "tracks_2d.csv").write_text("frame_idx,track_id\n", encoding="utf-8")
    (camera_dir / "overlay_free_play.mp4").write_bytes(b"\x00")
    (frames_dir / "frame_000001.jpg").write_bytes(b"\xff\xd8\xff")
    (frames_dir / "frame_index.csv").write_text("frame_idx\n1\n", encoding="utf-8")
    (intermediate_dir / "inference_raw.json").write_text("{}", encoding="utf-8")

    zip_path = tmp_path / "job" / "P001c_results.zip"
    make_result_zip(session_output_dir=session_dir, zip_path=zip_path)

    with ZipFile(zip_path) as archive:
        names = archive.namelist()
    assert "P001c/camera_a/tracks_2d.csv" in names
    assert "P001c/camera_a/overlay_free_play.mp4" in names
    assert not any("/frames/" in n for n in names), f"frames leaked: {names}"
    assert not any(
        "/intermediate/" in n for n in names
    ), f"intermediate leaked: {names}"


def test_build_results_readme_documents_files_schema_and_loading(
    tmp_path: Path,
) -> None:
    readme = build_results_readme(_job_config_two_cameras(tmp_path))

    # Session header and the block it documents.
    assert "# P001c" in readme
    assert "free_play" in readme
    # Points analysts at the final, analysis-ready files.
    assert "pose_3d_smooth.csv" in readme
    assert "tracks_2d_smooth.csv" in readme
    # Documents the full processing lineage including interpolation.
    assert "tracks_2d_interpolated.csv" in readme
    # Schema details a reader needs to interpret columns.
    assert "torso_distance_px" in readme
    assert "peak_lag_s" in readme
    assert "Positive = child follows parent" in readme
    # COCO keypoint mapping so kp_000.. are meaningful.
    assert "kp_000" in readme
    assert "nose" in readme
    # Canonical role ids after filtering.
    assert "0 = parent" in readme
    # A concrete load snippet for humans and LLMs.
    assert "pd.read_csv" in readme
    # Coordinates are image pixels, not meters, and z is a 0 placeholder
    # (the pose model is 2D). Guards against the earlier mislabel.
    assert "not meters" in readme
    assert "placeholder" in readme
    assert "position in meters" not in readme
    # Overlay draws every keypoint; filled ones are shown muted.
    assert "interpolated" in readme
    assert "muted" in readme


# ---------------------------------------------------------------------------
# Tests: phase split + anchor extraction + annotation flow (D2b)
# ---------------------------------------------------------------------------


def _job_config_two_cameras(tmp_path: Path) -> PortalJobConfig:
    return PortalJobConfig(
        job_id="job-001",
        session_id="P001c",
        video_a=tmp_path / "camera_a_rapid.mp4",
        video_b=tmp_path / "camera_b_rapid.mp4",
        job_dir=tmp_path / "job-001",
        output_root=tmp_path / "output",
        session_config_path=tmp_path / "job-001" / "session_config.json",
        blocks=[
            SessionBlock(name="free_play", start_s=100.0, end_s=220.0, color="green")
        ],
    )


def test_build_phase1_returns_only_pose_inference(tmp_path: Path) -> None:
    from portal.jobs import build_phase1_steps

    steps = build_phase1_steps(_job_config_two_cameras(tmp_path))
    assert [s.name for s in steps] == ["pose inference"]


def test_build_phase2_returns_per_camera_steps_without_anchors(tmp_path: Path) -> None:
    from portal.jobs import build_phase2_steps

    steps = build_phase2_steps(_job_config_two_cameras(tmp_path))
    names = [s.name for s in steps]
    assert "filter tracks camera_a" in names
    assert "filter tracks camera_b" in names
    assert "interpolate tracks camera_a" in names
    assert "smooth tracks camera_a" in names
    assert "overlay video camera_a free_play" in names
    # Without anchors, no --anchors-json flag anywhere
    joined = "\n".join(" ".join(s.argv) for s in steps)
    assert "--anchors-json" not in joined


def test_build_phase2_interpolates_between_filter_and_smooth(tmp_path: Path) -> None:
    from portal.jobs import build_phase2_steps

    config = _job_config_two_cameras(tmp_path)
    config.video_b = None  # camera A only, keeps the ordering assertion clear
    config.frame_rate = 8.0
    steps = build_phase2_steps(config)
    names = [s.name for s in steps]

    # Interpolation runs after the filter assigns roles and before smoothing,
    # so the gap-fill operates on the parent/child tracks and feeds the EMA.
    assert (
        names.index("filter tracks camera_a")
        < names.index("interpolate tracks camera_a")
        < names.index("smooth tracks camera_a")
    )

    by_name = {s.name: s.argv for s in steps}
    interp_argv = by_name["interpolate tracks camera_a"]
    # Reads the filtered CSVs, writes the interpolated CSVs at the analysis fps.
    assert "tracks_2d_filtered.csv" in interp_argv
    assert "pose_3d_filtered.csv" in interp_argv
    assert "tracks_2d_interpolated.csv" in interp_argv
    assert "pose_3d_interpolated.csv" in interp_argv
    assert "8.0" in interp_argv  # --target-fps matches config.frame_rate

    # Smoothing now consumes the interpolated CSVs, not the filtered ones.
    smooth_argv = by_name["smooth tracks camera_a"]
    assert "pose_3d_interpolated.csv" in smooth_argv
    assert "tracks_2d_interpolated.csv" in smooth_argv
    assert "pose_3d_filtered.csv" not in smooth_argv


def test_build_phase2_renders_one_overlay_per_block(tmp_path: Path) -> None:
    from portal.jobs import build_phase2_steps

    config = _job_config_two_cameras(tmp_path)
    # Two blocks for camera A: a long one (capped) and a short one (full).
    long_block = SessionBlock(name="free_play", start_s=100.0, end_s=400.0)
    short_block = SessionBlock(name="clap_synchrony", start_s=500.0, end_s=515.0)
    config.blocks_by_camera = {"camera_a": [long_block, short_block]}
    config.video_b = None  # camera A only, keeps the assertion focused

    steps = build_phase2_steps(config)
    overlay_steps = {
        s.name: s.argv for s in steps if s.name.startswith("overlay video")
    }

    # The long block gets a first- and a last-window clip; the short block
    # (under one window) gets only the full-block clip.
    assert set(overlay_steps) == {
        "overlay video camera_a free_play",
        "overlay video camera_a free_play (end)",
        "overlay video camera_a clap_synchrony",
    }

    free_play_argv = overlay_steps["overlay video camera_a free_play"]
    assert any(arg.endswith("overlay_free_play.mp4") for arg in free_play_argv)
    # 300s block, 30s cap -> first window 100s..130s.
    assert "1:40" in free_play_argv  # start 100s
    assert "2:10" in free_play_argv  # end 130s (capped)

    end_argv = overlay_steps["overlay video camera_a free_play (end)"]
    assert any(arg.endswith("overlay_free_play_end.mp4") for arg in end_argv)
    # Last window of the 100s..400s block: 370s..400s.
    assert "6:10" in end_argv  # start 370s
    assert "6:40" in end_argv  # end 400s

    clap_argv = overlay_steps["overlay video camera_a clap_synchrony"]
    assert any(arg.endswith("overlay_clap_synchrony.mp4") for arg in clap_argv)
    # 15s block (< 30s cap) renders in full once: 500s..515s, no separate end.
    assert "8:20" in clap_argv  # start 500s
    assert "8:35" in clap_argv  # end 515s (full block)


def test_build_phase2_injects_anchors_json_per_camera(tmp_path: Path) -> None:
    from portal.jobs import build_phase2_steps

    anchors_path_a = tmp_path / "anchors_a.json"
    anchors_path_b = tmp_path / "anchors_b.json"
    steps = build_phase2_steps(
        _job_config_two_cameras(tmp_path),
        anchors_json_by_camera={
            "camera_a": anchors_path_a,
            "camera_b": anchors_path_b,
        },
    )
    by_name = {s.name: s.argv for s in steps}
    assert "--anchors-json" in by_name["filter tracks camera_a"]
    assert str(anchors_path_a) in by_name["filter tracks camera_a"]
    assert "--anchors-json" in by_name["filter tracks camera_b"]
    assert str(anchors_path_b) in by_name["filter tracks camera_b"]
    # Other steps should not get the flag.
    assert "--anchors-json" not in by_name["smooth tracks camera_a"]


def test_annotate_page_track_buttons_use_dark_readable_text(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PORTAL_PASSWORD", "test-password")
    monkeypatch.setenv("PORTAL_SECRET_KEY", "test-secret")
    monkeypatch.setattr(portal_app, "JOB_ROOT", tmp_path)
    portal_app.SESSION_TOKENS.clear()
    session_token = portal_app.create_session_token()

    camera_dir = tmp_path / "job-001" / "anchors" / "camera_a"
    camera_dir.mkdir(parents=True)
    (camera_dir / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "anchor_idx": 0,
                    "image_filename": "anchor_000.jpg",
                    "frame_idx": 10,
                    "timestamp_s": 1.25,
                    "tracks": [
                        {"track_id": 1, "color_bgr": [0, 255, 0]},  # green
                        {"track_id": 2, "color_bgr": [0, 165, 255]},  # orange
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    response = portal_app.annotate_job("job-001", session_token)
    text = response.body.decode("utf-8")

    # The track buttons sit on a light tinted background; the global button
    # rule paints text white, which is unreadable there. Each track button
    # must override to dark text.
    assert "background: rgba(0,255,0,0.12); color: #17202a;" in text
    assert "background: rgba(255,165,0,0.12); color: #17202a;" in text


def test_build_processing_steps_is_phase1_plus_phase2(tmp_path: Path) -> None:
    from portal.jobs import (
        build_phase1_steps,
        build_phase2_steps,
        build_processing_steps,
    )

    cfg = _job_config_two_cameras(tmp_path)
    full = build_processing_steps(cfg)
    expected = build_phase1_steps(cfg) + build_phase2_steps(cfg)
    assert [s.name for s in full] == [s.name for s in expected]


def test_resolve_anchors_json_paths_finds_per_camera(tmp_path: Path) -> None:
    from portal.jobs import resolve_anchors_json_paths

    cfg = _job_config_two_cameras(tmp_path)
    (cfg.job_dir / "anchors" / "camera_a").mkdir(parents=True)
    (cfg.job_dir / "anchors" / "camera_a" / "anchors.json").write_text(
        "[]", encoding="utf-8"
    )
    paths = resolve_anchors_json_paths(cfg)
    assert "camera_a" in paths
    assert paths["camera_a"].exists()
    # camera_b's anchors.json was never written
    assert "camera_b" not in paths


def test_extract_job_anchor_frames_calls_extractor_per_camera(
    monkeypatch, tmp_path: Path
) -> None:
    from portal import jobs as portal_jobs

    cfg = _job_config_two_cameras(tmp_path)
    # The extractor reads tracks_2d.csv; create stubs so the per-camera
    # branches both fire.
    for cam in ("camera_a", "camera_b"):
        (cfg.session_output_dir / cam).mkdir(parents=True)
        (cfg.session_output_dir / cam / "tracks_2d.csv").write_text(
            "frame_idx,timestamp_s,track_id\n", encoding="utf-8"
        )

    seen: list[dict] = []

    def fake_extract(tracks_csv, source_video, output_dir, n_anchors, **_kwargs):
        seen.append(
            {
                "tracks_csv": str(tracks_csv),
                "source_video": str(source_video),
                "output_dir": str(output_dir),
                "n_anchors": n_anchors,
            }
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # Pretend we extracted n_anchors frames
        return [
            SimpleNamespace(
                anchor_idx=i,
                frame_idx=i * 10,
                timestamp_s=float(i),
                image_filename=f"anchor_{i:02d}.jpg",
                tracks=[],
            )
            for i in range(1, n_anchors + 1)
        ]

    monkeypatch.setattr(
        "video_analysis.anchor_frames.extract_anchor_frames", fake_extract
    )

    summary = portal_jobs.extract_job_anchor_frames(cfg, n_anchors=4)

    assert summary == {
        "camera_a": {
            "count": 4,
            "dir": str(cfg.job_dir / "anchors" / "camera_a"),
        },
        "camera_b": {
            "count": 4,
            "dir": str(cfg.job_dir / "anchors" / "camera_b"),
        },
    }
    assert {entry["source_video"] for entry in seen} == {
        str(cfg.video_a),
        str(cfg.video_b),
    }


def test_convert_anchor_submission_validates_and_remaps(tmp_path: Path) -> None:
    from portal.app import _convert_anchor_submission_to_filter_format

    manifest = [
        {
            "anchor_idx": 1,
            "frame_idx": 100,
            "tracks": [{"track_id": 5}, {"track_id": 7}],
        },
        {
            "anchor_idx": 2,
            "frame_idx": 200,
            "tracks": [{"track_id": 5}, {"track_id": 7}],
        },
    ]
    submission = [
        {"anchor_idx": 1, "parent_track_id": 5, "child_track_id": 7},
        {"anchor_idx": 2, "skipped": True},
    ]
    converted = _convert_anchor_submission_to_filter_format(submission, manifest)
    assert converted == [
        {"frame_idx": 100, "parent_track_id": 5, "child_track_id": 7},
        {"frame_idx": 200, "skipped": True},
    ]


def test_convert_anchor_submission_rejects_unknown_track_id() -> None:
    from portal.app import _convert_anchor_submission_to_filter_format

    manifest = [
        {
            "anchor_idx": 1,
            "frame_idx": 100,
            "tracks": [{"track_id": 5}, {"track_id": 7}],
        }
    ]
    submission = [{"anchor_idx": 1, "parent_track_id": 99, "child_track_id": 7}]
    with pytest.raises(portal_app.HTTPException) as error:
        _convert_anchor_submission_to_filter_format(submission, manifest)
    assert error.value.status_code == 400


def test_convert_anchor_submission_rejects_unknown_anchor_idx() -> None:
    from portal.app import _convert_anchor_submission_to_filter_format

    manifest = [
        {
            "anchor_idx": 1,
            "frame_idx": 100,
            "tracks": [{"track_id": 5}, {"track_id": 7}],
        }
    ]
    submission = [{"anchor_idx": 99, "parent_track_id": 5, "child_track_id": 7}]
    with pytest.raises(portal_app.HTTPException) as error:
        _convert_anchor_submission_to_filter_format(submission, manifest)
    assert error.value.status_code == 400


def test_anchor_image_endpoint_404s_when_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(portal_app, "JOB_ROOT", tmp_path)
    (tmp_path / "job-001").mkdir()
    with pytest.raises(portal_app.HTTPException) as error:
        portal_app.anchor_image("job-001", "camera_a", "anchor_01.jpg", None)
    assert error.value.status_code == 404


def test_anchor_image_endpoint_serves_jpeg(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(portal_app, "JOB_ROOT", tmp_path)
    anchor_dir = tmp_path / "job-001" / "anchors" / "camera_a"
    anchor_dir.mkdir(parents=True)
    anchor_path = anchor_dir / "anchor_01.jpg"
    anchor_path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")

    response = portal_app.anchor_image("job-001", "camera_a", "anchor_01.jpg", None)
    assert response.media_type == "image/jpeg"
    assert Path(response.path).name == "anchor_01.jpg"


def test_build_phase1_uses_offline_two_person_tracker(tmp_path: Path) -> None:
    from portal.jobs import build_phase1_steps

    argv = build_phase1_steps(_job_config_two_cameras(tmp_path))[0].argv
    assert argv[argv.index("--tracker-backend") + 1] == "offline"
    assert "botsort" not in argv


def test_build_phase2_uses_offline_association_when_cache_exists(
    tmp_path: Path,
) -> None:
    from portal.jobs import build_phase2_steps

    cfg = _job_config_two_cameras(tmp_path)
    for cam in ("camera_a", "camera_b"):
        cache = cfg.session_output_dir / cam / "intermediate" / "detections.npz"
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"")
    anchors_a = tmp_path / "anchors_a.json"
    steps = build_phase2_steps(cfg, anchors_json_by_camera={"camera_a": anchors_a})
    by_name = {s.name: s.argv for s in steps}
    filter_a = by_name["filter tracks camera_a"]
    assert "video_inference.offline_assoc" in filter_a
    assert filter_a[filter_a.index("--anchors-json") + 1] == str(anchors_a)
    filter_b = by_name["filter tracks camera_b"]
    assert "video_inference.offline_assoc" in filter_b
    assert "--anchors-json" not in filter_b


def test_build_phase2_falls_back_to_track_filter_without_cache(tmp_path: Path) -> None:
    from portal.jobs import build_phase2_steps

    steps = build_phase2_steps(_job_config_two_cameras(tmp_path))
    by_name = {s.name: s.argv for s in steps}
    assert "video_analysis.track_filter" in by_name["filter tracks camera_a"]


def test_submission_metadata_records_offline_tracker() -> None:
    metadata = build_submission_metadata(
        job_id="job",
        created_at="2026-10-09T00:00:00",
        safe_session_id="P001",
        blocks_text="free_play,0:10,0:40",
        exclusions_text="",
        video_a_name="a.mp4",
        video_b_name="",
        video_a_size=10,
        video_b_size=0,
        include_gaze=False,
    )
    assert metadata["processing"]["tracker_backend"] == "offline"
    assert metadata["processing"]["tracker_name"] == "two_person_offline"


def test_build_phase2_refines_keypoints_after_offline_association(
    tmp_path: Path,
) -> None:
    from portal.jobs import build_phase2_steps

    cfg = _job_config_two_cameras(tmp_path)
    cfg.video_b = None
    cache = cfg.session_output_dir / "camera_a" / "intermediate" / "detections.npz"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"")
    steps = build_phase2_steps(cfg)
    names = [s.name for s in steps]
    assert (
        names.index("filter tracks camera_a")
        < names.index("refine keypoints camera_a")
        < names.index("interpolate tracks camera_a")
    )
    by_name = {s.name: s.argv for s in steps}
    assert "video_inference.topdown_refine" in by_name["refine keypoints camera_a"]
    assert "pose_3d_refined.csv" in by_name["interpolate tracks camera_a"]

    cfg.refine_keypoints = False
    names_off = [s.name for s in build_phase2_steps(cfg)]
    assert "refine keypoints camera_a" not in names_off


def test_phase2_steps_are_tagged_with_their_camera(tmp_path: Path) -> None:
    from portal.jobs import build_phase2_steps

    steps = build_phase2_steps(_job_config_two_cameras(tmp_path))
    for step in steps:
        assert step.camera_id in {"camera_a", "camera_b"}
        assert step.camera_id in step.name


def test_phase1_requests_parallel_cameras(tmp_path: Path) -> None:
    from portal.jobs import build_phase1_steps

    cfg = _job_config_two_cameras(tmp_path)
    assert "--parallel-cameras" in build_phase1_steps(cfg)[0].argv
    cfg.parallel_cameras = False
    assert "--parallel-cameras" not in build_phase1_steps(cfg)[0].argv


def test_run_steps_runs_cameras_concurrently_and_merges_logs(
    tmp_path: Path, monkeypatch
) -> None:
    import threading

    from portal import jobs as portal_jobs
    from portal.jobs import CommandStep, run_steps

    barrier = threading.Barrier(2, timeout=5)

    def fake_run(step, cwd, log_path):
        if step.name.startswith("first"):
            barrier.wait()  # both cameras must be inside their first step at once
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"== {step.name} ==\n")

    monkeypatch.setattr(portal_jobs, "run_command_step", fake_run)
    steps = [
        CommandStep("first camera_a", [], "camera_a"),
        CommandStep("second camera_a", [], "camera_a"),
        CommandStep("first camera_b", [], "camera_b"),
        CommandStep("second camera_b", [], "camera_b"),
    ]
    seen = []
    log = tmp_path / "job.log"
    run_steps(
        steps,
        tmp_path,
        log,
        on_start=lambda i, n, s: seen.append(i),
        parallel_cameras=True,
    )
    assert sorted(seen) == [1, 2, 3, 4]
    text = log.read_text(encoding="utf-8")
    assert (
        text.index("first camera_a")
        < text.index("second camera_a")
        < text.index("first camera_b")
    )
    assert not list(tmp_path.glob("job.camera_*.log"))


def test_run_steps_reports_failure_after_other_camera_finishes(
    tmp_path: Path, monkeypatch
) -> None:
    import subprocess

    from portal import jobs as portal_jobs
    from portal.jobs import CommandStep, run_steps

    ran = []

    def fake_run(step, cwd, log_path):
        ran.append(step.name)
        if step.name == "boom camera_a":
            raise subprocess.CalledProcessError(1, ["x"])

    monkeypatch.setattr(portal_jobs, "run_command_step", fake_run)
    steps = [
        CommandStep("boom camera_a", [], "camera_a"),
        CommandStep("after camera_a", [], "camera_a"),
        CommandStep("ok camera_b", [], "camera_b"),
    ]
    with pytest.raises(subprocess.CalledProcessError):
        run_steps(steps, tmp_path, tmp_path / "job.log", parallel_cameras=True)
    assert "ok camera_b" in ran
    assert "after camera_a" not in ran


def test_run_steps_never_overlaps_gpu_heavy_steps(tmp_path: Path, monkeypatch) -> None:
    import threading
    import time

    from portal import jobs as portal_jobs
    from portal.jobs import CommandStep, run_steps

    active = 0
    peak = 0
    lock = threading.Lock()

    def fake_run(step, cwd, log_path):
        nonlocal active, peak
        if step.gpu_heavy:
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1

    monkeypatch.setattr(portal_jobs, "run_command_step", fake_run)
    steps = [
        CommandStep(f"refine {cam}", [], cam, gpu_heavy=True)
        for cam in ("camera_a", "camera_b")
    ]
    run_steps(steps, tmp_path, tmp_path / "job.log", parallel_cameras=True)
    assert peak == 1


def test_build_submission_metadata_applies_exclusions_per_camera_clock() -> None:
    # Each camera has its own clock, so each gets its own extra-person windows.
    metadata = build_submission_metadata(
        job_id="job-001",
        created_at="2026-05-14T13:00:00",
        safe_session_id="1015",
        blocks_text="grocery,16:00,26:50",
        blocks_text_b="grocery,14:48,25:54",
        exclusions_text="18:06,18:21,experimenter entered",
        exclusions_text_b="16:54,17:09,experimenter entered",
        video_a_name="a.mp4",
        video_b_name="b.mp4",
        video_a_size=1024,
        video_b_size=2048,
        include_gaze=False,
    )

    blocks = metadata["analysis_blocks_by_camera"]
    assert [(b["start_s"], b["end_s"]) for b in blocks["camera_a"]] == [
        (960.0, 1086.0),
        (1101.0, 1610.0),
    ]
    assert [(b["start_s"], b["end_s"]) for b in blocks["camera_b"]] == [
        (888.0, 1014.0),
        (1029.0, 1554.0),
    ]
    windows = metadata["extra_person_windows_by_camera"]
    assert windows["camera_b"][0]["start_s"] == 1014.0
    assert metadata["raw_inputs"]["extra_person_windows_text_b"] == (
        "16:54,17:09,experimenter entered"
    )


def test_build_submission_metadata_camera_b_exclusions_can_be_empty() -> None:
    metadata = build_submission_metadata(
        job_id="job-001",
        created_at="2026-05-14T13:00:00",
        safe_session_id="1015",
        blocks_text="grocery,16:00,26:50",
        blocks_text_b="grocery,14:48,25:54",
        exclusions_text="18:06,18:21,experimenter entered",
        exclusions_text_b="",
        video_a_name="a.mp4",
        video_b_name="b.mp4",
        video_a_size=1024,
        video_b_size=2048,
        include_gaze=False,
    )

    assert len(metadata["analysis_blocks_by_camera"]["camera_a"]) == 2
    assert len(metadata["analysis_blocks_by_camera"]["camera_b"]) == 1


def _write_session_metrics(output_root: Path, session_id: str, coverage: float) -> None:
    session = output_root / session_id
    session.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "session_id": session_id,
                "camera_id": "camera_a",
                "block": "storybook",
                "block_start_s": 100.0,
                "block_end_s": 200.0,
                "block_frames": 601,
                "coverage_both": coverage,
                "low_coverage": coverage < 0.5,
            }
        ]
    ).to_csv(session / "session_metrics.csv", index=False)


def test_job_page_shows_block_coverage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(portal_app, "JOB_ROOT", tmp_path / "portal_jobs")
    monkeypatch.setattr(portal_app, "OUTPUT_ROOT", tmp_path)
    job_dir = tmp_path / "portal_jobs" / "job-001"
    write_status(job_dir, {"state": "complete", "message": "Done."})
    (job_dir / "job_config.json").write_text(
        json.dumps({"session_id": "1012"}), encoding="utf-8"
    )
    _write_session_metrics(tmp_path, "1012", 0.21)

    text = portal_app.job_detail("job-001", None).body.decode("utf-8")

    assert "storybook" in text
    assert "21%" in text
    assert "Low" in text


def test_all_sessions_table_download(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(portal_app, "OUTPUT_ROOT", tmp_path)
    _write_session_metrics(tmp_path, "1010", 0.9)
    _write_session_metrics(tmp_path, "1012", 0.2)

    response = portal_app.download_all_sessions_table(None)

    table = pd.read_csv(response.path)
    assert sorted(table["session_id"].astype(str)) == ["1010", "1012"]


def _gaze_label_job(tmp_path: Path, monkeypatch) -> Path:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gaze_label_helpers", Path(__file__).with_name("test_portal_gaze_labels.py")
    )
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    _camera = helpers._camera

    monkeypatch.setattr(portal_app, "JOB_ROOT", tmp_path / "portal_jobs")
    monkeypatch.setattr(portal_app, "OUTPUT_ROOT", tmp_path)
    session = tmp_path / "1012"
    session.mkdir()
    _camera(session)
    job_dir = tmp_path / "portal_jobs" / "job-001"
    write_status(job_dir, {"state": "complete", "message": "Done."})
    (job_dir / "job_config.json").write_text(
        json.dumps(
            {
                "session_id": "1012",
                "frame_rate": 6.0,
                "blocks_by_camera": {
                    "camera_a": [{"name": "grocery", "start_s": 100.0, "end_s": 199.0}]
                },
            }
        ),
        encoding="utf-8",
    )
    return job_dir


def test_gaze_label_page_lists_frames(tmp_path: Path, monkeypatch) -> None:
    _gaze_label_job(tmp_path, monkeypatch)
    monkeypatch.setattr(portal_app, "csrf_token_for_session", lambda token: "csrf")

    text = portal_app.gaze_label_page("job-001", None).body.decode("utf-8")

    assert "The other person's face" in text
    assert '"image_name": "frame_' in text
    assert "/gaze-frames/" in text


def test_gaze_label_submission_is_saved_with_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    from portal.gaze_labels import load_gaze_labels

    _gaze_label_job(tmp_path, monkeypatch)
    entry = {
        "frame_idx": 600,
        "timestamp_s": 100.0,
        "person": "child",
        "target": "object",
        "point": None,
    }
    saved = portal_app._save_gaze_label_submission(
        "job-001", {"labeler": "RA", "labels": {"camera_a": [entry]}}
    )

    assert saved == {"camera_a": 1}
    assert load_gaze_labels(tmp_path / "1012" / "camera_a") == [entry]
    with pytest.raises(Exception):
        portal_app._save_gaze_label_submission(
            "job-001", {"labels": {"camera_z": [entry]}}
        )


def test_gaze_step_uses_page_by_default_and_gazelle_on_request(tmp_path: Path) -> None:
    from portal.jobs import build_phase2_steps

    cfg = _job_config_two_cameras(tmp_path)
    cfg.include_gaze = True
    gaze = [s for s in build_phase2_steps(cfg) if s.name.startswith("gaze inference")]
    assert len(gaze) == 2
    assert all("gaze_analysis.page_runner" in s.argv for s in gaze)

    cfg.gaze_model = "gazelle"
    gaze = [s for s in build_phase2_steps(cfg) if s.name.startswith("gaze inference")]
    assert all("gaze_analysis.gazelle_runner" in s.argv for s in gaze)


# ---- Frame timing (metrics-validity plan, phase 0) ----


def _write_extraction_info(config, camera_id: str, achieved: float) -> None:
    frames = config.session_output_dir / camera_id / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    (frames / "extraction_info.json").write_text(
        json.dumps({"achieved_frame_rate": achieved}), encoding="utf-8"
    )


def test_portal_defaults_give_even_5_fps() -> None:
    """Portal compression at 10 fps divides evenly into the 5 fps analysis rate."""
    from portal.jobs import PORTAL_DEFAULT_FRAME_RATE, CompressionSettings

    assert PORTAL_DEFAULT_FRAME_RATE == 5.0
    assert CompressionSettings().target_fps == 10.0


def test_session_frame_rate_uses_achieved_rate(tmp_path: Path) -> None:
    from portal.jobs import achieved_session_frame_rate

    config = _job_config_two_cameras(tmp_path)
    _write_extraction_info(config, "camera_a", 5.0)
    _write_extraction_info(config, "camera_b", 5.0)

    assert achieved_session_frame_rate(config) == 5.0


def test_session_frame_rate_falls_back_without_extraction_info(tmp_path: Path) -> None:
    from portal.jobs import achieved_session_frame_rate

    config = _job_config_two_cameras(tmp_path)

    assert achieved_session_frame_rate(config) == config.frame_rate


def test_session_frame_rate_mismatch_stops_with_plain_message(tmp_path: Path) -> None:
    from portal.jobs import achieved_session_frame_rate

    config = _job_config_two_cameras(tmp_path)
    _write_extraction_info(config, "camera_a", 5.0)
    _write_extraction_info(config, "camera_b", 4.0)

    with pytest.raises(ValueError, match="frames per second"):
        achieved_session_frame_rate(config)


def test_phase2_resume_uses_achieved_frame_rate(tmp_path: Path, monkeypatch) -> None:
    """Phase 2 rebuilds its config from the upload; it must re-read the rate."""
    import portal.jobs as jobs

    config = _job_config_two_cameras(tmp_path)
    config.job_dir.mkdir(parents=True, exist_ok=True)
    _write_extraction_info(config, "camera_a", 4.0)
    _write_extraction_info(config, "camera_b", 4.0)
    seen: list[float] = []

    def fake_build(cfg, anchors_json_by_camera=None):
        seen.append(cfg.frame_rate)
        raise RuntimeError("stop after step building")

    monkeypatch.setattr(jobs, "build_phase2_steps", fake_build)
    monkeypatch.setattr(jobs, "resolve_anchors_json_paths", lambda cfg: {})

    try:
        jobs.resume_portal_job_phase2(config, tmp_path)
    except RuntimeError:
        pass

    assert seen == [4.0]


def test_annotate_page_uses_the_callers_own_csrf_token(
    tmp_path: Path, monkeypatch
) -> None:
    """With two people logged in, each sees their own CSRF token.

    The page used to embed the most recently created session's token, so the
    earlier user's submit was rejected with 403.
    """
    monkeypatch.setenv("PORTAL_PASSWORD", "test-password")
    monkeypatch.setenv("PORTAL_SECRET_KEY", "test-secret")
    monkeypatch.setattr(portal_app, "JOB_ROOT", tmp_path)
    portal_app.SESSION_TOKENS.clear()
    first = portal_app.create_session_token()
    second = portal_app.create_session_token()
    # The second person logged in later, so their session expires later.
    portal_app.SESSION_TOKENS[second].expires_at += 600

    camera_dir = tmp_path / "job-001" / "anchors" / "camera_a"
    camera_dir.mkdir(parents=True)
    (camera_dir / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "anchor_idx": 0,
                    "image_filename": "anchor_000.jpg",
                    "frame_idx": 10,
                    "timestamp_s": 1.25,
                    "tracks": [{"track_id": 1, "color_bgr": [0, 255, 0]}],
                }
            ]
        ),
        encoding="utf-8",
    )

    text = portal_app.annotate_job("job-001", first).body.decode("utf-8")

    assert portal_app.csrf_token_for_session(first) in text
    assert portal_app.csrf_token_for_session(second) not in text
