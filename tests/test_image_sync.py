from __future__ import annotations

from pathlib import Path

import pytest

from image_host.core.sync_manager import SyncManager
from image_host.providers.stardots_provider import StarDotsProvider


class _FakeImageHost:
    def __init__(self, remote_images: list[dict[str, str]]) -> None:
        self._remote_images = remote_images
        self.config = {"provider": "stardots"}

    def get_image_list(self) -> list[dict[str, str]]:
        return list(self._remote_images)

    def upload_image(self, file_path: Path) -> dict[str, str]:
        raise NotImplementedError

    def delete_image(self, image_hash: str) -> bool:
        raise NotImplementedError

    def download_image(self, image_info: dict[str, str], save_path: Path) -> bool:
        raise NotImplementedError


def test_stardots_build_remote_filename_omits_separator_for_empty_category() -> None:
    provider = StarDotsProvider.__new__(StarDotsProvider)
    provider.DEFAULT_CATEGORY = "default"
    provider.CATEGORY_SEPARATOR = "@@CAT@@"
    provider._encode_category = lambda category: category

    remote_name = provider._build_remote_filename(
        {"filename": "demo.png", "category": ""}
    )

    assert remote_name == "demo.png"


def test_sync_manager_check_status_does_not_write_progress_to_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sync_manager = SyncManager(
        image_host=_FakeImageHost(
            [{"id": "remote/demo.png", "filename": "demo.png", "category": "remote"}]
        ),
        local_dir=tmp_path,
        upload_tracker=None,
    )
    sync_manager.file_handler.scan_local_images = lambda: [
        {
            "id": "local/demo.png",
            "path": str(tmp_path / "local" / "demo.png"),
            "filename": "demo.png",
            "category": "local",
        }
    ]

    status = sync_manager.check_sync_status()
    captured = capsys.readouterr()

    assert status["to_upload"]
    assert captured.out == ""
