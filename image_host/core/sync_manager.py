import logging
from pathlib import Path

from ..interfaces.image_host import ImageHostInterface
from .file_handler import FileHandler
from .upload_tracker import UploadTracker

logger = logging.getLogger(__name__)


class SyncManager:
    """同步管理器"""

    def __init__(
        self,
        image_host: ImageHostInterface,
        local_dir: Path,
        upload_tracker: UploadTracker | None = None,
    ):
        self.image_host = image_host
        self.file_handler = FileHandler(local_dir)
        self.upload_tracker = upload_tracker

    def _normalize_remote_id(self, remote_id: str, provider_name: str = None) -> str:
        """
        根据不同的图床提供商规范化远程文件ID

        Args:
            remote_id: 远程文件ID
            provider_name: 提供商名称

        Returns:
            规范化后的文件ID，用于与本地文件ID比较
        """
        if not provider_name:
            # 尝试从配置获取提供商名称
            if hasattr(self.image_host, "config") and self.image_host.config:
                provider_name = self.image_host.config.get("provider", "").lower()
            elif hasattr(self.image_host, "__class__"):
                provider_name = self.image_host.__class__.__name__.lower()

        # 统一转换为正斜杠
        normalized_id = remote_id.replace("\\", "/")

        # 根据不同提供商处理特定的前缀
        if provider_name:
            if "cloudflare_r2" in provider_name or "r2" in provider_name:
                # Cloudflare R2: 移除 memes/ 前缀
                if normalized_id.startswith("memes/"):
                    return normalized_id[6:]  # 移除"memes/"前缀
            elif "stardots" in provider_name:
                # Stardots: 保持原样（未来可能需要特殊处理）
                pass
            # 可以在这里添加其他提供商的处理逻辑

        return normalized_id

    def check_sync_status(self) -> dict[str, list[dict]]:
        """检查同步状态 - 简化版，只检查存在性"""
        logger.info("Scanning local meme files before sync")
        local_images = self.file_handler.scan_local_images()
        logger.info("Local file count: %s", len(local_images))

        logger.info("Fetching remote meme file list")
        remote_images = self.image_host.get_image_list()
        logger.info("Remote file count: %s", len(remote_images))

        # 上传：检查哪些文件没有上传记录
        to_upload = []
        if self.upload_tracker:
            for img in local_images:
                category = img.get("category", "")
                file_path = Path(img["path"])
                if not self.upload_tracker.is_uploaded(file_path, category):
                    to_upload.append(img)
            logger.info("Local files without upload record: %s", len(to_upload))
        else:
            # 如果没有上传追踪器，默认所有文件都需要上传
            to_upload = local_images
            logger.info(
                "Upload tracker disabled, marking all local files for upload: %s",
                len(to_upload),
            )

        # 获取提供商名称
        provider_name = None
        if hasattr(self.image_host, "config") and self.image_host.config:
            provider_name = self.image_host.config.get("provider", "").lower()

        # 下载：检查哪些文件本地不存在
        local_file_ids = {img["id"].replace("\\", "/") for img in local_images}
        to_download = []
        for img in remote_images:
            remote_id = img["id"].replace("\\", "/")
            normalized_remote_id = self._normalize_remote_id(remote_id, provider_name)
            if normalized_remote_id not in local_file_ids:
                to_download.append(img)
        logger.info("Remote files missing locally: %s", len(to_download))

        # 远程删除：检查哪些文件在云端存在但本地不存在
        to_delete_remote = to_download.copy()
        logger.info(
            "Remote-only files to delete for overwrite_to_remote: %s",
            len(to_delete_remote),
        )

        # 本地删除：检查哪些文件在本地存在但云端不存在
        remote_file_ids = set()
        for img in remote_images:
            remote_id = img["id"].replace("\\", "/")
            normalized_remote_id = self._normalize_remote_id(remote_id, provider_name)
            remote_file_ids.add(normalized_remote_id)

        to_delete_local = []
        for img in local_images:
            local_id = img["id"].replace("\\", "/")
            if local_id not in remote_file_ids:
                to_delete_local.append(img)
        logger.info(
            "Local-only files to delete for overwrite_from_remote: %s",
            len(to_delete_local),
        )

        return {
            "to_upload": to_upload,
            "to_download": to_download,
            "to_delete_local": to_delete_local,
            "to_delete_remote": to_delete_remote,
            "is_synced": not (
                to_upload or to_download or to_delete_local or to_delete_remote
            ),
        }

    def sync_to_remote(self) -> bool:
        """同步本地文件到远程 - 只上传未上传过的文件"""
        status = self.check_sync_status()

        if status.get("is_synced", False):
            logger.info("Local and remote are already aligned for upload")
            return True

        # 上传新文件
        to_upload = status["to_upload"]
        if to_upload:
            logger.info("Starting upload of %s files", len(to_upload))
            uploaded_count = 0
            skipped_count = 0

            for image in to_upload:
                file_path = Path(image["path"])
                category = image.get("category", "")

                try:
                    result = self.image_host.upload_image(file_path)
                    if self.upload_tracker:
                        remote_url = result.get("url", "")
                        self.upload_tracker.mark_uploaded(
                            file_path, category, remote_url
                        )
                    uploaded_count += 1
                except Exception as exc:
                    logger.warning(
                        "Upload failed for %s: %s",
                        file_path.name,
                        exc,
                    )
                    skipped_count += 1

            logger.info(
                "Upload finished: success=%s failed=%s",
                uploaded_count,
                skipped_count,
            )
        else:
            logger.info("No files need upload")

        return True

    def sync_from_remote(self) -> bool:
        """从远程同步文件到本地 - 只下载本地不存在的文件"""
        status = self.check_sync_status()

        if status.get("is_synced", False):
            logger.info("Local and remote are already aligned for download")
            return True

        # 下载新文件
        to_download = status["to_download"]
        if to_download:
            logger.info("Starting download of %s files", len(to_download))
            downloaded_count = 0
            skipped_count = 0

            for image in to_download:
                category = image.get("category", "")
                filename = image["filename"]
                try:
                    save_path = self.file_handler.get_file_path(category, filename)
                    if save_path.exists():
                        logger.info("Skipping existing local file: %s", filename)
                        skipped_count += 1
                        continue

                    if self.image_host.download_image(image, save_path):
                        downloaded_count += 1
                    else:
                        logger.warning("Download failed for %s", filename)
                        skipped_count += 1
                except Exception as exc:
                    logger.warning("Download failed for %s: %s", filename, exc)
                    skipped_count += 1

            logger.info(
                "Download finished: success=%s failed_or_skipped=%s",
                downloaded_count,
                skipped_count,
            )
        else:
            logger.info("No files need download")

        return True

    def overwrite_to_remote(self) -> bool:
        """从本地覆盖云端 - 让云端完全和本地一致"""
        status = self.check_sync_status()

        # 1. 上传本地多出的文件
        self.sync_to_remote()

        # 2. 删除云端多出的文件
        to_delete_remote = status.get("to_delete_remote", [])
        if to_delete_remote:
            logger.info(
                "Deleting %s remote-only files to overwrite remote state",
                len(to_delete_remote),
            )
            deleted_count = 0
            for img in to_delete_remote:
                try:
                    if self.image_host.delete_image(img["id"]):
                        deleted_count += 1
                except Exception as exc:
                    logger.warning(
                        "Failed to delete remote file %s: %s",
                        img["filename"],
                        exc,
                    )
            logger.info("Remote cleanup finished: deleted=%s", deleted_count)
        else:
            logger.info("No remote-only files need deletion")

        return True

    def overwrite_from_remote(self) -> bool:
        """从云端覆盖本地 - 让本地完全和云端一致"""
        status = self.check_sync_status()

        # 1. 下载本地缺失的文件
        self.sync_from_remote()

        # 2. 删除本地多出的文件
        to_delete_local = status.get("to_delete_local", [])
        if to_delete_local:
            logger.info(
                "Deleting %s local-only files to overwrite local state",
                len(to_delete_local),
            )
            deleted_count = 0
            for img in to_delete_local:
                try:
                    file_path = Path(img["path"])
                    if file_path.exists():
                        file_path.unlink()
                        deleted_count += 1
                        # 同时从上传记录中移除
                        if self.upload_tracker:
                            self.upload_tracker.remove_record(
                                file_path, img.get("category", "")
                            )
                except Exception as exc:
                    logger.warning(
                        "Failed to delete local file %s: %s",
                        img["filename"],
                        exc,
                    )
            logger.info("Local cleanup finished: deleted=%s", deleted_count)
        else:
            logger.info("No local-only files need deletion")

        return True
