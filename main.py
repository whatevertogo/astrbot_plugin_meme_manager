from __future__ import annotations

import asyncio
import json
import random
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from astrbot_sdk import (
    Context,
    EventResultType,
    Image,
    MessageChain,
    MessageEvent,
    MessageEventResult,
    Plain,
    Star,
)
from astrbot_sdk.clients.llm import LLMResponse
from astrbot_sdk.decorators import (
    http_api,
    on_command,
    on_event,
    on_message,
    priority,
    provide_capability,
    require_admin,
    validate_config,
)
from astrbot_sdk.errors import AstrBotError
from astrbot_sdk.llm.entities import ProviderRequest
from astrbot_sdk.message import component_to_payload_sync, payload_to_component
from category_manager import CategoryManager
from image_host.img_sync import ImageSync
from PIL import Image as PILImage
from pydantic import BaseModel, Field
from services import (
    DEFAULT_CATEGORY_DESCRIPTIONS,
    IMAGE_SUFFIXES,
    PluginPaths,
    build_paths,
    copy_image_to_category,
    dict_to_prompt_lines,
    ensure_runtime_layout,
    library_stats,
    list_category_files,
    sanitize_category_name,
)

FOUND_EMOTIONS_KEY = "meme_manager_found_emotions"
PENDING_IMAGES_KEY = "meme_manager_pending_images"
TEMP_FILES_KEY = "meme_manager_temp_files"
STREAM_BUFFER_KEY = "meme_manager_stream_buffer"
STREAMING_RESULT_TYPE = "streaming_finish"


def _serialize_components(components: list[Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for component in components:
        if isinstance(component, Image):
            payloads.append(component_to_payload_sync(component))
    return payloads


def _deserialize_images(payloads: Any) -> list[Image]:
    images: list[Image] = []
    if not isinstance(payloads, list):
        return images
    for payload in payloads:
        component = payload_to_component(payload)
        if isinstance(component, Image):
            images.append(component)
    return images


class StarDotsConfig(BaseModel):
    key: str = ""
    secret: str = ""
    space: str = ""


class CloudflareR2Config(BaseModel):
    account_id: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    bucket_name: str = ""
    public_url: str = ""


class ImageHostConfig(BaseModel):
    stardots: StarDotsConfig = Field(default_factory=StarDotsConfig)
    cloudflare_r2: CloudflareR2Config = Field(default_factory=CloudflareR2Config)


class PromptConfig(BaseModel):
    prompt_head: str = (
        "\n\n你在对话中需根据当前情境智能选用表情符号，表情需用&&包裹，例如：&&happy&&。"
        "\n\n[表情标签库]（当前可用）\n格式：标签 - 使用场景描述\n当前可用：\n"
    )
    prompt_tail_1: str = "\n\n=== 安全控制体系 ===\n1. 使用频率：\n   • 日常对话：最多"
    prompt_tail_2: str = (
        "个表情\n   • 专业咨询：≤1个\n2. 强制校验规则：\n"
        "   a) 仅使用当前列表存在的标签\n"
        "   b) 高风险主题可放弃使用表情\n"
        "   c) 表情含义需与上下文情绪一致\n"
        "3. 如果没有合适标签，可以不输出表情。"
    )


class MemeManagerConfig(BaseModel):
    image_host: str = "stardots"
    image_host_config: ImageHostConfig = Field(default_factory=ImageHostConfig)
    webui_port: int = 5000
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    emotion_llm_enabled: bool = False
    emotion_llm_provider_id: str = ""
    convert_static_to_gif: bool = False
    max_emotions_per_message: int = 2
    emotions_probability: int = 50
    strict_max_emotions_per_message: bool = True
    enable_loose_emotion_matching: bool = True
    enable_alternative_markup: bool = True
    remove_invalid_alternative_markup: bool = True
    enable_repeated_emotion_detection: bool = True
    high_confidence_emotions: list[str] = Field(
        default_factory=lambda: list(DEFAULT_CATEGORY_DESCRIPTIONS)
    )
    content_cleanup_rule: str = "&&[a-zA-Z]*&&"
    enable_mixed_message: bool = True
    mixed_message_probability: int = 50
    streaming_compatibility: bool = False


@dataclass(slots=True)
class UploadState:
    category: str
    expire_at: float


class AstrbotPluginMemeManager(Star):
    def __init__(self) -> None:
        super().__init__()
        self._ctx: Context | None = None
        self._config = MemeManagerConfig()
        self._plugin_dir = Path(__file__).resolve().parent
        self._paths: PluginPaths | None = None
        self._category_manager: CategoryManager | None = None
        self._img_sync: ImageSync | None = None
        self._upload_states: dict[str, UploadState] = {}

    @validate_config(model=MemeManagerConfig)
    async def _validate_config(self) -> None:
        return None

    async def on_start(self, ctx: Any | None = None) -> None:
        await super().on_start(ctx)
        runtime_ctx = self._require_ctx(ctx)
        self._ctx = runtime_ctx

        config_payload = await runtime_ctx.metadata.get_plugin_config()
        self._config = MemeManagerConfig.model_validate(config_payload or {})

        data_dir = await runtime_ctx.get_data_dir()
        self._paths = build_paths(self._plugin_dir, data_dir)
        ensure_runtime_layout(self._paths)

        self._category_manager = CategoryManager(self._paths)
        self._category_manager.sync_with_filesystem()
        self._img_sync = self._build_img_sync()

    async def on_stop(self, ctx: Any | None = None) -> None:
        if self._img_sync is not None:
            self._img_sync.stop_sync()
        await super().on_stop(ctx)

    def _require_ctx(self, ctx: Any | None = None) -> Context:
        active_ctx = ctx or self._ctx
        if not isinstance(active_ctx, Context):
            raise AstrBotError.internal_error(
                "meme_manager requires a valid runtime Context."
            )
        return active_ctx

    def _require_paths(self) -> PluginPaths:
        if self._paths is None:
            raise AstrBotError.internal_error("meme_manager paths are not initialized.")
        return self._paths

    def _require_category_manager(self) -> CategoryManager:
        if self._category_manager is None:
            raise AstrBotError.internal_error(
                "meme_manager category manager is not initialized."
            )
        return self._category_manager

    def _build_img_sync(self) -> ImageSync | None:
        paths = self._require_paths()
        provider_type = self._config.image_host.strip().lower()

        if provider_type == "cloudflare_r2":
            provider_config = self._config.image_host_config.cloudflare_r2.model_dump()
            required_fields = (
                provider_config["account_id"],
                provider_config["access_key_id"],
                provider_config["secret_access_key"],
                provider_config["bucket_name"],
            )
        else:
            provider_type = "stardots"
            provider_config = self._config.image_host_config.stardots.model_dump()
            required_fields = (
                provider_config["key"],
                provider_config["secret"],
                provider_config["space"],
            )

        if not all(str(item).strip() for item in required_fields):
            return None
        return ImageSync(provider_config, paths.memes_dir, provider_type=provider_type)

    def _upload_key(self, event: MessageEvent) -> str:
        return f"{event.session_id}:{event.user_id}"

    def _category_mapping(self) -> dict[str, str]:
        return self._require_category_manager().get_descriptions()

    def _is_command_text(self, text: str) -> bool:
        normalized = text.strip()
        return normalized.startswith("/")

    def _is_meme_manager_command(self, text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text.lstrip("/").strip())
        return normalized == "表情管理" or normalized.startswith("表情管理 ")

    def _clamp_probability(self, value: int) -> int:
        return max(0, min(100, int(value)))

    def _should_send(self, probability: int) -> bool:
        return random.randint(1, 100) <= self._clamp_probability(probability)

    def _system_prompt_suffix(self) -> str:
        categories = self._category_mapping()
        if not categories:
            return ""
        return (
            self._config.prompt.prompt_head
            + dict_to_prompt_lines(categories)
            + self._config.prompt.prompt_tail_1
            + str(self._config.max_emotions_per_message)
            + self._config.prompt.prompt_tail_2
        )

    def _is_position_in_thinking_tags(self, text: str, position: int) -> bool:
        thinking_pattern = re.compile(
            r"<think(?:ing)?>.*?</think(?:ing)?>",
            re.DOTALL | re.IGNORECASE,
        )
        return any(
            match.start() <= position < match.end()
            for match in thinking_pattern.finditer(text)
        )

    def _is_likely_emotion_markup(self, markup: str, text: str, position: int) -> bool:
        before_text = text[:position].strip()
        after_text = text[position + len(markup) :].strip()

        has_chinese_before = bool(
            re.search(r"[\u4e00-\u9fff]", before_text[-1:] if before_text else "")
        )
        has_chinese_after = bool(
            re.search(r"[\u4e00-\u9fff]", after_text[:1] if after_text else "")
        )
        if has_chinese_before or has_chinese_after:
            return True
        if re.fullmatch(r"\[\d+\]", markup):
            return False
        if " " in markup[1:-1]:
            return False

        english_context_before = bool(re.search(r"[a-zA-Z]\s+$", before_text))
        english_context_after = bool(re.search(r"^\s+[a-zA-Z]", after_text))
        return not (english_context_before and english_context_after)

    def _is_likely_emotion(
        self,
        word: str,
        text: str,
        position: int,
        valid_emotions: set[str],
    ) -> bool:
        del valid_emotions
        before_text = text[:position].strip()
        after_text = text[position + len(word) :].strip()

        english_context_before = bool(re.search(r"[a-zA-Z]\s+$", before_text))
        english_context_after = bool(re.search(r"^\s+[a-zA-Z]", after_text))
        if english_context_before or english_context_after:
            return False

        has_chinese_before = bool(
            re.search(r"[\u4e00-\u9fff]", before_text[-1:] if before_text else "")
        )
        has_chinese_after = bool(
            re.search(r"[\u4e00-\u9fff]", after_text[:1] if after_text else "")
        )
        if has_chinese_before or has_chinese_after:
            return True
        if not before_text or before_text.endswith(
            ("。", "，", "！", "？", ".", ",", ":", ";", "!", "?", "\n")
        ):
            return True
        if (not before_text or before_text[-1] in " \t\n.,!?;:'\"()[]{}") and (
            not after_text or after_text[0] in " \t\n.,!?;:'\"()[]{}"
        ):
            return True
        return word in self._config.high_confidence_emotions

    async def _select_emotions_with_llm(
        self,
        ctx: Context,
        text: str,
        valid_emotions: set[str],
    ) -> list[str]:
        if (
            not self._config.emotion_llm_enabled
            or not text.strip()
            or not valid_emotions
        ):
            return []

        prompt = (
            "你是表情标签选择器，只能从给定标签中选择。\n"
            '请返回 JSON，例如 {"emotions":["happy"]}。\n'
            "只输出 JSON，不要解释。\n"
            f"可用标签: {', '.join(sorted(valid_emotions))}\n"
            f"文本: {text}"
        )
        raw_response = await ctx.llm.chat(
            prompt,
            provider_id=self._config.emotion_llm_provider_id.strip() or None,
        )
        parsed: Any = None
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", raw_response)
            if match is not None:
                parsed = json.loads(match.group(0))

        emotions = parsed.get("emotions") if isinstance(parsed, dict) else []
        if isinstance(emotions, str):
            emotions = [emotions]
        if not isinstance(emotions, list):
            return []
        return [
            item
            for item in emotions
            if isinstance(item, str) and item in valid_emotions
        ]

    async def _extract_emotions(
        self,
        ctx: Context,
        text: str,
    ) -> tuple[str, list[str]]:
        valid_emotions = set(self._category_mapping())
        if not valid_emotions:
            return text, []

        clean_text = text
        found_emotions: list[str] = []

        strict_matches: list[tuple[str, str]] = []
        for match in re.finditer(r"&&([^&]+)&&", clean_text):
            original = match.group(0)
            emotion = match.group(1).strip()
            strict_matches.append(
                (original, emotion if emotion in valid_emotions else "")
            )
        for original, emotion in strict_matches:
            clean_text = clean_text.replace(original, "", 1)
            if emotion:
                found_emotions.append(emotion)

        if self._config.enable_alternative_markup:
            for pattern in (r"\[([^\[\]]+)\]", r"\(([^()]+)\)"):
                replacements: list[tuple[str, str]] = []
                invalid_markup: list[str] = []
                for match in re.finditer(pattern, clean_text):
                    original = match.group(0)
                    emotion = match.group(1).strip()
                    if emotion in valid_emotions and (
                        pattern.startswith(r"\[")
                        or self._is_likely_emotion_markup(
                            original,
                            clean_text,
                            match.start(),
                        )
                    ):
                        replacements.append((original, emotion))
                    elif self._config.remove_invalid_alternative_markup:
                        invalid_markup.append(original)

                for invalid in invalid_markup:
                    clean_text = clean_text.replace(invalid, "", 1)
                for original, emotion in replacements:
                    clean_text = clean_text.replace(original, "", 1)
                    found_emotions.append(emotion)

        if self._config.enable_repeated_emotion_detection:
            for emotion in valid_emotions:
                if len(emotion) < 3:
                    continue
                repeat_count = (
                    2 if emotion in self._config.high_confidence_emotions else 3
                )
                pattern = f"({re.escape(emotion)})\\1{{{repeat_count - 1},}}"
                for match in list(re.finditer(pattern, clean_text)):
                    if self._is_position_in_thinking_tags(clean_text, match.start()):
                        continue
                    clean_text = clean_text.replace(match.group(0), "", 1)
                    found_emotions.append(emotion)

        if self._config.enable_loose_emotion_matching:
            for emotion in valid_emotions:
                pattern = r"\b(" + re.escape(emotion) + r")\b"
                matches = list(re.finditer(pattern, clean_text))
                for match in reversed(matches):
                    if self._is_position_in_thinking_tags(clean_text, match.start()):
                        continue
                    word = match.group(1)
                    if not self._is_likely_emotion(
                        word,
                        clean_text,
                        match.start(),
                        valid_emotions,
                    ):
                        continue
                    found_emotions.append(word)
                    clean_text = (
                        clean_text[: match.start()]
                        + clean_text[match.start() + len(word) :]
                    )

        try:
            found_emotions.extend(
                await self._select_emotions_with_llm(ctx, clean_text, valid_emotions)
            )
        except Exception as exc:
            ctx.logger.warning("meme_manager emotion llm selection failed: %s", exc)

        deduped: list[str] = []
        seen: set[str] = set()
        for emotion in found_emotions:
            if emotion in seen:
                continue
            seen.add(emotion)
            deduped.append(emotion)
            if self._config.strict_max_emotions_per_message and (
                len(deduped) >= self._config.max_emotions_per_message
            ):
                break

        clean_text = re.sub(r"&&+", "", clean_text)
        return clean_text.strip(), deduped

    def _cleanup_text(self, text: str) -> str:
        cleaned = text
        if self._config.content_cleanup_rule:
            cleaned = re.sub(self._config.content_cleanup_rule, "", cleaned)
        cleaned = re.sub(r"&&+", "", cleaned)
        return cleaned.strip()

    @staticmethod
    def _strip_streaming_emotion_markup(text: str) -> tuple[str, str]:
        if not text:
            return "", ""
        safe_parts: list[str] = []
        cursor = 0
        while cursor < len(text):
            start = text.find("&&", cursor)
            if start < 0:
                safe_parts.append(text[cursor:])
                return "".join(safe_parts), ""
            end = text.find("&&", start + 2)
            if end < 0:
                safe_parts.append(text[cursor:start])
                return "".join(safe_parts), text[start:]
            safe_parts.append(text[cursor:start])
            cursor = end + 2
        return "".join(safe_parts), ""

    @staticmethod
    def _result_content_type(event: MessageEvent) -> str:
        raw = event.raw if isinstance(event.raw, dict) else {}
        return str(raw.get("result_content_type", "")).strip().lower()

    def _convert_to_gif_if_needed(self, image_path: Path) -> tuple[Path, Path | None]:
        if (
            not self._config.convert_static_to_gif
            or image_path.suffix.lower() == ".gif"
        ):
            return image_path, None

        try:
            with PILImage.open(image_path) as image:
                if (image.format or "").upper() == "GIF":
                    return image_path, None

                temp_file = Path(tempfile.gettempdir()) / (
                    f"meme_manager_{int(time.time() * 1000)}_{random.randint(1000, 9999)}.gif"
                )
                if image.mode in ("RGBA", "LA") or (
                    image.mode == "P" and "transparency" in image.info
                ):
                    background = PILImage.new("RGB", image.size, (255, 255, 255))
                    if image.mode == "P":
                        image = image.convert("RGBA")
                    background.paste(image, mask=image.split()[3])
                    image = background
                else:
                    image = image.convert("RGB")
                image.save(temp_file, "GIF")
                return temp_file, temp_file
        except Exception:
            return image_path, None

    def _pick_random_image(self, emotion: str) -> tuple[Path | None, Path | None]:
        paths = self._require_paths()
        emotion_dir = paths.memes_dir / emotion
        if not emotion_dir.exists():
            return None, None

        candidates = [
            item
            for item in emotion_dir.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not candidates:
            return None, None

        selected = random.choice(candidates)
        return self._convert_to_gif_if_needed(selected)

    def _build_image_components(
        self,
        emotions: list[str],
    ) -> tuple[list[Image], list[str]]:
        components: list[Image] = []
        temp_files: list[str] = []
        for emotion in emotions:
            selected, temp_path = self._pick_random_image(emotion)
            if selected is None:
                continue
            if temp_path is not None:
                temp_files.append(str(temp_path))
            components.append(Image.fromFileSystem(str(selected)))
        return components, temp_files

    @staticmethod
    def _http_first_value(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if isinstance(value, list):
            return str(value[0]).strip() if value else ""
        return str(value).strip() if value is not None else ""

    @staticmethod
    def _http_json_body(payload: dict[str, Any]) -> dict[str, Any]:
        body = payload.get("json_body")
        if isinstance(body, dict):
            return body
        text_body = str(payload.get("text_body", "")).strip()
        if not text_body:
            return {}
        try:
            parsed = json.loads(text_body)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _http_files(payload: dict[str, Any]) -> list[dict[str, Any]]:
        files = payload.get("files")
        if not isinstance(files, list):
            return []
        return [item for item in files if isinstance(item, dict)]

    @staticmethod
    def _http_mapping_first_value(
        payload: dict[str, Any],
        mapping_name: str,
        key: str,
    ) -> str:
        mapping = payload.get(mapping_name)
        if not isinstance(mapping, dict):
            return ""
        value = mapping.get(key)
        if isinstance(value, list):
            return str(value[0]).strip() if value else ""
        return str(value).strip() if value is not None else ""

    @classmethod
    def _http_int_value(
        cls,
        payload: dict[str, Any],
        mapping_name: str,
        key: str,
        *,
        default: int,
        minimum: int = 0,
        maximum: int | None = None,
    ) -> int:
        raw_value = cls._http_mapping_first_value(payload, mapping_name, key)
        if not raw_value:
            return default
        try:
            parsed = int(raw_value)
        except ValueError:
            return default
        value = max(minimum, parsed)
        if maximum is not None:
            value = min(maximum, value)
        return value

    @staticmethod
    def _json_response(
        body: dict[str, Any] | list[Any],
        *,
        status: int = 200,
    ) -> dict[str, Any]:
        return {"status": status, "body": body}

    @staticmethod
    def _error_response(message: str, *, status: int = 400) -> dict[str, Any]:
        return {"status": status, "body": {"message": message}}

    def _plugin_public_base(self) -> str:
        return "/plug/astrbot_plugin_meme_manager"

    def _plugin_api_base(self) -> str:
        return "/api/plug/astrbot_plugin_meme_manager"

    def _preview_url(self, category: str, filename: str) -> str:
        return (
            f"{self._plugin_public_base()}/image"
            f"?category={quote(category)}&filename={quote(filename)}"
        )

    def _category_dir(self, category: str) -> Path:
        return self._require_paths().memes_dir / category

    def _resolve_image_path(self, category: str, filename: str) -> Path | None:
        safe_category = sanitize_category_name(category)
        safe_filename = Path(filename).name
        if not safe_category or not safe_filename:
            return None
        file_path = self._category_dir(safe_category) / safe_filename
        if not file_path.exists() or not file_path.is_file():
            return None
        return file_path

    def _scan_emoji_payload(self) -> dict[str, list[dict[str, str]]]:
        result: dict[str, list[dict[str, str]]] = {}
        for category, files in list_category_files(
            self._require_paths().memes_dir
        ).items():
            result[category] = [
                {
                    "filename": filename,
                    "preview_url": self._preview_url(category, filename),
                }
                for filename in files
            ]
        return result

    def _library_payload(self) -> dict[str, Any]:
        files = list_category_files(self._require_paths().memes_dir)
        descriptions = self._category_mapping()
        categories: list[dict[str, Any]] = []
        total_files = 0
        for category in sorted(set(descriptions) | set(files)):
            count = len(files.get(category, []))
            total_files += count
            categories.append(
                {
                    "category": category,
                    "count": count,
                    "description": descriptions.get(category, "请添加描述"),
                }
            )
        return {
            "descriptions": descriptions,
            "categories": categories,
            "total_files": total_files,
        }

    async def _file_url_for_path(self, file_path: Path) -> str:
        ctx = self._require_ctx()
        token = await ctx.files.register_file(str(file_path), timeout=300)
        return f"/api/file/{token}"

    async def _emoji_page_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        category = sanitize_category_name(
            self._http_mapping_first_value(payload, "query", "category")
        )
        if not category:
            raise AstrBotError.invalid_input("category query parameter is required")

        offset = self._http_int_value(
            payload,
            "query",
            "offset",
            default=0,
            minimum=0,
        )
        limit = self._http_int_value(
            payload,
            "query",
            "limit",
            default=24,
            minimum=1,
            maximum=48,
        )
        files = list_category_files(self._require_paths().memes_dir)
        filenames = files.get(category, [])
        page_filenames = filenames[offset : offset + limit]
        file_paths = [
            self._resolve_image_path(category, filename) for filename in page_filenames
        ]
        preview_urls = await asyncio.gather(
            *[
                self._file_url_for_path(file_path)
                for file_path in file_paths
                if file_path is not None
            ]
        )
        items: list[dict[str, str]] = []
        url_index = 0
        for filename, file_path in zip(page_filenames, file_paths, strict=False):
            if file_path is None:
                continue
            items.append(
                {
                    "filename": filename,
                    "preview_url": preview_urls[url_index],
                }
            )
            url_index += 1
        next_offset = offset + len(items)
        return {
            "category": category,
            "total": len(filenames),
            "offset": offset,
            "limit": limit,
            "next_offset": next_offset,
            "has_more": next_offset < len(filenames),
            "items": items,
        }

    def _sync_config_payload(self) -> dict[str, Any]:
        missing_in_config, deleted_categories = (
            self._require_category_manager().get_sync_status()
        )
        return {
            "status": "ok",
            "differences": {
                "missing_in_config": missing_in_config,
                "deleted_categories": deleted_categories,
            },
        }

    def _cloud_sync_status_payload(self) -> dict[str, Any]:
        if self._img_sync is None:
            raise AstrBotError.invalid_input("图床服务未配置")
        return self._img_sync.check_status()

    def _sync_process_payload(self) -> dict[str, Any]:
        if self._img_sync is None or self._img_sync.sync_process is None:
            return {"completed": True, "success": True}

        process = self._img_sync.sync_process
        if process.is_alive():
            return {
                "completed": False,
                "pid": process.pid,
            }

        success = process.exitcode == 0
        self._img_sync.sync_process = None
        if success:
            self._require_category_manager().sync_with_filesystem()
        return {
            "completed": True,
            "success": success,
            "exit_code": process.exitcode,
        }

    def _start_background_sync(self, task: str) -> dict[str, Any]:
        if self._img_sync is None:
            return self._error_response("图床服务未配置")
        if (
            self._img_sync.sync_process is not None
            and self._img_sync.sync_process.is_alive()
        ):
            return self._error_response("已有正在运行的同步任务", status=409)

        self._img_sync.sync_process = self._img_sync._start_sync_process(task)
        return self._json_response(
            {
                "success": True,
                "task": task,
                "pid": self._img_sync.sync_process.pid,
            }
        )

    def _build_overview_html(self) -> str:
        html = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Meme Manager</title>
  <style>
    :root {
      --bg: #f5f7fb;
      --panel: #ffffff;
      --line: #d8dfeb;
      --text: #172033;
      --muted: #5f6b85;
      --accent: #2f6fed;
      --accent-soft: #e9f0ff;
      --danger: #d64545;
      --shadow: 0 14px 36px rgba(20, 32, 58, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", sans-serif;
      background: linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%);
      color: var(--text);
    }
    .page {
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }
    .hero, .sidebar, .category-card {
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      border-radius: 18px;
    }
    .hero {
      padding: 24px 28px;
      margin-bottom: 20px;
      display: flex;
      gap: 18px;
      align-items: flex-start;
      justify-content: space-between;
      flex-wrap: wrap;
    }
    .hero h1 { margin: 0 0 8px; font-size: 34px; }
    .hero p { margin: 0; color: var(--muted); }
    .hero-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .layout {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 20px;
    }
    .sidebar {
      padding: 18px;
      position: sticky;
      top: 16px;
      height: fit-content;
    }
    .sidebar h2, .category-card h3 {
      margin: 0 0 12px;
      font-size: 18px;
    }
    .status-box, .toolbar, .category-toolbar {
      display: grid;
      gap: 10px;
    }
    .toolbar {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 12px;
    }
    .status-item {
      padding: 12px 14px;
      border-radius: 14px;
      background: #f8faff;
      border: 1px solid var(--line);
    }
    .status-item strong { display: block; margin-bottom: 4px; }
    .content {
      display: grid;
      gap: 18px;
    }
    .button, button {
      border: 0;
      cursor: pointer;
      border-radius: 12px;
      padding: 11px 14px;
      font-size: 14px;
      font-weight: 600;
      transition: transform 120ms ease, opacity 120ms ease;
    }
    .button:hover, button:hover { transform: translateY(-1px); }
    .button-primary { background: var(--accent); color: #fff; }
    .button-soft { background: var(--accent-soft); color: var(--accent); }
    .button-danger { background: #fff1f1; color: var(--danger); }
    .button-muted { background: #eef2f8; color: var(--text); }
    .category-card {
      padding: 18px;
    }
    .category-header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 14px;
      flex-wrap: wrap;
    }
    .category-header p {
      margin: 6px 0 0;
      color: var(--muted);
      max-width: 720px;
    }
    .category-toolbar {
      grid-template-columns: repeat(4, minmax(0, max-content));
    }
    .emoji-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
      gap: 14px;
    }
    .emoji-item, .upload-card {
      border-radius: 16px;
      border: 1px solid var(--line);
      background: #fbfcff;
      overflow: hidden;
      min-height: 178px;
    }
    .emoji-item img {
      width: 100%;
      height: 140px;
      object-fit: contain;
      display: block;
      background: linear-gradient(180deg, #f8faff 0%, #eef3fb 100%);
    }
    .emoji-meta {
      padding: 10px;
      display: grid;
      gap: 8px;
    }
    .emoji-meta code {
      display: block;
      font-size: 12px;
      color: var(--muted);
      word-break: break-all;
    }
    .upload-card {
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 12px;
      border-style: dashed;
    }
    .upload-card input { display: none; }
    .notice {
      margin-top: 12px;
      padding: 12px 14px;
      border-radius: 12px;
      background: #fff7db;
      color: #7f5d00;
      border: 1px solid #f0d88b;
      display: none;
    }
    .empty-state {
      padding: 28px;
      text-align: center;
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.72);
    }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { position: static; }
      .toolbar, .category-toolbar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div>
        <h1>Meme Manager</h1>
        <p>SDK 版管理页。页面可直开，数据接口继续使用 Dashboard token 请求。</p>
      </div>
      <div class="hero-actions">
        <button class="button button-primary" id="refresh-btn">刷新图库</button>
        <button class="button button-soft" id="add-category-btn">新增分类</button>
      </div>
    </section>
    <div class="notice" id="notice"></div>
    <div class="layout">
      <aside class="sidebar">
        <h2>同步与统计</h2>
        <div class="status-box" id="summary-box"></div>
        <div class="toolbar">
          <button class="button button-muted" id="sync-config-btn">同步配置</button>
          <button class="button button-muted" id="check-config-btn">检查差异</button>
          <button class="button button-soft" id="upload-sync-btn">同步到云端</button>
          <button class="button button-soft" id="download-sync-btn">从云端同步</button>
        </div>
      </aside>
      <main class="content" id="category-list">
        <div class="empty-state">正在加载图库...</div>
      </main>
    </div>
  </div>
  <script>
    const API_BASE = "__API_BASE__";
    const state = {
      categories: [],
      descriptions: {},
      itemsByCategory: {},
      pageInfo: {},
      sync: null,
      cloud: null,
      polling: null,
    };

    function authHeaders(extra = {}) {
      const token = window.localStorage.getItem("token");
      return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
    }

    async function api(path, options = {}) {
      const response = await fetch(`${API_BASE}${path}`, {
        ...options,
        headers: authHeaders(options.headers || {}),
      });
      const contentType = response.headers.get("content-type") || "";
      const isJson = contentType.includes("application/json");
      const payload = isJson ? await response.json() : await response.text();
      if (!response.ok) {
        const message = typeof payload === "string"
          ? payload
          : payload?.message || payload?.error || `HTTP ${response.status}`;
        throw new Error(message);
      }
      return payload;
    }

    function showNotice(message, isError = false) {
      const notice = document.getElementById("notice");
      notice.style.display = "block";
      notice.textContent = message;
      notice.style.background = isError ? "#fff1f1" : "#fff7db";
      notice.style.borderColor = isError ? "#efb0b0" : "#f0d88b";
      notice.style.color = isError ? "#8b1e1e" : "#7f5d00";
    }

    function hideNotice() {
      document.getElementById("notice").style.display = "none";
    }

    async function refreshAll() {
      try {
        hideNotice();
        const [library, configStatus, cloudStatus] = await Promise.all([
          api("/api/library"),
          api("/api/sync/status"),
          api("/api/img_host/sync/status").catch((error) => ({ error: error.message })),
        ]);
        state.categories = library.categories || [];
        state.descriptions = library.descriptions || {};
        state.itemsByCategory = {};
        state.pageInfo = {};
        state.sync = configStatus;
        state.cloud = cloudStatus;
        renderSummary();
        renderCategories();
        if (state.categories.length) {
          void loadCategoryPage(state.categories[0].category, { reset: true });
        }
      } catch (error) {
        showNotice(`加载失败：${error.message}`, true);
      }
    }

    function renderSummary() {
      const summaryBox = document.getElementById("summary-box");
      const categoryCount = state.categories.length;
      const totalFiles = state.categories.reduce((count, item) => count + (item.count || 0), 0);
      const sync = state.sync?.differences || {};
      const cloud = state.cloud || {};
      summaryBox.innerHTML = `
        <div class="status-item"><strong>本地分类</strong>${categoryCount} 个</div>
        <div class="status-item"><strong>本地文件</strong>${totalFiles} 张</div>
        <div class="status-item"><strong>仅文件系统存在</strong>${(sync.missing_in_config || []).length} 个</div>
        <div class="status-item"><strong>仅配置存在</strong>${(sync.deleted_categories || []).length} 个</div>
        <div class="status-item"><strong>待上传</strong>${(cloud.to_upload || []).length} 个</div>
        <div class="status-item"><strong>待下载</strong>${(cloud.to_download || []).length} 个</div>
      `;
    }

    function renderCategories() {
      const container = document.getElementById("category-list");
      if (!state.categories.length) {
        container.innerHTML = '<div class="empty-state">当前还没有任何表情分类。</div>';
        return;
      }
      container.innerHTML = "";
      for (const item of state.categories) {
        const category = item.category;
        const items = state.itemsByCategory[category] || [];
        const pageInfo = state.pageInfo[category] || { has_more: (item.count || 0) > 0, loading: false };
        const card = document.createElement("section");
        card.className = "category-card";
        const description = item.description || state.descriptions?.[category] || "请添加描述";
        card.innerHTML = `
          <div class="category-header">
            <div>
              <h3>${category}</h3>
              <p style="margin-bottom:8px;color:#172033;font-weight:600;">共 ${item.count || 0} 张</p>
              <p>${description}</p>
            </div>
            <div class="category-toolbar">
              <button class="button button-muted" data-action="edit" data-category="${category}">改描述</button>
              <button class="button button-muted" data-action="rename" data-category="${category}">重命名</button>
              <button class="button button-danger" data-action="delete-category" data-category="${category}">删分类</button>
            </div>
          </div>
          <div class="emoji-grid"></div>
        `;
        const grid = card.querySelector(".emoji-grid");
        const uploadCard = document.createElement("label");
        uploadCard.className = "upload-card";
        uploadCard.innerHTML = `
          <div>
            <strong>上传到 ${category}</strong>
            <div style="margin-top:8px;color:#5f6b85;font-size:13px;">点击或拖拽图片到这里</div>
            <input type="file" multiple accept="image/*">
          </div>
        `;
        const fileInput = uploadCard.querySelector("input");
        fileInput.addEventListener("change", (event) => uploadFiles(category, event.target.files));
        uploadCard.addEventListener("dragover", (event) => event.preventDefault());
        uploadCard.addEventListener("drop", (event) => {
          event.preventDefault();
          uploadFiles(category, event.dataTransfer.files);
        });
        grid.appendChild(uploadCard);

        for (const item of items) {
          const emoji = document.createElement("article");
          emoji.className = "emoji-item";
          emoji.innerHTML = `
            <img src="${item.preview_url}" alt="${item.filename}">
            <div class="emoji-meta">
              <code>${item.filename}</code>
              <button class="button button-danger" data-action="delete-image" data-category="${category}" data-filename="${item.filename}">删除图片</button>
            </div>
          `;
          grid.appendChild(emoji);
        }

        if (!items.length) {
          const placeholder = document.createElement("div");
          placeholder.className = "empty-state";
          placeholder.textContent = pageInfo.loading ? "正在加载图片..." : "尚未加载该分类图片。";
          grid.appendChild(placeholder);
        }

        if ((item.count || 0) > 0) {
          const more = document.createElement("button");
          more.className = "button button-soft";
          more.dataset.action = "load-more";
          more.dataset.category = category;
          more.textContent = pageInfo.loading
            ? "加载中..."
            : items.length
              ? (pageInfo.has_more ? "加载更多" : "已加载完毕")
              : "加载图片";
          more.disabled = !!pageInfo.loading || !pageInfo.has_more && items.length > 0;
          card.appendChild(more);
        }
        container.appendChild(card);
      }
    }

    async function loadCategoryPage(category, { reset = false } = {}) {
      const current = state.pageInfo[category] || { next_offset: 0, has_more: true };
      if (current.loading) return;
      const offset = reset ? 0 : (current.next_offset || 0);
      state.pageInfo[category] = { ...current, loading: true };
      renderCategories();
      try {
        const payload = await api(`/api/emoji?category=${encodeURIComponent(category)}&offset=${offset}&limit=24`);
        const existing = reset ? [] : (state.itemsByCategory[category] || []);
        state.itemsByCategory[category] = existing.concat(payload.items || []);
        state.pageInfo[category] = {
          next_offset: payload.next_offset || state.itemsByCategory[category].length,
          has_more: !!payload.has_more,
          loading: false,
        };
      } catch (error) {
        state.pageInfo[category] = { ...current, loading: false };
        showNotice(`加载 ${category} 失败：${error.message}`, true);
      }
      renderCategories();
    }

    async function uploadFiles(category, fileList) {
      const files = Array.from(fileList || []).filter((file) => file.type.startsWith("image/"));
      if (!files.length) return;
      for (const file of files) {
        const formData = new FormData();
        formData.append("category", category);
        formData.append("image_file", file);
        try {
          await api("/api/emoji/add", { method: "POST", body: formData });
        } catch (error) {
          showNotice(`上传失败：${error.message}`, true);
          return;
        }
      }
      await refreshAll();
      showNotice(`已上传 ${files.length} 张图片到 ${category}`);
    }

    async function mutate(path, body, successMessage) {
      try {
        await api(path, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body || {}),
        });
        await refreshAll();
        if (successMessage) showNotice(successMessage);
      } catch (error) {
        showNotice(`操作失败：${error.message}`, true);
      }
    }

    async function runSync(path, message) {
      try {
        await api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
        showNotice(message);
        startSyncPolling();
      } catch (error) {
        showNotice(`同步失败：${error.message}`, true);
      }
    }

    function startSyncPolling() {
      if (state.polling) window.clearInterval(state.polling);
      state.polling = window.setInterval(async () => {
        try {
          const status = await api("/api/img_host/sync/check_process");
          if (status.completed) {
            window.clearInterval(state.polling);
            state.polling = null;
            await refreshAll();
            showNotice(status.success ? "图床同步完成" : "图床同步失败", !status.success);
          }
        } catch (error) {
          window.clearInterval(state.polling);
          state.polling = null;
          showNotice(`同步轮询失败：${error.message}`, true);
        }
      }, 2000);
    }

    document.addEventListener("click", async (event) => {
      const target = event.target.closest("button[data-action]");
      if (!target) return;
      const category = target.dataset.category;
      const filename = target.dataset.filename;
      const action = target.dataset.action;
      if (action === "delete-image" && category && filename) {
        if (confirm(`删除图片 ${filename}？`)) {
          await mutate("/api/emoji/delete", { category, image_file: filename }, `已删除 ${filename}`);
        }
      }
      if (action === "delete-category" && category) {
        if (confirm(`删除分类 ${category}？这会删除整个目录。`)) {
          await mutate("/api/category/delete", { category }, `已删除分类 ${category}`);
        }
      }
      if (action === "edit" && category) {
        const value = prompt(`输入 ${category} 的新描述`, state.descriptions?.[category] || "");
        if (value !== null) {
          await mutate("/api/category/update_description", { tag: category, description: value }, `已更新 ${category} 描述`);
        }
      }
      if (action === "rename" && category) {
        const value = prompt(`输入 ${category} 的新名称`, category);
        if (value && value !== category) {
          await mutate("/api/category/rename", { old_name: category, new_name: value }, `已重命名 ${category} -> ${value}`);
        }
      }
      if (action === "load-more" && category) {
        await loadCategoryPage(category);
      }
    });

    document.getElementById("refresh-btn").addEventListener("click", refreshAll);
    document.getElementById("check-config-btn").addEventListener("click", async () => {
      await refreshAll();
      showNotice("已刷新配置差异状态");
    });
    document.getElementById("sync-config-btn").addEventListener("click", async () => {
      await mutate("/api/sync/config", {}, "已同步配置与文件夹结构");
    });
    document.getElementById("upload-sync-btn").addEventListener("click", async () => {
      await runSync("/api/img_host/sync/upload", "正在同步到云端...");
    });
    document.getElementById("download-sync-btn").addEventListener("click", async () => {
      await runSync("/api/img_host/sync/download", "正在从云端同步...");
    });
    document.getElementById("add-category-btn").addEventListener("click", async () => {
      const category = prompt("输入新分类名称");
      if (!category) return;
      const description = prompt("输入分类描述", "请添加描述") || "请添加描述";
      await mutate("/api/category/restore", { category, description }, `已创建分类 ${category}`);
    });

    if (!window.localStorage.getItem("token")) {
      showNotice("当前页面需要已登录 Dashboard 的同源 localStorage token 才能调用管理接口。请先登录 Dashboard 后再打开此页。", true);
    }
    refreshAll();
  </script>
</body>
</html>"""
        return html.replace("__API_BASE__", self._plugin_api_base())

    def _merge_components_with_images(
        self,
        components: list[Any],
        images: list[Image],
    ) -> list[Any]:
        if not images:
            return components
        if not components:
            return list(images)

        plain_indexes = [
            index
            for index, component in enumerate(components)
            if isinstance(component, Plain)
        ]
        if not plain_indexes:
            return components + list(images)

        merged = list(components)
        image_index = 0
        inserted = 0
        images_per_text = max(1, len(images) // len(plain_indexes))

        for plain_offset, plain_index in enumerate(plain_indexes):
            if image_index >= len(images):
                break
            if plain_offset == len(plain_indexes) - 1:
                count = len(images) - image_index
            else:
                count = min(images_per_text, len(images) - image_index)
            insert_at = plain_index + 1 + inserted
            for _ in range(count):
                merged.insert(insert_at, images[image_index])
                image_index += 1
                insert_at += 1
                inserted += 1
        return merged

    async def _save_images_to_category(
        self,
        event: MessageEvent,
        category: str,
    ) -> list[str]:
        paths = self._require_paths()
        target_dir = paths.memes_dir / category
        saved_files: list[str] = []

        for image in event.get_images():
            file_path = Path(await image.convert_to_file_path())
            saved_path = copy_image_to_category(
                source_path=file_path,
                target_dir=target_dir,
            )
            saved_files.append(saved_path.name)
        return saved_files

    @on_command(
        "查看图库",
        group="表情管理",
        description="List available meme categories",
        group_help="Legacy meme manager commands migrated to the SDK platform",
    )
    async def list_library(self, event: MessageEvent) -> None:
        mapping = self._category_mapping()
        stats, total = library_stats(self._require_paths().memes_dir)
        if not mapping:
            await event.reply("当前还没有可用的表情分类。")
            return
        lines = [f"当前图库分类 {len(mapping)} 个，共 {total} 张图片："]
        for category, description in sorted(mapping.items()):
            lines.append(f"- {category} ({stats.get(category, 0)}): {description}")
        await event.reply("\n".join(lines))

    @on_command(
        "开启管理后台", group="表情管理", description="Show SDK HTTP entrypoint"
    )
    @require_admin
    async def start_webui(self, event: MessageEvent) -> None:
        await event.reply(
            "SDK 版不再自启独立 WebUI 进程。请直接打开 "
            "`/plug/astrbot_plugin_meme_manager` 进入管理页，"
            "页面会通过 `/api/plug/astrbot_plugin_meme_manager/*` 调用后台接口。"
        )

    @on_command("关闭管理后台", group="表情管理", description="No-op for SDK runtime")
    @require_admin
    async def stop_webui(self, event: MessageEvent) -> None:
        await event.reply("SDK 版没有独立 WebUI 进程需要关闭。")

    @on_command(
        "添加表情", group="表情管理", description="Upload images into a category"
    )
    @require_admin
    async def upload_meme(
        self,
        event: MessageEvent,
        category: str | None = None,
    ) -> None:
        normalized_category = sanitize_category_name(category or "")
        if not normalized_category:
            await event.reply(
                "用法：/表情管理 添加表情 <类别>，然后在 30 秒内发送图片消息。"
            )
            return

        if normalized_category not in self._category_mapping():
            await event.reply(
                f"无效类别：{normalized_category}。请先用 `/表情管理 查看图库` 查看可用类别。"
            )
            return

        self._upload_states[self._upload_key(event)] = UploadState(
            category=normalized_category,
            expire_at=time.time() + 30,
        )
        await event.reply(
            f"请在 30 秒内发送要添加到 `{normalized_category}` 的图片，可一次发送多张。"
        )

    @on_command(
        "同步状态", group="表情管理", description="Show local/remote sync status"
    )
    async def check_sync_status(self, event: MessageEvent) -> None:
        if self._img_sync is None:
            await event.reply("当前未配置图床同步。")
            return

        status = self._img_sync.check_status()
        to_upload = status.get("to_upload", [])
        to_download = status.get("to_download", [])
        lines = [
            "图床同步状态：",
            f"- 待上传: {len(to_upload)}",
            f"- 待下载: {len(to_download)}",
        ]
        if to_upload:
            lines.append(
                "- 上传示例: " + ", ".join(item["filename"] for item in to_upload[:5])
            )
        if to_download:
            lines.append(
                "- 下载示例: " + ", ".join(item["filename"] for item in to_download[:5])
            )
        if not to_upload and not to_download:
            lines.append("- 本地与云端已同步")
        await event.reply("\n".join(lines))

    @on_command(
        "同步到云端", group="表情管理", description="Upload local memes to remote"
    )
    @require_admin
    async def sync_to_remote(self, event: MessageEvent) -> None:
        if self._img_sync is None:
            await event.reply("当前未配置图床同步。")
            return
        await event.reply("正在同步到云端...")
        success = await self._img_sync.start_sync("upload")
        await event.reply(
            "同步到云端完成。" if success else "同步到云端失败，请查看日志。"
        )

    @on_command(
        "从云端同步", group="表情管理", description="Download remote memes to local"
    )
    @require_admin
    async def sync_from_remote(self, event: MessageEvent) -> None:
        if self._img_sync is None:
            await event.reply("当前未配置图床同步。")
            return
        await event.reply("正在从云端同步...")
        success = await self._img_sync.start_sync("download")
        if success:
            self._require_category_manager().sync_with_filesystem()
        await event.reply(
            "从云端同步完成。" if success else "从云端同步失败，请查看日志。"
        )

    @on_command("覆盖到云端", group="表情管理", description="Make remote match local")
    @require_admin
    async def overwrite_to_remote(self, event: MessageEvent) -> None:
        if self._img_sync is None:
            await event.reply("当前未配置图床同步。")
            return
        await event.reply("正在覆盖到云端...")
        success = await self._img_sync.start_sync("overwrite_to_remote")
        await event.reply(
            "覆盖到云端完成。" if success else "覆盖到云端失败，请查看日志。"
        )

    @on_command("从云端覆盖", group="表情管理", description="Make local match remote")
    @require_admin
    async def overwrite_from_remote(self, event: MessageEvent) -> None:
        if self._img_sync is None:
            await event.reply("当前未配置图床同步。")
            return
        await event.reply("正在从云端覆盖本地...")
        success = await self._img_sync.start_sync("overwrite_from_remote")
        if success:
            self._require_category_manager().sync_with_filesystem()
        await event.reply(
            "从云端覆盖完成。" if success else "从云端覆盖失败，请查看日志。"
        )

    @on_command(
        "图库统计", group="表情管理", description="Show local and remote meme stats"
    )
    async def show_library_stats(self, event: MessageEvent) -> None:
        local_stats, local_total = library_stats(self._require_paths().memes_dir)
        lines = [
            "图库统计：",
            f"- 本地分类数: {len(local_stats)}",
            f"- 本地文件数: {local_total}",
        ]
        for category, count in sorted(local_stats.items()):
            lines.append(f"- {category}: {count}")

        if self._img_sync is not None:
            try:
                remote_files = self._img_sync.provider.get_image_list()
                lines.append(f"- 云端文件数: {len(remote_files)}")
            except Exception as exc:
                lines.append(f"- 云端统计失败: {exc}")
        else:
            lines.append("- 云端统计: 未配置图床")
        await event.reply("\n".join(lines))

    @on_message(
        regex=r"(?s)^.*$", description="Capture image upload flow for meme manager"
    )
    async def handle_upload_image(self, event: MessageEvent) -> None:
        upload_state = self._upload_states.get(self._upload_key(event))
        if upload_state is None:
            return
        if time.time() > upload_state.expire_at:
            self._upload_states.pop(self._upload_key(event), None)
            return

        images = event.get_images()
        if not images:
            await event.reply("请发送图片文件来完成上传。")
            return

        saved_files = await self._save_images_to_category(event, upload_state.category)
        self._upload_states.pop(self._upload_key(event), None)
        self._require_category_manager().sync_with_filesystem()

        lines = [f"已将 {len(saved_files)} 张图片加入 `{upload_state.category}`。"]
        if self._img_sync is not None:
            lines.append("如需同步图床，请执行 `/表情管理 同步到云端`。")
        await event.reply("\n".join(lines))

    @on_event("llm_request", description="Append meme guidance to the system prompt")
    @priority(99999)
    async def on_llm_request(
        self,
        event: MessageEvent,
        request: ProviderRequest,
    ) -> None:
        command_text = request.prompt or event.text
        if self._is_command_text(command_text) or self._is_meme_manager_command(
            command_text
        ):
            return
        suffix = self._system_prompt_suffix().strip()
        if not suffix:
            return
        request.system_prompt = (
            f"{request.system_prompt}\n{suffix}".strip()
            if request.system_prompt
            else suffix
        )

    @on_event("llm_response", description="Extract emotion tags from the LLM response")
    @priority(99999)
    async def on_llm_response(
        self,
        event: MessageEvent,
        response: LLMResponse,
        ctx: Context,
    ) -> None:
        if not response.text.strip():
            return
        clean_text, emotions = await self._extract_emotions(ctx, response.text)
        response.text = clean_text
        event.set_extra(FOUND_EMOTIONS_KEY, emotions)
        event.set_extra(STREAM_BUFFER_KEY, "")
        ctx.logger.info(
            "meme_manager llm_response: session={} emotions={} cleaned_text={}",
            event.session_id,
            emotions,
            clean_text,
        )

    @on_event(
        "streaming_delta",
        description="Strip meme markup from streaming text before it is sent",
    )
    @priority(99999)
    async def on_streaming_delta(
        self,
        event: MessageEvent,
        result: MessageEventResult,
    ) -> None:
        if result.type != EventResultType.CHAIN:
            return

        pending_buffer = str(event.get_extra(STREAM_BUFFER_KEY, ""))
        components: list[Any] = []
        for component in list(result.chain.components):
            if not isinstance(component, Plain):
                components.append(component)
                continue
            cleaned_text, pending_buffer = self._strip_streaming_emotion_markup(
                pending_buffer + component.text
            )
            if cleaned_text.strip():
                components.append(Plain(cleaned_text, convert=False))

        event.set_extra(STREAM_BUFFER_KEY, pending_buffer)
        if components:
            result.type = EventResultType.CHAIN
            result.chain = MessageChain(components)
        else:
            result.type = EventResultType.EMPTY
            result.chain = MessageChain([])

        ctx = self._require_ctx()
        ctx.logger.debug(
            "meme_manager streaming_delta: session={} components={} pending_buffer={}",
            event.session_id,
            len(result.chain.components),
            pending_buffer,
        )

    @on_event(
        "decorating_result", description="Attach selected meme images to the reply"
    )
    @priority(99999)
    async def on_decorating_result(
        self,
        event: MessageEvent,
        result: MessageEventResult,
    ) -> None:
        if result.type not in {EventResultType.CHAIN, EventResultType.EMPTY}:
            return
        if result.type == EventResultType.EMPTY and not event.get_extra(
            FOUND_EMOTIONS_KEY
        ):
            return

        components: list[Any] = []
        for component in list(result.chain.components):
            if isinstance(component, Plain):
                cleaned = self._cleanup_text(component.text)
                if cleaned:
                    components.append(Plain(cleaned, convert=False))
            else:
                components.append(component)

        emotions = [
            item
            for item in event.get_extra(FOUND_EMOTIONS_KEY, [])
            if isinstance(item, str)
        ]
        is_streaming_finish = self._result_content_type(event) == STREAMING_RESULT_TYPE
        ctx = self._require_ctx()
        ctx.logger.info(
            "meme_manager decorating_result: session={} emotions={} streaming_finish={} result_type={}",
            event.session_id,
            emotions,
            is_streaming_finish,
            result.type.value,
        )
        if emotions and self._should_send(self._config.emotions_probability):
            image_components, temp_files = self._build_image_components(emotions)
            ctx.logger.info(
                "meme_manager image_selection: session={} selected_images={} temp_files={}",
                event.session_id,
                len(image_components),
                len(temp_files),
            )
            if temp_files:
                existing = list(event.get_extra(TEMP_FILES_KEY, []))
                event.set_extra(TEMP_FILES_KEY, existing + temp_files)
            if image_components:
                if (
                    not is_streaming_finish
                    and self._config.enable_mixed_message
                    and self._should_send(self._config.mixed_message_probability)
                ):
                    components = self._merge_components_with_images(
                        components,
                        image_components,
                    )
                    ctx.logger.info(
                        "meme_manager decorating_result: session={} merged_images_into_chain={}",
                        event.session_id,
                        len(image_components),
                    )
                else:
                    event.set_extra(
                        PENDING_IMAGES_KEY,
                        _serialize_components(image_components),
                    )
                    ctx.logger.info(
                        "meme_manager decorating_result: session={} deferred_images={}",
                        event.session_id,
                        len(image_components),
                    )
        elif emotions:
            ctx.logger.info(
                "meme_manager decorating_result: session={} skipped_by_probability emotions_probability={}",
                event.session_id,
                self._config.emotions_probability,
            )
        else:
            ctx.logger.info(
                "meme_manager decorating_result: session={} no_emotions_found",
                event.session_id,
            )

        if components:
            result.type = EventResultType.CHAIN
            result.chain = MessageChain(components)
        else:
            result.type = EventResultType.EMPTY
            result.chain = MessageChain([])

    @on_event(
        "after_message_sent",
        description="Send deferred meme images after the main reply",
    )
    async def after_message_sent(self, event: MessageEvent) -> None:
        pending_images = _deserialize_images(event.get_extra(PENDING_IMAGES_KEY, []))
        ctx = self._require_ctx()
        ctx.logger.info(
            "meme_manager after_message_sent: session={} pending_images={}",
            event.session_id,
            len(pending_images),
        )
        sent_images = 0
        try:
            for image in pending_images:
                if isinstance(image, Image):
                    await event.reply_chain([image])
                    sent_images += 1
            ctx.logger.info(
                "meme_manager after_message_sent: session={} sent_images={}",
                event.session_id,
                sent_images,
            )
        finally:
            for temp_file in event.get_extra(TEMP_FILES_KEY, []):
                temp_path = Path(str(temp_file))
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
            event.set_extra(PENDING_IMAGES_KEY, [])
            event.set_extra(TEMP_FILES_KEY, [])
            event.set_extra(STREAM_BUFFER_KEY, "")

    @http_api(
        "/astrbot_plugin_meme_manager",
        methods=["GET"],
        description="SDK meme manager overview page",
    )
    @provide_capability(
        name="astrbot_plugin_meme_manager.overview",
        description="Return the SDK meme manager overview page",
    )
    async def overview(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return {
            "status": 200,
            "headers": {"Content-Type": "text/html; charset=utf-8"},
            "body": self._build_overview_html(),
        }

    @http_api(
        "/astrbot_plugin_meme_manager/image",
        methods=["GET"],
        description="Serve a public meme preview image",
    )
    @provide_capability(
        name="astrbot_plugin_meme_manager.image",
        description="Return a preview image for the SDK meme manager page",
    )
    async def http_image(self, payload: dict[str, Any]) -> dict[str, Any]:
        category = self._http_mapping_first_value(payload, "query", "category")
        filename = self._http_mapping_first_value(payload, "query", "filename")
        image_path = self._resolve_image_path(category, filename)
        if image_path is None:
            return self._error_response("图片不存在", status=404)
        file_url = await self._file_url_for_path(image_path)
        return {
            "status": 302,
            "headers": {
                "Location": file_url,
                "Cache-Control": "no-store",
            },
            "body": "",
        }

    @http_api(
        "/astrbot_plugin_meme_manager/api/image",
        methods=["GET"],
        description="Return a public file URL for a meme preview image",
    )
    @provide_capability(
        name="astrbot_plugin_meme_manager.image_data",
        description="Return a public file URL for the management page preview image",
    )
    async def http_image_data(self, payload: dict[str, Any]) -> dict[str, Any]:
        category = self._http_mapping_first_value(payload, "query", "category")
        filename = self._http_mapping_first_value(payload, "query", "filename")
        image_path = self._resolve_image_path(category, filename)
        if image_path is None:
            return self._error_response("图片不存在", status=404)
        file_url = await self._file_url_for_path(image_path)
        return self._json_response(
            {
                "category": category,
                "filename": filename,
                "url": file_url,
            }
        )

    @http_api(
        "/astrbot_plugin_meme_manager/api/library",
        methods=["GET"],
        description="Return category descriptions and local files",
    )
    @provide_capability(
        name="astrbot_plugin_meme_manager.library",
        description="Return meme library JSON data",
    )
    async def http_library(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return self._json_response(self._library_payload())

    @http_api(
        "/astrbot_plugin_meme_manager/api/emoji",
        methods=["GET"],
        description="Return grouped meme files for the management page",
    )
    @provide_capability(
        name="astrbot_plugin_meme_manager.emoji",
        description="Return grouped meme files with preview URLs",
    )
    async def http_emoji(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json_response(await self._emoji_page_payload(payload))

    @http_api(
        "/astrbot_plugin_meme_manager/api/emotions",
        methods=["GET"],
        description="Return category descriptions for the management page",
    )
    @provide_capability(
        name="astrbot_plugin_meme_manager.emotions",
        description="Return meme category descriptions",
    )
    async def http_emotions(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return self._json_response(self._category_mapping())

    @http_api(
        "/astrbot_plugin_meme_manager/api/emoji/add",
        methods=["POST"],
        description="Upload one or more images into a meme category",
    )
    @provide_capability(
        name="astrbot_plugin_meme_manager.emoji_add",
        description="Upload meme images into a category",
    )
    async def http_add_emoji(self, payload: dict[str, Any]) -> dict[str, Any]:
        category = sanitize_category_name(
            self._http_mapping_first_value(payload, "form", "category")
            or self._http_json_body(payload).get("category", "")
        )
        if not category:
            return self._error_response("没有指定类别")

        uploaded_files = [
            item
            for item in self._http_files(payload)
            if item.get("field_name") == "image_file"
        ]
        if not uploaded_files:
            return self._error_response("没有找到上传的图片文件")

        target_dir = self._category_dir(category)
        saved_files: list[str] = []
        temp_paths: list[Path] = []
        try:
            for file_item in uploaded_files:
                temp_path = Path(str(file_item.get("path", "")))
                if not temp_path.exists():
                    continue
                temp_paths.append(temp_path)
                preferred_name = Path(str(file_item.get("filename", ""))).stem
                saved_path = copy_image_to_category(
                    source_path=temp_path,
                    target_dir=target_dir,
                    preferred_name=preferred_name or None,
                )
                saved_files.append(saved_path.name)
        finally:
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)

        if not saved_files:
            return self._error_response("没有成功保存任何图片", status=500)

        self._require_category_manager().sync_with_filesystem()
        return self._json_response(
            {
                "message": "表情包添加成功",
                "category": category,
                "saved_files": saved_files,
            },
            status=201,
        )

    @http_api(
        "/astrbot_plugin_meme_manager/api/emoji/delete",
        methods=["POST"],
        description="Delete a single meme image from a category",
    )
    @provide_capability(
        name="astrbot_plugin_meme_manager.emoji_delete",
        description="Delete a meme image from a category",
    )
    async def http_delete_emoji(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = self._http_json_body(payload)
        category = sanitize_category_name(str(body.get("category", "")))
        filename = Path(str(body.get("image_file", ""))).name
        if not category or not filename:
            return self._error_response("Category and image file are required")

        image_path = self._resolve_image_path(category, filename)
        if image_path is None:
            return self._error_response("Emoji not found", status=404)

        image_path.unlink(missing_ok=True)
        self._require_category_manager().sync_with_filesystem()
        return self._json_response(
            {
                "message": "Emoji deleted successfully",
                "category": category,
                "filename": filename,
            }
        )

    @http_api(
        "/astrbot_plugin_meme_manager/api/stats",
        methods=["GET"],
        description="Return local and remote meme library stats",
    )
    @provide_capability(
        name="astrbot_plugin_meme_manager.stats",
        description="Return meme statistics JSON data",
    )
    async def http_stats(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        per_category, total = library_stats(self._require_paths().memes_dir)
        body: dict[str, Any] = {
            "local": {
                "per_category": per_category,
                "total": total,
            }
        }
        if self._img_sync is not None:
            try:
                body["remote"] = {
                    "total": len(self._img_sync.provider.get_image_list())
                }
            except Exception as exc:
                body["remote_error"] = str(exc)
        return self._json_response(body)

    @http_api(
        "/astrbot_plugin_meme_manager/api/sync/status",
        methods=["GET"],
        description="Return config/filesystem drift information",
    )
    @provide_capability(
        name="astrbot_plugin_meme_manager.sync_status",
        description="Return config/filesystem drift information",
    )
    async def http_sync_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return self._json_response(self._sync_config_payload())

    @http_api(
        "/astrbot_plugin_meme_manager/api/config-sync",
        methods=["GET"],
        description="Return category/config drift information",
    )
    @provide_capability(
        name="astrbot_plugin_meme_manager.config_sync",
        description="Return category/config drift JSON data",
    )
    async def http_config_sync(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        sync_payload = self._sync_config_payload()
        differences = sync_payload["differences"]
        return self._json_response(
            {
                "only_in_filesystem": differences["missing_in_config"],
                "only_in_config": differences["deleted_categories"],
            }
        )

    @http_api(
        "/astrbot_plugin_meme_manager/api/sync/config",
        methods=["POST"],
        description="Sync category config with the filesystem layout",
    )
    @provide_capability(
        name="astrbot_plugin_meme_manager.sync_config",
        description="Sync category config with the filesystem layout",
    )
    async def http_sync_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        self._require_category_manager().sync_with_filesystem()
        return self._json_response({"message": "配置同步成功"})

    @http_api(
        "/astrbot_plugin_meme_manager/api/category/delete",
        methods=["POST"],
        description="Delete a meme category and its files",
    )
    @provide_capability(
        name="astrbot_plugin_meme_manager.category_delete",
        description="Delete a meme category and its files",
    )
    async def http_delete_category(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = self._http_json_body(payload)
        category = sanitize_category_name(str(body.get("category", "")))
        if not category:
            return self._error_response("Category is required")
        self._require_category_manager().delete_category(category)
        return self._json_response({"message": "Category deleted successfully"})

    @http_api(
        "/astrbot_plugin_meme_manager/api/category/update_description",
        methods=["POST"],
        description="Update the description for a meme category",
    )
    @provide_capability(
        name="astrbot_plugin_meme_manager.category_update_description",
        description="Update the description for a meme category",
    )
    async def http_update_category_description(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        body = self._http_json_body(payload)
        category = sanitize_category_name(str(body.get("tag", "")))
        description = str(body.get("description", "")).strip()
        if not category or not description:
            return self._error_response("Category and description are required")
        self._require_category_manager().update_description(category, description)
        return self._json_response({"category": category, "description": description})

    @http_api(
        "/astrbot_plugin_meme_manager/api/category/restore",
        methods=["POST"],
        description="Create or restore a meme category",
    )
    @provide_capability(
        name="astrbot_plugin_meme_manager.category_restore",
        description="Create or restore a meme category",
    )
    async def http_restore_category(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = self._http_json_body(payload)
        category = sanitize_category_name(str(body.get("category", "")))
        description = str(body.get("description", "")).strip() or "请添加描述"
        if not category:
            return self._error_response("Category is required")
        self._category_dir(category).mkdir(parents=True, exist_ok=True)
        self._require_category_manager().update_description(category, description)
        return self._json_response(
            {
                "message": "Category created successfully",
                "description": description,
            }
        )

    @http_api(
        "/astrbot_plugin_meme_manager/api/category/rename",
        methods=["POST"],
        description="Rename a meme category",
    )
    @provide_capability(
        name="astrbot_plugin_meme_manager.category_rename",
        description="Rename a meme category",
    )
    async def http_rename_category(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = self._http_json_body(payload)
        old_name = sanitize_category_name(str(body.get("old_name", "")))
        new_name = sanitize_category_name(str(body.get("new_name", "")))
        if not old_name or not new_name:
            return self._error_response("Old and new category names are required")
        if not self._require_category_manager().rename_category(old_name, new_name):
            return self._error_response("Failed to rename category", status=409)
        return self._json_response({"message": "Category renamed successfully"})

    @http_api(
        "/astrbot_plugin_meme_manager/api/img_host/sync/status",
        methods=["GET"],
        description="Return cloud image host sync status",
    )
    @provide_capability(
        name="astrbot_plugin_meme_manager.img_host_sync_status",
        description="Return cloud image host sync status",
    )
    async def http_img_host_sync_status(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        del payload
        if self._img_sync is None:
            return self._error_response("图床服务未配置")
        return self._json_response(self._cloud_sync_status_payload())

    @http_api(
        "/astrbot_plugin_meme_manager/api/img_host/sync/upload",
        methods=["POST"],
        description="Start a background upload sync",
    )
    @provide_capability(
        name="astrbot_plugin_meme_manager.img_host_sync_upload",
        description="Start a background upload sync",
    )
    async def http_img_host_sync_upload(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        del payload
        return self._start_background_sync("upload")

    @http_api(
        "/astrbot_plugin_meme_manager/api/img_host/sync/download",
        methods=["POST"],
        description="Start a background download sync",
    )
    @provide_capability(
        name="astrbot_plugin_meme_manager.img_host_sync_download",
        description="Start a background download sync",
    )
    async def http_img_host_sync_download(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        del payload
        return self._start_background_sync("download")

    @http_api(
        "/astrbot_plugin_meme_manager/api/img_host/sync/check_process",
        methods=["GET"],
        description="Return the current background sync process status",
    )
    @provide_capability(
        name="astrbot_plugin_meme_manager.img_host_sync_check_process",
        description="Return the current background sync process status",
    )
    async def http_img_host_sync_check_process(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        del payload
        return self._json_response(self._sync_process_payload())
