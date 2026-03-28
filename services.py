from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

DEFAULT_CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "angry": "当对话包含抱怨、批评或激烈反对时使用（如用户投诉/观点反驳）",
    "happy": "用于成功确认、积极反馈或庆祝场景（问题解决/获得成就）",
    "sad": "表达伤心, 歉意、遗憾或安慰场景（遇到挫折/传达坏消息）",
    "surprised": "响应超出预期的信息（重大发现/意外转折）注意：轻微惊讶慎用",
    "confused": "请求澄清或表达理解障碍时（概念模糊/逻辑矛盾）或对于用户的请求感到困惑",
    "color": "社交场景中的暧昧表达（调情）使用频率≤1次/对话",
    "cpu": "技术讨论中表示思维卡顿（复杂问题/需要加载时间）",
    "fool": "自嘲或缓和气氛的幽默场景（小失误/无伤大雅的玩笑）",
    "givemoney": "涉及报酬讨论时使用（服务付费/奖励机制）需配合明确金额",
    "like": "表达对事物或观点的喜爱（美食/艺术/优秀方案）",
    "see": "表示偷瞄或持续关注（监控进度/观察变化）常与时间词搭配",
    "shy": "涉及隐私话题或收到赞美时（个人故事/外貌评价）",
    "work": "工作流程相关场景（任务分配/进度汇报）",
    "reply": "等待用户反馈时（提问后/需要确认）最长间隔30分钟",
    "meow": "卖萌或萌系互动场景（宠物话题/安抚情绪）慎用于正式场合",
    "baka": "轻微责备或吐槽（低级错误/可爱型抱怨）禁用程度：友善级",
    "morning": "早安问候专用（UTC时间6:00-10:00）跨时区需换算",
    "sleep": "涉及作息场景（熬夜/疲劳/休息建议）",
    "sigh": "表达无奈, 无语或感慨（重复问题/历史遗留难题）",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


@dataclass(slots=True)
class PluginPaths:
    plugin_dir: Path
    data_dir: Path
    memes_dir: Path
    memes_data_path: Path
    temp_dir: Path
    bundled_memes_dir: Path


def build_paths(plugin_dir: Path, data_dir: Path) -> PluginPaths:
    return PluginPaths(
        plugin_dir=plugin_dir,
        data_dir=data_dir,
        memes_dir=data_dir / "memes",
        memes_data_path=data_dir / "memes_data.json",
        temp_dir=data_dir / "temp",
        bundled_memes_dir=plugin_dir / "memes",
    )


def ensure_runtime_layout(paths: PluginPaths) -> None:
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.memes_dir.mkdir(parents=True, exist_ok=True)
    paths.temp_dir.mkdir(parents=True, exist_ok=True)
    if _memes_dir_is_effectively_empty(paths.memes_dir):
        copy_default_memes(paths)
    if not paths.memes_data_path.exists():
        save_json(paths.memes_data_path, DEFAULT_CATEGORY_DESCRIPTIONS)


def _memes_dir_is_effectively_empty(memes_dir: Path) -> bool:
    if not memes_dir.exists():
        return True
    for child in memes_dir.iterdir():
        if child.name.startswith("."):
            continue
        return False
    return True


def copy_default_memes(paths: PluginPaths) -> None:
    source_dir = paths.bundled_memes_dir
    if not source_dir.exists():
        return
    for source_child in source_dir.iterdir():
        target_child = paths.memes_dir / source_child.name
        if source_child.is_dir():
            if target_child.exists():
                continue
            shutil.copytree(source_child, target_child)
        elif not target_child.exists():
            shutil.copy2(source_child, target_child)


def load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(default or {})


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def dict_to_prompt_lines(mapping: dict[str, str]) -> str:
    return "\n".join(f"{key} - {value}" for key, value in mapping.items())


def sanitize_category_name(category: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]", "_", category.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("._")
    return normalized


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def detect_image_suffix(file_path: Path) -> str:
    try:
        with PILImage.open(file_path) as image:
            image_format = (image.format or "").lower()
    except Exception:
        return file_path.suffix.lower() or ".bin"
    return {
        "jpeg": ".jpg",
        "png": ".png",
        "gif": ".gif",
        "webp": ".webp",
        "bmp": ".bmp",
    }.get(image_format, file_path.suffix.lower() or ".bin")


def copy_image_to_category(
    *,
    source_path: Path,
    target_dir: Path,
    preferred_name: str | None = None,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = detect_image_suffix(source_path)
    file_name = preferred_name.strip() if preferred_name else source_path.stem
    file_name = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]", "_", file_name).strip("._")
    if not file_name:
        file_name = next(tempfile._get_candidate_names())  # noqa: SLF001
    target_path = ensure_unique_path(target_dir / f"{file_name}{suffix}")
    shutil.copy2(source_path, target_path)
    return target_path


def list_category_files(memes_dir: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if not memes_dir.exists():
        return result
    for category_dir in sorted(item for item in memes_dir.iterdir() if item.is_dir()):
        result[category_dir.name] = sorted(
            child.name
            for child in category_dir.iterdir()
            if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
        )
    return result


def library_stats(memes_dir: Path) -> tuple[dict[str, int], int]:
    per_category: dict[str, int] = {}
    total = 0
    for category, files in list_category_files(memes_dir).items():
        per_category[category] = len(files)
        total += len(files)
    return per_category, total
