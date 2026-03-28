from __future__ import annotations

import time
from pathlib import Path

import pytest
from PIL import Image as PILImage

from astrbot_sdk import EventResultType, Image, MessageChain, MessageEventResult, Plain
from astrbot_sdk.clients.llm import LLMResponse
from astrbot_sdk.llm.entities import ProviderRequest
from astrbot_sdk.message import payload_to_component
from astrbot_sdk.testing import MockContext, MockMessageEvent, PluginHarness
from main import (
    FOUND_EMOTIONS_KEY,
    PENDING_IMAGES_KEY,
    STREAM_BUFFER_KEY,
    AstrbotPluginMemeManager,
    UploadState,
    _deserialize_images,
)


def create_test_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (8, 8), (255, 0, 0)).save(path, format="PNG")
    return path


@pytest.mark.asyncio
async def test_on_start_initializes_runtime_data() -> None:
    plugin = AstrbotPluginMemeManager()
    ctx = MockContext(
        plugin_id="astrbot_plugin_meme_manager",
        plugin_metadata={"display_name": "Meme Manager"},
    )

    await plugin.on_start(ctx)

    data_dir = await ctx.get_data_dir()
    assert (data_dir / "memes").exists()
    assert (data_dir / "memes_data.json").exists()
    assert any((data_dir / "memes").iterdir())


@pytest.mark.asyncio
async def test_upload_flow_saves_image_to_category(tmp_path: Path) -> None:
    plugin = AstrbotPluginMemeManager()
    ctx = MockContext(
        plugin_id="astrbot_plugin_meme_manager",
        plugin_metadata={"display_name": "Meme Manager"},
    )
    await plugin.on_start(ctx)

    category = next(iter(plugin._category_mapping()))
    source_image = create_test_image(tmp_path / "upload.png")
    event = MockMessageEvent(
        text="image upload",
        context=ctx,
        raw={"messages": [Image.fromFileSystem(str(source_image)).toDict()]},
    )
    plugin._upload_states[plugin._upload_key(event)] = UploadState(
        category=category,
        expire_at=time.time() + 30,
    )

    await plugin.handle_upload_image(event)

    data_dir = await ctx.get_data_dir()
    saved_files = list((data_dir / "memes" / category).iterdir())
    assert any(item.name.endswith(".png") for item in saved_files)
    assert any("已将" in reply for reply in event.replies)


@pytest.mark.asyncio
async def test_llm_response_and_result_decoration_attach_pending_images() -> None:
    plugin = AstrbotPluginMemeManager()
    ctx = MockContext(
        plugin_id="astrbot_plugin_meme_manager",
        plugin_metadata={"display_name": "Meme Manager"},
    )
    await plugin.on_start(ctx)
    plugin._config.emotions_probability = 100
    plugin._config.enable_mixed_message = False

    event = MockMessageEvent(text="你好", context=ctx)
    response = LLMResponse(text="你好 &&happy&&")

    await plugin.on_llm_response(event, response, ctx)

    result = MessageEventResult(
        type=EventResultType.CHAIN,
        chain=MessageChain([Plain(response.text, convert=False)]),
    )
    await plugin.on_decorating_result(event, result)

    assert event.get_extra(FOUND_EMOTIONS_KEY) == ["happy"]
    assert result.chain.get_plain_text() == "你好"
    pending_images = event.get_extra(PENDING_IMAGES_KEY)
    assert pending_images
    assert all(isinstance(item, dict) for item in pending_images)
    assert all(isinstance(payload_to_component(item), Image) for item in pending_images)

    await plugin.after_message_sent(event)

    assert event.get_extra(PENDING_IMAGES_KEY) == []
    assert any(record.kind == "chain" for record in ctx.sent_messages)


@pytest.mark.asyncio
async def test_streaming_delta_strips_emotion_markup_across_chunks() -> None:
    plugin = AstrbotPluginMemeManager()
    ctx = MockContext(
        plugin_id="astrbot_plugin_meme_manager",
        plugin_metadata={"display_name": "Meme Manager"},
    )
    await plugin.on_start(ctx)

    event = MockMessageEvent(
        text="早上好",
        context=ctx,
        raw={"result_content_type": "streaming_delta"},
    )
    first = MessageEventResult(
        type=EventResultType.CHAIN,
        chain=MessageChain([Plain("早上好 &&se", convert=False)]),
    )
    second = MessageEventResult(
        type=EventResultType.CHAIN,
        chain=MessageChain([Plain("e&& &&happy&&", convert=False)]),
    )

    await plugin.on_streaming_delta(event, first)
    await plugin.on_streaming_delta(event, second)

    assert first.chain.get_plain_text() == "早上好 "
    assert first.type == EventResultType.CHAIN
    assert second.type == EventResultType.EMPTY
    assert event.get_extra(STREAM_BUFFER_KEY) == ""


@pytest.mark.asyncio
async def test_streaming_finish_defers_images_instead_of_merging() -> None:
    plugin = AstrbotPluginMemeManager()
    ctx = MockContext(
        plugin_id="astrbot_plugin_meme_manager",
        plugin_metadata={"display_name": "Meme Manager"},
    )
    await plugin.on_start(ctx)
    plugin._config.emotions_probability = 100
    plugin._config.enable_mixed_message = True
    plugin._config.mixed_message_probability = 100

    event = MockMessageEvent(
        text="你好",
        context=ctx,
        raw={"result_content_type": "streaming_finish"},
    )
    event.set_extra(FOUND_EMOTIONS_KEY, ["happy"])
    result = MessageEventResult(
        type=EventResultType.CHAIN,
        chain=MessageChain([Plain("你好", convert=False)]),
    )

    await plugin.on_decorating_result(event, result)

    pending_images = event.get_extra(PENDING_IMAGES_KEY)
    assert pending_images
    assert all(isinstance(item, dict) for item in pending_images)
    assert all(isinstance(item, Image) for item in _deserialize_images(pending_images))
    assert result.chain.get_plain_text() == "你好"


@pytest.mark.asyncio
async def test_http_library_returns_descriptions_and_files() -> None:
    plugin = AstrbotPluginMemeManager()
    ctx = MockContext(
        plugin_id="astrbot_plugin_meme_manager",
        plugin_metadata={"display_name": "Meme Manager"},
    )
    await plugin.on_start(ctx)

    response = await plugin.http_library({})

    assert response["status"] == 200
    assert "descriptions" in response["body"]
    assert "files" in response["body"]


@pytest.mark.asyncio
async def test_command_dispatch_lists_library() -> None:
    plugin_dir = Path(__file__).resolve().parents[1]

    async with PluginHarness.from_plugin_dir(plugin_dir) as harness:
        records = await harness.dispatch_text("表情管理 查看图库")

    assert any("当前图库分类" in (record.text or "") for record in records)


@pytest.mark.asyncio
@pytest.mark.parametrize("command_text", ["表情管理", "/表情管理", "/ 表情管理"])
async def test_group_root_command_returns_help(command_text: str) -> None:
    plugin_dir = Path(__file__).resolve().parents[1]

    async with PluginHarness.from_plugin_dir(plugin_dir) as harness:
        records = await harness.dispatch_text(command_text)

    assert any("表情管理命令" in (record.text or "") for record in records)


@pytest.mark.asyncio
async def test_on_llm_request_skips_prompt_injection_for_commands() -> None:
    plugin = AstrbotPluginMemeManager()
    ctx = MockContext(
        plugin_id="astrbot_plugin_meme_manager",
        plugin_metadata={"display_name": "Meme Manager"},
    )
    await plugin.on_start(ctx)

    event = MockMessageEvent(text="/表情管理", context=ctx)
    request = ProviderRequest(prompt="/表情管理", system_prompt="base")

    await plugin.on_llm_request(event, request)

    assert request.system_prompt == "base"
