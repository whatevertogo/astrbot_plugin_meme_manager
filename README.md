# astrbot_plugin_meme_manager

SDK migration of the legacy `meme_manager` plugin.

What is migrated:

- `/表情管理` command group
- image upload flow for category-based meme storage
- LLM prompt injection and response tag extraction
- result decoration plus deferred image sending
- optional remote image-host sync
- SDK HTTP overview and JSON endpoints

What is intentionally not migrated 1:1:

- the legacy standalone Quart/Hypercorn WebUI process
- multipart upload and static-file serving from the old backend

The SDK version keeps the data model and command workflow, but exposes:

- a public overview page at `/plug/{plugin_id}`
- JSON APIs at `/api/plug/{plugin_id}` and `/api/plug/{plugin_id}/api/*`

instead of booting its own server.

## Validation

```bash
astrbot-sdk validate --plugin-dir .
python -m pytest tests/test_plugin.py -v
```
