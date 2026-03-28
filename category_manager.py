from __future__ import annotations

import shutil

from services import DEFAULT_CATEGORY_DESCRIPTIONS, PluginPaths, load_json, save_json


class CategoryManager:
    def __init__(self, paths: PluginPaths) -> None:
        self.paths = paths
        if not self.paths.memes_data_path.exists():
            save_json(self.paths.memes_data_path, DEFAULT_CATEGORY_DESCRIPTIONS)
        self.descriptions = self._load_descriptions()

    def _load_descriptions(self) -> dict[str, str]:
        return load_json(self.paths.memes_data_path, DEFAULT_CATEGORY_DESCRIPTIONS)

    def reload(self) -> None:
        self.descriptions = self._load_descriptions()

    def get_descriptions(self) -> dict[str, str]:
        return dict(self.descriptions)

    def get_local_categories(self) -> set[str]:
        if not self.paths.memes_dir.exists():
            return set()
        return {
            item.name
            for item in self.paths.memes_dir.iterdir()
            if item.is_dir() and not item.name.startswith(".")
        }

    def get_sync_status(self) -> tuple[list[str], list[str]]:
        local_categories = self.get_local_categories()
        config_categories = set(self.descriptions)
        return (
            sorted(local_categories - config_categories),
            sorted(config_categories - local_categories),
        )

    def sync_with_filesystem(self) -> bool:
        changed = False
        local_categories = self.get_local_categories()
        for category in sorted(local_categories):
            if category not in self.descriptions:
                self.descriptions[category] = "请添加描述"
                changed = True
        if changed:
            save_json(self.paths.memes_data_path, self.descriptions)
        return True

    def update_description(self, category: str, description: str) -> bool:
        self.descriptions[category] = description
        save_json(self.paths.memes_data_path, self.descriptions)
        return True

    def rename_category(self, old_name: str, new_name: str) -> bool:
        if old_name not in self.descriptions:
            return False
        old_path = self.paths.memes_dir / old_name
        new_path = self.paths.memes_dir / new_name
        if new_path.exists():
            return False
        if old_path.exists():
            old_path.rename(new_path)
        description = self.descriptions.pop(old_name)
        self.descriptions[new_name] = description
        save_json(self.paths.memes_data_path, self.descriptions)
        return True

    def delete_category(self, category: str) -> bool:
        self.descriptions.pop(category, None)
        save_json(self.paths.memes_data_path, self.descriptions)
        category_path = self.paths.memes_dir / category
        if category_path.exists():
            shutil.rmtree(category_path)
        return True
