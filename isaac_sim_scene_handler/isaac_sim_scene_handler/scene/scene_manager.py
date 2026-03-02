from __future__ import annotations

from typing import List
from typing import Tuple
from typing import Dict
from typing import Optional

import importlib
import inspect
import pkgutil
from pathlib import Path

import time

from isaac_sim_scene_handler.scene.scene_orchestrator import SceneOrchestrator


class SceneManager:
    _scenes: List[SceneOrchestrator]

    def __init__(self):
        self._scenes = []

    def add_scene(self, package_name: str) -> Tuple[int, Tuple[float, float, float], Dict]:
        # assert self._verify_package(package_name, 'Scene')

        self._set_stop_flag(True)

        # ! Not a full solution
        scene_class = self.import_class_from_path(
            package_name,
            "scene.py",
            "Scene",
        )

        # scene_class = self._import_class(package_name, 'Scene')
        id = len(self._scenes)
        self._scenes.append(scene_class(scene_id=id, path=package_name))

        offset = self._calculate_offset(id)

        self._set_stop_flag(False)

        return (id, offset, self._scenes[id]._config)

    def import_class_from_path(self, package_path: str, module_file: str, class_name: str):
        package_path = Path(package_path)
        module_path = package_path / module_file

        if not module_path.exists():
            raise FileNotFoundError(f"Module file not found: {module_path}")

        spec = importlib.util.spec_from_file_location(
            module_path.stem,
            module_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec for {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        cls = getattr(module, class_name, None)
        if not inspect.isclass(cls):
            raise ImportError(f"{class_name} not found in {module_path}")

        return cls

    def _verify_package(self, package_name: str, class_name: str) -> bool:
        assert package_name is not None
        assert class_name is not None

        try:
            package = importlib.import_module(package_name)
        except ImportError:
            print(f"{package_name} module is not found!")
            return False

        # top-level ellenőrzés
        if inspect.isclass(getattr(package, class_name, None)):
            return True

        # almodulok bejárása
        if hasattr(package, "__path__"):
            for _, modname, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                try:
                    module = importlib.import_module(modname)
                except Exception:
                    continue

                if inspect.isclass(getattr(module, class_name, None)):
                    return True

        print(f"{class_name} class is not found!")
        return False

    def _import_class(self, module_name: str, class_name: str):
        module = importlib.import_module(module_name)
        try:
            cls = getattr(module, class_name)
        except AttributeError:
            raise ImportError(f"Module '{module_name}' does not define '{class_name}'")
        return cls

    def _calculate_offset(self, scene_id: int) -> Tuple[float, float, float]:
        print("Calculating the offsets")
        offset = [0.0, 0.0, 0.0]
        for idx in range(scene_id):
            offset[1] = (
                offset[1]
                + self._scenes[idx].bounding_box["yp"]
                + self._scenes[idx].bounding_box["yn"]
            )

        offset[1] = (
            offset[1]
            - self._scenes[0].bounding_box["yn"]
            + self._scenes[scene_id].bounding_box["yp"]
        )

        offset[0] = offset[0] - self._scenes[scene_id].origin[0]
        offset[1] = offset[1] - self._scenes[scene_id].origin[1]
        offset[2] = offset[2] - self._scenes[scene_id].origin[2]

        return tuple(offset)

    def reset_scene(self, scene_id: int, seed: Optional[int]) -> bool:
        return self._scenes[scene_id].reset(seed)

    def randomize_scene(self, scene_id: int, seed: Optional[int]) -> bool:
        return self._scenes[scene_id].randomize(seed)

    def is_success(self, scene_id: int) -> bool:
        return self._scenes[scene_id].is_success()

    def _set_stop_flag(self, value: bool):
        if not value:
            for scene in self._scenes:
                scene._stop_flag.clear()
            return

        for scene in self._scenes:
            scene._stop_flag.set()

        while any(scene._running.is_set() for scene in self._scenes):
            time.sleep(0.1)
