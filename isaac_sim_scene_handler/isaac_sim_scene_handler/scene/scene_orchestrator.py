from __future__ import annotations

from typing import Tuple
from typing import Optional
from typing import List
from typing import Dict

import yaml

from abc import ABC
from abc import abstractmethod

from threading import Event

from importlib import resources
from pathlib import Path


class SceneOrchestrator(ABC):

    scene_id: int

    _offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    _pkg_name: str

    _config: dict
    _usd_path: str
    bounding_box: dict
    origin: list

    _running: Event = Event()
    _stop_flag: Event = Event()

    def __init__(self, scene_id: int, path: Optional[str], config_path: str = "/config/init.yaml"):

        self._scene_id = scene_id

        # Getting init.yaml
        package_name = self.__class__.__module__.split(".")[0]

        if path is None:
            self._path = resources.files(package_name)
            config_file = resources.files(package_name).joinpath(config_path)
        else:
            self._path = Path(path)
            config_file = Path(f"{path}/{config_path}")

        # with resources.files(package_name).joinpath(config_path).open('r') as f:
        with config_file.open("r") as f:
            self._config = yaml.safe_load(f)

        self._get_usd_params(package_name)

        self._get_limits()
        self._get_origin()

    def _get_usd_params(self, package_name):
        self._usd_path = self._path.joinpath(self._config["usd_path"])

    def _get_limits(self):
        limits: Optional[dict] = self._config.get("limits", None)

        assert limits is not None

        # Creating bounding box
        self.bounding_box = {
            "xp": limits.get("xp", 0.0),
            "xn": limits.get("xn", 0.0),
            "yp": limits.get("yp", 0.0),
            "yn": limits.get("yn", 0.0),
            "zp": limits.get("zp", 0.0),
            "zn": limits.get("zn", 0.0),
        }

    def _get_origin(self):
        self.origin = self._config.get("origin", [0.0, 0.0, 0.0])

        assert self.origin is not None

    def create_robot_graphs(self):
        robot_list: List[Dict] = []
        for name, path in self._config["robots"].items():
            robot_list.append(
                {"namespace": f"/{name}", "articulation_root": f"{path}", "path": f"/{name}"}
            )
        return robot_list

    def create_camera_graphs(self):
        camera_list: List[Dict] = []
        for name, data in self._config["cameras"].items():
            path = data["path"]
            assert path is not None
            width = data["width"]
            assert width is not None
            height = data["height"]
            assert height is not None
            topic = data["topic"]
            assert topic is not None

            camera_list.append(
                {
                    "camera_path": f"{path}",
                    "path": f"/{name}",
                    "width": width,
                    "height": height,
                    "frame": f"{name}",
                    "topic": f"{topic}",
                }
            )
        return camera_list

    @abstractmethod
    def reset(self) -> bool:
        raise NotImplementedError("Reset function is not implemented for this scene.")

    @abstractmethod
    def randomize(self, seed: Optional[int]) -> bool:
        raise NotImplementedError("Randomize function is not implemented.")

    @abstractmethod
    def is_success(self) -> bool:
        raise NotImplementedError("Success checking is not implemented.")
