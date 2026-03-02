from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

from isaac_sim_scene_handler.types import conversion

# * Geometry classes wrapping common representations used in Isaac Sim and ROS


def _vec3(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float).reshape(3)
    if not np.all(np.isfinite(a)):
        raise ValueError("Vector has non-finite values.")
    return a


# -----------------------------
# Point / Vector3
# -----------------------------


@dataclass(slots=True)
class Point:
    coordinates: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinates", _vec3(self.coordinates))

    @staticmethod
    def zero() -> Point:
        return Point()

    @staticmethod
    def from_numpy(arr: np.ndarray) -> Point:
        return Point(coordinates=_vec3(arr))

    @staticmethod
    def from_ros(p: conversion.RosPoint) -> Point:
        return Point.from_numpy(conversion.gf_to_vec3(conversion.ros_point_to_gf(p)))

    # @staticmethod
    # def from_gf(v: conversion.Gf.Vec3d) -> Point:
    #     return Point.from_numpy(conversion.gf_to_vec3(v))

    # @staticmethod
    # def from_carb(v: conversion.carb.Float3) -> Point:
    #     return Point.from_gf(conversion.carb_to_gf_vec3(v))

    def to_numpy(self) -> np.ndarray:
        return self.coordinates.copy()

    def to_ros(self) -> conversion.RosPoint:
        return conversion.vec3_to_ros_point(self.coordinates)

    # def to_gf(self) -> conversion.Gf.Vec3d:
    #     return conversion.vec3_to_gf(self.coordinates)

    # def to_carb(self) -> conversion.carb.Float3:
    #     return conversion.gf_to_carb_vec3(self.to_gf())

    def __add__(self, other: Vector3 | np.ndarray) -> Point:
        if isinstance(other, Vector3):
            return Point.from_numpy(self.coordinates + other.v)
        return Point.from_numpy(self.coordinates + _vec3(other))

    def __sub__(self, other: Point | Vector3 | np.ndarray) -> Vector3:
        if isinstance(other, Point):
            return Vector3.from_numpy(self.coordinates - other.coordinates)
        if isinstance(other, Vector3):
            return Vector3.from_numpy(self.coordinates - other.v)
        return Vector3.from_numpy(self.coordinates - _vec3(other))


@dataclass(slots=True)
class Vector3:
    v: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    def __post_init__(self) -> None:
        object.__setattr__(self, "v", _vec3(self.v))

    @staticmethod
    def zero() -> Vector3:
        return Vector3()

    @staticmethod
    def from_numpy(arr: np.ndarray) -> Vector3:
        return Vector3(v=_vec3(arr))

    @staticmethod
    def from_ros(v: conversion.RosVector3) -> Vector3:
        return Vector3.from_numpy(conversion.gf_to_vec3(conversion.ros_vec_to_gf(v)))

    # @staticmethod
    # def from_gf(v: conversion.Gf.Vec3d) -> Vector3:
    #     return Vector3.from_numpy(conversion.gf_to_vec3(v))

    # @staticmethod
    # def from_carb(v: conversion.carb.Float3) -> Vector3:
    #     return Vector3.from_gf(conversion.carb_to_gf_vec3(v))

    def to_numpy(self) -> np.ndarray:
        return self.v.copy()

    def to_ros(self) -> conversion.RosVector3:
        return conversion.vec3_to_ros_vec(self.v)

    # def to_gf(self) -> conversion.Gf.Vec3d:
    #     return conversion.vec3_to_gf(self.v)

    # def to_carb(self) -> conversion.carb.Float3:
    #     return conversion.gf_to_carb_vec3(self.to_gf())

    def __add__(self, other: Vector3 | np.ndarray) -> Vector3:
        if isinstance(other, Vector3):
            return Vector3.from_numpy(self.v + other.v)
        return Vector3.from_numpy(self.v + _vec3(other))

    def __sub__(self, other: Vector3 | np.ndarray) -> Vector3:
        if isinstance(other, Vector3):
            return Vector3.from_numpy(self.v - other.v)
        return Vector3.from_numpy(self.v - _vec3(other))

    def __neg__(self) -> Vector3:
        return Vector3.from_numpy(-self.v)


# -----------------------------
# Rotation
# -----------------------------
@dataclass(slots=True)
class Rotation:
    rot: R = field(default_factory=R.identity)
    scale_factor: float = 1.0

    @staticmethod
    def identity() -> Rotation:
        return Rotation()

    @staticmethod
    def from_scipy(rot: R, scale_factor: float = 1.0) -> Rotation:
        return Rotation(rot=rot, scale_factor=float(scale_factor))

    # @staticmethod
    # def from_gf_quat(q: conversion.Gf.Quatd) -> Rotation:
    #     return Rotation.from_scipy(conversion.gf_quat_to_scipy_rot(q))

    @staticmethod
    def from_ros_quat(q: conversion.RosQuaternion) -> Rotation:
        return Rotation.from_scipy(conversion.ros_quat_to_scipy_rot(q))

    # @staticmethod
    # def from_gf_matrix(m: conversion.Gf.Matrix3d) -> Rotation:
    #     return Rotation(
    #         rot=conversion.gf_matrix_to_scipy_rot(m),
    #         scale_factor=conversion.scale_factor_from_gf_matrix(m),
    #     )

    # @staticmethod
    # def from_carb(quat: conversion.carb.Float4) -> Rotation:
    #     return Rotation.from_gf_quat(conversion.carb_to_gf_quat(quat))

    def to_scipy(self) -> R:
        return self.rot

    def to_numpy(self) -> np.ndarray:
        return self.rot.as_matrix()

    def to_numpy_quat(self) -> np.ndarray:
        return self.rot.as_quat()

    # def to_gf_quat(self) -> conversion.Gf.Quatd:
    #     return conversion.scipy_rot_to_gf_quat(self.rot)

    def to_ros_quat(self) -> conversion.RosQuaternion:
        return conversion.scipy_rot_to_ros_quat(self.rot)

    # def to_gf_matrix(self) -> conversion.Gf.Matrix3d:
    #     return conversion.scipy_rot_to_gf_matrix(self.rot)

    # def to_carb(self) -> conversion.carb.Float4:
    #     return conversion.gf_quat_to_carb_quat(self.to_gf_quat())

    def inv(self) -> Rotation:
        return Rotation(rot=self.rot.inv(), scale_factor=self.scale_factor)

    def __mul__(self, other: Rotation) -> Rotation:
        return Rotation(
            rot=self.rot * other.rot,
            scale_factor=self.scale_factor * other.scale_factor,
        )

    def __invert__(self) -> Rotation:
        return self.inv()

    def apply(self, x: Point | Vector3 | np.ndarray) -> np.ndarray:
        if isinstance(x, Point):
            v = x.coordinates
        elif isinstance(x, Vector3):
            v = x.v
        else:
            v = _vec3(x)
        return self.rot.apply(v)


# -----------------------------
# Pose
# -----------------------------
@dataclass(slots=True)
class Pose:
    position: Point = field(default_factory=Point.zero)
    orientation: Rotation = field(default_factory=Rotation.identity)

    @staticmethod
    def identity() -> Pose:
        return Pose()

    @staticmethod
    def from_numpy(position: np.ndarray, orientation: R) -> Pose:
        return Pose(
            position=Point.from_numpy(position),
            orientation=Rotation.from_scipy(orientation),
        )

    @staticmethod
    def from_ros_pose(p: conversion.RosPose) -> Pose:
        pos = conversion.gf_to_vec3(conversion.ros_point_to_gf(p.position))
        rot = conversion.ros_quat_to_scipy_rot(p.orientation)
        return Pose.from_numpy(pos, rot)

    @staticmethod
    def from_ros_transform(t: conversion.RosTransform) -> Pose:
        pos = conversion.gf_to_vec3(conversion.ros_vec_to_gf(t.translation))
        rot = conversion.ros_quat_to_scipy_rot(t.rotation)
        return Pose.from_numpy(pos, rot)

    # @staticmethod
    # def from_gf_vec_quat(pos: conversion.Gf.Vec3d, quat: conversion.Gf.Quatd) -> Pose:
    #     return Pose.from_numpy(conversion.gf_to_vec3(pos), conversion.gf_quat_to_scipy_rot(quat))

    # @staticmethod
    # def from_gf_tuple(pose: tuple[conversion.Gf.Vec3d, conversion.Gf.Quatd]) -> Pose:
    #     return Pose.from_gf_vec_quat(pose[0], pose[1])

    def to_ros(self) -> conversion.RosPose:
        return conversion.RosPose(
            position=self.position.to_ros(),
            orientation=self.orientation.to_ros_quat(),
        )

    def to_numpy(self) -> np.ndarray:
        T = np.eye(4, dtype=float)
        T[:3, :3] = self.orientation.to_numpy()
        T[:3, 3] = self.position.to_numpy()
        return T

    # def to_gf(self) -> tuple[conversion.Gf.Vec3d, conversion.Gf.Quatd]:
    #     return self.position.to_gf(), self.orientation.to_gf_quat()

    # def to_carb(self) -> tuple[conversion.carb.Float3, conversion.carb.Float4]:
    #     return self.position.to_carb(), self.orientation.to_carb()

    def to_Transform(self) -> conversion.RosTransform:
        return conversion.RosTransform(
            translation=self.position.to_ros(),
            rotation=self.orientation.to_ros_quat(),
        )

    def inv(self) -> Pose:
        inv_rot = self.orientation.inv()
        inv_pos = -inv_rot.apply(self.position)
        return Pose(position=Point.from_numpy(inv_pos), orientation=inv_rot)

    def __invert__(self) -> Pose:
        return self.inv()

    def __mul__(self, other: Pose) -> Pose:
        # t = t1 + R1 * t2
        pos = self.position.to_numpy() + self.orientation.apply(other.position)
        rot = self.orientation * other.orientation
        return Pose(position=Point.from_numpy(pos), orientation=rot)


# -----------------------------
# Transform
# -----------------------------
@dataclass(slots=True)
class Transform:
    translation: Vector3 = field(default_factory=Vector3.zero)
    rotation: Rotation = field(default_factory=Rotation.identity)

    @staticmethod
    def identity() -> Transform:
        return Transform()

    @staticmethod
    def from_ros(t: conversion.RosTransform) -> Transform:
        tr = conversion.gf_to_vec3(conversion.ros_vec_to_gf(t.translation))
        rot = conversion.ros_quat_to_scipy_rot(t.rotation)
        return Transform(translation=Vector3.from_numpy(tr), rotation=Rotation.from_scipy(rot))

    @staticmethod
    def from_numpy(position: np.ndarray, orientation: R) -> Transform:
        return Transform(
            position=Vector3.from_numpy(position),
            orientation=Rotation.from_scipy(orientation),
        )

    # @staticmethod
    # def from_gf(translation: conversion.Gf.Vec3d, rotation: conversion.Gf.Quatd) -> Transform:
    #     return Transform(
    #         translation=Vector3.from_gf(translation),
    #         rotation=Rotation.from_gf_quat(rotation),
    #     )

    def to_ros(self) -> conversion.RosTransform:
        return conversion.RosTransform(
            translation=self.translation.to_ros(),
            rotation=self.rotation.to_ros_quat(),
        )

    def to_numpy(self) -> np.ndarray:
        T = np.eye(4, dtype=float)
        T[:3, :3] = self.rotation.to_numpy()
        T[:3, 3] = self.translation.to_numpy()
        return T

    # def to_gf(self) -> tuple[conversion.Gf.Vec3d, conversion.Gf.Quatd]:
    #     return self.translation.to_gf(), self.rotation.to_gf_quat()

    # def to_carb(self) -> tuple[conversion.carb.Float3, conversion.carb.Float4]:
    #     return self.translation.to_carb(), self.rotation.to_carb()

    def to_pose(self) -> Pose:
        return Pose(
            position=Point.from_numpy(self.translation.to_numpy()),
            orientation=self.rotation,
        )

    def inv(self) -> Transform:
        inv_rot = self.rotation.inv()
        inv_trans = -inv_rot.apply(self.translation)
        return Transform(translation=Vector3.from_numpy(inv_trans), rotation=inv_rot)

    def __invert__(self) -> Transform:
        return self.inv()

    def __mul__(self, other: Transform | Pose) -> Transform:
        # t = t1 + R1 * t2
        if isinstance(other, Transform):
            pos = self.translation.to_numpy() + self.rotation.apply(other.translation)
            rot = self.rotation * other.rotation
        else:
            pos = self.translation.to_numpy() + self.rotation.apply(other.position)
            rot = self.rotation * other.orientation
        return Transform(translation=Vector3.from_numpy(pos), rotation=rot)
