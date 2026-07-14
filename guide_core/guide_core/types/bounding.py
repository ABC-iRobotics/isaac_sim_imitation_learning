from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import fcl
import numpy as np

# * Axis-Aligned (AABB) and Oriented (OBB) bounding-volume types plus a manager
# * exposing the logical operations (overlap / containment / union / intersection)
# * for one, two, and many objects.
# *
# * Overlap and distance queries are delegated to FCL (the Flexible Collision
# * Library, via ``python-fcl`` -- the same engine MoveIt uses) so we do not
# * hand-maintain separating-axis math. Each box is realised as an ``fcl.Box``
# * with a rigid transform. FCL reports a *signed* distance (negative ==
# * penetration depth), which gives a clean overlap rule::
# *
# *     a and b overlap  <=>  signed_distance(a, b) <= tolerance
# *
# * Conventions
# *   AABB : ``lower``/``upper`` corners (3,), lower <= upper.
# *   OBB  : ``center`` (3,), ``axes`` (3, 3) with each *row* a unit local axis
# *          (a proper rotation), and ``half_extents`` (3,) >= 0.

_EPS = 1e-9


def _vec3(arr: np.ndarray, name: str = "vector") -> np.ndarray:
    a = np.asarray(arr, dtype=float).reshape(3)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} has non-finite values.")
    return a


# -----------------------------
# AABB
# -----------------------------


@dataclass(slots=True)
class AABB:
    """Axis-aligned bounding box defined by its lower and upper corners."""

    lower: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    upper: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    def __post_init__(self) -> None:
        self.lower = _vec3(self.lower, "lower")
        self.upper = _vec3(self.upper, "upper")
        if np.any(self.lower > self.upper):
            raise ValueError("AABB lower corner must be <= upper corner on every axis.")

    # ---- constructors ---------------------------------------------------
    @classmethod
    def from_min_max(cls, lower: Sequence[float], upper: Sequence[float]) -> AABB:
        return cls(lower=np.asarray(lower, dtype=float), upper=np.asarray(upper, dtype=float))

    @classmethod
    def from_center_extents(cls, center: Sequence[float], half_extents: Sequence[float]) -> AABB:
        c = _vec3(center, "center")
        he = np.abs(_vec3(half_extents, "half_extents"))
        return cls(lower=c - he, upper=c + he)

    @classmethod
    def from_isaac(cls, aabb: Sequence[float]) -> AABB:
        """Build from Isaac Sim ``compute_aabb`` output ``[minx,miny,minz,maxx,maxy,maxz]``."""
        a = np.asarray(aabb, dtype=float).reshape(6)
        return cls(lower=a[:3], upper=a[3:])

    @classmethod
    def from_points(cls, points: np.ndarray) -> AABB:
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        if p.size == 0:
            raise ValueError("Cannot build an AABB from zero points.")
        return cls(lower=p.min(axis=0), upper=p.max(axis=0))

    # ---- properties -----------------------------------------------------
    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.lower + self.upper)

    @property
    def half_extents(self) -> np.ndarray:
        return 0.5 * (self.upper - self.lower)

    @property
    def size(self) -> np.ndarray:
        return self.upper - self.lower

    @property
    def volume(self) -> float:
        return float(np.prod(self.size))

    def corners(self) -> np.ndarray:
        """Return the 8 corners as a (8, 3) array."""
        lo, hi = self.lower, self.upper
        return np.array(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
            dtype=float,
        )

    def to_obb(self) -> OBB:
        return OBB(center=self.center, axes=np.eye(3), half_extents=self.half_extents)

    def extents_transform(self) -> tuple[np.ndarray, np.ndarray]:
        """Full side lengths and the 4x4 pose used to build an ``fcl.Box``."""
        t = np.eye(4)
        t[:3, 3] = self.center
        return self.size, t


# -----------------------------
# OBB
# -----------------------------


@dataclass(slots=True)
class OBB:
    """Oriented bounding box: a center, three orthonormal axes and half-extents."""

    center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    axes: np.ndarray = field(default_factory=lambda: np.eye(3))
    half_extents: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    def __post_init__(self) -> None:
        self.center = _vec3(self.center, "center")
        self.half_extents = np.abs(_vec3(self.half_extents, "half_extents"))

        axes = np.asarray(self.axes, dtype=float).reshape(3, 3)
        if not np.all(np.isfinite(axes)):
            raise ValueError("OBB axes have non-finite values.")
        norms = np.linalg.norm(axes, axis=1)
        if np.any(norms == 0.0):
            raise ValueError("OBB axes must be non-zero vectors.")
        # Normalise each row so the box orientation is a clean rotation.
        self.axes = axes / norms[:, np.newaxis]

    # ---- constructors ---------------------------------------------------
    @classmethod
    def from_isaac(cls, centroid: Sequence[float], axes: np.ndarray, half_extents: Sequence[float]) -> OBB:
        """Build from Isaac Sim ``compute_obb`` output ``(centroid, axes, half_extent)``."""
        return cls(center=centroid, axes=axes, half_extents=half_extents)

    @classmethod
    def from_aabb(cls, aabb: AABB) -> OBB:
        return aabb.to_obb()

    # ---- properties -----------------------------------------------------
    @property
    def volume(self) -> float:
        return float(8.0 * np.prod(self.half_extents))

    def corners(self) -> np.ndarray:
        """Return the 8 corners as a (8, 3) array."""
        signs = np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float
        )
        # corner = center + sum_i (sign_i * half_extent_i * axis_i)
        return self.center + (signs * self.half_extents) @ self.axes

    def enclosing_aabb(self) -> AABB:
        """Smallest axis-aligned box that fully contains this OBB."""
        radius = (np.abs(self.axes) * self.half_extents[:, np.newaxis]).sum(axis=0)
        return AABB(lower=self.center - radius, upper=self.center + radius)

    def extents_transform(self) -> tuple[np.ndarray, np.ndarray]:
        """Full side lengths and the 4x4 pose used to build an ``fcl.Box``.

        ``axes`` rows are the world directions of the local x/y/z axes, so the
        local->world rotation is ``axes.T`` (those directions as columns).
        """
        t = np.eye(4)
        t[:3, :3] = self.axes.T
        t[:3, 3] = self.center
        return 2.0 * self.half_extents, t


BoundingVolume = AABB | OBB


# -----------------------------
# Manager: logical operations
# -----------------------------


class BoundingVolumeOps:
    """Logical operations between axis-aligned and oriented bounding boxes.

    Overlap/distance are delegated to FCL; containment and union/intersection
    regions are computed analytically (FCL has no such query). An instance
    carries a default ``tolerance`` -- a positive value grows the clash region
    (report contact within ``tolerance``); a negative value requires that much
    penetration before reporting overlap. Every method accepts a per-call
    ``tolerance`` override.

    Operations are grouped by arity:
      * one object    -> :meth:`volume`, :meth:`corners`, :meth:`contains_point`
      * two objects   -> :meth:`intersects`, :meth:`disjoint`, :meth:`contains`,
                         :meth:`distance`, :meth:`union`, :meth:`intersection`
      * many objects  -> :meth:`any_intersecting`, :meth:`all_disjoint`,
                         :meth:`colliding_pairs`, :meth:`intersection_matrix`,
                         :meth:`union_all`, :meth:`contains_all`,
                         :meth:`containment_mask`
    """

    def __init__(self, tolerance: float = 0.0) -> None:
        self.tolerance = float(tolerance)

    # ---- helpers --------------------------------------------------------
    def _tol(self, tolerance: float | None) -> float:
        return self.tolerance if tolerance is None else float(tolerance)

    @staticmethod
    def _as_obb(box: BoundingVolume) -> OBB:
        if isinstance(box, OBB):
            return box
        if isinstance(box, AABB):
            return box.to_obb()
        raise TypeError(f"Expected AABB or OBB, got {type(box).__name__}.")

    @staticmethod
    def _as_aabb(box: BoundingVolume) -> AABB:
        if isinstance(box, AABB):
            return box
        if isinstance(box, OBB):
            return box.enclosing_aabb()
        raise TypeError(f"Expected AABB or OBB, got {type(box).__name__}.")

    @staticmethod
    def _fcl_object(box: BoundingVolume) -> fcl.CollisionObject:
        if not isinstance(box, (AABB, OBB)):
            raise TypeError(f"Expected AABB or OBB, got {type(box).__name__}.")
        extents, transform = box.extents_transform()
        # FCL requires strictly positive side lengths.
        extents = np.maximum(np.asarray(extents, dtype=float), _EPS)
        geom = fcl.Box(float(extents[0]), float(extents[1]), float(extents[2]))
        pose = fcl.Transform(np.ascontiguousarray(transform[:3, :3]), transform[:3, 3])
        return fcl.CollisionObject(geom, pose)

    @staticmethod
    def _signed_distance(a: fcl.CollisionObject, b: fcl.CollisionObject) -> float:
        request = fcl.DistanceRequest(enable_signed_distance=True)
        result = fcl.DistanceResult()
        fcl.distance(a, b, request, result)
        return float(result.min_distance)

    # =====================================================================
    # One object
    # =====================================================================
    def volume(self, box: BoundingVolume) -> float:
        return box.volume

    def corners(self, box: BoundingVolume) -> np.ndarray:
        return box.corners()

    def contains_point(
        self, box: BoundingVolume, point: Sequence[float], tolerance: float | None = None
    ) -> bool:
        """Whether ``point`` lies inside ``box`` (within tolerance)."""
        tol = self._tol(tolerance)
        p = _vec3(point, "point")
        if isinstance(box, AABB):
            return bool(np.all(p >= box.lower - tol) and np.all(p <= box.upper + tol))
        obb = self._as_obb(box)
        local = obb.axes @ (p - obb.center)  # project into the OBB frame
        return bool(np.all(np.abs(local) <= obb.half_extents + tol))

    # =====================================================================
    # Two objects
    # =====================================================================
    def distance(self, a: BoundingVolume, b: BoundingVolume) -> float:
        """Signed separation: >0 gap, 0 touching, <0 penetration depth."""
        return self._signed_distance(self._fcl_object(a), self._fcl_object(b))

    def intersects(
        self, a: BoundingVolume, b: BoundingVolume, tolerance: float | None = None
    ) -> bool:
        """Whether ``a`` and ``b`` overlap (logical AND of the two volumes)."""
        return self.distance(a, b) <= self._tol(tolerance)

    def disjoint(
        self, a: BoundingVolume, b: BoundingVolume, tolerance: float | None = None
    ) -> bool:
        """Logical negation of :meth:`intersects`."""
        return not self.intersects(a, b, tolerance)

    def contains(
        self, container: BoundingVolume, other: BoundingVolume, tolerance: float | None = None
    ) -> bool:
        """Whether ``container`` fully encloses ``other`` (``other`` subset of ``container``).

        A box is the intersection of three slabs, so containment holds iff every
        corner of ``other`` lies inside ``container``.
        """
        tol = self._tol(tolerance)
        corners = self._as_obb(other).corners()
        if isinstance(container, AABB):
            return bool(
                np.all(corners >= container.lower - tol)
                and np.all(corners <= container.upper + tol)
            )
        obb = self._as_obb(container)
        local = (corners - obb.center) @ obb.axes.T  # corners in container frame
        return bool(np.all(np.abs(local) <= obb.half_extents + tol))

    def union(self, a: BoundingVolume, b: BoundingVolume) -> AABB:
        """Smallest AABB enclosing both volumes (logical OR bound)."""
        aa, bb = self._as_aabb(a), self._as_aabb(b)
        return AABB(lower=np.minimum(aa.lower, bb.lower), upper=np.maximum(aa.upper, bb.upper))

    def intersection(
        self, a: BoundingVolume, b: BoundingVolume, tolerance: float | None = None
    ) -> AABB | None:
        """Overlap region as an AABB, or ``None`` if the volumes are disjoint.

        For oriented boxes this is the intersection of their enclosing AABBs, so
        it is a conservative (outer) bound on the true intersection region.
        """
        tol = self._tol(tolerance)
        aa, bb = self._as_aabb(a), self._as_aabb(b)
        lower = np.maximum(aa.lower, bb.lower)
        upper = np.minimum(aa.upper, bb.upper)
        if np.any(lower > upper + tol):
            return None
        return AABB(lower=lower, upper=np.maximum(lower, upper))

    # =====================================================================
    # Many objects
    # =====================================================================
    def any_intersecting(
        self, boxes: Sequence[BoundingVolume], tolerance: float | None = None
    ) -> bool:
        """Whether *any* pair among ``boxes`` overlaps."""
        tol = self._tol(tolerance)
        objs = [self._fcl_object(bx) for bx in boxes]
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                if self._signed_distance(objs[i], objs[j]) <= tol:
                    return True
        return False

    def all_disjoint(
        self, boxes: Sequence[BoundingVolume], tolerance: float | None = None
    ) -> bool:
        """Whether *no* pair among ``boxes`` overlaps."""
        return not self.any_intersecting(boxes, tolerance)

    def colliding_pairs(
        self, boxes: Sequence[BoundingVolume], tolerance: float | None = None
    ) -> list[tuple[int, int]]:
        """Indices ``(i, j)`` with ``i < j`` of every overlapping pair."""
        tol = self._tol(tolerance)
        objs = [self._fcl_object(bx) for bx in boxes]
        pairs: list[tuple[int, int]] = []
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                if self._signed_distance(objs[i], objs[j]) <= tol:
                    pairs.append((i, j))
        return pairs

    def intersection_matrix(
        self, boxes: Sequence[BoundingVolume], tolerance: float | None = None
    ) -> np.ndarray:
        """Symmetric ``(N, N)`` boolean overlap matrix (diagonal is ``True``)."""
        tol = self._tol(tolerance)
        objs = [self._fcl_object(bx) for bx in boxes]
        n = len(objs)
        mat = np.eye(n, dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                hit = self._signed_distance(objs[i], objs[j]) <= tol
                mat[i, j] = mat[j, i] = hit
        return mat

    def union_all(self, boxes: Sequence[BoundingVolume]) -> AABB:
        """Smallest AABB enclosing every volume in ``boxes``."""
        aabbs = [self._as_aabb(bx) for bx in boxes]
        if not aabbs:
            raise ValueError("union_all requires at least one box.")
        lower = np.min([bx.lower for bx in aabbs], axis=0)
        upper = np.max([bx.upper for bx in aabbs], axis=0)
        return AABB(lower=lower, upper=upper)

    def contains_all(
        self,
        container: BoundingVolume,
        boxes: Sequence[BoundingVolume],
        tolerance: float | None = None,
    ) -> bool:
        """Whether ``container`` encloses every box in ``boxes``."""
        return all(self.contains(container, bx, tolerance) for bx in boxes)

    def containment_mask(
        self,
        container: BoundingVolume,
        boxes: Sequence[BoundingVolume],
        tolerance: float | None = None,
    ) -> np.ndarray:
        """Boolean array flagging which boxes are enclosed by ``container``."""
        return np.array([self.contains(container, bx, tolerance) for bx in boxes], dtype=bool)
