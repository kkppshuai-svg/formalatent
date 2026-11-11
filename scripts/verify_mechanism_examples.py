"""Replay one masked AI-CAD joint against an independent motion specification.

Part poses are translations in the rest world frame. Joint axes and pivots
are expressed in that frame as well. The selected joint moves while all
other joints remain at rest. This is a sampled rigid-body check.
"""

from itertools import combinations
import math

import cadquery as cq


SAMPLE_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
POSITION_TOLERANCE_MM = 1e-6
VOLUME_TOLERANCE_MM3 = 1e-6
SUPPORTED_JOINTS = {"fixed", "revolute", "prismatic"}
ZERO = (0.0, 0.0, 0.0)


def vector(values, label):
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{label} must have three coordinates")
    result = tuple(float(value) for value in values)
    if not all(map(math.isfinite, result)):
        raise ValueError(f"{label} must have finite coordinates")
    return result


def unit_axis(values, label):
    axis = vector(values, label)
    norm = math.hypot(*axis)
    if norm == 0 or not math.isfinite(norm):
        raise ValueError(f"{label} must be a finite nonzero axis")
    return tuple(value / norm for value in axis)


def expected_position(point, kind, axis, pivot, travel):
    """Analytic target position for the probe, independent of CAD placement."""
    if kind == "prismatic":
        return tuple(coord + travel * direction for coord, direction in zip(point, axis))
    x, y, z = (coord - origin for coord, origin in zip(point, pivot))
    a, b, c = axis
    cross = (b * z - c * y, c * x - a * z, a * y - b * x)
    dot = a * x + b * y + c * z
    cosine, sine = math.cos(travel), math.sin(travel)
    return tuple(
        origin + offset * cosine + skew * sine + direction * dot * (1 - cosine)
        for origin, offset, skew, direction in zip(pivot, (x, y, z), cross, axis)
    )


def cad_motion(shape, kind, axis, pivot, travel):
    if kind == "prismatic":
        return shape.translate(tuple(travel * direction for direction in axis))
    axis_end = tuple(origin + direction for origin, direction in zip(pivot, axis))
    return shape.rotate(pivot, axis_end, math.degrees(travel))


def rest_shapes(parts, build_part):
    shapes = {}
    for part_id, part in parts.items():
        pose = part.get("pose") or {}
        if vector(pose.get("rotate", ZERO), f"{part_id} rotation") != ZERO:
            raise ValueError(f"{part_id} requires a world-aligned rest frame")
        offset = vector(pose.get("translate", ZERO), f"{part_id} translation")
        built = build_part(part)
        solids = built.solids().vals()
        if not solids or any(
            not solid.isValid() or not math.isfinite(solid.Volume()) or solid.Volume() <= 0
            for solid in solids
        ):
            raise ValueError(f"{part_id} has invalid or empty CAD solids")
        shape = solids[0] if len(solids) == 1 else cq.Compound.makeCompound(solids)
        shapes[part_id] = shape.translate(offset)
    return shapes


def part_ancestors(parts, joints, selected_id, child):
    parent_of = {}
    for joint in joints:
        parent = joint["parent"]
        target = child if joint["id"] == selected_id else joint["child"]
        if parent != "world" and parent not in parts:
            raise ValueError(f"unknown joint parent {parent}")
        if target not in parts:
            raise ValueError(f"unknown joint child {target}")
        if target in parent_of:
            raise ValueError(f"multiple joints attach to {target}")
        parent_of[target] = parent

    ancestors = {}
    for part in parts:
        current = part
        lineage = set()
        while current in parent_of:
            if current in lineage:
                raise ValueError(f"joint cycle at {current}")
            lineage.add(current)
            current = parent_of[current]
        ancestors[part] = lineage | {part, current}
    return ancestors


def collision(shapes, pairs):
    bounds = {part_id: shape.BoundingBox() for part_id, shape in shapes.items()}
    for left, right in pairs:
        a, b = bounds[left], bounds[right]
        if any(
            min(getattr(a, f"{axis}max"), getattr(b, f"{axis}max"))
            <= max(getattr(a, f"{axis}min"), getattr(b, f"{axis}min"))
            for axis in "xyz"
        ):
            continue
        overlap = shapes[left].intersect(shapes[right]).Volume()
        if overlap > VOLUME_TOLERANCE_MM3:
            return {"parts": [left, right], "volumeMm3": overlap}
    return None


def result(replay_passed, motion_passed, reason, **details):
    return dict(replayPassed=replay_passed, motionPassed=motion_passed, reason=reason, **details)


def verify_candidates(example, build_part, candidate_ids=None):
    """Return geometry and motion outcomes keyed by candidate child part ID."""
    plan, probe = example["plan"], example["probe"]
    parts = {part["id"]: part for part in plan["parts"]}
    if (len(parts) != len(plan["parts"]) or not parts or "world" in parts
            or any(not isinstance(part, str) or not part for part in parts)):
        raise ValueError("part IDs must be nonempty and unique, and world is reserved")
    joints = plan["joints"]
    joints_by_id = {joint["id"]: joint for joint in joints}
    if len(joints_by_id) != len(joints) or probe["jointId"] not in joints_by_id:
        raise ValueError("joint IDs must be unique and include the probed joint")
    if any(joint["type"] not in SUPPORTED_JOINTS for joint in joints):
        raise ValueError("unsupported joint type in replay")
    selected = joints_by_id[probe["jointId"]]
    kind = selected["type"]
    if kind == "fixed":
        raise ValueError("the probed joint must be movable")
    if vector(selected["origin"].get("rpy", ZERO), "joint rotation") != ZERO:
        raise ValueError("the probed joint requires a world-aligned rest frame")
    pivot = vector(selected["origin"]["xyz"], "joint pivot")
    axis = unit_axis(selected["axis"], "joint axis")
    goal_pivot = vector(probe["pivot"], "probe pivot")
    goal_axis = unit_axis(probe["axis"], "probe axis")
    travel = float(probe["travel"])
    if not math.isfinite(travel) or travel == 0:
        raise ValueError("probe travel must be finite and nonzero")
    goal_kind = probe.get("type", kind)
    if goal_kind not in {"revolute", "prismatic"}:
        raise ValueError("unsupported probe motion")
    moving = {probe["movingPart"], *probe.get("movingParts", [])}
    stationary = set(probe["stationaryParts"])
    if moving & stationary or not (moving | stationary) <= parts.keys():
        raise ValueError("probe must identify disjoint moving and stationary CAD parts")
    candidates = [part for part in parts if part != selected["parent"]]
    if candidate_ids is not None:
        requested = list(candidate_ids)
        if len(set(requested)) != len(requested) or not set(requested).issubset(candidates):
            raise ValueError("requested children must be unique candidate part IDs")
        candidates = requested
    pairs = tuple(combinations(parts, 2))
    try:
        shapes = rest_shapes(parts, build_part)
        rest_collision = collision(shapes, pairs)
    except (ValueError, TypeError, RuntimeError) as exc:
        return {part: result(False, False, "geometry_build_failed", detail=str(exc)) for part in candidates}
    if rest_collision:
        return {part: result(False, False, "rest_interference", collision=rest_collision) for part in candidates}

    witnesses = {}
    for part in moving | stationary:
        center = shapes[part].Center().toTuple()
        offsets = (ZERO, (1, 0, 0), (0, 1, 0), (0, 0, 1))
        points = [tuple(x + dx for x, dx in zip(center, offset)) for offset in offsets]
        witnesses[part] = [(point, cq.Vertex.makeVertex(*point)) for point in points]

    outcomes = {}
    for child in candidates:
        try:
            ancestors = part_ancestors(parts, joints, selected["id"], child)
        except ValueError as exc:
            outcomes[child] = result(False, False, "invalid_joint_graph", detail=str(exc))
            continue
        actual_moving = {part for part, lineage in ancestors.items() if child in lineage}
        moving_pairs = [(a, b) for a, b in pairs if (a in actual_moving) != (b in actual_moving)]
        max_error = 0.0
        first_collision = None
        for fraction in SAMPLE_FRACTIONS:
            step = fraction * travel
            positioned = {
                part: cad_motion(shape, kind, axis, pivot, step) if part in actual_moving else shape
                for part, shape in shapes.items()
            }
            if first_collision is None:
                hit = collision(positioned, moving_pairs)
                if hit is not None:
                    first_collision = {"fraction": fraction, **hit}
            for part, points in witnesses.items():
                for point, vertex in points:
                    expected = expected_position(point, goal_kind, goal_axis, goal_pivot, step) if part in moving else point
                    actual = cad_motion(vertex, kind, axis, pivot, step).Center().toTuple() if part in actual_moving else point
                    max_error = max(max_error, math.dist(expected, actual))
        evidence = {"maxMotionErrorMm": max_error, "sampleFractions": list(SAMPLE_FRACTIONS)}
        if first_collision:
            outcomes[child] = result(True, False, "motion_interference", collision=first_collision, **evidence)
        elif max_error > POSITION_TOLERANCE_MM:
            outcomes[child] = result(True, False, "motion_goal_mismatch", **evidence)
        else:
            outcomes[child] = result(True, True, "passed", **evidence)
    return outcomes
