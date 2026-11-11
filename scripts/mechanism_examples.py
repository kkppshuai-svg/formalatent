"""Synthetic AI-CAD plans with independent single-joint motion goals.

Geometry is in millimetres in the rest world frame. Joint axes are world
vectors. Revolute travel is in radians and prismatic travel in millimetres.
The examples exercise directed joint recovery with rigid, low-detail solids.
"""

from dataclasses import dataclass
from random import Random


X = (1, 0, 0)
Y = (0, 1, 0)
Z = (0, 0, 1)
ORIGIN = (0, 0, 0)


@dataclass(frozen=True)
class Body:
    name: str
    role: str
    center: tuple
    size: tuple
    hole: bool = False

    def part(self, part_id, scale, shift):
        size = [round(value * scale, 6) for value in self.size]
        if len(size) == 3:
            stock = {"type": "base_box", "params": {"size": size, "center": True}}
        else:
            stock = {"type": "base_cylinder", "params": {
                "diameter": size[0], "height": size[1], "center": True
            }}
        nodes = [{"id": "stock", **stock}]
        if self.hole:
            nodes.append({
                "id": "bore", "type": "hole", "dependsOn": ["stock"],
                "params": {"diameter": round(0.25 * min(size[:-1]), 6),
                           "depth": size[-1], "center": [0, 0, 0], "axis": "z"},
            })
        return {
            "id": part_id, "name": self.name, "role": self.role, "featureTree": nodes,
            "pose": {"translate": scaled_point(self.center, scale, shift), "rotate": [0, 0, 0]},
        }


@dataclass(frozen=True)
class Family:
    name: str
    bodies: tuple
    fixed: tuple
    driver: tuple
    witness: str
    joint_type: str
    axis: tuple
    pivot: tuple
    travel: float


BASE = Body("base", "mount", (0, 0, -3), (34, 28, 6))
FAMILIES = (
    Family(
        "plate-hinge",
        (BASE, Body("leaf", "carrier", (9, 0, 2.2), (20, 8, 4), True)),
        (("world", "base"),), ("base", "leaf"), "leaf",
        "revolute", Z, ORIGIN, 0.32,
    ),
    Family(
        "linear-rail",
        (Body("base", "guide", (0, 0, -3), (50, 16, 6), True),
         Body("slide", "carrier", (-8, 0, 2.2), (12, 8, 4), True)),
        (("world", "base"),), ("base", "slide"), "slide",
        "prismatic", X, (-8, 0, 0), 10.0,
    ),
    Family(
        "bearing-turntable",
        (BASE, Body("bearing", "support", (0, 0, 2.05), (14, 4), True),
         Body("platter", "carrier", (4, 0, 6.2), (16, 4), True)),
        (("world", "base"), ("base", "bearing")), ("bearing", "platter"), "platter",
        "revolute", Z, (0, 0, 4), -0.26,
    ),
    Family(
        "nested-slide",
        (BASE, Body("bridge", "support", (0, 0, 2.05), (24, 30, 4)),
         Body("rail", "guide", (0, 0, 6.1), (8, 32, 4), True),
         Body("slide", "carrier", (0, -7, 10.2), (12, 8, 4))),
        (("world", "base"), ("base", "bridge"), ("bridge", "rail")),
        ("rail", "slide"), "slide", "prismatic", Y, (0, -7, 8), 9.0,
    ),
    Family(
        "swing-carrier",
        (BASE, Body("arm", "carrier", (10, 0, 2.2), (22, 6, 4), True),
         Body("payload", "payload", (17, 0, 6.3), (8, 4))),
        (("world", "base"), ("arm", "payload")), ("base", "arm"), "payload",
        "revolute", Z, ORIGIN, 0.29,
    ),
    Family(
        "telescoping-tool",
        (BASE, Body("rod", "guide", (0, 0, 4.2), (9, 8), True),
         Body("tool", "payload", (0, 0, 10.3), (13, 6, 4)),
         Body("stop", "support", (-14, 0, 2.1), (4, 10, 4))),
        (("world", "base"), ("base", "stop"), ("rod", "tool")),
        ("base", "rod"), "tool", "prismatic", Z, ORIGIN, 7.0,
    ),
    Family(
        "dual-support-gate",
        (BASE, Body("post", "support", (0, 0, 3.05), (8, 6), True),
         Body("brace", "support", (-12, 0, 2.1), (5, 10, 4)),
         Body("panel", "carrier", (11, 0, 8.2), (24, 9, 4)),
         Body("latch", "payload", (20, 0, 12.3), (5, 4, 4), True)),
        (("world", "base"), ("base", "post"), ("base", "brace"), ("panel", "latch")),
        ("post", "panel"), "latch", "revolute", Z, (0, 0, 6), 0.25,
    ),
    Family(
        "rack-mounted-lift",
        (BASE, Body("support", "support", (0, 0, 2.05), (20, 16, 4)),
         Body("rail", "guide", (0, 0, 6.1), (14, 12, 4)),
         Body("stage", "carrier", (0, 0, 10.25), (15, 11, 4), True),
         Body("fixture", "payload", (2, 0, 14.35), (7, 6, 4)),
         Body("tool", "payload", (2, 0, 18.45), (6, 4))),
        (("world", "base"), ("base", "support"), ("support", "rail"),
         ("stage", "fixture"), ("fixture", "tool")),
        ("rail", "stage"), "tool", "prismatic", Z, (0, 0, 8), 6.0,
    ),
)


def scaled_point(point, scale, shift):
    return [round(scale * value + offset, 6) for value, offset in zip(point, shift)]


def joint(joint_id, parent, child, kind, axis, pivot):
    return {
        "id": joint_id, "parent": parent, "child": child, "type": kind,
        "axis": list(axis), "origin": {"xyz": list(pivot), "rpy": [0, 0, 0]},
    }


def make_example(family, variant, rng):
    scale = rng.uniform(0.85, 1.2)
    shift = tuple(rng.uniform(-6, 6) for _ in range(3))
    moving_stock = next(body for body in family.bodies if body.name == family.driver[1])
    bodies = list(family.bodies)
    # These unconnected parts share role cues with the intended destination.
    # Substituting one keeps a valid joint tree but moves the wrong part.
    bodies.extend((
        Body("alternate-carrier", moving_stock.role, (0, 44, 4), moving_stock.size, variant % 2 == 1),
        Body("spare-guide", "guide", (0, -45, 4), (9, 8, 6)),
    ))
    if variant % 3 == 2:
        bodies.append(Body("spare-support", "support", (-46, -22, 4), (9, 6)))

    rng.shuffle(bodies)
    ids = {body.name: f"component-{index + 1}" for index, body in enumerate(bodies)}
    ids["world"] = "world"
    parts = [body.part(ids[body.name], scale, shift) for body in bodies]
    joints = [joint(f"fixture-{i + 1}", ids[parent], ids[child], "fixed", Z, list(shift))
              for i, (parent, child) in enumerate(family.fixed)]
    pivot = scaled_point(family.pivot, scale, shift)
    joints.append(joint("drive", ids[family.driver[0]], ids[family.driver[1]],
                        family.joint_type, family.axis, pivot))
    rng.shuffle(joints)

    driven = {family.driver[1]}
    for parent, child in family.fixed:
        if parent in driven:
            driven.add(child)
    travel = family.travel * (scale if family.joint_type == "prismatic" else 1)
    example_id = f"{family.name}-{variant + 1:02}"
    return {
        "id": example_id, "family": family.name,
        "plan": {"name": example_id, "engine": "cadquery", "parts": parts, "joints": joints},
        "probe": {
            "jointId": "drive", "movingPart": ids[family.witness],
            "stationaryParts": [ids[body.name] for body in bodies if body.name not in driven],
            "travel": round(travel, 6), "axis": list(family.axis), "pivot": pivot,
        },
    }


def generate_examples(variants=3, seed=42):
    """Yield reproducible examples grouped by the template that created them."""
    if variants < 1:
        raise ValueError("variants must be positive")
    rng = Random(seed)
    for family in FAMILIES:
        for variant in range(variants):
            yield make_example(family, variant, rng)
