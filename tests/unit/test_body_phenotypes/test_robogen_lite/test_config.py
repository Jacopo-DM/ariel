"""Test: robogen_lite configuration enums and constants."""

from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    NUM_OF_FACES,
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
    ModuleFaces,
    ModuleInstance,
    ModuleRotationsIdx,
    ModuleRotationsTheta,
    ModuleType,
)

HINGE_VALUE = 2
NONE_VALUE = 3
DEG_90_INDEX = 2
DEG_45_VALUE = 45
DEG_90_VALUE = 90
CORE_FACE_COUNT = 4
BRICK_FACE_COUNT = 5


def test_module_type_members() -> None:
    """ModuleType enum has all expected members."""
    assert ModuleType.CORE.value == 0
    assert ModuleType.BRICK.value == 1
    assert ModuleType.HINGE.value == HINGE_VALUE
    assert ModuleType.NONE.value == NONE_VALUE


def test_module_faces_members() -> None:
    """ModuleFaces enum has all six faces."""
    expected = {"FRONT", "BACK", "RIGHT", "LEFT", "TOP", "BOTTOM"}
    assert {f.name for f in ModuleFaces} == expected


def test_module_rotations_idx_members() -> None:
    """ModuleRotationsIdx enum has DEG_0, DEG_45, DEG_90."""
    assert ModuleRotationsIdx.DEG_0.value == 0
    assert ModuleRotationsIdx.DEG_45.value == 1
    assert ModuleRotationsIdx.DEG_90.value == DEG_90_INDEX


def test_module_rotations_theta_members() -> None:
    """ModuleRotationsTheta enum maps to correct degree values."""
    assert ModuleRotationsTheta.DEG_0.value == 0
    assert ModuleRotationsTheta.DEG_45.value == DEG_45_VALUE
    assert ModuleRotationsTheta.DEG_90.value == DEG_90_VALUE


def test_idx_of_core_is_zero() -> None:
    """IDX_OF_CORE is always 0."""
    assert IDX_OF_CORE == 0


def test_derived_constants() -> None:
    """Derived system parameter constants match enum lengths."""
    assert len(ModuleType) == NUM_OF_TYPES_OF_MODULES
    assert len(ModuleFaces) == NUM_OF_FACES
    assert len(ModuleRotationsIdx) == NUM_OF_ROTATIONS


def test_allowed_faces_core_has_four_faces() -> None:
    """CORE is allowed four faces (FRONT, BACK, RIGHT, LEFT)."""
    assert len(ALLOWED_FACES[ModuleType.CORE]) == CORE_FACE_COUNT


def test_allowed_faces_brick_has_five_faces() -> None:
    """BRICK is allowed five faces."""
    assert len(ALLOWED_FACES[ModuleType.BRICK]) == BRICK_FACE_COUNT


def test_allowed_faces_hinge_has_one_face() -> None:
    """HINGE is allowed exactly one face (FRONT)."""
    assert ALLOWED_FACES[ModuleType.HINGE] == [ModuleFaces.FRONT]


def test_allowed_rotations_core_only_deg0() -> None:
    """CORE only allows DEG_0 rotation."""
    assert ALLOWED_ROTATIONS[ModuleType.CORE] == [ModuleRotationsIdx.DEG_0]


def test_allowed_rotations_hinge_and_brick() -> None:
    """HINGE and BRICK share the same three allowed rotations."""
    expected = [
        ModuleRotationsIdx.DEG_0,
        ModuleRotationsIdx.DEG_45,
        ModuleRotationsIdx.DEG_90,
    ]
    assert ALLOWED_ROTATIONS[ModuleType.HINGE] == expected
    assert ALLOWED_ROTATIONS[ModuleType.BRICK] == expected


def test_module_instance_construction() -> None:
    """ModuleInstance can be constructed and stores its fields correctly."""
    instance = ModuleInstance(
        type=ModuleType.BRICK,
        rotation=ModuleRotationsIdx.DEG_0,
        links={ModuleFaces.FRONT: 1},
    )
    assert instance.type == ModuleType.BRICK
    assert instance.rotation == ModuleRotationsIdx.DEG_0
    assert instance.links[ModuleFaces.FRONT] == 1
