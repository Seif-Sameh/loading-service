from app.schemas import (
    Dimensions,
    HazmatClass,
    Rotation,
    UPRIGHT_ROTATIONS,
)


def test_dimensions_rotated_swaps_horizontal_axes_only_for_upright():
    d = Dimensions(length_mm=1000, width_mm=600, height_mm=1200)
    assert d.rotated(Rotation.LWH) == d
    swapped = d.rotated(Rotation.WLH)
    assert swapped.length_mm == 600 and swapped.width_mm == 1000
    assert swapped.height_mm == 1200


def test_dimensions_rotated_tips_for_non_upright():
    d = Dimensions(length_mm=1000, width_mm=600, height_mm=1200)
    tipped = d.rotated(Rotation.LHW)
    assert tipped.length_mm == 1000  # length unchanged
    assert tipped.width_mm == 1200  # formerly height
    assert tipped.height_mm == 600  # formerly width


def test_upright_rotations_constant():
    assert set(UPRIGHT_ROTATIONS) == {Rotation.LWH, Rotation.WLH}


def test_hazmat_class_enum_coerces_from_str():
    assert HazmatClass("3") is HazmatClass.C3
    assert HazmatClass("none") is HazmatClass.NONE
