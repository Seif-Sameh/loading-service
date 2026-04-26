from app.catalog.loader import (
    get_cargo_preset,
    get_container,
    imdg_table,
    list_cargo_presets,
    list_containers,
)
from app.schemas import HazmatClass


def test_list_containers_contains_all_iso_types():
    codes = {c.code.value for c in list_containers()}
    assert {"20GP", "40GP", "40HC", "45HC", "20RF", "40RF", "20OT", "20FR"} <= codes


def test_container_40hc_matches_iso_numbers():
    c = get_container("40HC")
    assert c.internal.length_mm == 12032
    assert c.internal.width_mm == 2352
    assert c.internal.height_mm == 2698
    assert c.payload_kg == 28800
    assert c.mgw_kg == 30480
    assert c.is_reefer is False


def test_list_cargo_presets_is_non_empty():
    presets = list_cargo_presets()
    assert len(presets) >= 10


def test_get_cargo_preset_materialises_pydantic_item():
    it = get_cargo_preset("eur_pallet_heavy", item_id="p1")
    assert it.weight_kg == 900
    assert it.dimensions.length_mm == 1200
    assert it.hazmat_class is HazmatClass.NONE


def test_hazmat_preset_sets_class_8():
    it = get_cargo_preset("hazmat_corrosive_drum", item_id="h1")
    assert it.hazmat_class is HazmatClass.C8


def test_imdg_table_dimensions():
    t = imdg_table()
    assert len(t.classes) == 10  # 1-9 + "none"
    assert len(t.matrix) == 10
    for row in t.matrix:
        assert len(row) == 10


def test_imdg_table_symmetry():
    t = imdg_table()
    for i in range(len(t.classes)):
        for j in range(len(t.classes)):
            assert t.matrix[i][j] == t.matrix[j][i], (
                f"non-symmetric at ({t.classes[i]}, {t.classes[j]})"
            )
