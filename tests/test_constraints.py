from app.catalog.loader import get_cargo_preset, get_container
from app.constraints.cog import CoGTracker
from app.constraints.imdg import pair_ok
from app.constraints.mask import is_placement_feasible
from app.constraints.reward import score_state
from app.schemas import (
    CandidateAction,
    Dimensions,
    HazmatClass,
    Placement,
    Position,
    Rotation,
)


def test_cog_tracker_updates_running_average():
    cont = get_container("40HC")
    cog = CoGTracker(container=cont)
    p1 = Placement(
        item_id="a",
        position=Position(x_mm=0, y_mm=0, z_mm=0),
        rotation=Rotation.LWH,
        rotated_dimensions=Dimensions(length_mm=1200, width_mm=800, height_mm=1000),
    )
    cog.add(p1, 500)
    p2 = Placement(
        item_id="b",
        position=Position(x_mm=cont.internal.length_mm - 1200, y_mm=0, z_mm=0),
        rotation=Rotation.LWH,
        rotated_dimensions=Dimensions(length_mm=1200, width_mm=800, height_mm=1000),
    )
    cog.add(p2, 500)
    # Equal weights at opposite ends along x → CoG should be near container centre.
    assert abs(cog.longitudinal_fraction - 0.5) < 0.01


def test_imdg_pair_ok_same_class_8():
    # Two class-8 items: matrix[8][8] = 0 (no restriction).
    assert pair_ok(HazmatClass.C8, HazmatClass.C8, fore_aft_mm=0, lateral_mm=0) is True


def test_imdg_pair_ok_explosives_vs_corrosive_blocked_close():
    # Class 1 vs class 8: code = 4 (separated by compartment) — impossible in one container.
    assert pair_ok(HazmatClass.C1, HazmatClass.C8, fore_aft_mm=999999, lateral_mm=999999) is False


def test_imdg_away_from_needs_at_least_three_metres():
    # Class 3 vs class 5 → code 2 (separated from). Needs 6 m fore-aft or 2.4 m lateral.
    assert pair_ok(HazmatClass.C3, HazmatClass.C5, fore_aft_mm=1000, lateral_mm=1000) is False
    assert pair_ok(HazmatClass.C3, HazmatClass.C5, fore_aft_mm=6500, lateral_mm=0) is True
    assert pair_ok(HazmatClass.C3, HazmatClass.C5, fore_aft_mm=0, lateral_mm=2500) is True


def test_is_placement_feasible_rejects_out_of_bounds():
    cont = get_container("20GP")
    item = get_cargo_preset("eur_pallet_heavy", item_id="x")
    # Place well past the container length.
    cand = CandidateAction(
        item_index=0,
        position=Position(x_mm=5000, y_mm=0, z_mm=0),
        rotation=Rotation.LWH,
        rotated_dimensions=Dimensions(length_mm=1200, width_mm=800, height_mm=1200),
    )
    assert not is_placement_feasible(
        cand, item, cont, placed=[], items_by_id={"x": item}, current_total_weight_kg=0.0
    )


def test_is_placement_feasible_rejects_orientation_lock_violation():
    cont = get_container("20GP")
    item = get_cargo_preset("steel_drum_200l", item_id="d")
    # this_side_up=True, but candidate uses a tipped rotation.
    cand = CandidateAction(
        item_index=0,
        position=Position(x_mm=0, y_mm=0, z_mm=0),
        rotation=Rotation.LHW,
        rotated_dimensions=Dimensions(length_mm=580, width_mm=880, height_mm=580),
    )
    assert not is_placement_feasible(
        cand, item, cont, placed=[], items_by_id={"d": item}, current_total_weight_kg=0.0
    )


def test_is_placement_feasible_rejects_reefer_requirement_mismatch():
    cont = get_container("40HC")  # not a reefer
    item = get_cargo_preset("reefer_fruit_pallet", item_id="r")
    cand = CandidateAction(
        item_index=0,
        position=Position(x_mm=0, y_mm=0, z_mm=0),
        rotation=Rotation.LWH,
        rotated_dimensions=Dimensions(length_mm=1200, width_mm=1000, height_mm=1600),
    )
    assert not is_placement_feasible(
        cand, item, cont, placed=[], items_by_id={"r": item}, current_total_weight_kg=0.0
    )


def test_score_state_empty_container_is_all_zero():
    cont = get_container("20GP")
    kpis, score = score_state(container=cont, placements=[], items_by_id={})
    assert kpis.utilization == 0.0
    assert kpis.weight_used == 0.0
    assert score == 0.0
