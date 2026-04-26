from app.env.heightmap import Heightmap
from app.env.packing_env import PackingEnv
from app.schemas import Position


def test_heightmap_drop_on_empty_is_zero(container_40hc):
    hm = Heightmap(container_40hc, resolution_mm=50)
    assert hm.drop_y(0, 0, 1200, 800) == 0


def test_heightmap_place_then_drop_matches_top(container_40hc):
    hm = Heightmap(container_40hc, resolution_mm=50)
    hm.place(Position(x_mm=0, y_mm=0, z_mm=0), l_mm=1200, w_mm=800, h_mm=1200)
    # A same-footprint item dropped at the origin should now rest on top.
    assert hm.drop_y(0, 0, 1200, 800) == 1200
    # An offset item that doesn't overlap should still drop to the floor.
    assert hm.drop_y(2000, 1000, 1000, 1000) == 0


def test_heightmap_support_ratio_floor_is_one(container_40hc):
    hm = Heightmap(container_40hc, resolution_mm=50)
    assert hm.support_ratio(0, 0, 0, 1000, 500) == 1.0


def test_packing_env_reset_yields_candidates(container_40hc, eur_pallets_10):
    env = PackingEnv(container=container_40hc, items=eur_pallets_10)
    obs, info = env.reset()
    assert "ems" in obs and "item" in obs and "mask" in obs
    assert info["n_placed"] == 0
    assert info["n_remaining"] == 10
    assert len(env.state.candidates) > 0


def test_packing_env_single_step_places_and_decrements(container_40hc, eur_pallets_10):
    env = PackingEnv(container=container_40hc, items=eur_pallets_10)
    env.reset()
    obs, reward, done, _, info = env.step(0)
    assert info["n_placed"] == 1
    assert info["n_remaining"] == 9
    assert len(env.state.placements) == 1
    assert reward != 0.0
    assert not done
