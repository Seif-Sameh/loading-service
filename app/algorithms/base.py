"""Algorithm interface shared by heuristics, GA, and RL."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

from app.env.packing_env import PackingEnv, PackingState
from app.schemas import (
    CandidateAction,
    CargoItem,
    Container,
    Placement,
    SolveResult,
)


class PackingAlgorithm(ABC):
    """Minimum contract: given a state with feasible candidates, pick one."""

    code: str = "base"
    display_name: str = "base"

    @abstractmethod
    def select(self, state: PackingState) -> int:
        """Return an index into ``state.candidates``."""

    def attach_env(self, env) -> None:  # noqa: ANN001
        """Optional hook: receive the live :class:`PackingEnv` reference each step.

        Default: no-op. Algorithms that need to roll the environment forward (e.g. ensemble /
        lookahead / MCTS) override this to keep a reference and fork it inside
        :meth:`select`. Called by :func:`solve` and :func:`iter_solve` before each step.
        """
        return None


@dataclass
class StepEvent:
    """Emitted after each step — fed to the WebSocket streamer."""

    step: int
    placement: Placement
    remaining: int


def solve(
    *,
    algorithm: PackingAlgorithm,
    container: Container,
    items: list[CargoItem],
    heightmap_resolution_mm: int = 10,
    max_candidates: int = 80,
    seed: int | None = None,
) -> tuple[SolveResult, list[StepEvent]]:
    """Run an algorithm end-to-end and collect step events for streaming."""
    env = PackingEnv(
        container=container,
        items=items,
        heightmap_resolution_mm=heightmap_resolution_mm,
        max_candidates=max_candidates,
        seed=seed,
    )

    events: list[StepEvent] = []
    t0 = time.perf_counter()
    done = False
    algorithm.attach_env(env)
    while not done:
        state = env.state
        if not state.candidates:
            break
        idx = algorithm.select(state)
        _, _, done, _, _ = env.step(idx)
        if env.state.placements and (
            not events or events[-1].placement is not env.state.placements[-1]
        ):
            events.append(
                StepEvent(
                    step=env.state.step_index,
                    placement=env.state.placements[-1],
                    remaining=len(env.state.items_remaining),
                )
            )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    score, kpis = env.final_score()
    unplaced = [it.id for it in env.state.unplaced + env.state.items_remaining]
    result = SolveResult(
        algorithm=algorithm.code,
        container_code=container.code,
        placements=env.state.placements,
        unplaced_item_ids=unplaced,
        kpis=kpis,
        elapsed_ms=elapsed_ms,
    )
    return result, events


def iter_solve(
    *,
    algorithm: PackingAlgorithm,
    container: Container,
    items: list[CargoItem],
    heightmap_resolution_mm: int = 10,
    max_candidates: int = 80,
    seed: int | None = None,
) -> Iterator[StepEvent | SolveResult]:
    """Stream placements as they happen (for WebSocket consumers). Yields StepEvent until
    done, then one final :class:`SolveResult`."""
    env = PackingEnv(
        container=container,
        items=items,
        heightmap_resolution_mm=heightmap_resolution_mm,
        max_candidates=max_candidates,
        seed=seed,
    )
    t0 = time.perf_counter()
    done = False
    algorithm.attach_env(env)
    while not done:
        state = env.state
        if not state.candidates:
            break
        idx = algorithm.select(state)
        _, _, done, _, _ = env.step(idx)
        yield StepEvent(
            step=env.state.step_index,
            placement=env.state.placements[-1],
            remaining=len(env.state.items_remaining),
        )
    score, kpis = env.final_score()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    yield SolveResult(
        algorithm=algorithm.code,
        container_code=container.code,
        placements=env.state.placements,
        unplaced_item_ids=[it.id for it in env.state.unplaced + env.state.items_remaining],
        kpis=kpis,
        elapsed_ms=elapsed_ms,
    )


def pick_default(state: PackingState) -> int:
    """Safety net: lowest-y, then lowest-x placement."""
    best = 0
    best_key: tuple[int, int, int] | None = None
    for i, c in enumerate(state.candidates):
        key = (c.position.y_mm, c.position.x_mm, c.position.z_mm)
        if best_key is None or key < best_key:
            best = i
            best_key = key
    return best
