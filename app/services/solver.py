"""Solver orchestration: runs an algorithm and returns/streams placements.

Stays unaware of FastAPI; the API layer adapts these generators to REST + WebSockets.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
import asyncio

from app.algorithms import get_algorithm
from app.algorithms.base import StepEvent, iter_solve, solve
from app.algorithms.ga import GeneticAlgorithm
from app.schemas import CargoItem, Container, SolveResult


class SolverService:
    @staticmethod
    def run(
        *,
        container: Container,
        items: list[CargoItem],
        algorithm: str = "extreme_points",
        seed: int | None = None,
    ) -> SolveResult:
        algo = get_algorithm(algorithm)
        if isinstance(algo, GeneticAlgorithm):
            algo.prepare(container, items)
        result, _ = solve(algorithm=algo, container=container, items=items, seed=seed)
        return result

    @staticmethod
    async def stream(
        *,
        container: Container,
        items: list[CargoItem],
        algorithm: str = "extreme_points",
        seed: int | None = None,
    ) -> AsyncIterator[StepEvent | SolveResult]:
        algo = get_algorithm(algorithm)
        if isinstance(algo, GeneticAlgorithm):
            algo.prepare(container, items)
        for event in iter_solve(
            algorithm=algo,
            container=container,
            items=items,
            seed=seed,
        ):
            yield event
            # Yield to the event loop between placements so other coroutines run.
            await asyncio.sleep(0)
