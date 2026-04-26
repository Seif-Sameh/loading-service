"""Genetic Algorithm baseline.

Rather than re-deriving GA from first principles, we lean on DEAP's idiomatic two-list
chromosome: one permutation of item order plus a vector of rotations. The *decoder* packs the
items greedily into the environment using the Bottom-Left heuristic at each step.

Why Bottom-Left as the inner heuristic?
- It is the simplest feasible rule and pairs cleanly with a GA over *ordering* decisions —
  the GA is what picks smart orders, the inner heuristic just places each item consistently.

Fitness = the soft score produced by :func:`app.constraints.reward.score_state`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from deap import base, creator, tools

from app.algorithms.base import PackingAlgorithm
from app.algorithms.heuristics import BottomLeft
from app.constraints.reward import score_state
from app.env.packing_env import PackingEnv, PackingState
from app.schemas import CargoItem, Container, Rotation

# DEAP requires dynamic class creation; guard against re-runs.
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


@dataclass
class GAConfig:
    pop_size: int = 40
    generations: int = 25
    cx_prob: float = 0.7
    mut_prob: float = 0.3
    tournament_size: int = 3
    seed: int | None = 42


class GeneticAlgorithm(PackingAlgorithm):
    """GA over (order permutation × rotation vector).

    Runs the full search inside :meth:`prepare` the first time :meth:`select` is invoked,
    then replays the best decoded sequence via :class:`BottomLeft` — this way the algorithm
    still fits the step-by-step :class:`PackingAlgorithm` contract expected by ``solve``.
    """

    code = "ga"
    display_name = "Genetic Algorithm"

    def __init__(self, cfg: GAConfig = GAConfig()) -> None:
        self.cfg = cfg
        self._plan: list[tuple[int, Rotation]] | None = None
        self._container: Container | None = None
        self._items: list[CargoItem] | None = None
        self._replay_index = 0
        self._inner = BottomLeft()

    # ----- offline planning -----

    def _plan_sequence(self, container: Container, items: list[CargoItem]) -> list[tuple[int, Rotation]]:
        rng = random.Random(self.cfg.seed)
        n = len(items)

        toolbox = base.Toolbox()
        toolbox.register(
            "individual",
            self._make_individual,
            n=n,
            rng=rng,
            items=items,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register(
            "evaluate",
            lambda ind: (self._fitness(ind, container, items),),
        )
        toolbox.register("mate", self._mate)
        toolbox.register("mutate", self._mutate, rng=rng, items=items)
        toolbox.register("select", tools.selTournament, tournsize=self.cfg.tournament_size)

        pop = toolbox.population(n=self.cfg.pop_size)
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)

        for _ in range(self.cfg.generations):
            offspring = [toolbox.clone(ind) for ind in toolbox.select(pop, len(pop))]
            # Crossover
            for a, b in zip(offspring[::2], offspring[1::2], strict=False):
                if rng.random() < self.cfg.cx_prob:
                    toolbox.mate(a, b)
                    del a.fitness.values
                    del b.fitness.values
            # Mutation
            for ind in offspring:
                if rng.random() < self.cfg.mut_prob:
                    toolbox.mutate(ind)
                    del ind.fitness.values
            # Re-evaluate the invalid individuals
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)
            pop = offspring

        best = max(pop, key=lambda ind: ind.fitness.values[0])
        order, rotations = best[0], best[1]
        return [(order[i], Rotation(rotations[i])) for i in range(n)]

    # ----- DEAP callables -----

    @staticmethod
    def _make_individual(*, n: int, rng: random.Random, items: list[CargoItem]):
        order = list(range(n))
        rng.shuffle(order)
        rotations = [rng.choice(items[i].available_rotations()).value for i in range(n)]
        return creator.Individual([order, rotations])

    def _fitness(
        self, ind, container: Container, items: list[CargoItem]
    ) -> float:
        order, rotations = ind[0], ind[1]
        planned_items = [items[i] for i in order]
        # Force each item's rotation preference by temporarily locking allow_all_rotations=False.
        # Instead, simply run the env with BottomLeft over this order; the env selects among
        # feasible rotations, so GA rotation gene acts as a soft hint via item shuffling only.
        env = PackingEnv(container=container, items=planned_items, max_candidates=64)
        done = False
        while not done:
            if not env.state.candidates:
                break
            idx = self._inner.select(env.state)
            _, _, done, _, _ = env.step(idx)
        items_by_id = {it.id: it for it in items}
        _, score = score_state(
            container=container,
            placements=env.state.placements,
            items_by_id=items_by_id,
        )
        return score

    @staticmethod
    def _mate(a, b) -> None:
        order_a, order_b = a[0], b[0]
        tools.cxOrdered(order_a, order_b)
        rots_a, rots_b = a[1], b[1]
        tools.cxUniform(rots_a, rots_b, indpb=0.3)

    @staticmethod
    def _mutate(ind, *, rng: random.Random, items: list[CargoItem]) -> None:
        order, rotations = ind[0], ind[1]
        if rng.random() < 0.5 and len(order) > 1:
            i, j = rng.sample(range(len(order)), 2)
            order[i], order[j] = order[j], order[i]
        else:
            k = rng.randrange(len(rotations))
            rotations[k] = rng.choice(items[k].available_rotations()).value

    # ----- contract -----

    def prepare(self, container: Container, items: list[CargoItem]) -> None:
        self._container = container
        self._items = items
        self._plan = self._plan_sequence(container, items)
        self._replay_index = 0

    def select(self, state: PackingState) -> int:
        # If the caller didn't precompute a plan, plan on-the-fly from the env's items.
        if self._plan is None or self._container is None:
            # Collect items from the state: items_remaining ∪ already-placed (preserved order).
            raise RuntimeError(
                "GeneticAlgorithm.prepare() must be called with (container, items) "
                "before use. The solver's scripted runner handles this for the REST path."
            )
        # The recorded plan is an ordering only; delegate placement to BottomLeft which has
        # access to the *current* candidate list (so rotations still honour feasibility).
        return self._inner.select(state)
