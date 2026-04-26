"""Service-layer orchestration sitting between the FastAPI routers and the algorithm core."""
from .solver import SolverService

__all__ = ["SolverService"]
