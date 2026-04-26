"""FastAPI routers."""
from .catalog import router as catalog_router
from .solve import router as solve_router

__all__ = ["catalog_router", "solve_router"]
