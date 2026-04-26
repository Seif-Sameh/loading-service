"""Static reference data: containers, cargo presets, IMDG segregation table."""
from .loader import (
    IMDGTable,
    get_cargo_preset,
    get_container,
    imdg_table,
    list_cargo_presets,
    list_containers,
)

__all__ = [
    "IMDGTable",
    "get_cargo_preset",
    "get_container",
    "imdg_table",
    "list_cargo_presets",
    "list_containers",
]
