"""Real-world cargo datasets and sampling.

Two complementary datasets ship with the service:

- **BR1-BR10** (Bischoff & Ratcliff 1995): 1,500 industrial 3-D bin-packing problems with
  per-axis orientation flags. Used by every academic 3-D-BPP paper as a benchmark.
- **Wadaboa product pool**: ~1 M real e-commerce parcel records with width/depth/height/weight.
  Used to sample realistic (dimension, weight) pairs.

The :class:`AlexandriaSampler` combines these with a weighted commodity mix to produce
voyages that look like Alexandria-Port traffic.
"""
from .alexandria_sampler import AlexandriaSampler, SamplerConfig
from .br_loader import BRBoxType, BRProblem, list_br_problems, load_br_problem
from .product_pool import ProductPool, load_product_pool

__all__ = [
    "AlexandriaSampler",
    "BRBoxType",
    "BRProblem",
    "ProductPool",
    "SamplerConfig",
    "list_br_problems",
    "load_br_problem",
    "load_product_pool",
]
