# loading-service

Container-loading optimization microservice for the Alexandria Port Digital Twin.

- **Algorithms:** geometric heuristics (BAF, BSSF, BLSF, BL, Extreme Points), Genetic Algorithm, PPO + Packing Transformer (trained offline).
- **Constraints:** weight, payload, floor load, orientation lock, reefer, IMDG segregation; soft scoring for CoG, stability, LIFO, stack order.
- **Interface:** FastAPI REST + WebSocket streaming for per-step placement animation.

See `docs/` for the full design document.

## Quickstart (dev)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
make test                      # run unit tests
make run                       # start FastAPI on :8009 (future phase)
```

## Project layout

```
app/
├── schemas.py              Pydantic DTOs used across layers
├── catalog/                Static container + cargo + IMDG data
├── constraints/            Hard feasibility mask + soft reward
├── env/                    Heightmap, EMS extraction, Gymnasium env
├── algorithms/             Heuristics, GA, PPO+Transformer
├── services/               Orchestration (solver, compare)
├── api/                    FastAPI routers
└── utils/                  Logging, RNG, helpers

models/                     Trained RL weights (gitignored, download separately)
notebooks/                  Training + benchmark notebooks (Colab/Kaggle)
tests/                      Pytest suites per layer
scripts/                    CLI entrypoints (benchmarks, data gen)
```

## Module ownership

- AI / RL algorithms, service code: Member 4 (AI/ML).
- REST/WS contract shared with frontend: Members 4 + 5.
- 3D asset conventions (cargo GLBs, container interior): Member 6.
