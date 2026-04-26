# Datasets

Provenance, licensing and re-generation instructions for every dataset shipped with the
loading-service.

---

## 1. Brunel OR-Library — BR1 … BR10 (Bischoff & Ratcliff)

- **What:** 1,500 industrial 3-D bin-packing problems, 100 boxes each, with allowed-orientation
  flags per axis. The de-facto academic benchmark for container loading; used in the
  DeepPack3D paper and most modern 3D-BPP literature.
- **Source:** Brunel University OR-Library — http://people.brunel.ac.uk/~mastjjb/jeb/orlib/thpackinfo.html
- **Files:** `br/br1.txt` … `br/br10.txt` (~600 KB total).
- **License:** OR-Library is a freely-redistributable academic resource. Cite:

  > Bischoff, E.E. & Ratcliff, M.S.W. (1995). *Issues in the development of approaches to
  > container loading.* OMEGA 23(4):377-390.

- **File format** (per problem):

  ```
  <problem_id> <seed_id>
  <container_length_cm> <container_width_cm> <container_height_cm>
  <n_box_types>
  <type_id> <l_cm> <vert_l> <w_cm> <vert_w> <h_cm> <vert_h> <quantity>
  ...
  ```

  `vert_*` is 1 if the corresponding side may be placed vertically, 0 otherwise.
  Container dimensions in BR1-7 are 587 × 233 × 220 cm — very close to a real 20GP.

---

## 2. Wadaboa "products" — 1,000,000 real e-commerce package records

- **What:** 1 M-row table of (width, depth, height, weight) drawn from real shipments.
  Used by the Q4RealBPP-DataGen package generator that backs the IEEE / arXiv 2304.14712
  benchmark.
- **Source:** https://github.com/Wadaboa/3d-bpp (released under MIT). Original pickle
  re-saved as **parquet** here for safety + speed.
- **File:** `raw/wadaboa_products.parquet` (~8 MB compressed). Columns:

  | column | unit | range |
  |---|---|---|
  | `width`  | mm | 89 – 1500 (after outlier clamp) |
  | `depth`  | mm | 50 – 1500 |
  | `height` | mm | 1  – 1500 |
  | `weight` | kg | 2  – 1500 |
  | `volume` | mm³ | – |

- **License:** MIT (per the upstream repo). Cite the upstream:

  > Romero & Marrara (2023). *Hybrid approach for solving real-world bin packing problem
  > instances using quantum annealers.* Sci Reports 13:11777.
  > arXiv:2304.14712, dataset https://doi.org/10.17632/y258s6d939

- **Outliers removed:** rows with any side > 1.5 m or weight > 1.5 t (≈ 0.4 % of the file).

---

## 3. Alexandria-Port commodity mix (`alexandria_cargo_mix.json`)

- **What:** Approximate proportional weighting of containerized cargo categories handled at
  Alexandria Port. Used by the sampler to produce realistic voyages.
- **Sources used to derive percentages:**
  - Mordor Intelligence — *Egypt Freight & Logistics Market Report* (2024-25).
  - CAPMAS / Daily News Egypt — *Egypt's foreign trade hits $140.6 bn in 2024.*
  - Alexandria Container & Cargo Handling Co — *Statistics page*.
  - ENSCT 2024 cargo throughput summary.
- **License:** Aggregated public statistics; categories and percentages are estimates and
  should be tuned with port-authority figures when available.

---

## Re-generating everything

```bash
python -m scripts.prepare_datasets
```

The script downloads BR files into `data/br/`, parses them to JSON, and converts the
Wadaboa pickle to parquet. Run from the service root.
