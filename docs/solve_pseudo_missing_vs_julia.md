# solve_pseudo: Status vs Julia Reference

Compared to the Julia reference (`ChebyshevFilterA2!`, `QuasiHermitianRayleighRitz_v3`, deflation, locking, bounds, outlier purge). Current status of the C++ `solve_pseudo` implementation.

---

## Implemented (aligned with Julia)

| Item | Status |
|------|--------|
| Lanczos for H², bounds (μ_1, μ_nevnex, b_sup) | Done |
| Filter on H² (filter_H2) | Done |
| QR after filter (with locked cols preserved) | Done |
| Rayleigh–Ritz (pseudo-Hermitian, rayleighRitz_v2) | Done |
| Residuals (Resd) for full 2·nevex block | Done |
| Index for locking: neg first (array order), pos reverse (Julia order) | Done |
| locking_pseudo + Lock(), cap at nev per sign | Done |
| Pointer shift after locking (ritzv, resid, degrees, etc.) | Done |
| calc_degrees_pseudo_H2 (adaptive degrees, ± pairing) | Done |
| Bounds update: new_mu_nevex index (nevex − n_found_pos)*0.95, upperb_nevnex | Done |
| Outlier purge: reinit ± pairs with \|λ\| ratio > 2 in boundary | Done |
| Final reorder: first nev = positive locked (sorted ascending) + Swap | Done |
| early_locked_residuals + set_early_locked_residuals | Done |
| Stop when n_found_pos ≥ nev | Done |

---

## Still missing or different vs Julia

### 1. **QR with S-inner product for deflation (Julia: S_product!)**

- **Julia:** `YV_qr = hcat(S_product!(Y_locked[:, 1:n_found]), V_qr)` then `CholeskyQR2!(YV_qr)`. Locked vectors are transformed by the pseudo-Hermitian metric **S** before being included in the QR.
- **C++:** QR is standard Euclidean (CholeskyQR / Householder) on `[locked cols | unconverged cols]`; locked columns are copied but not multiplied by S.
- **Impact:** For strict equivalence with Julia’s deflation in the S-metric, the C++ QR step for pseudo-Hermitian would need to use (or expose) an S-weighted Gram matrix when orthogonalizing against locked vectors. Currently it does not.

### 2. **lowerb / lambda_1 update**

- **Julia:** `lowerb` in `filter_state` is **not** updated from Ritz values; only `upperb_nevnex` is updated from `new_mu_nevex` when smaller.
- **C++:** We also update `lambda_1` from the Ritz pair with smallest residual (μ_1-style update). So we do one extra bounds update that Julia does not.

### 3. **Rayleigh–Ritz version (v2 vs v3)**

- **Julia:** `QuasiHermitianRayleighRitz_v3`.
- **C++:** `rayleighRitz_v2` (and `pseudo_hermitian_rayleighRitz_v2` in distributed).
- **Impact:** Possible differences in ordering, scaling, or handling of the ± pairs. Aligning with a “v3” implementation (if available) would match Julia exactly.

### 4. **CHASE_SAVE_RESIDUALS (residual file)**

- **solve():** When `CHASE_SAVE_RESIDUALS` is set, residual history is written to a file.
- **solve_pseudo:** No residual-file writing is wired; the same hook could be added for parity.

### 5. **PURGE_RATIO_THRESHOLD configurable**

- **Julia:** `PURGE_RATIO_THRESHOLD = 2.0` (constant in script).
- **C++:** Hardcoded `2.0` in algorithm.inc. Could be exposed via config for consistency.

### 6. **Optional: early-exit / “break” in locking loops**

- **Julia:** Can break out of the negative (or positive) locking loop once enough are locked; the code structure makes that explicit.
- **C++:** We achieve the same effect via the nev-per-sign cap and `n_skip_tail`; no functional gap, only a structural difference.

---

## Summary checklist (current)

| Item | Status |
|------|--------|
| Lanczos for H², bounds | Done |
| Filter on H² | Done |
| QR after filter | Done (Euclidean; no S-metric) |
| Rayleigh–Ritz (pseudo) | Done (v2) |
| Residuals | Done |
| Index (neg first, pos reverse) | Done |
| locking_pseudo + Lock, nev cap | Done |
| Pointer shift after locking | Done |
| calc_degrees_pseudo_H2 | Done |
| Bounds update (new_mu_nevex, upperb_nevnex) | Done |
| Outlier purge | Done |
| Final sort (positive first) + Swap | Done |
| early_locked_residuals | Done |
| **QR with S_product (deflation)** | **Missing** |
| **RR v3 (if desired)** | **v2 in C++** |
| **lowerb not updated from Ritz** | **We do update lambda_1** |
| CHASE_SAVE_RESIDUALS in solve_pseudo | Missing (optional) |
| Configurable purge threshold | Optional |

The main conceptual gap with the Julia reference is **QR with the S-inner product for deflation**; the rest is either aligned or minor (bounds update detail, RR version, residual file, purge threshold).
