# Plan: Extend pChASE CPU for Pseudo-Hermitian (solve_pseudo) Support

This document lists what to implement and what to be careful about when enabling the distributed CPU backend (`pchase_cpu`) for the pseudo-Hermitian solver (`solve_pseudo`).

---

## 0. Distributed vector layout in pchase_cpu (background)

- **Vectors are distributed** either by **row communicator** or **column communicator**.
- **Local part:** Each process holds a **subset of rows** and **full column indices**. So locally the shape is `(local_rows, g_cols)`:
  - Rows are distributed across the corresponding communicator.
  - Columns are **not** distributed along that dimension (each process has all column indices for its rows).
- **V1, V2:** Distributed within the **column communicator** → each process has `l_rows` = (its subset of global rows), `l_cols` = `g_cols`.
- **W1, W2:** Distributed within the **row communicator** → each process has a (possibly **different**) `l_rows` and full columns.
- Because the **MPI grid can be non-square**, the local row count for (V1, V2) and the local row count for (W1, W2) can **differ**. So you cannot assume `V1_->l_rows() == W1_->l_rows()`; combining data from V1 and W1 (e.g. `W1 = beta*W1 + gamma*V1`) requires either redistribution or a kernel that respects the two layouts.

**Implications for HEMM_H2:**

- The **result** of `H * vector` is stored in the **row-communicator** layout (same as W1). So any temporary for “tmp = H*V1” must have the **same layout as W1** (row comm, same `l_rows` as W1), not the same as V1.
- The step `W1[cols] = beta*W1[cols] + gamma*V1[cols]` mixes W1 (row comm) and V1 (column comm). They have **different row partitions**, so a plain local axpy is not valid. You must either redistribute V1 into row-comm layout first, or use an existing pattern (e.g. W2 used as redistribution target) and then add into W1.

---

## 1. Current state

- **chase_cpu (serial):** Full pseudo-Hermitian support: `HEMM_H2`, `ReinitColumns`, `LanczosDos`, `isPseudoHerm`, 2×nevex subspace, QR sign flip, pseudo RR, etc.
- **pchase_cpu:** Already has:
  - `isPseudoHerm`, `GetRitzvBlockSize()` → `2*nevex_` when pseudo
  - `ReinitColumns`, `LanczosDos`
  - Pseudo-Hermitian QR (flip lower half for locked)
  - Pseudo-Hermitian RR (`pseudo_hermitian_rayleighRitz`), Resd with `subSize = (is_pseudoHerm_ ? 2*nevex_ : nevex_) - locked_`
- **Missing / to verify:**
  - **HEMM_H2** is not implemented (throws).
  - Buffer sizes and some uses of `nevex_` may assume standard (nevex) subspace.
  - Lanczos path for H² bounds is algorithm-side only; backend is still “Lanczos on H”.

---

## 2. Implement HEMM_H2 (critical)

**Interface (from `algorithm/interface.hpp`):**
```cpp
void HEMM_H2(std::size_t nev, T alpha, T beta, T gamma,
             std::size_t offset_left, std::size_t offset_right = 0);
```
Semantics: one Chebyshev step for H² on a column range:  
`Vec2[cols] = alpha*H*(H*Vec1[cols]) + beta*Vec2[cols] + gamma*Vec1[cols]`.

**Serial (chase_cpu) pattern:**
1. `ncols = (offset_right < block) ? (block - offset_right) : 0`; if 0, swap Vec1/Vec2 and return.
2. `tmp = H * Vec1[cols]` (gemm).
3. Scale/add: `Vec2[cols] = beta*Vec2[cols] + gamma*Vec1[cols]`.
4. `Vec2[cols] += alpha * H * tmp` (gemm).
5. `Vec1_.swap(Vec2_)`.

**What to do in pchase_cpu:**

- Reuse the existing distributed HEMM pattern: column range `offset_left + locked_` and `ncols`; use the same next-op swap (V1_ ↔ W1_) so the rest of the algorithm still sees the correct “current” and “filtered” buffers.
- Two matvecs:
  1. **First matvec:** `tmp = H * V1_[cols]`.  
     You need a temporary multi-vector with the same **global** layout as V1_ (same row/column distribution). Options:
     - **Option A:** Use **W2_** as temporary (same type as W1_). Ensure W2_ has at least as many columns as V1_ and that no other step assumes W2_ is unchanged after HEMM_H2. Then:  
       `MatrixMultiplyMultiVectors(1, *Hmat_, *V1_, 0, *W2_, offset_left + locked_, ncols)`.
     - **Option B:** Add a dedicated member (e.g. `H2_tmp_` or a clone of W1_) used only in HEMM_H2, sized for at least `ncols` columns in the same distribution as V1_/W1_.
- 2. **Scale/add:** On the **local** portion of W1_ corresponding to `[offset_left + locked_, offset_left + locked_ + ncols)`, compute `W1_[cols] = beta*W1_[cols] + gamma*V1_[cols]` (local axpy/scal per column, respecting `l_ld` and column mapping).
- 3. **Second matvec:** `W1_[cols] += alpha * H * tmp` (tmp = W2_ or H2_tmp_).  
   Call `MatrixMultiplyMultiVectors(&alpha, *Hmat_, *W2_, &one, *W1_, offset_left + locked_, ncols)` (or with H2_tmp_ instead of W2_).
- 4. **Swap:** Toggle `next_` (e.g. set so that the next HEMM would use the same convention as serial: “current” and “filtered” are swapped). If serial does `Vec1_.swap(Vec2_)`, then effectively “result” is in the buffer that will be read as “current” next; in pchase you achieve that by `next_ = (next_ == NextOp::bAc) ? NextOp::cAb : NextOp::bAc` so that the next HEMM reads from the buffer that now holds the H²-filtered vectors.

**Careful:**

- **Column distribution:** `offset_left` and `ncols` are **global** column indices. `MatrixMultiplyMultiVectors(..., offset, subSize)` must receive the same global range. Check that the MPI kernel uses `offset` and `subSize` as global (e.g. only the process that owns the corresponding columns does the local gemv, or the kernel does the right gather/communicate). Match the semantics of the existing HEMM.
- **Temp buffer ownership:** If using W2_, ensure no other code path between HEMM_H2 and the next HEMM/QR/RR assumes W2_ is unmodified. If in doubt, use a dedicated H2 temp buffer.
- **Mixed precision:** If pchase_cpu uses single precision in HEMM when residuals are large, decide whether HEMM_H2 should use the same policy and reuse the same single-precision matvec path for both H*V and H*tmp.

---

## 3. Buffer and scalar sizes (2*nevex vs nevex)

**Ritz values and residuals:**

- Algorithm expects `GetRitzvBlockSize()` = `2*nevex` for pseudo, so the **user** (or wrapper) allocates `ritzv` and `resid` of size `2*nevex`.
- In pchase_cpu, `ritzv_` and `resid_` are `RedundantMatrix(nevex_, 1, ...)`. For pseudo they should be **2*nevex_** so that `ritzv_->l_data()` and `resid_->l_data()` have 2*nevex entries and indexing in RR/Resd/algorithm does not overflow.
- **Action:** In the constructor, when `is_pseudoHerm_` (or when `MatrixType::hermitian_type` is PseudoHermitian), allocate:
  - `ritzv_` with rows = `2*nevex_` (and lead 2*nevex_ if needed),
  - `resid_` with rows = `2*nevex_`.
- Ensure any code that sizes or loops over `ritzv_->l_data()` / `resid_->l_data()` uses the actual size (e.g. `GetRitzvBlockSize()` or a member like `block_size_ = is_pseudoHerm_ ? 2*nevex_ : nevex_`) instead of hard-coded `nevex_`.

**Workspace matrix A_:**

- For pseudo, serial chase_cpu uses `A_(3*2*nevex_, 2*nevex_)`; pchase_cpu currently uses `A_(nevex_, 3*nevex_)` for pseudo.
- **Action:** Check what `pseudo_hermitian_rayleighRitz` (and any cholQR / RR helper) expects for the workspace dimensions. Align A_ dimensions with that (and with serial if possible). If the MPI RR uses a different layout, document it and keep A_ consistent with the MPI routine.

---

## 4. QR and column counts

- After QR, the code copies “unconverged” columns from V1 to V2. It currently uses `nevex_ - locked_` in the lacpy.
- For pseudo, the unconverged count is `2*nevex_ - locked_`.
- **Action:** Replace with a variable such as `unconverged_cols = (is_pseudoHerm_ ? 2*nevex_ : nevex_) - locked_` (or `V1_->g_cols() - locked_` if V1_ is always created with the correct global column count) and use it in:
  - The lacpy that copies the unconverged block after QR (and any similar copy that assumes “rest of columns”).
- Also check any `nevex_ - locked_` in the QR path (e.g. condition number, cholQR) and use the same unconverged count for pseudo.

---

## 5. Lanczos and H² bounds

- `solve_pseudo` uses `lanczos_for_H2` in the algorithm layer, which calls the backend’s `Lanczos(M, numvec, ...)` on **H** (not H²), then computes H² bounds (μ_1, μ_nevnex, b_sup) and fills `ritzv_` with squared values and DOS quantile.
- So the backend only needs the standard Lanczos interface. For pseudo-Hermitian matrix type, the existing `lanczos_dispatch` already calls `pseudo_hermitian_lanczos`.
- **Action:** Confirm that:
  - The multi-vector passed to Lanczos has **2*nevex** columns when running solve_pseudo (driver/ChaseManager must allocate based on `GetRitzvBlockSize()` or equivalent).
  - `pseudo_hermitian_lanczos` returns Ritz values of **H**; the algorithm then squares them. No backend change for “Lanczos on H²” is required.

---

## 6. Swap, Lock, and column indices

- The algorithm swaps and locks columns in the range `[0, 2*nevex)` for pseudo. `Swap(i,j)` and `Lock(new_converged)` use global column indices.
- **Action:** Ensure `V1_->swap_ij(i,j)` (and V2_ if used) support full range including indices in `[nevex_, 2*nevex_)` when pseudo. Same for `Lock`: `locked_` is incremented by the algorithm; no change needed if the backend only uses `locked_` as an offset. Verify that all uses of `locked_` (HEMM, HEMM_H2, QR, RR, Resd, ReinitColumns) interpret it as “number of locked columns” and that the next active column range is always `[locked_, g_cols())` with `g_cols() == 2*nevex` for pseudo.

---

## 7. ReinitColumns and LanczosDos

- Already implemented; both use `fixednev + col_indices[c]` and local indexing. For pseudo, `fixednev` can be up to 2*nevex and column indices are in the unconverged block.
- **Action:** Only verify that when the problem is pseudo, the multi-vectors actually have 2*nevex columns so that no out-of-bound access occurs.

---

## 8. Testing and validation

- Add (or extend) a **distributed** test that:
  - Builds a pseudo-Hermitian matrix (e.g. BSE-style).
  - Calls the solver path that uses `solve_pseudo` (e.g. via the same high-level API as the serial pseudo BSE test).
  - Checks that the first `nev` eigenvalues are the requested positive ones and that residuals (e.g. ‖H*v − λ*v‖) are below tolerance.
- Compare (optionally) a few eigenvalues and residuals with the serial `chase_cpu` pseudo run on the same matrix and parameters.
- Run with a small number of MPI ranks (e.g. 2, 4) to ensure correct behavior with distributed data.

---

## 9. Summary checklist

| Item | Action | Careful about |
|------|--------|----------------|
| **HEMM_H2** | Implement two matvecs + scale/add + swap | Temp buffer (W2_ or dedicated), global column range, next_ swap, mixed precision |
| **ritzv_ / resid_ size** | Allocate 2*nevex_ for pseudo | Constructor and any code assuming nevex_ entries |
| **A_ dimensions** | Match pseudo_hermitian_rayleighRitz and serial if possible | Workspace layout in MPI RR |
| **QR lacpy / unconverged count** | Use 2*nevex_ - locked_ (or g_cols() - locked_) for pseudo | All places that assume “rest = nevex_ - locked_” |
| **Lanczos** | No backend change; ensure 2*nevex columns for pseudo | Driver allocates V with correct size |
| **Swap / Lock** | Support columns in [0, 2*nevex) | Global index interpretation in swap_ij |
| **ReinitColumns / LanczosDos** | Verify with 2*nevex columns | Bounds on col_indices and j |
| **Tests** | Distributed pseudo test | Same matrix as serial; residual check |

---

## 10. Optional later improvements

- **Performance:** Reuse a single allocated H2 temp buffer across HEMM_H2 calls instead of allocating per call.
- **Documentation:** Document in the code that for pseudo-Hermitian, the subspace size is 2*nevex and that HEMM_H2 implements one Chebyshev step for H² on the unconverged column range.
- **pchase_gpu:** The same HEMM_H2 and size/column checks will be needed there for pseudo support; this plan can be used as a template.

This plan should be enough to extend pchase_cpu for solve_pseudo support in a careful and consistent way.
