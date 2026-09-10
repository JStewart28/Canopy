# An abstract far-field backend for Canopy

**Status:** IN PROGRESS

## Problem

Canopy's far field is welded to the solid-harmonic expansion of $1/r$. A
downstream solver needs a far field for a **Plummer-softened** kernel,

$$
\varphi(r) = \left(r^2 + b\right)^{-1/2}, \qquad
K_l(\delta) = -\partial_l \varphi = \frac{\delta_l}{(\delta^2 + b)^{3/2}},
\qquad b > 0,
$$

and no finite solid-harmonic expansion can represent it: the expansion rests on
the addition theorems (Greengard Thms 5.22, 5.23, 5.26, cited in the kernel at
`src/Canopy_LaplaceKernel.hpp:265`, `:353-368` and `:676-683`), which require
harmonicity, and the only isotropic harmonic functions in 3D are
$\mathrm{const}$ and $1/r$.

Today Canopy works around this with a **near-field softening floor**: any pair
closer than `near_softening_factor * eps` is refused by the MAC and forced into
the softened direct sum (`mac_satisfied`,
`src/Canopy_CommunicationPlan.hpp:347-359`; the knob is `FmmConfig::near_softening_factor`,
`src/Canopy_Solver.hpp:78`). That keeps the answer defensible only where
$b$ is negligible, at the cost of pushing a large and configuration-dependent
fraction of pairs — measured between 5% and 97% by the downstream solver — out
of the $O(1)$-per-cell-pair M2L and into the $O(\mathrm{ncrit}^2)$-per-leaf-pair
direct sum.

Two replacement bases are candidates, and they want different things from the
sweeps:

| Basis | Coefficients | What the kernel supplies | Reference |
| --- | --- | --- | --- |
| **Solid-harmonic** (today) | $(P{+}1)(P{+}2)/2$ complex | harmonicity, as a series identity | `src/Canopy_LaplaceKernel.hpp` |
| **Cartesian-Taylor** | $\binom{p+3}{3}$ real | a derivative ladder $\partial^\alpha\varphi$ | [canopy-questions.md](canopy-questions.md) §§1-3 |
| **Black-box FMM** (Chebyshev) | $n^3$ real, several sets | $n^3\times n^3$ *evaluated numbers* | [canopy-bbFMM.md](canopy-bbFMM.md) |

**What is being built.** The abstraction that holds all three behind one solver
template parameter, with the existing solid-harmonic path ported onto it and
proven **bit-for-bit unchanged**, plus a second, non-harmonic basis carried far
enough to prove the contract is real. The two production bases are separate
work: Cartesian-Taylor is **T12**, a deliberately coarse task in this document;
black-box FMM has no task here at all and gets its own design.

**Requirements this abstraction must satisfy**, stated as the downstream solver
states them:

- **R-A** — the far field must be computable for $\varphi = (r^2+b)^{-1/2}$ with
  $b>0$, so that `near_softening_factor = 0` is a viable configuration.
- **R-B** — far-field relative error must be a *tunable truncation*, reachable
  to $10^{-10}$ per evaluation, not a fixed bias.
- **R-C** — `Solver` and `createSolver` must keep compiling verbatim for
  existing callers. There are six:
  `tests/tstMultiSolve.hpp:181`, `:745`, `:989`;
  `examples/02_full_fmm/example_full_fmm.cpp:36`;
  `examples/03_gravity_solve/gravity_solve.cpp:39`;
  `examples/04_nan_replay/nan_replay.cpp:66`.
- **R-D** — three simultaneous charge components with gradient output, as
  `NComps = 3` already provides.

**Out of scope.** Implementing either new basis beyond T12's coarse statement.
The tree builder, partitioner, MAC, dual-tree traversal, communication plan and
CSR — none of them are kernel-aware and none of them change. `Canopy_P2P.hpp`
does not change either: it never calls the kernel, and it already runs the
softened near kernel (see [Current state](#current-state)). Single-precision
support for the new bases is out of scope — see the `Scalar` row in
[Conventions](#conventions). The two `unit` test targets that do not compile are
out of scope too, and no task here adds tests to either — see
[Current state](#current-state).

## Approach

### Two axes, one solver parameter

`LaplaceKernel` bundles two independent concerns into one struct, and untangling
them is the whole design:

| Concern | Where it lives today |
| --- | --- |
| **Basis** — storage type, coefficient count, index map, symmetry, the five operators' algebra, five width normalizations | `src/Canopy_LaplaceKernel.hpp:153-159`, `:177-193`, `:226-229`, `:267-272`, `:369-372`, `:685-687`, `:795-799` |
| **Kernel** — $1/r$, entering as harmonicity and as $\rho^{-(n+j+1)}$ in the operator builder | `:265`, `:353-368`, `:600`, `:676-683` |

The two are not in bijection. The black-box basis is kernel-blind — the kernel
enters only as evaluated numbers. The Cartesian-Taylor basis needs a
kernel-specific derivative ladder. The solid-harmonic basis is welded to
harmonicity and can carry no other kernel at all.

The sweeps already take exactly one type and read everything else off it as
traits — `UpwardSweep<MemorySpace, ExecutionSpace, KernelType>`
(`src/Canopy_UpwardSweep.hpp:55`), `DownwardSweep` (`src/Canopy_DownwardSweep.hpp:100`),
`P2P`. **That slot is the right shape.** What goes into it becomes a
*basis* that names its kernel internally, so that "this basis generalizes to
other kernels" is expressible as `ChebyshevBasis<SoftPlummer, …>` versus
`ChebyshevBasis<Stokes, …>` without writing a second basis. The sweeps never see
the kernel axis.

Validity is enforced by what a basis requires of its kernel, checked at
instantiation:

| Basis | Requires of its kernel | Valid kernels |
| --- | --- | --- |
| Solid-harmonic | harmonicity, supplied as a tag rather than a callable | bare $1/r$ only |
| Cartesian-Taylor | `deriv_tensor(order, r)` — a ladder | anything with a derived ladder: $1/r$, $(r^2{+}b)^{-1/2}$ |
| Chebyshev | `evaluate(x, y)` | anything callable |

### The load-bearing decision: M2L becomes three kernel-owned stages

Eleven of the twelve sites in shared code that encode the solid-harmonic basis
are removed by a trait (see [Current state](#current-state)). The twelfth is
the fused M2L inner loop, `src/Canopy_DownwardSweep.hpp:1496-1526`, which
hardcodes the packed triangular storage index `n*(n+1)/2 + abs_m` (`:1508-1509`),
the flat source index `n*n+n+m` (`:1506`), the conjugate-symmetry expansion for
$m<0$ (`:1512-1517`), the $(n,m)$ loop bounds from `P_local` (`:1502-1504`),
complex accumulation (`:1501`, `:1518`) and a real/imag *split* scratch
accumulator chosen deliberately to halve shared-memory bank conflicts
(`:1454-1462`, `:1522-1524`). No trait removes this. The contraction is
basis-specific, and so is the scratch layout — which the sweep currently owns.

**M2L therefore becomes a three-stage, kernel-owned operation:**

1. `m2l_pre_cell` — optional, once per **source** cell;
2. `m2l_core` — the per-pair apply, reaching the operator set only through an
   integer `op_idx`;
3. `m2l_post_cell` — optional, once per **target** cell.

The sweep keeps the traversal, the CSR, the team-per-target launch, the scratch
allocation (sized by a trait) and the write-back. The operator set is **opaque
to the sweep**.

Why three stages and not one. The compressed shared-basis form that makes the
black-box basis representable at all is

$$
L^A \mathrel{+}= U_\ell \left( \sum_{\text{pairs}} C_{\rm key} \left(V_\ell^{\!\top} M^B\right) \right),
$$

and its whole flop advantage comes from computing $V_\ell^{\!\top}M^B$ **once per
source cell** and $U_\ell(\cdot)$ **once per target cell**, leaving only the
small $r\times r$ core inside the pair loop. Today's driver has nowhere to put a
per-cell pass: the loop at `:1489-1527` is strictly per-pair inside a per-target
team, and the only per-target work is the zeroing (`:1481-1487`) and the
write-back (`:1533-1542`). **The post-pass fits naturally where the write-back
already is; the pre-pass fits nowhere and is the one genuinely new structural
element.**

The same two hooks are what an FFT-accelerated M2L would need — forward
transform once per source cell, pointwise grid multiply per pair, inverse
transform once per target cell (see [canopy-kIndp.md](canopy-kIndp.md), "What
survives intact"). Two unrelated methods needing the same two hooks is the
reason to declare them now rather than discover them later.

The solid-harmonic and Cartesian-Taylor bases supply **no-op** pre/post passes,
so neither pays for this beyond the declaration.

### The bit-for-bit gate

The solid-harmonic path must come through this refactor with **identical bit
patterns**, and nothing else in the suite can detect a break: the tightest
full-pipeline assertion elsewhere is a $5\times10^{-2}$ relative bound on the
potential and $1\times10^{-1}$ on the gradient (`tests/tstMultiSolve.hpp:929-930`),
whose own comment says it exists to catch "a complete-regression bug" (`:925-928`).
The per-operator tests in `tests/tstLaplaceKernel.hpp` are the right granularity
but compare against analytic references with tolerances, not stored bytes — and
that target does not compile and is out of scope
([Current state](#current-state)), so it is not an available instrument either.

So **T1 builds this harness before anything is refactored**, and
**T3 performs the M2L move alone, against nothing else**, so that a bitwise
difference is attributable to one change. T3 is the gate for the entire
document: if the solid-harmonic M2L cannot move into the basis bit-identically,
the design falls back to the narrow abstraction of **R1**.

Bit-for-bit identity holds at one and two ranks, and **not above them**. The
M2L summation order itself is deterministic: `M2LPlan::interaction_lists` is a
`std::unordered_map` (`src/Canopy_CommunicationPlan.hpp:95-96`) whose iteration
order is not guaranteed, but `entries` is subsequently `std::sort`ed by
`(depth, target_idx)`, a total order over distinct targets, and within-entry
source order is the traversal's deterministic push order
(`src/Canopy_CommunicationPlan.hpp:481`). That argument settles the CSR given a
partition. It says nothing about the partition, which decides the CSR's *input*:
`TreePartitioner::partition_leaves` uses the Zoltan2 `multijagged` algorithm,
documented as non-deterministic at `src/Canopy_TreePartitioner.hpp:417-419`, and
computing it on rank 0 and broadcasting makes the assignment consistent within a
run but not across runs. Determinism of the sweep given a partition is not
determinism of the solve.

**The partition can also move between invocations inside a single process, and a
later task must not read that as its own doing.** At np=5 the three test bodies
run three separate solves. Those three drew the same cut through T1 and T2
(`n_unique_ops` 180/168/147/217/187) and have since been observed splitting into
two cuts within one job — one solve drawing 170 on rank 0 and 177 on rank 1
while the other two drew the earlier set. That moves the np=5 direct-sum
deviation in its 9th significant figure, `3.209361028925181e-07` against
`3.2093610310582619e-07`, still 3x under `LS_DIRECT_SUM_TOL`. It is the same
multijagged non-determinism and not a property of any change: it is reproduced by
unmodified `HEAD` (flux job `f3XWSVHd6FVh`, the control run T3 attributed it
with). **So a np 3-6 direct-sum or cross-rank deviation that moves only in its
9th or later significant figure and stays under its pinned tolerance is this.**
Confirm it the same way — stash the change, rebuild, re-run the gate from
unmodified `HEAD` — before recording it as a finding of the task. The np 1-2
bitwise half of the gate is unaffected, because the cut over one or two parts is
reproducible.

So the gate is three checks — **the Laplace-solve gate** — and every task below
is verified against all three:

| Ranks | What is asserted | Why it is the strongest thing available |
| --- | --- | --- |
| 1-2 | bitwise identity of `locals()`, the M2L operator table, the $A_{n,m}$ table and the realized key list | the multijagged cut over one or two parts is trivial and reproducible, so bit patterns are stable |
| 2-6 | the np=$k$ field reproduces the np=1 field to floating-point reassociation | the FMM answer is partition-independent as mathematics; only the summation order moves, so the bound is ~$10^{-13}$ rather than the truncation error |
| 1-6 | the field matches a direct sum at the accuracy the method delivers | the only check that the far field is the *right* field and not merely a self-consistent one |

**np ≤ 2 is sufficient for T3 specifically**, which is what keeps the document's
own gate intact. R1 — the scratch-layout regression that is the likely way the
M2L move fails — is per-target-cell arithmetic and presents at np=1. R6's shared
cells exist at np=1 already, since shared cells are the non-leaf cells at
depth ≤ `replication_depth` (`src/Canopy_CommunicationPlan.hpp:698`) and nothing
in that condition mentions the rank count. What np ≥ 3 uniquely adds is
rank-*count*-dependent indexing in the MPI packing loops, and the cross-rank
check of T1 covers that at a tolerance far tighter than any accuracy bound.

Making the partitioner reproducible run-to-run is a recorded limitation under
`README.md` "Known Issues", **not** a prerequisite for any task here. It is not
a task in this document.

### Conventions

| Choice | Value | Why |
| --- | --- | --- |
| Sweep template parameter name | stays `KernelType` | Renaming it touches nearly every line of three headers and would bury the diff that T1's harness must attribute. The concept is documented in the class comments instead. |
| New `Solver` parameter name | `FarField` | It selects a basis-plus-kernel composition, not a bare kernel. |
| Basis header naming | `src/Canopy_<Name>Basis.hpp` | Distinguishes a basis from `Canopy_LaplaceKernel.hpp`, which keeps its name (R-C: it is named in six call sites and in test fixtures). |
| Trait naming | `snake_case`, `static constexpr` or typedef | Matches `num_coeffs_per_cell` (`:157`), `m2l_num_src_coeffs` (`:488`). |
| Operator naming | `snake_case`, `KOKKOS_INLINE_FUNCTION static` | Matches `p2m_contribution`, `m2m_translate`, `l2l_translate`, `l2p_evaluate`. |
| Host-side operator construction | plain static member, **not** `KOKKOS_INLINE_FUNCTION` | `build_m2l_operators` runs once on host and may call LAPACK. Marking it device-callable would forbid that. |
| `Scalar` for new bases | `double` only, enforced by `static_assert` | A softened kernel has no scale invariance to exploit, so the new bases carry *physical* operators keyed by level; the FP32 conditioning argument that the width normalizations exist for does not transfer. R-B's $10^{-10}$ does not survive single precision regardless. The solid-harmonic basis keeps its live `float` path (`src/Canopy_DownwardSweep.hpp:301-302`; `tests/tstMultiSolve.hpp:1079`). |
| Failure on unsatisfiable contract | `static_assert` at instantiation | A basis asking for a capability the sweeps do not have must not compile. Never a runtime fallback that silently produces a different answer. |
| Failure on operator-table overflow | keep today's loud path | One `fprintf` warning (`:1038-1046`) plus routing to the per-pair fallback. Extended by T8 with a per-basis policy. |
| Test tier for new tests | `unit` | The `regression` tier is the ship gate and holds only `MultiSolve` (`tests/CMakeLists.txt:60-62`), and no task here runs it — see [Deliberate deviations](#deliberate-deviations). Promoting anything into it requires confirming with the user first, per the repository's own rule. |
| New test registration | add the name to `UNIT_MPI_TESTS` (`tests/CMakeLists.txt:47-55`) or `UNIT_SERIAL_TESTS` (`:35-38`) | Target becomes `Canopy_Test_<Name>_MPI_<DEVICE>`, tests `..._np_<N>` for `N` in 1-6. |
| Reference data | one committed file, `tests/data/laplace_solve_P6.txt` | Three bit-for-bit records — `(nprocs, rank)` of `(1,0)`, `(2,0)`, `(2,1)` — plus one np=1 field record in canonical `GlobalId` order and a hash of the initial global particle set. The particle set is a fixed global set of $N_{\rm total} = 600$ from seed `1234 + P`, identical at every rank count, which is what makes the np=$k$-versus-np=1 comparison definable at all. The committed drift check hashes that *initial* set and not the state the field record is taken at: the solve drives the particles, so their positions at the last step are not bit-identical across rank counts. |
| Test naming | `tests/tstLaplaceSolve.hpp` for the solve-level gate, `tests/tstLaplaceKernel.hpp` for the per-operator kernel tests | `Canopy_add_tests` maps a name to `tst<NAME>.hpp` and to `Canopy_Test_<NAME>_MPI_<DEVICE>` (`cmake/test_harness/test_harness.cmake:104-112`), so the file name, the `tests/CMakeLists.txt` entry and every exit criterion below move together. |
| Bit-for-bit artifact form | 64-bit hash plus extents for `locals()` and the operator table; full bit patterns for the $A_{n,m}$ table and `n_unique_ops` | The operator table costs $N_t N_s \cdot 16 = 28\cdot49\cdot16$, 21.4 KB per key at $P=6$; committing it in full is not affordable. A hash over the raw bytes is exactly as sensitive to a bitwise change and still attributes a failure to one artifact, which is all any exit criterion here asks. Hashes are computed with the in-repo FNV-1a so a committed value depends on no library version. |
| Provenance comments | required on any operator derived from a paper, spec or reference implementation | Name the source and the exact theorem/section on the routine, as `:265` and `:676` already do. |
| Units and conventions on declarations | required | Every width parameter states whether it is a half-width or a full width; every offset states its sign convention (`source − target` or the reverse); every operator states which normalized quantity it consumes and produces. These are not recoverable from the code. |

### Deliberate deviations

- **No task here runs the `regression` suite.** `CLAUDE.md` "Minimum test set"
  makes it the required gate before a code change ships; every exit criterion
  below names the Laplace-solve gate instead, and none names `regression`. The
  suite does not pass on unmodified code, it hangs intermittently, and the
  Laplace-solve gate is the sharper and far cheaper instrument — measurements
  and the failing set are in [Current state](#current-state). This scopes to
  this document's tasks and leaves the repository's ship gate as it is.
- **The operator-count cap is retained alongside the byte budget.** T8 makes the
  cap a memory budget, but keeps `M2L_OP_COUNT_CAP` (`:304`) as a floor:
  `effective_cap = min(M2L_OP_COUNT_CAP, byte_budget / bytes_per_key)`. A pure
  byte budget would change *which* pairs overflow into the per-pair fallback path
  — which is different arithmetic — and so would break the bit-for-bit
  requirement for reasons unrelated to the abstraction. With a 2 GB default
  budget and 58 KB per key at $P=8$, the count cap binds first and today's
  overflow set is provably unchanged.
- **`m2l_apply_operator` (`src/Canopy_LaplaceKernel.hpp:639-672`) is repurposed,
  not deleted.** It has no callers, but it is the operator-apply interface the
  fused kernel should have been going through and is roughly 80% of the
  solid-harmonic `m2l_core`. T3 grows it into `m2l_core` rather than deleting it
  and rewriting the same contraction.
- **`get_coeff_3d` (`:177-193`) and `Canopy_SphericalCoefficients.hpp:70-91`
  stay.** Both look like generic accessors that a generalization would remove.
  `get_coeff_3d` has six live callers inside the kernel (`:338`, `:464`, `:662`,
  `:762`, `:828`, `:840`) and becomes private implementation of the
  solid-harmonic basis. `get_coeff` is exercised by
  `tests/tstLaplaceKernel.hpp:441-470`.
- **The conformance basis lives in `tests/`, not `src/`.** It is a fixture that
  proves the contract, not a method anyone should solve with. Shipping it in
  `src/` would invite exactly that.

## Current state

Three things in this document are built. **T1 is complete** — the diagnostic
surface on the sweeps, the solve-level gate `tests/tstLaplaceSolve.hpp`, its
committed reference data `tests/data/laplace_solve_P6.txt`, and both pinned
tolerances (`LS_CROSS_RANK_TOL = 5.6e-10`, `LS_DIRECT_SUM_TOL = 9.63e-07`) all
exist and pass at ranks 1-6. **T2 is complete** — the five dead scaffolding
members and `has_mplus_symmetry` are deleted. **T3 is complete** — M2L is three
kernel-owned stages, `m2l_pre_cell` / `m2l_core` / `m2l_post_cell` over a
basis-sized raw-byte scratch, and the solid-harmonic path came through
bit-identical, so **R1 did not fire** and the narrow-abstraction fallback is not
needed. Nothing else here has been built. What follows is what is true of the
repository now.

**Line numbers in this document no longer track either sweep header**, and the
offset is not a single figure. Most citations here were written against the
pre-T1 numbering (commit `a6c90de`); T1 inserted 55 lines into
`Canopy_DownwardSweep.hpp`, T2 deleted 100 from it and 40 from
`Canopy_UpwardSweep.hpp`, and T3 rewrote `run_m2l_fused` and shifted everything
after it again. Measured net offsets, to be *added* to a cited line to reach the
current one:

- `src/Canopy_DownwardSweep.hpp`: `0` up to `:109`, `-1` through `:305`, `-3`
  through `:336`, `+8` through `:445`, `+52` through `:470`, `+40` through
  `:621`, `+39` through `:1071`, `+44` through `:1081`, `+43` through `:1418`,
  `+19` from `:1543` through `:1903`, `-65` through `:2123`, and `-66` beyond.
  **`run_m2l_fused` — pre-T1 `:1419-:1542` — was rewritten by T3 and no offset
  maps into it**; locate anything there by symbol. Spot-checks that hold:
  pre-T1 `:406` is `:414`, `:1725` is `:1744`, `:1782` is `:1801`, `:2073` is
  `:2008`.
- `src/Canopy_UpwardSweep.hpp`: T1 and T3 changed nothing here, so these
  citations were exact before T2. After it: `0` up to `:207`, `-5` through
  `:416`, and **`-40` from `:417` on**.
- `src/Canopy_MpiCoalescedExchange.hpp`: no task has touched this file, so every
  citation to it is exact.

T1's and T2's own citations were current when written, which is the post-T1
numbering; for those, only T2's deletions apply — `-1` past `:110`, `-3` past
`:307`, `-15` past `:532`, `-99` past `:2042` and `-100` past `:2180` in
`Canopy_DownwardSweep.hpp`, and `-5` past `:212`, `-40` past `:449` in
`Canopy_UpwardSweep.hpp`. Individual citations elsewhere in this document have
deliberately **not** been renumbered. Where a citation and the code disagree,
the code is authoritative — search for the named symbol rather than trusting
the number.

**Twelve sites in shared code encode the solid-harmonic basis.** Grouped by what
fixes each:

*(a) Five are dead. Deleting them removes the leak for free.* **Done in T2** —
all five, plus `has_mplus_symmetry`, are deleted; the table below is the
historical record of what was removed and its line numbers are pre-deletion.
Verified by repository-wide search over `src/`, `tests/` and `examples/`:

| Site | What it encodes | Callers |
| --- | --- | --- |
| `UpwardSweep::apply_p2m_normalization_bridge` (`src/Canopy_UpwardSweep.hpp:212`, `:418-449`) | $w^{n+1}$ scaling, `n*(n+1)/2+m`, `.real()/.imag()` | none |
| `DownwardSweep::apply_l2p_normalization_bridge` (`src/Canopy_DownwardSweep.hpp:524`, `:2006-2041`) | $w^n$, same index map | none |
| `DownwardSweep::scale_locals_at_depth` (`:531`, `:1959-2004`) | $w^{n\cdot\mathrm{sign}}$, same | none |
| `DownwardSweep::M2L_NUM_SRC` (`:306`) | duplicates `LaplaceKernel::m2l_num_src_coeffs` (`:488`) | none |
| `DownwardSweep::P` (`:110`) | expansion order | only `M2L_NUM_SRC`, itself dead |

`LaplaceKernel::has_mplus_symmetry` (`:159`) is also dead and is deleted with
them. `execute()`'s own comment already records that the bridges are obsolete —
"after step 5 every multipole/local in the pipeline is in scale-normalized form,
so no bridges are needed" (`:2176-2179`).

*(b) Five in shared code go to a `coeff_type` + `scalars_per_coeff` trait pair:*

| Site | Today |
| --- | --- |
| `src/Canopy_UpwardSweep.hpp:72-73` | `View<complex_type***, LayoutRight>` |
| `src/Canopy_DownwardSweep.hpp:117-118` | same, for locals |
| `src/Canopy_UpwardSweep.hpp:664`, `src/Canopy_DownwardSweep.hpp:2073` | `deep_copy(view, complex_type(0,0))` |
| `src/Canopy_DownwardSweep.hpp:406`, `:1725-1726` | `std::vector<complex_type>` snapshot, zero-filled |
| `src/Canopy_DownwardSweep.hpp:253-254`, `src/Canopy_UpwardSweep.hpp:179-180` | `CoalescedExchangeBuffers<complex_type, …>` |

The snapshot arithmetic (`:1778` subtract, `:1800` add) works unchanged for a
real type — `operator-` and `operator+` exist for both.

**The M2L operator table's element type is already basis-owned** and is not one
of the five. `src/Canopy_DownwardSweep.hpp:339-340` holds only
`using m2l_operators_type = typename KernelType::template m2l_operators_type<memory_space>;`,
with `_m2l_op_table` declared from it at `:344`; the literal
`Kokkos::View<complex_type***, Kokkos::LayoutLeft, MemorySpace>` lives in the
basis at `src/Canopy_LaplaceKernel.hpp:178-180`. Generalizing the element type
there is an edit to the basis alias, not to a sweep typedef, and it must move
together with `coeff_type` or the two can silently disagree — the sweep's
`m2l_op_table_view_type` (`src/Canopy_DownwardSweep.hpp:466`) is an alias of
`m2l_operators_type`, and `tests/tstLaplaceSolve.hpp:689-691` `static_assert`s
`LayoutLeft` through it.

*(c) Three assume complex arithmetic **structurally**, not by typedef. All three
are MPI packing:*

| Site | The structural assumption |
| --- | --- |
| `src/Canopy_MpiCoalescedExchange.hpp:72` | `using scalar_type = typename complex_type::value_type;` — a real `double` coefficient has no `::value_type`, so this function template does not compile. A hard compile failure, not a silent 2× waste. |
| `src/Canopy_MpiCoalescedExchange.hpp:96` | `per_cell_real = 2 * per_cell_complex` |
| `src/Canopy_UpwardSweep.hpp:534-535`, `:581-584`; `src/Canopy_DownwardSweep.hpp:1782-1788` | `reinterpret_cast<scalar_type*>(buf)` with count `2 * total_complex`, and `MPI_DOUBLE`/`MPI_FLOAT` chosen from `sizeof(scalar_type)` |

`coalesced_view_exchange` is **already** shape-generic in the other two extents —
it reads `view.extent(1)` and `view.extent(2)` at `:93-94` rather than
compile-time constants — so the later `sets_per_component` change costs nothing
there.

*(d) Two need a trait to supply a value shared code derives from harmonic
reasoning:*

- `src/Canopy_UpwardSweep.hpp:233-235` builds the $A_{n,m}$ table to $2P$ because
  "M2L accesses A at degree n+j where both n and j go up to P". Shared code owns
  a table whose *existence* is basis-specific. `DownwardSweep` borrows it
  (`:529`) and threads it into three operators (`:1099-1100`, `:1636-1639`,
  `:1688-1691`), plus `src/Canopy_UpwardSweep.hpp:501-504`.
- `src/Canopy_DownwardSweep.hpp:301-302` branches `M2L_KEY_DD_MAX` on
  `KernelType::scalar_type` being `float`, with a rationale (`:295-300`) derived
  entirely from the solid-harmonic scale normalization. For a non-homogeneous
  kernel that rescaling does not exist and the constant is meaningless.

*(e) One is not fixable by a trait and must move:* the fused M2L inner loop,
`src/Canopy_DownwardSweep.hpp:1496-1526`, described in
[Approach](#the-load-bearing-decision-m2l-becomes-three-kernel-owned-stages).

*(f) Two sites outside the sweeps, both bounded:*

- **Softening never reaches the far field.** `FmmConfig::softening`
  (`src/Canopy_Solver.hpp:69`) is routed to `_p2p.set_softening` and
  `_comm_plan.set_near_softening` only — in the constructor (`:169-178`) and in
  `_init_auto_softening` (`:680-717`). Nothing passes it to `_upward` or
  `_downward`, and `m2l_build_operator` is a **static** method taking
  `(dd, ix, iy, iz, A_table, T_out)` (`src/Canopy_LaplaceKernel.hpp:516-519`) —
  integers only. A softened basis needs $b$ *and* the physical unit width. T9
  supplies both.
- **P2P never calls the kernel.** It uses only `scalar_type` and
  `num_components` (`src/Canopy_P2P.hpp:70-72`) and inlines
  $1/\sqrt{r^2+\varepsilon^2}$ and $-q\,\delta/(r^2+\varepsilon^2)^{3/2}$
  directly (`:880-899`, and again in the inter-leaf kernel near `:1082`). For all
  three bases here this is **not** a leak — all three share that same softened
  near kernel and it is already the right one. It *is* a leak for the claim "this
  basis generalizes to other kernels": a fourth kernel would need P2P changed.
  Named and bounded; not done here.

**The M2L operator table is reached from five sites**, and only one is in device
code:

| Site | What it does |
| --- | --- |
| `src/Canopy_DownwardSweep.hpp:341-342` | declares `View<complex_type***, LayoutLeft>` |
| `:622-623` | resets it to a default-constructed view in `setup()` |
| `:1079-1109` | builds it, `(Nt, Ns, n_unique_ops)`, on host, serially, one `m2l_build_operator` call per key (`:1094-1101`), then one `deep_copy` (`:1106`) |
| `:1449` | captures it by value for the device lambda |
| `:1518` | indexes it — `op_table(out_idx, j, op_idx)` |

Only `:1518` is shape-committing. `LayoutLeft` is deliberate: a
`subview(_m2l_op_table, ALL, ALL, op_idx)` is a contiguous column-major
$(N_t, N_s)$ matrix consumable by a BLAS `gemm` without transpose (`:337-340`).

**The artifacts a bit-for-bit gate needs are reachable.** Alongside `locals()`,
`interaction_list_build_count()`, `total_fallback_pair_count()` and
`total_m2l_pair_count()`, `DownwardSweep` exposes `m2l_op_table()`,
`m2l_realized_keys()`, `m2l_n_unique_ops()` and `A_table()`, plus the public
aliases `m2l_op_table_view_type` and `m2l_key_type`. The realized key list is
retained in a private `std::vector<M2LKey> _m2l_realized_keys`, copied from the
function-local `ops` in `build_interaction_list_device` immediately after
`n_unique_ops` is computed; `ops` is copied rather than moved because the
table build still consumes it. The accessor is `m2l_n_unique_ops()` rather than
`n_unique_ops()` so it cannot be shadowed by that function-local. All of it is
additive and read-only and changes no arithmetic. `A_table()` was chosen over
adding `Solver::upward()`: `_A_table` is borrowed from the upward sweep in
`setup()`, so it is the same view, and the whole diagnostic surface stays on one
class. This surface is a diagnostic, not part of the supported runtime API —
the qualification already carried on `downward()` (`src/Canopy_Solver.hpp:477-479`)
applies to all of it. T7 and T9 read these.

**One test can locate a data file.** `Canopy_add_tests`
(`cmake/test_harness/test_harness.cmake:104-112`) sets no `WORKING_DIRECTORY` on
any `add_test` and defines no data directory, so the definition is applied per
generated target from `tests/CMakeLists.txt:79-82` instead. Only the solve-level
Laplace test reads a file; no other test under `tests/` opens one.

**The `regression` suite is not the verification instrument for any task here.**
Correctness for every task in this document is gated on the Laplace-solve gate
alone, `ctest -R Canopy_Test_LaplaceSolve_MPI_SERIAL`. This is a deliberate
deviation from the repository's ship gate in `CLAUDE.md` "Minimum test set",
scoped to this document's tasks; it does not change what gates a release. Three
things force it. The `regression` suite does not pass on unmodified code, so a
green run was never an available criterion. It hangs intermittently at np=3. And
the Laplace-solve gate is both sharper and roughly 60x cheaper in wall — 41 s,
against a run that needs more than 15 minutes when the hang fires.

**The measured `regression` baseline**, recorded here so no later task
re-derives it. At commit `64d1648` (flux job `f3XUAWAy6WFR`),
`ctest --output-on-failure -L regression -R MPI_SERIAL` fails six `MultiSolve`
tests at **all six** rank counts — `StableTree_Migrate`,
`IntermediateMotion_Rebalance`, `LargeMotion_Rebuild`, `AutoMaintain`,
`AutoRebalance`, `M2L_BinEdge_Fallback` — on a multi-step position/velocity
comparison at `fmm_tolerance = 1e-8` (`tests/tstMultiSolve.hpp:542,546`), with
measured relative errors of 3e-7 to 9e-6. A **seventh** test fails alongside
them: `SolveFusedM2L.FP32_smokeTest`, at np 2-6, passing only at np=1
(`max_grad_rel` 0.277 at np=2 rising to 0.339 at np=3 against a 5e-2 budget).
All seven predate every task here: np=1 reproduces the same digits
(`3.485035469067542e-07`, `6.8419528791564039e-07`, `9.1947965989306709e-06`)
that checking out the pre-T1 `src/Canopy_DownwardSweep.hpp` (commit `a6c90de`),
rebuilding and rerunning produces. **The np=3 hang is intermittent**, which is
what makes the suite unusable as a gate rather than merely slow: this baseline
ran all six rank counts in 62 s with no hang at all, while a rerun of the same
script over the same binary path (flux job `f3XUPJqSuqdh`) hung at np=3 for over
ten minutes and was cancelled. All of it is recorded under `README.md` "Known
Issues", and fixing any of it is not a prerequisite for any task here.

**Two `unit` test targets do not compile, and both are out of scope.** At commit
`64d1648` a `make -j 4 -k` over the whole tree fails exactly two targets, at
every backend:

- `Canopy_Test_LaplaceKernel_*` — 35 errors, "no matching function" for
  `p2m_contribution`, `m2m_translate`, `m2l_translate`, `l2l_translate` and
  `l2p_evaluate`.
- `Canopy_Test_P2P_*` — 3 errors; `tests/tstP2P.hpp:449` passes
  `std::array<double,3>` where `TreeBuilder`'s constructor
  (`src/Canopy_TreeBuilder.hpp:164-166`) takes `std::array<double,6>`. The
  signature drift traces to commit `8b0298e` "Refactor Solver constructor".

Both predate T2 — verified by stashing T2's diff and rebuilding at `64d1648` —
and neither carries the `regression` label. **They are repaired separately,
outside this document. No task here repairs either one, and no task here adds
tests to `tests/tstLaplaceKernel.hpp`.** The consequence worth stating is that
`ctest -L unit` cannot serve as this document's diagnostic layer until they are
fixed; the diagnostic layer is the individually-compiling component tests plus
`ctest -R Canopy_Test_LaplaceSolve_MPI_SERIAL`.

**The key, the cap and the overflow path.** The key struct is `{dd, ii, jj, kk}`
(`:308-318`) with an FNV-style hash (`:319-335`). The class comment at
`:280-294` describes the key as `(max_d, dd, ii, jj, kk)` — **the struct has no
`max_d`; the comment is stale and the code is authoritative**. `max_d` *is*
computed in the classify pass (`:870`) and discarded, so extending the key costs one
field, one `mix()` call and no new computation. The cap is a count,
`M2L_OP_COUNT_CAP = 32768` (`:304`); overflow assigns `op_idx = -1` with one
warning (`:1028-1047`) and routes those pairs to the per-pair `m2l_translate`
fallback (`:1231-1315` builds the tables, `:1599-1641` runs them).
`total_fallback_pair_count()` (`:424-430`) already exposes the count.

**Physical width was deliberately removed from the interaction-list builder.**
The classify pass is a pure-integer pipeline precisely so it produces
"bit-identical M2LKey output by construction" with "no per-source
`h_dc_for_filter` gather and no FP rounding" (`:778-787`), and
`half_width_at_depth` is computed **only under `CANOPY_ENABLE_DEBUG`**
(`:742-744`, `:762-776`) — so in a release build that code knows no physical
length at all. Reinstating a gather is not the fix and is not proposed:
$w_{\rm unit}(\mathrm{max\_d}) = w_{\rm root}/2^{\rm max\_d}$ is exact from
`TreeBuilder::root_box()` (already used at `src/Canopy_Solver.hpp:685`), one
array of `max_depth+1` doubles handed to the operator builder.

**Operator tables are rebuilt on every topology change.** The whole table is
rebuilt whenever `_interaction_list_dirty` (`:645-646`), which `setup()` sets
(`:637`) and `invalidate_interaction_list()` sets from
`src/Canopy_Solver.hpp:565` and `:610`. That is affordable for the
solid-harmonic and Cartesian tables and unaffordable for anything needing an SVD
per key. The mathematics says the tables depend only on
(level, offset, $b$) and not on particle positions — true, and false of this
code as written.

**The realized key count is unmeasured.** A tuning comment states it as
"globally ~16 k under MAC=0.5" (`:36-42`), which is 50× the textbook 316-offset
figure. This is a claim in a comment, not a measurement. T8 instruments it.

**Trilinos is already a required dependency**, found and marked `TYPE REQUIRED`
(`CMakeLists.txt:73-74`) for load balancing, with `${Trilinos_LIBRARIES}` linked
and `${Trilinos_INCLUDE_DIRS}` included unconditionally (`src/CMakeLists.txt:44`,
`:52`). Nothing in this document needs a new `find_package`.

**No task here is blocked on reading a dependency that has not been opened.**
Deliberately not read, and correctly deferred: the Trilinos/KokkosKernels dense
linear-algebra surface (`Teuchos_LAPACK.hpp`, `KokkosBatched_SVD_Decl.hpp`),
which only a black-box-FMM design needs; `Canopy_TreeBuilder.hpp` and
`Canopy_TreePartitioner.hpp` beyond `root_box()`, which no task touches; and
`Canopy_P2P.hpp` beyond the two sites cited above.

## Progress log

[abstract-solver-backend-progress-log.md](abstract-solver-backend-progress-log.md)
holds the session record: decisions and what forced them, signature changes,
measured numbers, and bugs that only running revealed. **Read it before starting
any task**, before changing a signature this document specifies, and before
reopening a question this document treats as settled. Each entry ends with an
`**Affects:**` line naming the later task IDs it changes, so scan those first.

## Task sequence

### T1 — The Laplace-solve harness gates the solid-harmonic path at every rank count — **DONE**

**Met.** `ctest -R Canopy_Test_LaplaceSolve_MPI_SERIAL` passes 6/6 at ranks 1-6
on unmodified code (flux job `f3XTw8Jdt7eo`, commit `4a351e1`), against
`tests/data/laplace_solve_P6.txt` generated at commit `fedf400` (flux job
`f3XKhLPnyd9H`). The configuration is frozen at charges uniform on
$[0.5, 1.5]$ and `num_steps = 12`, the latter chosen by the step-3 measurement
procedure: a 50-step trace first trips a degeneracy test at step 18, and 12 is
two thirds of that rounded down. `m2l_n_unique_ops() > 0` is asserted in
`bitForBitArtifacts` and `crossRankAgreement` and holds at all 21
(rank, rank-count) pairs — 686 at np=1, 111 to 386 per rank at np 2-6 — so the
far field is genuinely evaluated at the step every check reads, which is what
the 50-step configuration failed to do. `total_fallback_pair_count()` is 0
everywhere, keeping R4's discriminator intact. `LS_CROSS_RANK_TOL` is
$5.6\times10^{-10}$, 100x the worst measured cross-rank deviation of
$5.60\times10^{-12}$ (np=4, gradient); `LS_DIRECT_SUM_TOL` is
$9.63\times10^{-7}$, 3x the worst measured direct-sum deviation of
$3.21\times10^{-7}$ (np=2, potential). No cross-rank deviation came within
two orders of magnitude of the $10^{-9}$ stop-and-report threshold, so R8 did
not fire and stays retired at this configuration.

**The first sensitivity perturbation behaves exactly as required** (flux job
`f3XTnWDziu35`): with the `m` loop at `src/Canopy_DownwardSweep.hpp:1559`
reversed, `bitForBitArtifacts` fails on the `locals()` comparison and on
nothing else at np=1 rank 0 and both np=2 ranks — no operator-table, key-list,
`n_unique_ops`, $A_{n,m}$ or extent failure accompanies it — while
`crossRankAgreement` passes at np 2-6 and `matchesDirectSum` passes at np 1-6.

**The second perturbation does not behave as this document states it will, and
the reason is a property of the code rather than of the harness.** Offsetting
the running counter by one slot at *both* `:1832` (pack) and `:1854` (unpack)
is a **provable no-op**: the rotation is applied identically on both sides of
an elementwise `MPI_Allreduce`, so it commutes through, and the only residue
is proportional to `_shared_snapshot_buf`, which measures identically **0** at
every depth where shared cells exist at this configuration. Verified rather
than argued — flux job `f3XTprsFKehD` reproduces every cross-rank and
direct-sum deviation to all 17 digits of the unperturbed run, and
`bitForBitArtifacts` still passes at np 1-2. Offsetting the pack side alone
*does* corrupt the dataflow, and `crossRankAgreement` catches it at np 2-6 by
seven orders of magnitude (deviation $2.6\times10^{-2}$ against a
$5.6\times10^{-10}$ tolerance; flux job `f3XTuDYZg9G3`) — so the substantive
claim, that this test protects the shared-cell dataflow T4 and T10 rewrite, is
verified. What is **not** true is the clause "while np=1 is unaffected":
`bitForBitArtifacts` and `matchesDirectSum` both fail at np=1 under that
perturbation, because the shared-cell Allreduce path runs at np=1 too —
measured `nshared` is 1 at depth 0, 8 at depth 1 and 3-4 at depth 2. R6 carries
that measurement and what it means for T10's discriminator. No tolerance was
raised, no rank count dropped, and both perturbations were reverted and the
target rebuilt.

Full per-step trace, per-rank-count measurements, provenance and the
shared-cell diagnostic are in
[the progress log](abstract-solver-backend-progress-log.md#t1--the-laplace-solve-gate-12-step-harness-completed).

**Depends on:** none.

**Fill in:** `tests/tstLaplaceSolve.hpp`, the frozen-configuration block and
the generator in particular; regenerated reference data under `tests/data/`,
replacing the orphaned `tests/data/golden_solid_harmonic_P6.txt`.

**Reference:** `tests/tstMultiSolve.hpp:866-901` for the brute-force $N^2$
reference and `:798-852` for the MPI gather that feeds it; `:915-931` for how a
full-pipeline solve is driven and compared; `tests/tstDownwardSweep.hpp:1314-1332` for the
model of a compile-time layout assertion; the reachability facts in
[Current state](#current-state).

#### What already exists

`tests/tstLaplaceSolve.hpp` is built, formatted and committed, and carries all
three tests — `bitForBitArtifacts`, `crossRankAgreement` and `matchesDirectSum`
— over a time loop with `migrate` between steps. The names of step 1, the
rank-count-independent global generator of step 2 (bar its charge
distribution), the time loop of step 4, the
`GlobalId` pairing, the MPI gather, the direct sum, the mismatch dumps, the
regeneration gate and the reference-file parser are all in place and
demonstrated working. The diagnostic accessors — `m2l_op_table()`,
`m2l_realized_keys()`, `m2l_n_unique_ops()`, `A_table()` and the retained
`_m2l_realized_keys` member — are additive, read-only, and change no
arithmetic. `total_fallback_pair_count()` measures 0 at every rank and every
rank count. Committed data is read through a `CANOPY_TEST_DATA_DIR` compile
definition applied per generated target from `tests/CMakeLists.txt:79-82`.

**What is not done:** `tests/data/laplace_solve_P6.txt` is not committed, so
`ctest -R Canopy_Test_LaplaceSolve_MPI_SERIAL` fails at every rank count with
`cannot open reference data file`; `LS_CROSS_RANK_TOL` and `LS_DIRECT_SUM_TOL`
are placeholders rather than measured values; and neither sensitivity
perturbation has been run. All three are downstream of the frozen
configuration's charge distribution and step count, which steps 2 and 3 fix.
`tests/data/golden_solid_harmonic_P6.txt` is orphaned — keyed to the deleted
per-rank generator — and is deleted when the replacement is committed.

#### Why the gate is split by rank count

The partition is not reproducible run-to-run above two ranks.
`TreePartitioner::partition_leaves` uses the Zoltan2 `multijagged` algorithm,
which `src/Canopy_TreePartitioner.hpp:417-419` documents as non-deterministic;
computing on rank 0 and broadcasting makes the assignment consistent *within* a
run but not *across* runs. `num_cells` is identical across runs at every rank
count, so the tree build is deterministic and it is cell *ownership* that moves
— which moves each rank's interaction list, its realized key set, its operator
table and its `locals()`. With one or two parts the cut is trivial and comes out
the same every time; at three and above it does not. Measurements are in
[the progress log](abstract-solver-backend-progress-log.md#t1--the-golden-bit-for-bit-harness).

Making the partitioner deterministic is **not** a prerequisite for any task
here. It is a recorded limitation, documented under `README.md` "Known Issues",
and the gate is built around it instead:

- **np 1-2 — bit-for-bit.** Bitwise identity of the four internal artifacts.
- **np 2-6 — cross-rank agreement.** The solve at $k$ ranks must reproduce the
  np=1 solve to floating-point reassociation.
- **np 1-6 — direct-sum accuracy.** The field must be the right field.

#### Why the charges are one-signed and the interval is bounded

The harness exists to protect the far field, so the configuration must leave a
far field to evaluate at the step the checks read. Two properties of the
integrated problem decide that, and both are constraints rather than free
choices.

**Charges are one-signed.** With charges on $[-1, 1]$ and `softening = 0.0`, the
closest opposite-charge pair in a 600-particle set free-falls to contact well
inside any interval long enough to be worth integrating: the expected
closest-pair distance is $\approx 0.0099$, giving a relative acceleration of
$\approx 1.0\times10^{4}$ and a free-fall time of $\approx 1.4\times10^{-3}$
— about 14 steps at $dt = 10^{-4}$. The pair collides, both participants are
ejected at high velocity, the bounding box grows roughly thirtyfold, and with
`max_depth = 6` the tree cannot refine into the residual cloud: the solve
degenerates to 29 cells at which **no pair is MAC-admissible and `n_unique_ops`
is 0**. Every check then passes vacuously — `matchesDirectSum` to machine
precision because the answer is pure P2P, `bitForBitArtifacts` against an
operator table with zero realized columns — and the exit criterion's first
sensitivity perturbation cannot fire at all, because the `m` loop it reverses is
reached only through a CSR entry with a valid `op_idx`. Measurements are in
[the progress log](abstract-solver-backend-progress-log.md). Charges uniform on
$[0.5, 1.5]$, the distribution `tests/tstMultiSolve.hpp:200` uses for its
gravity tests, remove the opposite-sign singularity. `grad` is $\nabla\varphi$
and the update is `v += dt * g` (`tests/tstMultiSolve.hpp:376-386`), so a
one-signed set is mutually attracting — it still collapses, but as a cloud
rather than as a two-body singularity.

**The interval is bounded by measurement, not by assumption.** Attraction alone
does not guarantee the tree survives 50 steps: the closest *same*-sign pair
free-falls on a comparable timescale. The step count is therefore fixed by
tracing the run and reading where the tree degenerates, not by asserting a
number — see step 3.

#### Why the far field is not checked to $10^{-10}$

The solid-harmonic far-field error is a truncation controlled by
$\theta^{P+1}$. At the frozen configuration ($\theta = 0.5$, $P = 6$) that is
$0.5^7 \approx 8\times10^{-3}$, and the repository's own bound on this exact
configuration is $5\times10^{-2}$ on the potential
(`tests/tstMultiSolve.hpp:929-930`), whose comment records $\approx3\times10^{-6}$
at the well-conditioned $N=200\mathrm{k}$ production scale. Reaching
$10^{-10}$ against a direct sum requires one of two things, and neither is
available:

- **Raise $P$ at $\theta = 0.5$.** $P+1 \approx \log(10^{-10})/\log(0.5) \approx 33$,
  so $P \approx 32$. There $N_t = 561$ and $N_s = 1089$, so
  $561\cdot1089\cdot16 \approx 9.3$ MiB per key — roughly 15 GB of operator
  table per rank at the ~1600 realized keys measured at $P=6$. It would also
  exercise a code path nothing ships.
- **Lower $\theta$ at $P = 6$.** $\theta = 10^{-10/7} \approx 0.037$, at which
  no pair is MAC-admissible and every interaction degenerates to P2P — so the
  far field the test exists to protect is never evaluated.

"Reaches $10^{-10}$ against a direct sum" and "M2L is actually exercised" are
one constraint read in two directions, and no initial condition threads them.
The direct-sum check is therefore carried at the accuracy the method delivers,
measured rather than asserted, and the *tight* gate is cross-rank agreement,
which is bounded by reassociation and not by truncation.

Softening does not enter. The frozen configuration sets `softening = 0.0`, and
`FmmConfig::near_softening_factor` "Only has an effect when softening > 0"
(`src/Canopy_Solver.hpp:69-78`), so the MAC softening floor is inert and no pair
is forced to P2P by it. `total_fallback_pair_count()` measures 0 at every rank
count, confirming it.

**Do:**

1. **The names are fixed, and nothing may rename them again.** The harness
   macro maps a name to `tst<NAME>.hpp` and to the target
   `Canopy_Test_<NAME>_MPI_<DEVICE>`
   (`cmake/test_harness/test_harness.cmake:104-112`), so the solve-level gate is
   `tests/tstLaplaceSolve.hpp`, the `UNIT_MPI_TESTS` entry `LaplaceSolve`
   (`tests/CMakeLists.txt:50`) and the target
   `Canopy_Test_LaplaceSolve_MPI_<DEVICE>`; the per-operator kernel tests are
   `tests/tstLaplaceKernel.hpp` under `UNIT_SERIAL_TESTS` (`:37`). The
   regeneration environment variable is `CANOPY_LAPLACE_SOLVE_REGENERATE` and
   the two flux wrappers are
   `scripts/tuolumne/run_ctest_laplace_solve.flux` and
   `scripts/tuolumne/run_laplace_solve_regenerate.flux`. `tasks/todo_0.md:483`
   still names the old kernel-test path; it is a different document and is
   **out of scope**.

2. **The particle generator is a rank-count-independent global set.** A
   per-rank generator would make both the global particle set and the total $N$
   functions of the rank count, so np=1 and np=6 would be different physics
   problems and could not be compared to one another at all. Instead: fix
   $N_{\rm total} = 600$, seed
   `1234 + P` once, generate the identical global set on every rank, and have
   each rank keep the contiguous slice
   `[rank * N_total / nprocs, (rank+1) * N_total / nprocs)`. Positions stay
   uniform on `[0.05, 0.95]`; charges are uniform on `[0.5, 1.5]` — one-signed,
   for the reason above. Velocities start at zero. `setup()`
   redistributes particles, so the initial slicing does not affect the answer;
   every rank count from 1 to 6 divides 600 exactly, giving 100 per rank at
   np=6, and the $N^2$ reference is 360 k pairs, which is milliseconds. This is
   a deliberate departure from reusing `tests/tstMultiSolve.hpp:753-767`
   verbatim: that generator is per-rank, and makes the cross-rank comparison in
   step 6 impossible to define. Everything in this step but the charge
   distribution is already in place; the charge distribution is the change.

   The AoSoA gains a `GlobalId` member, `first_id_on_this_rank + i`, the same
   device `tests/tstMultiSolve.hpp:110-113` uses. It is what pairs a particle
   across rank counts. Position cannot serve as that key: migration scrambles
   the local ordering, and the solve moves the particles (step 4), so the
   positions the last step is evaluated at differ in their last bits between one
   rank count and another.

3. **Hold the rest of the configuration fixed forever:** `P = 6`, `NComps = 1`,
   `Scalar = double`, `mac_theta = 0.5`, `ncrit = 16`, `max_depth = 6`,
   `replication_depth = 2`, `imbalance_tolerance = 0.05`, the six box tolerances
   and `ncrit_tol` at `0.1`, `softening = 0.0`, `dt = 1.0e-4`,
   `drift_multiplier = 1.0`, and `migrate` as the between-step maintenance call.

   **`num_steps` is fixed by measurement and then frozen with the rest.** Trace
   the run at `num_steps = 50` on the one-signed set and record, per step, the
   cell count, `n_unique_ops`, the global position range and the maximum
   gradient magnitude. A step is degenerate when any of three things holds:
   `n_unique_ops` has fallen below half its step-0 value; the global position
   range exceeds the initial bounding box by more than 50% in any dimension; or
   the maximum gradient magnitude exceeds 100x its step-0 value. If no step
   through 50 is degenerate, `num_steps = 50`. Otherwise `num_steps` is two
   thirds of the first degenerate step, rounded down — a margin, so that a later
   task perturbing the trajectory does not push the gate over a cliff it was
   sitting on. Record the trace and the chosen value in the log. The trace
   instrumentation is temporary and is removed before the task finishes.

4. **Drive `num_steps` timesteps, and evaluate all three checks on the state
   after the last solve.** Each step is: `solve<Position, Charge>(particles,
   /*compute_gradient=*/true)`; then the symplectic-Euler update
   `v += dt * g; r += dt * drift_multiplier * v`, on device against the AoSoA
   slices, exactly as `tests/tstMultiSolve.hpp:365-392` writes it; then
   `solver.migrate<Position>(particles)`. The last solve's artifacts and field
   are what every check below reads. There is no update and no `migrate` after
   that solve: `migrate()` re-runs `downward.setup()`, which would overwrite
   `locals()` and the operator table before the gate reads them, and the update
   would move the particles away from the positions the field was evaluated at.
   The loop is therefore `num_steps` solves with `num_steps - 1` intervening
   maintenance steps.

   **`migrate`, and not `rebalance`, `rebuild` or `auto_maintain`.** Migrate
   moves particles to the ranks that already own their cells and never
   repartitions, so the np 1-2 bitwise gate faces one partition rather than one
   per step. The partitioner is not reproducible run-to-run above two ranks
   (`src/Canopy_TreePartitioner.hpp:417-419`), and every further invocation is
   another opportunity for the np=2 cut to stop coming out the same way.

   **`dt` is small deliberately.** The np=$k$ field differs from the np=1 field
   by reassociation at each step, that difference enters the velocity update,
   and the perturbed positions feed the next step. A large `dt` compounds it
   until the cross-rank deviation measures trajectory divergence instead of
   summation order, which is the one quantity step 6 exists to bound.

5. **`LaplaceSolve.bitForBitArtifacts` — np 1-2.** `GTEST_SKIP` at np ≥ 3 with a
   message naming the partitioner non-determinism and pointing at `README.md`
   "Known Issues", so `ctest` output shows the gate was skipped rather than
   passed. Four artifacts, taken from the last solve, are dumped to host
   and compared on their bit patterns — as `uint64_t` via `memcpy`, never as
   `double`, since `EXPECT_DOUBLE_EQ` has a tolerance and `NaN != NaN`:
   - `DownwardSweep::locals()` — hash and extents;
   - the M2L operator table over the realized key columns — hash and extents;
   - the $A_{n,m}$ table and its extent — full bit patterns;
   - the sorted realized key list — hash — and `n_unique_ops` — full.

   Only the last solve's artifacts are compared, and that is sufficient: the
   state at the last step is cumulative — the locals, the operator table and the
   realized key set there all descend from the positions the preceding solves
   produced — so a bitwise difference introduced at any earlier step is already
   carried into it.

   Keep the in-repo FNV-1a, the mismatch dumps that write full arrays to the
   build directory and name the paths in the failure message, and the layout
   `static_assert`s that `coeff_view_type` is `LayoutRight` and the operator
   table is `LayoutLeft`.

6. **`LaplaceSolve.crossRankAgreement` — np 2-6.** `GTEST_SKIP` at np=1, which
   is the reference. The FMM answer is partition-independent *as mathematics*,
   and that holds at every step: the interaction set is a function of the tree
   and not of ownership; cells at depth ≤ `replication_depth` are shared and
   their M2L runs on rank 0 alone, with the Allreduce summing rank 0's
   contribution against zeros (`src/Canopy_DownwardSweep.hpp:701-708`); every
   non-shared target is owned by exactly one rank; and `num_cells` is identical
   across runs at every rank count. Only the summation order moves when the
   partition moves, so a *single* np=$k$ solve reproduces a single np=1 solve to
   floating-point reassociation, order $10^{-13}$ relative.

   What that argument does not settle on its own is the size of the deviation
   after the full interval. Each step's $O(10^{-13})$ field difference enters the
   velocity update and is carried into the next step's positions, so the
   integrator amplifies it by a factor this document does not derive. The
   tolerance is therefore **measured, not derived**. It remains far tighter than
   any direct-sum bound can be, and it remains pointed squarely at what np ≥ 3
   uniquely risks: the MPI packing T4 rewrites and the shared-cell Allreduce T10
   rewrites.
   - Gather per-particle `GlobalId`, potential and gradient to rank 0, as
     `tests/tstMultiSolve.hpp:798-852` already does, and pair particles by
     `GlobalId`.
   - Compare against the committed np=1 record, normalizing by a **global**
     scale: $\max|\varphi|$ for the potential and $\max|\nabla\varphi|$ for the
     gradient over the gathered set. Never a per-particle relative error: the
     gradient components pass through zero wherever a particle's neighbours
     cancel, so a per-particle ratio measures that cancellation rather than
     accuracy. One scale rule for both fields keeps the two comparisons reading
     the same way.
   - **Measure the tolerance before pinning it.** Record the maximum normalized
     deviation at every rank count in the progress log, then pin the assertion
     at 100× the worst measured value. If the measured deviation exceeds
     $10^{-9}$ at any rank count, **stop, record the measurements in the log,
     and report** — pinning a tolerance above that would bury a real defect in
     the parallel dataflow.
   - **Before attributing such a deviation to the dataflow, re-measure the same
     configuration at `num_steps = 1`.** A one-step deviation at reassociation
     level with a full-interval deviation above the threshold is the integrator
     amplifying reassociation; a one-step deviation already above the threshold
     is R8. Record both numbers either way. Measured on unmodified code at
     `num_steps = 1`, the cross-rank deviation is $10^{-15}$ on the potential and
     $3\times10^{-13}$ on the gradient at every rank count from 2 to 6, which is
     what retires R8 at this configuration.
   - Assert `total_fallback_pair_count() == 0` here too, so R4's discriminator
     is still covered at 3-6 where `bitForBitArtifacts` no longer runs.

7. **`LaplaceSolve.matchesDirectSum` — np 1-6.** A brute-force $O(N^2)$
   reference on rank 0 in double precision, over the gathered last-step
   positions and charges, structured as `tests/tstMultiSolve.hpp:866-901`, with the same
   global-scale normalization as step 6 rather than that function's per-particle
   ratio, for the same cancellation reason. **Measure first at every rank count,
   record in the log, and pin at 3× the worst measured value.** This test
   overlaps `SolveFusedM2L.matchesPriorReference`
   (`tests/tstMultiSolve.hpp:915-931`) and is carried anyway: it is the only
   check in this harness that the far field is the *right* field.
   `crossRankAgreement` compares the solve against itself and would pass a
   uniformly wrong answer at every rank count.

8. **Regenerate and commit one reference-data file**, `tests/data/laplace_solve_P6.txt`,
   replacing `golden_solid_harmonic_P6.txt`. Keep the existing provenance header
   and record format. It holds:
   - three bit-for-bit records — `(nprocs, rank)` of `(1,0)`, `(2,0)`, `(2,1)` —
     each from that configuration's last solve, rather than the 21 that a 1-6
     bit-for-bit gate required;
   - one np=1 field record: 600 potentials and $3\times600$ gradient components
     as bit patterns in canonical `GlobalId` order (~41 KB). The np=1 run is
     bit-reproducible, which is what makes committing its output meaningful;
   - a hash of the initial global particle set — 600 positions and 600 charges,
     as bit patterns. That set is generated identically on every rank at every
     rank count, so the hash is exact by construction, which the last-step
     positions are not. It is checked first, so a generator drift fails with one
     legible message instead of 600 value mismatches.

   Record the commit the data was generated at, in the log, in the file's
   provenance header, and in the commit message.

9. Keep the data-directory plumbing as it is: `target_compile_definitions`
   setting `CANOPY_TEST_DATA_DIR` from `tests/CMakeLists.txt`, per generated
   target, **not** inside `Canopy_add_tests`
   (`cmake/test_harness/test_harness.cmake:104-112`) — that macro is shared by
   every test and no other test reads a data file.

**Exit criterion:** `ctest -R Canopy_Test_LaplaceSolve_MPI_SERIAL` passes at
ranks 1-6 on unmodified code, with `m2l_n_unique_ops() > 0` asserted at every
rank and every rank count, and both sensitivity perturbations behave as stated.

The `n_unique_ops` assertion is load-bearing rather than decorative: a
configuration whose tree has degenerated realizes zero M2L operators, at which
point `matchesDirectSum` passes to machine precision because the solve is pure
P2P, `bitForBitArtifacts` compares an operator table with zero realized columns,
and the first perturbation below cannot fail because the loop it reverses is
never entered. Every check passes and none of them measures the far field. The
assertion is what makes that state a failure instead of a pass. Put it in
`bitForBitArtifacts` and in `crossRankAgreement`, beside the existing
`total_fallback_pair_count() == 0`, so it is checked at all six rank counts.

The two perturbations:

- With the `m` loop at `src/Canopy_DownwardSweep.hpp:1559` reversed by hand
  (`for ( int m = n; m >= -n; m-- )` — mathematically identical and bitwise
  different; it is the only `for ( int m = -n; m <= n; m++ )` in the file),
  `bitForBitArtifacts` **fails on the `locals()` comparison specifically**, not
  merely somewhere, at np 1 and 2. `crossRankAgreement` and `matchesDirectSum`
  must **pass** under this perturbation: a reassociation-level difference is
  below their tolerances by construction. That is the expected outcome, not a
  coverage gap — it is what distinguishes the two regimes.
- With the shared-cell Allreduce pack/unpack perturbed — offset the running
  counter by one slot at `src/Canopy_DownwardSweep.hpp:1825-1835` and
  `:1847-1857` — `crossRankAgreement` **fails at np ≥ 2** while np=1 is
  unaffected. Without this check, the claim that this test protects T4 and T10
  is unverified.

Revert both perturbations and rebuild before finishing.


---

### T2 — Dead solid-harmonic scaffolding is deleted — **DONE**

**Depends on:** T1.

**Fill in:** `src/Canopy_UpwardSweep.hpp:212`, `:418-449`;
`src/Canopy_DownwardSweep.hpp:110`, `:306`, `:524`, `:531`, `:1959-2004`,
`:2006-2041`; `src/Canopy_LaplaceKernel.hpp:159`.

**Reference:** the callers table in [Current state](#current-state) (a),
re-verified by search before deleting.

**Do:**

1. Re-run the repository-wide search for each symbol across `src/`, `tests/` and
   `examples/` before deleting it. Do not trust the table; a caller may have
   landed since.
2. Delete the five dead members and `has_mplus_symmetry`. Delete
   `DownwardSweep::P` (`:110`) **only after** `M2L_NUM_SRC` (`:306`) is gone,
   since `P` is its only reader.
3. Do not touch `get_coeff_3d` (`src/Canopy_LaplaceKernel.hpp:177-193`) or
   `Canopy_SphericalCoefficients.hpp:70-91`. Both are live — see
   [Deliberate deviations](#deliberate-deviations).

**Exit criterion:** `ctest -R Canopy_Test_LaplaceSolve_MPI_SERIAL` passes the full
[Laplace-solve gate](#the-bit-for-bit-gate) — bit-for-bit at ranks 1-2,
`crossRankAgreement` at 2-6, `matchesDirectSum` at 1-6;
`ctest --output-on-failure -L regression -R MPI_SERIAL` shows exactly the six
pre-existing `MultiSolve` failures of [Current state](#current-state) and no
others, against a baseline run of the same command on the same checkout taken
before the deletion; and a search for each deleted symbol across `src/`,
`tests/` and `examples/` returns no hits.

**Met.** All six members are deleted — `apply_p2m_normalization_bridge`,
`apply_l2p_normalization_bridge`, `scale_locals_at_depth`, `M2L_NUM_SRC`,
`DownwardSweep::P` and `LaplaceKernel::has_mplus_symmetry` — 142 lines removed
across the three headers, `M2L_NUM_SRC` before `P` as required. Each symbol's
explanatory comment went with it; `execute()`'s comment kept its invariant and
lost the bridge clause. Neither sweep header was reformatted.

**The Laplace-solve gate passes at every rank count** (flux job `f3XUMSqyY4qV`):
`100% tests passed, 0 tests failed out of 6`, with `bitForBitArtifacts` green on
all three committed records at np 1-2, `crossRankAgreement` green at np 2-6 and
`matchesDirectSum` green at np 1-6. Stronger than pass/fail: every measured
deviation reproduces T1's pinned table **to every digit** at every rank count —
worst cross-rank `5.5987399483706545e-12` at np=4 against
`LS_CROSS_RANK_TOL = 5.6e-10`, worst direct-sum `3.2093610363952985e-07` at
np=2 against `LS_DIRECT_SUM_TOL = 9.63e-07` — as do `n_unique_ops` per rank
(686; 368/386; 273/204/329; 264/128/234/194; 180/168/147/217/187;
174/156/111/160/116/175), `initial_hash = 0xb6ad437608ad69b7`,
`locals_ext = (103,28,1)`, `a_extent = 169` and `fallback_pairs = 0`. Neither
tolerance was touched.

**Searches returned no hits.** After the deletion,
`grep -rn <symbol> src/ tests/ examples/` is empty for all five named symbols;
the only surviving whole-word `P` in `Canopy_DownwardSweep.hpp` is in four prose
comments (`:97`, `:295`, `:298`, `:490`, `:1505`). `get_coeff_3d` (7 references)
and `Canopy_SphericalCoefficients.hpp`'s `get_coeff` are untouched, per
[Deliberate deviations](#deliberate-deviations).

**The regression comparison was carried at np 1-2 and stopped there**, because
the post-deletion run hung at np=3. The pre-deletion baseline is flux job
**`f3XUAWAy6WFR`** on this checkout at commit `64d1648` — the run
[Current state](#current-state) records the failing set from; the post-deletion
run is flux job **`f3XUPJqSuqdh`**, cancelled after the hang. Over the np=1 and
np=2 overlap the two runs are **identical** — same failure sets and every
reported error value equal to the last digit, the partitioner-dependent np=2
values included. np=3-6 were not compared, and no later task carries a
`regression` clause to compare.

---

### T3 — M2L is a three-stage kernel-owned operation, solid-harmonic bit-identical — **DONE**

**This is the gate for the whole document.** Perform it alone. Do not fold any
part of T4 or later into it: the value of this task is that a bitwise difference
is attributable to exactly one change.

**Depends on:** T2.

**Fill in:** `src/Canopy_DownwardSweep.hpp:1431-1544` (`run_m2l_fused`), `:341-342`,
`:1449`, `:1518`; `src/Canopy_LaplaceKernel.hpp:639-672` (`m2l_apply_operator`,
grown into `m2l_core`).

**Reference:** the contraction being moved is `src/Canopy_DownwardSweep.hpp:1496-1526`;
the scratch-split rationale is `:1458-1462`; the write-back is `:1533-1542`.

**Do:**

1. Add to the basis contract, all `KOKKOS_INLINE_FUNCTION static`:
   - `m2l_scratch_bytes(int n_comps) -> size_t` — the per-team scratch the basis
     needs, in bytes. The sweep allocates exactly this and passes it in.
   - `m2l_pre_cell(team, M, src_cell, ops, scratch)`
   - `m2l_core(team, M, src_cell, ops, op_idx, scratch)`
   - `m2l_post_cell(team, scratch, L_out, tgt_cell, ops)`
   plus the typedef `m2l_operators_type`, **opaque to the sweep**.
2. For the solid-harmonic basis: `m2l_pre_cell` is a no-op; `m2l_core` is
   `:1496-1526` moved verbatim; `m2l_post_cell` is `:1533-1542` moved verbatim.
   The **real/imag split scratch must be preserved inside the basis** — the sweep
   hands over raw bytes, and the basis views them as two `scalar_type` arrays
   exactly as `:1454-1462` does. Handing the basis a `complex_type` scratch view
   instead is mathematically identical and **bitwise different**, and is the
   single most likely way this task fails.
3. Keep `Nt`, `NComps` and the loop bounds `constexpr` through the move
   (`:1443-1445`, `:1499`, `:1502-1504`). If any becomes a runtime value the
   fused kernel deoptimizes — a performance regression no correctness test sees.
4. The sweep retains the traversal, the CSR walk, the team launch, the zeroing
   (`:1481-1487`) and the scratch allocation. It must carry nothing but
   `int op_idx` and the CSR.

**Exit criterion:** `ctest -R Canopy_Test_LaplaceSolve_MPI_SERIAL` passes with
**identical bit patterns** on all four artifacts at ranks 1-2, and with
`crossRankAgreement` at 2-6 and `matchesDirectSum` at 1-6 passing at their pinned
tolerances. If the bit patterns differ, **stop and record the difference in the
log before changing anything else** — that is R1's trigger and it changes the
rest of the document. Ranks 1-2 are the whole bitwise gate for this task and that is
sufficient: the contraction being moved is per-target-cell arithmetic, so R1
presents at np=1 (see [The bit-for-bit gate](#the-bit-for-bit-gate)).
Additionally, `grep -n "complex_type" src/Canopy_DownwardSweep.hpp` must show no
hit inside `run_m2l_fused`'s body.

**Met.** Commit `49a88de`, flux job **`f3XWVoKhkc5u`** (tuolumne1040, Cray clang
20.0.0, `RelWithDebInfo`, Kokkos SERIAL): `ctest -R
Canopy_Test_LaplaceSolve_MPI_SERIAL` reports **6/6 passed**, and every measured
deviation reproduces T1's pinned table to all 17 digits at **every** rank count
1-6 — cross-rank 4.1994107222659022e-13 / 2.1570013757642702e-12 (np=2),
1.114294857300434e-12 / 5.5987399483706545e-12 (np=4, the worst, against
`LS_CROSS_RANK_TOL = 5.6e-10`), and direct-sum 3.2093610331931809e-07 /
4.2399302231264458e-08 (np=1) through 3.2093610299905848e-07 (np=6) against
`LS_DIRECT_SUM_TOL = 9.63e-07`. `fallback_pairs` is 0 at every rank and rank
count, so **R4**'s discriminator is intact. `bitForBitArtifacts` passes at np 1-2
on all four artifacts, so **R1 did not fire**: the solid-harmonic M2L moved into
the basis bit-identically and the rest of this document stands as written. The
`complex_type` grep returns no hit inside `run_m2l_fused`'s 73-line body.

**R3 did fire, mildly, and is recorded rather than fixed.** The M2L-kernel timer
rises from 0.053-0.055 s to 0.062-0.068 s summed over 24 solves (about +18%),
against 1.5% on the downward sweep and 0.5% on `solve()`. It is not the scratch
byte-fill and not the per-pair accumulator-view construction — both were
measured and neither accounts for it. Numbers, variants and method are in the
progress log; no exit criterion depends on this.

---

### T4 — Coefficient storage and MPI packing are basis-agnostic — **NOT STARTED**

**Depends on:** T3.

**Fill in:** the five shared-code typedef sites of
[Current state](#current-state) (b), the basis-owned `m2l_operators_type` alias
named alongside them, and the three structural sites of (c).

**Reference:** `src/Canopy_MpiCoalescedExchange.hpp:72`, `:93-94`, `:96`;
`src/Canopy_UpwardSweep.hpp:534-535`, `:581-584`;
`src/Canopy_DownwardSweep.hpp:1782-1788`.

**Do:**

1. Add three traits to the contract:
   - `coeff_type` — replaces `complex_type` (`src/Canopy_LaplaceKernel.hpp:155`).
     For the solid-harmonic basis this is `Kokkos::complex<Scalar>` **exactly**;
     a "generalization" to `struct { Scalar re, im; }` or to split real/imag
     planes changes the layout and breaks T1.
   - `component_scalar_type` — the real scalar the MPI packing sees.
   - `scalars_per_coeff` — 2 for the solid-harmonic basis, 1 for a real basis.
2. Carry the operator table's element type with `coeff_type`: the basis's
   `m2l_operators_type<MemorySpace>` (`src/Canopy_LaplaceKernel.hpp:178-180`)
   becomes `Kokkos::View<coeff_type***, Kokkos::LayoutLeft, MemorySpace>`. For
   the solid-harmonic basis `coeff_type` *is* `Kokkos::complex<Scalar>`, so this
   is a rename that changes no bits; it exists so the operator-table element type
   and the coefficient element type cannot be set independently and silently
   disagree. `tests/tstLaplaceSolve.hpp:689-691` `static_assert`s `LayoutLeft`
   through `DownwardSweep::m2l_op_table_view_type`, which is an alias of
   `m2l_operators_type` — that test must keep compiling untouched.
3. Replace `typename complex_type::value_type`
   (`src/Canopy_MpiCoalescedExchange.hpp:72`) with the trait, and the three
   literal `2 *` factors (`:96`; `src/Canopy_UpwardSweep.hpp:583`;
   `src/Canopy_DownwardSweep.hpp:1788`) with `scalars_per_coeff`.
4. Leave `view.extent(1)` / `view.extent(2)` (`:93-94`) alone — already generic.

**Signature changes and their callers.** `coalesced_view_exchange`
(`src/Canopy_MpiCoalescedExchange.hpp:64-70`) keeps its signature; only its body
changes. `CoalescedExchangeBuffers<complex_type, …>` becomes
`CoalescedExchangeBuffers<coeff_type, …>` at two declaration sites —
`src/Canopy_DownwardSweep.hpp:253-254` and `src/Canopy_UpwardSweep.hpp:179-180`
— and has no other users.

**Exit criterion:** `ctest -R Canopy_Test_LaplaceSolve_MPI_SERIAL` passes the full
[Laplace-solve gate](#the-bit-for-bit-gate) — bit-for-bit at ranks 1-2,
`crossRankAgreement` at 2-6, `matchesDirectSum` at 1-6
(`scalars_per_coeff == 2` must reproduce today's packing exactly — bit-for-bit at
1-2 pins the layout, and `crossRankAgreement` at 3-6 is what pins the
rank-count-dependent indexing this task rewrites); and a
`static_assert` that `sizeof(coeff_type) == scalars_per_coeff * sizeof(component_scalar_type)`
holds for the solid-harmonic basis.

---

### T5 — Auxiliary tables are owned by the basis — **NOT STARTED**

**Depends on:** T4.

**Fill in:** `src/Canopy_UpwardSweep.hpp:233-235`, `:501-504`;
`src/Canopy_DownwardSweep.hpp:529`, `:1099-1100`, `:1636-1639`, `:1688-1691`.

**Reference:** `build_A_coefficients` (`src/Canopy_SphericalCoefficients.hpp:132`),
whose `2*P` argument (`src/Canopy_UpwardSweep.hpp:235`) exists because M2L reaches
degree $n+j$.

**Do:**

1. Add `aux_tables_type` and `build_aux_tables(order, params) -> aux` to the
   contract. The solid-harmonic basis returns the $A_{n,m}$ view built to $2P$; a
   basis needing nothing returns an empty struct.
2. Replace the `A_table` parameter with `aux` in every operator signature, and
   delete `DownwardSweep::_A_table` (`:264`, borrowed at `:529`).
3. Do **not** change the `2*P` argument. `m2l_build_operator` already `continue`s
   on `A == 0` (`src/Canopy_LaplaceKernel.hpp:612-614`), so a table built one
   degree short produces a *quietly wrong* operator rather than a crash.

**Signature changes and their callers.** Four operators lose `A_table` and gain
`aux`: `m2m_translate` (`src/Canopy_LaplaceKernel.hpp:273-278`), called at
`src/Canopy_UpwardSweep.hpp:501-504`; `m2l_translate` (`:373-378`), called at
`src/Canopy_DownwardSweep.hpp:1636-1639`; `l2l_translate` (`:688-693`), called at
`src/Canopy_DownwardSweep.hpp:1688-1691`; `m2l_build_operator` (`:516-519`),
called at `src/Canopy_DownwardSweep.hpp:1099-1100`. `p2m_contribution` and
`l2p_evaluate` never took it.

**Exit criterion:** `ctest -R Canopy_Test_LaplaceSolve_MPI_SERIAL` passes the full
[Laplace-solve gate](#the-bit-for-bit-gate) — bit-for-bit at ranks 1-2,
`crossRankAgreement` at 2-6, `matchesDirectSum` at 1-6 — the
$A_{n,m}$ artifact and its extent, compared at ranks 1-2, pin the `2*P` argument;
and `grep -n "_A_table" src/Canopy_DownwardSweep.hpp` returns no
hits.

---

### T6 — A non-harmonic conformance basis drives the full pipeline — **NOT STARTED**

This is the first proof that the contract is real rather than a rename. It is a
fixture, not a method: low accuracy, but an **exactly checkable** far field.

**Depends on:** T5.

**Fill in:** new `tests/CanopyTest_MonopoleBasis.hpp`; new
`tests/tstFarFieldContract.hpp`; `tests/CMakeLists.txt` `UNIT_MPI_TESTS`
(`:47-55`).

**Reference:** the trait and operator contract as it stands after T5;
`tests/tstDownwardSweep.hpp:57` for how a test instantiates a basis and drives
the sweeps directly, without going through `Solver`.

**Do:**

1. Write `MonopoleBasis<Scalar, Order, NComps>`: one **real** coefficient per
   cell per component, holding the cell's total charge. `coeff_type = Scalar`,
   `scalars_per_coeff = 1`, `num_coeffs_per_cell = 1`,
   `m2l_num_src_coeffs = 1`, `sets_per_component = 1`,
   `aux_tables_type` = empty struct.
2. Its operators: P2M sums charge; M2M sums children; M2L is
   $L^A \mathrel{+}= q^B / |c_A - c_B|$ against a one-entry operator table;
   `m2l_pre_cell` and `m2l_post_cell` are no-ops; L2L copies the parent's local
   to each child; L2P returns the local as the potential and zero as the
   gradient. Every one of these is exactly reproducible on host.
3. Write `tstFarFieldContract.hpp` driving `UpwardSweep`/`DownwardSweep` with it
   and comparing `locals()` against a host computation of the same sum over the
   same interaction list, at `EXPECT_DOUBLE_EQ`.
4. Add a negative test: a basis declaring
   `sizeof(coeff_type) != scalars_per_coeff * sizeof(component_scalar_type)`
   must fail to compile. Guard it behind a
   `#ifdef CANOPY_TEST_EXPECT_COMPILE_FAILURE` block and document in the header
   how to run it by hand, since CTest cannot assert a compile failure here.

**Exit criterion:** `ctest -R Canopy_Test_FarFieldContract_MPI_SERIAL` passes at
ranks 1-6; the Laplace-solve gate still passes all three checks
(`ctest -R Canopy_Test_LaplaceSolve_MPI_SERIAL`); and deliberately breaking one
trait on `MonopoleBasis` — set
`scalars_per_coeff = 2` while leaving `coeff_type = Scalar` — makes the
conformance test fail rather than pass with wrong numbers. Restore it before
finishing.

---

### T7 — The M2L key carries depth, chosen by the basis — **NOT STARTED**

**Depends on:** T3.

**Fill in:** `src/Canopy_DownwardSweep.hpp:280-294` (the stale comment),
`:301-302`, `:308-318`, `:319-335`, and the classify pass around `:866-919`;
`tests/CanopyTest_MonopoleBasis.hpp`.

**Reference:** `max_d` is already computed in the classify pass and discarded;
`decode_morton` (`:51-70`) is what produces the depths.

**Do:**

1. Add `max_d` to `M2LKey` (`:308-318`) and one `mix()` call to `M2LKeyHash`
   (`:319-335`). Have the classify pass **always** emit it — no branch.
2. Add `canonicalize_key(key) -> key` to the contract, applied **before**
   hashing. The solid-harmonic basis **zeroes `max_d`**, reproducing today's key
   set exactly; a softened basis returns the key unchanged. This is a trait call,
   not an `if constexpr` in shared code.
3. Add `key_needs_level` as a `constexpr bool` documenting the same fact for
   readers and for the byte accounting T8 needs. It must agree with
   `canonicalize_key`; assert that in the conformance test.
4. Replace the `float` branch at `:301-302` with an `m2l_key_dd_max` trait. The
   solid-harmonic basis returns today's values (4 for `float`, 6 otherwise) so the
   guard behaves identically.
5. Fix the stale comment at `:280-294` to describe the key the code now builds.
6. Raise `MonopoleBasis` to `key_needs_level = true` and identity
   `canonicalize_key`, so both branches are exercised.

**Exit criterion:** `ctest -R Canopy_Test_LaplaceSolve_MPI_SERIAL` passes the full
[Laplace-solve gate](#the-bit-for-bit-gate) — bit-for-bit at ranks 1-2,
`crossRankAgreement` at 2-6, `matchesDirectSum` at 1-6.
The sorted-key-list and `n_unique_ops` artifacts pin that the solid-harmonic key
set is unchanged, and they are compared **only at ranks 1-2** — above two ranks
the realized key set moves with the partition, so there is nothing stable to
compare it against. That is the one place this split costs real coverage: a
`canonicalize_key` bug that only manifests at a depth reached at higher rank
counts would be caught by `crossRankAgreement` as a field difference, not
attributed to the key set. Also `ctest -R Canopy_Test_FarFieldContract_MPI_SERIAL`
passes and its assertion that `MonopoleBasis` realizes **strictly more** distinct
keys than a `max_d`-zeroing basis on the same tree holds — proving the level
actually reaches the key rather than being silently dropped. Run that assertion
at np=1, where the key set is reproducible.

---

### T8 — The operator-table cap is a memory budget with a per-basis overflow policy — **NOT STARTED**

**Depends on:** T7.

**Fill in:** `src/Canopy_DownwardSweep.hpp:304`, `:1028-1047`, `:1070`.

**Reference:** the overflow path is `:1028-1047` (assign `op_idx = -1`, warn
once); the fallback tables are built at `:1231-1315` and run at `:1599-1641`;
`total_fallback_pair_count()` is `:424-430`.

**Do:**

1. Add `bytes_per_key` as a `constexpr size_t` to the contract. Solid-harmonic at
   $P=8$: $N_t N_s \cdot 16 = 45\cdot81\cdot16$, about 58 KB.
2. Compute
   `effective_cap = min(M2L_OP_COUNT_CAP, byte_budget / KernelType::bytes_per_key)`
   and use it at `:1028-1029`. The budget is a new `FmmConfig` field defaulting
   to 2 GB. Retaining the count cap is deliberate — see
   [Deliberate deviations](#deliberate-deviations).
3. Add an `m2l_overflow_policy` trait taking one of two enumerators:
   - `M2LOverflow::PerPairTranslate` — today's behavior, and what the
     solid-harmonic and Cartesian-Taylor bases use. Requires the basis to define
     `m2l_translate`.
   - `M2LOverflow::EscalateToP2P` — the pair is handed to the direct sum instead.
     **No path exists in the downward sweep to trigger this.** A basis selecting
     it must fail with a `static_assert` naming the fact that the escalation path
     is unimplemented. Do not add a lenient fallback; a basis that cannot
     evaluate its own operator per pair and cannot escalate must not silently
     produce a partial far field.
4. Emit `n_unique_ops` (`:1070`) and `n_unique_ops * bytes_per_key` under
   `CANOPY_ENABLE_PROFILING`, so the realized key count stops being a claim in a
   comment. Record the measured number in the progress log.

**Exit criterion:** `ctest -R Canopy_Test_LaplaceSolve_MPI_SERIAL` passes the full
[Laplace-solve gate](#the-bit-for-bit-gate) — bit-for-bit at ranks 1-2,
`crossRankAgreement` at 2-6, `matchesDirectSum` at 1-6,
with `total_fallback_pair_count() == 0` asserted at every rank count — both
`bitForBitArtifacts` and `crossRankAgreement` check it, so the discriminator
survives at 3-6; a test that sets the byte
budget low enough to bind before the count cap drives
`total_fallback_pair_count() > 0` and **still produces the same potential to
$5\times10^{-2}$** (the fallback path is different arithmetic, not wrong
arithmetic); and a basis declaring `EscalateToP2P` fails to compile with the
named message.

---

### T9 — Operator construction splits into a persistent cache and a per-tree map — **NOT STARTED**

**Depends on:** T8.

**Fill in:** `src/Canopy_DownwardSweep.hpp:645-646`, `:1079-1109`;
`src/Canopy_Solver.hpp:69`, `:169-178`, `:565`, `:610`, `:685`.

**Reference:** the rebuild trigger is `_interaction_list_dirty` (`:645-646`), set
by `setup()` (`:637`) and by `invalidate_interaction_list()`
(`src/Canopy_Solver.hpp:565`, `:610`).

**Do:**

1. Split the rebuild at `:1079-1109` into two pieces:
   - a **geometry-keyed operator cache**, keyed by the canonicalized `M2LKey`,
     persisting across topology changes — a key already built is never rebuilt;
   - a **per-tree key→`op_idx` map**, rebuilt on the dirty flag exactly as today.
2. Replace `m2l_build_operator(dd, ix, iy, iz, aux, T_out)` with
   `build_m2l_operators(keys[], unit_w[], kernel_params) -> ops`, a **host**
   method (not `KOKKOS_INLINE_FUNCTION`) called only for keys the cache lacks.
   It receives the whole missing key set at once so a basis that batches its
   construction can.
3. Plumb `kernel_params` — carrying at minimum the softening $b$ — from
   `FmmConfig::softening` (`src/Canopy_Solver.hpp:69`) through `Solver` into
   `_downward` before the table build. Today it reaches only `_p2p` and
   `_comm_plan` (`:169-178`, `:680-717`).
4. Supply `unit_w` as an array of `max_depth+1` half-widths computed as
   $w_{\rm root}/2^{d}$ from `TreeBuilder::root_box()` (`src/Canopy_Solver.hpp:685`).
   **Do not reinstate a per-source gather of cell centers** — the classify pass
   stays a pure-integer pipeline (`:778-787`). Integer keys stay integer; only
   the builder sees lengths.
5. State on the declaration whether `unit_w` is a half-width or a full width, and
   whether the offset convention is source-minus-target. The existing key comment
   uses half-widths and source-minus-target (`:280-294`, corrected in T7).

**Signature changes and their callers.** `m2l_build_operator` is removed and
replaced; its single caller is `src/Canopy_DownwardSweep.hpp:1099-1100`. The
solid-harmonic basis's `m2l_build_operator` body
(`src/Canopy_LaplaceKernel.hpp:516-630`) moves inside the new method's per-key
loop unchanged.

**Exit criterion:** `ctest -R Canopy_Test_LaplaceSolve_MPI_SERIAL` passes the full
[Laplace-solve gate](#the-bit-for-bit-gate) — bit-for-bit at ranks 1-2,
`crossRankAgreement` at 2-6, `matchesDirectSum` at 1-6;
and a test that calls `invalidate_interaction_list()` and re-solves
shows `interaction_list_build_count()` (`:236-239`) incremented while a new
counter on the operator cache shows **zero** keys rebuilt — proving the split is
real. A cache that silently rebuilds everything would pass the bit-for-bit test
and fail this one.

---

### T10 — Locals carry multiple sets per component — **NOT STARTED**

**Depends on:** T6, T4.

**Fill in:** `src/Canopy_DownwardSweep.hpp:534`, `:1723-1740`, `:1760-1802`;
`tests/CanopyTest_MonopoleBasis.hpp`.

**Reference:** `coalesced_view_exchange` already reads `view.extent(2)`
(`src/Canopy_MpiCoalescedExchange.hpp:94`) and needs nothing. Only the allocation
and the two hand-rolled Allreduce loops change.

**Do:**

1. Add `sets_per_component` as a `constexpr int` to the contract, 1 for the
   solid-harmonic basis.
2. Change the `_locals` allocation (`:534`) third extent from `NComps` to
   `NComps * sets_per_component`, and the same in the snapshot pack (`:1723-1740`,
   `per_cell_complex` at `:1724`) and the Allreduce pack/unpack (`:1760-1802`,
   `per_cell_complex` at `:1760`).

   **Give the three loops one slot expression rather than three running
   counters.** The snapshot pack, the Allreduce pack and the Allreduce unpack each
   hand-roll their own `idx++` over the same `(ci, c)` nest, and the third factor
   is what makes them able to disagree. A single
   `shared_slot(i, ci, c, s) -> int`, used by all three and stating its own
   flattening order on the declaration, removes that class by construction — see
   R6, where it is the most loudly-caught class but also the easiest not to write.
3. Raise `MonopoleBasis` to `sets_per_component = 2`, where set 0 is the monopole
   potential and set 1 is a **deliberately distinct** quantity — the monopole
   scaled by the cell half-width. A packing bug that aliases the two sets is
   invisible if they hold the same numbers.
4. Extend the conformance test to check both sets independently across the
   shared-cell Allreduce, at ranks 1-6 — shared cells exist at np=1 too (R6).
5. **Assert that the slot map is a bijection onto the full slot range**, in the
   conformance fixture or under `CANOPY_ENABLE_DEBUG`: mark each buffer slot as
   the pack writes it, then require every slot written exactly once and the total
   to equal `nshared * coeffs_per_cell * NComps * sets_per_component`. This is the
   property that separates an inert relabeling from the two classes that corrupt
   an answer — aliasing, which writes a slot twice, and truncation, which leaves
   slots unwritten. It needs no reference data, no configuration change, and holds
   at every rank count.

**Exit criterion:** `ctest -R Canopy_Test_LaplaceSolve_MPI_SERIAL` passes the full
[Laplace-solve gate](#the-bit-for-bit-gate) — bit-for-bit at ranks 1-2,
`crossRankAgreement` at 2-6, `matchesDirectSum` at 1-6
(`sets_per_component == 1` reproduces today's shapes exactly; per R6 this gate
catches a pack/unpack disagreement everywhere, aliasing at np 1-2 through
`bitForBitArtifacts`, and truncation at np 2-6 through `crossRankAgreement`); and
`ctest -R Canopy_Test_FarFieldContract_MPI_SERIAL` passes at ranks 1-6 with both
sets checked, failing if set 1 is replaced by a copy of set 0, and with the
step-5 slot-coverage assertion holding at every rank count — verified by
deliberately shortening the packed slot range so it fails, then restoring it.

---

### T11 — `Solver` selects the far field, with existing callers unchanged — **NOT STARTED**

**Depends on:** T9, T10.

**Fill in:** `src/Canopy_Solver.hpp:104-105`, `:112`, `:719-727`.

**Reference:** the six existing instantiations listed under R-A..R-D in
[Problem](#problem).

**Do:**

1. Add a **defaulted template template parameter** to `Solver` (`:104-105`):
   `template <class, int, int> class FarField = LaplaceKernel`, and change the
   typedef at `:112` to `using kernel_type = FarField<Scalar, P_ORDER, NComps>;`.
2. Give `createSolver` (`:719-727`) the same defaulted parameter and forward it.
3. Document in the class comment (`:97-101`) that `P_ORDER` is now "the basis's
   order knob" — $P$ for a solid-harmonic basis, $p$ for Taylor, $n$ for
   Chebyshev. Different quantities, same slot.
4. Delete the FD gradient (`src/Canopy_LaplaceKernel.hpp:851-879`)? **No** — that
   is not this task. `l2p_evaluate`'s signature (`:800-804`) is already generic:
   it names no basis concept, taking the locals view, a cell index, an offset and
   a width, and returning `phi[NComps]` plus a 2-D accessor. The finite
   difference stays until a basis that can differentiate analytically replaces it,
   which is T12's business.

**Exit criterion:** all six existing instantiations compile **unmodified** — do
not touch `tests/tstMultiSolve.hpp` or the three examples; the Laplace-solve gate
passes all three checks (`ctest -R Canopy_Test_LaplaceSolve_MPI_SERIAL`);
and a new compile-only test instantiates
`Solver<TEST_MEMSPACE, TEST_EXECSPACE, double, 1, 1, MonopoleBasis>` successfully.

---

### T12 — A Cartesian-Taylor basis — **NOT STARTED** — **COARSE**

This task is stated coarsely on purpose. It is where the mathematics lives, and
its fine-grained design cannot be written before the contract above is real and
its own open questions are answered.

**Depends on:** T11.

**Fill in:** new `src/Canopy_CartesianTaylorBasis.hpp`; one line in
`src/CMakeLists.txt` `HEADERS_PUBLIC` (`:3-16`); new tests and one line in
`tests/CMakeLists.txt`.

**Reference:** [canopy-questions.md](canopy-questions.md) §§1-3 for the
derivative ladder $\partial_a P_m = -(2m{+}1)\,r_a P_{m+1}$ with
$P_m = w^{-(2m+1)/2}$, $w = r^2 + b$, and for the multi-index recurrence for
$b_{k+e_i}$; [canopy-kernel-rec.md](canopy-kernel-rec.md), "Convergence per DOF",
for what accuracy each order $p$ buys.

**Do:**

- Coefficients are real, $\binom{p+3}{3}$ per cell, so `coeff_type = Scalar` and
  `scalars_per_coeff = 1`. `sets_per_component = 1`.
- M2L is the sole kernel-touching operator: $\ell_p^A = \sum_q (-1)^{|q|}\,b_{p+q}(R)\,M_q^B$.
  M2M and L2L are binomial Taylor shifts that never see the kernel. L2P
  differentiates the local expansion analytically, which deletes the finite
  difference at `src/Canopy_LaplaceKernel.hpp:851-879` for this basis.
- `m2l_pre_cell` and `m2l_post_cell` are no-ops. `m2l_overflow_policy` is
  `PerPairTranslate` — the ladder is device-evaluable, at a register cost worth
  measuring.
- `key_needs_level = true`, `canonicalize_key` is the identity: softening
  introduces the absolute length $\sqrt{b}$ and destroys the scale invariance the
  solid-harmonic operators exploit.
- Set `Scalar = double` by `static_assert`.

**Additional information needed** — each of these must be answered before a
fine-grained design is possible, and none is answerable from the code:

1. **What order $p$ is required?** The downstream solver's accuracy requirement
   decides it, and the answer decides whether this basis is viable at all: at
   standard admissibility each order buys between 0.24 and 0.48 decades while the
   DOF count grows as $\binom{p+3}{3}\sim p^3/6$, so $10^{-6}$ wants $p\approx11$-24
   ([canopy-kernel-rec.md](canopy-kernel-rec.md), "Convergence per DOF"). R-B's
   $10^{-10}$ is likely out of reach for this basis. **This question is prior to
   the task, not part of it.**
2. **What is the realized key count and what do the tables cost?** T8's
   instrumentation answers it. At $p=4$ the tables are 9.8 KB per key; the
   multiplier is what is unknown.
3. **What sign and normalization convention links the moment definition to the
   $(-1)^{|q|}$ multiplier?** This is the one place the reference author flagged
   as needing care, and getting it subtly wrong produces a plausible-looking
   field rather than an obvious failure. The order-2 tensors are already
   implemented in the reference treecode and give a component-level oracle.
4. **How are symmetric tensors indexed?** Hand-derived Cartesian FMMs habitually
   go wrong here. The index map is a design decision, not an implementation
   detail, and it should be fixed and unit-tested before any operator is written.

**Exit criterion:** deferred to this task's own design. At minimum it must
include: a per-operator unit test against the reference treecode's order-2
tensors; a full-pipeline solve with `near_softening_factor = 0` matching a direct
softened sum to the tolerance answered in question 1; and
the Laplace-solve gate still passing all three checks
(`ctest -R Canopy_Test_LaplaceSolve_MPI_SERIAL`), since this task adds a basis and
changes no shared code.

## Known risks

**R1 — the solid-harmonic M2L cannot move into the basis bit-identically.** The
most likely cause is the scratch: the real/imag split accumulator
(`src/Canopy_DownwardSweep.hpp:1454-1462`, `:1522-1524`) is a sweep-owned
optimization, and any scratch abstraction that hands the basis a `complex_type`
view produces a mathematically identical, bitwise different sum. **Presents as:**
`bitForBitArtifacts` failing on `locals()` at ranks 1-2 while the operator table, $A_{n,m}$ table
and key list all match. **Distinguished from R2** by exactly that: if the
operator table also differs, the cause is the table build, not the contraction.
**Do:** fall back to the narrow abstraction — keep T2, T4, T5, T7, T9, T10, T11
and drop T3, T8's three-stage assumptions and `m2l_pre_cell`/`m2l_post_cell`,
leaving the M2L apply solid-harmonic-specific and bit-identical by construction.
T12 remains fully viable under that fallback; a black-box basis would then need
its own M2L driver, duplicating roughly 150 lines of team launch, CSR walk,
scratch and write-back.

**R2 — `canonicalize_key` does not reproduce today's key set.** Zeroing `max_d`
should be exactly today's key, but a hash change or a classify-pass reordering
would multiply the solid-harmonic table by occupied depth. **Presents as:**
`bitForBitArtifacts` failing on the sorted-key-list and `n_unique_ops` artifacts,
with `locals()` also wrong. **Diagnosable only at ranks 1-2**: above two ranks the
realized key set moves with the partition, so a key-set difference there is
indistinguishable from partitioner drift and the artifacts are not compared. At
3-6 the same bug presents instead as a `crossRankAgreement` failure, which says
the field is wrong but not which artifact caused it. **Do:** reproduce at np=1 or
np=2 and compare the sorted key lists directly from the mismatch dump the test
writes; if the sets differ only in `max_d` being non-zero, the canonicalization is
not being applied before hashing.

**R3 — a trait indirection deoptimizes the fused M2L kernel.**
`num_coeffs_per_cell`, `Nt` and `NComps` are `constexpr` and drive unrolling
(`:1443-1445`, `:1499`). If any becomes a runtime value the kernel slows down
with no correctness signal. **Presents as:** every test passing and the solve
being slower.

**This risk has already fired once, at T3**, at roughly +18% on the M2L kernel —
which is +1.5% on the downward sweep and +0.5% on `solve()`. Two candidate causes
were tested and both excluded: making the sweep's zero-fill word-granular rather
than byte-granular, and replacing the basis's unmanaged accumulator views with
raw `scalar_type*`. The cost is intrinsic to putting the contraction behind the
basis interface. It is recorded, not fixed; no exit criterion depends on it.

**Do:** measure with the committed `scripts/tuolumne/run_laplace_solve_profile.flux`
against a **second build tree**, `build-tuolumne-prof/`, configured from the same
`run_cmake_tuolumne.sh` with `Canopy_ENABLE_PROFILING=ON` and
`Canopy_PROFILING_LEVEL=2`. The committed `build-tuolumne/` is configured
`Canopy_ENABLE_PROFILING=OFF` and emits no `[Canopy Diagnostics]` line at all, so
reading `CANOPY_PRINT_SOLVE_BREAKDOWN` (`src/Canopy_Solver.hpp:238`) there
measures nothing; reconfiguring it instead would put the bitwise gate and the
timing measurement in different build configurations. The figure is
`M2L kernel (all depths)` from the `DownwardSweep::execute()` table, summed over
the 24 solves one np=1 invocation performs. **T4's "before" number is T3's, not
unmodified code's:** 0.053-0.055 s before T3, 0.062-0.068 s after it. Record the
new number in the log against that pair.

**R4 — the byte budget changes which pairs overflow.** The overflow set decides
which pairs take the per-pair path, which is *different arithmetic* from the
operator path. **Presents as:** the Laplace-solve gate failing while
`total_fallback_pair_count()` has become non-zero. **Do:** that counter is the
discriminator — assert it is 0 for the frozen configuration, which the retained
count cap guarantees at $P=8$.

**R5 — the operator cache holds stale operators across a topology change.** T9's
cache persists deliberately; if a basis's operator depends on anything beyond the
canonicalized key and `kernel_params`, persistence is a correctness bug rather
than an optimization. **Presents as:** correct results on the first solve and
drifting results after a `rebalance` — which the Laplace-solve gate, a single solve,
would not catch. **Do:** T9's exit criterion requires a re-solve after
`invalidate_interaction_list()`; extend it to assert the potential is unchanged
across that re-solve. Any basis whose operator depends on particle positions must
declare so and opt out of the cache.

**R6 — `sets_per_component != 1` breaks the shared-cell Allreduce.** The two
hand-rolled pack/unpack loops (`:1774-1779`, `:1796-1801`) index by a running
counter, which is easy to get wrong when a third factor enters.

**Shared cells are not a multi-rank phenomenon.** A cell is shared when
`depth <= _replication_depth && !is_leaf` (`src/Canopy_CommunicationPlan.hpp:698`)
— a function of the tree alone, with no rank-count condition — and
`allreduce_shared_locals_at_depth` is called unconditionally
(`src/Canopy_DownwardSweep.hpp:2050`), returning early only on `nshared == 0`
(`:1776-1777`). At the frozen configuration np=1 has 1 shared cell at depth 0, 8
at depth 1 and 3-4 at depth 2 — the same counts as np=2.

**What np=1 does instead is make the function an exact identity, structurally.**
At one rank `MPI_Allreduce` copies send to recv, so the unpack computes
`Snap[k] + (L(j) - Snap[k])`, which is `L(j)`. The snapshot terms cancel whatever
they hold, and the slot expression `k` cancels with itself however wrong it is.
So at np=1 `_locals` is unchanged by this function **whenever its own pack and
unpack agree** — this does not depend on the snapshot being zero, and it is why
the pack-side-only perturbation T1 ran does corrupt np=1 while a both-sides one
does not.

Let $\sigma$ be the slot map the loops actually realize. Four classes follow, and
they are not equally dangerous:

| Class | np=1 | np ≥ 2 | Caught by |
| --- | --- | --- | --- |
| pack and unpack disagree | broken | broken | everything, by seven orders of magnitude — a one-slot rotation of the pack side alone measures $2.6\times10^{-2}$ against a $5.6\times10^{-10}$ cross-rank and a $9.63\times10^{-7}$ direct-sum tolerance |
| **aliasing** — $\sigma$ not injective | slot $A$ receives $L(B)$ | broken | `bitForBitArtifacts` at np 1-2 |
| **truncation** — the slot range misses a set | identity, correct | that set's M2L delta is never summed across ranks | `crossRankAgreement` at np 2-6 |
| $\sigma$ a bijection, applied consistently | identity | residue $(P-1)\big(S[j] - S[\sigma(j)]\big)$, and $S \equiv 0$ | nothing |

**The one invisible class is also the one that does not matter.** A bijection
applied uniformly is an arbitrary internal relabeling: `_shared_snapshot_buf`
(`src/Canopy_DownwardSweep.hpp:414`) is private and read only by these two
functions, so no correctness claim rests on which slot holds which coefficient.
The two classes that corrupt an answer — aliasing and truncation, which are the
likely slips when a third factor enters — are each already covered, one at np=1
and one at np ≥ 2. $S$ is identically zero at every shared depth at this
configuration, because no pair is MAC-admissible at depth 0 or 1 under
$\theta = 0.5$; that is what empties the fourth row, and it is not worth
engineering around.

**Distinguished from R1** by which checks fail: R1 moves `locals()` at np 1-2
with the operator table, $A_{n,m}$ table and key list all matching, and leaves
`matchesDirectSum` passing, because it is a reassociation-level difference. A
disagreeing or aliasing packing error fails `matchesDirectSum` too, four orders
of magnitude above truncation.

**Do:** the coverage above is a property of the gate, not something to add tests
for. What is worth building is in T10, steps 2 and 5: one slot expression shared
by all three loops, so "pack and unpack disagree" cannot be written; and a
slot-coverage assertion that $\sigma$ is a bijection onto the full slot range,
which is what separates the inert relabeling from aliasing and truncation. Do not
reach for a perturbation meant to fail at np ≥ 2 while sparing np=1 — no slot
offset behaves that way, for the identity reason above. Do not pin the exchange
buffer's bytes as a further bit-for-bit artifact either: it would lock an internal
ordering that T10 legitimately changes, so it would fire on T10 by construction,
be re-baselined, and catch nothing.

**R7 — the realized key count makes a compressed-operator basis unbuildable.**
Not a risk to this abstraction, but to whether it is worth building. If the
"~16 k under MAC=0.5" figure (`:36-42`) is right, an uncompressed $n^3\times n^3$
operator at $n=6$ costs roughly 17.6 TB per rank before the key even carries a
level. **Presents as:** T8's instrumentation reporting a key count in the
thousands. **Do:** record the measured number in the log against T8. If
compression at every usable order cannot fit in available memory, the black-box
basis is not buildable here, and the correct response is to build T12 standalone
— which this task sequence already supports, since T12 depends on nothing that a
black-box basis uniquely needs.

**R8 — the FMM answer is not partition-independent, and `crossRankAgreement` has
no valid reference.** The cross-rank half of the gate rests on the claim that
only the summation order moves when the partition moves: the interaction set is a
function of the tree, shared cells run their M2L on rank 0 alone
(`src/Canopy_DownwardSweep.hpp:701-708`), and every non-shared target has exactly
one owner. If some quantity in the pipeline is in fact a function of ownership —
a per-rank truncation, a locally-derived bound, an accumulation over locally-held
cells only — then the np=$k$ field differs from the np=1 field by far more than
reassociation and the test has no stable reference at all. **Presents as:** T1's
measurement step finding a deviation above $10^{-9}$ at some rank count on
unmodified code, before any refactor has happened, **and that deviation still
being present at `num_steps = 1`**. Crossing the threshold over the full
interval but not at one step is the integrator amplifying reassociation, not an
answer that depends on the partition. Measured at `num_steps = 1` on unmodified
code the deviation is $10^{-15}$ on the potential and $3\times10^{-13}$ on the
gradient at every rank count from 2 to 6, so this risk does not fire at the
frozen configuration; it stays recorded because T4 and T10 rewrite the dataflow
the measurement covers. **Distinguished from a genuine packing bug** by exactly that
timing: this fires on the current tree, a packing bug fires only after T4 or
T10. **Do:** T1's stop-and-report clause exists for
this. Do not raise the tolerance to accommodate it — a partition-dependent answer
is a defect in the parallel FMM, and it must be found and recorded before the
cross-rank gate is relied on. If it proves real and unfixable within T1, the
np 3-6 regime falls back to `matchesDirectSum` alone, and T4 and T10 lose their
tight multi-rank gate — which is a material weakening of this document's
verification strategy and belongs in the log.
