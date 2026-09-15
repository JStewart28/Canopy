# A Cartesian-Taylor far-field basis for Canopy — progress log

Session record for cartesian-taylor-basis. Companion to
[cartesian-taylor-basis.md](cartesian-taylor-basis.md), which holds the design,
the task sequence and the risks; this file holds what actually happened, in
order.

**Read this when** you need the reasoning behind a decision the design states
flatly, the measured numbers behind a claim, or the history of a file you are
about to change. The design says *what is true now*; the log says *how it got
that way and what was tried on the route*.

**Append to it** at the end of any task that makes a decision, changes a
signature, measures something, or finds a bug. Add a new `## <task ID>` section
at the bottom, named for the task it records, so
[cartesian-taylor-basis.md](cartesian-taylor-basis.md) can cite it by ID. No
dates: the order of the sections is the chronology. If a session covers more than
one task, name them all; if it belongs to no task, name the topic.

**End each section with `**Affects:**`** — the later task IDs whose stated plan
this entry changes, one clause each on how, or `none`. A finding that invalidates
a later task is worthless if the session starting that task has to read the whole
log to notice it; this line is the index that makes it findable.

Worth recording, because none of it is recoverable from the code afterwards:
semantic decisions and what forced them, signature changes and why they could not
stay as they were, bugs that only running revealed, measured numbers, and
approaches tried that did not work. Record too where the implementation departed
from the task's stated **Do** steps, and why — a task marked `**DONE**` that was
done differently than it was written is the quietest way for a design to stop
describing the code.

Four things this basis in particular will want back later, so record them where
they arise:

- **Every perturbation outcome the exit criteria call for.** T1, T2, T3 and T4
  each name a failure direction — a wrong recurrence coefficient, a broken trait,
  a dropped $(-1)^{|q|}$, `LaplaceKernel` at `near_softening_factor = 0`. Each is
  evidence that a check has teeth, and each is cheap to run once and impossible
  to reconstruct afterwards from a passing suite.
- **The achieved accuracy at both $\theta$, with the full configuration it was
  measured at.** The pinned constants live in the test, but the configuration
  behind them — particle count, `ncrit`, `max_depth`, `replication_depth`,
  `softening`, rank counts, flux job — belongs here.
- **Every `Solver` member body that turned out to assume something
  `LaplaceKernel`-specific** (**R7**). T4 is the first work that instantiates
  them on another basis, and that list is what the next basis author needs.
- **The cache figures across a drifting bounding box** (**R6**):
  `m2l_op_keys_built_count()` per rank per build, against T9's measured zero for
  a level-blind basis. T5 exists to produce exactly this.

## Session 0 — the design

Written before any code. Recorded here because three decisions in
[cartesian-taylor-basis.md](cartesian-taylor-basis.md) are stated flatly there and
their reasoning is not reconstructible from the code.

### The target is the regularization unblock, not $10^{-10}$

`canopy-questions.md` §5 is the governing statement: the blob correction dies as
$(\delta/R)^{2(k+1)}$, so a low order beats the regularization problem at standard
admissibility, and that is what makes `near_softening_factor = 0` viable. The
downstream solver's R-B requirement of a tunable $10^{-10}$ is **not** in scope
for this basis and no task pursues it — at 0.24 to 0.48 decades per order against
$\binom{p+3}{3}$ DOF (`canopy-kernel-rec.md`, "Convergence per DOF"), $10^{-6}$
alone wants $p \approx 11$–$24$. That band is black-box FMM's, and
`canopy-bbFMM.md` holds its design. The two claims — "$p=2$ is enough" and
"Cartesian Taylor cannot reach $10^{-6}$" — are not in tension; they are about
different problems.

$p$ stays the existing `P_ORDER` template slot on `Solver`, already documented as
the basis's order knob. No new template parameter.

### The accuracy bar, and why it is measured at two $\theta$

The reference treecode's far-field solve is documented at **~1e-3 relative
velocity at `theta = 0.3`**, at **`order = 2`** — moments G/D/Q, i.e. orders
0/1/2. Both figures were read from the reference repository at
`~/research-bridges/zmodel-steve/zmodel3d-amr/`: the order from `treecode.py:103`
and `_expansion_batch` at `:56-79`, the accuracy from `README.md:71` and
`PHYSICS.md:137`, with the measured table (5.9e-4 / 1.1e-3 / 9.4e-4 at
N = 642 / 2562 / 10242) in `PARALLELIZATION.md:24-30`. This confirms
`canopy-questions.md` §2 and §5 rather than merely being consistent with them.

**The reference's admissibility is not Canopy's**, and this is why T4 runs two
arms rather than one. The reference uses a Barnes-Hut radius MAC,
`node.radius < theta * s` at $\theta = 0.3$ (`treecode.py:99`, `:121`); Canopy
uses $R^2\theta^2 > 3(w_a+w_b)^2$ at $\theta = 0.5$
(`src/Canopy_CommunicationPlan.hpp:340-346`), which admits pairs the reference
would refuse. Taylor truncation goes as $(c\,w/R)^{p+1}$, so the same order does
not deliver the same accuracy under the two. Asserting 1e-3 at $\theta = 0.5$
would be holding the basis to a bar the reference never claimed at that
admissibility, and an implementing session would have no way to tell a real
defect from an admissibility difference. So the pass condition is 1e-3 at
$\theta = 0.3$ — matched admissibility, an apples-to-apples claim against the
reference — with the $\theta = 0.5$ value measured and **pinned** beside it the
way `LS_DIRECT_SUM_TOL` is, so a later regression is visible rather than merely
inside a loose bound.

### The order-2 oracle is in-repo, deliberately

`treecode.py` and `_expansion_batch` appear nowhere in this repository. Rather
than make the test depend on a file that is not here, the oracle for $|k| \le 3$
is the closed-form tensor set in `canopy-questions.md` §2, hand-coded in
`tests/tstCartesianTaylor.hpp`, with the §3 multi-index recurrence checked against
it — §3 itself states the recurrence reproduces $b_\emptyset$, $b_{e_i}$ and
$b_{2e_i}$, and T1 turns that statement into an assertion. Above $|k| = 3$ the
available oracle is a Richardson-extrapolated finite difference of $\varphi$.

### Two facts about the reference that are easy to get wrong

- **Its `blob` is not $\varepsilon$.** `blob` is the quantity added to $r^2$, and
  at the solver's default `use_matlab_blob = True` it equals `eps`, not `eps**2`
  (`zmodel3d/mesh_solver.py:394`). Canopy's `M2LKernelParams::softening` is a
  **length** whose square is $b$. Anything comparing a Canopy number against a
  reference number must match $b$, not $\varepsilon$.
- **Its `theta` default is 0.3 and its `ncrit` is 64** (`treecode.py:99`, `:104`),
  neither of which matches Canopy's frozen gate configuration.

### Three-scalar-passes, not a vector kernel

`canopy-questions.md` §4 peels the cross product off: with
$[K\times\gamma]_i = \varepsilon_{ilm}K_l\gamma_m$ and $K_l = -\partial_l\varphi$,
the far field is three scalar-$\varphi$ passes, one per strength component,
recombined by $\varepsilon_{ilm}$ at the end. Canopy's `NComps` already provides
those three independent passes, so the basis needs no vector machinery, and the
$\varepsilon$ recombination belongs to the downstream solver and is out of scope.

### `m2l_post_cell` is not a no-op

Recorded because an earlier coarse statement of this work said both per-cell M2L
hooks were no-ops, and that is true of only one of them. `m2l_pre_cell` **is** a
no-op — this basis contracts the source multipole directly and has no per-source
work to hoist, so the half-placed-hook gap recorded at T3 of
`abstract-solver-backend-progress-log.md` does not bite here. But
`m2l_post_cell` is **the only stage that writes to the locals view** and carries
the accumulator flush; both existing bases write it that way
(`src/Canopy_LaplaceKernel.hpp:1155-1195`,
`tests/CanopyTest_MonopoleBasis.hpp:785-810`). A no-op `m2l_post_cell` compiles
and leaves every local zero, which no compile check would catch.

### What was read, and what was deliberately not

Read in full: `tests/CanopyTest_MonopoleBasis.hpp`, `src/Canopy_FarFieldContract.hpp`,
`canopy-questions.md`, `treecode.py`. Read in part:
`src/Canopy_LaplaceKernel.hpp` (the contract block, the M2L key and
operator-builder block, the three M2L stages, `l2p_evaluate`),
`tests/tstFarFieldContract.hpp`, `tests/tstLaplaceSolve.hpp`'s tolerance block
and direct-sum body, both sweeps' guards, setters, counters and operator call
sites, `src/Canopy_Solver.hpp`'s `FarField` slot and the two push helpers,
`tests/CMakeLists.txt`, `src/CMakeLists.txt`,
`cmake/test_harness/test_harness.cmake`.

**Not read, deliberately:** `src/Canopy_P2P.hpp` beyond confirming it squares a
softening length into `_softening2` and applies it as `r^2 + eps2` (`:103-106`,
`:799-803`, `:1082`) — no task changes it; `src/Canopy_TreeBuilder.hpp` and
`src/Canopy_TreePartitioner.hpp` beyond the partitioner's documented
non-determinism; the interiors of `run_m2l_fused` and the shared-cell Allreduce,
which no task here edits and whose contract with a basis is fully stated by the
three stage signatures and the raw-byte scratch. T4 is the first task that could
need any of the three, and only if a `Solver` member body turns out to assume
something `LaplaceKernel`-specific (**R7**).

### Line numbers measured against the working tree

Stated here because three of them were carried forward stale from earlier
documents and a later reader may meet the old values elsewhere:
`src/CMakeLists.txt` `HEADERS_PUBLIC` is `:3-18`; `tests/CMakeLists.txt`
`UNIT_MPI_TESTS` is `:48-58` and `UNIT_SERIAL_TESTS` is `:36-39`; the finite
difference `l2p_evaluate` replaces is `src/Canopy_LaplaceKernel.hpp:1378-1407`,
inside `l2p_evaluate` (`:1327-1332`); `createSolver` is
`src/Canopy_Solver.hpp:816-825`.

**Affects:** **T1** — the oracle is `canopy-questions.md` §2 hand-coded in-repo,
not the reference treecode, so T1 needs no file outside this repository. **T3** —
the $R$ sign is the load-bearing one: the sweep's key offset is
$c_{\rm source}-c_{\rm target}$ (`src/Canopy_DownwardSweep.hpp:1464-1481`) while
§4's $R = c_A - c_B$ is the reverse, so $R = -(ii,jj,kk)\cdot w_{\rm unit}$; and
the $1/q!$ belongs in the moment, never in $b_k$. **T4** — the pass condition is
1e-3 at $\theta = 0.3$ with the $\theta = 0.5$ value pinned alongside, and
`softening` must be set explicitly because `FmmConfig::softening = -1.0` selects
auto-softening whose effective $\varepsilon$ moves with the distribution.

## T1 — the index map and the derivative ladder

Implemented `src/Canopy_CartesianTaylorBasis.hpp` (slot map + $b_k$ evaluator,
nothing else), `tests/tstCartesianTaylor.hpp` (three bodies), one line in each
of the two `CMakeLists.txt`, and
`scripts/tuolumne/run_ctest_cartesian_taylor_serial.flux`.

### The total order on multi-indices

**Degree-graded, then ascending lexicographic in $(k_x, k_y)$** — with
$k_z = |k| - k_x - k_y$ determined, so the pair fixes the triple. Stated on the
declaration in the header, which is where the reasoning belongs; repeated here
because the *rejected* alternative is not recoverable from the code.

Degree-graded is the load-bearing half and downstream code depends on it. The
M2L needs $b_{p+q}$ out to $|p+q| = 2p$ while the moments only run to
$|q| \le p$, so under a graded order the order-$p$ slot table is a **prefix** of
the order-$2p$ table and one flat index is valid in both. A non-graded order
(plain lexicographic on the triple, say) would need two maps and a translation
between them at every M2L contraction. It is also why `slot()` takes no order
argument: the degree is read off the multi-index itself and the answer does not
move when $p$ changes, which is what makes the prefix property checkable rather
than merely intended. The bijection test asserts it directly — every slot lands
inside its own degree block.

The within-degree half is arbitrary. It is pinned only so that it is written
down: ascending in $k_x$, then in $k_y$. Closed forms:

$$
\mathrm{slot}(k) = \binom{n+2}{3} + k_x (n+1) - \frac{k_x(k_x-1)}{2} + k_y,
\qquad n = |k|
$$

with $\binom{n+2}{3}$ the count of multi-indices of degree $< n$ and
$\mathrm{num\_slots}(p) = \binom{p+3}{3}$. The inverse walks the degree and then
the $k_x$-block; both walks are $O(|k|)$ and no table is built.

### What the §3 recurrence turned out to need that §3 does not say

Three things, all of which had to be decided to turn the formula into a table
fill:

1. **§3 gives no base case.** It produces $b_{k+e_i}$ from lower orders and
   stops. The base is §1's $b_\emptyset = P_0 = w^{-1/2}$.
2. **§3 does not say which $i$ to step in.** It states the recurrence for
   $b_{k+e_i}$, but filling a table means going the other way: for each target
   multi-index $m$, choosing a decomposition $m = k + e_i$. Any $i$ with
   $m_i > 0$ gives the same $b_m$; the implementation takes the lowest so the
   sweep is deterministic and the same value comes out on host and device.
3. **"Any $b$ carrying a negative component is identically zero" never needs to
   be implemented as a lookup.** Every term whose index would go negative has a
   coefficient that vanishes on *exactly* the same condition, so guarding on the
   coefficient is exact rather than an approximation and no zero-padded table is
   required. Term by term, with $j \ne i$ unless stated:
   - $-k_i b_{k-e_i}$: negative iff $k_i = 0$, and the coefficient is $k_i$.
   - $-2 k_j r_j b_{k+e_i-e_j}$: component $j$ is $k_j - 1$, negative iff
     $k_j = 0$, and the coefficient is $k_j$. At $j = i$ the index is $k$ itself
     and is never negative.
   - $-k_j(k_j{-}1) b_{k+e_i-2e_j}$: component $j$ is $k_j - 2$, negative iff
     $k_j \le 1$, and the coefficient is $k_j(k_j-1)$. At $j = i$ the index is
     $k - e_i$, negative iff $k_i = 0$, and the same coefficient vanishes.

   This is worth having written down: a reader who implements the "≡ 0" clause
   literally will allocate a padded table and a bounds-checked accessor for
   nothing.

Consequence of (3) plus §3's own "orders $|k|$ and $|k|-1$" statement — which
holds including the $j = i$ term of the first sum, where the index is $b_k$
itself at degree $|k|$ — is that **one forward sweep in ascending degree fills
the table in place with no scratch buffer**. That is what makes the evaluator
device-callable with a caller-provided array and no allocation.

### Signatures

```cpp
namespace Canopy::CartesianTaylor {
  KOKKOS_INLINE_FUNCTION constexpr int slot_degree_base( int n );      // C(n+2,3)
  KOKKOS_INLINE_FUNCTION constexpr int num_slots_at_degree( int n );   // C(n+2,2)
  KOKKOS_INLINE_FUNCTION constexpr int num_slots( int p );             // C(p+3,3)
  KOKKOS_INLINE_FUNCTION constexpr int slot( int kx, int ky, int kz );
  KOKKOS_INLINE_FUNCTION void inverse_slot( int s, int k[3] );
  KOKKOS_INLINE_FUNCTION void derivative_ladder( const double r[3], double b,
                                                 int max_order, double* out );
}
```

`b` is **softening squared**, the quantity added to $r^2$ — not $\varepsilon$.
`b <= 0` calls `Kokkos::abort` with a message naming the convention; it is a
precondition, not a defaultable argument, because $b = 0$ at $r = 0$ divides by
zero. `out` is caller-provided and holds **raw** $\partial^k\varphi$ with no
$1/k!$. Nothing in the file is a contract member: no trait, no typedef, no
`static_assert` on the basis, no operator, and no sweep or `Solver` is
instantiated.

### Decisions carried in from the task statement

- **The finite-difference check runs at $|k| = 4$ only** — that is $2p$ at the
  $p = 2$ every later task uses. The bijection assertion still runs at orders 0
  through 6 because it is cheap and it is where hand-derived Cartesian FMMs
  actually break (**R2**). The FD oracle is deliberately *not* extended to
  match: its tolerance is scale-dependent at each order, and **R8** requires it
  be re-measured rather than assumed if $p$ is ever raised.
- **The `_valgrind` CTest variant is not a gate.** `Canopy_add_tests` registers
  `Canopy_Test_CartesianTaylor_SERIAL_valgrind` beside the real test because
  valgrind is found in `build-tuolumne/`
  (`cmake/test_harness/test_harness.cmake:157-162`). The batch script's `-R` is
  anchored, `'^Canopy_Test_CartesianTaylor_SERIAL$'`, which is what excludes it.
  It was neither made to pass nor disabled.

### Measured: the sampled $(r, b)$ and the tolerances they hold at

40 samples. $b \in \{10^{-6},\ 6.25\times10^{-4},\ 10^{-2},\ 1\}$ — the second
is the downstream solver's $\varepsilon = 0.025$ squared, with the set spanning
roughly three decades either side of it. For each $b$: $r = 0$ exactly (legal,
because $b > 0$), and $|r| / \sqrt b \in \{0.01,\ 1,\ 100\}$ in three
directions — axis-aligned $(1,0,0)$ so the $\delta_{ab}$ terms of §2 stand
alone, a generic unit vector $(0.36, -0.48, 0.80)$, and the diagonal
$(1,1,1)/\sqrt3$. The $b \to 0$ limit is not sampled; it is a precondition
violation, not an edge case.

| Check | Tolerance | Achieved (worst over all 40 samples) | Margin |
| --- | --- | --- | --- |
| §2 closed forms, $\vert k\vert \le 3$ | $10^{-12}$ | $2.911\times10^{-15}$ at $k = (3,0,0)$ | 343× |
| Finite difference, $\vert k\vert = 4$ | $10^{-5}$ | $7.321\times10^{-7}$ at $k = (4,0,0)$, $b = 6.25\times10^{-4}$ | 13.7× |

Green flux jobs: `f3YTi9NMzAE3` (first clean run) and `f3YTwT7yiC2K` (re-run
after both perturbations were reverted, on the exact tree committed at the
checkpoint). Both 3/3 bodies, ctest rc 0, identical figures.

Both errors are measured against the **natural scale of a $|k|$-th derivative**,
$\varphi / L^{|k|}$ with $L = \sqrt w$, not relative to the value itself.
Relative-to-value is unusable here: individual components vanish identically at
the sampled $r$ (any $r_a = 0$ kills the odd terms) and the test would divide by
zero. This is stated on both assertions.

**The FD oracle needed two Richardson steps, not one.** The first implementation
used the single step the design describes, $R_1(h) = (4D(h/2) - D(h))/3$, at
$h = L/64$. A standalone replica of the map, the recurrence and the oracle —
written to size the tolerance before spending a queue slot — showed that
configuration achieving $7.8\times10^{-6}$ against a $10^{-5}$ tolerance: a
1.3× margin, sitting exactly on the truncation/roundoff crossover. Widening the
tolerance would have been the wrong fix, because this check is the *only* oracle
above $|k| = 3$ and its sharpness is what bounds how small an index-map or
recurrence error has to be to slip through (**R2**, **R8**). Sharpening the
oracle instead: a second Richardson step,
$R_2(h) = (16 R_1(h/2) - R_1(h))/15$, kills the $h^4$ term and leaves $O(h^6)$.
The composed central stencils carry even powers of $h$ only, so the second step
is valid for the same reason the first is.

Divisor scan at $|k| = 4$, worst case over the whole sample set, two steps:

| $h$ | worst | regime |
| --- | --- | --- |
| $L/8$ | $1.3\times10^{-4}$ | truncation-limited, falling as $(h/L)^6$ |
| $L/16$ | $1.9\times10^{-6}$ | |
| $L/32$ | $4.3\times10^{-7}$ | **the floor** — truncation and roundoff balanced |
| $L/64$ | $7.2\times10^{-6}$ | roundoff-limited, rising as $(L/h)^4$ |

$h = L/32$ is what shipped. The measured $7.3\times10^{-7}$ in the table above
is the on-machine figure and sits just above the replica's $4.3\times10^{-7}$,
the difference being `-ffp-contract` and libm. $4\times10^{-7}$ is about as
good as a double-precision 4th-derivative difference gets; 13.7× is therefore
the real margin available, not a number that can be improved by tuning $h$.

Nondimensionalizing the step as $h = L/32$ rather than fixing it absolutely is
what lets one tolerance hold across three decades of $b$ and four of
$|r|/\sqrt b$: $L = \sqrt{r^2 + b}$ is the scale $\varphi$ actually varies on at
every sampled point.

### Both perturbations

Each was built and run as its own flux job, and each was reverted by inverting
the edit rather than by `git checkout`, which would have discarded the task's
uncommitted work.

**Perturbation A — the §3 coefficient $-2\sum_j k_j r_j \to -\sum_j k_j r_j$**
(job `f3YTv3AMK12P`, ctest rc 8). `closed_forms` and `finite_difference` both
failed; `index_map_bijection` still passed, correctly, since the map is
untouched. First failure, naming the multi-index:

```
canopy-questions.md §3 recurrence disagrees with the §2 closed form at
multi-index (2,0,0), |k| = 2: recurrence -999650068.7390641, closed form
-999550093.73468971, |diff| / (phi/L^|k|) = 9.999e-05 > 1e-12;
at r = (1e-05,0,0), b = 1e-06
```

This is exactly the predicted direction: $|k| \le 1$ is untouched because the
perturbed sum is empty at $k = 0$, and the error appears at $|k| = 2$. Worth
noting how *small* it is — $10^{-4}$ of scale, four decades above the tolerance
but nowhere near an obvious blow-up. A test built on a loose relative tolerance
would have missed it.

**Perturbation B — `inverse_slot`'s $k_y$ and $k_z$ swapped** (job
`f3YTvodPfdYT`, ctest rc 8). All three bodies failed. First failure:

```
inverse_slot( slot(0,0,1) = 1 ) gave (0,1,0), expected (0,0,1)
```

`closed_forms` then failed at $(0,0,1)$ and `finite_difference` at $(0,0,4)$,
because `derivative_ladder` uses `inverse_slot` to walk each degree and so
inherits the break. That coupling is worth recording for whoever perturbs this
next: the map and the ladder are not independently testable in this
implementation, and a map failure will always present as three red bodies rather
than one.

### An optimization not taken

`derivative_ladder` calls `inverse_slot` once per slot and `slot` up to seven
times per slot, all recomputed on every call. In the M2L inner loop this is per
box pair. A precomputed table of $(m, i, k, \text{term slots})$ built once on
host would remove all of it. Not done: it is a performance refinement, not a
correctness issue, and T1 declares no contract member to hang a host-built table
off. Flagged here rather than in `README.md` "Future Optimizations" because the
call site that would pay for it does not exist until T2.

**Affects:** **T2** — the signatures above are what to build the contract on:
`slot`/`inverse_slot`/`num_slots` are free functions in
`Canopy::CartesianTaylor` taking no order parameter, and `derivative_ladder`
takes `( const double r[3], double b, int max_order, double* out )` with `b`
softening *squared* and `out` caller-provided of length `num_slots(max_order)`.
The graded order makes `num_coeffs_per_cell = num_slots(P_ORDER)` and the M2L's
`num_slots(2*P_ORDER)` table share one index, so no second map is needed. T2
adds the traits and operators; it should not need to change anything in this
file. **T3** — the $b_k$ are raw $\partial^k\varphi$ with no $1/k!$, as the
header states, so the $(-1)^{|q|}$ and the factorial placement T3 pins are
entirely T3's business and nothing here pre-empts them. **T4** — if accuracy
misses at $p = 2$, re-run this test's `finite_difference` body at the specific
$(r, b)$ scales T4 uses before touching the basis (**R2**); the 13.7× margin is
measured at the scales tabulated above and nowhere else. **T2, T3** — the flux
script `scripts/tuolumne/run_ctest_cartesian_taylor_serial.flux` is reusable
unchanged; its `-R` is anchored to exclude the valgrind variant.

## T2 — the contract surface and the kernel-blind operators

Added `Canopy::CartesianTaylorBasis<Scalar, P_ORDER, NComps>` to
`src/Canopy_CartesianTaylorBasis.hpp` — the whole contract surface plus four
complete operators — and four bodies to `tests/tstCartesianTaylor.hpp`. Nothing
in T1's `Canopy::CartesianTaylor` namespace changed, as T1's **Affects:** line
predicted; the class is built on those free functions and adds nothing to them.
One README entry and three document corrections ride in the same commit.

Flux jobs: `f3YUdrmXQa1d` (first clean run, 7/7, ctest rc 0) and `f3YUfS9vvt9m`
(re-run after both perturbations were reverted, on the exact tree committed at
the checkpoint — 7/7, rc 0, byte-identical figures).

### Decisions carried in from the task statement

- **`build_aux_tables` returns an empty `aux_tables_type`**, as `MonopoleBasis`
  does. T1's multi-index tables are *not* precomputed here even though step 3
  leaves the choice open. The only call sites that would pay for a table are
  `m2l_core` and `build_m2l_operators`, both of which abort until T3 — so a
  table built now has no consumer and no way to be measured against the
  recompute it replaces, and **R3** is the reason it must be measured rather
  than assumed. Tracked in `README.md` "Future Optimizations" instead; the
  declaration on `aux_tables_type` says so and says why.
- **The accumulator layout is fixed in T2**, not deferred. See below.
- **`num_coeffs_per_cell` and `m2l_num_src_coeffs` are both spelled
  `Canopy::CartesianTaylor::num_slots( P_ORDER )`**, T1's own already-`constexpr`
  function, which already equals $\binom{p+3}{3}$. No second binomial helper was
  written: two spellings of one count is how the M2L's shared flat index stops
  being shared.
- **The compile-only body instantiates at `P_ORDER = 2, NComps = 3`** (T4's and
  the downstream solver's shape) and at `NComps = 1` beside it (the existing
  `Solver` call sites' shape).
- **The README edit rides in this task's checkpoint commit.**

### The M2L accumulator layout, and why it had to be settled here

```
n_acc = num_coeffs_per_cell * NComps * sets_per_component     (= 10 * 3 * 1)
acc_slot( out_idx, c, s ) = out_idx * num_comp_slots + comp_set_slot( c, s )
```

**Coefficient-major, then component, then set.** Two reasons, and neither is
"MonopoleBasis does it":

1. It is the **same nesting as the locals view** `(cell, coeff,
   comp_set_slot)`, which makes `m2l_post_cell` a slot-for-slot walk of one
   cell's slice rather than a transpose. That matters because `m2l_post_cell`
   runs once per target cell per team and a transposed flush would stride the
   locals view on its fastest axis.
2. It is the nesting **T3's contraction wants**. The M2L reads
   `M_full(source_cell, q_slot, c)`, whose component axis is likewise innermost,
   so with this layout the component loop is stride-1 on *both* sides of the
   multiply-accumulate.

Fixing it in T2 was forced, not chosen: `m2l_post_cell` is written in this task
and it **reads** the accumulator. `m2l_scratch_bytes` and `acc_slot` therefore
both had to be settled here even though `m2l_core`, which fills the scratch,
aborts. Deferring the layout to T3 would have left the flush indexing something
whose shape nobody had decided. **T3 does not get to choose this** — `acc_slot`
is the whole of the contract between the two stages.

At `sets_per_component = 1` the expression collapses to
`out_idx * NComps + c` and `comp_set_slot(c, 0) == c` exactly, which is the
`Deliberate deviations` claim made concrete. The set factor is carried through
`comp_set_slot`/`acc_slot` anyway so that a later set count changes one
expression rather than six.

### The single-source-of-truth functions T2 leaves for T3

Two, both `KOKKOS_INLINE_FUNCTION static` on the basis:

- **`taylor_monomials( dx, dy, dz, t[num_coeffs_per_cell] )`** — fills
  `t[slot(k)] = d^k / k!` for every $|k| \le p$. **All four** kernel-blind
  operators are this table contracted against a coefficient array (P2M against
  a charge, M2M and L2L against a coefficient sum, L2P against the local), so
  it has exactly one home. Built as an outer product of three per-axis running
  products `pf[a][j] = d_a^j / j!`, so each entry costs two multiplies and no
  `pow()` or factorial division appears in an inner loop. **The $1/k!$ is here
  and nowhere else on these paths** — not in the $b_k$, which are raw
  derivatives, and not applied a second time by any caller.
- **`taylor_accumulate( acc, a, b )`** — the one multiply-accumulate step,
  written as two statements with the product in a named local, per the
  Conventions row on `-ffp-contract`. `MonopoleBasis::m2l_accumulate` is the
  same device.

T3's own single source of truth — the M2L operator entry — is a *third*
function and does not exist yet. It should be written the same way, and
`taylor_accumulate` is already there for it to use.

### Where the contract as documented did not match what the sweeps demanded

Three, all now corrected in `tasks/cartesian-taylor-basis.md` as part of this
task:

1. **T2's `**Fill in**` line contradicted its own steps 4 and 5.** It said the
   task fills everything "except the three M2L stages' bodies and
   `build_m2l_operators`'s body", which would have deferred `m2l_post_cell`.
   No later task's `**Fill in**` names `m2l_post_cell` either, so following the
   summary literally would have left it unwritten **permanently** — and that is
   exactly the failure `Deliberate deviations` records: a no-op `m2l_post_cell`
   compiles cleanly and leaves every local coefficient zero, because it is the
   only stage that writes the locals view. Reworded to name `m2l_core`'s body
   and `build_m2l_operators`'s body only.
2. **Step 6 used macros that do not exist in a SERIAL unit test.**
   `TEST_MS`/`TEST_ES` are template parameter names local to
   `tests/tstFarFieldContract.hpp`'s fixtures. The macros the category header
   defines are `TEST_MEMSPACE` and `TEST_EXECSPACE`
   (`cmake/test_harness/TestSERIAL_Category.hpp:16-17`).
3. **`m2l_accumulator_type` is not what the contract preamble claims.** The
   preamble says every listed member is "reached by `UpwardSweep`,
   `DownwardSweep`, `P2P` or `Solver`, or is required to keep one of them
   well-formed." This one is neither: no sweep names it. Both existing bases
   declare it only to reinterpret the sweep's raw scratch bytes inside their own
   M2L stages (`src/Canopy_LaplaceKernel.hpp:330`,
   `tests/CanopyTest_MonopoleBasis.hpp:327`). The sweep's obligation stops at
   `m2l_scratch_bytes`, which is what it actually reads. Noted in the document
   as **basis-internal**.

Also corrected: the document's top-level `**Status:**` still read `NOT STARTED`
with T1 already DONE. Now `IN PROGRESS — T1 and T2 DONE; T3 next`.

One thing the document got **right** that a nearby comment gets wrong, worth
restating because copying the wrong one is a compile error deep in a sweep:
`build_aux_tables` takes **two** arguments, `( int order, const
Canopy::M2LKernelParams& )`, per the real call sites at
`src/Canopy_UpwardSweep.hpp:305` and `src/Canopy_DownwardSweep.hpp:1789`. The
comment at `tests/CanopyTest_MonopoleBasis.hpp:297` shows a one-argument
spelling and is stale.

### What the four operators are, and why every check hit the roundoff floor

`p2m_contribution` (atomic — one thread per particle, many particles per leaf),
`m2m_translate` and `l2l_translate` (non-atomic — one team per parent), and
`l2p_evaluate` with the **analytic** gradient, which deletes the central finite
difference at `src/Canopy_LaplaceKernel.hpp:1378-1407` *for this basis only*.
Note $|p - e_i| \le p - 1$, so the same order-$p$ shift table serves the
potential and the gradient; no second table and no larger one is needed.

Every width (`w_self`, `w_child`, `w_parent`) is ignored, and each declaration
**says so and says why** rather than silently dropping it: the solid-harmonic
basis divides by `w_self^{n+1}` because a scale-invariant kernel makes the
normalized operator depend on the offset alone, and a softened kernel has no
scale invariance — $b$ is a fixed $\mathrm{length}^2$ and does not rescale with
the cell. There is no normalization to divide out and the coefficients carried
here are physical.

All four are **exact rational arithmetic**, not approximations, which is why
every measured deviation below sits at the roundoff floor rather than at a
truncation level. That is the expected result and it is what makes these
tolerances meaningful: there is no truncation error for a loose bound to hide
inside.

| Body | Check | Worst, $p=2$ | Worst, $p=4$ |
| --- | --- | --- | --- |
| `m2m_shift` | P2M vs brute-force $\sum_j d_j^q/q!\,s_{jc}$, over $\max\lvert M\rvert$ | $7.547\times10^{-18}$ | $7.547\times10^{-18}$ |
| `m2m_shift` | M2M vs **direct P2M about the parent center** | $1.208\times10^{-16}$ | $1.208\times10^{-16}$ |
| `m2m_shift` | M2M round trip by $-s$ | $6.038\times10^{-17}$ | $6.038\times10^{-17}$ |
| `l2l_shift` | L2L round trip by $-s$ | $4.441\times10^{-16}$ | $8.882\times10^{-15}$ |
| `l2l_shift` | child expansion vs parent expansion at the same **physical point** | $8.882\times10^{-16}$ | $1.776\times10^{-15}$ |
| `l2p_evaluation` | potential vs brute-force $\sum_p a^p\ell_p/p!$ | $2.220\times10^{-16}$ | $1.388\times10^{-16}$ |
| `l2p_evaluation` | **analytic** gradient vs Richardson-extrapolated FD | $7.105\times10^{-15}$ | $2.576\times10^{-14}$ |

The three `m2m_shift` figures being *identical* at $p = 2$ and $p = 4$ is not a
sign that the $p=4$ instantiation silently ran at $p=2$ — the body asserts
`num_coeffs_per_cell == num_slots(P)`, which is 10 against 35. The worst
*absolute* deviation simply lands on a low-degree slot at both orders, because
the sampled offsets are below 0.5 and the $1/k!$ shrinks the high-degree
moments by orders of magnitude, so those slots contribute error far below the
degree-0 and degree-1 ones. The scale the ratio is taken against,
$\max\lvert M\rvert$, is the total charge at slot 0 in both cases.

### Choices inside the test bodies

- **Every body runs at two orders, $p = 2$ and $p = 4$.** At $p = 2$ several of
  these checks degenerate — the shift sums have very few terms — and a body that
  only ever ran at the shipping order would not notice a degree-dependent
  indexing error. $p=4$ costs nothing here (no tree, no MPI, 35 slots).
- **The oracles share no code with the basis.** The reference monomial
  $d^k/k!$ is brute-force repeated multiplication with an explicit factorial
  (`CT2::monoOverFact`), never `taylor_monomials`. Checking `taylor_monomials`
  against itself would pass under any *consistent* indexing error, which is the
  failure mode **R2** is about.
- **M2M is checked against a direct P2M about the parent center, not only by
  round trip.** The M2M is exact rather than truncated — expanding
  $(y - c_{\rm par})^q$ produces only terms at degree $\le |q| \le p$, so
  nothing is cut — which makes the moments of the *same particles* taken
  directly about the parent center an exact reference. A round trip alone
  cannot distinguish the right shift from a wrong-but-invertible one, and this
  is also what gives `p2m_contribution` real coverage: it is the only operator
  the task statement's step 7 does not name.
- **L2L is likewise checked against the polynomial it is supposed to
  preserve** — the shifted child expansion evaluated at a point, against the
  parent expansion at the *same physical point* — for the same reason, and
  because it exercises `l2p_evaluate` on a second input.
- **The gradient's FD oracle needed only one Richardson step**, unlike T1's
  ladder, which needed two. The reason is structural and worth writing down
  because it stops holding: $u$ here is a **polynomial of degree $p$**, so the
  central difference carries $h^2/6\,u''' + h^4/120\,u^{(5)} + \dots$ in which
  every term above the degree vanishes identically. One step kills the $h^2$
  term, so at $p \le 5$ the extrapolated difference equals the analytic
  derivative *in exact arithmetic* and the only residual is cancellation
  roundoff, $O(\varepsilon\,|u|/h)$. The tolerance is written against that scale
  — $10^3\varepsilon\,|u|/h$ — rather than as a pinned constant, precisely so
  that a later session raising $p$ above 5 sees the assumption stated. This is
  the opposite situation from T1's, where the oracle was the sharpness bottleneck.
- **`ASSERT_EQ( Basis::sets_per_component, 1 )` sits inside `l2l_shift`.** If
  that trait ever moves, the `(component, set)` flattening stops collapsing to
  `c` and this body is the first thing that must be revisited; the assertion is
  where a later session finds that out.

### Both perturbations

Each was built on the login node — both are compile-time failures, so neither
needed a job — and each was reverted **by inverting the edit**, not by
`git checkout`, which would have discarded the task's uncommitted work. The
tree was rebuilt clean and re-run green afterwards (`f3YUfS9vvt9m`).

**Perturbation A — `sets_per_component = 1` → `0`.** Fails at
`src/Canopy_DownwardSweep.hpp:154`, the guard the task statement names, reached
through `Solver`'s `downward_type` member:

```
/g/g20/stewartj/research-bridges/canopy-dev/Canopy/src/Canopy_DownwardSweep.hpp:154:20: error: static assertion failed due to requirement 'sets_per_component >= 1': DownwardSweep: the basis declares sets_per_component < 1, so its locals view would have a degenerate third extent and every local coefficient read would be out of bounds
  154 |     static_assert( sets_per_component >= 1,
      |                    ^~~~~~~~~~~~~~~~~~~~~~~
/g/g20/stewartj/research-bridges/canopy-dev/Canopy/src/Canopy_Solver.hpp:143:42: note: in instantiation of template class 'Canopy::DownwardSweep<Kokkos::HostSpace, Kokkos::Serial, Canopy::CartesianTaylorBasis<double, 2, 3>>' requested here
  143 |     using potential_view_type = typename downward_type::potential_view_type;
      |                                          ^
/g/g20/stewartj/research-bridges/canopy-dev/Canopy/tests/tstCartesianTaylor.hpp:1144:20: note: in instantiation of template class 'Canopy::Solver<Kokkos::HostSpace, Kokkos::Serial, double, 2, 3, Canopy::CartesianTaylorBasis>' requested here
 1144 |     static_assert( sizeof( Solver3 ) > 0,
      |                    ^
/g/g20/stewartj/research-bridges/canopy-dev/Canopy/src/Canopy_DownwardSweep.hpp:154:39: note: expression evaluates to '0 >= 1'
  154 |     static_assert( sets_per_component >= 1,
      |                    ~~~~~~~~~~~~~~~~~~~^~~~
```

4 errors total: the assert fires twice, once per `Solver` instantiation
(`NComps = 3` and `NComps = 1`), each followed by a cascade
`error: no type named 'gradient_view_type' in 'Canopy::DownwardSweep<...>'` —
clang marks the class invalid after a failed class-scope assert, so every later
member lookup on it fails too.

**Perturbation B — `scalars_per_coeff = 1` → `2`.** Fails at
`src/Canopy_UpwardSweep.hpp:74`, the guard the task statement names, reached
through `Solver`'s `_upward` data member:

```
/g/g20/stewartj/research-bridges/canopy-dev/Canopy/src/Canopy_UpwardSweep.hpp:74:20: error: static assertion failed due to requirement 'sizeof(double) == scalars_per_coeff * sizeof(double)': UpwardSweep: the basis's coeff_type is not scalars_per_coeff contiguous component_scalar_type, so the shared-cell Allreduce would transfer the wrong byte count
   74 |     static_assert( sizeof( coeff_type ) ==
      |                    ^~~~~~~~~~~~~~~~~~~~~~~
   75 |                        scalars_per_coeff * sizeof( component_scalar_type ),
      |                        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/g/g20/stewartj/research-bridges/canopy-dev/Canopy/src/Canopy_Solver.hpp:693:17: note: in instantiation of template class 'Canopy::UpwardSweep<Kokkos::HostSpace, Kokkos::Serial, Canopy::CartesianTaylorBasis<double, 2, 3>>' requested here
  693 |     upward_type _upward;
      |                 ^
/g/g20/stewartj/research-bridges/canopy-dev/Canopy/tests/tstCartesianTaylor.hpp:1144:20: note: in instantiation of template class 'Canopy::Solver<Kokkos::HostSpace, Kokkos::Serial, double, 2, 3, Canopy::CartesianTaylorBasis>' requested here
 1144 |     static_assert( sizeof( Solver3 ) > 0,
      |                    ^
/g/g20/stewartj/research-bridges/canopy-dev/Canopy/src/Canopy_UpwardSweep.hpp:74:41: note: expression evaluates to '8 == 16'
```

19 errors total. Three *distinct* static asserts fire, which is the interesting
part and not noise:

```
src/Canopy_CartesianTaylorBasis.hpp:383  CartesianTaylorBasis: coeff_type is not scalars_per_coeff contiguous component_scalar_type, ...
src/Canopy_UpwardSweep.hpp:74            UpwardSweep: the basis's coeff_type is not scalars_per_coeff contiguous component_scalar_type, ...
src/Canopy_DownwardSweep.hpp:124         DownwardSweep: the basis's coeff_type is not scalars_per_coeff contiguous component_scalar_type, ...
```

Two things follow. First, the "clang reports only the **first** failing
class-scope assert per class instantiation" rule the document's contract section
closes with is **per class**, not per translation unit: these three live in
three different classes and all three are reported. Second, the basis's own
assert firing alongside the sweeps' is what marks `CartesianTaylorBasis` invalid
and produces the trailing
`error: no member named 'aux_tables_type' in 'Canopy::CartesianTaylorBasis<...>'`
cascade in the *test* file. A future session perturbing a trait the basis also
guards should expect that noise and read past it to the sweep diagnostic — the
sweep guard is the one under test.

Neither perturbation was satisfied vacuously: both fired *through* `Solver`'s
data members, which is the whole point of the `sizeof( Solver ) > 0` body.

### Not done, deliberately

`build_m2l_operators`, `m2l_core` and `m2l_translate` all `Kokkos::abort` with a
message naming T3. `m2l_translate` aborts too, though step 5 names only the
other two: it is the `PerPairTranslate` fallback, and a silently-empty fallback
would drop every overflowing pair's contribution — the same defined-but-wrong
failure mode step 5 exists to prevent (**R4** makes it worse: the two paths
disagreeing decides the answer). Nothing calls any of them before T3; the
`sizeof` body instantiates class bodies, not member bodies.

**Affects:**

- **T3** — inherits three things it does not get to re-decide. (i) The
  accumulator layout `acc_slot(out_idx, c, s) = out_idx * num_comp_slots +
  comp_set_slot(c, s)`, coefficient-major, because `m2l_post_cell` already reads
  it; `m2l_core` must fill exactly that. (ii) `taylor_accumulate` already exists
  as the `-ffp-contract`-guarded multiply-accumulate — use it rather than
  writing a second one. (iii) The counts: `num_coeffs_per_cell` and
  `m2l_num_src_coeffs` are both `Canopy::CartesianTaylor::num_slots( P_ORDER )`,
  and the M2L's $b_{p+q}$ table is `num_slots( 2 * P_ORDER )` sharing the same
  flat index by T1's graded order, so **no second map and no translation between
  maps**. T3's own single-source-of-truth function for the operator entry is the
  third such function and does not exist yet. Also: all three aborting members
  are `Kokkos::abort`s to *replace*, not empty bodies to fill — the abort text
  naming T3 is how a wrong-but-running solve was made impossible, so removing an
  abort and leaving a partial body reopens exactly that hole.
- **T4** — **R7** stands untouched by this task's clean pass. The `sizeof` body
  instantiates class bodies only, so `Solver::solve()` has still never been
  compiled against a non-`LaplaceKernel` basis. Expect T4's first failures in
  `src/Canopy_Solver.hpp`, read them as expected, and record every `Solver`
  member body that turns out to assume something `LaplaceKernel`-specific. Also:
  `l2p_evaluate` writes `phi_out` with `=` and not `+=`, matching the sweep's
  accumulate-afterwards at `src/Canopy_DownwardSweep.hpp:2672-2674`; if T4 sees
  doubled potentials, that convention is the first thing to check.
- **T5** — `key_needs_level = true` is declared, so **R6** applies in full: the
  operator cache empties on every rebuild whose root half-width moves. T5's
  measurement is of that, and the declaration in the header states the coupling.
- **T2, T3** — `scripts/tuolumne/run_ctest_cartesian_taylor_serial.flux` was
  reused **unchanged**, as T1 intended, and remains reusable by T3. Its
  anchored `-R` is still what excludes the valgrind variant; that variant was
  neither made to pass nor disabled.
