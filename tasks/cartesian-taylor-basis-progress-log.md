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
noting how *small* it is — $10^{-4}$ of scale, eight decades above the tolerance
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

## T3 — the M2L, and the sign and normalization convention

Replaced the three `Kokkos::abort` bodies in
`src/Canopy_CartesianTaylorBasis.hpp` — `build_m2l_operators`, `m2l_core` and
`m2l_translate` — added the operator's single source of truth and its two
sizing traits, gave `aux_tables_type` its one scalar, and added five bodies to
`tests/tstCartesianTaylor.hpp`. One README correction rides in the final
commit. Nothing in T1's `Canopy::CartesianTaylor` namespace changed, and
nothing outside those three files was touched.

Flux jobs, all through the unchanged
`scripts/tuolumne/run_ctest_cartesian_taylor_serial.flux`: `f3YeTspuFk3q`
(first full pass, 12/12, run with a deliberately non-gating bound so the
end-to-end constant could be *measured* before it was pinned), `f3YeUnMzVdk3`
(12/12 with the pinned bound — the checkpoint tree), `f3YeVguWwKE7`,
`f3YeX8BoVNNB`, `f3YeXxRkMxHM` (the three perturbations, rc 8 each), and
`f3YeYsBmmaa3` (12/12 on the exact tree committed, byte-identical figures).

### The operator, and why it takes `R` and not a key

```cpp
KOKKOS_INLINE_FUNCTION
static void m2l_operator_block( const Scalar R[3], Scalar b,
                                Scalar ( &op )[m2l_op_entries] );
```

`op[ m2l_op_index(p_slot, q_slot) ] = (-1)^{|q|} b_{p+q}(R)`, target-slot-major,
so `build_m2l_operators` copies it into `ops(p, q, j)` with no transpose.
`R = c_target - c_source`; `b` is softening **squared**.

**It is parameterized on the physical `(R, b)` and has no `dd` dependence, and
that is forced rather than chosen.** `m2l_translate` is handed two cell
centers and two half-widths and **cannot recover `max_d` from them**, so a
key-parameterized operator would be unreachable from the fallback path — and
without a shared operator function the fused-versus-fallback comparison could
only ever be approximate. `MonopoleBasis::m2l_translate` reconstructs a key
and is **not** the model here: its local is dimensionless and normalized, so
it needs `F(dd)` to convert between "separation in deeper-cell half-widths"
and the source's own scale. A physical operator has no normalization to undo.

One consequence, expected rather than debugged: **two keys differing only in
`dd` get bit-identical columns.** `m2l_parity_identity` asserts that directly
(keys 0 and 1 of its list differ only in `dd`), so the duplication is pinned as
a property rather than left to be rediscovered as a suspected bug.

### Where `b` had to live, which was not obvious

`m2l_translate` is a **device** operator and the fallback call site
(`src/Canopy_DownwardSweep.hpp:2338-2347`) hands it the team, the multipoles,
the source cell, the physical offset, two half-widths, `aux`, and the target
slice — **no `M2LKernelParams`**. So for a basis whose operator depends on a
kernel parameter, `aux_tables_type` is the *only* channel from
`build_aux_tables` to a device operator. It grew exactly one member:

```cpp
template <class MemorySpace> struct aux_tables_type { double b = 0.0; };
```

**`b` is softening SQUARED**, and `build_aux_tables` is the one place on this
path where the square is taken. `build_m2l_operators` does **not** read `aux` —
it has `kernel_params` directly and squares it itself — so the square appears
in exactly two places, one per path, each with the units on the declaration.

The default of `0.0` is load-bearing and is why `m2l_translate` **guards**
`aux.b > 0` rather than trusting it: a sweep a test drives directly is never
handed a configuration and runs at `M2LKernelParams::softening = 0`, the
unsoftened kernel this basis has no expansion of. `build_aux_tables` itself
deliberately does **not** reject `eps <= 0`, because `UpwardSweep::setup` calls
it unconditionally including for sweeps that never run an M2L; the rejection
belongs at the point of use.

### Decisions carried in from the task statement

- **The fused-versus-fallback body constructs its geometry so the two paths'
  $R$ are bit-identical, and asserts exact equality.** Root half-width 1.0 (a
  power of two), `unit_w[d] = 2^-d`, and both cell centers exact dyadic
  multiples of `unit_w[3] = 0.125`, so `c_s - c_t` reproduces
  `(ii,jj,kk) * unit_w[max_d]` bit for bit. The body **asserts that premise
  first** (`ASSERT_EQ` on each axis, with a message saying an exact comparison
  is the wrong assertion if it fails) so a later session that moves the
  geometry off the dyadic grid is told why its `ASSERT_EQ`s started failing
  instead of concluding the operator drifted.
- **The accumulator layout is T2's**, unchanged: `m2l_core` fills exactly
  `acc_slot(out_idx, c, s)` and `m2l_post_cell` reads it as written.
- **`taylor_accumulate` is reused**, not reimplemented, in both M2L paths.
- **One flat index.** `m2l_ladder_slots = num_slots(2 * P_ORDER)` and the
  order-$p$ slot table is its prefix by T1's graded order, so `slot(p+q)`
  indexes the ladder directly. No second map exists.
- **`num_coeffs_per_cell`, `sets_per_component` and `m2l_scratch_bytes` stayed
  `constexpr`** (**R3**), and the two traits T3 added — `m2l_op_entries` and
  `m2l_ladder_slots` — are `constexpr` too, because both are local-array
  extents inside device operators.
- **The `_valgrind` variant was left exactly as it is**; the script's anchored
  `-R` still excludes it.

### Measured

Every figure is against the **sum of the term magnitudes** of the check, not
against its result. The terms alternate in sign and cancel, so a relative test
against the result would be a test of the cancellation rather than of the
operator. Tolerance $10^{-13}$ of that scale throughout; the tightest of the
three finite figures sits 84x below it, the other two at 500x or better.

| Body | Check | Achieved |
| --- | --- | --- |
| `m2l_ell0_closed_forms` | $\ell_0$ vs §2's closed forms, $p = 2$, 16 geometries | $2.000\times10^{-16}$ |
| `m2l_ell0_closed_forms` | the same at $p = 3$ | $1.961\times10^{-16}$ |
| `m2l_p1_contraction` | $\lvert p\rvert = 1$ vs the $K/dK/ddK$ contraction, $p = 2$ | $1.190\times10^{-15}$ |
| `m2l_parity_identity` | 7 keys, $R$ and $S$ independently sourced | $0$, bitwise |
| `m2l_fused_vs_fallback` | 5 keys, single pair, plus a 3-pair accumulation | $0$, `ASSERT_EQ` |

`m2l_end_to_end`, P2M → M2L → L2P against a direct softened sum, $p = 2$,
$\varepsilon = 0.025$ (so $b = 6.25\times10^{-4}$), both boxes at **half-width**
$W = 0.5$ with sources at $\lvert d\rvert \le 0.48$ and probes at
$\lvert a\rvert \le 0.48$ — a near-worst-case interior sample for that $W$,
not a comfortable one:

| $R/W$ | $R$ | relative error | achieved $c$ | bound $(1.25\,W/R)^3$ |
| --- | --- | --- | --- | --- |
| 8 | 4.0 | $2.3370\times10^{-3}$ | 1.062 | $3.815\times10^{-3}$ |
| 16 | 8.0 | $2.3456\times10^{-4}$ | 0.987 | $4.768\times10^{-4}$ |
| 32 | 16.0 | $2.5298\times10^{-5}$ | 0.939 | $5.961\times10^{-5}$ |

The error falls by 9.96x and 9.27x per doubling of $R/W$ against the $8\times$
a $(W/R)^{p+1}$ law predicts, i.e. slightly **faster** than third order over
this range, which is the finite-source-extent effect and not an anomaly. The
constant $c = 1.25$ was pinned **after** `f3YeTspuFk3q` measured 1.062 as the
worst achieved value; it is stated on the declaration that raising it to
accommodate a failure would silently change what the body means.

### Two parity facts worth having written down

**The parity identity is bitwise, not merely to round-off, and that is not a
coincidence.** $b_n(S)$ and $b_n(R)$ with $R = -S$ differ termwise by exact
sign flips — $w$ is identical and every term of the §3 recurrence carries a
fixed parity of $r$ factors — and both sides sum in the same slot order, so
every intermediate matches bit for bit. The achieved $0$ is therefore the
*expected* value and a non-zero one would mean something real. This does not
make the check vacuous: it is vacuous only if the two arguments come from one
source, which is exactly what the independent sourcing prevents, and
perturbation B proves it fires.

**The exit criterion's predicted failure direction was half right, and the
half that was wrong is informative.** It says omitting the negation of $R$
"must fail the parity check at the first odd $|p|$". Measured, it fails at
$|p| = 0$: with the sign multiplier still in place the perturbed left side is
$(-1)^{|p|}\sum_q b_{p+q}(R) M_q$ against a right side of
$\sum_q (-1)^{|q|} b_{p+q}(R) M_q$, and those differ on the odd-$|q|$ terms at
**every** $|p|$, even ones included. The "first odd $|p|$" prediction is
correct for the *both-dropped* build, where the two errors cancel exactly at
$|p| = 0$ and the first surviving disagreement is at $|p| = 1$ — which is what
was observed, with $\rm lhs = -\rm rhs$.

### The three perturbations

Each was applied, built on the login node, run as its own job, and reverted
**by inverting the edit** — never by `git checkout`, which would have discarded
the task's uncommitted work. The tree was rebuilt and re-run green afterwards
(`f3YeYsBmmaa3`), and `git diff HEAD` was empty against the checkpoint commit
before that run.

**A — the $(-1)^{|q|}$ multiplier dropped** (`sign` forced to $+1$ in
`m2l_operator_block`). `f3YeVguWwKE7`, rc 8, 4 of 12 bodies failing. First
failure is `m2l_ell0_closed_forms` at $R = (2,0,0)$, $b = 6.25\times10^{-4}$:
got $0.62584$, want $0.85579$, difference $0.22995$ against a tolerance of
$9.27\times10^{-14}$ — twelve orders of magnitude, at the first odd $|q|$ as
**R1** predicts. `m2l_p1_contraction`, `m2l_parity_identity` and
`m2l_end_to_end` fail too. **`m2l_fused_vs_fallback` passes**, correctly and
importantly: both paths reach the same perturbed operator, so it is an
agreement check and never a convention check, and a session reading a green
line there under a wrong operator should not take it as evidence about the
convention.

**B — the negation of $R$ omitted** in `build_m2l_operators`'s key-to-$R$ path.
`f3YeX8BoVNNB`, rc 8, 2 of 12 failing: `m2l_parity_identity` (at $|p| = 0$,
lhs $1.53785$ vs rhs $0.72480$, tolerance $1.96\times10^{-13}$) and
`m2l_fused_vs_fallback` (only the table path was perturbed, so the two paths
genuinely disagree). **$\ell_0$ and the $|p|=1$ contraction pass**, because
both call `m2l_operator_block` with their own $R$ and never go through the key
path. That split — sign errors caught by the closed forms, $R$-sense errors
caught by the parity identity — is the discrimination the exit criterion asks
for, and it only exists because the parity check's two arguments are sourced
independently.

**C — both dropped.** `f3YeXxRkMxHM`, rc 8, 5 of 12 failing — every M2L body.
The parity identity fails at $|p| = 1$ with lhs $= -0.665732$ and
rhs $= +0.665732$, i.e. exactly $\rm lhs = -\rm rhs$: the two errors cancel at
$|p| = 0$ and nowhere above it. The errors do **not** cancel for $|p| > 0$,
which is what the criterion required a both-dropped build to show.

### An optimization not taken, again

T1 flagged a host-built table of `(m, i, k, term slots)` for
`derivative_ladder`, and **T3 is the first task with a call site that would pay
for it**: `m2l_operator_block` evaluates one full ladder per key in
`build_m2l_operators` and one per pair in `m2l_translate`. It is still not
built. **R3** records a *measured* +18% M2L regression from a comparable
indirection, with two candidate micro-causes tested and excluded, so a table
here has to be measured against the recompute it replaces and not assumed
faster — and no exit criterion in this document depends on a timing figure.
The README entry was updated to say the call site now exists rather than that
it does not yet. Still worth doing; it needs `build-tuolumne-prof/` and an
`M2L kernel (all depths)` figure on either side, which is a task of its own.

**Affects:**

- **T4** — the single-source-of-truth function's final signature is
  `static void m2l_operator_block( const Scalar R[3], Scalar b, Scalar
  (&op)[m2l_op_entries] )`, `KOKKOS_INLINE_FUNCTION`, host- and
  device-callable, allocating nothing, with `R = c_target - c_source` and `b`
  softening **squared**. Any host reference T4 wants must go through it.
  **On the accuracy margin: there is very little at the admissibility edge.**
  The measured relative error at $R/W = 8$ — which is barely inside Canopy's
  $\theta = 0.5$ MAC, $R > 2\sqrt3\,(w_a{+}w_b) = 6.93\,W$ — is
  $2.34\times10^{-3}$, **above** the 1e-3 bar, and 1e-3 is only reached
  somewhere around $R/W \approx 13$ ($2.35\times10^{-4}$ at $R/W = 16$). This
  is the quantitative form of the design's "the reference's admissibility is
  not Canopy's": the 1e-3 figure transfers at $\theta = 0.3$ and should **not**
  be expected at $\theta = 0.5$, and T4's two-arm structure is what makes that
  visible rather than looking like a defect. A single-pair bound is not a
  whole-solve error, so T4 must measure rather than extrapolate from this — but
  it should expect the $\theta = 0.5$ arm to land near 1e-3 and not far below
  it, and should not read a miss there as a basis defect without first checking
  the $\theta = 0.3$ arm.
  **R7 still stands**: nothing here instantiated `Solver::solve()`, so expect
  T4's first failures in `src/Canopy_Solver.hpp`.
- **T4** — `m2l_translate` and `build_m2l_operators` both **abort** on a
  non-positive softening, and `build_m2l_operators` additionally on
  `max_d` outside `[0, n_levels)` and on `unit_w[max_d] <= 0`. T4 sets an
  explicit positive softening, so none should fire; if one does, it is a
  configuration-ordering bug in the solve setup and not a basis defect, and the
  message names which value it was.
- **T5** — unchanged by this task. `key_needs_level = true` still holds and
  **R6** still applies in full.
- **T4, T5** — `scripts/tuolumne/run_ctest_cartesian_taylor_serial.flux` was
  reused **unchanged** for the third task running. Its header comment still
  lists only T1's three bodies; the target now has twelve, and the anchored
  `-R` is what keeps the valgrind variant out of every one of these runs.

## T4 — a full-pipeline solve at `near_softening_factor = 0`

Added `tests/tstCartesianTaylorSolve.hpp` (two gating bodies over a multi-step
harness parameterized on order, basis and domain span), one line in
`tests/CMakeLists.txt` `UNIT_MPI_TESTS`, and
`scripts/tuolumne/run_ctest_cartesian_taylor.flux`. Two stale line references
in `tasks/cartesian-taylor-basis.md` corrected. **No file under `src/`
changed.**

Two things in this task were **not** what the task statement assumed, and both
are the real output: `Solver` needed no edit at all, and the
$\theta = 0.3$ arm had to move to $p = 3$.

Flux jobs, all `pdebug`, all on a saturated queue (40/40 nodes allocated, so
each waited 20-30 min in `SCHED`): `f3YfM7Ldtv5D` (first full run),
`f3YfdTuJH59H` (R2's mandated T1 re-run, the `LaplaceKernel` failure direction
and a $p = 3$ sweep), `f3YfgoXqJe2j` (re-measurement after the domain was
corrected), `f3YfjP7sDgw9` (gate at $p = 2$, red), `f3YfvsefN86T` (the domain
span sweep), `f3Yg13MRtyp3` ($\theta = 0.5$ measured at the anchored span,
np 1-6) and `f3Yg3EA788xw` (the exit-criterion job on the exact committed
tree).

### R7 did not fire, and that is the headline

**`src/Canopy_Solver.hpp` needed no edit.** `setup()`, `solve()`, `migrate()`
and `rebalance()` all instantiated against `CartesianTaylorBasis` and compiled
clean on the first build. The list of `Solver` member bodies that turned out to
assume something `LaplaceKernel`-specific — which T4's statement, **R7** and
T2's **Affects:** line all expected to be this task's deliverable — **is
empty.**

That is only credible because the same binary also drove
`LaplaceKernel<double, 2, 3>` through the identical harness for the
failure-direction arm: both bases reach `Solver::solve()` through one code
path, so the clean build is not an artifact of the new basis resembling the old
one at the declaration level. `Solver` is basis-agnostic as written.

### Why the $\theta = 0.3$ arm runs at $p = 3$

At $p = 2$ the $\theta = 0.3$ **gradient** is $8.996\times10^{-3}$ against the
$10^{-3}$ bar, while the **potential** meets it at $2.655\times10^{-4}$. The
three risks the task names as the diagnostics were all **excluded by
measurement** before anything was changed:

- **Not R1.** R1 predicts a miss "by one to three orders at every rank count
  and at both $\theta$". The potential is $2.655\times10^{-4}$ — three decades
  better than a convention error permits, and a misplaced $(-1)^{|q|}$ or
  $1/q!$ corrupts the $|q| = 1$ terms that dominate $\varphi$ and
  $\nabla\varphi$ alike. T3's `m2l_ell0_closed_forms` and `m2l_p1_contraction`
  are green.
- **Not R2**, and R2's instruction earned its place. It requires T1's oracle be
  re-run at the specific $(r, b)$ scales T4 uses, and those turned out never to
  have been sampled: T4's band is $|r|/\sqrt b \in [9.28, 74.2]$, lying
  **entirely between** T1's `1.0` and `100.0`. Re-run with
  `scales = {0.01, 1.0, 9.276, 15.46, 74.2, 100.0}`, `closed_forms` went from
  40 to 76 samples (so the new points did run) with the worst deviation moving
  only from $2.911\times10^{-15}$ to $5.465\times10^{-15}$ against a
  $10^{-12}$ tolerance, and `finite_difference` at $|k| = 4$ was **unchanged**
  at $7.321\times10^{-7}$. All 12 bodies green.
- **Not R4.** `total_fallback_pair_count()` is ~300 per rank at
  $\theta = 0.3$ and **exactly 0** at $\theta = 0.5$, while both arms show the
  same error structure. Worth writing down because it is deducible rather than
  measured: `max_depth = 6` bounds $|dd| \le 6$ identically, so the
  `m2l_key_dd_max = 6` guard **can never fire** here. Every fallback is the
  sweep's own `|offset| <= 32` guard (`src/Canopy_DownwardSweep.hpp:526`),
  tripped by deep-versus-deep pairs separated far enough for the tighter
  $\theta = 0.3$ MAC.

**It is the target-side L2P truncation.** The gradient of a degree-$p$ Taylor
local is a degree-$(p-1)$ polynomial, so $\nabla\varphi$ truncates one order
before $\varphi$. The fingerprint is the ratio of the **absolute** errors,
which is $1/W$ for a target-side term and would be $1/R$ for a source-side one:

| domain half-span | $W$ (depth-4 leaf) | abs. grad / abs. pot | $1/W$ | agreement |
| --- | --- | --- | --- | --- |
| 0.45 | 0.033463 | 29.47 | 29.88 | 1.4% |
| 0.06 | 0.004471 | 218.5 | 223.7 | 2.3% |

$1/R$ at the MAC edge is 2.59 and 19.4 — off by an order in both. The error is
the potential's error differentiated over the **cell half-width**, on two
domains differing 12-fold in scale.

**`treecode.py` has no target-side expansion at all**, so its documented
$\sim 10^{-3}$ **velocity** accuracy is a source-side-only figure — the same
truncation order as this solve's **potential**. Applying it to an FMM's
gradient at the same $p$ compares quantities of different order. This is the
same species of mismatch the design document already records for admissibility
("the 1e-3 figure transfers only at matched admissibility"), one level deeper:
it also transfers only at matched **expansion structure**.

Running the $\theta = 0.3$ arm at $p = 3$ gives $\nabla\varphi$ the truncation
order the reference's velocity has, which is what makes the comparison a
comparison. **This was a decision taken by the task owner, not by measurement**
— the alternative, restating the bar as a claim about the potential, was
equally available. `CTS_P` stays 2 for the $\theta = 0.5$ arm and for the
particle-set seed, so the two arms now differ in order as well as
admissibility; the $\theta = 0.5$ arm's job is to record what Canopy's own
default delivers at the reference's order, and it has no bar to meet.

Note **R8**: T1's finite-difference oracle is validated at $|k| = 4 = 2p$ for
$p = 2$ only. The $\theta = 0.3$ arm now runs at $p = 3$, whose ladder reaches
$|k| = 6$, so **that arm is outside what T1's oracle has checked.** R8 says the
FD check must be re-run rather than assumed if $p$ is raised. It has not been.
This is the one loose end T4 leaves, and it is cheap to close: re-run
`Canopy_Test_CartesianTaylor_SERIAL`'s `finite_difference` body at
`max_k = 6`, re-measuring the step divisor rather than widening the tolerance.

### The domain span, which was got wrong first and then anchored

The first configuration put the particles on `[0.05, 0.95]^3`, copied from
`tstLaplaceSolve.hpp`. **It could not tell the two bases apart**, and the
`LaplaceKernel` failure direction did not fire:

| half-span | basis | $\theta = 0.3$ potential | $\theta = 0.3$ gradient |
| --- | --- | --- | --- |
| 0.45 | CartesianTaylor $p{=}2$ | $3.433\times10^{-4}$ | $8.102\times10^{-3}$ |
| 0.45 | **LaplaceKernel** | $1.258\times10^{-3}$ | $7.513\times10^{-3}$ |

`LaplaceKernel` was $3.7\times$ worse on the potential and **better** on the
gradient — no failure direction at all, and a comparison that demonstrated
nothing while looking reasonable.

The cause is dimensionless and has nothing to do with the particle count. The
MAC puts the closest admissible pair at $R = 11.55\,W$, and the unsoftened far
field's relative error from ignoring $b$ is $b/(2R^2)$; what decides whether
that is large is therefore $\varepsilon/W$ and **nothing else**. With a
half-span $h$, $W = 1.2h/16$ and $R = 0.866h$, so $R/\varepsilon = 34.6\,h$.
At $h = 0.45$, $R = 15.5\,\varepsilon$ and $b/(2R^2) = 2.1\times10^{-3}$ — the
same order as the $p = 2$ truncation.

**Two requirements pull opposite ways**, and both are stated in T4: the failure
direction wants a small $h$, the $10^{-3}$ gradient wants a large one. Rather
than slide $h$ until a number landed, the value was fixed on a stated principle
**before** the sweep: $h = 0.1155$ is the half-span at which the closest
admissible pair sits at exactly $R = 4\varepsilon$, the **default**
`FmmConfig::near_softening_factor` — the separation at which Canopy itself
declares an unsoftened far field unsafe and forces the pair back to softened
P2P. A basis accurate there while `LaplaceKernel` is not is precisely the claim
this test exists to make.

The sweep (`f3YfvsefN86T`, $\theta = 0.3$, np 1-2, this basis at $p = 3$ and
`LaplaceKernel` at $p = 2$, same particle set) then confirmed it:

| $h$ | $R/\varepsilon$ | CT $p{=}3$ gradient | LaplaceKernel $p{=}2$ potential | ratio |
| --- | --- | --- | --- | --- |
| 0.060 | 2.08 | $1.1412\times10^{-3}$ MISS | $6.907\times10^{-2}$ | 3171x |
| **0.1155** | **4.00** | $7.0132\times10^{-4}$ **PASS** | $1.894\times10^{-2}$ | **999x** |
| 0.231 | 8.00 | $7.5834\times10^{-4}$ PASS | $4.728\times10^{-3}$ | 207x |

The failure direction weakens monotonically with $h$, as $b/(2R^2)$ requires;
the gradient has a shallow minimum near the anchor. **Nothing else about the
problem changes with $h$** — the tree is a function of the dimensionless
geometry, and the sweep says so: `n_cells` 2936 / 2913 / 2916 and realized keys
26260 / 26180 / 26570.

**This is the lesson worth carrying forward: a solve test for a softened
far-field basis is only a test of the softening when $\varepsilon$ is
comparable to the admissible separation, and Canopy's MAC makes that a
statement about $\varepsilon/W$, not about $\varepsilon$.**

### The configuration behind both pinned constants

8640 particles global (divisible by every rank count 1-6) on a cube of
half-span 0.1155 about 0.5, `ncrit = 8`, `max_depth = 6`,
`replication_depth = 2`, `softening = 0.025` with
`near_softening_factor = 0`, `NComps = 3`, charges one-signed on [0.5, 1.5]
drawn **independently per component** from seed `90210 + CTS_P`, SERIAL
backend, np 1-6. $\theta = 0.3$ at $p = 3$, $\theta = 0.5$ at $p = 2$.

**The particle count and `ncrit` are chosen together and are not free.** The
tree refines while a cell holds more than `ncrit`, so 8640 at `ncrit = 8`
reaches depth 4, a $16^3$ leaf grid. The $\theta = 0.3$ near field reaches
$11.55\,W = 5.77$ cell widths, a neighbourhood of about 800 cells — under 20%
of $16^3$, so most pairs are far field. At depth 3 ($8^3 = 512$ cells) that
same neighbourhood covers the **whole grid** and only corner pairs stay
admissible: `m2l_n_unique_ops() > 0` would still hold while the far field
carried almost none of the field, and the accuracy check would pass without
measuring anything. Lowering the particle count reopens exactly that hole.

**The step schedule** is `solve -> migrate -> solve -> rebalance -> solve ->
migrate -> solve`. `rebalance()` repartitions, rebuilds the comm plan and
invalidates the interaction list unconditionally, so the last two solves face
an operator table rebuilt after the root box moved — **R5**'s stale-operator
path, and the reason this file does the opposite of
`tstLaplaceSolve.hpp:860-871`, which uses `migrate()` and never `rebalance()`
because it has a np 1-2 **bitwise** gate and needs one partition. No
drifting-operator symptom appeared: the six rank counts agree to 15 significant
figures.

**Measured, np 1-6** (`f3Yg13MRtyp3`, and reproduced on `f3Yg3EA788xw`):

| arm | potential | gradient | gate |
| --- | --- | --- | --- |
| $\theta = 0.3$, $p = 3$ | $1.9263\times10^{-5}$ | $7.0718\times10^{-4}$ | 1e-3 bar, 1.41x margin |
| $\theta = 0.5$, $p = 2$ | $9.9667\times10^{-4}$ | $1.8652\times10^{-2}$ | pinned 3.74e-2 |

`CTS_DEV_TOL_THETA_REF` is **the bar itself**, not 2x the measurement — 2x
would be $1.41\times10^{-3}$, above the bar and a weaker gate than the exit
criterion. `CTS_DEV_TOL_THETA_CANOPY` is 2x its measured worst.

**Counters**, $\theta = 0.3$ / $\theta = 0.5$ at np=1: `n_unique_ops`
25438 / 7374, `total_fallback_pair_count()` 246 / 0, effective operator cap
32768 in both (the count cap, far above the realized key count, as **R4**
expects). `n_cells` 2941, root half-width 0.13842.

### Decisions carried in from the task statement

- **`near_softening_factor = 0` with an explicit `softening = 0.025`.** The
  default $-1.0$ selects auto-softening, whose effective $\varepsilon$ moves
  with the distribution and would make both constants unpinnable. With
  `near_softening_factor = 0` the floor in
  `CommunicationPlan::mac_satisfied` (`src/Canopy_CommunicationPlan.hpp:354`)
  is skipped outright — it is guarded on `_near_softening_k > 0.0`.
- **The direct sum is softened**, $\phi_i = \sum_{j\ne i} q_j (r^2+b)^{-1/2}$
  and $\nabla\phi_i = -\sum_{j\ne i} q_j d (r^2+b)^{-3/2}$ with
  $b = \varepsilon^2$, matching `src/Canopy_P2P.hpp:885` and the sign
  convention of `tests/tstLaplaceSolve.hpp:1535-1538`.
- **Deviations are normalized by the global field scale**, not per particle.
- **No potential doubling appeared.** `l2p_evaluate` writes `phi_out` with
  `=`, matching the sweep's accumulate at
  `src/Canopy_DownwardSweep.hpp:2673-2674`; T2's **Affects:** line flagged this
  and it needed no action.
- **Nothing was added to `REGRESSION_MPI_TESTS`**, `ctest -L regression` was
  not run, `clang-format` was not run, `CANOPY_LAPLACE_SOLVE_REGENERATE` was
  never set and `tests/data/laplace_solve_P6.txt` was not regenerated.

### One departure from the task statement

**The harness is parameterized on order, basis and domain half-span**, where
the statement implies a single spelling. T4 requires a `LaplaceKernel` arm, the
diagnosis required a $p = 3$ arm, and the domain had to be swept — with a
parameterized harness each is an **added `TEST` body** rather than an edit to
the code under test, so the shipped path is provably the one that was measured.
The particle set does not depend on order or basis (the seed is
`90210 + CTS_P`, a constant), so arms are directly comparable. `dt` is derived
from the span by `cts_dt_for_half_span` under the $L^{3/2}$ law so spans are
compared at a constant dimensionless trajectory.

### The `LaplaceKernel` failure direction

Run as T1, T2 and T3 ran their perturbations: a temporary **additive** edit,
built on the login node, submitted as its own job, reverted **by inverting the
edit** — never `git checkout`, which would have discarded this task's
uncommitted work. `tests/tstCartesianTaylor.hpp` was confirmed byte-identical
to `HEAD` afterwards, and every diagnostic `TEST` body was deleted before the
checkpoint.

At the shipped configuration it fails as T4 requires: **$1.894\times10^{-2}$
potential against this basis's $1.896\times10^{-5}$ — 999x, three orders of
magnitude** — and $5.168\times10^{-2}$ gradient. That is the unsoftened
multipole far field measured against a softened reference at
`near_softening_factor = 0`, at the exact separation Canopy's own default
declares unsafe for it.

**Affects:**

- **T5** — inherits the multi-step harness ready-made, parameterized on order,
  basis and span, already doing `migrate / rebalance / migrate` across 4
  solves, already printing `n_unique_ops`, `total_fallback_pair_count()` and
  `m2l_effective_op_cap()` per rank. T5 adds `m2l_op_keys_built_count()`,
  `m2l_op_cache_size()` and `interaction_list_build_count()` to the same
  `[ct-solve]` line. **Its "drifting root box" premise is satisfied but only
  just**: the root half-width moves in its 8th significant figure between arms
  (0.138419475 against 0.138419418), enough to change `set_root_half_width` and
  empty the cache, but T5 should raise `CTS_DT` for a visibly drifting box —
  and must scale it by $L^{3/2}$ via `cts_dt_for_half_span`, not copy
  `tstLaplaceSolve.hpp`'s.
- **T5, and any later basis** — **R7 is closed.** `src/Canopy_Solver.hpp` is
  basis-agnostic as written. Do not budget for `Solver` edits.
- **R8 is now live and was not closed here.** The $\theta = 0.3$ arm runs at
  $p = 3$, whose ladder reaches $|k| = 6$, while T1's finite-difference oracle
  is validated only at $|k| = 4$. Re-run that body at `max_k = 6`, re-measuring
  the Richardson step divisor rather than widening the tolerance, before
  leaning further on $p = 3$.
- **R6 is untouched and unmeasured** by this task; T5 owns its magnitude.
- **Anyone writing a solve test for this basis** — the $\varepsilon/W$
  argument is the load-bearing one. A domain on which the MAC puts every
  admissible pair at $R \gg \varepsilon$ makes `CartesianTaylorBasis` and
  `LaplaceKernel` indistinguishable, and a test written there proves nothing
  while looking green.

## T5 — the operator cache across a drifting bounding box

Added two measurement bodies and a per-step `[ct-cache]` line to
`tests/tstCartesianTaylorSolve.hpp`, a `dt_scale` parameter threading through
the harness and `runArm`, the constant `CTS_DT_SCALE_DRIFT`, and
`scripts/tuolumne/run_ctest_cartesian_taylor_t5.flux`. One stale comment in
that test corrected. **No file under `src/` changed** — T5 measures **R6**'s
magnitude and designs no fix.

Flux jobs, all `pdebug`: `f3ZcosWuders` (the `dt_scale` probe, np 1-2,
`--time-limit=8`, backfilled immediately), `f3ZcrXqDFqZH` (the measurement
every figure below is quoted from, np 1-6, `--time-limit=20`) and
`f3ZcvgKiGjaw` (the same job re-run on the **exact committed tree** — the only
difference from `f3ZcrXqDFqZH` is comment text and three wrapped lines, and it
reproduces every counter digit for digit: 336 builds, 0 keys retained,
`keys_built` 7370 / 14830 / 22230 / 30012 and
25406 / 50962 / 76786 / 103254 at np = 1). Every queue was short, unlike every
T4 job. `ctest -V -R Canopy_Test_CartesianTaylorSolve_MPI_SERIAL`: **6/6
passed** on both, 76.1 s.

### The headline: the cache retains nothing, and the figure is exactly zero

At **every one of the 336 builds** in the exit-criterion job — 4 solves x
(1+2+3+4+5+6) ranks x 4 arms — the per-step increment in
`m2l_op_keys_built_count()` equals `m2l_op_cache_size()` after that build,
**exactly**. Summed over all of them, the number of cached operators that
survived a rebuild is **0**.

That is **R6** confirmed in full and at face value: `key_needs_level = true`
means `set_root_half_width` empties the entire cache on any change to the root
half-width (`src/Canopy_DownwardSweep.hpp:406-416`), the box drifts on every
rebuild of a moving-particle solve, and `Solver::_push_root_half_width()`
(`src/Canopy_Solver.hpp:756-770`) pushes the new width in before every
`_downward.setup()`. Both root half-widths on the `[ct-cache]` line — the
sweep's own copy, which is what the clearing rule compares, and the builder's
box — agreed on all 336 lines, so the mechanism is visible end to end rather
than inferred.

**Per-rank `m2l_op_keys_built_count()` after the 4th build, against the cache
size at that point** (`f3ZcrXqDFqZH`, `dt_scale = 3`, both arms $p = 2$):

$\theta = 0.5$ (`CTS_THETA_CANOPY`):

| np | keys_built, per rank | cache_size, per rank | ratio |
| --- | --- | --- | --- |
| 1 | 30012 | 7782 | 3.86 |
| 2 | 26409, 23238 | 6755, 6006 | 3.89 |
| 3 | 25704, 21234, 19928 | 6549, 5459, 5154 | 3.90 |
| 4 | 24150, 19945, 21219, 18996 | 6117, 5143, 5473, 4870 | 3.90 |
| 5 | 22362, 21183, 18313, 19793, 17443 | 5776, 5335, 4734, 5083, 4458 | 3.90 |
| 6 | 21217, 20077, 17475, 19655, 19647, 15188 | 5413, 5045, 4547, 5172, 4910, 3912 | 3.91 |

$\theta = 0.3$ (`CTS_THETA_REF`):

| np | keys_built, per rank | cache_size, per rank | ratio |
| --- | --- | --- | --- |
| 1 | 103254 | 26468 | 3.90 |
| 2 | 83702, 81025 | 21223, 20701 | 3.93 |
| 3 | 76161, 73354, 66226 | 19234, 18815, 16813 | 3.93 |
| 4 | 67280, 64223, 65429, 61031 | 16835, 16443, 16576, 15902 | 3.92 |
| 5 | 60585, 68126, 56784, 60019, 53389 | 15497, 16892, 14684, 15136, 13816 | 3.93 |
| 6 | 53326, 61543, 51616, 55490, 60309, 46020 | 13402, 15303, 13443, 14652, 14988, 12047 | 3.92 |

The ratio is **the number of builds**, 4, less the few percent by which the key
set grows over the run. `interaction_list_build_count()` is 1/2/3/4 across the
four solves at every rank, so there is one operator construction per
interaction-list build and never a reuse.

**The two consecutive rebuilds the exit criterion asks for, in full**
(`f3ZcrXqDFqZH`, np = 1, per build: cumulative `keys_built`, the increment, and
the cache size after it):

| build | $\theta = 0.5$ | $\theta = 0.3$ |
| --- | --- | --- |
| 1 | 7370 (+7370, cache 7370) | 25406 (+25406, cache 25406) |
| 2 | 14830 (+7460, cache 7460) | 50962 (+25556, cache 25556) |
| 3 | 22230 (+7400, cache 7400) | 76786 (+25824, cache 25824) |
| 4 | 30012 (+7782, cache 7782) | 103254 (+26468, cache 26468) |

The increment tracks the *current* key count, which moves by a few percent as
the tree changes — so R6's "climbing by roughly the full key count at every
build" is right, with "roughly" doing no work beyond that drift in the key set
itself.

### Stated against T9's baseline, which is not like-for-like

T9 measured `m2l_op_keys_built_count() == m2l_op_cache_size()` at every rank of
every rank count with `interaction_list_build_count() == 2` — **zero keys
rebuilt** ([abstract-solver-backend-progress-log.md](abstract-solver-backend-progress-log.md)
§T9). Here the same two counters give **100% rebuilt at every build**: 3.9x the
cached key count constructed over four solves where T9's basis constructed 1.0x
over two.

**The comparison is legitimate but it is not apples to apples, and the
difference cuts both ways.** T9 drove `DownwardSweep` directly on
`LaplaceKernel` (`key_needs_level = false`), across one
`invalidate_interaction_list()`, with a root box that **never moved** — not
through a moving-particle solve. So T9's zero isolates the cache's own
behaviour under a topology change, while T5's 3.9x is the compound of a basis
that keys on level and a box that drifts. Neither number alone says what a
level-keyed basis would do on a *static* box: `set_root_half_width` is a no-op
on an unchanged value, so a static distribution would give T9's zero on this
basis too. **The cost measured here is the cost of motion, not of the basis.**

### The trajectory: `CTS_DT` was not raised, and a multiplier was added instead

T4's **Affects:** line told this session to raise `CTS_DT`. That would have
moved the trajectory under `CTS_DEV_TOL_THETA_REF` (the 1e-3 bar itself, met
with a 1.41x margin) and `CTS_DEV_TOL_THETA_CANOPY`, both measured at
`CTS_DT = 1.0e-5`, because `cts_dt_for_half_span` reads the constant on behalf
of both gating arms. So `CTS_DT` is **untouched** and the amplification is a
runtime multiplier — `CTS_DT_SCALE_DRIFT`, applied to the $L^{3/2}$-scaled `dt`
inside the step loop and defaulting to 1.0 on `runArm`, which leaves both
gating arms on the exact trajectory they were measured on.

**`CTS_DT_SCALE_DRIFT = 3.0` is pinned from a sweep, not chosen.** Probe
`f3ZcosWuders`, $\theta = 0.5$, $p = 2$, np 1-2, reading
`builder_root_half_width` at step 0..3 (displacement goes as $dt^2$ under
symplectic Euler from rest, so drift is quadratic in the multiplier):

| `dt_scale` | per-step drift in the root half-width | net | verdict |
| --- | --- | --- | --- |
| 1 | -0.020%, -0.039%, -0.059% | -0.118% | too small |
| **3** | **-0.177%, -0.349%, -0.399%** | **-0.922%** | **shipped** |
| 10 | -1.574%, -2.927%, -4.630% | -8.879% | saturates the cap |
| 30 | -12.9%, -29.0%, **+252%** | +118% | degenerate |

`dt_scale = 10` was the first choice and was **rejected on measurement**: at
$\theta = 0.3$ its cache reaches 32528 and then *exactly* 32768 operators — the
effective operator-count cap — so the cache overflows, is emptied and refilled,
which overlays that mechanism (T9's `opTableByteBudget` path) on the R6
measurement and clamps `n_unique_ops`. The $\theta = 0.5$ arm is unaffected at
10, but one constant keeps the two arms comparable. At 30 the cloud collapses
through its own centre and re-expands, the box more than doubling on the last
step, and the distribution is no longer the one the configuration was chosen
for. At 3 the 4th significant figure of the box moves at every build, the cap
stays unbound in both arms, and the contraction is monotone and under 1%.

**The conclusion does not depend on the constant.** Retention was zero at
`dt_scale` 1, 3, 10 and 30 alike, and zero in the two *gating* arms as well
(they emit `[ct-cache]` at `dt_scale = 1`: $\theta = 0.3$ builds
25406 / 25428 / 25438 / 25438 for a cumulative 101710). The multiplier only
buys a box whose drift is legible in the log.

### A correction to T4's characterization of the drift

T4 recorded that "the root half-width differs between the two arms in its 8th
significant figure, 0.138419475 against 0.138419418" and inferred the shipped
trajectory was barely enough. **That figure is the difference between the two
arms' final boxes, not the drift.** Measured per step here, the box on the
shipped trajectory goes 0.138583467 → 0.138555330 → 0.138500974 →
0.138419418, a **-0.118%** drift in its **4th** significant figure — ten
thousand times the inter-arm difference, and already more than enough to empty
the cache at every build. The knobs comment in the test now says both things
and distinguishes them.

### What this says about the two fixes R6 suggests

**R6 floats two candidate fixes and the measurement bears on one of them.**
Tolerating "a root-width change that is an exact power of two" would **not
fire on this workload**: the realized drift is a smooth monotone contraction of
0.18% to 0.40% per build, nowhere near a power of two, and the same is true of
any distribution whose bounding box is recomputed from moving particles. The
other candidate — keying the cache on the physical $R$ rather than on the level
— is untouched by anything measured here. A fix has 3.9x to beat, which is to
say it has to turn four operator constructions into one.

### The bodies, and one departure from the task statement

The two measurement bodies, `operatorCacheAcrossDriftThetaCanopy` and
`operatorCacheAcrossDriftThetaRef`, call `with_cartesian_taylor_solve`
**directly** rather than through `runArm`, with a no-op callback. They make no
accuracy claim and assert no deviation: they run a longer trajectory than the
gating arms, and neither pinned tolerance was measured on it. Both run at
`CTS_P = 2`, so — the consequence worth holding onto — **no arm added here
evaluates the derivative ladder above $|k| = 4$ and R8 does not bear on any
number above.**

**Departure:** `dt_scale` is a **required** positional parameter on
`with_cartesian_taylor_solve` and defaulted only on `runArm`, where the task
entry asks for it defaulted on both. A default is unusable there because the
parameter precedes the `Fn&& after` callback. This is exactly the shape
`pos_half_span` already has — required on the harness, defaulted on `runArm` —
which is the pattern the task entry names, and the requirement it exists for is
met: both gating bodies call `runArm` without the argument and run at
`dt_scale = 1.0`.

The per-step line is tagged **`[ct-cache]`** and not `[ct-solve]`: the latter
belongs to the once-per-run configuration echo and the once-per-run deviation
line, and a later session greps for one or the other.

### The gating arms came through unmoved

Both reproduce T4's figures at all six rank counts, to every digit printed:

| arm | potential | gradient | T4 (`f3Yg13MRtyp3`) |
| --- | --- | --- | --- |
| $\theta = 0.3$, $p = 3$ | 1.9263340835e-05 | 7.0717918545e-04 | identical |
| $\theta = 0.5$, $p = 2$ | 9.9666798509e-04 | 1.8651556395e-02 | identical |

(np 3-6 differ from np 1-2 in the 12th significant figure of the potential,
1.9263340835103815e-05 against 1.9263340835190165e-05, exactly as T4 recorded.)
The added `[ct-cache]` line is echo-only and the one assertion added to the
shared harness holds in every arm, so nothing about what the gating bodies
compute moved.

### The failure-direction outcome

`EXPECT_GT( m2l_op_keys_built_count(), 0 )` after the first build, spelled
`EXPECT` and not `ASSERT` for the reason the `n_unique_ops` guard beside it
gives — a fatal assertion returns from the harness and the `MPI_Gather` /
`MPI_Gatherv` below it are collective, so one rank leaving early hangs the rest
until the walltime and destroys the log the numbers have to be read out of.

**It held at every rank of every rank count**, smallest observed value 3757
(np = 6, rank 5, $\theta = 0.5$). **No perturbation was run to make it fire**,
and the honest reason is that it cannot be made to fire independently in the
current code: at the first build there is no prior cache, so
`m2l_op_keys_built_count() == 0` implies `m2l_n_unique_ops() == 0`, which the
pre-existing guard catches first. It is a redundancy against a future change —
a pre-populated or shared operator table would zero it while the far field
stayed live — and it is what makes "3.9x rebuilt" a statement about a real
cache rather than a difference of zeros.

### The stale comment corrected

`tests/tstCartesianTaylorSolve.hpp`, on `matchesDirectSumThetaCanopy`, stated
the measured $\theta = 0.5$ arm as "8.67e-04 potential, 2.16e-02 gradient".
Neither figure appears anywhere in T4's record, and both are inconsistent with
the shipped `CTS_DEV_TOL_THETA_CANOPY = 3.74e-02` (2x the measured worst) and
with the constant's own provenance comment twenty lines above it. Replaced with
9.9667e-04 and 1.8652e-02 and the job they came from. The "2.6x the theta = 0.3
gradient and 3.3x its potential" clause was recomputed against the log's
$\theta = 0.3$ figures and is now 26x and 52x — but it is **stated with the
caveat that it conflates two variables**, because that arm runs at $p = 3$ and
this one at $p = 2$, so the ratio is not a measurement of admissibility alone.
The old ratios were consistent with the $p = 2$ $\theta = 0.3$ diagnostic
figures (2.07x and 3.75x against 8.996e-03 and 2.655e-04), which is most likely
where they came from — a pre-$p{=}3$ draft of the comment.

**Affects:**

- **T6 — nothing measured here changes it.** T6 re-runs T1's
  finite-difference oracle at $|k| = 6$ because the $\theta = 0.3$ **gating**
  arm runs at $p = 3$ (**R8**), and that arm is untouched by this task: it runs
  at `dt_scale = 1` on the trajectory it was measured on and reproduces T4's
  figures exactly. Both arms T5 **added** are at $p = 2$, so they reach only
  $|k| = 4$ and R8 does not reach them. T6 starts exactly as written.
- **Any session extending the drift measurement** — `dt_scale` above ~10
  stops measuring R6 cleanly: the $\theta = 0.3$ cache saturates the 32768
  operator-count cap and the overflow-and-refill path overlays the level-keying
  one. Raise the cap first, or stay at or below 3.
- **Whoever fixes R6** — the target is 3.9x, and the "exact power of two"
  tolerance R6 suggests does not fire on a moving-particle box. See above.
- **`m2l_op_keys_built_count()` has no per-build reset**, which is why the
  measurement had to go inside the step loop. If a later session wants
  per-build figures without parsing cumulative differences, that is a `src/`
  change and T5 deliberately made none.

## T6 — the derivative ladder validated at $|k| = 6$

One file changed, `tests/tstCartesianTaylor.hpp`. Nothing under `src/`, nothing
in `tests/tstCartesianTaylorSolve.hpp`, no other test body, no change to the
flux script, no change to `p_order`.

### What the oracle could not do before

Three separate things, and only the first is the one T6's title names:

1. `max_k = 2 * p_order = 4` stopped the check at degree 4. Fixed with a body-
   local `fd_max_k = 6`; `p_order` stays 2 because it also sizes the ladder
   `closed_forms` allocates, and the shift and M2L bodies already choose their
   own orders explicitly.
2. **`stencil1D` could not evaluate a per-axis derivative above order 4 and did
   not say so.** Its `default:` branch returned the order-4 stencil for *any*
   higher order, and `fdDerivative` then divided by $h^5$ or $h^6$ anyway. Every
   degree-5 and degree-6 multi-index with a single component of 5 or 6 —
   $(5,0,0)$, $(6,0,0)$, $(5,1,0)$ and permutations — would have been
   differenced with the wrong operator and compared against a correct ladder
   value. This is the failure mode that looks exactly like a recurrence bug, so
   it is worth being explicit: simply raising `max_k` to 6 without touching
   `stencil1D` would have produced a *red* test blaming §3. Orders 5 and 6 are
   now explicit — weights $(-\frac12, 2, -\frac52, 0, \frac52, -2, \frac12)$ and
   $(1, -6, 15, -20, 15, -6, 1)$ at offsets $-3 \ldots 3$, antisymmetric and
   symmetric respectively, so both error expansions carry even powers of $h$
   only and both Richardson steps stay valid unchanged. The arrays in
   `stencil1D` and the locals in `fdDerivative` widened 5 → 7, and `default:`
   is now `std::abort()` behind a message naming the order.
3. The sample set bracketed the $p = 3$ arm's band instead of entering it.
   `scales` permanently gains 9.276, 15.46 and 74.2 (§T4 measured these once);
   40 samples → 76.

### The divisor scan, on this machine

Run as a temporary loop inside the body over eleven candidates, read out of
**job `f3Ze4jwB9ggF`**, then deleted. Worst $|{\rm diff}|/(\varphi/L^{|k|})$
over all 76 samples, two Richardson steps:

| $h$ | $\vert k\vert = 4$ | $\vert k\vert = 5$ | $\vert k\vert = 6$ |
| --- | --- | --- | --- |
| $L/2$ | 3.1895e+01 | 2.1647e+02 | 1.2928e+03 |
| $L/3$ | 9.0864e-02 | 2.3917e+02 | 1.4189e+03 |
| $L/4$ | 1.1218e-02 | 1.8076e+00 | 1.0837e+01 |
| $L/6$ | 7.9408e-04 | 8.2201e-02 | 4.9289e-01 |
| $L/8$ | 1.3193e-04 | 1.2272e-02 | 7.3593e-02 |
| $L/12$ | 1.1048e-05 | 9.6088e-04 | 5.7912e-03 |
| $L/16$ | 1.9448e-06 | 1.6500e-04 | **1.1843e-03** |
| $L/24$ | **3.0798e-07** | **2.5308e-05** | 6.9018e-03 |
| $L/32$ | 7.3209e-07 | 6.3866e-05 | 4.3151e-02 |
| $L/48$ | 3.8231e-06 | 5.2008e-04 | 5.4051e-01 |
| $L/64$ | 8.3097e-06 | 2.6844e-03 | 2.5210e+00 |

**The floor moves up in $h$ with the degree, and that is the whole case for
three divisors rather than one.** Roundoff enters as $(L/h)^{|k|}$ while
truncation still falls as $(h/L)^6$ after both Richardson steps, so the
crossover shifts coarser at every order. The cost of getting it wrong is
asymmetric and large: $|k| = 6$ run at $|k| = 4$'s divisor is 36× worse
(4.3e-2 against 1.2e-3), and at $L/64$ it is off by three orders. A single
tolerance across the three degrees would have had to be sized for $|k| = 6$ and
would have loosened $|k| = 4$ by four orders — the outcome step 4 forbids.

Both regimes are visible at every degree, which is what says the floor is a
real minimum and not the end of the scanned range. Note $L/3$ at $|k| = 5$ and
$6$ being *worse* than $L/2$: at $h \sim L$ the seven-point stencil samples
$\varphi$ out to $\pm 3h$, far outside the region the Taylor expansion the
truncation estimate assumes is valid in, so the "truncation-limited, falling as
$(h/L)^6$" description only starts holding from about $L/4$ down.

**Measuring on-machine rather than with a replica mattered.** §T1 sized the
shipped divisor with a standalone replica that reported 4.3e-7 at $|k| = 4$
where the binary reports 7.321e-7 at the same $h$ — 1.7× optimistic, from
`-ffp-contract` and libm. At the 10× margin rule used below, that 1.7× is most
of a degree's headroom.

### Pinned, and the margins achieved

`fd_h_divisor[3]` and `fd_tol[3]`, both indexed by degree − 4, both named on the
assertion together. From **job `f3Ze6Hh1Fp8X`** (rc 0, 12/12) and reproduced
exactly by **job `f3Ze8yUzNVSo`** (rc 0, 12/12, the post-revert re-run on the
tree as committed):

| $\vert k\vert$ | divisor | tol | achieved | multi-index | $(r, b)$ | margin |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | $L/24$ | 4e-6 | 3.080e-07 | $(4,0,0)$ | $b = 10^{-6}$, $\vert r\vert/\sqrt b = 74.2$ | 13.0× |
| 5 | $L/24$ | 3e-4 | 2.531e-05 | $(5,0,0)$ | $b = 6.25\times10^{-4}$, $\vert r\vert/\sqrt b = 100$ | 11.9× |
| 6 | $L/16$ | 2e-2 | 1.184e-03 | $(6,0,0)$ | $b = 6.25\times10^{-4}$, $\vert r\vert/\sqrt b = 15.46$ | 16.9× |

The rule, stated in the file: **the smallest one-significant-digit value at
least 10× the measured worst.** Chosen to match T1's 13.7× rather than to round
to a decade, because a decade would have given $|k| = 6$ an 84× margin — a
tolerance a regression could drift a long way inside of. No tolerance was
widened to make a degree pass; none needed to be.

Every worst case is at a **single-axis** multi-index, $(n,0,0)$ at all three
degrees. That is the oracle's own resolution limit rather than the recurrence's:
a single-axis derivative uses the widest 1-D stencil (7 points, weight sum 64 at
order 6) and the largest $1/h^n$, so it carries the most cancellation. Mixed
indices like $(2,2,2)$ spread the same total order over three narrow stencils
and come out better.

**$|k| = 4$ ended up sharper than it shipped**, 3.080e-07 against T1's
7.321e-07. This is not the interior scales: $L/32$ still measures
7.3209e-07 on the 76-sample set, matching T1 and T4 to four digits. It is that
T1's four-point scan $\{8, 16, 32, 64\}$ straddled a floor that sits at 24.

**The deviation grows about 40× per degree** (3.1e-7 → 2.5e-5 → 1.2e-3). Read
this as oracle resolution, not recurrence conditioning: it tracks the roundoff
floor of a $|k|$-th difference, which is exactly what rises with $|k|$, and the
perturbation figures below show the recurrence itself is being resolved to well
inside each bound.

### `closed_forms`, re-pinned

Both bodies read `buildSamples`, so the interior scales moved this one too:
**76 samples, worst 5.465e-15 at $k = (3,0,0)$** against the unchanged
$10^{-12}$, a 183× margin. The 40-sample 2.911e-15 is not carried forward. The
figure now appears in the body's header comment as well as in its printf, which
it did not before. It agrees with §T4's one-off measurement of the same thing.

### The perturbation, at each degree

**Job `f3Ze7qtDXeRD`, rc 8.** The §3 coefficient $k_j(k_j-1)$ multiplied by
1.001 — one coefficient, 0.1%, and deliberately *not* T1's perturbation, so the
two cover different terms of the recurrence. A 0.1% error is also a far subtler
probe than T1's $-2 \to -1$, which is the point: the question is whether degrees
5 and 6 are exercised, not whether they notice a broken recurrence.

| $\vert k\vert$ | perturbed deviation | tolerance | over by |
| --- | --- | --- | --- |
| 4 | 2.600e-02 | 4e-6 | 6500× |
| 5 | 2.739e-01 | 3e-4 | 913× |
| 6 | 2.843e+00 | 2e-2 | 142× |

All three fire, so the two new degrees are exercised and no single low-degree
slot is carrying the failure. `closed_forms` also failed, at $(1,0,2)$,
$|k| = 3$ — correct, since the perturbed term first contributes at target degree
3 (it needs $k_j \ge 2$, so $|k| \ge 2$, so $|k+e_i| \ge 3$), which is also why
$|k| \le 2$ stayed green. `m2l_ell0_closed_forms` and `m2l_p1_contraction`
failed downstream; `index_map_bijection` correctly did not.

**Reading the per-degree evidence required a second temporary edit.**
`ASSERT_LE` inside the sample loop aborts the body at the first failure, which
under a perturbation is always degree 4 — the run would have proved degree 4 and
said nothing about 5 and 6. For the perturbation job only, the in-loop assertion
was suppressed and three `EXPECT_LE(worst[d], fd_tol[d])` added after the
per-degree printf, giving one failure line per degree. Both that edit and the
`src/` perturbation were reverted **by inverting them**, never by
`git checkout`, which would have discarded this task's uncommitted work;
`src/Canopy_CartesianTaylorBasis.hpp` verified identical to `git HEAD` and
`tests/tstCartesianTaylor.hpp` byte-identical (`cmp`) to the pre-perturbation
checkpoint, and `f3Ze8yUzNVSo` then reproduced `f3Ze6Hh1Fp8X`'s figures exactly
on that tree.

### Flux jobs

| job | what | rc |
| --- | --- | --- |
| `f3Ze4jwB9ggF` | the 11-divisor scan; `finite_difference` red on purpose, still on placeholder tolerances | 8 |
| `f3Ze6Hh1Fp8X` | the pinned divisors and tolerances, scan loop deleted | 0 |
| `f3Ze7qtDXeRD` | the 0.1% perturbation of $k_j(k_j-1)$ + the per-degree probe | 8 |
| `f3Ze8yUzNVSo` | re-run after both edits were inverted, on the committed tree | 0 |

All four via `scripts/tuolumne/run_ctest_cartesian_taylor_serial.flux`,
**unchanged**. Its header comment lists only T1's three bodies and says T2 and
T3 reuse it, while the target now has twelve — stale, deliberately left for
whoever next edits that file, since it is not a T6 edit.

### R8

**Closed for $p = 3$.** The $\theta = 0.3$ gating arm of
`tstCartesianTaylorSolve.hpp` runs at $p = 3$, its ladder runs to $|k| = 6$, and
every degree of it is now measured against an oracle that is independent of the
§3 recurrence, at the $(r, b)$ scales the arm actually uses, with a
double-digit margin at each degree and a demonstrated failure at each degree.
R8's predicted presentation — the FD check failing above $|k| = 4$, worst at
small $|r|/\sqrt b$ where $w$ is smallest — did not occur: the ladder holds, and
the growth with $|k|$ that is present is the oracle's, not the recurrence's.

**R8 remains open above $p = 3$**, on the same terms as before, with one
improvement: the oracle no longer degrades silently there. `stencil1D` aborts on
per-axis order 7 naming the order, so a session that raises $p$ gets a
diagnostic instead of a mis-differenced comparison. Raising $p$ means adding
stencils, re-scanning the divisor at each new degree, and re-pinning — the
divisor table above is the shape to expect, not figures to reuse.

**Affects:**

- **Any session raising $p$ above 3** — do all three of: add the per-axis
  stencils (the abort tells you which order), extend `fd_max_k` and the two
  length-3 constant arrays, and re-scan the divisor **per degree** on-machine.
  The floor moves coarser with every order; do not carry $L/16$ over.
- **Anyone re-measuring $|k| = 4$ against T1's record** — the shipped divisor is
  now $L/24$ and the figure 3.080e-07, not $L/32$ and 7.321e-07. T1's number is
  still reproducible at $L/32$ on the current sample set and the log above says
  so; the change is the divisor, not the arithmetic.
- **`closed_forms`' worst figure is 5.465e-15 over 76 samples.** Any later
  session adding to `buildSamples` moves both bodies and must re-pin both, in
  the printf *and* in the two header comments.
- **Nothing in `src/` changed**, so no solve figure anywhere in this log moves.
