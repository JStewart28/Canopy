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
