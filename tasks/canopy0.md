# Canopy as the far-field engine for a distributed vortex-sheet solver

**Status:** IN PROGRESS — the survey below is complete; every task is NOT STARTED.

## Problem

A downstream application wants to use Canopy as the far-field engine for a
**distributed unstructured vortex-sheet solver**. That consumer is unlike the
workloads Canopy has been driven with so far, in five ways that all bear on the
API:

1. **Its kernel is a softened $1/r^2$ field, evaluated as a contraction.**
   The quantity it needs is not a potential and not a force; it is
   $\sum_s K(x_t,y_s) \times S_s$ with
   $K = \delta/(b + r^2)^{3/2}$, $\delta = x_t - y_s$,
   $r^2 = |\delta|^2$ — the Birkhoff–Rott integral — and, in a second
   configuration, the dot-product contraction $\sum_s (\delta \cdot G_s)/(b + r^2)^{3/2}$
   against a different vector source. The softening $b$ is not
   a numerical guard: it is the sheet thickness, a physical parameter of the
   model, and the reference answer the solver is validated against is a direct
   sum of exactly that kernel. A representative value is
   $b = \varepsilon^2 = 6.25\times10^{-4}$, i.e.
   $\varepsilon = \sqrt b = 2.5\times10^{-2}$, on a domain of extent $O(1)$.

2. **Its sources are owned entities of a distributed mesh, not free particles.**
   Each source is a mesh vertex owned by exactly one rank under a decomposition
   the *mesh* owns. The result must land back on that vertex, on that rank. The
   solver cannot adopt a decomposition chosen by the far-field engine, because
   every other operator in the solver — differentiation, remeshing, halo
   exchange, I/O — is expressed in the mesh's own layout.

3. **It calls the far field three times per timestep, not once.** Its time
   integrator is a three-stage Runge–Kutta; each stage moves every source and
   then requires a fresh evaluation. So a "one setup, one solve per step"
   cost model does not describe it: whatever the engine needs between
   evaluations is paid three times per step, not once.

4. **Its sources lie on a two-dimensional sheet embedded in three dimensions**,
   not in a volume. At late time the sheet rolls up and approaches itself, so
   the *geometric* separation between two parts of the surface falls toward
   $\sqrt b$ while the *along-surface* distance between them stays large.

5. **It is validated against a direct summation of the same kernel** at a
   tolerance it must state in advance. Its direct solver's own reproducibility
   floor across rank counts is $\sim\!10^{-15}$ relative, so the achievable
   comparison tolerance is set entirely by the far-field engine's accuracy, and
   the consumer needs that number to be *knowable*, not discovered by tuning.

This document records a **read-only survey** of Canopy answering five specific
questions from that consumer, and then one task per gap the survey found. The
survey opened no other dependency and changed no Canopy code.

The consumer's reference algorithm is a Barnes–Hut treecode
(`~/research-bridges/zmodel-steve/zmodel3d-amr/zmodel3d/treecode.py`), described
in `~/spack_envs/tuolumne_beatnik/beatnik/tasks/treecode.md`. Canopy's softened
far field, `CartesianTaylorBasis`, was built and measured under its own design,
`tasks/cartesian-taylor-basis.md`; the accuracy figures this document quotes for
it are that design's measurements. Every figure quoted for `LaplaceKernel` with
softening enabled is an estimate — C1 is the task that measures it.

**Out of scope:** the consumer's own adapter layer; any change to the consumer's
configuration surface; performance work not implied by one of the tasks below;
and the choice between Canopy and any other far-field method.

## Read this first

Four assumptions a reader coming to Canopy from that consumer's side is likely
to hold, and what is actually true.

**"The multipole far field expands the softened kernel."** Only if the caller
selects the basis that does. The far-field basis is the `FarField` template
parameter of `Solver` (`src/Canopy_Solver.hpp:165-174`), and its default,
`LaplaceKernel`, expands the **unsoftened** $1/r$ kernel. With that default,
softening exists only in the near-field P2P kernel
(`src/Canopy_P2P.hpp:799-803`, `883-896`) and is kept relevant by a floor that
pushes close pairs out of M2L and into P2P (`src/Canopy_Solver.hpp:71-78`,
`src/Canopy_CommunicationPlan.hpp:353-367`). The second basis,
`CartesianTaylorBasis` (`src/Canopy_CartesianTaylorBasis.hpp:355-356`), expands
the softened kernel $(r^2+b)^{-1/2}$ itself, and runs at
`near_softening_factor = 0`. See **F1**, **F6** and **C1**.

**"So a softened far field means a patch to the existing operators."** It does
not. `LaplaceKernel` implements all five operators in the solid-harmonic basis
with Greengard theorem citations on each — `p2m_contribution`
(`src/Canopy_LaplaceKernel.hpp:403`), `m2m_translate` (`:446`, Thm 5.22),
`m2l_translate` (`:551`, Thm 5.23), `l2l_translate` (`:1220`, Thm 5.26),
`l2p_evaluate` (`:1337`), plus a precomputed-operator M2L path
(`build_m2l_operators`, `:887`). Those theorems cannot carry softening, which is
why the softened far field is a separate basis rather than a change to these
routines. See **F6**.

**"`setSources` then `solve` maps onto Canopy's `setup` then `solve`."** It maps
onto `setup`/`auto_maintain` plus `solve`, and the split is not where a caller
would put it. `solve()` reads current positions but uses the leaf membership,
communication plan and P2P neighbour lists cached by the *last* setup or
maintenance call (`src/Canopy_UpwardSweep.hpp:700-705` iterates
`_leaves_at_depth_local`; `src/Canopy_P2P.hpp:819`, `:834` use cached
`_leaf_particle_offsets` / `_particle_to_league`). Calling `solve()` after moving
particles, without a maintenance call, does not fail — it silently evaluates a
wrong near/far partition. See **F3** and **C3**.

**"Canopy evaluates at the caller's particles, in the caller's order."** It
evaluates at *its own* particles: `setup()` and every maintenance path migrate
particles across ranks and permute the local array, and
`TreePartitioner::migrate_particles` states outright that "the within-AoSoA order
after migration is unspecified" (`src/Canopy_TreePartitioner.hpp:854-858`). There
is no identity, global-ID, or inverse-permutation facility anywhere in `src/`.
See **F3** and **C2**.

## Findings

### F1 — Kernel generality and the form of the softening

**Canopy exposes two far-field bases, chosen at compile time.** `Solver` takes
`template <class, int, int> class FarField = LaplaceKernel` and sets
`kernel_type = FarField<Scalar, P_ORDER, NComps>`
(`src/Canopy_Solver.hpp:165-174`). A `FarField` type is a basis-plus-kernel
composition that supplies every far-field operator (`README.md:15-33`); the
contract is checked by `src/Canopy_FarFieldContract.hpp`. Two exist:

- `LaplaceKernel` (`src/Canopy_LaplaceKernel.hpp:153-155`), the default — a
  complex solid-harmonic expansion of the unsoftened $1/r$ kernel, any
  `Scalar`, any `P_ORDER`.
- `CartesianTaylorBasis` (`src/Canopy_CartesianTaylorBasis.hpp:355-356`) — a
  real Cartesian-Taylor expansion of the softened $(r^2+b)^{-1/2}$, `double`
  only (`:360`), with `P_ORDER` read as the Taylor order $p$.

The kernel is not otherwise pluggable: a consumer selects one of these two, not
an arbitrary kernel.

**What it can express instead is a fixed *set* of Laplace solves.** `NComps` is
the number of simultaneous independent charge components, and the header already
names this consumer's use case: "3 for Biot-Savart via three parallel Laplace
solves" (`src/Canopy_LaplaceKernel.hpp:136-138`). With `NComps = 3` and
`compute_gradient = true`, one traversal produces a $3\times3$ tensor per
target (`src/Canopy_DownwardSweep.hpp:174-180`, gradient shaped
`(num_particles, NComps, 3)`).

**The softening is the same functional form the consumer needs, not merely an
analogue.** Canopy's near field is Plummer softening applied as
$r^2 \rightarrow r^2 + \varepsilon^2$ in every pairwise term
(`src/Canopy_P2P.hpp:799-803`), and the gradient it accumulates is

```
inv_r  = 1 / sqrt(r2 + eps2);   inv_r3 = inv_r^3
g[c]  -= q[c] * d * inv_r3                       // d = x_target - x_source
```

(`src/Canopy_P2P.hpp:883-896`). That is exactly
$-\sum_s q_s\,\delta/(r^2+\varepsilon^2)^{3/2}$: the consumer's
$K = \delta/(b+r^2)^{3/2}$ with $\varepsilon^2 = b$, up to an overall
sign. A consumer therefore sets `FmmConfig::softening = sqrt(b)` — an explicit
non-negative value, which also suppresses the distribution-based auto-softening
that would otherwise be derived once at first setup and frozen
(`src/Canopy_Solver.hpp:196-209`, `806-847`). No functional-form conversion is
needed and no reinterpretation of $b$ is needed.

**With `LaplaceKernel`, only the near field is softened.** This is the single
most consequential property of the default configuration. Its multipole far
field expands $1/r$ unsoftened;
accuracy in the far field is bought by *excluding* every pair where softening
matters, via a floor in the acceptance criterion: an M2L pair is rejected
whenever the cell-centre separation $R \le \texttt{near\_softening\_factor} \cdot \varepsilon$
(`src/Canopy_CommunicationPlan.hpp:353-367`). A rejected
pair falls through the dual-tree traversal to a leaf-leaf P2P pair
(`src/Canopy_CommunicationPlan.hpp:629-640`), so the near/far partition stays a
partition and the close pairs do get the softened kernel.

For pairs that *are* taken by M2L, the evaluated kernel is not the consumer's
kernel, and the difference is a **systematic bias that does not shrink with
expansion order**. Per pair, at separation $R$:

| quantity | exact vs. unsoftened relative error |
| --- | --- |
| potential $1/\sqrt{R^2+\varepsilon^2}$ | $\approx \tfrac12 \varepsilon^2/R^2$ |
| gradient $\delta/(R^2+\varepsilon^2)^{3/2}$ | $\approx \tfrac32 \varepsilon^2/R^2$ |

The bound quoted in `README.md:64-74` — "far-field relative softening error
`~ 1/(2·factor²)`, ≈3% at the default `4`" — is the **potential's**. A consumer
that uses the gradient (as this one does; both of its contractions are gradient
contractions) sees three times that: $\tfrac32/\text{factor}^2 \approx 9.4\%$
at the default `near_softening_factor = 4`. Inverting for a target far-field
fidelity $\tau$ gives a required exclusion radius
$R > \varepsilon\sqrt{1.5/\tau}$:

| target $\tau$ | required `near_softening_factor` | exclusion radius at $\varepsilon = 2.5\times10^{-2}$ |
| --- | --- | --- |
| $10^{-2}$ | 12.2 | 0.31 |
| $10^{-3}$ | 38.7 | 0.97 |
| $10^{-6}$ | 1225 | 30.6 |

On a domain of extent $O(1)$ the $10^{-3}$ row already makes the near
field the entire domain — i.e. $O(N^2)$ — and the $10^{-6}$ row is
unreachable at any cost. **There is no setting of `near_softening_factor` that
delivers a $10^{-6}$-accurate softened-kernel evaluation with a bare-kernel
far field.** With `LaplaceKernel`, the consumer's accuracy claim is bounded at
the $10^{-2}$–$10^{-3}$ level; measuring and documenting that bound is task
**C1**. A far field that carries the softening is `CartesianTaylorBasis`, and
what it achieves and costs is **F6**.

Two smaller notes on the kernel:

- P2P skips any pair with $r^2 < 10^{-24}$ (`src/Canopy_P2P.hpp:881`).
  For a consumer whose target set equals its source set this is the
  self-interaction, whose contribution to the *gradient* contraction is exactly
  zero ($\delta = 0$), so skipping it is correct rather than merely
  tolerable. It also silently drops genuinely coincident distinct sources, which
  for this kernel likewise contribute zero to the gradient.
- The sweeps are kernel-blind. The effective softening length reaches them only
  as `M2LKernelParams` (`src/Canopy_FarFieldContract.hpp:114`), pushed by
  `Solver::_push_m2l_kernel_params` (`src/Canopy_Solver.hpp:777-783`) into
  `set_m2l_kernel_params` on each sweep (`src/Canopy_UpwardSweep.hpp:195-198`,
  `src/Canopy_DownwardSweep.hpp:399-406`). `LaplaceKernel` ignores it
  (`src/Canopy_LaplaceKernel.hpp:319`, `:898`); `CartesianTaylorBasis` uses it
  as $b = \varepsilon^2$.

### F2 — The cross-product contraction

**It cannot be folded into the kernel, and it does not need three separate
solves.** Neither basis has a hook for a caller-supplied contraction — the
`FarField` operators (listed for `LaplaceKernel` at
`src/Canopy_LaplaceKernel.hpp:145-150`) are fixed. But `NComps = 3` with `compute_gradient = true` yields, per target
$i$, the full tensor

$$
  T_{cj}(i) \;=\; \texttt{gradient}(i,c,j) \;=\; -\sum_s q_{c,s}\,
  \frac{\delta_j}{(r^2+\varepsilon^2)^{3/2}},
$$

from **one** tree traversal and **one** ghost exchange
(`src/Canopy_Solver.hpp:277-318`; `src/Canopy_DownwardSweep.hpp:174-180`). Load
the three charge components with the three components of the vector source, and
both contractions the consumer needs are purely local post-processing of that
$3\times3$ tensor:

- cross product: $u_i = -\epsilon_{ijk}\,T_{kj}$ — a nine-term local
  reduction, no communication;
- dot product: $\Psi = -\operatorname{tr} T = -\sum_c T_{cc}$ — a
  three-term local reduction.

So the two evaluations the consumer needs share the tensor *only* when they share
the source vector. They do not: the velocity contraction is against the sheet
strength and the scalar contraction is against a different vector field. That is
**two `solve()` calls with different charges over the same tree**, not one, and
both want the gradient and neither wants the potential. Canopy supports two
`solve()` calls over one tree directly (`solve()` re-reads the charge slice each
call and zeroes its outputs first, `src/Canopy_Solver.hpp:284-301`) — but it
always allocates, zeroes and accumulates the potential even when only the
gradient is wanted (`src/Canopy_Solver.hpp:288-290`; P2P accumulates
`phi[c]` unconditionally, `src/Canopy_P2P.hpp:891`). See **C8**.

Keeping the contraction *out* of the kernel is also the right design and not
merely the available one: the cross product is linear in the source strength, so
it commutes with any expansion of the kernel. The reference treecode bakes the
cross product into its own kernel evaluation
(`treecode.py:56-81`) and thereby loses that separation. `CartesianTaylorBasis`
keeps it: it delivers $\varphi$ and $\nabla\varphi$ per component and nothing
inside Canopy recombines them (`tasks/cartesian-taylor-basis.md`, "The far field
is three scalar passes, not a vector kernel").

`NComps` is a **compile-time** template parameter, as are `P_ORDER` and
`FarField` (`src/Canopy_Solver.hpp:165-167`). A consumer whose expansion order is a runtime
configuration value cannot pass it through. See **C7**.

### F3 — Tree reuse across integrator stages

Answered against the real `setup`/maintenance/`solve` split. Three separate
problems, in increasing severity.

**(a) `solve()` after motion, with no maintenance, is silently wrong.** The
upward sweep runs P2M over the leaf lists cached at setup
(`src/Canopy_UpwardSweep.hpp:700-705`) and P2P uses the cached particle→leaf
mapping (`src/Canopy_P2P.hpp:819`, `:834`); the M2L interaction list is likewise
cached and only invalidated on a topology change
(`src/Canopy_Solver.hpp:641-643`, `688-689`). So a particle that has moved out of
its leaf still contributes to its old leaf's multipole and still gets its old
leaf's near-field list. No error is raised. The failure mode is a plausible,
slightly wrong field — the worst kind for a consumer whose only check is a
tolerance.

**(b) The cheapest maintenance path is not cheap.** `migrate()` — documented as
"cheapest maintenance … tree topology assumed unchanged"
(`src/Canopy_Solver.hpp:320-332`) — performs, per call:

- a full `TreeBuilder::build()` from current positions
  (`src/Canopy_Solver.hpp:347-350`), which is two `MPI_Allreduce` for the
  bounding box (`src/Canopy_TreeBuilder.hpp:491-492`) plus **one
  `MPI_Allreduce` per octree level** over the candidate-cell counts
  (`src/Canopy_TreeBuilder.hpp:896-897`), with the per-level candidate marshalling
  and the resulting globally-replicated cell list assembled on the host
  (`src/Canopy_TreeBuilder.hpp:812-935`);
- a host-side `std::unordered_set` over every cell key, twice, to detect
  topology change (`src/Canopy_Solver.hpp:341-365`);
- a particle redistribution (`src/Canopy_Solver.hpp:374-377`);
- then `_finish_topology_stable`, which builds the tree **again**, re-sorts the
  array by leaf, and re-runs all three sweep setups
  (`src/Canopy_Solver.hpp:700-726`).

There is no API for "the positions moved by less than a cell width; refresh only
the leaf membership and the P2P offsets". For a consumer paying this three times
per timestep, that is the difference between a usable and an unusable far field.
See **C3**.

**(c) On a deforming surface, `auto_maintain` will pick `Rebalance`, not
`Migrate`, almost every time.** The decision is: rebuild on bounding-box escape,
else `Rebalance` if *any* cell key differs from the previous tree, else
`Migrate` (`src/Canopy_Solver.hpp:426-545`). A rolling-up sheet changes its
occupancy pattern continuously, so the cell-key set changes essentially every
stage. `Rebalance` adds a ParMETIS repartition and a full communication-plan
rebuild — and the communication-plan rebuild is a **serial host-side dual-tree
traversal over the globally replicated cell tree, executed on every rank**
(`src/Canopy_CommunicationPlan.hpp:555-671`). So the expensive path is the
common path, three times per step. With `CartesianTaylorBasis` it is more
expensive still: a rebalance that moves the root box empties that basis's whole
M2L operator cache, so every operator is rebuilt (**F6**).

**The decomposition is reproducible run to run; a HIP solve is not.**
`TreePartitioner::partition_cells` runs a distributed
`ParMETIS_V3_PartKway` / `ParMETIS_V3_AdaptiveRepart` with a fixed seed
(`src/Canopy_TreePartitioner.hpp:582-603`) and shares the result with
`MPI_Allgatherv` (`:614-625`). `README.md:529-547` records the consequence: two
`MultiSolve` passes at SERIAL np 2–6 and HIP np 2–4 give the identical ownership
map on every partition and refresh, and on SERIAL the solve output is identical
between passes at every rank count. On a HIP `ExecutionSpace` it is not — device
reductions accumulate in a run-dependent order, giving relative differences of
6e-12 to 2e-4 between passes at np 2–4, and differences even at np 1. So a SERIAL
run supports a bitwise claim at fixed rank count; a HIP run supports none, and
needs a measured noise floor before any tolerance is set against it. See **C4**.

**(d) Every maintenance path re-decomposes and permutes.** `partition`,
`repartition` and `redistribute` all migrate particles between ranks and reorder
the local array, and the order after migration is explicitly unspecified
(`src/Canopy_TreePartitioner.hpp:854-858`). Nothing in `src/` carries a
caller-supplied identity through that: the only global IDs in the tree are
ParMETIS vertex numbers for cells, assigned by supplier rank and Morton position
(`src/Canopy_TreePartitioner.hpp:480-496`). Migration packs whole AoSoA tuples
(`src/Canopy_TreePartitioner.hpp:945-1050`), so a caller-added identity member
*would* travel with its particle — but the caller must then run its own reverse
exchange to get results home, once per stage, in addition to Canopy's. For a
consumer whose decomposition is fixed by a mesh, this is the interface's
central mismatch. See **C2**.

### F4 — `ncrit`, `mac_theta`, `max_depth` on an unstructured sheet

| knob | in Canopy | notes for a consumer |
| --- | --- | --- |
| `ncrit` | `FmmConfig::ncrit` (`src/Canopy_Solver.hpp:55`), runtime. A cell becomes a leaf when its **global** count $\le$ `ncrit` or it hits `max_depth` (`src/Canopy_TreeBuilder.hpp:922`). | Same meaning as in any Barnes–Hut/FMM code; a value of 64 transfers directly. Note the refinement test is on the *global* count, so leaves are balanced in total occupancy, not per-rank occupancy. |
| `mac_theta` | `FmmConfig::mac_theta` (`src/Canopy_Solver.hpp:68`), runtime, default 0.5. The predicate is the exafmm spherical MAC: accept M2L iff $R\theta > \sqrt3\,(h_A + h_B)$, with exact ties rejected by a $10^{-10}$ relative margin (`src/Canopy_CommunicationPlan.hpp:338-353`). | A **different predicate** from a Barnes–Hut opening angle, so an inherited numeric value does not carry its meaning across. It does carry its *direction*: smaller is more conservative, more M2L pairs, more accurate at fixed order. A value of 0.3 is conservative under Canopy's predicate too, and is exercised by one existing test (`tests/tstMultiSolve.hpp:1197-1212`). |
| `max_depth` | `FmmConfig::max_depth` (`src/Canopy_Solver.hpp:56`), runtime, **hard maximum 19**, enforced by a throw in the `TreeBuilder` constructor because the Morton key is a `uint64_t` (`src/Canopy_TreeBuilder.hpp:50`, `251-256`). | Has **no counterpart** in a treecode-derived parameter set; a consumer must choose it. It is coupled to the bounding box: the finest cell width is (root box width) / $2^{\text{max\_depth}}$. |

**Do the structured-workload values carry over to a sheet?** Partly, and the
part that does not is untested. The tree is *count*-adaptive, so a
two-dimensional source support in a three-dimensional box is handled without
special-casing: empty cells are dropped outright
(`src/Canopy_TreeBuilder.hpp:907-908`), and reaching $N/\texttt{ncrit}$
leaves on a surface costs roughly $\log_4(N/\texttt{ncrit})$ levels rather
than $\log_8$, i.e. a *deeper* tree than a volumetric distribution of the
same count — which is what makes the depth-19 ceiling worth checking rather than
assuming. Three sheet-specific effects have no coverage:

- **Anisotropic leaf occupancy.** A cube cell straddling a sheet has its sources
  on a plane through it, so the circumradius $\sqrt3 h$ used by the MAC
  overstates the actual source extent. The MAC stays conservative (safe), but
  the achieved accuracy at a given `mac_theta` is not the volumetric one, and no
  test measures it. **There is a cheap, kernel-independent improvement here.**
  The reference treecode uses the *exact* node radius $\max_j |y_j - c|$
  (`treecode.py:31`) where Canopy uses the geometric
  $\sqrt3\,(h_A+h_B)$ (`src/Canopy_CommunicationPlan.hpp:345`). The exact radius
  is tighter on a sheet by construction, is computable per cell in the upward
  sweep at negligible cost, and would reduce the number of pairs demoted to P2P
  at fixed accuracy — a win that is independent of every kernel question in F1
  and F6. It interacts with the near-softening floor, so it must be **measured**
  as part of **C5** rather than assumed.
- **Self-approach at roll-up.** Two along-surface-distant parts of the sheet come
  within $\sqrt b$ geometrically. `README.md:64-74` already names exactly
  this case — "a clustering system whose cells shrink below the softening length
  (e.g. a vortex sheet at full roll-up) gets a spurious, far too large far-field
  and blows up" — and with `LaplaceKernel` the near-softening floor is the
  mitigation. So the mechanism is anticipated; what is unmeasured is the
  *cost*, since the floor converts a growing fraction of pairs to P2P as the
  sheet tightens. `CartesianTaylorBasis` needs no floor, so on that basis the
  self-approach costs nothing extra in P2P; what is unmeasured there is the
  accuracy once the softening dominates the cell width.
- **Bounding-box sensitivity.** `TreeBuilder::compute_global_bounding_box` takes
  a raw global min/max (`src/Canopy_TreeBuilder.hpp:491-492`), and
  `README.md:321-332` records that a single outlier inflates the root box until
  a dense cluster collapses into one max-depth leaf, making the
  $O(N_{\text{leaf}}^2)$ P2P effectively hang. A surface that develops a
  single spurious vertex hits this.

**Every accuracy figure Canopy currently carries was measured on a uniform or
clustered *volumetric* random distribution.** `tests/tstSingleSolve.hpp`,
`tests/tstMultiSolve.hpp` and `tests/tstCartesianTaylorSolve.hpp` place
particles randomly in a box; none has a surface, sheet or manifold case
(`grep -i "surface\|sheet\|manifold" tests/` returns only unrelated prose).
Only the `CartesianTaylorBasis` figures are softened. The validated envelope is:

| test | configuration | tolerance met |
| --- | --- | --- |
| `SingleSolve.PotentialAndGradientNComps3` | $P=8$, `ncrit` 16, `max_depth` 6, softening 0, 500 particles/rank, volumetric random; all nine gradient components vs. brute force (`tests/tstSingleSolve.hpp:79-87`, `365-375`, `413-432`) | $10^{-3}$ — but **fails at exactly 4 ranks**, see F5 |
| `MultiSolve` suite | `LaplaceKernel`, $P=8$ (`tests/tstMultiSolve.hpp:88`), `mac_theta` 0.3–0.5, `ncrit` 8–16, `max_depth` 6–8, `softening = 0` (`:432`, `:1301`, `:1561`); end-of-run position and velocity against a brute-force shadow trajectory, per test (`tests/tstMultiSolve.hpp:1041-1229`) | position $3.5\times10^{-9}$–$6.4\times10^{-3}$, velocity $3.4\times10^{-6}$–$3.5\times10^{-2}$, by test |
| `CartesianTaylorSolve` | `CartesianTaylorBasis`, `NComps` 3, 8640 particles on a cube of half-span 0.1155, `ncrit` 8, `max_depth` 6, `softening` 0.025, `near_softening_factor` 0, 4 solves with migrate/rebalance between, np 1–6; vs. a direct **softened** sum (`tests/tstCartesianTaylorSolve.hpp:135-155`, `:326-368`) | $p=3$, `mac_theta` 0.3: potential $1.9\times10^{-5}$, gradient $7.1\times10^{-4}$ (bar $10^{-3}$; at $p=2$ the gradient is $9.0\times10^{-3}$); $p=2$, `mac_theta` 0.5: potential $1.0\times10^{-3}$, gradient $1.87\times10^{-2}$. Global-scale-normalized; SERIAL np 1–6 and HIP np 1–4 |

There is therefore **no measured parameter set for a sheet, and no measured
accuracy figure for `LaplaceKernel` with softening enabled**. Producing one requires
compiling and running; it cannot be settled by reading. This is task **C5**, and
it is the task that decides whether the consumer's whole approach is viable.

### F5 — Open defects on this path

**The `Rebalance` NIC-registration-cache defect is fixed.**
`TreePartitioner::migrate_particles` no longer routes through
`Cabana::Distributor`/`Cabana::migrate`; it packs outgoing tuples into per-peer
subviews of **one** persistent registered send region and posts one
`MPI_Isend` per peer, with matching receives in one persistent recv region, so
peak concurrent registrations are $O(1)$ per direction regardless of peer
count (`src/Canopy_TreePartitioner.hpp:825-858`,
`src/Canopy_RegisteredBufferPool.hpp:24-56`). The same pooling already covers the
M2L/L2L `coalesced_view_exchange` and the P2P ghost gather. `README.md:260-282`
records this as fixing the `dreg_evict NO_SPACE` deadlock on many-way
migrations, and as removing the need for a patched Cabana fork (the MPI element
type is one whole tuple, so a single peer's payload may exceed 2 GiB without
overflowing MPI's signed-`int` count).

This mattered acutely for this consumer, because per F3(c) `Rebalance` is its
*common* path rather than a rare one. As fixed, the defect does not affect it.
One residual remains, and it is a scaling concern rather than a defect: peer
discovery in migration is a single `MPI_Alltoall` of `comm_size` ints on every
`Rebalance` (`README.md:341-357`), which for a three-stage integrator is three
$O(\text{comm\_size})$ collectives per timestep on top of everything else.
That is folded into **C3** as motivation, not raised as its own task.

**Two other open defects sit on or beside this path.**

- **The exact code path this consumer needs is out of the regression gate.**
  `SingleSolve.PotentialNComps3` and `SingleSolve.PotentialAndGradientNComps3`
  — three components, gradient, versus brute force — fail at exactly 4 ranks
  (`max_pot_rel_err = 0.00196` vs. a $10^{-3}$ budget), pass at 1, 2, 3, 5
  and 6, and the whole `SingleSolve` binary additionally leaks state that
  deadlocks a later test in the same `ctest` process; it is consequently labelled
  `unit`, not `regression` (`README.md:614-642`). So the multi-component
  gradient path — the one thing this consumer's correctness rests on — is
  neither gated nor correct at one of the rank counts it will be run at. This is
  **C6**.
- The FP32 fused-M2L failure at $\ge 2$ ranks (`README.md:562-586`) does
  **not** affect a consumer running in double precision, and no task is raised
  for it here.

**One coverage gap rather than a defect:** every test in `tests/` gives every
rank the same non-zero particle count. A consumer distributing an unstructured
mesh can have a rank that owns **zero** sources, and must still enter every
collective. Nothing in `src/` obviously mishandles it — the per-level
`MPI_Allreduce` sums zero local counts, the replicated cell list is identical on
every rank, and the ParMETIS solve runs on a communicator split to exclude
ranks that supply no vertices (`src/Canopy_TreePartitioner.hpp:559-567`) — but
"reads as though it should work" is not coverage. This is **C10**.

### F6 — The softened far field, and what it costs

`CartesianTaylorBasis` is Canopy's softened far field. Its design, derivations
and measurements are `tasks/cartesian-taylor-basis.md` and its progress log;
this finding records only what a consumer choosing between the two bases needs.

**(a) Why it is a separate basis.** `LaplaceKernel`'s M2M/M2L/L2L are the
solid-harmonic addition theorems (Greengard Thms 5.22, 5.23, 5.26, cited at
`src/Canopy_LaplaceKernel.hpp:446`, `:551`, `:1220`). Those theorems hold
*because* $1/r$ is harmonic. The softened potential is not:

$$
\nabla^2 (r^2+b)^{-1/2} \;=\; -\,\frac{3b}{(r^2+b)^{5/2}} \;\ne\; 0 .
$$

So no softened coefficient fed to `m2l_translate` evaluates the softened kernel.
A Cartesian Taylor basis needs only *smoothness* of the kernel, never
harmonicity, and carries $b$ inside $w = r^2 + b$ at every derivative order
(`src/Canopy_CartesianTaylorBasis.hpp:26-36`). Only its M2L knows the kernel;
P2M, M2M, L2L and L2P are kernel-blind Taylor shifts.

**(b) Storage.** The sweeps take their coefficient type from the basis —
`coeff_type = KernelType::coeff_type` and `View<coeff_type***>`
(`src/Canopy_UpwardSweep.hpp:70`, `:102-103`;
`src/Canopy_DownwardSweep.hpp:122`, `:171-172`), with the per-cell count from
`KernelType::num_coeffs_per_cell` (`src/Canopy_UpwardSweep.hpp:96`,
`src/Canopy_DownwardSweep.hpp:147`) and auxiliary tables from the opaque
`KernelType::build_aux_tables` (`src/Canopy_UpwardSweep.hpp:302-306`).
`LaplaceKernel` stores complex coefficients, $(P+1)(P+2)/2$ per cell, plus the
$A_{n,m}$ table to $2P$; `CartesianTaylorBasis` stores real ones,
$(p+1)(p+2)(p+3)/6$ per cell (`src/Canopy_CartesianTaylorBasis.hpp:379`).

**(c) Softening destroys scale invariance, and the operator cache pays for it.**
`LaplaceKernel` scale-normalizes every operator against cell half-width because
$1/r$ is homogeneous of degree $-1$: P2M produces $\bar M = M/w^{n+1}$
(`src/Canopy_LaplaceKernel.hpp:399-402`), M2M applies $(w_c/w_p)^{j+1}$
(`:440-445`), M2L expands $\rho^{-(n+j+1)}$ as $(w_s/\rho)^{n+1}(w_t/\rho)^j$
(`:547-550`), L2L applies $(w_c/w_p)^j$ (`:1217-1219`), L2P consumes
$\bar L = L\,w^n$ (`:1332-1336`). Its precomputed M2L operators therefore depend
only on the key $(dd, ii, jj, kk)$, with no physical length entering
(`:874-884`, `:892-895`), declared by `key_needs_level = false` (`:721`).

A softened kernel has a length scale, $\sqrt b$. `CartesianTaylorBasis` keeps
physical, un-normalized coefficients, is `double` only
(`src/Canopy_CartesianTaylorBasis.hpp:360`), and declares
`key_needs_level = true` (`:486`): its operator key carries the level, the sweep
hands its builder per-level physical half-widths (`set_root_half_width`,
`src/Canopy_DownwardSweep.hpp:413-450`), and a change of root half-width or of
softening empties the whole cache (`:376-406`). A rebalance on a moving
distribution moves the root box, so on this basis **no operator survives a
rebalance** — measured as zero keys retained across 336 builds
(`tasks/cartesian-taylor-basis.md`, Current state and **R6**). Per F3(c),
rebalance is this consumer's common path.

**(d) The basis that admits softening does not scale to high accuracy.** A
Taylor expansion truncated at order $p$ has relative error
$\sim (c\,W/R)^{p+1}$ with $W$ the source box width and $c \in [1, \sqrt3]$,
while its coefficient count grows as $\binom{p+3}{3} \sim p^3/6$ against
$O(p^2)$ for solid harmonics. Measured (F4's table): $7.1\times10^{-4}$ on the
gradient at $p = 3$, `mac_theta` 0.3; $1.87\times10^{-2}$ at $p = 2$, `mac_theta`
0.5. The gradient of a degree-$p$ local is degree $p-1$, so the gradient
truncates one order before the potential. $10^{-6}$ needs $p \approx 11$–$24$
(364 to 2925 coefficients per cell; `tasks/cartesian-taylor-basis.md`, "The
target is the regularization unblock"). So a softened far field lives in the
$10^{-3}$ regime, and $10^{-6}$ is out of reach at any order this basis is
practical at.

### F7 — `LaplaceKernel`'s L2P gradient is a finite difference, not analytic

`src/Canopy_LaplaceKernel.hpp:1388-1418` evaluates the far-field gradient by a
central difference: six extra potential evaluations at
$h = 10^{-5} w_{\rm self}$, with an in-code `TODO: replace with analytical
derivatives` and a comment recording that a previously *fixed* step size was the
root cause of a premature full-rollup NaN. Two consequences:

- **Part of the gradient error C1 and C5 are about to measure is
  finite-difference error, not softening bias and not truncation.** At the
  roundoff/truncation optimum the relative FD error is
  $\sim\!10^{-10}$–$10^{-11}$, so it does not threaten a $10^{-3}$
  budget — but it is a **third** plateau in the `P_ORDER` scan that C1 step 1
  and risk **R1** are built around, and R1's "truncation falls, bias plateaus"
  discriminator has to account for it.
- `CartesianTaylorBasis::l2p_evaluate` already returns an analytic gradient
  (`src/Canopy_CartesianTaylorBasis.hpp:1453-1490`). The finite difference is
  `LaplaceKernel`'s alone.

Replacing the FD with analytic solid-harmonic derivatives in `LaplaceKernel` is
independent of every kernel question above and is task **C11**.

## Approach

Each finding above that blocks or degrades the consumer becomes one task below.
The tasks are independent except where stated: **C1** and **C5** together decide
which basis the consumer should run and whether the approach is viable at all,
and should be done first; **C2** and
**C3** are the interface changes that make a three-stage integrator affordable;
**C4**, **C6** and **C10** are correctness and confidence work; **C7**, **C8**,
**C9** and **C11** are small API and quality items that can be taken at any
time, though C11 is worth doing *before* C1's scan is interpreted.

### Conventions

| Choice | Rule |
| --- | --- |
| Library style | header-only under `src/`, `Canopy_` prefix, `namespace Canopy` (`detail` for internals, as `src/Canopy_RegisteredBufferPool.hpp:21-23`) |
| Parallelism | Kokkos + Cabana + MPI; no serial-only signatures |
| Configuration | one new knob goes in `FmmConfig` (`src/Canopy_Solver.hpp:53-136`) with a defaulted member and a comment stating units and the meaning of the default; never a new constructor parameter |
| New parameters | prefer an enum or tag type over a bool or magic number; a mode selector is an enum |
| Failure behavior | a violated precondition throws (`std::runtime_error`, as `src/Canopy_TreeBuilder.hpp:251-256`); never return a truncated or best-effort field |
| Comments | state units, sign convention, and which side of a difference is which, on the declaration; the sign of the gradient output is the single most misread thing in this API |
| Provenance | cite the paper, section or upstream code any new operator is derived from, on the routine |
| Test tier | new correctness tests are `regression` and must pass at ranks 1–6; a test that cannot yet pass at all six is `unit` **and** its exclusion is recorded in `README.md` "Known Issues" with the rank counts that fail |
| Accuracy claims | every stated tolerance names the distribution, the rank counts, `P_ORDER`, `ncrit`, `max_depth`, `mac_theta` and the softening it was measured at. A tolerance without that list is not a claim |

### Deliberate deviations

- **No task adds a third far-field basis or a treecode (M2P) mode.** The
  consumer's kernel is reachable as a set of gradient solves (F2) on either
  existing basis, and `CartesianTaylorBasis` already carries the softening with
  analytic gradients at the reference treecode's accuracy (F4, F6). A treecode
  mode would reach the same $\sim\!10^{-3}$ at $O(N\log N)$ instead of $O(N)$.
- **No task makes HIP solves bitwise reproducible.** The partition is already
  deterministic (F3(c)); the remaining HIP spread comes from device reduction
  order, which is recorded in `README.md` Known Issues and is not specific to
  this consumer. C4 states the guarantee per backend instead.
- **The consumer's configuration surface is fixed and cannot absorb these
  gaps.** No task below may be closed by asking the consumer to add a knob.

## Current state

Everything described in **Findings** is the state of the library as surveyed.
Concretely, and stated as what is *not* true:

- The default basis, `LaplaceKernel`, does not carry softening in the far
  field, and nothing reports the resulting bias. A consumer that leaves
  `FarField` at its default gets a wrong-by-a-known-formula answer with no
  indication, and `README.md:64-74` understates the gradient's share of it by a
  factor of three (F1).
- `CartesianTaylorBasis` does carry it, measured at $7.1\times10^{-4}$ on the
  gradient (F4), but only on a volumetric distribution, and its operator cache
  is emptied by every rebalance that moves the root box (F6(c)).
- There is no accuracy measurement for `LaplaceKernel` with softening enabled,
  and none for either basis on a non-volumetric source distribution.
- `LaplaceKernel`'s far-field gradient is a finite difference of the potential,
  not an analytic derivative (F7).
- There is no way to get results back in the caller's particle order or on the
  caller's ranks.
- There is no maintenance path cheaper than a full global tree build.
- The three-component gradient path is not in the regression gate and is known
  wrong at 4 ranks.
- `solve()` called after motion without maintenance returns a
  **defined-but-wrong** field rather than raising. This is the most dangerous
  single property of the current API for a new consumer.

## Progress log

`tasks/canopy0-progress-log.md` holds what actually happened: the reasoning behind
decisions this document states flatly, measured numbers, and things only running
revealed. **Read it before implementing any task, changing any signature, or
reopening a question this document treats as settled** — in particular before
choosing a tolerance, since a measured number in the log always outranks an
estimate here.

## Task sequence

### C1 — Measure and document `LaplaceKernel`'s softening bias — **NOT STARTED**

**Depends on:** none. (Interacts with C11: C11 removes one of the three error
plateaus step 1 will see, so doing C11 first makes the scan easier to read.)

**Fill in:** one new test in `tests/` plus its registration in
`tests/CMakeLists.txt` under `REGRESSION_MPI_TESTS`; `README.md` (the
`near_softening_factor` paragraph, `:64-74`). No `src/` change.

**Reference:** the softened near-field kernel
(`src/Canopy_P2P.hpp:799-803`, `883-896`); the floor that keeps the unsoftened
far field usable (`src/Canopy_CommunicationPlan.hpp:353-367`); the error bound
and its potential-vs-gradient factor of three, tabulated in **F1**; the softened
basis, its measured accuracy and its cost, in **F4** and **F6**; the
finite-difference gradient, in **F7**. `tests/tstCartesianTaylorSolve.hpp` is
the pattern for a softened direct-sum comparison: its explicit positive
`softening` (`:44-47`), its global-scale normalization, and its
measured-then-pinned tolerance comment (`:326-368`). Its harness is
parameterized on the order and the basis precisely so a `LaplaceKernel` arm is
an added `TEST` body (`:408-420`), and its comment at `:194-204` records that
arm already: at `near_softening_factor = 0` and closest admissible pairs at
$R/\varepsilon$ = 2–8, `LaplaceKernel` at $P = 2$ misses the softened potential
by $4.7\times10^{-3}$ to $6.9\times10^{-2}$.

**Do:**
1. **Measure first.** With `FarField = LaplaceKernel`, evaluate a softened
   configuration (`softening = 2.5e-2`, domain extent $O(1)$) against a
   brute-force sum of the *same softened* kernel, reporting max relative error
   on **both** the potential and all `NComps × 3` gradient components, as a
   function of `near_softening_factor` over at least {4, 8, 16, 32} and of
   `P_ORDER`. The point is to show the gradient error plateauing with `P_ORDER`
   — that plateau is the softening bias, and its independence from `P_ORDER` is
   what distinguishes it from truncation error. Read the scan against F7: unless
   C11 has landed, the finite-difference L2P contributes a *second*, much lower
   plateau ($\sim\!10^{-10}$), so "a plateau" is not by itself the softening
   bias. Record the P2P pair count at each factor beside the error.
2. Run the same configuration once through `CartesianTaylorBasis` at
   `near_softening_factor = 0`, at $p = 2$ and $p = 3$, as the comparison: the
   error there falls with $p$ and shows no bias plateau.
3. Record both scans in the progress log. In `README.md` next to
   `near_softening_factor`, correct the quoted bound to state that it is the
   **potential's** and that the gradient's is three times larger, state the
   measured achievable gradient fidelity at each factor, and state that the
   far field that carries the softening is `FarField = CartesianTaylorBasis`
   at `near_softening_factor = 0`, with its measured accuracy and the cost F6
   records (double only; operator cache emptied by a rebalance that moves the
   root box).

**Exit criterion:** the new test passes under
`ctest --output-on-failure -R '^Canopy_Test_<Stem>_MPI_SERIAL_np_[1-6]$'` and
asserts a **stated, measured** relative-error bound on all gradient components
for `LaplaceKernel` in a softened configuration; and it fails, for the
softening-bias reason specifically, when `near_softening_factor` is set to 1 —
demonstrating the test is sensitive to the effect it exists to bound, rather
than passing on slack. `README.md` states the achievable far-field gradient
fidelity for both bases.

---

### C2 — Return results in the caller's particle order, on the caller's ranks — **NOT STARTED**

**Depends on:** none.

**Fill in:** `src/Canopy_Solver.hpp` (public API and `_full_setup`,
`_finish_topology_change`, `_finish_topology_stable`),
`src/Canopy_TreePartitioner.hpp` (retain the forward map already computed by
`migrate_particles` and `sort_particles_by_leaf`), `README.md`.

**Reference:** `src/Canopy_TreePartitioner.hpp:825-858` (migration semantics and
the explicit "order after migration is unspecified"); `:945-1050` (the pack/unpack
that already knows every particle's destination); `:480-496` (the only existing
global IDs, which number cells for ParMETIS, not particles);
`src/Canopy_Solver.hpp:598-604` (`sort_particles_by_leaf`, the second reordering).

**Do:**
1. Decide and record which of two shapes to expose: **(i)** an inverse map —
   Canopy retains, per local particle, the origin rank and origin index, and
   exposes a method that scatters `potential()`/`gradient()` back to the
   caller's pre-`setup` layout; or **(ii)** a caller-identity passthrough — a
   documented convention that the caller adds an identity member to its AoSoA
   and Canopy guarantees it travels intact, plus a helper that builds the
   reverse exchange from it. (i) is the smaller API and the larger internal
   change; (ii) is the reverse. Recommendation: **(i)**, because the destination
   information already exists inside `migrate_particles` at the moment it is
   needed and is thrown away, whereas (ii) makes every consumer reimplement the
   same exchange.
2. Implement it so it composes with *every* path that reorders: `setup`,
   `rebuild`, `rebalance`, `migrate`, `auto_maintain`, and the
   `sort_particles_by_leaf` permutation inside each. A map that is correct after
   `setup` and stale after `auto_maintain` is worse than none.
3. State on the declaration whether the returned ordering is the caller's
   ordering *at the most recent setup* or *at construction*, and which ranks the
   values land on.

**Exit criterion:** a `regression` test at ranks 1–6 in which each rank creates
particles with a known per-rank identity, runs `setup`, then `auto_maintain`
enough times to force at least one `Rebalance` (assert the returned
`MaintenanceAction`), and recovers `gradient()` values matching a brute-force
reference **indexed by the caller's original local index on the caller's
original rank**; and the same test fails with a clear error, not a wrong answer,
if the scatter-back is requested before any `setup`.

---

### C3 — A cheap per-evaluation refresh for multi-stage time integrators — **NOT STARTED**

**Depends on:** none. (Interacts with C2: if C2 lands first, the refresh must
keep C2's map valid.)

**Fill in:** `src/Canopy_Solver.hpp` (new public method plus the internals it
needs), `src/Canopy_TreeBuilder.hpp` (a keys-only recompute against the existing
cell list), `src/Canopy_P2P.hpp` / `src/Canopy_UpwardSweep.hpp` (refresh cached
per-leaf offsets without a full `setup`), `README.md`.

**Reference:** `src/Canopy_Solver.hpp:320-384` (`migrate`, and what it actually
costs); `:700-726` (`_finish_topology_stable`, the second full build);
`src/Canopy_TreeBuilder.hpp:812-935` (per-level `MPI_Allreduce` and host-side
cell-list assembly); `src/Canopy_CommunicationPlan.hpp:555-671` (the serial
host-side dual-tree traversal run on every rank per plan rebuild);
`src/Canopy_TreeBuilder.hpp:404-410` (`apply_particle_permutation`, precedent for
updating keys without a rebuild); `README.md:341-357` (the
$O(\text{comm\_size})$ `MPI_Alltoall` per `Rebalance`).

**Do:**
1. Add a maintenance path cheaper than `migrate()` for the case "positions moved;
   the cell list is unchanged and every particle is still in a cell owned by
   this rank". It must recompute particle→leaf keys against the *existing* cell
   list and refresh the cached leaf offsets, with **no** tree rebuild, **no**
   repartition, **no** communication-plan rebuild, and — in the common case — no
   particle exchange at all.
2. It must verify its own precondition rather than assume it. If any particle
   left its owning rank's cells, fail over to the existing path (and say which,
   via the returned `MaintenanceAction`) rather than returning a wrong field.
3. Do not change `migrate`/`rebalance`/`rebuild` semantics; this is an addition.
   `auto_maintain` gains this as its new cheapest branch, ahead of `Migrate`.

**Exit criterion:** a `regression` test at ranks 1–6 that performs three
successive small displacements and evaluations per step (the three-stage
integrator pattern) and asserts (a) the new path is selected for each — via the
returned `MaintenanceAction` — and (b) the resulting gradient matches, to the
tolerance C5 establishes, the result of a full `rebuild()` + `solve()` at the same
positions; plus a case where a particle is displaced across a cell boundary and
the assertion is that the path *declines* and reports the fallback it took.

---

### C4 — State and gate the reproducibility guarantee per backend — **NOT STARTED**

**Depends on:** none.

**Fill in:** a new test in `tests/` plus its registration in
`tests/CMakeLists.txt` under `REGRESSION_MPI_TESTS`; `README.md` (the
partitioner's description and the Known Issue "HIP solves are not
bit-reproducible run to run; the cell partition is", `:529-547`). No `src/`
change.

**Reference:** **F3(c)**; `src/Canopy_TreePartitioner.hpp:582-603` (the
fixed-seed distributed ParMETIS solve) and `:614-625` (the `MPI_Allgatherv`
that shares it); `README.md:529-547` (two `MultiSolve` passes give identical
ownership maps at SERIAL np 2–6 and HIP np 2–4, identical SERIAL output at every
np, and HIP output differing by relative 6e-12 to 2e-4);
`tests/tstLaplaceSolve.hpp:1-65` (the bitwise-comparison pattern, by bit
pattern rather than by tolerance).

**Do:**
1. Add a `regression` test that runs the same configuration twice in one
   process at a fixed rank count on `Kokkos::Serial`, with at least one
   `rebalance()` between solves so the ParMETIS repartition is exercised, and
   asserts the two `gradient()` results are **bitwise** identical — compared by
   bit pattern, since NaN ≠ NaN. Skip it on every non-Serial backend, stating why
   in the skip message.
2. In `README.md`, next to the partitioner's description, state the guarantee:
   bitwise-identical results run to run at a fixed rank count on SERIAL; the
   same partition but not the same bits on HIP; and no bitwise agreement across
   *rank counts* on either, since reduction order changes with the
   decomposition.

**Exit criterion:**
`ctest --output-on-failure -R '^Canopy_Test_<Stem>_MPI_SERIAL_np_[1-6]$'` passes
with the bitwise assertion live at every rank count; the same test fails when one
run is perturbed by a single ulp in one particle's position — showing the
comparison is bitwise rather than tolerant; and `README.md` states the per-backend
guarantee.

---

### C5 — Accuracy on a two-dimensional source distribution, and a validated parameter set — **NOT STARTED**

**Depends on:** none. Should be read together with C1 — C1 varies the softening
at fixed distribution; C5 varies the distribution. The primary configuration is
`FarField = CartesianTaylorBasis` at `near_softening_factor = 0`, the far field
that carries the softening (F6); `LaplaceKernel` with its floor is the
comparison.

**Fill in:** a new test in `tests/` plus its `tests/CMakeLists.txt` registration;
`README.md` (a validated-parameters table). Step 5 additionally touches
`src/Canopy_UpwardSweep.hpp` and `src/Canopy_CommunicationPlan.hpp`.

**Reference:** `tests/tstSingleSolve.hpp:79-87`, `252-376` (the brute-force comparison
harness to reuse, including its rank-0 gather), `:365-375` (how the tolerance is
asserted), `:389-431` (the existing parameter choices);
`tests/tstCartesianTaylorSolve.hpp` (the softened direct-sum harness,
parameterized on order and basis, `:408-420`; its configuration and pinned
figures, `:135-155`, `:326-368`); `tests/tstMultiSolve.hpp:88` (`P_ORDER = 8`),
`:432`, `:1301` and `:1561` (`softening = 0`), `:1041-1229` (the per-test
trajectory tolerances currently met);
`src/Canopy_TreeBuilder.hpp:251-256` (the depth-19 ceiling this task must check
against a deeper, surface-driven tree); `README.md:321-332` (the
bounding-box outlier limitation, which a surface with one stray source hits);
`src/Canopy_CommunicationPlan.hpp:345` (the geometric $\sqrt3(h_A+h_B)$
source-extent bound) and `treecode.py:31` (the exact $\max_j|y_j - c|$
alternative), per **F4**.

**Do:**
1. Add a source distribution on a **two-dimensional manifold embedded in three
   dimensions** — a sphere is sufficient and is trivially generated — with
   sources on the surface only, `NComps = 3`, `compute_gradient = true`, softening
   set to a physically-motivated non-zero value, compared against a brute-force
   sum of the softened kernel. Run it on `CartesianTaylorBasis` at
   `near_softening_factor = 0`, and on `LaplaceKernel` at its default floor.
2. Sweep `ncrit`, `mac_theta`, `max_depth` and `P_ORDER` (the Taylor order $p$
   on `CartesianTaylorBasis`) and record the achieved
   max relative gradient error for each combination in the progress log. Include
   at least one case where the required depth for the target `ncrit` approaches
   the ceiling, and report the depth actually reached.
3. Add a self-approaching case: two surface patches brought to within a few
   times the softening length. On `LaplaceKernel` the near-softening floor
   engages; on `CartesianTaylorBasis` it is off and the softening dominates the
   cell width. Report both the error and the P2P pair count on each basis, since
   the cost is the thing that decides viability here.
4. Publish the resulting validated parameter set in `README.md`, with the full
   qualification list the conventions table requires.
5. **Measure the exact-node-radius alternative** (F4). Compute
   $\max_j |y_j - c|$ per cell in the upward sweep, use it in `mac_satisfied`
   in place of $\sqrt3\,h$, and report — on the same surface cases as steps 1–3
   — the change in achieved gradient error *and* in P2P pair count. It is
   kernel-independent and expected to help on a sheet, but it interacts with the
   near-softening floor, so it must be measured rather than assumed; if it wins,
   it becomes a `FmmConfig` mode enum per the conventions table, not a silent
   change of predicate.

**Exit criterion:** a `regression` test passes at ranks 1–6 asserting a stated
max relative gradient error for a surface distribution with non-zero softening
on `CartesianTaylorBasis`; `README.md` carries the validated `(FarField, ncrit,
mac_theta, max_depth, P_ORDER, softening, near_softening_factor)` set and the
error it achieves; the test
fails when `mac_theta` is loosened by 2× — showing it is measuring the
approximation rather than passing on a slack budget; and the progress log carries
the exact-radius-vs-geometric-radius comparison from step 5, with a recorded
decision either way.

---

### C6 — Return the three-component gradient path to the regression gate — **NOT STARTED**

**Depends on:** none.

**Fill in:** whatever the np=4 investigation implicates — most likely
`src/Canopy_TreePartitioner.hpp` or `src/Canopy_CommunicationPlan.hpp`; plus the
teardown path exercised by `tests/tstSingleSolve.hpp`; plus
`tests/CMakeLists.txt` (label change) and `README.md` (removing the Known Issue).

**Reference:** `README.md:614-642` (both symptoms: a $2\times$-over-budget
accuracy failure at exactly 4 ranks, and a state leak that deadlocks a later
test in the same `ctest` process); `tests/tstSingleSolve.hpp:365-375` (the
assertion and its $10^{-3}$ budget); `:413-432` (the two failing cases).

**Do:**
1. Fix the np=4 accuracy failure. The rank-count-specific signature points at a
   partition or decomposition edge case rather than at the expansion; the first
   diagnostic worth running is whether the np=4 leaf assignment produces an
   ownership or replication pattern absent at 3 and 5 ranks. Determine whether
   the correct outcome is a bug fix or a re-justified budget, and record which
   with the evidence — do **not** simply widen the tolerance.
2. Fix the teardown so a failed solve cannot poison a later test in the same
   process.
3. Relabel the suite `regression` and remove the Known Issue entry.

**Exit criterion:** `ctest -L regression` passes at ranks 1–6 with the
three-component potential-and-gradient cases included, in the same `ctest`
invocation as the rest of the suite (no deadlock); and `README.md` no longer
carries the `SingleSolve` Known Issue.

---

### C7 — Expansion order and component count selectable at runtime — **NOT STARTED**

**Depends on:** none.

**Fill in:** `src/Canopy_Solver.hpp` (`createSolver`, or a new dispatch),
`README.md`.

**Reference:** `src/Canopy_Solver.hpp:165-174` (`P_ORDER`, `NComps` and the
`FarField` basis are template parameters); `:853-862` (`createSolver`, the
existing factory, which inherits the same template parameters).
`CartesianTaylorBasis` is `double` only (`src/Canopy_CartesianTaylorBasis.hpp:360`)
and reads `P_ORDER` as the Taylor order $p$, so the supported set of orders is
per basis.

**Do:** provide a factory that accepts an expansion order — and optionally a
component count — as **runtime** values and dispatches to a documented,
explicitly enumerated set of instantiations for a given `FarField`, throwing for
a value outside that set. `FarField` stays a compile-time choice: it is a type,
and selecting it changes what the result means, not just how accurate it is. Do not template the entire consumer-visible API on a value the consumer
holds at runtime, and do not silently round an unsupported order to a supported
one. Document the supported set and the compile-time cost of extending it.

**Exit criterion:** a `unit` test constructs a solver for each supported order
through the runtime factory and gets results identical to the directly
instantiated template, and gets a thrown exception naming the supported set for
an unsupported order.

---

### C8 — Gradient-only solve — **NOT STARTED**

**Depends on:** none.

**Fill in:** `src/Canopy_Solver.hpp` (`solve`), `src/Canopy_P2P.hpp`,
`src/Canopy_DownwardSweep.hpp`, `README.md`.

**Reference:** `src/Canopy_Solver.hpp:277-318` (`solve` always allocates and
zeroes the potential, and `compute_gradient` is the only selector);
`src/Canopy_P2P.hpp:891` (`phi[c]` is accumulated unconditionally inside the
innermost pair loop).

**Do:** replace the `bool compute_gradient` parameter with an enum selecting
`Potential`, `Gradient`, or `Both`, and skip the potential's allocation,
zeroing, and per-pair accumulation when it is not requested. Enumerate and
update all callers of `solve()`, `DownwardSweep::execute` and `P2P::execute` —
including every test in `tests/` and every example in `examples/` — rather than
adding an overload alongside the bool.

Note the interaction with **F7**: `LaplaceKernel`'s far-field gradient is
computed *from* potential evaluations, so on that basis a `Gradient`-only
far-field path cannot skip the potential internally until C11 lands.
`CartesianTaylorBasis`'s gradient is analytic and has no such dependency.
Skipping the potential's *output* allocation, zeroing and P2P accumulation is
valid on both and is what this task asks for.

**Exit criterion:** an existing gradient test passes unchanged through the new
enum, a `Gradient`-only solve leaves `potential()` zero-extent, and the full
`unit` + `regression` suites build and pass at ranks 1–6 with no remaining
`bool` overload in the tree.

---

### C9 — `FmmConfig` cannot be default-constructed safely — **NOT STARTED**

**Depends on:** none.

**Fill in:** `src/Canopy_Solver.hpp` (`FmmConfig`), `README.md`.

**Reference:** `src/Canopy_Solver.hpp:53-57` — every other member of
`FmmConfig` has a default initializer; `ncrit` and `max_depth` do not, so
`FmmConfig cfg;` followed by setting only some fields reads uninitialized
memory and builds an arbitrary tree. The README's parameter table lists their
defaults as "—" (`README.md:49-50`), which documents the hazard rather than
removing it.

**Do:** give both members either a defensible default initializer or a value
that is unambiguously invalid and checked in the `Solver` constructor with a
throw naming the unset field. Prefer the latter, since no default value for
`max_depth` is correct independently of the domain.

**Exit criterion:** a `unit` test constructs `FmmConfig` without setting `ncrit`
or `max_depth`, passes it to the `Solver` constructor, and gets a thrown
exception naming the unset field — rather than a built tree.

---

### C10 — A rank that owns zero sources — **NOT STARTED**

**Depends on:** none.

**Fill in:** a case added to an existing test in `tests/`; whatever `src/` path it
implicates.

**Reference:** every test in `tests/` gives every rank the same non-zero
`num_particles_per_rank` (e.g. `tests/tstSingleSolve.hpp:389-431`), so the
zero-particle rank is uncovered. The paths it must survive are the per-level
`MPI_Allreduce` over candidate counts (`src/Canopy_TreeBuilder.hpp:896-897`),
the bounding-box reduction (`:491-492`), the distributed ParMETIS solve on a
communicator split to exclude ranks that supply no vertices, and the
`MPI_Allgatherv` that shares its result
(`src/Canopy_TreePartitioner.hpp:559-625`; vertices are supplied by a block rule,
`:453-478`), and the P2P and ghost-gather loops
over a zero-length local set (`src/Canopy_P2P.hpp:834-838`).

**Do:** add a case where at least one rank — including, in one variant, rank 0
itself, the root of any rank-0 gather or print — starts with zero local
particles, and one where a rank is left with zero after migration. Assert
completion and correctness, not merely absence of a hang. If the bounding-box
reduction over an empty local set is what breaks, fix it there rather than
special-casing the caller.

**Exit criterion:** a `regression` test passes at ranks 2–6 with one rank
holding zero particles at `setup` (and, in a second variant, rank 0 holding
zero), producing gradients on the non-empty ranks that match a brute-force
reference; the test times out or fails, rather than silently passing, if a
collective is skipped on the empty rank.

---

### C11 — Analytic far-field gradient in place of the finite difference — **NOT STARTED**

**Depends on:** none. Worth doing **before** C1 step 1's scan is interpreted, so
that scan has two error sources to separate rather than three.

**Fill in:** `src/Canopy_LaplaceKernel.hpp` (`l2p_evaluate`, `:1339`, and its
header comment `:1313-1337`); `tests/tstLaplaceKernel.hpp`;
`tests/data/laplace_solve_P6.txt`; the cross-reference comment at
`src/Canopy_CartesianTaylorBasis.hpp:1463-1466`, which cites the
finite difference by line; `README.md` (Known Issues, and any stated accuracy
that changes).

**Reference:** `src/Canopy_LaplaceKernel.hpp:1388-1418` — the six-point central
difference at $h = 10^{-5} w_{\rm self}$, its in-code
`TODO: replace with analytical derivatives`, and the comment recording that a
previously fixed step size caused a premature full-rollup NaN; `:1333-1337` (the
$\bar L = L\,w^n$ normalization the analytic form must respect); **F7**.
`CartesianTaylorBasis::l2p_evaluate`
(`src/Canopy_CartesianTaylorBasis.hpp:1453-1490`) already returns an analytic
gradient and is the precedent for how the declaration states it.

`tests/tstLaplaceKernel.hpp` does not compile: it calls `p2m_contribution`,
`m2m_translate`, `m2l_translate`, `l2l_translate` and `l2p_evaluate` with
argument lists that no longer match their declarations — e.g. `testL2PGradient`
(`:795`) omits `w_self` — 35 errors, recorded in `README.md` Known Issues under
"Two `unit` test targets do not compile".

`tests/tstLaplaceSolve.hpp` drives a frozen configuration for 12 timesteps whose
gradient feeds the velocity update, then gates the final state bit-for-bit
(`bitForBitArtifacts`, np 1–2, Kokkos::Serial only) and against the committed
np=1 field (`crossRankAgreement`, np 2–6), both from
`tests/data/laplace_solve_P6.txt`. Nothing in the default test path rewrites that
file (`tstLaplaceSolve.hpp:56-65`), so any change to the gradient's bits fails
both gates by construction. Regeneration is
`scripts/tuolumne/run_laplace_solve_regenerate.flux`.

**Do:**
1. Update `tests/tstLaplaceKernel.hpp` to the current operator signatures —
   signature changes only, no change to what any test asserts — so
   `Canopy_Test_LaplaceKernel_SERIAL` builds and passes against the finite
   difference. Remove the `LaplaceKernel` half of the README Known Issue; the
   `P2P` half stays.
2. Differentiate the local expansion analytically in the solid-harmonic basis
   and evaluate the gradient directly, citing the identity used on the routine
   per the conventions table. Keep the width normalization consistent with
   `l2p_evaluate`'s existing $\bar L$ convention. Remove the step-size heuristic
   and the TODO. State on the declaration the sign convention of the returned
   gradient, since F2 records that as the most misread thing in the API. Update
   the `CartesianTaylorBasis` cross-reference comment to match.
3. Add a `unit` test in `tests/tstLaplaceKernel.hpp` comparing the analytic
   gradient with the finite difference, and tighten `testL2PGradient`'s
   tolerance (`1e-7`, set for the finite difference) to what the analytic form
   achieves.
4. Only after step 3 passes, regenerate `tests/data/laplace_solve_P6.txt` with
   `run_laplace_solve_regenerate.flux` so the committed reference data matches
   the new code. `LS_CROSS_RANK_TOL` and `LS_DIRECT_SUM_TOL` stay unchanged.

**Out of scope:** FP32. `SolveFusedM2L.FP32_smokeTest` stays commented out and
disabled (`tests/tstMultiSolve.hpp:1643-1700`); FP32 is not a production
configuration.

**Exit criterion:** all of the following pass, run with anchored regexes:
- `ctest --output-on-failure -R '^Canopy_Test_LaplaceKernel_SERIAL$'`, including
  a test showing the analytic gradient agreeing with the finite difference to the
  finite difference's own accuracy ($\sim\!10^{-8}$ relative or better), and
  `testL2PGradient` at a tolerance tighter than `1e-7`;
- `ctest --output-on-failure -R '^Canopy_Test_(LaplaceSolve|MultiSolve|DownwardSweep)_MPI_SERIAL_np_[1-6]$'`
  and `-R '^Canopy_Test_(LaplaceSolve|MultiSolve|DownwardSweep)_MPI_HIP_np_[1-4]$'`
  (flux cannot place HIP at np 5–6, `systems/tuolumne/claude.md` §5), with
  `LaplaceSolve` against the regenerated reference data and its tolerances
  unchanged;
- `ctest --output-on-failure -R '^Canopy_Test_SingleSolve_MPI_SERIAL_np_[12356]$'`
  — np=4 is C6's known failure and is not part of this gate.

No existing tolerance in these tests is loosened. `LaplaceSolve` against the
*old* reference data fails after the change — confirming the regeneration was
required rather than incidental. The progress log records the measured gradient
error floor before and after, so C1's scan can be read against it.

## Known risks

**R1 — The softening bias is mistaken for expansion error, and "fixed" by
raising `P_ORDER`.** Both present as a relative-error figure above budget.
The distinguishing measurement is the *scan*: truncation error falls as
`P_ORDER` rises; the softening bias plateaus. C1 step 1 exists to produce that
scan, and no tolerance anywhere should be set before it has been read. Note the
scan has **three** components, not two: per F7 the finite-difference L2P
contributes its own, much lower ($\sim\!10^{-10}$) plateau, so "the error
stopped falling" is not by itself evidence of softening bias. C11 removes that
confound.

**R2 — A tolerance gets tuned to hide C6's np=4 defect.** The np=4 failure is
$\approx 2\times$ over a $10^{-3}$ budget — close enough that widening
the budget looks defensible. It is not, until the rank-count-specific mechanism
is understood: an error that appears at exactly one rank count is a
decomposition bug signature, not a budget signature. C6 requires the evidence
either way.

**R3 — HIP run-to-run noise masquerades as a regression.** The partition is
deterministic, but a HIP solve differs between runs by relative 6e-12 to 2e-4
(F3(c)) because device reductions accumulate in a run-dependent order. Once a
consumer compares a HIP run against a reference, that spread is
indistinguishable from a real change, and the likely outcome is a tolerance
loosened until the noise fits — which then hides real regressions of the same
magnitude. Set bitwise gates on SERIAL only (C4), and measure the HIP spread on
the configuration in question before setting any HIP tolerance.

**R4 — C3's cheap path is implemented as an optimistic one.** The dangerous
version of C3 assumes its precondition and returns a wrong field when it is
violated — exactly the failure mode that already exists for `solve()` after
motion (F3(a)). Its exit criterion therefore requires a case where the
precondition is violated *and the path declines*, not merely a case where it
succeeds.

**R5 — C5 measures a distribution that is not the hard case.** A uniform
sphere is a two-dimensional distribution but it is not self-approaching, and the
self-approach is where the softening, the tree depth, the near-field cost and
the bounding box all degrade at once. A C5 that reports a clean number on a
smooth sphere and stops has answered the easy half; step 3 is the half that
decides viability.

**R6 — A gap gets closed on the consumer's side instead.** Several findings here
(C1's bias, C2's reordering, C3's cost) can be worked around by the consumer at
the price of a wrong answer, a duplicated exchange, or a tripled cost per
timestep. Those are not closures. Each task above is either done in this library
or accepted by name, with the accepted consequence written down.

**R7 — `CartesianTaylorBasis` is chosen on accuracy alone and its maintenance
cost surfaces later.** Every rebalance that moves the root box empties that
basis's M2L operator cache (F6(c)), and per F3(c) rebalance is this consumer's
common path, three times per step. An accuracy sweep (C5) does not show that
cost. `DownwardSweep::m2l_op_keys_built_count()`
(`src/Canopy_DownwardSweep.hpp:469-480`) counts cache misses; record it beside
every accuracy figure C3 and C5 take on this basis, so the per-stage rebuild is
measured rather than discovered.

**R8 — The $10^{-6}$ target survives unexamined.** `LaplaceKernel` with a
floor is bounded at $10^{-2}$–$10^{-3}$ on the gradient (F1), and
`CartesianTaylorBasis` measures $7.1\times10^{-4}$ at $p = 3$ (F4); F6(d) puts
$10^{-6}$ at $p \approx 11$–$24$ on that basis, which pays $O(p^3)$ coefficients
per cell. If $10^{-6}$ is a hard consumer requirement rather than an
aspiration, that cost should be established before anything here is built. Establish what the consumer
actually needs first; it is the cheapest question on this list to answer.
