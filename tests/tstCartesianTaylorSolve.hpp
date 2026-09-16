/****************************************************************************
 * Copyright (c) 2025 by the Canopy authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Canopy library. Canopy is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

// ===========================================================================
// tstCartesianTaylorSolve — the solve-level accuracy gate for
// CartesianTaylorBasis. T4 of tasks/cartesian-taylor-basis.md.
//
// One multi-step FMM solve is driven through Solver at np 1-6 and its
// last-step field is compared against a brute-force O(N^2) SOFTENED direct
// sum, at two admissibilities:
//
//   matchesDirectSumThetaRef     mac_theta = 0.3, the reference treecode's
//                                admissibility. This is the matched-
//                                admissibility comparison, and the one the
//                                1e-3 accuracy claim is made against.
//   matchesDirectSumThetaCanopy  mac_theta = 0.5, Canopy's own frozen gate.
//                                Its deviation is measured and pinned; it is
//                                NOT expected to reach 1e-3 (see below).
//
// WHY THE FAR FIELD HERE IS SOFTENED AND THE NEAR FIELD IS NOT SPECIAL.
// LaplaceKernel's multipole far field is the UNSOFTENED 1/r kernel, so a
// solve with a positive softening has to keep every near pair out of the M2L
// (FmmConfig::near_softening_factor, default 4). CartesianTaylorBasis expands
// the SOFTENED kernel (r^2 + b)^{-1/2} directly, so that floor is not needed
// and is switched off here: near_softening_factor = 0. That is the whole
// point of the basis, and it is why the reference this file compares against
// is a softened sum rather than an unsoftened one:
//
//   phi_i        =  sum_{j != i} q_j (r^2 + b)^{-1/2}
//   grad phi_i   = -sum_{j != i} q_j d (r^2 + b)^{-3/2},  d = x_i - x_j
//
// with b = softening^2 — the same convention P2P uses (Canopy_P2P.hpp:803,
// :885) and the same sign convention tstLaplaceSolve.hpp:1535-1538 uses for
// the unsoftened case.
//
// THE SOFTENING IS EXPLICIT AND POSITIVE. FmmConfig::softening defaults to
// -1.0, which selects distribution-based auto-softening whose effective eps
// moves with the particle distribution; the tolerances below would then be
// unpinnable. CTS_SOFTENING is the downstream solver's 0.025.
//
// WHY THE TWO ARMS DO NOT LAND IN THE SAME PLACE. Canopy's MAC is
// R^2 theta^2 > 3 (w_a + w_b)^2, so equal-sized cells are admissible at
// R > 11.55 W for theta = 0.3 and at R > 6.93 W for theta = 0.5. Taylor
// truncation goes as (c W/R)^{p+1}, so the theta = 0.5 arm admits pairs the
// theta = 0.3 arm refuses and is necessarily the less accurate of the two.
// tasks/cartesian-taylor-basis-progress-log.md §T3 measured a single pair at
// R/W = 8 — just inside the theta = 0.5 MAC — at 2.34e-3 relative, ABOVE the
// 1e-3 bar. A whole-solve deviation is not a single-pair bound (the near
// field is exact and the global-scale normalization is over the whole field),
// so both arms are measured rather than extrapolated, but a theta = 0.5
// figure near 1e-3 is the expected result and not a defect.
// ===========================================================================

#include "Canopy_CartesianTaylorBasis.hpp"
#include "Canopy_Helpers.hpp"
#include "Canopy_Solver.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace Test
{
//---------------------------------------------------------------------------//

namespace CartesianTaylorSolveTest
{

// ---------------------------------------------------------------------------
// The configuration. The pinned theta = 0.5 constant below was measured at
// exactly this one, and its provenance comment names it again.
//
// CTS_NUM_PARTICLES is the GLOBAL total, sliced contiguously per rank, so the
// same physics problem is solved at every rank count. 8640 is divisible by
// every rank count from 1 to 6.
//
// CTS_NUM_PARTICLES and CTS_NCRIT ARE CHOSEN TOGETHER, TO MAKE THE theta =
// 0.3 ARM NON-VACUOUS. The tree refines while a cell holds more than ncrit
// particles, so a uniform set of N reaches depth d when N / 8^d ~ ncrit; at
// 8640 and ncrit = 8 that is depth 4, a 16^3 leaf grid. The near field at
// theta = 0.3 reaches 11.55 W = 5.77 cell widths, which is a neighbourhood of
// about 800 cells — under 20% of a 16^3 grid, so most pairs are far field. At
// depth 3 (an 8^3 grid, 512 cells) that same neighbourhood covers the WHOLE
// grid and only a handful of corner pairs stay admissible: the far field is
// then live by the m2l_n_unique_ops() > 0 guard while carrying almost none of
// the field, and the accuracy check passes without measuring anything. That
// is the failure mode the particle count exists to prevent, and it is why
// lowering CTS_NUM_PARTICLES to make the test cheaper is not a free change.
//
// CTS_DT and CTS_NUM_STEPS. The solve is multi-step so the operator cache is
// exercised across a topology change (risk R5): this basis's operators depend
// on unit_w, which is not part of the cache's key premise, and the only thing
// that saves it is set_root_half_width clearing the cache for a
// key_needs_level basis. The step schedule is
//
//   solve -> migrate -> solve -> REBALANCE -> solve -> migrate -> solve
//
// i.e. 4 solves with 3 intervening maintenance steps, the middle one a
// rebalance(). rebalance() repartitions, rebuilds the comm plan and
// invalidates the interaction list unconditionally, so the last two solves
// run against a tree that was torn down and rebuilt after the first operator
// table was constructed. tstLaplaceSolve.hpp:860-871 uses migrate() and never
// rebalance() for the opposite reason — it has a np 1-2 BITWISE gate and
// needs a single partition. This file has no bitwise gate, so it wants the
// repartition rather than avoiding it.
//
// CTS_DT is small: the point of the motion is to move the root bounding box
// (which empties the operator cache) and to change the tree topology, not to
// integrate anything physical. Charges are ONE-SIGNED, as
// tstLaplaceSolve.hpp:775-785 explains at length: a two-signed set at this
// softening still collapses, and a degenerate tree makes every check here
// pass vacuously.
// ---------------------------------------------------------------------------
static constexpr int CTS_P = 2; // the reference treecode's order
static constexpr int CTS_NCOMPS = 3;
static constexpr int CTS_NUM_PARTICLES = 8640; // global total
static constexpr int CTS_NCRIT = 8;
static constexpr int CTS_MAX_DEPTH = 6;
static constexpr int CTS_REPLICATION_DEPTH = 2;
static constexpr int CTS_NUM_STEPS = 4;
static constexpr double CTS_DT = 1.0e-5;

// ---------------------------------------------------------------------------
// THE SPATIAL EXTENT OF THE PARTICLE SET, AND WHY IT IS NOT [0.05, 0.95].
//
// This is the one configuration choice that decides whether this test can tell
// CartesianTaylorBasis apart from LaplaceKernel at all, and it was got WRONG
// first: the original [0.05, 0.95] (copied from tstLaplaceSolve.hpp) was
// measured on job f3YfdTuJH59H and the LaplaceKernel failure-direction arm did
// NOT fire — 1.26e-03 potential and 7.51e-03 gradient against this basis's
// 3.43e-04 and 8.10e-03, i.e. LaplaceKernel was 3.7x worse on the potential
// and slightly BETTER on the gradient. T4's failure direction requires orders
// of magnitude. The numbers and the reasoning are in the T4 section of
// tasks/cartesian-taylor-basis-progress-log.md.
//
// The cause is dimensionless and has nothing to do with the particle count.
// Canopy's MAC is R^2 theta^2 > 3 (w_a + w_b)^2, so the CLOSEST admissible
// pair sits at R = 11.55 W (theta = 0.3), and the unsoftened far field's
// relative error from ignoring b is b / (2 R^2). What decides whether that is
// large is therefore eps / W — the softening measured in cell half-widths —
// and nothing else. On [0.05, 0.95] the root half-width is 0.5354, a depth-4
// leaf is W = 0.03346, and R = 11.55 W = 0.3865 = 15.5 eps, where
// b / (2 R^2) = 2.1e-03: the same order as the p = 2 truncation error, so the
// two bases are indistinguishable. The softened far field is only worth having
// where the admissible separation is a FEW eps, which is exactly the regime
// the downstream solver runs in and the reason this basis exists.
//
// CTS_POS_LO/HI shrink the cloud until that holds while keeping eps at the
// downstream solver's 0.025, which the task statement pins. A span of 0.12
// gives a root half-width of 0.072 (the 0.1 per-face padding included), a
// depth-4 leaf of W = 0.0045, and closest admissible separations of
// R = 2.08 eps at theta = 0.3 and R = 1.25 eps at theta = 0.5, where
// b / (2 R^2) is 11% and 32%. Nothing else about the problem changes: the tree,
// the cell counts, the realized key counts and the fallback counts are all
// functions of the DIMENSIONLESS geometry and are identical either way.
//
// CTS_DT is scaled with the domain so the dimensionless trajectory is the one
// the original measured. Self-gravity gives g ~ M / L^2 and hence a dynamical
// time ~ sqrt(L^3 / M), so dt scales as L^(3/2): (0.072 / 0.5354)^(3/2) =
// 0.0494, and 2.0e-04 * 0.0494 = 9.9e-06, rounded to 1.0e-05. Leaving dt at
// 2.0e-04 would move every particle several domain widths in four steps.
// ---------------------------------------------------------------------------
static constexpr double CTS_POS_LO = 0.44;
static constexpr double CTS_POS_HI = 0.56;

// The softening LENGTH, eps. b = eps^2 is what is added to r^2. The
// downstream solver's value; stated as a length because that is what
// FmmConfig::softening is (P2P and the basis each square it themselves).
static constexpr double CTS_SOFTENING = 0.025;

// The two admissibilities. 0.3 is treecode.py's Barnes-Hut theta; 0.5 is
// Canopy's FmmConfig default and the value tstLaplaceSolve freezes.
static constexpr double CTS_THETA_REF = 0.3;
static constexpr double CTS_THETA_CANOPY = 0.5;

// ---------------------------------------------------------------------------
// THE ACCURACY BAR, which is not a measurement.
//
// 1e-3 relative is the reference treecode's DOCUMENTED accuracy at its own
// order (2) and its own admissibility (theta = 0.3) — README.md:71 and
// PHYSICS.md:137 of the downstream solver's repository, with 5.9e-4 / 1.1e-3
// / 9.4e-4 measured at N = 642 / 2562 / 10242 in PARALLELIZATION.md:24-30.
// See tasks/cartesian-taylor-basis.md, "The reference implementation, and
// what it fixes".
//
// This is the exit criterion of T4 and it is not a tolerance to be tuned. The
// static_assert below is what stops a later session from raising the pinned
// theta = 0.3 constant past it to accommodate a miss; if the measurement
// moves above the bar, the diagnostics to work are R1 and R2 in
// tasks/cartesian-taylor-basis.md, in that order.
// ---------------------------------------------------------------------------
static constexpr double CTS_REFERENCE_BAR = 1.0e-3;

// ---------------------------------------------------------------------------
// MEASURED, THEN PINNED — and one of the two is NOT a measurement.
//
// Both figures below were measured on job f3YfgoXqJe2j, at the configuration
// above and nowhere else: 8640 particles global on [0.44, 0.56]^3, ncrit = 8,
// max_depth = 6, replication_depth = 2, softening = 0.025 (b = 6.25e-4) with
// near_softening_factor = 0, P_ORDER = 2, NComps = 3, 4 solves with
// migrate / rebalance / migrate between them, dt = 1.0e-5, charges one-signed
// on [0.5, 1.5] drawn per component from seed 90210 + CTS_P, SERIAL backend.
// Each is the worst of potential and gradient, normalized by the GLOBAL field
// scale (max |phi|, max |grad phi| over the direct sum) as
// tstLaplaceSolve.hpp:1546-1552 normalizes. np 1 and np 2 agreed to 1e-16.
//
//   theta = 0.3   potential 2.6552722557554608e-04   gradient 8.9961055614e-03
//   theta = 0.5   potential 8.6730576409738364e-04   gradient 2.1590006996e-02
//
// CTS_DEV_TOL_THETA_CANOPY is 2x the worst theta = 0.5 figure, 2.159e-02.
//
// CTS_DEV_TOL_THETA_REF IS NOT A MEASUREMENT. It is CTS_REFERENCE_BAR itself,
// and the theta = 0.3 GRADIENT does not meet it: 8.996e-03 against 1.0e-03, a
// factor of 9. That assertion FAILS, deliberately and knowingly, and it is
// left failing rather than widened. The potential meets the bar with room
// (2.66e-04). The full attribution is in the T4 section of
// tasks/cartesian-taylor-basis-progress-log.md; in short, and measured rather
// than argued:
//
//   * It is not R1. A convention error costs one to three orders on BOTH
//     fields; the potential is 2.66e-04.
//   * It is not R2. T1's oracle was re-run at this configuration's actual
//     (r, b) band, |r|/sqrt(b) in [9.28, 74.2], which the original sample set
//     never covered: 76 samples, worst 5.465e-15 against a 1e-12 tolerance,
//     and the |k| = 4 finite difference unchanged at 7.321e-07.
//   * It is not R4. The fallback count is 300 per rank at theta = 0.3 and
//     exactly 0 at theta = 0.5, and both arms show the same error structure.
//
//   * It IS the target-side L2P truncation, which the reference treecode does
//     not have. The gradient of a degree-p local is a degree-(p-1) polynomial,
//     so grad phi truncates one order before phi. The fingerprint is exact:
//     the ABSOLUTE error ratio grad/pot is 218.5 against 1/W = 223.7 for a
//     depth-4 leaf (W = 0.004471) — the potential's error differentiated over
//     the CELL HALF-WIDTH, not over R. The same ratio held to 2% on the
//     earlier [0.05, 0.95] domain (29.47 against 1/W = 29.88), so it is a
//     property of the method and not of this geometry.
//
// The reference treecode evaluates its source expansion with NO target-side
// expansion, so its documented ~1e-3 VELOCITY accuracy is a source-side-only
// figure — the same order as this file's POTENTIAL, which meets it. Applying
// it to an FMM's gradient at the same p compares quantities of different
// truncation order. Raising the order fixes it and confirms the diagnosis:
// at p = 3 the theta = 0.3 gradient falls to 1.141e-03 and the potential to
// 2.178e-05, the gradient improving by 7.9x per order as truncation requires.
//
// DO NOT widen CTS_DEV_TOL_THETA_REF to make this green. Either the bar is
// restated as a claim about the potential, or the gradient arm runs at p = 3;
// both are decisions about what T4 means and belong to whoever owns the task,
// not to this file. The static_assert below is what keeps the choice visible.
// ---------------------------------------------------------------------------
static constexpr double CTS_DEV_TOL_THETA_REF = CTS_REFERENCE_BAR;
static constexpr double CTS_DEV_TOL_THETA_CANOPY = 4.32e-02;

static_assert( CTS_DEV_TOL_THETA_REF <= CTS_REFERENCE_BAR,
               "The pinned theta = 0.3 tolerance is above the reference "
               "treecode's documented 1e-3 accuracy. T4's exit criterion is "
               "that bar, not this constant; raising this past it silently "
               "changes what the test means. Work R1 and R2 in "
               "tasks/cartesian-taylor-basis.md instead." );

// AoSoA member indices. Velocity and GlobalId exist for the time loop:
// GlobalId is what pairs a particle back to the global set after migration
// has scrambled the local ordering.
enum FieldIdx
{
    Position = 0,
    Charge = 1,
    Velocity = 2,
    GlobalId = 3
};

// The solver under test.
//
// PARAMETERIZED ON THE ORDER AND THE BASIS, and not because the shipped tests
// need it: both pass CTS_P and CartesianTaylorBasis. It is parameterized so
// that a DIAGNOSTIC arm — the LaplaceKernel failure direction this task
// records, or an order sweep — is an added TEST body rather than an edit to
// the harness, and so that T5's measurement body can drive the same multi-step
// solve without copying it. The particle set does NOT depend on either
// parameter (the seed is 90210 + CTS_P, a constant), so every arm solves the
// same physics problem and their deviations are directly comparable.
template <class MemorySpace, class ExecutionSpace, int P,
          template <class, int, int> class Basis>
using solver_type =
    Canopy::Solver<MemorySpace, ExecutionSpace, double, P, CTS_NCOMPS, Basis>;

// ---------------------------------------------------------------------------
// The gathered last-step state, on rank 0 only, in GlobalId order.
// ---------------------------------------------------------------------------
struct GatheredState
{
    bool valid = false;
    std::string err;
    int n = 0;
    std::vector<double> pos; // 3 * n
    std::vector<double> chg; // NComps * n
    std::vector<double> pot; // NComps * n
    std::vector<double> grad; // 3 * NComps * n
};

// ---------------------------------------------------------------------------
// Drive the configuration above at `mac_theta` for CTS_NUM_STEPS solves and
// hand the gathered last-step state to `after`.
//
// Every rank asserts its own far field was live before anything is compared.
// ---------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace, int P,
          template <class, int, int> class Basis, class Fn>
void with_cartesian_taylor_solve( double mac_theta, Fn&& after )
{
    using Scalar = double;
    using DataTypes = Cabana::MemberTypes<Scalar[3],          // Position
                                          Scalar[CTS_NCOMPS], // Charge
                                          Scalar[3],          // Velocity
                                          int>;               // GlobalId
    using AoSoA_t = Cabana::AoSoA<DataTypes, MemorySpace>;
    using AoSoA_ht = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
    using Solver_t = solver_type<MemorySpace, ExecutionSpace, P, Basis>;

    int rank = 0, nprocs = 1;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    const int n_total = CTS_NUM_PARTICLES;

    // The global particle set, generated identically on every rank at every
    // rank count. Positions on [CTS_POS_LO, CTS_POS_HI]^3; charges one-signed on
    // [0.5, 1.5], drawn INDEPENDENTLY PER COMPONENT so that a component-
    // crosstalk bug in the M2L accumulator's (component, set) flattening
    // shows up as a wrong field rather than as three copies of a right one.
    std::vector<double> g_pos( 3 * n_total );
    std::vector<double> g_chg( CTS_NCOMPS * n_total );
    {
        std::mt19937 gen( 90210 + CTS_P );
        std::uniform_real_distribution<double> pos_dist( CTS_POS_LO, CTS_POS_HI );
        std::uniform_real_distribution<double> q_dist( 0.5, 1.5 );
        for ( int i = 0; i < n_total; i++ )
        {
            g_pos[3 * i + 0] = pos_dist( gen );
            g_pos[3 * i + 1] = pos_dist( gen );
            g_pos[3 * i + 2] = pos_dist( gen );
            for ( int c = 0; c < CTS_NCOMPS; c++ )
                g_chg[CTS_NCOMPS * i + c] = q_dist( gen );
        }
    }

    const int i_begin = rank * n_total / nprocs;
    const int i_end = ( rank + 1 ) * n_total / nprocs;
    const int n_local_initial = i_end - i_begin;

    AoSoA_ht particles_h( "particles_h", n_local_initial );
    {
        auto hp = Cabana::slice<Position>( particles_h );
        auto hq = Cabana::slice<Charge>( particles_h );
        auto hv = Cabana::slice<Velocity>( particles_h );
        auto hid = Cabana::slice<GlobalId>( particles_h );
        for ( int i = 0; i < n_local_initial; i++ )
        {
            const int g = i_begin + i;
            for ( int d = 0; d < 3; d++ )
            {
                hp( i, d ) = g_pos[3 * g + d];
                hv( i, d ) = 0.0;
            }
            for ( int c = 0; c < CTS_NCOMPS; c++ )
                hq( i, c ) = g_chg[CTS_NCOMPS * g + c];
            hid( i ) = g;
        }
    }
    AoSoA_t particles( "particles", n_local_initial );
    Cabana::deep_copy( particles, particles_h );

    Canopy::FmmConfig cfg;
    cfg.ncrit = CTS_NCRIT;
    cfg.max_depth = CTS_MAX_DEPTH;
    cfg.xmin_tol = cfg.xmax_tol = 0.1;
    cfg.ymin_tol = cfg.ymax_tol = 0.1;
    cfg.zmin_tol = cfg.zmax_tol = 0.1;
    cfg.ncrit_tol = 0.1;
    cfg.replication_depth = CTS_REPLICATION_DEPTH;
    cfg.imbalance_tolerance = 0.05;
    cfg.mac_theta = mac_theta;
    // An EXPLICIT positive softening: the default -1.0 selects auto-softening
    // and the tolerances would not be pinnable. See the header.
    cfg.softening = CTS_SOFTENING;
    // 0 disables the near-field softening floor. LaplaceKernel needs that
    // floor because its far field is unsoftened; this basis's far field is
    // the softened kernel itself, so the floor would only narrow the far
    // field for no reason and hide exactly what this test measures.
    cfg.near_softening_factor = 0.0;

    Solver_t solver( MPI_COMM_WORLD, cfg );
    solver.template setup<Position, Charge>( particles, n_local_initial );

    // -----------------------------------------------------------------------
    // The step loop: solve -> migrate -> solve -> rebalance -> solve ->
    // migrate -> solve. The update and the maintenance are omitted after the
    // last solve, as tstLaplaceSolve.hpp:872-890 omits them, because every
    // check below reads that solve's field and maintenance would overwrite
    // the locals and the operator table while the update would move the
    // particles away from the positions the field was evaluated at.
    // -----------------------------------------------------------------------
    for ( int step = 0; step < CTS_NUM_STEPS; step++ )
    {
        solver.template solve<Position, Charge>( particles,
                                                 /*compute_gradient=*/true );

        if ( step + 1 == CTS_NUM_STEPS )
            break;

        // Symplectic Euler on COMPONENT 0's gradient. The other two
        // components are carried for their own sake — they are solved and
        // checked, but they do not move anything, so the trajectory stays a
        // function of one field and is reproducible across rank counts up to
        // reassociation.
        const int n_local = solver.num_local_particles();
        auto positions = Cabana::slice<Position>( particles );
        auto velocities = Cabana::slice<Velocity>( particles );
        auto grad = solver.gradient();
        const double dt_local = CTS_DT;
        Kokkos::parallel_for(
            "CartesianTaylorSolve::integrate",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_local ),
            KOKKOS_LAMBDA( int i ) {
                for ( int d = 0; d < 3; d++ )
                {
                    velocities( i, d ) += dt_local * grad( i, 0, d );
                    positions( i, d ) += dt_local * velocities( i, d );
                }
            } );
        Kokkos::fence();

        // The middle maintenance step is the rebalance. It is what
        // repartitions, rebuilds the comm plan and invalidates the
        // interaction list, so the last two solves face an operator table
        // built from scratch after the root box has moved (R5).
        if ( step == 1 )
            solver.template rebalance<Position>( particles );
        else
            solver.template migrate<Position>( particles );
    }

    const auto& ds = solver.downward();
    const int n_unique_ops = ds.m2l_n_unique_ops();
    const long long fallback_pairs = ds.total_fallback_pair_count();

    double root_hw = 0.0;
    {
        const auto& box = solver.builder().root_box();
        for ( int d = 0; d < 3; d++ )
            root_hw = std::max( root_hw, 0.5 * ( box.max[d] - box.min[d] ) );
    }

    // Always echo, so a later failure is attributable from a ctest -V log
    // without a rebuild (T4 step 6). A non-zero fallback count is NOT a
    // failure — T3's m2l_fused_vs_fallback proves the two paths agree
    // bitwise — but it changes which arithmetic ran, so it must be visible
    // (R4). At P_ORDER = 2 the count cap (32768) binds far above any
    // realized key count, so 0 is the expected value.
    std::printf( "[ct-solve] theta=%.17g p_order=%d nprocs=%d rank=%d steps=%d "
                 "n_particles=%d ncrit=%d max_depth=%d softening=%.17g "
                 "n_cells=%zu root_half_width=%.17g n_unique_ops=%d "
                 "fallback_pairs=%lld op_cap=%d\n",
                 mac_theta, P, nprocs, rank, CTS_NUM_STEPS, CTS_NUM_PARTICLES,
                 CTS_NCRIT, CTS_MAX_DEPTH, CTS_SOFTENING,
                 solver.builder().cells().size(), root_hw, n_unique_ops,
                 fallback_pairs, ds.m2l_effective_op_cap() );
    std::fflush( stdout );

    const std::string tag = "(theta=" + std::to_string( mac_theta ) +
                            " nprocs=" + std::to_string( nprocs ) +
                            " rank=" + std::to_string( rank ) + ")";

    // The degeneracy guard, on EVERY rank and in BOTH arms. Canopy's MAC is
    // R^2 theta^2 > 3 (w_a + w_b)^2, so theta = 0.3 needs R > 11.55 W against
    // 6.93 W at theta = 0.5; a tree too shallow for the tighter MAC admits no
    // pair at all, and the accuracy comparison below would then be a test of
    // P2P. See tstLaplaceSolve.hpp:1211-1212, which asserts the same thing
    // for the same reason.
    //
    // EXPECT and not ASSERT, deliberately. A fatal assertion returns from this
    // function, and the MPI_Gather/MPI_Gatherv below are COLLECTIVE: one rank
    // leaving early would hang every other rank until the job's walltime,
    // which destroys the very log the failure has to be read out of. The
    // non-fatal form fails the test and still lets the gather complete.
    EXPECT_GT( n_unique_ops, 0 )
        << "m2l_n_unique_ops() is 0: no pair is MAC-admissible, so the far "
           "field this test exists to measure was never evaluated and the "
           "direct-sum comparison below would pass vacuously "
        << tag;

    // -----------------------------------------------------------------------
    // Gather the last-step state to rank 0 in GlobalId order. Particles pair
    // to the global set by GlobalId and by nothing else: migration scrambles
    // the local ordering and the solve has moved the particles.
    // -----------------------------------------------------------------------
    GatheredState gs;
    {
        const int n_local = solver.num_local_particles();
        auto positions = Cabana::slice<Position>( particles );
        auto charges = Cabana::slice<Charge>( particles );
        auto gids = Cabana::slice<GlobalId>( particles );
        auto h_pos = Canopy::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          positions, "h_pos" );
        auto h_chg = Canopy::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          charges, "h_chg" );
        auto h_gid = Canopy::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          gids, "h_gid" );
        auto h_pot = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          solver.potential() );
        auto h_grad = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                           solver.gradient() );

        std::vector<double> l_pos( 3 * n_local );
        std::vector<double> l_chg( CTS_NCOMPS * n_local );
        std::vector<double> l_pot( CTS_NCOMPS * n_local );
        std::vector<double> l_grad( 3 * CTS_NCOMPS * n_local );
        std::vector<int> l_gid( n_local );
        for ( int i = 0; i < n_local; i++ )
        {
            for ( int d = 0; d < 3; d++ )
                l_pos[3 * i + d] = h_pos( i, d );
            for ( int c = 0; c < CTS_NCOMPS; c++ )
            {
                l_chg[CTS_NCOMPS * i + c] = h_chg( i, c );
                l_pot[CTS_NCOMPS * i + c] = h_pot( i, c );
                for ( int d = 0; d < 3; d++ )
                    l_grad[3 * ( CTS_NCOMPS * i + c ) + d] = h_grad( i, c, d );
            }
            l_gid[i] = h_gid( i );
        }

        std::vector<int> all_n( nprocs, 0 );
        MPI_Gather( &n_local, 1, MPI_INT, all_n.data(), 1, MPI_INT, 0,
                    MPI_COMM_WORLD );
        std::vector<int> c1( nprocs, 0 ), d1( nprocs, 0 );
        std::vector<int> c3( nprocs, 0 ), d3( nprocs, 0 );
        std::vector<int> cq( nprocs, 0 ), dq( nprocs, 0 );
        std::vector<int> cg( nprocs, 0 ), dg( nprocs, 0 );
        int total = 0;
        if ( rank == 0 )
        {
            for ( int k = 0; k < nprocs; k++ )
            {
                c1[k] = all_n[k];
                c3[k] = 3 * all_n[k];
                cq[k] = CTS_NCOMPS * all_n[k];
                cg[k] = 3 * CTS_NCOMPS * all_n[k];
                total += all_n[k];
            }
            for ( int k = 1; k < nprocs; k++ )
            {
                d1[k] = d1[k - 1] + c1[k - 1];
                d3[k] = d3[k - 1] + c3[k - 1];
                dq[k] = dq[k - 1] + cq[k - 1];
                dg[k] = dg[k - 1] + cg[k - 1];
            }
        }

        std::vector<double> a_pos, a_chg, a_pot, a_grad;
        std::vector<int> a_gid;
        if ( rank == 0 )
        {
            a_pos.resize( 3 * total );
            a_chg.resize( CTS_NCOMPS * total );
            a_pot.resize( CTS_NCOMPS * total );
            a_grad.resize( 3 * CTS_NCOMPS * total );
            a_gid.resize( total );
        }
        MPI_Gatherv( l_pos.data(), 3 * n_local, MPI_DOUBLE, a_pos.data(),
                     c3.data(), d3.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD );
        MPI_Gatherv( l_chg.data(), CTS_NCOMPS * n_local, MPI_DOUBLE,
                     a_chg.data(), cq.data(), dq.data(), MPI_DOUBLE, 0,
                     MPI_COMM_WORLD );
        MPI_Gatherv( l_pot.data(), CTS_NCOMPS * n_local, MPI_DOUBLE,
                     a_pot.data(), cq.data(), dq.data(), MPI_DOUBLE, 0,
                     MPI_COMM_WORLD );
        MPI_Gatherv( l_grad.data(), 3 * CTS_NCOMPS * n_local, MPI_DOUBLE,
                     a_grad.data(), cg.data(), dg.data(), MPI_DOUBLE, 0,
                     MPI_COMM_WORLD );
        MPI_Gatherv( l_gid.data(), n_local, MPI_INT, a_gid.data(), c1.data(),
                     d1.data(), MPI_INT, 0, MPI_COMM_WORLD );

        if ( rank == 0 )
        {
            gs.n = n_total;
            gs.pos.assign( 3 * n_total, 0.0 );
            gs.chg.assign( CTS_NCOMPS * n_total, 0.0 );
            gs.pot.assign( CTS_NCOMPS * n_total, 0.0 );
            gs.grad.assign( 3 * CTS_NCOMPS * n_total, 0.0 );
            std::vector<int> seen( n_total, 0 );
            std::ostringstream err;
            if ( total != n_total )
                err << "gathered " << total << " particles, expected "
                    << n_total << "; ";
            for ( int i = 0; i < total && err.str().empty(); i++ )
            {
                const int g = a_gid[i];
                if ( g < 0 || g >= n_total )
                {
                    err << "GlobalId " << g << " out of range; ";
                    break;
                }
                if ( seen[g]++ )
                {
                    err << "GlobalId " << g << " appears twice; ";
                    break;
                }
                for ( int d = 0; d < 3; d++ )
                    gs.pos[3 * g + d] = a_pos[3 * i + d];
                for ( int c = 0; c < CTS_NCOMPS; c++ )
                {
                    gs.chg[CTS_NCOMPS * g + c] = a_chg[CTS_NCOMPS * i + c];
                    gs.pot[CTS_NCOMPS * g + c] = a_pot[CTS_NCOMPS * i + c];
                    for ( int d = 0; d < 3; d++ )
                        gs.grad[3 * ( CTS_NCOMPS * g + c ) + d] =
                            a_grad[3 * ( CTS_NCOMPS * i + c ) + d];
                }
            }
            for ( int g = 0; g < n_total && err.str().empty(); g++ )
                if ( seen[g] != 1 )
                {
                    err << "GlobalId " << g << " missing from gathered set; ";
                    break;
                }
            gs.err = err.str();
            gs.valid = gs.err.empty();
        }
    }

    after( gs, nprocs, rank );
}

// ---------------------------------------------------------------------------
// The brute-force O(N^2) SOFTENED reference over the gathered last-step
// state, and the global-scale normalization the deviations are taken in.
//
//   phi_i      =  sum_{j != i} q_j (r^2 + b)^{-1/2}
//   grad phi_i = -sum_{j != i} q_j d (r^2 + b)^{-3/2},   d = x_i - x_j
//
// b = CTS_SOFTENING^2. The distance work is done ONCE per (i, j) and the
// component loop is inside it: the geometry is shared by all NComps solves
// and this is an O(N^2) body run 12 times per suite.
//
// The scales are GLOBAL maxima — max |phi| and max |grad phi| over the
// reference field — not per-particle ratios, for the cancellation reason
// tstLaplaceSolve.hpp:1546-1549 gives: a one-signed charge set still has
// particles near the centre of the cloud whose gradient nearly cancels, and
// a per-particle ratio there measures the cancellation rather than the
// method.
// ---------------------------------------------------------------------------
inline void direct_softened_sum( const GatheredState& gs,
                                 std::vector<double>& bf_pot,
                                 std::vector<double>& bf_grad )
{
    const int n = gs.n;
    const double b = CTS_SOFTENING * CTS_SOFTENING;
    bf_pot.assign( CTS_NCOMPS * n, 0.0 );
    bf_grad.assign( 3 * CTS_NCOMPS * n, 0.0 );

    for ( int i = 0; i < n; i++ )
    {
        double phi[CTS_NCOMPS] = { 0.0 };
        double gx[CTS_NCOMPS] = { 0.0 };
        double gy[CTS_NCOMPS] = { 0.0 };
        double gz[CTS_NCOMPS] = { 0.0 };
        for ( int j = 0; j < n; j++ )
        {
            if ( j == i )
                continue;
            const double dx = gs.pos[3 * i + 0] - gs.pos[3 * j + 0];
            const double dy = gs.pos[3 * i + 1] - gs.pos[3 * j + 1];
            const double dz = gs.pos[3 * i + 2] - gs.pos[3 * j + 2];
            const double inv_r =
                1.0 / std::sqrt( dx * dx + dy * dy + dz * dz + b );
            const double inv_r3 = inv_r * inv_r * inv_r;
            for ( int c = 0; c < CTS_NCOMPS; c++ )
            {
                const double qj = gs.chg[CTS_NCOMPS * j + c];
                phi[c] += qj * inv_r;
                gx[c] -= qj * dx * inv_r3;
                gy[c] -= qj * dy * inv_r3;
                gz[c] -= qj * dz * inv_r3;
            }
        }
        for ( int c = 0; c < CTS_NCOMPS; c++ )
        {
            bf_pot[CTS_NCOMPS * i + c] = phi[c];
            bf_grad[3 * ( CTS_NCOMPS * i + c ) + 0] = gx[c];
            bf_grad[3 * ( CTS_NCOMPS * i + c ) + 1] = gy[c];
            bf_grad[3 * ( CTS_NCOMPS * i + c ) + 2] = gz[c];
        }
    }
}

// ---------------------------------------------------------------------------
// One arm: drive the solve at `mac_theta`, compare against the direct
// softened sum, print the deviations, and assert them against `tol`.
// ---------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace, int P,
          template <class, int, int> class Basis>
void runArm( double mac_theta, double tol, const char* arm )
{
    with_cartesian_taylor_solve<MemorySpace, ExecutionSpace, P, Basis>(
        mac_theta,
        [mac_theta, tol, arm]( const GatheredState& gs, int nprocs, int rank )
        {
            if ( rank != 0 )
                return;

            ASSERT_TRUE( gs.valid )
                << "gathered last-step state is invalid: " << gs.err;

            const int n = gs.n;
            std::vector<double> bf_pot, bf_grad;
            direct_softened_sum( gs, bf_pot, bf_grad );

            double pot_scale = 0.0, grad_scale = 0.0;
            for ( int i = 0; i < CTS_NCOMPS * n; i++ )
            {
                pot_scale = std::max( pot_scale, std::abs( bf_pot[i] ) );
                const double gm =
                    std::sqrt( bf_grad[3 * i + 0] * bf_grad[3 * i + 0] +
                               bf_grad[3 * i + 1] * bf_grad[3 * i + 1] +
                               bf_grad[3 * i + 2] * bf_grad[3 * i + 2] );
                grad_scale = std::max( grad_scale, gm );
            }
            ASSERT_GT( pot_scale, 0.0 );
            ASSERT_GT( grad_scale, 0.0 );

            double max_pot_dev = 0.0, max_grad_dev = 0.0;
            for ( int i = 0; i < CTS_NCOMPS * n; i++ )
            {
                max_pot_dev = std::max(
                    max_pot_dev,
                    std::abs( gs.pot[i] - bf_pot[i] ) / pot_scale );
                const double dx = gs.grad[3 * i + 0] - bf_grad[3 * i + 0];
                const double dy = gs.grad[3 * i + 1] - bf_grad[3 * i + 1];
                const double dz = gs.grad[3 * i + 2] - bf_grad[3 * i + 2];
                max_grad_dev = std::max(
                    max_grad_dev,
                    std::sqrt( dx * dx + dy * dy + dz * dz ) / grad_scale );
            }

            std::printf( "[ct-solve] theta=%.17g p_order=%d nprocs=%d arm=%s "
                         "direct_softened_sum max_pot_dev=%.17g "
                         "max_grad_dev=%.17g tol=%.17g bar=%.17g "
                         "pot_scale=%.17g grad_scale=%.17g\n",
                         mac_theta, P, nprocs, arm, max_pot_dev, max_grad_dev,
                         tol, CTS_REFERENCE_BAR, pot_scale, grad_scale );
            std::fflush( stdout );

            EXPECT_LT( max_pot_dev, tol )
                << "np=" << nprocs << " theta=" << mac_theta
                << ": the potential does not match the direct SOFTENED sum at "
                   "the pinned deviation. This is the whole claim of the "
                   "basis — a far field that is accurate at "
                   "near_softening_factor = 0 — so read it as a defect, not "
                   "as a tolerance to widen. R1 then R2 in "
                   "tasks/cartesian-taylor-basis.md.";
            EXPECT_LT( max_grad_dev, tol )
                << "np=" << nprocs << " theta=" << mac_theta
                << ": the gradient does not match the direct SOFTENED sum at "
                   "the pinned deviation. R1 then R2 in "
                   "tasks/cartesian-taylor-basis.md.";
        } );
}

} // namespace CartesianTaylorSolveTest

//---------------------------------------------------------------------------//
// np 1-6, mac_theta = 0.3 — the reference treecode's admissibility. This is
// the arm the 1e-3 accuracy claim is made against, and CTS_DEV_TOL_THETA_REF
// is static_asserted to sit at or below that bar.
//---------------------------------------------------------------------------//
TEST( CartesianTaylorSolve, matchesDirectSumThetaRef )
{
    CartesianTaylorSolveTest::runArm<TEST_MEMSPACE, TEST_EXECSPACE,
                                     CartesianTaylorSolveTest::CTS_P,
                                     Canopy::CartesianTaylorBasis>(
        CartesianTaylorSolveTest::CTS_THETA_REF,
        CartesianTaylorSolveTest::CTS_DEV_TOL_THETA_REF, "theta_ref" );
}

//---------------------------------------------------------------------------//
// np 1-6, mac_theta = 0.5 — Canopy's own default and the value
// tstLaplaceSolve freezes. Measured and pinned beside the 0.3 arm, and it
// lands ABOVE the reference bar on both fields (8.67e-04 potential,
// 2.16e-02 gradient) because this MAC admits pairs at R ~ 6.93 W where the
// p = 2 Taylor truncation is largest — 2.6x the theta = 0.3 gradient and
// 3.3x its potential. That ordering is the expected one and is the point of
// running two arms: the 1e-3 figure transfers at matched admissibility and
// nowhere else. This arm PASSES against its own pinned constant.
//---------------------------------------------------------------------------//
TEST( CartesianTaylorSolve, matchesDirectSumThetaCanopy )
{
    CartesianTaylorSolveTest::runArm<TEST_MEMSPACE, TEST_EXECSPACE,
                                     CartesianTaylorSolveTest::CTS_P,
                                     Canopy::CartesianTaylorBasis>(
        CartesianTaylorSolveTest::CTS_THETA_CANOPY,
        CartesianTaylorSolveTest::CTS_DEV_TOL_THETA_CANOPY, "theta_canopy" );
}

//---------------------------------------------------------------------------//

} // end namespace Test
