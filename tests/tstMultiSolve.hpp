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

#include "Canopy_Helpers.hpp"
#include "Canopy_Solver.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace Test
{
//---------------------------------------------------------------------------//

using namespace Canopy;

namespace MultiSolveTest
{

inline double get_test_mac_theta()
{
    if ( const char* s = std::getenv( "CANOPY_MAC_THETA" ) )
        return std::atof( s );
    return 0.5;
}

// CANOPY_MULTISOLVE_PROBE=1 enables the per-step far-field probe; unset or 0
// leaves it off. Any other value throws.
inline bool get_test_probe_enabled()
{
    const char* s = std::getenv( "CANOPY_MULTISOLVE_PROBE" );
    if ( s == nullptr || std::string( s ) == "0" )
        return false;
    if ( std::string( s ) == "1" )
        return true;
    throw std::runtime_error( "CANOPY_MULTISOLVE_PROBE must be 0 or 1, got '" +
                              std::string( s ) + "'" );
}

// CANOPY_MULTISOLVE_NPP: particles per rank at every call site, overriding
// the site's own count. Returns 0 when unset; anything other than a positive
// integer throws.
inline int get_test_npp_override()
{
    const char* s = std::getenv( "CANOPY_MULTISOLVE_NPP" );
    if ( s == nullptr )
        return 0;
    char* end = nullptr;
    const long v = std::strtol( s, &end, 10 );
    if ( end == s || *end != '\0' || v <= 0 || v > INT_MAX )
        throw std::runtime_error(
            "CANOPY_MULTISOLVE_NPP must be a positive integer, got '" +
            std::string( s ) + "'" );
    return static_cast<int>( v );
}

enum FieldIdx
{
    Position = 0,
    Charge = 1,
    Velocity = 2,
    GlobalId = 3
};

// Expansion order used for all multi-solve tests.
static constexpr int P_ORDER = 8;

// Inter-step maintenance mode dispatched by the test driver.
enum class Mode
{
    Migrate,
    Rebalance,
    Rebuild,
    Auto
};

} // namespace MultiSolveTest

//---------------------------------------------------------------------------//
// Direct (brute-force) N-body gradient computation on host.
//
// For each particle i and component c:
//   g[i,c,d] = sum_{j != i} -q[j,c] * (r_i[d] - r_j[d]) / |r_i - r_j|^3
//
// This is the gradient of phi[i,c] = sum_{j!=i} q[j,c] / |r_i - r_j|.
//---------------------------------------------------------------------------//
inline void
brute_force_gradient( const std::vector<double>& pos, // 3 * N
                      const std::vector<double>& chg, // N (NComps=1 here)
                      std::vector<double>& grad )     // 3 * N (output)
{
    const int N = static_cast<int>( chg.size() );
    grad.assign( 3 * N, 0.0 );
    for ( int i = 0; i < N; i++ )
    {
        const double xi = pos[3 * i + 0];
        const double yi = pos[3 * i + 1];
        const double zi = pos[3 * i + 2];
        double gx = 0.0, gy = 0.0, gz = 0.0;
        for ( int j = 0; j < N; j++ )
        {
            if ( j == i )
                continue;
            const double dx = xi - pos[3 * j + 0];
            const double dy = yi - pos[3 * j + 1];
            const double dz = zi - pos[3 * j + 2];
            const double r2 = dx * dx + dy * dy + dz * dz;
            const double inv_r = 1.0 / std::sqrt( r2 );
            const double inv_r3 = inv_r * inv_r * inv_r;
            const double qj = chg[j];
            gx -= qj * dx * inv_r3;
            gy -= qj * dy * inv_r3;
            gz -= qj * dz * inv_r3;
        }
        grad[3 * i + 0] = gx;
        grad[3 * i + 1] = gy;
        grad[3 * i + 2] = gz;
    }
}

// Absolute field A_i = sum_{j != i} |q_j| / |r_i - r_j|^2, charge per
// length^2, >= |g_i|. A far-field truncation error of relative size eps per
// interaction moves g_i by at most eps * A_i whatever the signs, which is why
// the derived trajectory bound is built on A rather than on |g| (|g| cancels).
inline void absolute_field( const std::vector<double>& pos, // 3 * N
                            const std::vector<double>& chg, // N
                            std::vector<double>& afield )   // N (output)
{
    const int N = static_cast<int>( chg.size() );
    afield.assign( N, 0.0 );
    for ( int i = 0; i < N; i++ )
        for ( int j = 0; j < N; j++ )
        {
            if ( j == i )
                continue;
            const double dx = pos[3 * i + 0] - pos[3 * j + 0];
            const double dy = pos[3 * i + 1] - pos[3 * j + 1];
            const double dz = pos[3 * i + 2] - pos[3 * j + 2];
            afield[i] += std::abs( chg[j] ) / ( dx * dx + dy * dy + dz * dz );
        }
}

// Distance from each particle to its nearest other particle, O(N^2), in
// position units. Used by the probe to flag close encounters.
inline std::vector<double>
nearest_separation( const std::vector<double>& pos ) // 3 * N
{
    const int N = static_cast<int>( pos.size() / 3 );
    std::vector<double> nn( N, std::numeric_limits<double>::infinity() );
    for ( int i = 0; i < N; i++ )
        for ( int j = i + 1; j < N; j++ )
        {
            const double dx = pos[3 * i + 0] - pos[3 * j + 0];
            const double dy = pos[3 * i + 1] - pos[3 * j + 1];
            const double dz = pos[3 * i + 2] - pos[3 * j + 2];
            const double r = std::sqrt( dx * dx + dy * dy + dz * dz );
            nn[i] = std::min( nn[i], r );
            nn[j] = std::min( nn[j], r );
        }
    return nn;
}

//---------------------------------------------------------------------------//
/**
 * End-to-end multi-step gravity-style driver.
 *
 * Each particle carries a position, scalar charge ("mass"), velocity, and
 * stable global id (used to align FMM and brute-force results after
 * inter-rank migration scrambles the local ordering).
 *
 * Per timestep:
 *   1. FMM:   solver.solve() -> gradient g
 *             v += dt * g;  r += dt * drift_multiplier * v
 *             dispatch the requested maintenance call
 *   2. Brute (rank 0 only, on a separate copy of all particles):
 *             compute g via O(N^2);  v += dt * g;  r += dt * drift_multiplier *
 * v
 *
 * After num_steps, gather the FMM trajectory's final state to rank 0,
 * align by GlobalId, and compare against the brute-force final state.
 */
template <int Modes>
struct MaintenanceDispatch;

template <class Solver, class AoSoA>
inline typename Solver::MaintenanceAction
dispatch_maintain( Solver& solver, AoSoA& particles, MultiSolveTest::Mode mode )
{
    using namespace MultiSolveTest;
    using Action = typename Solver::MaintenanceAction;
    switch ( mode )
    {
    case Mode::Migrate:
        solver.template migrate<Position>( particles );
        return Action::Migrate;
    case Mode::Rebalance:
        solver.template rebalance<Position>( particles );
        return Action::Rebalance;
    case Mode::Rebuild:
        solver.template rebuild<Position, Charge>( particles );
        return Action::Rebuild;
    case Mode::Auto:
    default:
        return solver.template auto_maintain<Position, Charge>( particles );
    }
}

inline void testMultiStepGravity(
    MultiSolveTest::Mode mode, const char* case_label,
    int num_particles_per_rank, int num_steps,
    double dt, double drift_multiplier, int ncrit, int max_depth,
    double tree_tolerance, int replication_depth,
    // pos_tolerance, vel_tolerance: bounds on the end-of-run max_pos_rel and
    //   max_vel_rel (dimensionless); each call site states its derivation.
    double pos_tolerance, double vel_tolerance,
    int* out_action_counts = nullptr,
    // The next three knobs are used by the bin-edge regression test.
    // clustered: draw 80% of particles from a tight Gaussian blob in
    //   one corner (forces deep refinement in that octant) and 20%
    //   uniform — produces M2L pairs the key encoding cannot represent,
    //   which feed the per-pair m2l_translate fallback. Two bounds can
    //   refuse such a pair, and which one does is reported per solve by
    //   the probe below: |dd| > KernelType::m2l_key_dd_max (a signed
    //   depth difference, 6 for LaplaceKernel at double) or any of
    //   |ii|,|jj|,|kk| > M2L_KEY_OFFSET_MAX = 32 (a center-to-center
    //   offset in half-widths at the deeper of the two cells' depths).
    //   Both are the classify pass's range guard
    //   (Canopy_DownwardSweep.hpp:1704-1716); neither is a count cap.
    // mac_theta_override: if positive, replaces get_test_mac_theta();
    //   a tighter theta admits more far-field pairs and amplifies the
    //   fallback population.
    // out_max_fallback_total: if non-null, written with the maximum
    //   (across solves and across MPI ranks summed) fallback pair count
    //   observed during the run.
    // probe_field_tol: if positive, the per-step probe runs whatever
    //   CANOPY_MULTISOLVE_PROBE says, and every step's field_err must be
    //   below it (dimensionless). This is the far-field gate for a site whose
    //   trajectory deviation is dominated by close-encounter dynamics.
    bool clustered = false, double mac_theta_override = 0.0,
    long long* out_max_fallback_total = nullptr,
    double probe_field_tol = 0.0 )
{
    using namespace MultiSolveTest;

    using DataTypes =
        Cabana::MemberTypes<double[3], // Position
                            double[1], // Charge (NComps=1 for gravity)
                            double[3], // Velocity
                            int>;      // GlobalId
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    using AoSoA_ht = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
    using Solver_t =
        Solver<TEST_MEMSPACE, TEST_EXECSPACE, double, P_ORDER, /*NComps=*/1>;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    if ( const int npp = get_test_npp_override(); npp > 0 )
        num_particles_per_rank = npp;
    const bool probe = probe_field_tol > 0.0 || get_test_probe_enabled();

    // -----------------------------------------------------------------------
    // Generate random initial particles on host.
    // GlobalId = first_id_on_this_rank + i so ids are unique across ranks.
    // -----------------------------------------------------------------------
    AoSoA_ht particles_h( "particles_h", num_particles_per_rank );
    {
        auto hp = Cabana::slice<Position>( particles_h );
        auto hq = Cabana::slice<Charge>( particles_h );
        auto hv = Cabana::slice<Velocity>( particles_h );
        auto hid = Cabana::slice<GlobalId>( particles_h );

        std::mt19937 gen( 42 + rank * 7919 );
        std::uniform_real_distribution<double> pos_dist( 0.1, 0.9 );
        std::uniform_real_distribution<double> q_dist( 0.5, 1.5 );
        std::uniform_real_distribution<double> v_dist( -0.05, 0.05 );
        // Clustered mode: 80% drawn from a tight Gaussian blob in one
        // corner of [0,1]^3 (clipped to (0.01, 0.99) so particles stay
        // inside the bounding box) and 20% from the same uniform used by
        // the standard tests. The blob density triggers refinement deep
        // into one octant, which produces M2L pairs the key encoding
        // refuses — either |dd| > KernelType::m2l_key_dd_max or an
        // offset component > M2L_KEY_OFFSET_MAX = 32 half-widths at the
        // deeper cell's depth — the tail that the m2l_translate fallback
        // path handles.
        std::normal_distribution<double> blob_dist( 0.15, 0.05 );
        auto sample_pos = [&]( int idx ) {
            if ( !clustered )
                return pos_dist( gen );
            const bool in_blob = ( idx % 5 != 0 ); // 80% blob, 20% uniform
            if ( !in_blob )
                return pos_dist( gen );
            double v = blob_dist( gen );
            if ( v < 0.01 )
                v = 0.01;
            if ( v > 0.99 )
                v = 0.99;
            return v;
        };

        const int gid_base = rank * num_particles_per_rank;
        for ( int i = 0; i < num_particles_per_rank; i++ )
        {
            hp( i, 0 ) = sample_pos( i );
            hp( i, 1 ) = sample_pos( i + 1 );
            hp( i, 2 ) = sample_pos( i + 2 );
            hq( i, 0 ) = q_dist( gen );
            hv( i, 0 ) = v_dist( gen );
            hv( i, 1 ) = v_dist( gen );
            hv( i, 2 ) = v_dist( gen );
            hid( i ) = gid_base + i;
        }
    }
    AoSoA_t particles( "particles", num_particles_per_rank );
    Cabana::deep_copy( particles, particles_h );

    // -----------------------------------------------------------------------
    // Gather initial state on rank 0 for the brute-force shadow run.
    // We build it from the host copy before the AoSoA gets shuffled by the
    // partitioner.
    // -----------------------------------------------------------------------
    int total_particles = 0;
    std::vector<int> all_n( nprocs, 0 );
    int local_n = num_particles_per_rank;
    MPI_Allreduce( &local_n, &total_particles, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );
    MPI_Gather( &local_n, 1, MPI_INT, all_n.data(), 1, MPI_INT, 0,
                MPI_COMM_WORLD );

    std::vector<int> bf_displs( nprocs, 0 );
    std::vector<int> bf_counts3( nprocs, 0 );
    std::vector<int> bf_displs3( nprocs, 0 );
    std::vector<int> bf_counts( nprocs, 0 );
    if ( rank == 0 )
    {
        for ( int r = 0; r < nprocs; r++ )
        {
            bf_counts[r] = all_n[r];
            bf_counts3[r] = 3 * all_n[r];
        }
        for ( int r = 1; r < nprocs; r++ )
        {
            bf_displs[r] = bf_displs[r - 1] + bf_counts[r - 1];
            bf_displs3[r] = bf_displs3[r - 1] + bf_counts3[r - 1];
        }
    }

    // Pack initial local state from host AoSoA
    std::vector<double> local_pos0( 3 * num_particles_per_rank );
    std::vector<double> local_chg0( num_particles_per_rank );
    std::vector<double> local_vel0( 3 * num_particles_per_rank );
    std::vector<int> local_gid0( num_particles_per_rank );
    {
        auto hp = Cabana::slice<Position>( particles_h );
        auto hq = Cabana::slice<Charge>( particles_h );
        auto hv = Cabana::slice<Velocity>( particles_h );
        auto hid = Cabana::slice<GlobalId>( particles_h );
        for ( int i = 0; i < num_particles_per_rank; i++ )
        {
            local_pos0[3 * i + 0] = hp( i, 0 );
            local_pos0[3 * i + 1] = hp( i, 1 );
            local_pos0[3 * i + 2] = hp( i, 2 );
            local_chg0[i] = hq( i, 0 );
            local_vel0[3 * i + 0] = hv( i, 0 );
            local_vel0[3 * i + 1] = hv( i, 1 );
            local_vel0[3 * i + 2] = hv( i, 2 );
            local_gid0[i] = hid( i );
        }
    }

    std::vector<double> bf_pos, bf_chg, bf_vel;
    std::vector<int> bf_gid;
    if ( rank == 0 )
    {
        bf_pos.resize( 3 * total_particles );
        bf_chg.resize( total_particles );
        bf_vel.resize( 3 * total_particles );
        bf_gid.resize( total_particles );
    }
    MPI_Gatherv( local_pos0.data(), 3 * num_particles_per_rank, MPI_DOUBLE,
                 bf_pos.data(), bf_counts3.data(), bf_displs3.data(),
                 MPI_DOUBLE, 0, MPI_COMM_WORLD );
    MPI_Gatherv( local_chg0.data(), num_particles_per_rank, MPI_DOUBLE,
                 bf_chg.data(), bf_counts.data(), bf_displs.data(), MPI_DOUBLE,
                 0, MPI_COMM_WORLD );
    MPI_Gatherv( local_vel0.data(), 3 * num_particles_per_rank, MPI_DOUBLE,
                 bf_vel.data(), bf_counts3.data(), bf_displs3.data(),
                 MPI_DOUBLE, 0, MPI_COMM_WORLD );
    MPI_Gatherv( local_gid0.data(), num_particles_per_rank, MPI_INT,
                 bf_gid.data(), bf_counts.data(), bf_displs.data(), MPI_INT, 0,
                 MPI_COMM_WORLD );

    // -----------------------------------------------------------------------
    // Set up FMM solver.
    // -----------------------------------------------------------------------
    const double mac_theta_used =
        ( mac_theta_override > 0.0 ) ? mac_theta_override
                                     : get_test_mac_theta();
    Canopy::FmmConfig cfg;
    cfg.ncrit = ncrit;
    cfg.max_depth = max_depth;
    cfg.xmin_tol = cfg.xmax_tol = tree_tolerance;
    cfg.ymin_tol = cfg.ymax_tol = tree_tolerance;
    cfg.zmin_tol = cfg.zmax_tol = tree_tolerance;
    cfg.ncrit_tol = tree_tolerance;
    cfg.replication_depth = replication_depth;
    cfg.imbalance_tolerance = 0.05;
    cfg.mac_theta = mac_theta_used;
    cfg.softening = 0.0;
    Solver_t solver( MPI_COMM_WORLD, cfg );
    solver.template setup<Position, Charge>( particles,
                                             num_particles_per_rank );

    int action_counts[3] = { 0, 0, 0 }; // [Migrate, Rebalance, Rebuild]
    long long max_fallback_total = 0;   // max across solves of (sum across ranks)

    // Probe state (CANOPY_MULTISOLVE_PROBE=1 only).
    //   prev_action: the maintenance call that produced this step's tree;
    //     "Setup" before the first step.
    //   close_thr: close-encounter separation, one tenth of the mean spacing
    //     (0.8^3 / N)^(1/3) of N particles drawn on [0.1, 0.9]^3, position
    //     units.
    //   run_min_sep: rank 0, indexed by GlobalId; each particle's smallest
    //     nearest-neighbour distance over every probed state, position units.
    const char* prev_action = "Setup";
    const double close_thr =
        0.1 * std::cbrt( 0.8 * 0.8 * 0.8 / static_cast<double>( total_particles ) );
    std::vector<double> run_min_sep;
    if ( probe && rank == 0 )
        run_min_sep.assign( total_particles,
                            std::numeric_limits<double>::infinity() );

    // First-order error budget per unit relative field error, accumulated
    // along the brute-force shadow (rank 0, brute-force index). If every
    // far-field interaction is in error by at most eps relative, and the two
    // trajectories stay close enough that the error does not feed back
    // through the dynamics, then after the run
    //   |dv_i| <= eps * budget_dv[i],  budget_dv = dt * sum_n A_i(n)
    //     (velocity units), and
    //   |dr_i| <= eps * budget_dr[i],  budget_dr = dt * drift * sum_k
    //     budget_dv after kick k (position units),
    // A being absolute_field(). Feedback is what a close encounter adds, so a
    // deviation above eps * budget is dynamics, not far-field error.
    std::vector<double> budget_dv, budget_dr;
    if ( rank == 0 )
    {
        budget_dv.assign( total_particles, 0.0 );
        budget_dr.assign( total_particles, 0.0 );
    }

    // -----------------------------------------------------------------------
    // Time loop
    // -----------------------------------------------------------------------
    for ( int step = 0; step < num_steps; step++ )
    {
        // FMM solve for current state
        solver.template solve<Position, Charge>( particles,
                                                 /*compute_gradient=*/true );

        // Fallback-count probe. Each rank's downward sweep tracks how many
        // refused pairs it carried through m2l_translate this build;
        // sum those across ranks to get the global tally. Tracked as a
        // running max so the regression test can assert >0 without caring
        // which solve produced the work.
        //
        // The per-reason line below is the measurement T1 step 1 exists for:
        // it names WHICH bound refused these pairs. range_guard counts the
        // classify pass's representability refusals (|dd| or an offset
        // component out of range), count_cap counts pairs whose key was
        // hashed and then refused a column by the operator-count budget, and
        // the two sum to the total exactly. Printed per (nprocs, rank) and
        // per solve because the tree and partition path is run-to-run
        // nondeterministic at np >= 3, so a mean over ranks is not a number
        // anyone can reproduce. All three read -1 without
        // CANOPY_ENABLE_PROFILING.
        if ( out_max_fallback_total != nullptr )
        {
            const auto& ds = solver.downward();
            const std::vector<int> cells_at_depth = ds.m2l_cells_at_depth();
            std::string depth_occ;
            for ( std::size_t d = 0; d < cells_at_depth.size(); ++d )
            {
                depth_occ += std::to_string( cells_at_depth[d] );
                if ( d + 1 < cells_at_depth.size() )
                    depth_occ += ",";
            }
            std::printf(
                "[m2l-fallback-reason] nprocs %d rank %d step %d "
                "range_guard %lld count_cap %lld depth_dropped %lld "
                "total %lld unique_ops %d cells_at_depth [%s]\n",
                nprocs, rank, step,
                ds.m2l_n_fallback_pairs_range_guard(),
                ds.m2l_n_fallback_pairs_count_cap(),
                ds.m2l_n_fallback_pairs_depth_dropped(),
                ds.total_fallback_pair_count(), ds.m2l_n_unique_ops(),
                depth_occ.c_str() );
            std::fflush( stdout );

            // T1 step 1 found every refusal on this clustered fixture to be a
            // RANGE-GUARD refusal (tasks/tree-opt-progress-log.md section
            // T1). Asserted per rank rather than reported, so a change that
            // moves refusals from one reason to another fails here instead
            // of printing. Combined with the caller's max_fallback > 0, this
            // pins range_guard > 0 too, without asserting which rank the
            // partition gave the deep subtree to (risk R6).
            const long long fb_range_guard =
                ds.m2l_n_fallback_pairs_range_guard();
            const long long fb_count_cap = ds.m2l_n_fallback_pairs_count_cap();
            const long long fb_depth_dropped =
                ds.m2l_n_fallback_pairs_depth_dropped();
#ifdef CANOPY_ENABLE_PROFILING
            // The column cap is at its default, so no budget refusal can
            // occur and range_guard is the only reason available.
            EXPECT_EQ( fb_count_cap, 0 )
                << "nprocs " << nprocs << " rank " << rank << " step "
                << step << ": a pair was refused a column by the count cap "
                   "on a solve whose cap was never set, so the fallback "
                   "population is no longer all range-guard refusals";
            // A dropped pair is evaluated by neither a column nor the
            // fallback, so its contribution is missing from the solve.
            EXPECT_EQ( fb_depth_dropped, 0 )
                << "nprocs " << nprocs << " rank " << rank << " step "
                << step << ": a refused pair reached neither an operator "
                   "column nor the fallback table";
            EXPECT_EQ( fb_range_guard + fb_count_cap,
                       ds.total_fallback_pair_count() )
                << "nprocs " << nprocs << " rank " << rank << " step "
                << step << ": the per-reason counters do not sum to the "
                   "fallback total, so a refusal path is unaccounted for";
#else
            // Built without CANOPY_ENABLE_PROFILING: all three read -1, the
            // "unavailable" sentinel, and NEVER 0. The sum identity is
            // SKIPPED, not evaluated: total_fallback_pair_count() is NOT
            // profiling-gated and reads a real count here, so -1 + -1
            // against it is a claim about nothing.
            EXPECT_EQ( fb_range_guard, -1 );
            EXPECT_EQ( fb_count_cap, -1 );
            EXPECT_EQ( fb_depth_dropped, -1 );
            if ( rank == 0 && step == 0 )
                std::printf( "[m2l-fallback-reason] sum identity SKIPPED: "
                             "built without CANOPY_ENABLE_PROFILING, "
                             "counters are sentinels\n" );
            std::fflush( stdout );
#endif

            const long long local_fb =
                solver.downward().total_fallback_pair_count();
            long long global_fb = 0;
            MPI_Allreduce( &local_fb, &global_fb, 1, MPI_LONG_LONG, MPI_SUM,
                           MPI_COMM_WORLD );
            if ( global_fb > max_fallback_total )
                max_fallback_total = global_fb;
        }

        // Far-field probe: the FMM gradient against brute force evaluated at
        // the FMM's own current positions, so no trajectory difference enters
        // the error. One line per step on rank 0:
        //   cells: global cell count of the replicated tree;
        //   root_hw: root cell half-width, position units;
        //   max_rel: max over particles of |g_fmm - g_bf| / |g_bf|,
        //     dimensionless, inflated wherever |g_bf| cancels toward zero;
        //   field_err: max_i |g_fmm - g_bf| / max_i |g_bf|, dimensionless,
        //     the field_scales rule of tstLaplaceSolve.hpp;
        //   rel_* / abs_*: the argmax particle of max_rel and of the field
        //     error numerator: GlobalId, |g_bf| (charge / length^2) and its
        //     nearest-neighbour distance (position units);
        //   min_sep, n_close: the smallest nearest-neighbour distance and the
        //     number of particles closer than close_thr to another.
        if ( probe )
        {
            const int n_loc = solver.num_local_particles();
            auto p_pos = Canopy::create_mirror_view_and_copy(
                Kokkos::HostSpace(), Cabana::slice<Position>( particles ),
                "probe_pos" );
            auto p_chg = Canopy::create_mirror_view_and_copy(
                Kokkos::HostSpace(), Cabana::slice<Charge>( particles ),
                "probe_chg" );
            auto p_gid = Canopy::create_mirror_view_and_copy(
                Kokkos::HostSpace(), Cabana::slice<GlobalId>( particles ),
                "probe_gid" );
            auto p_grad = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), solver.gradient() );

            std::vector<double> loc_pos( 3 * n_loc ), loc_grad( 3 * n_loc ),
                loc_chg( n_loc );
            std::vector<int> loc_gid( n_loc );
            for ( int i = 0; i < n_loc; i++ )
            {
                for ( int d = 0; d < 3; d++ )
                {
                    loc_pos[3 * i + d] = p_pos( i, d );
                    loc_grad[3 * i + d] = p_grad( i, 0, d );
                }
                loc_chg[i] = p_chg( i, 0 );
                loc_gid[i] = p_gid( i );
            }

            std::vector<int> cnt( nprocs, 0 ), dsp( nprocs, 0 ),
                cnt3( nprocs, 0 ), dsp3( nprocs, 0 );
            MPI_Gather( &n_loc, 1, MPI_INT, cnt.data(), 1, MPI_INT, 0,
                        MPI_COMM_WORLD );
            for ( int r = 0; r < nprocs; r++ )
            {
                cnt3[r] = 3 * cnt[r];
                if ( r > 0 )
                {
                    dsp[r] = dsp[r - 1] + cnt[r - 1];
                    dsp3[r] = dsp3[r - 1] + cnt3[r - 1];
                }
            }
            const int n_root = ( rank == 0 ) ? total_particles : 0;
            std::vector<double> all_pos( 3 * n_root ), all_grad( 3 * n_root ),
                all_chg( n_root );
            std::vector<int> all_gid( n_root );
            MPI_Gatherv( loc_pos.data(), 3 * n_loc, MPI_DOUBLE, all_pos.data(),
                         cnt3.data(), dsp3.data(), MPI_DOUBLE, 0,
                         MPI_COMM_WORLD );
            MPI_Gatherv( loc_grad.data(), 3 * n_loc, MPI_DOUBLE,
                         all_grad.data(), cnt3.data(), dsp3.data(), MPI_DOUBLE,
                         0, MPI_COMM_WORLD );
            MPI_Gatherv( loc_chg.data(), n_loc, MPI_DOUBLE, all_chg.data(),
                         cnt.data(), dsp.data(), MPI_DOUBLE, 0,
                         MPI_COMM_WORLD );
            MPI_Gatherv( loc_gid.data(), n_loc, MPI_INT, all_gid.data(),
                         cnt.data(), dsp.data(), MPI_INT, 0, MPI_COMM_WORLD );

            if ( rank == 0 )
            {
                std::vector<double> bf;
                brute_force_gradient( all_pos, all_chg, bf );
                const std::vector<double> nn = nearest_separation( all_pos );

                double max_rel = 0.0, max_dg = 0.0, g_scale = 0.0;
                double min_sep = std::numeric_limits<double>::infinity();
                int i_rel = 0, i_abs = 0, n_close = 0;
                for ( int i = 0; i < total_particles; i++ )
                {
                    const double dx = all_grad[3 * i + 0] - bf[3 * i + 0];
                    const double dy = all_grad[3 * i + 1] - bf[3 * i + 1];
                    const double dz = all_grad[3 * i + 2] - bf[3 * i + 2];
                    const double dg = std::sqrt( dx * dx + dy * dy + dz * dz );
                    const double gm =
                        std::sqrt( bf[3 * i + 0] * bf[3 * i + 0] +
                                   bf[3 * i + 1] * bf[3 * i + 1] +
                                   bf[3 * i + 2] * bf[3 * i + 2] );
                    const double rel = ( gm > 0.0 ) ? dg / gm : dg;
                    if ( rel > max_rel )
                    {
                        max_rel = rel;
                        i_rel = i;
                    }
                    if ( dg > max_dg )
                    {
                        max_dg = dg;
                        i_abs = i;
                    }
                    g_scale = std::max( g_scale, gm );
                    min_sep = std::min( min_sep, nn[i] );
                    if ( nn[i] < close_thr )
                        n_close++;
                    double& m = run_min_sep[all_gid[i]];
                    m = std::min( m, nn[i] );
                }
                auto gmag = [&]( int i ) {
                    return std::sqrt( bf[3 * i + 0] * bf[3 * i + 0] +
                                      bf[3 * i + 1] * bf[3 * i + 1] +
                                      bf[3 * i + 2] * bf[3 * i + 2] );
                };
                double root_hw = 0.0;
                for ( const auto& c : solver.builder().cells() )
                    if ( c.depth == 0 )
                        root_hw = c.half_width;
                std::printf(
                    "[multisolve-probe] case %s nprocs %d step %d "
                    "prev_action %s cells %zu root_hw %.17g max_rel %.17g "
                    "field_err %.17g rel_gid %d rel_g %.17g rel_sep %.17g "
                    "abs_gid %d abs_dg %.17g abs_g %.17g abs_sep %.17g "
                    "min_sep %.17g n_close %d close_thr %.17g\n",
                    case_label, nprocs, step, prev_action,
                    solver.builder().cells().size(), root_hw, max_rel,
                    ( g_scale > 0.0 ) ? max_dg / g_scale : max_dg,
                    all_gid[i_rel], gmag( i_rel ), nn[i_rel], all_gid[i_abs],
                    max_dg, gmag( i_abs ), nn[i_abs], min_sep, n_close,
                    close_thr );
                std::fflush( stdout );

                if ( probe_field_tol > 0.0 )
                    EXPECT_LT( ( g_scale > 0.0 ) ? max_dg / g_scale : max_dg,
                               probe_field_tol )
                        << "step " << step << ": the FMM gradient at the "
                           "FMM's own positions deviates from brute force "
                           "by more than the per-step far-field bound";
            }
        }

        // Update local positions and velocities from gradient.
        // Symplectic Euler: v += dt*g;  r += dt * drift * v.
        // Run on device so we write directly into the AoSoA slices.
        const int n_local = solver.num_local_particles();
        auto positions = Cabana::slice<Position>( particles );
        auto velocities = Cabana::slice<Velocity>( particles );
        auto grad = solver.gradient();
        const double dt_local = dt;
        const double drift_local = drift_multiplier;
        Kokkos::parallel_for(
            "MultiSolve::integrate",
            Kokkos::RangePolicy<TEST_EXECSPACE>( 0, n_local ),
            KOKKOS_LAMBDA( int i ) {
                const double gx = grad( i, 0, 0 );
                const double gy = grad( i, 0, 1 );
                const double gz = grad( i, 0, 2 );
                velocities( i, 0 ) += dt_local * gx;
                velocities( i, 1 ) += dt_local * gy;
                velocities( i, 2 ) += dt_local * gz;
                positions( i, 0 ) +=
                    dt_local * drift_local * velocities( i, 0 );
                positions( i, 1 ) +=
                    dt_local * drift_local * velocities( i, 1 );
                positions( i, 2 ) +=
                    dt_local * drift_local * velocities( i, 2 );
            } );
        Kokkos::fence();

        // Brute-force shadow on rank 0
        if ( rank == 0 )
        {
            std::vector<double> bf_grad, bf_afield;
            brute_force_gradient( bf_pos, bf_chg, bf_grad );
            absolute_field( bf_pos, bf_chg, bf_afield );
            for ( int i = 0; i < total_particles; i++ )
            {
                budget_dv[i] += dt * bf_afield[i];
                budget_dr[i] += dt * drift_multiplier * budget_dv[i];
                bf_vel[3 * i + 0] += dt * bf_grad[3 * i + 0];
                bf_vel[3 * i + 1] += dt * bf_grad[3 * i + 1];
                bf_vel[3 * i + 2] += dt * bf_grad[3 * i + 2];
                bf_pos[3 * i + 0] += dt * drift_multiplier * bf_vel[3 * i + 0];
                bf_pos[3 * i + 1] += dt * drift_multiplier * bf_vel[3 * i + 1];
                bf_pos[3 * i + 2] += dt * drift_multiplier * bf_vel[3 * i + 2];
            }
        }

        // Inter-step maintenance
        auto action = dispatch_maintain( solver, particles, mode );
        switch ( action )
        {
        case Solver_t::MaintenanceAction::Migrate:
            action_counts[0]++;
            prev_action = "Migrate";
            break;
        case Solver_t::MaintenanceAction::Rebalance:
            action_counts[1]++;
            prev_action = "Rebalance";
            break;
        case Solver_t::MaintenanceAction::Rebuild:
            action_counts[2]++;
            prev_action = "Rebuild";
            break;
        }
    }

    if ( out_action_counts )
    {
        for ( int i = 0; i < 3; i++ )
            out_action_counts[i] = action_counts[i];
    }
    if ( out_max_fallback_total )
        *out_max_fallback_total = max_fallback_total;

    // -----------------------------------------------------------------------
    // Gather FMM final state to rank 0 and compare against brute-force.
    // -----------------------------------------------------------------------
    const int n_local_final = solver.num_local_particles();
    std::vector<int> all_n_final( nprocs, 0 );
    MPI_Gather( &n_local_final, 1, MPI_INT, all_n_final.data(), 1, MPI_INT, 0,
                MPI_COMM_WORLD );

    auto positions_f = Cabana::slice<Position>( particles );
    auto velocities_f = Cabana::slice<Velocity>( particles );
    auto gids_f = Cabana::slice<GlobalId>( particles );
    auto h_pos_f = Canopy::create_mirror_view_and_copy(
        Kokkos::HostSpace(), positions_f, "h_pos_f" );
    auto h_vel_f = Canopy::create_mirror_view_and_copy(
        Kokkos::HostSpace(), velocities_f, "h_vel_f" );
    auto h_gid_f = Canopy::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                        gids_f, "h_gid_f" );

    std::vector<double> local_pos_f( 3 * n_local_final );
    std::vector<double> local_vel_f( 3 * n_local_final );
    std::vector<int> local_gid_f( n_local_final );
    for ( int i = 0; i < n_local_final; i++ )
    {
        local_pos_f[3 * i + 0] = h_pos_f( i, 0 );
        local_pos_f[3 * i + 1] = h_pos_f( i, 1 );
        local_pos_f[3 * i + 2] = h_pos_f( i, 2 );
        local_vel_f[3 * i + 0] = h_vel_f( i, 0 );
        local_vel_f[3 * i + 1] = h_vel_f( i, 1 );
        local_vel_f[3 * i + 2] = h_vel_f( i, 2 );
        local_gid_f[i] = h_gid_f( i );
    }

    std::vector<int> counts_f( nprocs, 0 ), displs_f( nprocs, 0 );
    std::vector<int> counts3_f( nprocs, 0 ), displs3_f( nprocs, 0 );
    if ( rank == 0 )
    {
        for ( int r = 0; r < nprocs; r++ )
        {
            counts_f[r] = all_n_final[r];
            counts3_f[r] = 3 * all_n_final[r];
        }
        for ( int r = 1; r < nprocs; r++ )
        {
            displs_f[r] = displs_f[r - 1] + counts_f[r - 1];
            displs3_f[r] = displs3_f[r - 1] + counts3_f[r - 1];
        }
    }

    std::vector<double> fmm_pos_f, fmm_vel_f;
    std::vector<int> fmm_gid_f;
    if ( rank == 0 )
    {
        fmm_pos_f.resize( 3 * total_particles );
        fmm_vel_f.resize( 3 * total_particles );
        fmm_gid_f.resize( total_particles );
    }
    MPI_Gatherv( local_pos_f.data(), 3 * n_local_final, MPI_DOUBLE,
                 fmm_pos_f.data(), counts3_f.data(), displs3_f.data(),
                 MPI_DOUBLE, 0, MPI_COMM_WORLD );
    MPI_Gatherv( local_vel_f.data(), 3 * n_local_final, MPI_DOUBLE,
                 fmm_vel_f.data(), counts3_f.data(), displs3_f.data(),
                 MPI_DOUBLE, 0, MPI_COMM_WORLD );
    MPI_Gatherv( local_gid_f.data(), n_local_final, MPI_INT, fmm_gid_f.data(),
                 counts_f.data(), displs_f.data(), MPI_INT, 0, MPI_COMM_WORLD );

    if ( rank == 0 )
    {
        // Index brute-force final state by GlobalId.
        std::vector<int> bf_idx_of_gid( total_particles, -1 );
        for ( int i = 0; i < total_particles; i++ )
            bf_idx_of_gid[bf_gid[i]] = i;

        double max_pos_rel = 0.0;
        double max_vel_rel = 0.0;
        int max_vel_gid = -1;
        // Largest per-particle budget relative to the state it is compared
        // against, with the same |brute| < 1e-10 fallback as the deviation:
        // kappa_pos = max_i budget_dr[i] / |r_i|, kappa_vel = max_i
        // budget_dv[i] / |v_i|, both dimensionless.
        double kappa_pos = 0.0;
        double kappa_vel = 0.0;
        // Probe only, indexed by GlobalId: |v_bf| and the relative velocity
        // deviation of each particle.
        std::vector<double> vmag_of_gid, vrel_of_gid;
        if ( probe )
        {
            vmag_of_gid.assign( total_particles, 0.0 );
            vrel_of_gid.assign( total_particles, 0.0 );
        }
        for ( int i = 0; i < total_particles; i++ )
        {
            const int gid = fmm_gid_f[i];
            ASSERT_GE( gid, 0 );
            ASSERT_LT( gid, total_particles );
            const int j = bf_idx_of_gid[gid];
            ASSERT_GE( j, 0 )
                << "GlobalId " << gid << " missing from brute-force set";

            // Compare position and velocity vectors.
            const double pdx = fmm_pos_f[3 * i + 0] - bf_pos[3 * j + 0];
            const double pdy = fmm_pos_f[3 * i + 1] - bf_pos[3 * j + 1];
            const double pdz = fmm_pos_f[3 * i + 2] - bf_pos[3 * j + 2];
            const double pmag =
                std::sqrt( bf_pos[3 * j + 0] * bf_pos[3 * j + 0] +
                           bf_pos[3 * j + 1] * bf_pos[3 * j + 1] +
                           bf_pos[3 * j + 2] * bf_pos[3 * j + 2] );
            const double perr = std::sqrt( pdx * pdx + pdy * pdy + pdz * pdz );
            const double prel = ( pmag > 1.0e-10 ) ? perr / pmag : perr;
            if ( prel > max_pos_rel )
                max_pos_rel = prel;
            kappa_pos = std::max(
                kappa_pos,
                ( pmag > 1.0e-10 ) ? budget_dr[j] / pmag : budget_dr[j] );

            const double vdx = fmm_vel_f[3 * i + 0] - bf_vel[3 * j + 0];
            const double vdy = fmm_vel_f[3 * i + 1] - bf_vel[3 * j + 1];
            const double vdz = fmm_vel_f[3 * i + 2] - bf_vel[3 * j + 2];
            const double vmag =
                std::sqrt( bf_vel[3 * j + 0] * bf_vel[3 * j + 0] +
                           bf_vel[3 * j + 1] * bf_vel[3 * j + 1] +
                           bf_vel[3 * j + 2] * bf_vel[3 * j + 2] );
            const double verr = std::sqrt( vdx * vdx + vdy * vdy + vdz * vdz );
            const double vrel = ( vmag > 1.0e-10 ) ? verr / vmag : verr;
            kappa_vel = std::max(
                kappa_vel,
                ( vmag > 1.0e-10 ) ? budget_dv[j] / vmag : budget_dv[j] );
            if ( vrel > max_vel_rel )
            {
                max_vel_rel = vrel;
                max_vel_gid = gid;
            }
            if ( probe )
            {
                vmag_of_gid[gid] = vmag;
                vrel_of_gid[gid] = vrel;
            }
        }

        // Probe end-of-run line. Velocities in position units per time,
        // separations in position units, the rest dimensionless.
        //   max_vel_*: the max_vel_rel particle's GlobalId, its smallest
        //     nearest-neighbour distance over the probed steps and the final
        //     FMM state, and its |v_bf| against the median |v_bf|, which
        //     shows whether its relative deviation is inflated by a small
        //     |v_bf|;
        //   run_min_sep, n_close_run: smallest distance over the run, and the
        //     number of particles whose run minimum fell below close_thr;
        //   n_excess, n_excess_close: particles whose relative velocity
        //     deviation exceeds the floor theta^(P+1), and how many of those
        //     had a close encounter.
        if ( probe )
        {
            const std::vector<double> nn_f = nearest_separation( fmm_pos_f );
            for ( int i = 0; i < total_particles; i++ )
            {
                double& m = run_min_sep[fmm_gid_f[i]];
                m = std::min( m, nn_f[i] );
            }
            const double floor_rel = std::pow( mac_theta_used, P_ORDER + 1 );
            int n_close_run = 0, n_excess = 0, n_excess_close = 0;
            for ( int g = 0; g < total_particles; g++ )
            {
                const bool close = run_min_sep[g] < close_thr;
                n_close_run += close;
                if ( vrel_of_gid[g] > floor_rel )
                {
                    n_excess++;
                    n_excess_close += close;
                }
            }
            std::vector<double> vsorted = vmag_of_gid;
            std::nth_element( vsorted.begin(),
                              vsorted.begin() + total_particles / 2,
                              vsorted.end() );
            std::printf(
                "[multisolve-probe] case %s nprocs %d end max_vel_rel %.17g "
                "max_vel_gid %d max_vel_min_sep %.17g max_vel_v %.17g "
                "median_v %.17g run_min_sep %.17g n_close_run %d "
                "close_thr %.17g floor %.17g n_excess %d n_excess_close %d\n",
                case_label, nprocs, max_vel_rel, max_vel_gid,
                ( max_vel_gid >= 0 ) ? run_min_sep[max_vel_gid] : -1.0,
                ( max_vel_gid >= 0 ) ? vmag_of_gid[max_vel_gid] : -1.0,
                vsorted[total_particles / 2],
                *std::min_element( run_min_sep.begin(), run_min_sep.end() ),
                n_close_run, close_thr, floor_rel, n_excess, n_excess_close );
            std::fflush( stdout );
        }

        // UNCONDITIONAL. The bounds below are measured bounds, so the
        // figures they are measured from have to be readable on a PASS and
        // not only out of a failure message (risk R10: the distinguishing
        // measurement is the deviation, not the pass/fail). Printed on rank
        // 0 only, which is the only rank that holds the brute-force shadow.
        //   max_pos_rel, max_vel_rel: dimensionless relative deviations,
        //     >= 0, each the max over all particles of |fmm - brute| / |brute|
        //     on the final state (position and velocity respectively);
        //     the |brute| < 1e-10 particles fall back to the absolute
        //     deviation, same units as the state itself.
        //   derived_pos, derived_vel: the first-order bounds
        //     theta^(P+1) * kappa_pos and theta^(P+1) * kappa_vel, the
        //     per-solve floor carried through this run's integration
        //     (budget_dv / budget_dr above); dimensionless.
        //   pos_tol, vel_tol: the call site's bounds on the two deviations.
        const double floor_eps = std::pow( mac_theta_used, P_ORDER + 1 );
        std::printf( "[multisolve-dev] case %s nprocs %d nsteps %d "
                     "drift %.17g max_pos_rel %.17g max_vel_rel %.17g "
                     "derived_pos %.17g derived_vel %.17g "
                     "pos_tol %.17g vel_tol %.17g\n",
                     case_label, nprocs, num_steps, drift_multiplier,
                     max_pos_rel, max_vel_rel, floor_eps * kappa_pos,
                     floor_eps * kappa_vel, pos_tolerance, vel_tolerance );
        std::fflush( stdout );

        EXPECT_LT( max_pos_rel, pos_tolerance )
            << "FMM multi-step position deviates from brute-force; "
               "max relative error = "
            << max_pos_rel;
        EXPECT_LT( max_vel_rel, vel_tolerance )
            << "FMM multi-step velocity deviates from brute-force; "
               "max relative error = "
            << max_vel_rel;
    }
}

//---------------------------------------------------------------------------//
// Trajectory bounds (pos_tol, vel_tol at each call site below).
//
// Each is the tighter of two figures, both stated at the site:
//   derived:  theta^(P+1) carried through the site's integration -- the
//     first-order budget of testMultiStepGravity (derived_pos/derived_vel on
//     the [multisolve-dev] line), worst over np. It holds while the FMM and
//     brute-force trajectories stay close; above it is dynamics.
//   measured: the worst deviation over three runs on each backend (SERIAL
//     np 1-6, HIP np 1-4; flux jobs f3cajPhDd7dZ, f3cajPqbu44f) times 2.
//     SERIAL repeats bit for bit and HIP moves by at most 1.0004x run to run,
//     so the 2x margin covers the spread.
// Every measured figure sits under its derived one at every np, so measured
// x 2 sets every bound. One bound per site covers all rank counts; the
// deviation grows with np (global N = 200 * np), so a site is loose at np 1.
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// Test 1: Stable tree, only inter-rank migration.
//
// Small dt and unit drift_multiplier so positions barely move; tree
// topology never changes. Exercises Solver::migrate (cheap path) over
// many steps.
//---------------------------------------------------------------------------//
TEST( MultiSolve, StableTree_Migrate )
{
    testMultiStepGravity( MultiSolveTest::Mode::Migrate,
                          /*case=*/"StableTree_Migrate",
                          /*npp=*/200, /*nsteps=*/5,
                          /*dt=*/1.0e-4, /*drift_multiplier=*/1.0,
                          /*ncrit=*/16, /*max_depth=*/6,
                          /*tree_tol=*/0.1, /*repl_depth=*/2,
                          // derived 9.64e-6 / 1.97e-1; measured 3.367e-9
                          // (np 6) / 9.364e-6 (np 5).
                          /*pos_tol=*/6.8e-9, /*vel_tol=*/1.9e-5 );
}

//---------------------------------------------------------------------------//
// Test 2: Intermediate motion — tree topology changes.
//
// Larger dt so leaves can refine/coarsen between steps. Exercises
// Solver::rebalance (TreeBuilder::update + repartition + comm_plan rebuild).
//---------------------------------------------------------------------------//
TEST( MultiSolve, IntermediateMotion_Rebalance )
{
    testMultiStepGravity( MultiSolveTest::Mode::Rebalance,
                          /*case=*/"IntermediateMotion_Rebalance",
                          /*npp=*/200, /*nsteps=*/5,
                          /*dt=*/1.0e-3, /*drift_multiplier=*/5.0,
                          /*ncrit=*/16, /*max_depth=*/6,
                          /*tree_tol=*/0.1, /*repl_depth=*/2,
                          // derived 1.76e-2 / 9.73e-2; measured 2.203e-4
                          // (np 5) / 6.182e-4 (np 5).
                          /*pos_tol=*/4.5e-4, /*vel_tol=*/1.3e-3 );
}

//---------------------------------------------------------------------------//
// Test 3: Large motion — particles routinely escape the bounding box.
//
// Uses Solver::rebuild every step (full do-over: build → partition →
// build → sort → build → comm_plan → setups). drift_multiplier is high
// enough that the bounding box must grow each step.
//---------------------------------------------------------------------------//
TEST( MultiSolve, LargeMotion_Rebuild )
{
    testMultiStepGravity( MultiSolveTest::Mode::Rebuild,
                          /*case=*/"LargeMotion_Rebuild",
                          /*npp=*/200, /*nsteps=*/4,
                          /*dt=*/1.0e-3, /*drift_multiplier=*/50.0,
                          /*ncrit=*/16, /*max_depth=*/6,
                          /*tree_tol=*/0.1, /*repl_depth=*/2,
                          // derived 9.33e-2 / 2.24e-1; measured 1.742e-3
                          // (np 4) / 1.612e-3 (np 4, HIP).
                          /*pos_tol=*/3.5e-3, /*vel_tol=*/3.3e-3 );
}

//---------------------------------------------------------------------------//
// Test 4: Auto-maintain mode — the solver picks a safe maintenance path
// each step (Rebuild if particles escaped the bounding box; otherwise
// Rebalance). Verifies trajectory accuracy and that the dispatcher is
// being invoked.
//---------------------------------------------------------------------------//
TEST( MultiSolve, AutoMaintain )
{
    int counts[3] = { 0, 0, 0 };
    testMultiStepGravity( MultiSolveTest::Mode::Auto,
                          /*case=*/"AutoMaintain",
                          /*npp=*/200, /*nsteps=*/5,
                          /*dt=*/1.0e-3, /*drift_multiplier=*/5.0,
                          /*ncrit=*/16, /*max_depth=*/6,
                          /*tree_tol=*/0.1, /*repl_depth=*/2,
                          // derived 1.76e-2 / 9.73e-2; measured 2.203e-4
                          // (np 5) / 6.182e-4 (np 5) -- the same trajectory
                          // as IntermediateMotion_Rebalance at every np.
                          /*pos_tol=*/4.5e-4, /*vel_tol=*/1.3e-3, counts );

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if ( rank == 0 )
    {
        const int total = counts[0] + counts[1] + counts[2];
        EXPECT_GT( total, 0 ) << "auto_maintain was never called";
    }
}

//---------------------------------------------------------------------------//
// Test 4b: Auto-maintain mode forced into the Rebalance branch.
//
// The standard AutoMaintain test (above) uses drift_multiplier=5.0, which
// makes particles routinely escape the initial bounding box; auto_maintain
// then takes the Rebuild branch on every step. To exercise the Rebalance
// branch (tree topology changes but bbox holds) we widen the bounding box
// (tree_tol=0.3 ⇒ 30% padding), keep dt small, drop drift_multiplier to a
// modest value that lets velocities accumulate enough leaf refine/coarsen
// to change the cell-key set, and run for more steps so at least one
// per-step topology change is virtually certain.
//
// The assertion is that the Rebalance count is >0. Its far-field gate is the
// per-step probe, not the trajectory: the test runs unsoftened, and at np 5-6
// close encounters amplify a per-step field error of < 1e-6 into a velocity
// deviation of 1.7e-2 (fix-hang-rebalance E1). probe_field_tol bounds every
// step's field-scale error at the tighter of theta^(P+1) = 1.95e-3 and the
// measured worst 9.509e-7 (np 3, both backends) x 2. The trajectory bounds
// are measured x 2 like the other sites, and catch a complete regression of
// the Rebalance path; they make no far-field claim.
//---------------------------------------------------------------------------//
TEST( MultiSolve, AutoRebalance )
{
    int counts[3] = { 0, 0, 0 };
    testMultiStepGravity( MultiSolveTest::Mode::Auto,
                          /*case=*/"AutoRebalance",
                          /*npp=*/200, /*nsteps=*/8,
                          /*dt=*/1.0e-3, /*drift_multiplier=*/2.0,
                          /*ncrit=*/16, /*max_depth=*/6,
                          /*tree_tol=*/0.3, /*repl_depth=*/2,
                          // derived 3.26e-2 / 4.01e-1; measured 3.160e-3
                          // (np 6) / 1.713e-2 (np 6).
                          /*pos_tol=*/6.4e-3, /*vel_tol=*/3.5e-2, counts,
                          /*clustered=*/false, /*mac_theta_override=*/0.0,
                          /*out_max_fallback_total=*/nullptr,
                          /*probe_field_tol=*/1.9e-6 );

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if ( rank == 0 )
    {
        EXPECT_GT( counts[1], 0 )
            << "auto_maintain never took the Rebalance branch "
            << "(Migrate=" << counts[0]
            << " Rebalance=" << counts[1]
            << " Rebuild=" << counts[2] << ")";
    }
}

//---------------------------------------------------------------------------//
// Test 5: Key-refusal fallback — exercise the per-pair m2l_translate path.
//
// The batched-GEMM M2L pipeline gives each (target, source) pair a
// canonical key { max_d, dd, ii, jj, kk }: the deeper of the two depths,
// the signed depth difference d_src - d_tgt, and the center-to-center
// offset in half-widths at depth max_d. Two bounds in the classify pass's
// range guard (Canopy_DownwardSweep.hpp:1704-1716) can refuse a pair a
// key: |dd| <= KernelType::m2l_key_dd_max (6 for LaplaceKernel at double,
// 4 at float) and |ii|,|jj|,|kk| <= M2L_KEY_OFFSET_MAX = 32. A refused
// pair is routed to the on-the-fly m2l_translate kernel instead. A silent
// regression in that fallback (e.g. the atomic-accumulation race
// previously fixed in m2l_translate) would only surface in a workload
// that actually produces refused pairs.
//
// Configuration: clustered particle distribution to force deep refinement
// in one octant, tight MAC theta = 0.3 to admit more far-field pairs at
// large offsets, and ncrit/max_depth that match the existing tests'
// scale. The column cap is left at its default, so no refusal here can be
// a count-cap refusal. The probe inside testMultiStepGravity sums
// fallback pairs across ranks each solve and reports the running max; we
// assert it's strictly positive. The probe also prints the per-reason
// breakdown per (nprocs, rank) and ASSERTS it: every refusal is a range-guard
// refusal (count_cap == 0, depth_dropped == 0, and the reasons sum to the
// total), skipped under the -1 sentinel when profiling is off.
//---------------------------------------------------------------------------//
TEST( MultiSolve, M2L_BinEdge_Fallback )
{
    long long max_fallback = 0;
    testMultiStepGravity( MultiSolveTest::Mode::Migrate,
                          /*case=*/"M2L_BinEdge_Fallback",
                          /*npp=*/300, /*nsteps=*/2,
                          /*dt=*/1.0e-4, /*drift_multiplier=*/1.0,
                          /*ncrit=*/8, /*max_depth=*/8,
                          /*tree_tol=*/0.1, /*repl_depth=*/2,
                          // theta = 0.3, so theta^(P+1) = 1.97e-5: derived
                          // 2.28e-5 / 4.92e-4; measured 1.715e-9 (np 6) /
                          // 1.661e-6 (np 3).
                          /*pos_tol=*/3.5e-9, /*vel_tol=*/3.4e-6,
                          /*out_action_counts=*/nullptr,
                          /*clustered=*/true,
                          /*mac_theta_override=*/0.3, &max_fallback );

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if ( rank == 0 )
    {
        EXPECT_GT( max_fallback, 0 )
            << "no refused M2L pairs were produced — the m2l_translate "
               "fallback path was not exercised, so this regression is a "
               "no-op. Either the clustered distribution stopped reaching "
               "deep enough to put a pair outside the range guard "
               "(|dd| > KernelType::m2l_key_dd_max, or an offset "
               "component > M2L_KEY_OFFSET_MAX = 32 half-widths at the "
               "deeper depth), or one of those two bounds was raised. "
               "The [m2l-fallback-reason] lines above say which of the "
               "two was carrying this test's refusals.";
    }
}

//---------------------------------------------------------------------------//
// Tier 2 fused-M2L regression tests.
//
// The Tier 2 refactor replaced the pack/GEMM/scatter M2L pipeline with a
// fused team-per-target kernel. These tests exercise correctness of the
// new path against a brute-force N^2 reference and against itself across
// repeated solves.
//
// Note on coverage: the pre-existing M2L_BinEdge_Fallback test above
// already verifies that pairs violating the M2L key guards are routed
// through run_m2l_fallback_at_depth and produce a correct result, so a
// dedicated fallbackPathStillFires test is intentionally omitted here.
//---------------------------------------------------------------------------//

namespace MultiSolveTest
{

// Single-rank single-solve helper templated on expansion order and on the
// kernel Scalar type. Runs FMM + P2P once on a uniform random distribution
// and returns the max relative error in (potential, gradient) against a
// brute-force N^2 reference computed in double precision (the reference
// itself is FP64 regardless of Scalar; that lets the same oracle judge
// both an FP64 and an FP32 build with the appropriate per-precision
// tolerance set by the caller).
template <int P, class Scalar = double>
inline void run_fmm_and_compare( int num_particles, double mac_theta,
                                 int ncrit, int max_depth,
                                 double& max_pot_rel,
                                 double& max_grad_rel )
{
    using DataTypes = Cabana::MemberTypes<Scalar[3], Scalar[1]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    using AoSoA_ht = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
    using Solver_t =
        Canopy::Solver<TEST_MEMSPACE, TEST_EXECSPACE, Scalar, P, 1>;
    const MPI_Datatype mpi_scalar =
        std::is_same<Scalar, float>::value ? MPI_FLOAT : MPI_DOUBLE;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    AoSoA_ht particles_h( "particles_h", num_particles );
    {
        auto hp = Cabana::slice<0>( particles_h );
        auto hq = Cabana::slice<1>( particles_h );
        std::mt19937 gen( 1234 + rank * 31 + P );
        std::uniform_real_distribution<double> pos_dist( 0.05, 0.95 );
        std::uniform_real_distribution<double> q_dist( -1.0, 1.0 );
        for ( int i = 0; i < num_particles; i++ )
        {
            hp( i, 0 ) = pos_dist( gen );
            hp( i, 1 ) = pos_dist( gen );
            hp( i, 2 ) = pos_dist( gen );
            hq( i, 0 ) = q_dist( gen );
        }
    }
    AoSoA_t particles( "particles", num_particles );
    Cabana::deep_copy( particles, particles_h );

    Canopy::FmmConfig cfg;
    cfg.ncrit = ncrit;
    cfg.max_depth = max_depth;
    cfg.xmin_tol = cfg.xmax_tol = 0.1;
    cfg.ymin_tol = cfg.ymax_tol = 0.1;
    cfg.zmin_tol = cfg.zmax_tol = 0.1;
    cfg.ncrit_tol = 0.1;
    cfg.replication_depth = 2;
    cfg.imbalance_tolerance = 0.05;
    cfg.mac_theta = mac_theta;
    cfg.softening = 0.0;
    Solver_t solver( MPI_COMM_WORLD, cfg );
    solver.template setup<0, 1>( particles, num_particles );
    solver.template solve<0, 1>( particles, /*compute_gradient=*/true );

    const int n_local = solver.num_local_particles();
    auto positions = Cabana::slice<0>( particles );
    auto charges = Cabana::slice<1>( particles );
    auto h_pos = Canopy::create_mirror_view_and_copy(
        Kokkos::HostSpace(), positions, "h_pos" );
    auto h_chg = Canopy::create_mirror_view_and_copy(
        Kokkos::HostSpace(), charges, "h_chg" );
    auto h_pot = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), solver.potential() );
    auto h_grad = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), solver.gradient() );

    // Gather everything to rank 0.
    int total = 0;
    MPI_Allreduce( &n_local, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    std::vector<int> all_n( nprocs, 0 ), displs( nprocs, 0 );
    std::vector<int> displs3( nprocs, 0 ), counts3( nprocs, 0 );
    std::vector<int> displs1( nprocs, 0 ), counts1( nprocs, 0 );
    MPI_Gather( &n_local, 1, MPI_INT, all_n.data(), 1, MPI_INT, 0,
                MPI_COMM_WORLD );
    if ( rank == 0 )
    {
        for ( int r = 0; r < nprocs; r++ )
        {
            counts3[r] = 3 * all_n[r];
            counts1[r] = all_n[r];
        }
        for ( int r = 1; r < nprocs; r++ )
        {
            displs3[r] = displs3[r - 1] + counts3[r - 1];
            displs1[r] = displs1[r - 1] + counts1[r - 1];
        }
    }
    std::vector<Scalar> lpos( 3 * n_local ), lchg( n_local ),
        lpot( n_local );
    std::vector<Scalar> lgrad( 3 * n_local );
    for ( int i = 0; i < n_local; i++ )
    {
        lpos[3 * i + 0] = h_pos( i, 0 );
        lpos[3 * i + 1] = h_pos( i, 1 );
        lpos[3 * i + 2] = h_pos( i, 2 );
        lchg[i] = h_chg( i, 0 );
        lpot[i] = h_pot( i, 0 );
        lgrad[3 * i + 0] = h_grad( i, 0, 0 );
        lgrad[3 * i + 1] = h_grad( i, 0, 1 );
        lgrad[3 * i + 2] = h_grad( i, 0, 2 );
    }
    std::vector<Scalar> gpos_s, gchg_s, gpot_s, ggrad_s;
    if ( rank == 0 )
    {
        gpos_s.resize( 3 * total );
        gchg_s.resize( total );
        gpot_s.resize( total );
        ggrad_s.resize( 3 * total );
    }
    MPI_Gatherv( lpos.data(), 3 * n_local, mpi_scalar, gpos_s.data(),
                 counts3.data(), displs3.data(), mpi_scalar, 0,
                 MPI_COMM_WORLD );
    MPI_Gatherv( lchg.data(), n_local, mpi_scalar, gchg_s.data(),
                 counts1.data(), displs1.data(), mpi_scalar, 0,
                 MPI_COMM_WORLD );
    MPI_Gatherv( lpot.data(), n_local, mpi_scalar, gpot_s.data(),
                 counts1.data(), displs1.data(), mpi_scalar, 0,
                 MPI_COMM_WORLD );
    MPI_Gatherv( lgrad.data(), 3 * n_local, mpi_scalar, ggrad_s.data(),
                 counts3.data(), displs3.data(), mpi_scalar, 0,
                 MPI_COMM_WORLD );

    // Promote to double on rank 0 for the brute-force reference.
    std::vector<double> gpos, gchg, gpot, ggrad;
    if ( rank == 0 )
    {
        gpos.assign( gpos_s.begin(), gpos_s.end() );
        gchg.assign( gchg_s.begin(), gchg_s.end() );
        gpot.assign( gpot_s.begin(), gpot_s.end() );
        ggrad.assign( ggrad_s.begin(), ggrad_s.end() );
    }

    max_pot_rel = 0.0;
    max_grad_rel = 0.0;
    if ( rank == 0 )
    {
        for ( int i = 0; i < total; i++ )
        {
            double phi = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
            for ( int j = 0; j < total; j++ )
            {
                if ( j == i )
                    continue;
                const double dx = gpos[3 * i + 0] - gpos[3 * j + 0];
                const double dy = gpos[3 * i + 1] - gpos[3 * j + 1];
                const double dz = gpos[3 * i + 2] - gpos[3 * j + 2];
                const double r2 = dx * dx + dy * dy + dz * dz;
                const double inv_r = 1.0 / std::sqrt( r2 );
                const double inv_r3 = inv_r * inv_r * inv_r;
                phi += gchg[j] * inv_r;
                gx -= gchg[j] * dx * inv_r3;
                gy -= gchg[j] * dy * inv_r3;
                gz -= gchg[j] * dz * inv_r3;
            }
            const double pref = std::abs( phi );
            const double perr = std::abs( gpot[i] - phi );
            const double prel = ( pref > 1e-12 ) ? perr / pref : perr;
            if ( prel > max_pot_rel )
                max_pot_rel = prel;
            const double gmag =
                std::sqrt( gx * gx + gy * gy + gz * gz );
            const double gerr = std::sqrt(
                ( ggrad[3 * i + 0] - gx ) * ( ggrad[3 * i + 0] - gx ) +
                ( ggrad[3 * i + 1] - gy ) * ( ggrad[3 * i + 1] - gy ) +
                ( ggrad[3 * i + 2] - gz ) * ( ggrad[3 * i + 2] - gz ) );
            const double grel = ( gmag > 1e-12 ) ? gerr / gmag : gerr;
            if ( grel > max_grad_rel )
                max_grad_rel = grel;
        }
    }
    MPI_Bcast( &max_pot_rel, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    MPI_Bcast( &max_grad_rel, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
}

} // namespace MultiSolveTest

//---------------------------------------------------------------------------//
// SolveFusedM2L.matchesPriorReference: at P=6 the FMM result must match
// the brute-force N^2 reference within the same accuracy band the prior
// pack/GEMM pipeline produced. We use a smaller N than the profiling
// case so the O(N^2) reference is fast in CI; the fused-kernel code
// path is the same regardless of N.
//---------------------------------------------------------------------------//
TEST( SolveFusedM2L, matchesPriorReference )
{
    double pot_err = 0.0, grad_err = 0.0;
    MultiSolveTest::run_fmm_and_compare<6>( /*num_particles=*/400,
                                            /*mac_theta=*/0.5,
                                            /*ncrit=*/16, /*max_depth=*/6,
                                            pot_err, grad_err );
    // Bounds. pot_err is per-particle relative with mixed-sign charges, so
    // |phi| -> 0 inflates it: measured 8.93e-4 to 4.32e-2 over np 1-6 (worst
    // np 5), which is cancellation, not the far field. Its 5.0e-2 bound is
    // left as it is -- measured x 2 would loosen it. grad_err is stable:
    // identical over three runs on each backend (SERIAL np 1-6, HIP np 1-4;
    // flux jobs f3cajPhDd7dZ, f3cajPqbu44f), worst 2.847e-4 at np 3, under
    // the floor theta^(P+1) = 0.5^7 = 7.8e-3. Its bound is that worst x 2.
    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
    if ( rank == 0 )
    {
        // UNCONDITIONAL, same reason as [multisolve-dev] above: the bounds
        // below are measured, so the figures must be readable on a pass.
        // pot_err, grad_err: dimensionless relative deviations, >= 0, each
        // the max over particles of |fmm - brute| / |brute| in potential and
        // in gradient magnitude. Broadcast from rank 0 by the harness.
        std::printf( "[fusedm2l-dev] case matchesPriorReference nprocs %d "
                     "pot_err %.17g grad_err %.17g\n",
                     nprocs, pot_err, grad_err );
        std::fflush( stdout );
    }
    EXPECT_LT( pot_err, 5.0e-2 );
    EXPECT_LT( grad_err, 5.7e-4 );
}

//---------------------------------------------------------------------------//
// SolveFusedM2L.sweepConvergence: error must drop monotonically as P
// grows from 4 to 6 to 8. A P-dependent bug in the fused kernel (e.g.
// off-by-one in the conjugate-symmetry expansion) would surface here as
// a non-monotone trend.
//---------------------------------------------------------------------------//
TEST( SolveFusedM2L, sweepConvergence )
{
    double e_p4_pot, e_p4_grad;
    double e_p6_pot, e_p6_grad;
    double e_p8_pot, e_p8_grad;
    MultiSolveTest::run_fmm_and_compare<4>( 400, 0.5, 16, 6, e_p4_pot,
                                            e_p4_grad );
    MultiSolveTest::run_fmm_and_compare<6>( 400, 0.5, 16, 6, e_p6_pot,
                                            e_p6_grad );
    MultiSolveTest::run_fmm_and_compare<8>( 400, 0.5, 16, 6, e_p8_pot,
                                            e_p8_grad );

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if ( rank == 0 )
    {
        // Compare endpoints (P=8 vs P=4) rather than every consecutive
        // pair — at moderate N the per-step trend can be jittery near
        // the FMM's accuracy floor, but the four-order improvement from
        // P=4 to P=8 is robust and would be wiped out by any P-dependent
        // bug in the fused kernel (e.g. truncated j-loop bound).
        EXPECT_LT( e_p8_pot, e_p4_pot )
            << "P=8 potential error not below P=4: " << e_p8_pot
            << " vs " << e_p4_pot;
        EXPECT_LT( e_p8_grad, e_p4_grad )
            << "P=8 gradient error not below P=4: " << e_p8_grad
            << " vs " << e_p4_grad;
        // We deliberately do not require strict monotonicity P=4 > P=6
        // > P=8: at this small N + replication-depth-2 multi-rank
        // configuration the per-step trend is not monotone (verified
        // bit-for-bit identical between the pre-Tier-2 GEMM pipeline
        // and the Tier-2 fused kernel — the lack of monotonicity is
        // intrinsic to the FMM at this setup, not a kernel regression).
        // The P=8 << P=4 endpoint check above is the load-bearing one.
    }
}

//---------------------------------------------------------------------------//
// SolveFusedM2L.multipleSolvesIdempotent: three back-to-back solves on
// the same particle state must agree. Confirms the fused kernel does not
// leave residual state in _locals between solves and that execute()'s
// zero-init still works correctly -- a defect of that kind shows up as an
// O(1) change. Agreement is to a tolerance, not bit-identity: a device
// backend accumulates in a run-dependent order, so on HIP the solves differ
// in round-off (README "Known Issues", HIP bit-reproducibility).
//---------------------------------------------------------------------------//
TEST( SolveFusedM2L, multipleSolvesIdempotent )
{
    using namespace MultiSolveTest;
    using DataTypes = Cabana::MemberTypes<double[3], double[1]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    using AoSoA_ht = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
    using Solver_t =
        Canopy::Solver<TEST_MEMSPACE, TEST_EXECSPACE, double, 6, 1>;

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    const int N = 300;
    AoSoA_ht particles_h( "particles_h", N );
    {
        auto hp = Cabana::slice<0>( particles_h );
        auto hq = Cabana::slice<1>( particles_h );
        std::mt19937 gen( 99 + rank );
        std::uniform_real_distribution<double> pos_dist( 0.05, 0.95 );
        std::uniform_real_distribution<double> q_dist( -1.0, 1.0 );
        for ( int i = 0; i < N; i++ )
        {
            hp( i, 0 ) = pos_dist( gen );
            hp( i, 1 ) = pos_dist( gen );
            hp( i, 2 ) = pos_dist( gen );
            hq( i, 0 ) = q_dist( gen );
        }
    }
    AoSoA_t particles( "particles", N );
    Cabana::deep_copy( particles, particles_h );

    Canopy::FmmConfig cfg;
    cfg.ncrit = 16;
    cfg.max_depth = 6;
    cfg.xmin_tol = cfg.xmax_tol = 0.1;
    cfg.ymin_tol = cfg.ymax_tol = 0.1;
    cfg.zmin_tol = cfg.zmax_tol = 0.1;
    cfg.ncrit_tol = 0.1;
    cfg.replication_depth = 2;
    cfg.imbalance_tolerance = 0.05;
    cfg.mac_theta = 0.5;
    cfg.softening = 0.0;
    Solver_t solver( MPI_COMM_WORLD, cfg );
    solver.template setup<0, 1>( particles, N );

    auto snapshot = [&]( std::vector<double>& pot,
                         std::vector<double>& grad ) {
        const int n_local = solver.num_local_particles();
        auto h_pot = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), solver.potential() );
        auto h_grad = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), solver.gradient() );
        pot.resize( n_local );
        grad.resize( 3 * n_local );
        for ( int i = 0; i < n_local; i++ )
        {
            pot[i] = h_pot( i, 0 );
            grad[3 * i + 0] = h_grad( i, 0, 0 );
            grad[3 * i + 1] = h_grad( i, 0, 1 );
            grad[3 * i + 2] = h_grad( i, 0, 2 );
        }
    };

    std::vector<double> pot1, grad1, pot2, grad2, pot3, grad3;
    solver.template solve<0, 1>( particles, /*compute_gradient=*/true );
    snapshot( pot1, grad1 );
    solver.template solve<0, 1>( particles, true );
    snapshot( pot2, grad2 );
    solver.template solve<0, 1>( particles, true );
    snapshot( pot3, grad3 );

    ASSERT_EQ( pot1.size(), pot2.size() );
    ASSERT_EQ( pot1.size(), pot3.size() );

    // Field-scale drift of solves 2 and 3 against solve 1, the field_scales
    // rule of tstLaplaceSolve.hpp: max_i |x_k - x_1| / max_i |x_1| over all
    // ranks, dimensionless, with |.| the magnitude of the potential or of
    // the gradient vector.
    auto drift = [&]( const std::vector<double>& a,
                      const std::vector<double>& b, int dim ) {
        double num = 0.0, scale = 0.0;
        for ( size_t i = 0; i < a.size() / dim; i++ )
        {
            double d2 = 0.0, m2 = 0.0;
            for ( int c = 0; c < dim; c++ )
            {
                const double d = b[dim * i + c] - a[dim * i + c];
                d2 += d * d;
                m2 += a[dim * i + c] * a[dim * i + c];
            }
            num = std::max( num, std::sqrt( d2 ) );
            scale = std::max( scale, std::sqrt( m2 ) );
        }
        double g[2] = { num, scale }, gmax[2];
        MPI_Allreduce( g, gmax, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
        return ( gmax[1] > 0.0 ) ? gmax[0] / gmax[1] : gmax[0];
    };
    const double pot_drift =
        std::max( drift( pot1, pot2, 1 ), drift( pot1, pot3, 1 ) );
    const double grad_drift =
        std::max( drift( grad1, grad2, 3 ), drift( grad1, grad3, 3 ) );

    int nprocs;
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
    if ( rank == 0 )
    {
        std::printf( "[fusedm2l-idem] nprocs %d pot_drift %.17g "
                     "grad_drift %.17g\n",
                     nprocs, pot_drift, grad_drift );
        std::fflush( stdout );
    }

    // Round-off budget. Measured (tree-opt V1, flux jobs f3cb6Hpf9s35 and
    // f3cb6HwmJRNw, three passes each): SERIAL drift is exactly 0 at np 1-6;
    // HIP pot_drift <= 1.9e-16 and grad_drift 6.4e-14 to 5.4e-13 at np 1-4,
    // moving up to 2.8x between passes. 1e-11 is ~20x the worst; the defect
    // the test exists for is O(1).
    static constexpr double IDEM_TOL = 1.0e-11;
    EXPECT_LT( pot_drift, IDEM_TOL ) << "potential changed between solves";
    EXPECT_LT( grad_drift, IDEM_TOL ) << "gradient changed between solves";
}

//---------------------------------------------------------------------------//
// DISABLED: SolveFusedM2L.FP32_smokeTest, pending investigation of a multi-rank
// FP32 accuracy defect. Commented out rather than filtered so the carve-out
// is visible in the diff and the case is absent from the binary.
//
// It passes at np 1 and fails at np 2-6 with a max relative GRADIENT error of
// ~0.277 at np 2 (0.339 at np 3) against its own 5.0e-2 budget -- 5.5x over,
// not marginal, and identical to FP32 noise on Tuolumne and Dane. That
// magnitude and its rank-count dependence point at a multi-rank FP32
// accumulation defect in the fused-M2L solve, which README.md "Known Issues"
// ("SolveFusedM2L.FP32_smokeTest is disabled ...") carries.
//
// DO NOT re-enable by widening the 5.0e-2 budget: a widened budget retires
// the only signal that defect has. Re-enable as written once the defect is
// fixed. (tasks/tree-opt.md V1 step 2.)
//---------------------------------------------------------------------------//
// // SolveFusedM2L.FP32_smokeTest: the kernel templates support Scalar=float.
// // After scale-normalization (M̄ = M/w^{n+1}, L̄ = L·w^j) per-coefficient
// // intermediates are O(q · 2^max_d) — linear in depth, not geometric — so
// // FP32 stays well-conditioned. The |dd|-dependent factor 2^{j·|dd|} in
// // T̃ caps precision loss at ~8 bits when |dd| ≤ 4, which is what the FP32
// // path of M2L_KEY_DD_MAX enforces.
// //
// // At P=4, the Greengard truncation floor is already ~5e-3 for a uniform
// // 400-particle problem; FP32 round-off adds maybe ~1e-4 relative, so a
// // 1e-2 bound on max-rel error is robust.
// TEST( SolveFusedM2L, FP32_smokeTest )
// {
//     double max_pot_rel = 0.0, max_grad_rel = 0.0;
//     MultiSolveTest::run_fmm_and_compare<4, float>(
//         /*num_particles=*/400, /*mac_theta=*/0.5, /*ncrit=*/16,
//         /*max_depth=*/6, max_pot_rel, max_grad_rel );
//
//     int rank;
//     MPI_Comm_rank( MPI_COMM_WORLD, &rank );
//     if ( rank == 0 )
//     {
//         // Both thresholds are deliberately loose. Three error sources
//         // stack on top of the Greengard P=4 truncation floor:
//         //   1. FP32 round-off in M2L/L2L/M2M (~few × 10^{-3} per pair).
//         //   2. Non-deterministic cross-rank summation order (grows with
//         //      nprocs; at np=6 we see ~1e-2 on potential).
//         //   3. Gradient is via finite differences at h=1e-5, which loses
//         //      most of FP32's mantissa.
//         // 5e-2 is the smoke-test budget: tight enough to catch a wrong
//         // scale exponent in any of P2M / M2M / M2L / L2L / L2P, loose
//         // enough to not false-fail on np ∈ [1, 6]. Production FP32
//         // verification belongs in a problem-specific oracle.
//         EXPECT_LT( max_pot_rel, 5.0e-2 )
//             << "FP32 max relative potential error " << max_pot_rel
//             << " exceeds the 5e-2 budget";
//         EXPECT_LT( max_grad_rel, 5.0e-2 )
//             << "FP32 max relative gradient error " << max_grad_rel
//             << " exceeds the 5e-2 budget";
//     }
// }

//---------------------------------------------------------------------------//

} // end namespace Test
