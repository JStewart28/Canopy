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

// ---------------------------------------------------------------------------
// TEMPORARY DIAGNOSTIC (tasks/rebalance_error.md).
//
// Separates the two candidate explanations for why the MultiSolve rebalance
// tests need a looser fmm_tol than the stable-tree test:
//
//   (A) the FMM force itself is less accurate when the tree is rebuilt, or
//   (B) the force error is the same and the *trajectory* metric amplifies it,
//       because the rebalance tests also use a 10x larger dt and a 5x/50x
//       larger drift multiplier.
//
// Per step it reports BOTH:
//   force_err  - max relative error of the FMM gradient against a brute-force
//                N^2 reference evaluated at the FMM's OWN current positions.
//                Independent of trajectory divergence: a pure measure of FMM
//                accuracy for the current tree.
//   pos_err    - max relative position error of the FMM trajectory against an
//                independent brute-force shadow trajectory (this is what
//                tstMultiSolve.hpp asserts on).
// ---------------------------------------------------------------------------

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
#include <vector>

namespace Test
{
namespace RebalanceDiag
{

using namespace Canopy;

enum FieldIdx
{
    Position = 0,
    Charge = 1,
    Velocity = 2,
    GlobalId = 3
};

static constexpr int P_ORDER = 8;

enum class Mode
{
    Migrate,
    Rebalance,
    Rebuild
};

inline const char* mode_name( Mode m )
{
    switch ( m )
    {
    case Mode::Migrate:
        return "Migrate";
    case Mode::Rebalance:
        return "Rebalance";
    default:
        return "Rebuild";
    }
}

// Brute-force N^2 gradient (and potential is not needed here).
inline void bf_gradient( const std::vector<double>& pos,
                         const std::vector<double>& chg,
                         std::vector<double>& grad )
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
            const double inv_r = 1.0 / std::sqrt( dx * dx + dy * dy + dz * dz );
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

// ---------------------------------------------------------------------------
// The driver.
// ---------------------------------------------------------------------------
inline void run_diag( Mode mode, int npp, int nsteps, double dt,
                      double drift_multiplier, int ncrit, int max_depth,
                      double tree_tol, int repl_depth )
{
    using DataTypes = Cabana::MemberTypes<double[3], double[1], double[3], int>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    using AoSoA_ht = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
    using Solver_t =
        Solver<TEST_MEMSPACE, TEST_EXECSPACE, double, P_ORDER, /*NComps=*/1>;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    // ---- initial particles (identical generator to tstMultiSolve.hpp) -----
    AoSoA_ht particles_h( "particles_h", npp );
    {
        auto hp = Cabana::slice<Position>( particles_h );
        auto hq = Cabana::slice<Charge>( particles_h );
        auto hv = Cabana::slice<Velocity>( particles_h );
        auto hid = Cabana::slice<GlobalId>( particles_h );
        std::mt19937 gen( 42 + rank * 7919 );
        std::uniform_real_distribution<double> pos_dist( 0.1, 0.9 );
        std::uniform_real_distribution<double> q_dist( 0.5, 1.5 );
        std::uniform_real_distribution<double> v_dist( -0.05, 0.05 );
        const int gid_base = rank * npp;
        for ( int i = 0; i < npp; i++ )
        {
            hp( i, 0 ) = pos_dist( gen );
            hp( i, 1 ) = pos_dist( gen );
            hp( i, 2 ) = pos_dist( gen );
            hq( i, 0 ) = q_dist( gen );
            hv( i, 0 ) = v_dist( gen );
            hv( i, 1 ) = v_dist( gen );
            hv( i, 2 ) = v_dist( gen );
            hid( i ) = gid_base + i;
        }
    }
    AoSoA_t particles( "particles", npp );
    Cabana::deep_copy( particles, particles_h );

    const int total = npp * nprocs;

    // ---- gather counts for the shadow run --------------------------------
    std::vector<int> cnt1( nprocs, npp ), dsp1( nprocs, 0 );
    std::vector<int> cnt3( nprocs, 3 * npp ), dsp3( nprocs, 0 );
    for ( int r = 1; r < nprocs; r++ )
    {
        dsp1[r] = dsp1[r - 1] + cnt1[r - 1];
        dsp3[r] = dsp3[r - 1] + cnt3[r - 1];
    }

    std::vector<double> lp( 3 * npp ), lq( npp ), lv( 3 * npp );
    std::vector<int> lg( npp );
    {
        auto hp = Cabana::slice<Position>( particles_h );
        auto hq = Cabana::slice<Charge>( particles_h );
        auto hv = Cabana::slice<Velocity>( particles_h );
        auto hid = Cabana::slice<GlobalId>( particles_h );
        for ( int i = 0; i < npp; i++ )
        {
            for ( int d = 0; d < 3; d++ )
            {
                lp[3 * i + d] = hp( i, d );
                lv[3 * i + d] = hv( i, d );
            }
            lq[i] = hq( i, 0 );
            lg[i] = hid( i );
        }
    }

    // Shadow (brute-force) trajectory state on rank 0, indexed by gid.
    std::vector<double> sh_pos( 3 * total ), sh_chg( total ), sh_vel( 3 * total );
    {
        std::vector<double> g_pos( 3 * total ), g_chg( total ), g_vel( 3 * total );
        std::vector<int> g_gid( total );
        MPI_Allgatherv( lp.data(), 3 * npp, MPI_DOUBLE, g_pos.data(),
                        cnt3.data(), dsp3.data(), MPI_DOUBLE, MPI_COMM_WORLD );
        MPI_Allgatherv( lq.data(), npp, MPI_DOUBLE, g_chg.data(), cnt1.data(),
                        dsp1.data(), MPI_DOUBLE, MPI_COMM_WORLD );
        MPI_Allgatherv( lv.data(), 3 * npp, MPI_DOUBLE, g_vel.data(),
                        cnt3.data(), dsp3.data(), MPI_DOUBLE, MPI_COMM_WORLD );
        MPI_Allgatherv( lg.data(), npp, MPI_INT, g_gid.data(), cnt1.data(),
                        dsp1.data(), MPI_INT, MPI_COMM_WORLD );
        // Reorder into gid order so lookups are direct.
        for ( int i = 0; i < total; i++ )
        {
            const int g = g_gid[i];
            sh_chg[g] = g_chg[i];
            for ( int d = 0; d < 3; d++ )
            {
                sh_pos[3 * g + d] = g_pos[3 * i + d];
                sh_vel[3 * g + d] = g_vel[3 * i + d];
            }
        }
    }

    // ---- solver ----------------------------------------------------------
    Canopy::FmmConfig cfg;
    cfg.ncrit = ncrit;
    cfg.max_depth = max_depth;
    cfg.xmin_tol = cfg.xmax_tol = tree_tol;
    cfg.ymin_tol = cfg.ymax_tol = tree_tol;
    cfg.zmin_tol = cfg.zmax_tol = tree_tol;
    cfg.ncrit_tol = tree_tol;
    cfg.replication_depth = repl_depth;
    cfg.imbalance_tolerance = 0.05;
    cfg.mac_theta = 0.5;
    cfg.softening = 0.0;
    Solver_t solver( MPI_COMM_WORLD, cfg );
    solver.template setup<Position, Charge>( particles, npp );

    if ( rank == 0 )
        std::printf( "\n### DIAG mode=%s np=%d dt=%.1e drift=%.1f "
                     "ncrit=%d max_depth=%d tree_tol=%.2f\n"
                     "%-5s %8s %12s %12s %12s %12s %12s %12s\n",
                     mode_name( mode ), nprocs, dt, drift_multiplier, ncrit,
                     max_depth, tree_tol, "step", "ncells", "box_extent",
                     "max_|v|", "force_max", "force_rms", "pos_max_rel",
                     "vel_max_rel" );

    for ( int step = 0; step < nsteps; step++ )
    {
        solver.template solve<Position, Charge>( particles,
                                                 /*compute_gradient=*/true );

        const int n_local = solver.num_local_particles();

        // ---- gather FMM state (positions/charges/gradient/gid) ----------
        auto pos_s = Cabana::slice<Position>( particles );
        auto chg_s = Cabana::slice<Charge>( particles );
        auto gid_s = Cabana::slice<GlobalId>( particles );
        auto h_pos = Canopy::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          pos_s, "hp" );
        auto h_chg = Canopy::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          chg_s, "hq" );
        auto h_gid = Canopy::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          gid_s, "hg" );
        auto h_grad = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                           solver.gradient() );

        std::vector<double> sp( 3 * n_local ), sq( n_local ), sg( 3 * n_local );
        std::vector<int> sgid( n_local );
        for ( int i = 0; i < n_local; i++ )
        {
            for ( int d = 0; d < 3; d++ )
            {
                sp[3 * i + d] = h_pos( i, d );
                sg[3 * i + d] = h_grad( i, 0, d );
            }
            sq[i] = h_chg( i, 0 );
            sgid[i] = h_gid( i );
        }

        std::vector<int> an( nprocs, 0 ), c1( nprocs, 0 ), d1( nprocs, 0 );
        std::vector<int> c3( nprocs, 0 ), d3( nprocs, 0 );
        MPI_Allgather( &n_local, 1, MPI_INT, an.data(), 1, MPI_INT,
                       MPI_COMM_WORLD );
        for ( int r = 0; r < nprocs; r++ )
        {
            c1[r] = an[r];
            c3[r] = 3 * an[r];
        }
        for ( int r = 1; r < nprocs; r++ )
        {
            d1[r] = d1[r - 1] + c1[r - 1];
            d3[r] = d3[r - 1] + c3[r - 1];
        }

        std::vector<double> gp( 3 * total ), gq( total ), gg( 3 * total );
        std::vector<int> gid( total );
        MPI_Allgatherv( sp.data(), 3 * n_local, MPI_DOUBLE, gp.data(),
                        c3.data(), d3.data(), MPI_DOUBLE, MPI_COMM_WORLD );
        MPI_Allgatherv( sq.data(), n_local, MPI_DOUBLE, gq.data(), c1.data(),
                        d1.data(), MPI_DOUBLE, MPI_COMM_WORLD );
        MPI_Allgatherv( sg.data(), 3 * n_local, MPI_DOUBLE, gg.data(),
                        c3.data(), d3.data(), MPI_DOUBLE, MPI_COMM_WORLD );
        MPI_Allgatherv( sgid.data(), n_local, MPI_INT, gid.data(), c1.data(),
                        d1.data(), MPI_INT, MPI_COMM_WORLD );

        double force_max = 0.0, force_rms = 0.0;
        if ( rank == 0 )
        {
            // Instantaneous FMM force error at the FMM's own positions.
            std::vector<double> ref;
            bf_gradient( gp, gq, ref );
            double acc = 0.0;
            for ( int i = 0; i < total; i++ )
            {
                const double rx = ref[3 * i + 0], ry = ref[3 * i + 1],
                             rz = ref[3 * i + 2];
                const double mag = std::sqrt( rx * rx + ry * ry + rz * rz );
                const double ex = gg[3 * i + 0] - rx;
                const double ey = gg[3 * i + 1] - ry;
                const double ez = gg[3 * i + 2] - rz;
                const double err = std::sqrt( ex * ex + ey * ey + ez * ez );
                const double rel = ( mag > 1e-12 ) ? err / mag : err;
                if ( rel > force_max )
                    force_max = rel;
                acc += rel * rel;
            }
            force_rms = std::sqrt( acc / total );
        }

        // ---- integrate FMM trajectory -----------------------------------
        auto positions = Cabana::slice<Position>( particles );
        auto velocities = Cabana::slice<Velocity>( particles );
        auto grad = solver.gradient();
        const double dt_l = dt, drift_l = drift_multiplier;
        Kokkos::parallel_for(
            "diag::integrate",
            Kokkos::RangePolicy<TEST_EXECSPACE>( 0, n_local ),
            KOKKOS_LAMBDA( int i ) {
                for ( int d = 0; d < 3; d++ )
                {
                    velocities( i, d ) += dt_l * grad( i, 0, d );
                    positions( i, d ) += dt_l * drift_l * velocities( i, d );
                }
            } );
        Kokkos::fence();

        // ---- integrate brute-force shadow trajectory --------------------
        double pos_max = 0.0, vel_max = 0.0;
        if ( rank == 0 )
        {
            std::vector<double> shg;
            bf_gradient( sh_pos, sh_chg, shg );
            for ( int i = 0; i < total; i++ )
                for ( int d = 0; d < 3; d++ )
                {
                    sh_vel[3 * i + d] += dt * shg[3 * i + d];
                    sh_pos[3 * i + d] += dt * drift_multiplier * sh_vel[3 * i + d];
                }
        }

        // ---- trajectory comparison (needs post-integration FMM state) ---
        {
            auto h_pos2 = Canopy::create_mirror_view_and_copy(
                Kokkos::HostSpace(), positions, "hp2" );
            auto h_vel2 = Canopy::create_mirror_view_and_copy(
                Kokkos::HostSpace(), velocities, "hv2" );
            std::vector<double> tp( 3 * n_local ), tv( 3 * n_local );
            for ( int i = 0; i < n_local; i++ )
                for ( int d = 0; d < 3; d++ )
                {
                    tp[3 * i + d] = h_pos2( i, d );
                    tv[3 * i + d] = h_vel2( i, d );
                }
            std::vector<double> ap( 3 * total ), av( 3 * total );
            MPI_Allgatherv( tp.data(), 3 * n_local, MPI_DOUBLE, ap.data(),
                            c3.data(), d3.data(), MPI_DOUBLE, MPI_COMM_WORLD );
            MPI_Allgatherv( tv.data(), 3 * n_local, MPI_DOUBLE, av.data(),
                            c3.data(), d3.data(), MPI_DOUBLE, MPI_COMM_WORLD );
            if ( rank == 0 )
            {
                for ( int i = 0; i < total; i++ )
                {
                    const int g = gid[i];
                    double pm = 0.0, pe = 0.0, vm = 0.0, ve = 0.0;
                    for ( int d = 0; d < 3; d++ )
                    {
                        const double b = sh_pos[3 * g + d];
                        pm += b * b;
                        const double e = ap[3 * i + d] - b;
                        pe += e * e;
                        const double bv = sh_vel[3 * g + d];
                        vm += bv * bv;
                        const double ev = av[3 * i + d] - bv;
                        ve += ev * ev;
                    }
                    pm = std::sqrt( pm );
                    vm = std::sqrt( vm );
                    pe = std::sqrt( pe );
                    ve = std::sqrt( ve );
                    const double pr = ( pm > 1e-10 ) ? pe / pm : pe;
                    const double vr = ( vm > 1e-10 ) ? ve / vm : ve;
                    if ( pr > pos_max )
                        pos_max = pr;
                    if ( vr > vel_max )
                        vel_max = vr;
                }
            }
        }

        const size_t ncells = solver.builder().cells().size();
        // Longest edge of the root bounding box, and the largest velocity
        // magnitude anywhere in the system (both computed on rank 0 from the
        // already-gathered state, so they are global).
        const auto& bb = solver.builder().root_box();
        double box_extent = 0.0;
        for ( int d = 0; d < 3; d++ )
            box_extent = std::max( box_extent, bb.max[d] - bb.min[d] );
        double vmax_glob = 0.0;
        if ( rank == 0 )
            for ( int i = 0; i < total; i++ )
            {
                double m = 0.0;
                for ( int d = 0; d < 3; d++ )
                    m += sh_vel[3 * i + d] * sh_vel[3 * i + d];
                vmax_glob = std::max( vmax_glob, std::sqrt( m ) );
            }
        if ( rank == 0 )
            std::printf( "%-5d %8zu %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e\n",
                         step, ncells, box_extent, vmax_glob, force_max,
                         force_rms, pos_max, vel_max );

        // ---- maintenance -------------------------------------------------
        switch ( mode )
        {
        case Mode::Migrate:
            solver.template migrate<Position>( particles );
            break;
        case Mode::Rebalance:
            solver.template rebalance<Position>( particles );
            break;
        case Mode::Rebuild:
            solver.template rebuild<Position, Charge>( particles );
            break;
        }
    }
    if ( rank == 0 )
        std::fflush( stdout );
}

} // namespace RebalanceDiag

//---------------------------------------------------------------------------//
// The 3x3 matrix: each maintenance mode run at each test's (dt, drift).
// If the mode is what matters, rows within a column differ. If the
// integrator settings are what matters, columns differ and rows do not.
//---------------------------------------------------------------------------//

#define DIAG_CASE( NAME, MODE, DT, DRIFT, NSTEPS )                            \
    TEST( RebalanceDiag, NAME )                                               \
    {                                                                         \
        RebalanceDiag::run_diag( RebalanceDiag::Mode::MODE, /*npp=*/200,      \
                                 NSTEPS, DT, DRIFT, /*ncrit=*/16,             \
                                 /*max_depth=*/6, /*tree_tol=*/0.1,           \
                                 /*repl_depth=*/2 );                          \
    }

// Column 1: the StableTree_Migrate integrator settings (dt=1e-4, drift=1).
DIAG_CASE( A_Migrate_dt1e4_drift1, Migrate, 1.0e-4, 1.0, 5 )
DIAG_CASE( A_Rebalance_dt1e4_drift1, Rebalance, 1.0e-4, 1.0, 5 )
DIAG_CASE( A_Rebuild_dt1e4_drift1, Rebuild, 1.0e-4, 1.0, 5 )

// Column 2: the IntermediateMotion_Rebalance settings (dt=1e-3, drift=5).
DIAG_CASE( B_Migrate_dt1e3_drift5, Migrate, 1.0e-3, 5.0, 5 )
DIAG_CASE( B_Rebalance_dt1e3_drift5, Rebalance, 1.0e-3, 5.0, 5 )
DIAG_CASE( B_Rebuild_dt1e3_drift5, Rebuild, 1.0e-3, 5.0, 5 )

// Column 3: the LargeMotion_Rebuild settings (dt=1e-3, drift=50).
DIAG_CASE( C_Migrate_dt1e3_drift50, Migrate, 1.0e-3, 50.0, 4 )
DIAG_CASE( C_Rebalance_dt1e3_drift50, Rebalance, 1.0e-3, 50.0, 4 )
DIAG_CASE( C_Rebuild_dt1e3_drift50, Rebuild, 1.0e-3, 50.0, 4 )

//---------------------------------------------------------------------------//

} // end namespace Test
