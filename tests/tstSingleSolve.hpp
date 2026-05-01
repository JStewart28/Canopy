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

#include "Canopy_CommunicationPlan.hpp"
#include "Canopy_DownwardSweep.hpp"
#include "Canopy_Helpers.hpp"
#include "Canopy_LaplaceKernel.hpp"
#include "Canopy_P2P.hpp"
#include "Canopy_TreeBuilder.hpp"
#include "Canopy_TreePartitioner.hpp"
#include "Canopy_UpwardSweep.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>
#include <random>
#include <vector>

namespace Test
{
//---------------------------------------------------------------------------//

using namespace Canopy;

namespace SingleSolveTest
{

enum FieldIdx
{
    Position = 0,
    Charge = 1
};

// Expansion order used for all single-solve tests.
static constexpr int P_ORDER = 8;

} // namespace SingleSolveTest

//---------------------------------------------------------------------------//
/**
 * End-to-end distributed FMM + P2P solve and comparison against a
 * brute-force O(N²) direct sum.
 *
 * NComps controls the number of simultaneous solves (charge components):
 *   NComps == 1  — one charge and one potential per particle
 *   NComps == 3  — three charges and three potentials per particle
 *
 * The full pipeline is:
 *   build → partition → rebuild → sort_by_leaf → rebuild → comm_plan
 *   → UpwardSweep (P2M + M2M) → DownwardSweep (M2L + L2L + L2P, far-field)
 *   → P2P (direct near-field, one component at a time)
 *
 * The combined FMM + P2P result is gathered to rank 0, which computes the
 * reference N-body sum and checks that the max relative error over all
 * particles and components is below fmm_tolerance.
 *
 * When compute_gradient is true, the same comparison is made for each of
 * the NComps × 3 gradient components.
 *
 * Brute-force potential at particle i, component c:
 *   phi_ref[i,c] = sum_{j≠i} q[j,c] / |r_i - r_j|
 *
 * Brute-force gradient at particle i, component c, spatial axis d:
 *   grad_ref[i,c,d] = sum_{j≠i} -q[j,c] * (r_i[d] - r_j[d]) / |r_i - r_j|^3
 */
template <int NComps>
void testFullSolve( bool compute_gradient, int num_particles_per_rank,
                    int ncrit, int max_depth, double tree_tolerance,
                    int replication_depth, double fmm_tolerance )
{
    using namespace SingleSolveTest;

    using Kernel = LaplaceKernel<double, P_ORDER, NComps>;
    using DataTypes = Cabana::MemberTypes<double[3], double[NComps]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    using AoSoA_ht = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
    using UpSweep = UpwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel>;
    using DwnSweep = DownwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel>;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    // -----------------------------------------------------------------------
    // Generate random particles
    // -----------------------------------------------------------------------
    AoSoA_ht particles_h( "particles_h", num_particles_per_rank );
    {
        auto hp = Cabana::slice<Position>( particles_h );
        auto hq = Cabana::slice<Charge>( particles_h );

        std::mt19937 gen( 42 + rank * ( NComps + 1 ) );
        std::uniform_real_distribution<double> pos_dist( 0.0, 1.0 );
        std::uniform_real_distribution<double> q_dist( -1.0, 1.0 );
        for ( int i = 0; i < num_particles_per_rank; i++ )
        {
            hp( i, 0 ) = pos_dist( gen );
            hp( i, 1 ) = pos_dist( gen );
            hp( i, 2 ) = pos_dist( gen );
            for ( int c = 0; c < NComps; c++ )
                hq( i, c ) = q_dist( gen );
        }
    }
    AoSoA_t particles( "particles", num_particles_per_rank );
    Cabana::deep_copy( particles, particles_h );

    // -----------------------------------------------------------------------
    // Pipeline setup
    // -----------------------------------------------------------------------
    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, tree_tolerance, tree_tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> comm_plan(
        MPI_COMM_WORLD );
    UpSweep upward( MPI_COMM_WORLD );
    DwnSweep downward( MPI_COMM_WORLD );
    P2P<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> p2p( MPI_COMM_WORLD );

    // Step 1: build tree on initial particle distribution
    auto positions = Cabana::slice<Position>( particles );
    builder.build( positions, num_particles_per_rank );

    // Step 2: partition (migrates particles across ranks)
    partitioner.partition( builder, particles, num_particles_per_rank );
    int num_local = partitioner.num_local_particles();

    // Step 3: rebuild tree for migrated particles
    positions = Cabana::slice<Position>( particles );
    builder.build( positions, num_local );

    // Step 4: sort AoSoA so each leaf's particles are contiguous.
    // particle_keys are stale after this call.
    partitioner.sort_particles_by_leaf( builder, particles );

    // Step 5: rebuild so particle_keys match the sorted AoSoA order.
    positions = Cabana::slice<Position>( particles );
    builder.build( positions, num_local );

    // Step 6: build communication plan
    comm_plan.build( builder.cells(), partitioner.ownership(),
                     partitioner.cell_owner_map(), replication_depth );

    // Step 7: setup sweeps and P2P
    upward.setup( builder.cells(), partitioner.cell_owner_map(),
                  builder.particle_keys(), num_local );
    downward.setup( upward, num_local );
    p2p.setup( builder, partitioner, comm_plan );

    // -----------------------------------------------------------------------
    // Execute: UpwardSweep → DownwardSweep → P2P
    // -----------------------------------------------------------------------
    auto charges = Cabana::slice<Charge>( particles );

    // Upward sweep: P2M at leaves, M2M up the tree
    upward.execute( charges, positions, comm_plan );

    // Allocate output views.  Both DownwardSweep and P2P ADD to these, so
    // they must be zero before execute().  A zero-extent gradient view is
    // passed when gradient evaluation is skipped.
    using pot_view = typename DwnSweep::potential_view_type;
    using grad_view = typename DwnSweep::gradient_view_type;
    pot_view potential( "potential", num_local );
    grad_view gradient( "gradient", compute_gradient ? num_local : 0 );
    Kokkos::deep_copy( potential, 0.0 );
    if ( compute_gradient )
        Kokkos::deep_copy( gradient, 0.0 );

    // Downward sweep: M2L, L2L, L2P — far-field contribution
    downward.execute( upward.multipoles(), positions, potential, gradient,
                      compute_gradient, comm_plan );

    // P2P: near-field (direct) contribution. All NComps are evaluated in
    // a single execute() call sharing one ghost-particle halo exchange.
    p2p.execute( positions, charges, potential, gradient, compute_gradient );

    // -----------------------------------------------------------------------
    // Gather all particle data and results to rank 0 for comparison.
    //
    // Use the Canopy helper to copy Cabana slices to host Kokkos views,
    // then pack into flat double buffers for MPI_Gatherv.
    // -----------------------------------------------------------------------

    // Copy slices to host (must use Canopy helper; Kokkos::create_mirror_view
    // does not support Cabana slice sources).
    auto h_pos = Canopy::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                      positions, "h_pos" );
    auto h_chg = Canopy::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                      charges, "h_chg" );

    // Copy output Kokkos views to host
    auto h_pot =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), potential );
    auto h_grad =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), gradient );

    // Pack into flat buffers (row-major: particle outermost)
    const int chg_stride = NComps;
    const int pot_stride = NComps;
    const int grad_stride = NComps * 3;

    std::vector<double> local_pos_buf( 3 * num_local );
    std::vector<double> local_chg_buf( chg_stride * num_local );
    std::vector<double> local_pot_buf( pot_stride * num_local );
    std::vector<double> local_grad_buf;
    if ( compute_gradient )
        local_grad_buf.resize( grad_stride * num_local );

    for ( int i = 0; i < num_local; i++ )
    {
        local_pos_buf[3 * i + 0] = h_pos( i, 0 );
        local_pos_buf[3 * i + 1] = h_pos( i, 1 );
        local_pos_buf[3 * i + 2] = h_pos( i, 2 );
        for ( int c = 0; c < NComps; c++ )
        {
            local_chg_buf[i * NComps + c] = h_chg( i, c );
            local_pot_buf[i * NComps + c] = h_pot( i, c );
        }
        if ( compute_gradient )
        {
            for ( int c = 0; c < NComps; c++ )
                for ( int d = 0; d < 3; d++ )
                    local_grad_buf[i * grad_stride + c * 3 + d] =
                        h_grad( i, c, d );
        }
    }

    // Gather particle counts
    std::vector<int> all_num_local( nprocs, 0 );
    MPI_Gather( &num_local, 1, MPI_INT, all_num_local.data(), 1, MPI_INT, 0,
                MPI_COMM_WORLD );

    // Build displacements and receive buffers on rank 0
    int total_particles = 0;
    std::vector<int> pos_counts( nprocs, 0 ), pos_displs( nprocs, 0 );
    std::vector<int> chg_counts( nprocs, 0 ), chg_displs( nprocs, 0 );
    std::vector<int> pot_counts( nprocs, 0 ), pot_displs( nprocs, 0 );
    std::vector<int> grad_counts( nprocs, 0 ), grad_displs( nprocs, 0 );
    std::vector<double> gathered_pos, gathered_chg, gathered_pot, gathered_grad;

    if ( rank == 0 )
    {
        for ( int r = 0; r < nprocs; r++ )
        {
            pos_counts[r] = 3 * all_num_local[r];
            chg_counts[r] = chg_stride * all_num_local[r];
            pot_counts[r] = pot_stride * all_num_local[r];
            grad_counts[r] = grad_stride * all_num_local[r];
            total_particles += all_num_local[r];
        }
        for ( int r = 1; r < nprocs; r++ )
        {
            pos_displs[r] = pos_displs[r - 1] + pos_counts[r - 1];
            chg_displs[r] = chg_displs[r - 1] + chg_counts[r - 1];
            pot_displs[r] = pot_displs[r - 1] + pot_counts[r - 1];
            grad_displs[r] = grad_displs[r - 1] + grad_counts[r - 1];
        }
        gathered_pos.resize( 3 * total_particles );
        gathered_chg.resize( chg_stride * total_particles );
        gathered_pot.resize( pot_stride * total_particles );
        if ( compute_gradient )
            gathered_grad.resize( grad_stride * total_particles );
    }

    MPI_Gatherv( local_pos_buf.data(), 3 * num_local, MPI_DOUBLE,
                 gathered_pos.data(), pos_counts.data(), pos_displs.data(),
                 MPI_DOUBLE, 0, MPI_COMM_WORLD );
    MPI_Gatherv( local_chg_buf.data(), chg_stride * num_local, MPI_DOUBLE,
                 gathered_chg.data(), chg_counts.data(), chg_displs.data(),
                 MPI_DOUBLE, 0, MPI_COMM_WORLD );
    MPI_Gatherv( local_pot_buf.data(), pot_stride * num_local, MPI_DOUBLE,
                 gathered_pot.data(), pot_counts.data(), pot_displs.data(),
                 MPI_DOUBLE, 0, MPI_COMM_WORLD );
    if ( compute_gradient )
        MPI_Gatherv( local_grad_buf.data(), grad_stride * num_local, MPI_DOUBLE,
                     gathered_grad.data(), grad_counts.data(),
                     grad_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD );

    // -----------------------------------------------------------------------
    // On rank 0: compute brute-force N-body sum and compare.
    // -----------------------------------------------------------------------
    if ( rank == 0 )
    {
        double max_pot_rel_err = 0.0;
        double max_grad_rel_err = 0.0;

        for ( int i = 0; i < total_particles; i++ )
        {
            const double xi = gathered_pos[3 * i + 0];
            const double yi = gathered_pos[3 * i + 1];
            const double zi = gathered_pos[3 * i + 2];

            for ( int c = 0; c < NComps; c++ )
            {
                double phi_ref = 0.0;
                double gx_ref = 0.0, gy_ref = 0.0, gz_ref = 0.0;

                for ( int j = 0; j < total_particles; j++ )
                {
                    if ( j == i )
                        continue;
                    const double dx = xi - gathered_pos[3 * j + 0];
                    const double dy = yi - gathered_pos[3 * j + 1];
                    const double dz = zi - gathered_pos[3 * j + 2];
                    const double r2 = dx * dx + dy * dy + dz * dz;
                    const double inv_r = 1.0 / std::sqrt( r2 );
                    const double inv_r3 = inv_r * inv_r * inv_r;
                    const double qjc = gathered_chg[j * NComps + c];

                    phi_ref += qjc * inv_r;
                    if ( compute_gradient )
                    {
                        gx_ref -= qjc * dx * inv_r3;
                        gy_ref -= qjc * dy * inv_r3;
                        gz_ref -= qjc * dz * inv_r3;
                    }
                }

                // Potential relative error
                const double phi_fmm = gathered_pot[i * NComps + c];
                const double ref_mag = std::abs( phi_ref );
                const double pot_err = std::abs( phi_fmm - phi_ref );
                const double pot_rel =
                    ( ref_mag > 1.0e-10 ) ? pot_err / ref_mag : pot_err;
                if ( pot_rel > max_pot_rel_err )
                    max_pot_rel_err = pot_rel;

                // Gradient relative error
                if ( compute_gradient )
                {
                    const double gx_fmm =
                        gathered_grad[i * grad_stride + c * 3 + 0];
                    const double gy_fmm =
                        gathered_grad[i * grad_stride + c * 3 + 1];
                    const double gz_fmm =
                        gathered_grad[i * grad_stride + c * 3 + 2];

                    const double grad_mag = std::sqrt(
                        gx_ref * gx_ref + gy_ref * gy_ref + gz_ref * gz_ref );
                    const double grad_err =
                        std::max( { std::abs( gx_fmm - gx_ref ),
                                    std::abs( gy_fmm - gy_ref ),
                                    std::abs( gz_fmm - gz_ref ) } );
                    const double grad_rel =
                        ( grad_mag > 1.0e-10 ) ? grad_err / grad_mag : grad_err;
                    if ( grad_rel > max_grad_rel_err )
                        max_grad_rel_err = grad_rel;
                }
            }
        }

        EXPECT_LT( max_pot_rel_err, fmm_tolerance )
            << "FMM+P2P potential (NComps=" << NComps
            << ") deviates from brute-force N-body sum; "
               "max relative error = "
            << max_pot_rel_err;

        if ( compute_gradient )
            EXPECT_LT( max_grad_rel_err, fmm_tolerance )
                << "FMM+P2P gradient (NComps=" << NComps
                << ") deviates from brute-force N-body gradient; "
                   "max relative error = "
                << max_grad_rel_err;
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//
// Parameters: num_particles_per_rank, ncrit, max_depth, tree_tolerance,
//             replication_depth, fmm_tolerance.
//
// num_particles_per_rank is kept modest (200) so the O(N²) brute-force
// reference on rank 0 remains fast.
//
// fmm_tolerance of 1e-3 is a conservative bound for P_ORDER=6 with a
// random uniform particle distribution.
//---------------------------------------------------------------------------//

/**
 * Test 1: Laplace single-component (NComps=1) — potential only.
 * Each particle carries one charge; verify that the combined FMM+P2P
 * potential matches the brute-force direct sum over all particle pairs.
 */
TEST( SingleSolve, PotentialNComps1 )
{
    testFullSolve<1>( false, 500, 16, 6, 0.1, 2, 1.0e-3 );
}

/**
 * Test 2: Laplace single-component (NComps=1) — potential and forces.
 * Same as Test 1 but also verifies the gradient (force) via the same
 * brute-force comparison.
 */
TEST( SingleSolve, PotentialAndGradientNComps1 )
{
    testFullSolve<1>( true, 500, 16, 6, 0.1, 2, 1.0e-3 );
}

/**
 * Test 3: Laplace multi-component (NComps=3) — potential only.
 * Each particle carries three independent charges; the FMM runs all
 * three simultaneously. Each component's potential is compared
 * independently against its own brute-force direct sum.
 */
TEST( SingleSolve, PotentialNComps3 )
{
    testFullSolve<3>( false, 500, 16, 6, 0.1, 2, 1.0e-3 );
}

/**
 * Test 4: Laplace multi-component (NComps=3) — potential and forces.
 * Same as Test 3 but also verifies all nine gradient components
 * (NComps × 3 spatial directions) against the brute-force reference.
 */
TEST( SingleSolve, PotentialAndGradientNComps3 )
{
    testFullSolve<3>( true, 500, 16, 6, 0.1, 2, 1.0e-3 );
}

//---------------------------------------------------------------------------//

} // end namespace Test
