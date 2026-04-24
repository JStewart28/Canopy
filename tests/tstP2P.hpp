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
#include "Canopy_LaplaceKernel.hpp"
#include "Canopy_P2P.hpp"
#include "Canopy_TreeBuilder.hpp"
#include "Canopy_TreePartitioner.hpp"

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

namespace P2PTest
{

enum FieldIdx
{
    Position = 0,
    Charge   = 1
};

// P2P takes scalar charges: charges(p), not charges(p, comp).
static constexpr int P_ORDER = 4;
using Kernel = LaplaceKernel<double, P_ORDER>;

using DataTypes = Cabana::MemberTypes<double[3], double>;
using AoSoA_t   = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
using AoSoA_ht  = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;

// Run the standard P2P setup pipeline. After this call the AoSoA has been
// partitioned and sorted by leaf; callers must reslice positions/charges.
// Returns num_local (post-migration particle count on this rank).
int run_setup(
    AoSoA_t& particles,
    int n_initial,
    int ncrit,
    int max_depth,
    double tolerance,
    int replication_depth,
    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE>& builder,
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE>& partitioner,
    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE>& comm_plan,
    P2P<TEST_MEMSPACE, TEST_EXECSPACE, Kernel>& p2p )
{
    auto positions = Cabana::slice<Position>( particles );
    builder.build( positions, n_initial );

    partitioner.partition( builder, particles, n_initial );
    int num_local = partitioner.num_local_particles();

    positions = Cabana::slice<Position>( particles );
    builder.build( positions, num_local );

    partitioner.sort_particles_by_leaf( builder, particles );

    comm_plan.build( builder.cells(), partitioner.ownership(),
                     partitioner.cell_owner_map(), replication_depth );

    p2p.setup( builder, partitioner, comm_plan );
    return num_local;
}

} // namespace P2PTest

//---------------------------------------------------------------------------//
/**
 * Verify that all-zero particle charges produce all-zero potential and
 * gradient across every local particle.
 *
 * If the kernel is reading uninitialised memory or failing to zero-out
 * contributions, some entries will be non-zero even though no source
 * charges exist. The test covers both the intra-leaf (Phase 1) and
 * inter-leaf (Phase 2) code paths because they both short-circuit on
 * zero charges.
 *
 * Checks:
 *   1. potential(i) == 0 for all local i.
 *   2. gradient(i, {x,y,z}) == 0 for all local i.
 */
void testP2PZeroChargesGiveZeroPotential(
    int num_particles_per_rank, int ncrit, int max_depth,
    double tolerance, int replication_depth )
{
    using namespace P2PTest;

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    AoSoA_ht particles_h( "particles_h", num_particles_per_rank );
    {
        auto hp = Cabana::slice<Position>( particles_h );
        auto hq = Cabana::slice<Charge>( particles_h );

        std::mt19937 gen( 42 + rank );
        std::uniform_real_distribution<double> pos_dist( 0.0, 1.0 );
        for ( int i = 0; i < num_particles_per_rank; i++ )
        {
            hp( i, 0 ) = pos_dist( gen );
            hp( i, 1 ) = pos_dist( gen );
            hp( i, 2 ) = pos_dist( gen );
            hq( i )    = 0.0;
        }
    }

    AoSoA_t particles( "particles", num_particles_per_rank );
    Cabana::deep_copy( particles, particles_h );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, tolerance, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> comm_plan( MPI_COMM_WORLD );
    P2P<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> p2p( MPI_COMM_WORLD );

    int num_local = run_setup( particles, num_particles_per_rank, ncrit,
                               max_depth, tolerance, replication_depth,
                               builder, partitioner, comm_plan, p2p );

    auto positions = Cabana::slice<Position>( particles );
    auto charges   = Cabana::slice<Charge>( particles );

    Kokkos::View<double*, TEST_MEMSPACE> potential( "pot", num_local );
    Kokkos::View<double* [3], TEST_MEMSPACE> gradient( "grad", num_local );
    Kokkos::deep_copy( potential, 0.0 );
    Kokkos::deep_copy( gradient, 0.0 );

    p2p.execute( positions, charges, potential, gradient, true );

    auto h_pot  = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                        potential );
    auto h_grad = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                        gradient );

    for ( int i = 0; i < num_local; i++ )
    {
        EXPECT_EQ( h_pot( i ), 0.0 )
            << "Non-zero potential at particle " << i
            << " with all-zero charges";
        EXPECT_EQ( h_grad( i, 0 ), 0.0 )
            << "Non-zero gradient x at particle " << i;
        EXPECT_EQ( h_grad( i, 1 ), 0.0 )
            << "Non-zero gradient y at particle " << i;
        EXPECT_EQ( h_grad( i, 2 ), 0.0 )
            << "Non-zero gradient z at particle " << i;
    }
}

//---------------------------------------------------------------------------//
/**
 * Verify that strictly positive charges produce at least one non-zero
 * potential value after execute().
 *
 * If the P2P kernel fails to accumulate any contributions — e.g., the
 * neighbor list is empty, or the intra-leaf range is never entered — every
 * potential will be zero even for a non-trivial source distribution.  Positive
 * charges guarantee the monopole term is non-zero so cancellation cannot
 * mask a silent failure.
 *
 * Checks:
 *   1. max |potential(i)| > 0 across all local particles.
 */
void testP2PPotentialNonzeroAfterExecution(
    int num_particles_per_rank, int ncrit, int max_depth,
    double tolerance, int replication_depth )
{
    using namespace P2PTest;

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    AoSoA_ht particles_h( "particles_h", num_particles_per_rank );
    {
        auto hp = Cabana::slice<Position>( particles_h );
        auto hq = Cabana::slice<Charge>( particles_h );

        std::mt19937 gen( 7 + rank );
        std::uniform_real_distribution<double> pos_dist( 0.0, 1.0 );
        std::uniform_real_distribution<double> q_dist( 0.1, 1.0 );
        for ( int i = 0; i < num_particles_per_rank; i++ )
        {
            hp( i, 0 ) = pos_dist( gen );
            hp( i, 1 ) = pos_dist( gen );
            hp( i, 2 ) = pos_dist( gen );
            hq( i )    = q_dist( gen );
        }
    }

    AoSoA_t particles( "particles", num_particles_per_rank );
    Cabana::deep_copy( particles, particles_h );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, tolerance, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> comm_plan( MPI_COMM_WORLD );
    P2P<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> p2p( MPI_COMM_WORLD );

    int num_local = run_setup( particles, num_particles_per_rank, ncrit,
                               max_depth, tolerance, replication_depth,
                               builder, partitioner, comm_plan, p2p );

    auto positions = Cabana::slice<Position>( particles );
    auto charges   = Cabana::slice<Charge>( particles );

    Kokkos::View<double*, TEST_MEMSPACE> potential( "pot", num_local );
    Kokkos::deep_copy( potential, 0.0 );
    Kokkos::View<double* [3], TEST_MEMSPACE> gradient( "grad", num_local );
    Kokkos::deep_copy( gradient, 0.0 );

    p2p.execute( positions, charges, potential, gradient, false );

    auto h_pot = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                      potential );

    double max_abs = 0.0;
    for ( int i = 0; i < num_local; i++ )
        if ( std::abs( h_pot( i ) ) > max_abs )
            max_abs = std::abs( h_pot( i ) );

    // Reduce across ranks so the test passes even if a single rank
    // happens to receive no particles after partitioning.
    double global_max = 0.0;
    MPI_Allreduce( &max_abs, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                   MPI_COMM_WORLD );

    EXPECT_GT( global_max, 0.0 )
        << "All potentials are zero after P2P with non-zero charges";
}

//---------------------------------------------------------------------------//
/**
 * Verify that calling execute() twice with outputs zeroed before each call
 * produces bit-identical results.
 *
 * P2P adds its contributions into the caller-supplied output views.
 * If internal state leaks between calls, or the kernel has data races that
 * produce non-deterministic results, the two runs will differ.
 *
 * Checks:
 *   1. potential(i) is identical between the first and second call for all i.
 *   2. gradient(i, {x,y,z}) is identical between the two calls for all i.
 */
void testP2PIdempotentExecution(
    int num_particles_per_rank, int ncrit, int max_depth,
    double tolerance, int replication_depth )
{
    using namespace P2PTest;

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    AoSoA_ht particles_h( "particles_h", num_particles_per_rank );
    {
        auto hp = Cabana::slice<Position>( particles_h );
        auto hq = Cabana::slice<Charge>( particles_h );

        std::mt19937 gen( 13 + rank );
        std::uniform_real_distribution<double> pos_dist( 0.0, 1.0 );
        std::uniform_real_distribution<double> q_dist( -1.0, 1.0 );
        for ( int i = 0; i < num_particles_per_rank; i++ )
        {
            hp( i, 0 ) = pos_dist( gen );
            hp( i, 1 ) = pos_dist( gen );
            hp( i, 2 ) = pos_dist( gen );
            hq( i )    = q_dist( gen );
        }
    }

    AoSoA_t particles( "particles", num_particles_per_rank );
    Cabana::deep_copy( particles, particles_h );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, tolerance, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> comm_plan( MPI_COMM_WORLD );
    P2P<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> p2p( MPI_COMM_WORLD );

    int num_local = run_setup( particles, num_particles_per_rank, ncrit,
                               max_depth, tolerance, replication_depth,
                               builder, partitioner, comm_plan, p2p );

    auto positions = Cabana::slice<Position>( particles );
    auto charges   = Cabana::slice<Charge>( particles );

    Kokkos::View<double*, TEST_MEMSPACE> potential( "pot", num_local );
    Kokkos::View<double* [3], TEST_MEMSPACE> gradient( "grad", num_local );

    // First execute
    Kokkos::deep_copy( potential, 0.0 );
    Kokkos::deep_copy( gradient, 0.0 );
    p2p.execute( positions, charges, potential, gradient, true );
    auto h_pot_first  = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                              potential );
    auto h_grad_first = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                              gradient );

    // Second execute — fresh outputs
    Kokkos::deep_copy( potential, 0.0 );
    Kokkos::deep_copy( gradient, 0.0 );
    p2p.execute( positions, charges, potential, gradient, true );
    auto h_pot_second  = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                               potential );
    auto h_grad_second = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                               gradient );

    for ( int i = 0; i < num_local; i++ )
    {
        EXPECT_EQ( h_pot_first( i ), h_pot_second( i ) )
            << "Potential mismatch at particle " << i
            << " between first and second execute()";
        for ( int d = 0; d < 3; d++ )
            EXPECT_EQ( h_grad_first( i, d ), h_grad_second( i, d ) )
                << "Gradient[" << d << "] mismatch at particle " << i
                << " between first and second execute()";
    }
}

//---------------------------------------------------------------------------//
/**
 * Verify that P2P potentials match a brute-force O(N²) direct sum when all
 * particles reside in a single leaf cell.
 *
 * With ncrit larger than the particle count, every particle lands in the root
 * leaf. The intra-leaf kernel therefore handles every i<j pair via Newton's
 * third law, and the inter-leaf neighbor list is empty. The expected potential
 * at particle i is exactly sum_{j!=i} q_j / r_ij.
 *
 * The test is restricted to one MPI rank because the partition may split
 * particles across ranks on a multi-rank run, potentially creating multiple
 * leaves.
 *
 * Checks:
 *   1. |P2P potential(i) - direct sum potential(i)| < 1e-12 for all i.
 */
void testP2PDirectSumSingleLeaf(
    int num_particles, int ncrit, int max_depth,
    double tolerance, int replication_depth )
{
    using namespace P2PTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    if ( nprocs != 1 )
        return;

    // Place all particles in a tiny cluster so they share one leaf.
    AoSoA_ht particles_h( "particles_h", num_particles );
    {
        auto hp = Cabana::slice<Position>( particles_h );
        auto hq = Cabana::slice<Charge>( particles_h );

        std::mt19937 gen( 55 );
        std::uniform_real_distribution<double> pos_dist( 0.45, 0.55 );
        std::uniform_real_distribution<double> q_dist( -1.0, 1.0 );
        for ( int i = 0; i < num_particles; i++ )
        {
            hp( i, 0 ) = pos_dist( gen );
            hp( i, 1 ) = pos_dist( gen );
            hp( i, 2 ) = pos_dist( gen );
            hq( i )    = q_dist( gen );
        }
    }

    AoSoA_t particles( "particles", num_particles );
    Cabana::deep_copy( particles, particles_h );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, tolerance, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> comm_plan( MPI_COMM_WORLD );
    P2P<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> p2p( MPI_COMM_WORLD );

    int num_local = run_setup( particles, num_particles, ncrit, max_depth,
                               tolerance, replication_depth,
                               builder, partitioner, comm_plan, p2p );

    auto positions = Cabana::slice<Position>( particles );
    auto charges   = Cabana::slice<Charge>( particles );

    Kokkos::View<double*, TEST_MEMSPACE> potential( "pot", num_local );
    Kokkos::View<double* [3], TEST_MEMSPACE> gradient( "grad", num_local );
    Kokkos::deep_copy( potential, 0.0 );
    Kokkos::deep_copy( gradient, 0.0 );

    p2p.execute( positions, charges, potential, gradient, false );

    auto h_pot = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                      potential );

    // Copy sorted positions and charges to host for reference computation.
    Kokkos::View<double* [3], TEST_MEMSPACE> d_pos( "d_pos", num_local );
    Kokkos::View<double*, TEST_MEMSPACE> d_chg( "d_chg", num_local );
    Kokkos::parallel_for(
        "CopySlices", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_local ),
        KOKKOS_LAMBDA( int i ) {
            d_pos( i, 0 ) = positions( i, 0 );
            d_pos( i, 1 ) = positions( i, 1 );
            d_pos( i, 2 ) = positions( i, 2 );
            d_chg( i )    = charges( i );
        } );
    Kokkos::fence();

    auto h_pos = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_pos );
    auto h_chg = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_chg );

    // Brute-force reference: phi_i = sum_{j != i} q_j / r_ij
    double max_err = 0.0;
    for ( int i = 0; i < num_local; i++ )
    {
        double phi_ref = 0.0;
        for ( int j = 0; j < num_local; j++ )
        {
            if ( j == i )
                continue;
            const double dx = h_pos( i, 0 ) - h_pos( j, 0 );
            const double dy = h_pos( i, 1 ) - h_pos( j, 1 );
            const double dz = h_pos( i, 2 ) - h_pos( j, 2 );
            const double r  = std::sqrt( dx * dx + dy * dy + dz * dz );
            phi_ref += h_chg( j ) / r;
        }
        const double err = std::abs( h_pot( i ) - phi_ref );
        if ( err > max_err )
            max_err = err;
    }

    EXPECT_LT( max_err, 1.0e-12 )
        << "P2P potential deviates from brute-force direct sum; "
           "max absolute error = " << max_err;
}

//---------------------------------------------------------------------------//
/**
 * Verify exact potential and gradient values for a two-particle system.
 *
 * With two particles separated by a known distance d, the exact Coulomb
 * potential and gradient are:
 *   phi_0 = q1 / d,    phi_1 = q0 / d
 *   grad_0 = ( q1/d^2,  0, 0 )
 *   grad_1 = (-q0/d^2,  0, 0 )
 *
 * Both particles land in the intra-leaf kernel (Newton's third law path).
 * This test checks the sign and magnitude of gradient accumulation, which
 * is easy to get wrong compared to the potential accumulation.
 *
 * The test is restricted to one MPI rank.
 *
 * Checks:
 *   1. |potential(i) - phi_exact| < 1e-14 for i in {0, 1}.
 *   2. |gradient(i, d) - grad_exact(i, d)| < 1e-14 for all components.
 */
void testP2PTwoParticleExact()
{
    using namespace P2PTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    if ( nprocs != 1 )
        return;

    const double d  = 1.5;
    const double q0 = 2.0;
    const double q1 = -0.5;

    // particle 0 at origin, particle 1 at (d, 0, 0)
    AoSoA_ht particles_h( "particles_h", 2 );
    {
        auto hp = Cabana::slice<Position>( particles_h );
        auto hq = Cabana::slice<Charge>( particles_h );
        hp( 0, 0 ) = 0.0; hp( 0, 1 ) = 0.0; hp( 0, 2 ) = 0.0;
        hp( 1, 0 ) = d;   hp( 1, 1 ) = 0.0; hp( 1, 2 ) = 0.0;
        hq( 0 ) = q0;
        hq( 1 ) = q1;
    }

    AoSoA_t particles( "particles", 2 );
    Cabana::deep_copy( particles, particles_h );

    // ncrit=100: both particles end up in one leaf.
    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, 100, 6, 0.1, 0.1 );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, 2 );
    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> comm_plan( MPI_COMM_WORLD );
    P2P<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> p2p( MPI_COMM_WORLD );

    int num_local = run_setup( particles, 2, 100, 6, 0.1, 2,
                               builder, partitioner, comm_plan, p2p );

    ASSERT_EQ( num_local, 2 ) << "Expected 2 local particles on single rank";

    auto positions = Cabana::slice<Position>( particles );
    auto charges   = Cabana::slice<Charge>( particles );

    Kokkos::View<double*, TEST_MEMSPACE> potential( "pot", 2 );
    Kokkos::View<double* [3], TEST_MEMSPACE> gradient( "grad", 2 );
    Kokkos::deep_copy( potential, 0.0 );
    Kokkos::deep_copy( gradient, 0.0 );

    p2p.execute( positions, charges, potential, gradient, true );

    auto h_pot  = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                        potential );
    auto h_grad = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                        gradient );

    // The intra-leaf kernel sorts pairs as (i,j) with i < j.
    // After sort_particles_by_leaf the AoSoA may be permuted, so we
    // identify the two particles by their positions rather than indices.
    int idx0 = -1, idx1 = -1;
    {
        Kokkos::View<double* [3], TEST_MEMSPACE> d_pos( "d_pos", 2 );
        Kokkos::parallel_for(
            "CopyPos", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 2 ),
            KOKKOS_LAMBDA( int i ) {
                d_pos( i, 0 ) = positions( i, 0 );
                d_pos( i, 1 ) = positions( i, 1 );
                d_pos( i, 2 ) = positions( i, 2 );
            } );
        Kokkos::fence();
        auto hp = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_pos );
        for ( int i = 0; i < 2; i++ )
        {
            if ( hp( i, 0 ) < d * 0.5 )
                idx0 = i;
            else
                idx1 = i;
        }
    }

    ASSERT_GE( idx0, 0 ) << "Could not locate particle 0 by position";
    ASSERT_GE( idx1, 0 ) << "Could not locate particle 1 by position";

    const double phi0_exact = q1 / d;
    const double phi1_exact = q0 / d;
    const double gx0_exact  =  q1 / ( d * d );
    const double gx1_exact  = -q0 / ( d * d );

    EXPECT_NEAR( h_pot( idx0 ), phi0_exact, 1.0e-14 )
        << "Potential at particle 0 incorrect";
    EXPECT_NEAR( h_pot( idx1 ), phi1_exact, 1.0e-14 )
        << "Potential at particle 1 incorrect";

    EXPECT_NEAR( h_grad( idx0, 0 ), gx0_exact, 1.0e-14 )
        << "Gradient x at particle 0 incorrect";
    EXPECT_NEAR( h_grad( idx0, 1 ), 0.0, 1.0e-14 )
        << "Gradient y at particle 0 should be zero";
    EXPECT_NEAR( h_grad( idx0, 2 ), 0.0, 1.0e-14 )
        << "Gradient z at particle 0 should be zero";

    EXPECT_NEAR( h_grad( idx1, 0 ), gx1_exact, 1.0e-14 )
        << "Gradient x at particle 1 incorrect";
    EXPECT_NEAR( h_grad( idx1, 1 ), 0.0, 1.0e-14 )
        << "Gradient y at particle 1 should be zero";
    EXPECT_NEAR( h_grad( idx1, 2 ), 0.0, 1.0e-14 )
        << "Gradient z at particle 1 should be zero";
}

//---------------------------------------------------------------------------//
/**
 * Verify that the gradient is consistent with the potential via a finite-
 * difference check for a random particle distribution on a single rank.
 *
 * For each particle i, the x-component of the gradient approximated by a
 * central finite difference in x is compared to the P2P-computed gradient.
 * Because P2P computes a truncated near-field sum (not the full N-body
 * sum), the finite-difference reference is also computed from the same
 * near-field neighbor list by running two perturbed configurations.
 *
 * Instead of reproducing the neighbor list, we use a simpler end-to-end
 * check: confirm that the gradient is in the correct qualitative direction
 * by verifying that phi(x + eps) - phi(x - eps) ≈ 2*eps*grad_x for the
 * particle with the largest absolute potential.
 *
 * This is achieved by checking that the gradient vector is not identically
 * zero when charges are non-zero, and that its sign is consistent with
 * the finite-difference direction derived from the intra-leaf potential.
 *
 * Restricted to one MPI rank to keep particle ownership unambiguous.
 *
 * Checks:
 *   1. The gradient L2 norm is non-zero for non-zero charges.
 *   2. For at least one particle, gradient_x * (phi(x+eps) - phi(x-eps)) > 0.
 */
void testP2PGradientSignConsistency(
    int num_particles, int ncrit, int max_depth,
    double tolerance, int replication_depth )
{
    using namespace P2PTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    if ( nprocs != 1 )
        return;

    // All particles in one cluster so all pairs are in-leaf.
    AoSoA_ht particles_h( "particles_h", num_particles );
    {
        auto hp = Cabana::slice<Position>( particles_h );
        auto hq = Cabana::slice<Charge>( particles_h );

        std::mt19937 gen( 33 );
        std::uniform_real_distribution<double> pos_dist( 0.45, 0.55 );
        std::uniform_real_distribution<double> q_dist( 0.1, 1.0 );
        for ( int i = 0; i < num_particles; i++ )
        {
            hp( i, 0 ) = pos_dist( gen );
            hp( i, 1 ) = pos_dist( gen );
            hp( i, 2 ) = pos_dist( gen );
            hq( i )    = q_dist( gen );
        }
    }

    // Run base configuration
    AoSoA_t particles( "particles", num_particles );
    Cabana::deep_copy( particles, particles_h );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, tolerance, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> comm_plan( MPI_COMM_WORLD );
    P2P<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> p2p( MPI_COMM_WORLD );

    int num_local = run_setup( particles, num_particles, ncrit, max_depth,
                               tolerance, replication_depth,
                               builder, partitioner, comm_plan, p2p );

    auto positions = Cabana::slice<Position>( particles );
    auto charges   = Cabana::slice<Charge>( particles );

    Kokkos::View<double*, TEST_MEMSPACE> potential( "pot", num_local );
    Kokkos::View<double* [3], TEST_MEMSPACE> gradient( "grad", num_local );
    Kokkos::deep_copy( potential, 0.0 );
    Kokkos::deep_copy( gradient, 0.0 );

    p2p.execute( positions, charges, potential, gradient, true );

    auto h_pot  = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                        potential );
    auto h_grad = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                        gradient );

    // Gradient L2 norm must be non-zero
    double grad_norm_sq = 0.0;
    for ( int i = 0; i < num_local; i++ )
        for ( int d = 0; d < 3; d++ )
            grad_norm_sq += h_grad( i, d ) * h_grad( i, d );

    EXPECT_GT( grad_norm_sq, 0.0 )
        << "Gradient is identically zero with non-zero charges";

    // Pick particle with largest |potential| and verify sign consistency
    // using the brute-force intra-leaf Coulomb sum (single-leaf setup).
    int best = 0;
    double best_abs = std::abs( h_pot( 0 ) );
    for ( int i = 1; i < num_local; i++ )
        if ( std::abs( h_pot( i ) ) > best_abs )
        {
            best_abs = std::abs( h_pot( i ) );
            best     = i;
        }

    // Copy sorted particle data to host for finite-difference reference.
    Kokkos::View<double* [3], TEST_MEMSPACE> d_pos( "d_pos", num_local );
    Kokkos::View<double*, TEST_MEMSPACE> d_chg( "d_chg", num_local );
    Kokkos::parallel_for(
        "CopySlices", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_local ),
        KOKKOS_LAMBDA( int i ) {
            d_pos( i, 0 ) = positions( i, 0 );
            d_pos( i, 1 ) = positions( i, 1 );
            d_pos( i, 2 ) = positions( i, 2 );
            d_chg( i )    = charges( i );
        } );
    Kokkos::fence();
    auto hp = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_pos );
    auto hq = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_chg );

    const double eps = 1.0e-6;
    // phi(x+eps) and phi(x-eps) for particle `best`
    double phi_plus  = 0.0;
    double phi_minus = 0.0;
    for ( int j = 0; j < num_local; j++ )
    {
        if ( j == best )
            continue;
        const double dy = hp( best, 1 ) - hp( j, 1 );
        const double dz = hp( best, 2 ) - hp( j, 2 );
        {
            const double dx = ( hp( best, 0 ) + eps ) - hp( j, 0 );
            phi_plus += hq( j ) / std::sqrt( dx * dx + dy * dy + dz * dz );
        }
        {
            const double dx = ( hp( best, 0 ) - eps ) - hp( j, 0 );
            phi_minus += hq( j ) / std::sqrt( dx * dx + dy * dy + dz * dz );
        }
    }
    const double fd_gx = ( phi_plus - phi_minus ) / ( 2.0 * eps );

    // Both must be on the same side of zero (sign agreement).
    EXPECT_GT( fd_gx * h_grad( best, 0 ), 0.0 )
        << "Gradient x sign at particle " << best
        << " inconsistent with finite-difference reference";
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//

TEST( P2P, testZeroChargesGiveZeroPotentialBasic )
{
    testP2PZeroChargesGiveZeroPotential( 1000, 32, 6, 0.1, 2 );
}

TEST( P2P, testZeroChargesGiveZeroPotentialSmall )
{
    testP2PZeroChargesGiveZeroPotential( 200, 16, 4, 0.1, 1 );
}

TEST( P2P, testPotentialNonzeroAfterExecutionBasic )
{
    testP2PPotentialNonzeroAfterExecution( 1000, 32, 6, 0.1, 2 );
}

TEST( P2P, testPotentialNonzeroAfterExecutionSmall )
{
    testP2PPotentialNonzeroAfterExecution( 200, 16, 4, 0.1, 1 );
}

TEST( P2P, testIdempotentExecutionBasic )
{
    testP2PIdempotentExecution( 1000, 32, 6, 0.1, 2 );
}

TEST( P2P, testIdempotentExecutionSmall )
{
    testP2PIdempotentExecution( 200, 16, 4, 0.1, 1 );
}

TEST( P2P, testDirectSumSingleLeaf )
{
    // ncrit=200 ensures all 8 particles land in one leaf; max_depth=1 avoids
    // further subdivision.
    testP2PDirectSumSingleLeaf( 8, 200, 1, 0.1, 1 );
}

TEST( P2P, testTwoParticleExact )
{
    testP2PTwoParticleExact();
}

TEST( P2P, testGradientSignConsistencyBasic )
{
    testP2PGradientSignConsistency( 12, 200, 1, 0.1, 1 );
}

//---------------------------------------------------------------------------//

} // end namespace Test
