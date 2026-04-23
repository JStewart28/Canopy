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
#include "Canopy_LaplaceKernel.hpp"
#include "Canopy_SphericalCoefficients.hpp"
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

namespace DownwardSweepTest
{

enum FieldIdx
{
    Position = 0,
    Charge   = 1
};

static constexpr int P_ORDER = 6;
using Kernel = LaplaceKernel<double, P_ORDER>;

using DataTypes = Cabana::MemberTypes<double[3], double[Kernel::num_components]>;
using AoSoA_t  = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
using AoSoA_ht = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;

// Generate particles with random positions in [0, 1)^3 and strictly positive
// charges in [0.1, 1.0] so the monopole moment is guaranteed non-zero.
void generate_test_particles( AoSoA_t& particles, int num_particles, int rank )
{
    AoSoA_ht particles_h( "particles_h", num_particles );
    auto h_pos = Cabana::slice<Position>( particles_h );
    auto h_q   = Cabana::slice<Charge>( particles_h );

    std::mt19937 gen( 42 + rank );
    std::uniform_real_distribution<double> pos_dist( 0.0, 1.0 );
    std::uniform_real_distribution<double> q_dist( 0.1, 1.0 );

    for ( int i = 0; i < num_particles; i++ )
    {
        h_pos( i, 0 ) = pos_dist( gen );
        h_pos( i, 1 ) = pos_dist( gen );
        h_pos( i, 2 ) = pos_dist( gen );
        h_q( i, 0 )   = q_dist( gen );
    }

    particles.resize( num_particles );
    Cabana::deep_copy( particles, particles_h );
}

} // namespace DownwardSweepTest

//---------------------------------------------------------------------------//
/**
 * Verify that zero particle charges produce all-zero local coefficients and
 * all-zero particle potentials after the full upward + downward sweep.
 *
 * With zero charges P2M produces zero leaf multipoles; M2M propagates zeros
 * upward; M2L translating zero multipoles produces zero local increments;
 * L2L translating zero locals yields zero locals at children; L2P evaluating
 * a zero local returns zero potential. Any non-zero result indicates
 * uninitialized storage or an incorrect accumulation path.
 *
 * Checks:
 *   1. Every local coefficient in every cell is exactly zero.
 *   2. Every particle potential is exactly zero.
 */
void testZeroChargesGiveZeroLocalsAndPotential(
    int num_particles_per_rank, int ncrit, int max_depth,
    double tolerance, int replication_depth )
{
    using namespace DownwardSweepTest;

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    AoSoA_ht particles_h( "particles_h", num_particles_per_rank );
    {
        auto h_pos = Cabana::slice<Position>( particles_h );
        auto h_q   = Cabana::slice<Charge>( particles_h );

        std::mt19937 gen( 42 + rank );
        std::uniform_real_distribution<double> pos_dist( 0.0, 1.0 );

        for ( int i = 0; i < num_particles_per_rank; i++ )
        {
            h_pos( i, 0 ) = pos_dist( gen );
            h_pos( i, 1 ) = pos_dist( gen );
            h_pos( i, 2 ) = pos_dist( gen );
            h_q( i, 0 )   = 0.0;
        }
    }

    AoSoA_t particles( "particles", num_particles_per_rank );
    Cabana::deep_copy( particles, particles_h );

    auto positions = Cabana::slice<Position>( particles );
    auto charges   = Cabana::slice<Charge>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, tolerance, tolerance );
    builder.build( positions, num_particles_per_rank );

    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    partitioner.partition( builder, particles, num_particles_per_rank );
    int num_local = partitioner.num_local_particles();

    positions = Cabana::slice<Position>( particles );
    charges   = Cabana::slice<Charge>( particles );
    builder.build( positions, num_local );

    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> comm_plan( MPI_COMM_WORLD );
    comm_plan.build( builder.cells(), partitioner.ownership(),
                     partitioner.cell_owner_map(), replication_depth );

    UpwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> upward( MPI_COMM_WORLD );
    upward.setup( builder.cells(), partitioner.cell_owner_map(),
                  builder.particle_keys(), num_local );
    upward.execute( charges, positions, comm_plan );

    DownwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> downward( MPI_COMM_WORLD );
    downward.setup( upward, num_local );

    auto potential = downward.allocate_potential( num_local );
    auto gradient  = downward.allocate_gradient( num_local );
    Kokkos::deep_copy( potential, 0.0 );

    downward.execute( upward.multipoles(), positions, potential, gradient,
                      false, comm_plan );

    // All local coefficients must be zero
    auto h_L = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), downward.locals() );
    const int num_cells = static_cast<int>( h_L.extent( 0 ) );
    for ( int c = 0; c < num_cells; c++ )
        for ( int idx = 0; idx < Kernel::num_coeffs_per_cell; idx++ )
        {
            EXPECT_EQ( h_L( c, idx, 0 ).real(), 0.0 )
                << "Non-zero local real at cell " << c << " coeff " << idx;
            EXPECT_EQ( h_L( c, idx, 0 ).imag(), 0.0 )
                << "Non-zero local imag at cell " << c << " coeff " << idx;
        }

    // All particle potentials must be zero
    auto h_phi = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), potential );
    for ( int p = 0; p < num_local; p++ )
        EXPECT_EQ( h_phi( p, 0 ), 0.0 )
            << "Non-zero potential at particle " << p << " with zero charges";
}

//---------------------------------------------------------------------------//
/**
 * Verify that non-zero particle charges produce at least one non-zero local
 * coefficient and at least one non-zero particle potential.
 *
 * With strictly positive charges the upward sweep produces non-zero
 * multipoles. The M2L step must propagate these into non-zero local
 * contributions at cells with non-empty interaction lists. L2P must then
 * produce a non-zero potential at those cells' particles. An all-zero result
 * indicates the sweep is not processing any interaction-list entries or the
 * storage is not being read correctly.
 *
 * The check is global (MPI_Allreduce over ranks) so the test passes even
 * when a single rank happens to own no cells with non-trivial interaction
 * lists.
 *
 * Checks:
 *   1. The global maximum absolute local coefficient is > 0.
 *   2. The global maximum absolute particle potential is > 0.
 */
void testLocalsAndPotentialNonzeroAfterExecute(
    int num_particles_per_rank, int ncrit, int max_depth,
    double tolerance, int replication_depth )
{
    using namespace DownwardSweepTest;

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    AoSoA_t particles( "particles", num_particles_per_rank );
    generate_test_particles( particles, num_particles_per_rank, rank );

    auto positions = Cabana::slice<Position>( particles );
    auto charges   = Cabana::slice<Charge>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, tolerance, tolerance );
    builder.build( positions, num_particles_per_rank );

    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    partitioner.partition( builder, particles, num_particles_per_rank );
    int num_local = partitioner.num_local_particles();

    positions = Cabana::slice<Position>( particles );
    charges   = Cabana::slice<Charge>( particles );
    builder.build( positions, num_local );

    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> comm_plan( MPI_COMM_WORLD );
    comm_plan.build( builder.cells(), partitioner.ownership(),
                     partitioner.cell_owner_map(), replication_depth );

    UpwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> upward( MPI_COMM_WORLD );
    upward.setup( builder.cells(), partitioner.cell_owner_map(),
                  builder.particle_keys(), num_local );
    upward.execute( charges, positions, comm_plan );

    DownwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> downward( MPI_COMM_WORLD );
    downward.setup( upward, num_local );

    auto potential = downward.allocate_potential( num_local );
    auto gradient  = downward.allocate_gradient( num_local );
    Kokkos::deep_copy( potential, 0.0 );

    downward.execute( upward.multipoles(), positions, potential, gradient,
                      false, comm_plan );

    // Reduce max |local| across all ranks
    auto h_L = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), downward.locals() );
    const int num_cells = static_cast<int>( h_L.extent( 0 ) );
    double local_max_abs = 0.0;
    for ( int c = 0; c < num_cells; c++ )
        for ( int idx = 0; idx < Kernel::num_coeffs_per_cell; idx++ )
        {
            const auto v = h_L( c, idx, 0 );
            const double m =
                std::sqrt( v.real() * v.real() + v.imag() * v.imag() );
            if ( m > local_max_abs )
                local_max_abs = m;
        }

    double global_max_local = 0.0;
    MPI_Allreduce( &local_max_abs, &global_max_local, 1, MPI_DOUBLE, MPI_MAX,
                   MPI_COMM_WORLD );
    EXPECT_GT( global_max_local, 0.0 )
        << "All local coefficients are zero after sweep with non-zero charges";

    // Reduce max |potential| across all ranks
    auto h_phi = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), potential );
    double local_phi_max = 0.0;
    for ( int p = 0; p < num_local; p++ )
        if ( std::abs( h_phi( p, 0 ) ) > local_phi_max )
            local_phi_max = std::abs( h_phi( p, 0 ) );

    double global_phi_max = 0.0;
    MPI_Allreduce( &local_phi_max, &global_phi_max, 1, MPI_DOUBLE, MPI_MAX,
                   MPI_COMM_WORLD );
    EXPECT_GT( global_phi_max, 0.0 )
        << "All potentials are zero after sweep with non-zero charges";
}

//---------------------------------------------------------------------------//
/**
 * Verify that calling execute() twice on the same sweep objects and inputs
 * produces bit-identical local coefficients and particle potentials.
 *
 * execute() zeros _locals at the start of every call, so results must not
 * accumulate across invocations or depend on leftover state from prior runs.
 * The potential output view is zeroed by the caller before each call; L2P
 * uses += so a non-zero initial value would cause a mismatch.
 *
 * Checks:
 *   1. Real and imaginary parts of every local coefficient match exactly
 *      between the first and second execute() calls.
 *   2. Every particle potential value matches exactly between the two calls.
 */
void testIdempotentExecution(
    int num_particles_per_rank, int ncrit, int max_depth,
    double tolerance, int replication_depth )
{
    using namespace DownwardSweepTest;

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    AoSoA_t particles( "particles", num_particles_per_rank );
    generate_test_particles( particles, num_particles_per_rank, rank );

    auto positions = Cabana::slice<Position>( particles );
    auto charges   = Cabana::slice<Charge>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, tolerance, tolerance );
    builder.build( positions, num_particles_per_rank );

    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    partitioner.partition( builder, particles, num_particles_per_rank );
    int num_local = partitioner.num_local_particles();

    positions = Cabana::slice<Position>( particles );
    charges   = Cabana::slice<Charge>( particles );
    builder.build( positions, num_local );

    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> comm_plan( MPI_COMM_WORLD );
    comm_plan.build( builder.cells(), partitioner.ownership(),
                     partitioner.cell_owner_map(), replication_depth );

    UpwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> upward( MPI_COMM_WORLD );
    upward.setup( builder.cells(), partitioner.cell_owner_map(),
                  builder.particle_keys(), num_local );
    upward.execute( charges, positions, comm_plan );

    DownwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> downward( MPI_COMM_WORLD );
    downward.setup( upward, num_local );

    // First execute — snapshot locals and potential
    auto potential1 = downward.allocate_potential( num_local );
    auto gradient1  = downward.allocate_gradient( num_local );
    Kokkos::deep_copy( potential1, 0.0 );
    downward.execute( upward.multipoles(), positions, potential1, gradient1,
                      false, comm_plan );

    auto h_L1   = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), downward.locals() );
    auto h_phi1 = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), potential1 );

    // Second execute — must give bit-identical results
    auto potential2 = downward.allocate_potential( num_local );
    auto gradient2  = downward.allocate_gradient( num_local );
    Kokkos::deep_copy( potential2, 0.0 );
    downward.execute( upward.multipoles(), positions, potential2, gradient2,
                      false, comm_plan );

    auto h_L2   = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), downward.locals() );
    auto h_phi2 = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), potential2 );

    const int num_cells = static_cast<int>( h_L1.extent( 0 ) );
    for ( int c = 0; c < num_cells; c++ )
        for ( int idx = 0; idx < Kernel::num_coeffs_per_cell; idx++ )
        {
            EXPECT_EQ( h_L1( c, idx, 0 ).real(), h_L2( c, idx, 0 ).real() )
                << "Local real mismatch at cell " << c << " coeff " << idx;
            EXPECT_EQ( h_L1( c, idx, 0 ).imag(), h_L2( c, idx, 0 ).imag() )
                << "Local imag mismatch at cell " << c << " coeff " << idx;
        }

    for ( int p = 0; p < num_local; p++ )
        EXPECT_EQ( h_phi1( p, 0 ), h_phi2( p, 0 ) )
            << "Potential mismatch at particle " << p
            << " between first and second execute()";
}

//---------------------------------------------------------------------------//
/**
 * Verify that the FMM far-field potential at target particles approximates
 * the direct Coulomb sum from a well-separated source cluster (single rank).
 *
 * Setup:
 *   Sources: particles in [0.0, 0.2]^3, charges in [0.5, 1.5].
 *   Targets: particles in [0.8, 1.0]^3, charges = 0 (no P2M contribution).
 *
 * Because targets carry zero charge, only source cells produce non-zero
 * multipoles. At depth 2 the source cluster is entirely within one octree
 * cell and the target cluster is entirely within the diagonally opposite
 * cell; these two cells are in each other's interaction list (their depth-1
 * parents are adjacent). M2L translates the source multipole to the target
 * cell local; L2L propagates the local to the target leaves; L2P evaluates
 * it at each target particle.
 *
 * The direct reference for target particle j is:
 *   phi_ref(j) = sum_{i = sources} q_i / |r_j - r_i|
 *
 * Checks (single-rank runs only):
 *   1. At least one target particle exists in the post-partition data.
 *   2. Max relative error over all target particles is below error_tol.
 */
void testL2PApproximatesDirectSumSingleRank(
    int num_sources, int num_targets, int ncrit, int max_depth,
    double tolerance, int replication_depth, double error_tol )
{
    using namespace DownwardSweepTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    if ( nprocs != 1 )
        return;

    const int N = num_sources + num_targets;

    AoSoA_ht particles_h( "particles_h", N );
    {
        auto h_pos = Cabana::slice<Position>( particles_h );
        auto h_q   = Cabana::slice<Charge>( particles_h );

        std::mt19937 gen( 777 );
        std::uniform_real_distribution<double> src_dist( 0.0, 0.2 );
        std::uniform_real_distribution<double> tgt_dist( 0.8, 1.0 );
        std::uniform_real_distribution<double> q_dist( 0.5, 1.5 );

        for ( int i = 0; i < num_sources; i++ )
        {
            h_pos( i, 0 ) = src_dist( gen );
            h_pos( i, 1 ) = src_dist( gen );
            h_pos( i, 2 ) = src_dist( gen );
            h_q( i, 0 )   = q_dist( gen );
        }
        for ( int i = num_sources; i < N; i++ )
        {
            h_pos( i, 0 ) = tgt_dist( gen );
            h_pos( i, 1 ) = tgt_dist( gen );
            h_pos( i, 2 ) = tgt_dist( gen );
            h_q( i, 0 )   = 0.0;
        }
    }

    AoSoA_t particles( "particles", N );
    Cabana::deep_copy( particles, particles_h );

    auto positions = Cabana::slice<Position>( particles );
    auto charges   = Cabana::slice<Charge>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, tolerance, tolerance );
    builder.build( positions, N );

    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    partitioner.partition( builder, particles, N );
    int num_local = partitioner.num_local_particles();

    positions = Cabana::slice<Position>( particles );
    charges   = Cabana::slice<Charge>( particles );
    builder.build( positions, num_local );

    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> comm_plan( MPI_COMM_WORLD );
    comm_plan.build( builder.cells(), partitioner.ownership(),
                     partitioner.cell_owner_map(), replication_depth );

    UpwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> upward( MPI_COMM_WORLD );
    upward.setup( builder.cells(), partitioner.cell_owner_map(),
                  builder.particle_keys(), num_local );
    upward.execute( charges, positions, comm_plan );

    DownwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> downward( MPI_COMM_WORLD );
    downward.setup( upward, num_local );

    auto potential = downward.allocate_potential( num_local );
    auto gradient  = downward.allocate_gradient( num_local );
    Kokkos::deep_copy( potential, 0.0 );

    downward.execute( upward.multipoles(), positions, potential, gradient,
                      false, comm_plan );

    // Copy positions, charges, and potential to host for comparison
    Kokkos::View<double* [3], TEST_MEMSPACE> d_pos( "d_pos", num_local );
    Kokkos::View<double*, TEST_MEMSPACE> d_crg( "d_crg", num_local );
    Kokkos::parallel_for(
        "CopyForRef",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_local ),
        KOKKOS_LAMBDA( int i ) {
            d_pos( i, 0 ) = positions( i, 0 );
            d_pos( i, 1 ) = positions( i, 1 );
            d_pos( i, 2 ) = positions( i, 2 );
            d_crg( i )    = charges( i, 0 );
        } );
    Kokkos::fence();

    auto h_pos = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_pos );
    auto h_crg = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_crg );
    auto h_phi = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), potential );

    // For each target particle (charge == 0), compare FMM potential to
    // direct sum over all source particles (charge != 0).
    double max_rel_err = 0.0;
    int n_checked = 0;
    for ( int p = 0; p < num_local; p++ )
    {
        if ( h_crg( p ) != 0.0 )
            continue;

        const double tx = h_pos( p, 0 );
        const double ty = h_pos( p, 1 );
        const double tz = h_pos( p, 2 );

        double ref = 0.0;
        for ( int s = 0; s < num_local; s++ )
        {
            if ( h_crg( s ) == 0.0 )
                continue;
            const double dx   = tx - h_pos( s, 0 );
            const double dy   = ty - h_pos( s, 1 );
            const double dz   = tz - h_pos( s, 2 );
            const double dist = std::sqrt( dx * dx + dy * dy + dz * dz );
            if ( dist > 0.0 )
                ref += h_crg( s ) / dist;
        }

        const double fmm = h_phi( p, 0 );
        const double rel_err =
            ( std::abs( ref ) > 1e-14 )
                ? std::abs( fmm - ref ) / std::abs( ref )
                : std::abs( fmm - ref );

        if ( rel_err > max_rel_err )
            max_rel_err = rel_err;
        n_checked++;
    }

    EXPECT_GT( n_checked, 0 )
        << "No target particles found; check two-cluster placement";
    EXPECT_LT( max_rel_err, error_tol )
        << "FMM L2P deviates from direct Coulomb sum; "
           "max relative error = " << max_rel_err;
}

//---------------------------------------------------------------------------//
/**
 * Verify that the FMM far-field potential approximates the direct Coulomb sum
 * for the two-cluster setup on multiple MPI ranks.
 *
 * The geometry and physics are the same as testL2PApproximatesDirectSumSingleRank.
 * This test is silently skipped on a single rank; the single-rank case is
 * covered by testL2PApproximatesDirectSumSingleRank.
 *
 * After partitioning, exchange_multipoles_for_m2l communicates source cell
 * multipoles to the ranks that own target cells. Each rank evaluates L2P at
 * its local target particles. MPI_Gatherv collects all particle data on
 * rank 0, which computes the global reference and checks accuracy.
 *
 * Checks (on rank 0 only; all other ranks participate in the gather):
 *   1. At least one target particle exists in the gathered data.
 *   2. Max relative error over all target particles is below error_tol.
 */
void testL2PApproximatesDirectSumMultiRank(
    int num_sources, int num_targets, int ncrit, int max_depth,
    double tolerance, int replication_depth, double error_tol )
{
    using namespace DownwardSweepTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    if ( nprocs == 1 )
        return;

    const int N = num_sources + num_targets;

    // Each rank generates a distinct set of particles in the two-cluster
    // geometry so that the total source/target populations scale with nprocs.
    AoSoA_ht particles_h( "particles_h", N );
    {
        auto h_pos = Cabana::slice<Position>( particles_h );
        auto h_q   = Cabana::slice<Charge>( particles_h );

        std::mt19937 gen( 777 + rank );
        std::uniform_real_distribution<double> src_dist( 0.0, 0.2 );
        std::uniform_real_distribution<double> tgt_dist( 0.8, 1.0 );
        std::uniform_real_distribution<double> q_dist( 0.5, 1.5 );

        for ( int i = 0; i < num_sources; i++ )
        {
            h_pos( i, 0 ) = src_dist( gen );
            h_pos( i, 1 ) = src_dist( gen );
            h_pos( i, 2 ) = src_dist( gen );
            h_q( i, 0 )   = q_dist( gen );
        }
        for ( int i = num_sources; i < N; i++ )
        {
            h_pos( i, 0 ) = tgt_dist( gen );
            h_pos( i, 1 ) = tgt_dist( gen );
            h_pos( i, 2 ) = tgt_dist( gen );
            h_q( i, 0 )   = 0.0;
        }
    }

    AoSoA_t particles( "particles", N );
    Cabana::deep_copy( particles, particles_h );

    auto positions = Cabana::slice<Position>( particles );
    auto charges   = Cabana::slice<Charge>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, tolerance, tolerance );
    builder.build( positions, N );

    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    partitioner.partition( builder, particles, N );
    int num_local = partitioner.num_local_particles();

    positions = Cabana::slice<Position>( particles );
    charges   = Cabana::slice<Charge>( particles );
    builder.build( positions, num_local );

    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> comm_plan( MPI_COMM_WORLD );
    comm_plan.build( builder.cells(), partitioner.ownership(),
                     partitioner.cell_owner_map(), replication_depth );

    UpwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> upward( MPI_COMM_WORLD );
    upward.setup( builder.cells(), partitioner.cell_owner_map(),
                  builder.particle_keys(), num_local );
    upward.execute( charges, positions, comm_plan );

    DownwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> downward( MPI_COMM_WORLD );
    downward.setup( upward, num_local );

    auto potential = downward.allocate_potential( num_local );
    auto gradient  = downward.allocate_gradient( num_local );
    Kokkos::deep_copy( potential, 0.0 );

    downward.execute( upward.multipoles(), positions, potential, gradient,
                      false, comm_plan );

    // Copy local data to host
    Kokkos::View<double* [3], TEST_MEMSPACE> d_pos( "d_pos", num_local );
    Kokkos::View<double*, TEST_MEMSPACE> d_crg( "d_crg", num_local );
    Kokkos::parallel_for(
        "CopyForRef",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_local ),
        KOKKOS_LAMBDA( int i ) {
            d_pos( i, 0 ) = positions( i, 0 );
            d_pos( i, 1 ) = positions( i, 1 );
            d_pos( i, 2 ) = positions( i, 2 );
            d_crg( i )    = charges( i, 0 );
        } );
    Kokkos::fence();

    auto h_pos_l = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_pos );
    auto h_crg_l = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_crg );
    auto h_phi_l = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), potential );

    // Pack each particle as (x, y, z, charge, potential) for gathering
    std::vector<double> local_buf( 5 * num_local );
    for ( int i = 0; i < num_local; i++ )
    {
        local_buf[5 * i + 0] = h_pos_l( i, 0 );
        local_buf[5 * i + 1] = h_pos_l( i, 1 );
        local_buf[5 * i + 2] = h_pos_l( i, 2 );
        local_buf[5 * i + 3] = h_crg_l( i );
        local_buf[5 * i + 4] = h_phi_l( i, 0 );
    }

    std::vector<int> all_num_local( nprocs, 0 );
    MPI_Gather( &num_local, 1, MPI_INT,
                all_num_local.data(), 1, MPI_INT, 0, MPI_COMM_WORLD );

    int total_particles = 0;
    std::vector<int> counts( nprocs, 0 ), displs( nprocs, 0 );
    std::vector<double> gathered;

    if ( rank == 0 )
    {
        for ( int r = 0; r < nprocs; r++ )
        {
            counts[r] = 5 * all_num_local[r];
            total_particles += all_num_local[r];
        }
        for ( int r = 1; r < nprocs; r++ )
            displs[r] = displs[r - 1] + counts[r - 1];
        gathered.resize( 5 * total_particles );
    }

    MPI_Gatherv( local_buf.data(), 5 * num_local, MPI_DOUBLE,
                 gathered.data(), counts.data(), displs.data(),
                 MPI_DOUBLE, 0, MPI_COMM_WORLD );

    if ( rank == 0 )
    {
        double max_rel_err = 0.0;
        int n_checked = 0;

        for ( int p = 0; p < total_particles; p++ )
        {
            if ( gathered[5 * p + 3] != 0.0 )
                continue; // source particle — skip

            const double tx = gathered[5 * p + 0];
            const double ty = gathered[5 * p + 1];
            const double tz = gathered[5 * p + 2];

            double ref = 0.0;
            for ( int s = 0; s < total_particles; s++ )
            {
                if ( gathered[5 * s + 3] == 0.0 )
                    continue;
                const double dx   = tx - gathered[5 * s + 0];
                const double dy   = ty - gathered[5 * s + 1];
                const double dz   = tz - gathered[5 * s + 2];
                const double dist =
                    std::sqrt( dx * dx + dy * dy + dz * dz );
                if ( dist > 0.0 )
                    ref += gathered[5 * s + 3] / dist;
            }

            const double fmm = gathered[5 * p + 4];
            const double rel_err =
                ( std::abs( ref ) > 1e-14 )
                    ? std::abs( fmm - ref ) / std::abs( ref )
                    : std::abs( fmm - ref );

            if ( rel_err > max_rel_err )
                max_rel_err = rel_err;
            n_checked++;
        }

        EXPECT_GT( n_checked, 0 )
            << "No target particles gathered on rank 0";
        EXPECT_LT( max_rel_err, error_tol )
            << "FMM L2P multi-rank deviates from direct Coulomb sum; "
               "max relative error = " << max_rel_err;
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//

TEST( DownwardSweep, testZeroChargesGiveZeroLocalsAndPotentialBasic )
{
    testZeroChargesGiveZeroLocalsAndPotential( 1000, 32, 6, 0.1, 2 );
}

TEST( DownwardSweep, testZeroChargesGiveZeroLocalsAndPotentialSmall )
{
    testZeroChargesGiveZeroLocalsAndPotential( 200, 16, 4, 0.1, 1 );
}

TEST( DownwardSweep, testLocalsAndPotentialNonzeroAfterExecuteBasic )
{
    testLocalsAndPotentialNonzeroAfterExecute( 1000, 32, 6, 0.1, 2 );
}

TEST( DownwardSweep, testLocalsAndPotentialNonzeroAfterExecuteSmall )
{
    testLocalsAndPotentialNonzeroAfterExecute( 200, 16, 4, 0.1, 1 );
}

TEST( DownwardSweep, testIdempotentExecutionBasic )
{
    testIdempotentExecution( 1000, 32, 6, 0.1, 2 );
}

TEST( DownwardSweep, testIdempotentExecutionSmall )
{
    testIdempotentExecution( 200, 16, 4, 0.1, 1 );
}

TEST( DownwardSweep, testL2PApproximatesDirectSumSingleRankBasic )
{
    testL2PApproximatesDirectSumSingleRank( 100, 100, 8, 5, 0.1, 2, 1e-3 );
}

TEST( DownwardSweep, testL2PApproximatesDirectSumSingleRankSmall )
{
    testL2PApproximatesDirectSumSingleRank( 40, 40, 4, 4, 0.1, 1, 1e-3 );
}

TEST( DownwardSweep, testL2PApproximatesDirectSumMultiRankBasic )
{
    testL2PApproximatesDirectSumMultiRank( 100, 100, 8, 5, 0.1, 2, 1e-3 );
}

TEST( DownwardSweep, testL2PApproximatesDirectSumMultiRankSmall )
{
    testL2PApproximatesDirectSumMultiRank( 40, 40, 4, 4, 0.1, 1, 1e-3 );
}

//---------------------------------------------------------------------------//

} // end namespace Test
