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

#include "Canopy_CartesianTaylorBasis.hpp"
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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace Test
{
//---------------------------------------------------------------------------//

using namespace Canopy;

namespace DownwardSweepTest
{

// Read CANOPY_MAC_THETA env var to allow rerunning tests at different MAC
// values without rebuilding. Defaults to 0.5 (Canopy default).
inline double get_test_mac_theta()
{
    if ( const char* s = std::getenv( "CANOPY_MAC_THETA" ) )
        return std::atof( s );
    return 0.5;
}

enum FieldIdx
{
    Position = 0,
    Charge = 1
};

static constexpr int P_ORDER = 6;
using Kernel = LaplaceKernel<double, P_ORDER>;

using DataTypes =
    Cabana::MemberTypes<double[3], double[Kernel::num_components]>;
using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
using AoSoA_ht = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;

// Generate particles with random positions in [0, 1)^3 and strictly positive
// charges in [0.1, 1.0] so the monopole moment is guaranteed non-zero.
void generate_test_particles( AoSoA_t& particles, int num_particles, int rank )
{
    AoSoA_ht particles_h( "particles_h", num_particles );
    auto h_pos = Cabana::slice<Position>( particles_h );
    auto h_q = Cabana::slice<Charge>( particles_h );

    std::mt19937 gen( 42 + rank );
    std::uniform_real_distribution<double> pos_dist( 0.0, 1.0 );
    std::uniform_real_distribution<double> q_dist( 0.1, 1.0 );

    for ( int i = 0; i < num_particles; i++ )
    {
        h_pos( i, 0 ) = pos_dist( gen );
        h_pos( i, 1 ) = pos_dist( gen );
        h_pos( i, 2 ) = pos_dist( gen );
        h_q( i, 0 ) = q_dist( gen );
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
void testZeroChargesGiveZeroLocalsAndPotential( int num_particles_per_rank,
                                                int ncrit, int max_depth,
                                                double tolerance,
                                                int replication_depth )
{
    using namespace DownwardSweepTest;

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    AoSoA_ht particles_h( "particles_h", num_particles_per_rank );
    {
        auto h_pos = Cabana::slice<Position>( particles_h );
        auto h_q = Cabana::slice<Charge>( particles_h );

        std::mt19937 gen( 42 + rank );
        std::uniform_real_distribution<double> pos_dist( 0.0, 1.0 );

        for ( int i = 0; i < num_particles_per_rank; i++ )
        {
            h_pos( i, 0 ) = pos_dist( gen );
            h_pos( i, 1 ) = pos_dist( gen );
            h_pos( i, 2 ) = pos_dist( gen );
            h_q( i, 0 ) = 0.0;
        }
    }

    AoSoA_t particles( "particles", num_particles_per_rank );
    Cabana::deep_copy( particles, particles_h );

    auto positions = Cabana::slice<Position>( particles );
    auto charges = Cabana::slice<Charge>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );
    builder.build( positions, num_particles_per_rank );

    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    partitioner.partition( builder, particles, num_particles_per_rank );
    int num_local = partitioner.num_local_particles();

    positions = Cabana::slice<Position>( particles );
    charges = Cabana::slice<Charge>( particles );
    builder.build( positions, num_local );

    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> comm_plan(
        MPI_COMM_WORLD, get_test_mac_theta() );
    comm_plan.build( builder.cells(), partitioner.ownership(),
                     partitioner.cell_owner_map(), replication_depth );

    UpwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> upward( MPI_COMM_WORLD );
    upward.setup( builder.cells(), partitioner.cell_owner_map(),
                  builder.particle_keys(), num_local );
    upward.execute( charges, positions, comm_plan );

    DownwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> downward(
        MPI_COMM_WORLD );
    downward.setup( upward, num_local );

    auto potential = downward.allocate_potential( num_local );
    auto gradient = downward.allocate_gradient( num_local );
    Kokkos::deep_copy( potential, 0.0 );

    downward.execute( upward.multipoles(), positions, potential, gradient,
                      false, comm_plan );

    // All local coefficients must be zero
    auto h_L = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                    downward.locals() );
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
    auto h_phi =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), potential );
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
void testLocalsAndPotentialNonzeroAfterExecute( int num_particles_per_rank,
                                                int ncrit, int max_depth,
                                                double tolerance,
                                                int replication_depth )
{
    using namespace DownwardSweepTest;

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    AoSoA_t particles( "particles", num_particles_per_rank );
    generate_test_particles( particles, num_particles_per_rank, rank );

    auto positions = Cabana::slice<Position>( particles );
    auto charges = Cabana::slice<Charge>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );
    builder.build( positions, num_particles_per_rank );

    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    partitioner.partition( builder, particles, num_particles_per_rank );
    int num_local = partitioner.num_local_particles();

    positions = Cabana::slice<Position>( particles );
    charges = Cabana::slice<Charge>( particles );
    builder.build( positions, num_local );

    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> comm_plan(
        MPI_COMM_WORLD, get_test_mac_theta() );
    comm_plan.build( builder.cells(), partitioner.ownership(),
                     partitioner.cell_owner_map(), replication_depth );

    UpwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> upward( MPI_COMM_WORLD );
    upward.setup( builder.cells(), partitioner.cell_owner_map(),
                  builder.particle_keys(), num_local );
    upward.execute( charges, positions, comm_plan );

    DownwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> downward(
        MPI_COMM_WORLD );
    downward.setup( upward, num_local );

    auto potential = downward.allocate_potential( num_local );
    auto gradient = downward.allocate_gradient( num_local );
    Kokkos::deep_copy( potential, 0.0 );

    downward.execute( upward.multipoles(), positions, potential, gradient,
                      false, comm_plan );

    // Reduce max |local| across all ranks
    auto h_L = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                    downward.locals() );
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
    auto h_phi =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), potential );
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
void testIdempotentExecution( int num_particles_per_rank, int ncrit,
                              int max_depth, double tolerance,
                              int replication_depth )
{
    using namespace DownwardSweepTest;

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    AoSoA_t particles( "particles", num_particles_per_rank );
    generate_test_particles( particles, num_particles_per_rank, rank );

    auto positions = Cabana::slice<Position>( particles );
    auto charges = Cabana::slice<Charge>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );
    builder.build( positions, num_particles_per_rank );

    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    partitioner.partition( builder, particles, num_particles_per_rank );
    int num_local = partitioner.num_local_particles();

    positions = Cabana::slice<Position>( particles );
    charges = Cabana::slice<Charge>( particles );
    builder.build( positions, num_local );

    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> comm_plan(
        MPI_COMM_WORLD, get_test_mac_theta() );
    comm_plan.build( builder.cells(), partitioner.ownership(),
                     partitioner.cell_owner_map(), replication_depth );

    UpwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> upward( MPI_COMM_WORLD );
    upward.setup( builder.cells(), partitioner.cell_owner_map(),
                  builder.particle_keys(), num_local );
    upward.execute( charges, positions, comm_plan );

    DownwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> downward(
        MPI_COMM_WORLD );
    downward.setup( upward, num_local );

    // First execute — snapshot locals and potential
    auto potential1 = downward.allocate_potential( num_local );
    auto gradient1 = downward.allocate_gradient( num_local );
    Kokkos::deep_copy( potential1, 0.0 );
    downward.execute( upward.multipoles(), positions, potential1, gradient1,
                      false, comm_plan );

    auto h_L1 = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                     downward.locals() );
    auto h_phi1 =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), potential1 );

    // Second execute — must give bit-identical results
    auto potential2 = downward.allocate_potential( num_local );
    auto gradient2 = downward.allocate_gradient( num_local );
    Kokkos::deep_copy( potential2, 0.0 );
    downward.execute( upward.multipoles(), positions, potential2, gradient2,
                      false, comm_plan );

    auto h_L2 = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                     downward.locals() );
    auto h_phi2 =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), potential2 );

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
void testL2PApproximatesDirectSumSingleRank( int num_sources, int num_targets,
                                             int ncrit, int max_depth,
                                             double tolerance,
                                             int replication_depth,
                                             double error_tol )
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
        auto h_q = Cabana::slice<Charge>( particles_h );

        std::mt19937 gen( 777 );
        std::uniform_real_distribution<double> src_dist( 0.0, 0.2 );
        std::uniform_real_distribution<double> tgt_dist( 0.8, 1.0 );
        std::uniform_real_distribution<double> q_dist( 0.5, 1.5 );

        for ( int i = 0; i < num_sources; i++ )
        {
            h_pos( i, 0 ) = src_dist( gen );
            h_pos( i, 1 ) = src_dist( gen );
            h_pos( i, 2 ) = src_dist( gen );
            h_q( i, 0 ) = q_dist( gen );
        }
        for ( int i = num_sources; i < N; i++ )
        {
            h_pos( i, 0 ) = tgt_dist( gen );
            h_pos( i, 1 ) = tgt_dist( gen );
            h_pos( i, 2 ) = tgt_dist( gen );
            h_q( i, 0 ) = 0.0;
        }
    }

    AoSoA_t particles( "particles", N );
    Cabana::deep_copy( particles, particles_h );

    auto positions = Cabana::slice<Position>( particles );
    auto charges = Cabana::slice<Charge>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );
    builder.build( positions, N );

    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    partitioner.partition( builder, particles, N );
    int num_local = partitioner.num_local_particles();

    positions = Cabana::slice<Position>( particles );
    charges = Cabana::slice<Charge>( particles );
    builder.build( positions, num_local );

    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> comm_plan(
        MPI_COMM_WORLD, get_test_mac_theta() );
    comm_plan.build( builder.cells(), partitioner.ownership(),
                     partitioner.cell_owner_map(), replication_depth );

    UpwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> upward( MPI_COMM_WORLD );
    upward.setup( builder.cells(), partitioner.cell_owner_map(),
                  builder.particle_keys(), num_local );
    upward.execute( charges, positions, comm_plan );

    DownwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> downward(
        MPI_COMM_WORLD );
    downward.setup( upward, num_local );

    auto potential = downward.allocate_potential( num_local );
    auto gradient = downward.allocate_gradient( num_local );
    Kokkos::deep_copy( potential, 0.0 );

    downward.execute( upward.multipoles(), positions, potential, gradient,
                      false, comm_plan );

    // Copy positions, charges, and potential to host for comparison
    Kokkos::View<double* [3], TEST_MEMSPACE> d_pos( "d_pos", num_local );
    Kokkos::View<double*, TEST_MEMSPACE> d_crg( "d_crg", num_local );
    Kokkos::parallel_for(
        "CopyForRef", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_local ),
        KOKKOS_LAMBDA( int i ) {
            d_pos( i, 0 ) = positions( i, 0 );
            d_pos( i, 1 ) = positions( i, 1 );
            d_pos( i, 2 ) = positions( i, 2 );
            d_crg( i ) = charges( i, 0 );
        } );
    Kokkos::fence();

    auto h_pos =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_pos );
    auto h_crg =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_crg );
    auto h_phi =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), potential );

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
            const double dx = tx - h_pos( s, 0 );
            const double dy = ty - h_pos( s, 1 );
            const double dz = tz - h_pos( s, 2 );
            const double dist = std::sqrt( dx * dx + dy * dy + dz * dz );
            if ( dist > 0.0 )
                ref += h_crg( s ) / dist;
        }

        const double fmm = h_phi( p, 0 );
        const double rel_err = ( std::abs( ref ) > 1e-14 )
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
           "max relative error = "
        << max_rel_err;
}

//---------------------------------------------------------------------------//
/**
 * Verify that the FMM far-field potential approximates the direct Coulomb sum
 * for the two-cluster setup on multiple MPI ranks.
 *
 * The geometry and physics are the same as
 * testL2PApproximatesDirectSumSingleRank. This test is silently skipped on a
 * single rank; the single-rank case is covered by
 * testL2PApproximatesDirectSumSingleRank.
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
void testL2PApproximatesDirectSumMultiRank( int num_sources, int num_targets,
                                            int ncrit, int max_depth,
                                            double tolerance,
                                            int replication_depth,
                                            double error_tol )
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
        auto h_q = Cabana::slice<Charge>( particles_h );

        std::mt19937 gen( 777 + rank );
        std::uniform_real_distribution<double> src_dist( 0.0, 0.2 );
        std::uniform_real_distribution<double> tgt_dist( 0.8, 1.0 );
        std::uniform_real_distribution<double> q_dist( 0.5, 1.5 );

        for ( int i = 0; i < num_sources; i++ )
        {
            h_pos( i, 0 ) = src_dist( gen );
            h_pos( i, 1 ) = src_dist( gen );
            h_pos( i, 2 ) = src_dist( gen );
            h_q( i, 0 ) = q_dist( gen );
        }
        for ( int i = num_sources; i < N; i++ )
        {
            h_pos( i, 0 ) = tgt_dist( gen );
            h_pos( i, 1 ) = tgt_dist( gen );
            h_pos( i, 2 ) = tgt_dist( gen );
            h_q( i, 0 ) = 0.0;
        }
    }

    AoSoA_t particles( "particles", N );
    Cabana::deep_copy( particles, particles_h );

    auto positions = Cabana::slice<Position>( particles );
    auto charges = Cabana::slice<Charge>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );
    builder.build( positions, N );

    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    partitioner.partition( builder, particles, N );
    int num_local = partitioner.num_local_particles();

    positions = Cabana::slice<Position>( particles );
    charges = Cabana::slice<Charge>( particles );
    builder.build( positions, num_local );

    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> comm_plan(
        MPI_COMM_WORLD, get_test_mac_theta() );
    comm_plan.build( builder.cells(), partitioner.ownership(),
                     partitioner.cell_owner_map(), replication_depth );

    UpwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> upward( MPI_COMM_WORLD );
    upward.setup( builder.cells(), partitioner.cell_owner_map(),
                  builder.particle_keys(), num_local );
    upward.execute( charges, positions, comm_plan );

    DownwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> downward(
        MPI_COMM_WORLD );
    downward.setup( upward, num_local );

    auto potential = downward.allocate_potential( num_local );
    auto gradient = downward.allocate_gradient( num_local );
    Kokkos::deep_copy( potential, 0.0 );

    downward.execute( upward.multipoles(), positions, potential, gradient,
                      false, comm_plan );

    // Copy local data to host
    Kokkos::View<double* [3], TEST_MEMSPACE> d_pos( "d_pos", num_local );
    Kokkos::View<double*, TEST_MEMSPACE> d_crg( "d_crg", num_local );
    Kokkos::parallel_for(
        "CopyForRef", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_local ),
        KOKKOS_LAMBDA( int i ) {
            d_pos( i, 0 ) = positions( i, 0 );
            d_pos( i, 1 ) = positions( i, 1 );
            d_pos( i, 2 ) = positions( i, 2 );
            d_crg( i ) = charges( i, 0 );
        } );
    Kokkos::fence();

    auto h_pos_l =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_pos );
    auto h_crg_l =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_crg );
    auto h_phi_l =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), potential );

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
    MPI_Gather( &num_local, 1, MPI_INT, all_num_local.data(), 1, MPI_INT, 0,
                MPI_COMM_WORLD );

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

    MPI_Gatherv( local_buf.data(), 5 * num_local, MPI_DOUBLE, gathered.data(),
                 counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD );

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
                const double dx = tx - gathered[5 * s + 0];
                const double dy = ty - gathered[5 * s + 1];
                const double dz = tz - gathered[5 * s + 2];
                const double dist = std::sqrt( dx * dx + dy * dy + dz * dz );
                if ( dist > 0.0 )
                    ref += gathered[5 * s + 3] / dist;
            }

            const double fmm = gathered[5 * p + 4];
            const double rel_err = ( std::abs( ref ) > 1e-14 )
                                       ? std::abs( fmm - ref ) / std::abs( ref )
                                       : std::abs( fmm - ref );

            if ( rel_err > max_rel_err )
                max_rel_err = rel_err;
            n_checked++;
        }

        EXPECT_GT( n_checked, 0 ) << "No target particles gathered on rank 0";
        EXPECT_LT( max_rel_err, error_tol )
            << "FMM L2P multi-rank deviates from direct Coulomb sum; "
               "max relative error = "
            << max_rel_err;
    }
}

//---------------------------------------------------------------------------//
/**
 * Verify two structural invariants of the M2L interaction lists.
 *
 * In Greengard's adaptive FMM, for every cell C the interaction list (List 2)
 * consists of children of C's parent's neighbors that are not neighbors of C
 * itself. Two consequences must hold for any pair of cells (A, B) that this
 * rank can observe in `m2l_plan().interaction_lists`:
 *
 *   (a) **Same-depth symmetry.** If A and B are at the same depth, then
 *       B is in A's interaction list iff A is in B's interaction list.
 *       The relation "same-depth, parents adjacent, cells not adjacent" is
 *       symmetric, so any asymmetry indicates a neighbor-finding bug.
 *
 *   (b) **M2L / P2P disjointness.** If A is a leaf and B is in A's M2L
 *       interaction list, then B (or any descendant of B) cannot appear
 *       in A's P2P neighbor list. The two lists partition the far/near
 *       interactions; overlap means the same source-target pair is being
 *       evaluated twice.
 *
 * These invariants catch the failure mode of the previous probe-point
 * neighbor finder: probes at ±2*hw misidentify adjacency on adaptive
 * trees, producing asymmetric "B in A's list but not A in B's list"
 * pairs and overlapping near/far classification.
 *
 * This test runs single-rank only so all cells are visible to the checks.
 */
void testM2LListInvariants( int num_particles, int ncrit, int max_depth,
                            double tolerance, int replication_depth )
{
    using namespace DownwardSweepTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    if ( nprocs != 1 )
        return;

    AoSoA_t particles( "particles", num_particles );
    generate_test_particles( particles, num_particles, rank );

    auto positions = Cabana::slice<Position>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );
    builder.build( positions, num_particles );

    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    partitioner.partition( builder, particles, num_particles );
    int num_local = partitioner.num_local_particles();

    positions = Cabana::slice<Position>( particles );
    builder.build( positions, num_local );

    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> comm_plan(
        MPI_COMM_WORLD, get_test_mac_theta() );
    comm_plan.build( builder.cells(), partitioner.ownership(),
                     partitioner.cell_owner_map(), replication_depth );

    const auto& ilists = comm_plan.m2l_plan().interaction_lists;
    const auto& p2p_lists = comm_plan.p2p_plan().neighbor_lists;

    // Build a depth lookup from the cell list.
    std::unordered_map<MortonKey, int> depth_of;
    std::unordered_map<MortonKey, bool> is_leaf_of;
    for ( const auto& ci : builder.cells() )
    {
        depth_of[ci.key] = ci.depth;
        is_leaf_of[ci.key] = ci.is_leaf;
    }

    // (a) Same-depth symmetry of m2l interaction lists.
    int n_same_depth_pairs = 0;
    int n_asymmetric = 0;
    for ( const auto& kv : ilists )
    {
        MortonKey A = kv.first;
        for ( const auto& [B, b_idx] : kv.second )
        {
            (void)b_idx;
            auto da = depth_of.find( A );
            auto db = depth_of.find( B );
            if ( da == depth_of.end() || db == depth_of.end() )
                continue;
            if ( da->second != db->second )
                continue;

            n_same_depth_pairs++;
            auto it = ilists.find( B );
            bool reciprocal = false;
            if ( it != ilists.end() )
            {
                for ( const auto& [k, k_idx] : it->second )
                {
                    (void)k_idx;
                    if ( k == A )
                    {
                        reciprocal = true;
                        break;
                    }
                }
            }
            if ( !reciprocal )
            {
                n_asymmetric++;
                EXPECT_TRUE( false )
                    << "M2L list asymmetry: cell " << A << " has " << B
                    << " in its list at same depth " << da->second
                    << ", but reciprocal is missing";
                if ( n_asymmetric >= 5 )
                    break;
            }
        }
        if ( n_asymmetric >= 5 )
            break;
    }
    EXPECT_GT( n_same_depth_pairs, 0 )
        << "No same-depth M2L pairs observed; tree may be too shallow";

    // (b) M2L / P2P disjointness for leaves.
    int n_overlaps = 0;
    for ( const auto& kv : ilists )
    {
        MortonKey A = kv.first;
        auto la_it = is_leaf_of.find( A );
        if ( la_it == is_leaf_of.end() || !la_it->second )
            continue;

        auto pp_it = p2p_lists.find( A );
        if ( pp_it == p2p_lists.end() )
            continue;
        std::unordered_set<MortonKey> p2p_set( pp_it->second.begin(),
                                               pp_it->second.end() );
        for ( const auto& [B, b_idx] : kv.second )
        {
            (void)b_idx;
            if ( p2p_set.count( B ) )
            {
                n_overlaps++;
                EXPECT_TRUE( false ) << "Cell " << A << " has " << B
                                     << " in BOTH M2L and P2P lists "
                                        "(near/far classification overlap)";
                if ( n_overlaps >= 5 )
                    break;
            }
        }
        if ( n_overlaps >= 5 )
            break;
    }
}

//---------------------------------------------------------------------------//
/**
 * Verify L2P matches a direct Coulomb sum on a deliberately non-uniform
 * particle distribution.
 *
 * The geometry forces an adaptive tree: a dense source cluster in
 * [0.0, 0.15]^3 (refines deeply because of high local particle count) and a
 * sparse target region in [0.6, 1.0]^3 (refines shallowly). Cells of
 * different sizes neighbor each other across the cluster boundary, so the
 * M2L interaction list at fine target/source cells must include coarser
 * cells from the opposite side. This is the regime where the previous
 * probe-point neighbor finder produced incorrect lists.
 *
 * Same direct-sum reference and tolerance check as the uniform two-cluster
 * variants. Single-rank only.
 */
void testL2PApproximatesDirectSumAdaptive( int num_dense_sources,
                                           int num_sparse_targets, int ncrit,
                                           int max_depth, double tolerance,
                                           int replication_depth,
                                           double error_tol )
{
    using namespace DownwardSweepTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    if ( nprocs != 1 )
        return;

    const int N = num_dense_sources + num_sparse_targets;

    AoSoA_ht particles_h( "particles_h", N );
    {
        auto h_pos = Cabana::slice<Position>( particles_h );
        auto h_q = Cabana::slice<Charge>( particles_h );

        std::mt19937 gen( 31337 );
        std::uniform_real_distribution<double> dense_dist( 0.0, 0.15 );
        std::uniform_real_distribution<double> sparse_dist( 0.6, 1.0 );
        std::uniform_real_distribution<double> q_dist( 0.5, 1.5 );

        for ( int i = 0; i < num_dense_sources; i++ )
        {
            h_pos( i, 0 ) = dense_dist( gen );
            h_pos( i, 1 ) = dense_dist( gen );
            h_pos( i, 2 ) = dense_dist( gen );
            h_q( i, 0 ) = q_dist( gen );
        }
        for ( int i = num_dense_sources; i < N; i++ )
        {
            h_pos( i, 0 ) = sparse_dist( gen );
            h_pos( i, 1 ) = sparse_dist( gen );
            h_pos( i, 2 ) = sparse_dist( gen );
            h_q( i, 0 ) = 0.0;
        }
    }

    AoSoA_t particles( "particles", N );
    Cabana::deep_copy( particles, particles_h );

    auto positions = Cabana::slice<Position>( particles );
    auto charges = Cabana::slice<Charge>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );
    builder.build( positions, N );

    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    partitioner.partition( builder, particles, N );
    int num_local = partitioner.num_local_particles();

    positions = Cabana::slice<Position>( particles );
    charges = Cabana::slice<Charge>( particles );
    builder.build( positions, num_local );

    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> comm_plan(
        MPI_COMM_WORLD, get_test_mac_theta() );
    comm_plan.build( builder.cells(), partitioner.ownership(),
                     partitioner.cell_owner_map(), replication_depth );

    UpwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> upward( MPI_COMM_WORLD );
    upward.setup( builder.cells(), partitioner.cell_owner_map(),
                  builder.particle_keys(), num_local );
    upward.execute( charges, positions, comm_plan );

    DownwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> downward(
        MPI_COMM_WORLD );
    downward.setup( upward, num_local );

    auto potential = downward.allocate_potential( num_local );
    auto gradient = downward.allocate_gradient( num_local );
    Kokkos::deep_copy( potential, 0.0 );

    downward.execute( upward.multipoles(), positions, potential, gradient,
                      false, comm_plan );

    Kokkos::View<double* [3], TEST_MEMSPACE> d_pos( "d_pos", num_local );
    Kokkos::View<double*, TEST_MEMSPACE> d_crg( "d_crg", num_local );
    Kokkos::parallel_for(
        "CopyForRefAdaptive",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_local ),
        KOKKOS_LAMBDA( int i ) {
            d_pos( i, 0 ) = positions( i, 0 );
            d_pos( i, 1 ) = positions( i, 1 );
            d_pos( i, 2 ) = positions( i, 2 );
            d_crg( i ) = charges( i, 0 );
        } );
    Kokkos::fence();

    auto h_pos =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_pos );
    auto h_crg =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_crg );
    auto h_phi =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), potential );

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
            const double dx = tx - h_pos( s, 0 );
            const double dy = ty - h_pos( s, 1 );
            const double dz = tz - h_pos( s, 2 );
            const double dist = std::sqrt( dx * dx + dy * dy + dz * dz );
            if ( dist > 0.0 )
                ref += h_crg( s ) / dist;
        }

        const double fmm = h_phi( p, 0 );
        const double rel_err = ( std::abs( ref ) > 1e-14 )
                                   ? std::abs( fmm - ref ) / std::abs( ref )
                                   : std::abs( fmm - ref );

        if ( rel_err > max_rel_err )
            max_rel_err = rel_err;
        n_checked++;
    }

    EXPECT_GT( n_checked, 0 )
        << "No target particles found in adaptive geometry";
    EXPECT_LT( max_rel_err, error_tol )
        << "FMM L2P deviates from direct Coulomb sum on adaptive tree; "
           "max relative error = "
        << max_rel_err;
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

TEST( DownwardSweep, testM2LListInvariantsBasic )
{
    testM2LListInvariants( 1000, 32, 6, 0.1, 2 );
}

TEST( DownwardSweep, testM2LListInvariantsSmall )
{
    testM2LListInvariants( 200, 16, 4, 0.1, 1 );
}

TEST( DownwardSweep, testL2PApproximatesDirectSumAdaptiveBasic )
{
    testL2PApproximatesDirectSumAdaptive( 200, 100, 8, 5, 0.1, 2, 1e-3 );
}

TEST( DownwardSweep, testL2PApproximatesDirectSumAdaptiveSmall )
{
    testL2PApproximatesDirectSumAdaptive( 80, 40, 4, 4, 0.1, 1, 1e-3 );
}

//---------------------------------------------------------------------------//
// DownwardSweepCaching: verify that build_interaction_list_device is a
// no-op when the dirty flag is clear, and that invalidate_interaction_list
// forces a rebuild. Both tests assert the second-solve answer matches the
// first within tolerance for the existing P=6 problem.
//---------------------------------------------------------------------------//

namespace DownwardSweepTest
{
template <class TEST_MS, class TEST_ES>
struct CachingFixture
{
    using kernel = Kernel;
    int num_particles = 1000;
    int ncrit = 32;
    int max_depth = 6;
    double tolerance = 0.1;
    int replication_depth = 2;

    AoSoA_t particles{ "particles", 0 };
    TreeBuilder<TEST_MS, TEST_ES> builder;
    TreePartitioner<TEST_MS, TEST_ES> partitioner;
    CommunicationPlan<TEST_MS, TEST_ES> comm_plan;
    UpwardSweep<TEST_MS, TEST_ES, kernel> upward;
    DownwardSweep<TEST_MS, TEST_ES, kernel> downward;
    int num_local = 0;

    CachingFixture()
        : builder( MPI_COMM_WORLD, ncrit, max_depth,
                   std::array<double, 6>{ tolerance, tolerance, tolerance, tolerance, tolerance, tolerance },
                   tolerance )
        , partitioner( MPI_COMM_WORLD, replication_depth )
        , comm_plan( MPI_COMM_WORLD, get_test_mac_theta() )
        , upward( MPI_COMM_WORLD )
        , downward( MPI_COMM_WORLD )
    {
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );

        particles = AoSoA_t( "particles", num_particles );
        generate_test_particles( particles, num_particles, rank );

        auto positions = Cabana::slice<Position>( particles );
        builder.build( positions, num_particles );
        partitioner.partition( builder, particles, num_particles );
        num_local = partitioner.num_local_particles();

        positions = Cabana::slice<Position>( particles );
        builder.build( positions, num_local );

        comm_plan.build( builder.cells(), partitioner.ownership(),
                         partitioner.cell_owner_map(), replication_depth );

        upward.setup( builder.cells(), partitioner.cell_owner_map(),
                      builder.particle_keys(), num_local );
        upward.execute( Cabana::slice<Charge>( particles ),
                        Cabana::slice<Position>( particles ), comm_plan );

        downward.setup( upward, num_local );
    }
};

template <class TEST_MS, class TEST_ES>
void testSkipsRebuildWhenClean()
{
    CachingFixture<TEST_MS, TEST_ES> fix;

    auto positions = Cabana::slice<Position>( fix.particles );

    auto pot1 = fix.downward.allocate_potential( fix.num_local );
    auto grad1 = fix.downward.allocate_gradient( fix.num_local );
    Kokkos::deep_copy( pot1, 0.0 );
    fix.downward.execute( fix.upward.multipoles(), positions, pot1, grad1,
                          false, fix.comm_plan );

    const int build_count_after_first =
        fix.downward.interaction_list_build_count();
    EXPECT_EQ( build_count_after_first, 1 );

    auto pot2 = fix.downward.allocate_potential( fix.num_local );
    auto grad2 = fix.downward.allocate_gradient( fix.num_local );
    Kokkos::deep_copy( pot2, 0.0 );
    fix.downward.execute( fix.upward.multipoles(), positions, pot2, grad2,
                          false, fix.comm_plan );

    EXPECT_EQ( fix.downward.interaction_list_build_count(),
               build_count_after_first );

    auto h1 = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), pot1 );
    auto h2 = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), pot2 );
    for ( int p = 0; p < fix.num_local; p++ )
    {
        const double a = h1( p, 0 );
        const double b = h2( p, 0 );
        const double scale = std::max( 1.0, std::abs( a ) );
        EXPECT_NEAR( a, b, 1e-12 * scale );
    }
}

template <class TEST_MS, class TEST_ES>
void testRebuildsAfterInvalidate()
{
    CachingFixture<TEST_MS, TEST_ES> fix;

    auto positions = Cabana::slice<Position>( fix.particles );

    auto pot1 = fix.downward.allocate_potential( fix.num_local );
    auto grad1 = fix.downward.allocate_gradient( fix.num_local );
    Kokkos::deep_copy( pot1, 0.0 );
    fix.downward.execute( fix.upward.multipoles(), positions, pot1, grad1,
                          false, fix.comm_plan );

    const int build_count_after_first =
        fix.downward.interaction_list_build_count();
    EXPECT_EQ( build_count_after_first, 1 );

    // The persistent operator cache after the first build: one column per
    // realized canonical key, and every one of them a miss, since the cache
    // started empty.
    const int cache_size_after_first = fix.downward.m2l_op_cache_size();
    const long long keys_built_after_first =
        fix.downward.m2l_op_keys_built_count();
    EXPECT_GT( cache_size_after_first, 0 );
    EXPECT_EQ( keys_built_after_first,
               static_cast<long long>( cache_size_after_first ) );

    fix.downward.invalidate_interaction_list();

    auto pot2 = fix.downward.allocate_potential( fix.num_local );
    auto grad2 = fix.downward.allocate_gradient( fix.num_local );
    Kokkos::deep_copy( pot2, 0.0 );
    fix.downward.execute( fix.upward.multipoles(), positions, pot2, grad2,
                          false, fix.comm_plan );

    EXPECT_EQ( fix.downward.interaction_list_build_count(),
               build_count_after_first + 1 );

    // THE OPERATOR CACHE IS THE POINT OF THIS ADDITION. The interaction list
    // was rebuilt — the assertion above says so — but the tree did not change,
    // so every canonical key the rebuild realized was already in the cache and
    // KernelType::build_m2l_operators must not have been called again. A cache
    // that silently rebuilt everything would be invisible in every other
    // number this class exposes, including a bit-for-bit comparison of the
    // operator table, because it would rebuild the same bits.
    //
    // This is where the split T9 introduced is actually measured: the per-tree
    // key -> op_idx map is rebuilt on the dirty flag, the geometry-keyed
    // operator cache is not.
    EXPECT_EQ( fix.downward.m2l_op_keys_built_count(), keys_built_after_first );
    EXPECT_EQ( fix.downward.m2l_op_cache_size(), cache_size_after_first );

    int caching_rank = 0;
    MPI_Comm_rank( MPI_COMM_WORLD, &caching_rank );
    std::printf( "[downward-caching] rank %d builds=%d cache_keys=%d "
                 "keys_built=%lld\n",
                 caching_rank, fix.downward.interaction_list_build_count(),
                 fix.downward.m2l_op_cache_size(),
                 fix.downward.m2l_op_keys_built_count() );
    std::fflush( stdout );

    // ...and the potential is unchanged across the re-solve. Together with the
    // zero-rebuild assertion above, this is the check that a PERSISTING cache
    // has not gone stale: a cached operator that no longer matched the tree
    // would move the field here while the counters stayed silent.
    auto h1 = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), pot1 );
    auto h2 = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), pot2 );
    for ( int p = 0; p < fix.num_local; p++ )
    {
        const double a = h1( p, 0 );
        const double b = h2( p, 0 );
        const double scale = std::max( 1.0, std::abs( a ) );
        EXPECT_NEAR( a, b, 1e-12 * scale );
    }
}
} // namespace DownwardSweepTest

TEST( DownwardSweepCaching, skipsRebuildWhenClean )
{
    DownwardSweepTest::testSkipsRebuildWhenClean<TEST_MEMSPACE,
                                                 TEST_EXECSPACE>();
}

TEST( DownwardSweepCaching, rebuildsAfterInvalidate )
{
    DownwardSweepTest::testRebuildsAfterInvalidate<TEST_MEMSPACE,
                                                   TEST_EXECSPACE>();
}

//---------------------------------------------------------------------------//

namespace DownwardSweepTest
{
template <class TEST_MEMSPACE_, class TEST_EXECSPACE_>
void testLayoutTagIsLayoutRight()
{
    using DS = DownwardSweep<TEST_MEMSPACE_, TEST_EXECSPACE_, Kernel>;
    using US = UpwardSweep<TEST_MEMSPACE_, TEST_EXECSPACE_, Kernel>;
    static_assert(
        std::is_same<typename DS::coeff_view_type::array_layout,
                     Kokkos::LayoutRight>::value,
        "DownwardSweep::coeff_view_type must be LayoutRight" );
    static_assert(
        std::is_same<typename US::coeff_view_type::array_layout,
                     Kokkos::LayoutRight>::value,
        "UpwardSweep::coeff_view_type must be LayoutRight" );
}
} // namespace DownwardSweepTest

TEST( DownwardSweepLayout, layoutTagIsLayoutRight )
{
    DownwardSweepTest::testLayoutTagIsLayoutRight<TEST_MEMSPACE,
                                                  TEST_EXECSPACE>();
}

//---------------------------------------------------------------------------//
// DownwardSweepTwoScale: the non-uniform fixture the tree-opt measurements
// are taken on (T1 of tasks/tree-opt.md).
//
// WHY A NEW DRAW RATHER THAN THE CLUSTERED ONE IN tstMultiSolve.hpp. That
// draw's refusals were measured per-reason first (T1 step 1, recorded in
// tasks/tree-opt-progress-log.md): all of them are range-guard refusals, and
// within the range guard all of them are OFFSET refusals -- its occupancy
// never passes depth 6, so |dd| can never exceed LaplaceKernel's
// m2l_key_dd_max of 6 and the dd half of the guard is unreachable there. It
// has no large depth difference to offer, which is the one property the tasks
// built on this fixture need. Hence a two-scale draw: a dense cluster that
// refines several levels below ncrit, beside a sparse halo whose cells reach
// ncrit immediately and stop as shallow leaves. A single Gaussian does not
// guarantee that; a large density ratio over a short distance does.
//---------------------------------------------------------------------------//

namespace DownwardSweepTest
{

// Geometry of the two-scale draw, in DOMAIN units on [0,1)^3.
//
// The blob is a cube of half-width TWO_SCALE_BLOB_HALF_WIDTH centred on
// TWO_SCALE_BLOB_CENTER. Its edge, 2 * 0.01 = 0.02, is shorter than a cell at
// depth 5 (width 2^-5 = 0.031), so with ncrit = 8 the refinement inside it
// does not terminate until depth 7-8 -- while the uniform halo, a few hundred
// particles over the whole domain, reaches ncrit at depth 2-4. That spread is
// the fixture's reason for existing; testTwoScaleTreeHasShallowAndDeepLeaves
// asserts it rather than trusting it.
constexpr double TWO_SCALE_BLOB_CENTER = 0.15;
constexpr double TWO_SCALE_BLOB_HALF_WIDTH = 0.01;
// Particles in the blob, as a fraction of the per-rank count.
constexpr double TWO_SCALE_BLOB_FRACTION = 0.875;
// The skirt (A1 of tasks/tree-opt.md): a sparse cube about the blob centre,
// half-width 0.04, so the cells beside the blob at depth 5 (width ~0.0375)
// are occupied, with a few particles each, and stop as leaves against the
// blob's depth 7-8 leaves. Without it those cells are empty, so they are not
// in the cell list, and no shallow leaf TOUCHES a deep one: the touching-leaf
// level difference was 1 at np 2, 5 and 6. The remainder of the per-rank
// count, after blob and skirt, is the halo.
constexpr double TWO_SCALE_SKIRT_HALF_WIDTH = 0.04;
constexpr double TWO_SCALE_SKIRT_FRACTION = 0.05;

// Two-scale positions in [0, 1)^3 with strictly positive charges in
// [0.1, 1.0], seeded per rank so each rank draws a different halo while the
// blob stays in the same corner for all of them.
void generate_two_scale_particles( AoSoA_t& particles, int num_particles,
                                   int rank )
{
    AoSoA_ht particles_h( "particles_h", num_particles );
    auto h_pos = Cabana::slice<Position>( particles_h );
    auto h_q = Cabana::slice<Charge>( particles_h );

    std::mt19937 gen( 42 + rank * 7919 );
    std::uniform_real_distribution<double> halo_dist( 0.0, 1.0 );
    std::uniform_real_distribution<double> blob_dist(
        TWO_SCALE_BLOB_CENTER - TWO_SCALE_BLOB_HALF_WIDTH,
        TWO_SCALE_BLOB_CENTER + TWO_SCALE_BLOB_HALF_WIDTH );
    std::uniform_real_distribution<double> skirt_dist(
        TWO_SCALE_BLOB_CENTER - TWO_SCALE_SKIRT_HALF_WIDTH,
        TWO_SCALE_BLOB_CENTER + TWO_SCALE_SKIRT_HALF_WIDTH );
    std::uniform_real_distribution<double> q_dist( 0.1, 1.0 );

    const int num_blob =
        static_cast<int>( TWO_SCALE_BLOB_FRACTION * num_particles );
    const int num_skirt =
        static_cast<int>( TWO_SCALE_SKIRT_FRACTION * num_particles );

    for ( int i = 0; i < num_particles; i++ )
    {
        auto& dist = i < num_blob               ? blob_dist
                     : i < num_blob + num_skirt ? skirt_dist
                                                : halo_dist;
        for ( int d = 0; d < 3; d++ )
            h_pos( i, d ) = dist( gen );
        h_q( i, 0 ) = q_dist( gen );
    }

    particles.resize( num_particles );
    Cabana::deep_copy( particles, particles_h );
}

// Geometry of the graded draw (B0b of tasks/tree-opt.md), in DOMAIN units:
// r = R_MIN * (R_MAX / R_MIN)^u, u ~ U(0,1), isotropic about the centre.
// Density falls as r^-3, so the leaf depth drops one level per octave of r
// and every shell boundary is a one-level step -- the geometry that admits
// the most |dd| = 1 pairs inside the range guard. Six octaves put leaves at
// depths 2-8 at ncrit = 8, max_depth = 8.
constexpr double TWO_SCALE_GRADED_CENTER = 0.5;
constexpr double TWO_SCALE_GRADED_R_MAX = 0.45;
constexpr double TWO_SCALE_GRADED_R_MIN = TWO_SCALE_GRADED_R_MAX / 64.0;

// Graded positions, charges as generate_two_scale_particles, seeded per
// rank. A position outside [0, 1)^3 is REJECTED and redrawn, not clamped:
// clamping would pile particles onto the box faces. At the constants above
// the sphere lies inside the box, so the rejection never fires.
void generate_graded_particles( AoSoA_t& particles, int num_particles,
                                int rank )
{
    AoSoA_ht particles_h( "particles_h", num_particles );
    auto h_pos = Cabana::slice<Position>( particles_h );
    auto h_q = Cabana::slice<Charge>( particles_h );

    std::mt19937 gen( 42 + rank * 7919 );
    std::uniform_real_distribution<double> u_dist( 0.0, 1.0 );
    std::uniform_real_distribution<double> cos_dist( -1.0, 1.0 );
    std::uniform_real_distribution<double> phi_dist( 0.0,
                                                     2.0 * std::acos( -1.0 ) );
    std::uniform_real_distribution<double> q_dist( 0.1, 1.0 );

    for ( int i = 0; i < num_particles; i++ )
    {
        double x[3];
        bool inside = false;
        while ( !inside )
        {
            const double r =
                TWO_SCALE_GRADED_R_MIN *
                std::pow( TWO_SCALE_GRADED_R_MAX / TWO_SCALE_GRADED_R_MIN,
                          u_dist( gen ) );
            const double ct = cos_dist( gen );
            const double st = std::sqrt( 1.0 - ct * ct );
            const double phi = phi_dist( gen );
            x[0] = TWO_SCALE_GRADED_CENTER + r * st * std::cos( phi );
            x[1] = TWO_SCALE_GRADED_CENTER + r * st * std::sin( phi );
            x[2] = TWO_SCALE_GRADED_CENTER + r * ct;
            inside = true;
            for ( int d = 0; d < 3; d++ )
                inside = inside && x[d] >= 0.0 && x[d] < 1.0;
        }
        for ( int d = 0; d < 3; d++ )
            h_pos( i, d ) = x[d];
        h_q( i, 0 ) = q_dist( gen );
    }

    particles.resize( num_particles );
    Cabana::deep_copy( particles, particles_h );
}

// Which draw TwoScaleFixture builds its tree over.
enum class TwoScaleDraw
{
    TwoScale, // generate_two_scale_particles: T1's fixture
    Graded    // generate_graded_particles: B0b's cross-level-dense tree
};

inline const char* to_string( TwoScaleDraw draw )
{
    return draw == TwoScaleDraw::Graded ? "graded" : "two-scale";
}

// The fixture: a built tree, partition, communication plan and upward sweep
// over the selected draw (two-scale by default), with the downward sweep set
// up and ready to execute. Same shape as CachingFixture above, and the same two-phase build
// (global tree, partition, rebuild on the local particles).
//
// TEMPLATED ON THE FAR-FIELD TYPE, not fixed to this file's `Kernel`, because
// B0 reads its numbers out of this same fixture for both CartesianTaylorBasis
// and LaplaceKernel and the two must be the same tree. The particle AoSoA is
// shared with the rest of the file, so a far-field type with a different
// component count would silently mis-size the charge slice; the static_assert
// makes that a compile error instead.
//
// THE COLUMN CAP IS LEFT AT ITS DEFAULT, deliberately. An uncapped sweep can
// refuse a pair a column for exactly one reason, representability, so a
// non-zero range_guard reading here is unambiguous and count_cap must be 0.
template <class TEST_MS, class TEST_ES, class FarField = Kernel>
struct TwoScaleFixture
{
    using kernel = FarField;
    static_assert( FarField::num_components == Kernel::num_components,
                   "TwoScaleFixture: the far-field type's component count "
                   "must match the particle AoSoA's charge extent" );

    // GLOBAL particle count, split across ranks below -- NOT per rank. The
    // tree is built from the global set, so a per-rank count would make the
    // tree deeper and the blob's interaction list larger at every added rank:
    // measured, a per-rank 1200 ran the two cases in 8 s at np 4 and had not
    // finished at 300 s by np 5. A fixed global count also makes the rank
    // counts comparable to each other, which is the whole point of reading
    // B0/A1/C1's numbers per (nprocs, rank) off one fixture.
    int num_particles_global = 1200;
    int ncrit = 8;          // leaf capacity, in particles (global count)
    int max_depth = 8;      // depth cap; the blob reaches 7-8 at this ncrit
    double tolerance = 0.1; // tree bounding-box padding, domain units
    int replication_depth = 2;
    // MAC opening angle, dimensionless; a constructor argument because it
    // reaches comm_plan in the initializer list. The default 0.3 is tighter
    // than the solver's 0.5 on purpose: a tighter theta descends further
    // before admitting a pair, which is what puts admitted pairs at the deep
    // end of the tree where an offset can exceed M2L_KEY_OFFSET_MAX = 32
    // half-widths at that depth.
    double mac_theta;
    TwoScaleDraw draw;
    // Plummer softening LENGTH eps, domain units; the kernel's b is eps^2.
    // Positive because CartesianTaylorBasis aborts on b <= 0. It moves no
    // integer key and no interaction-list entry, and nothing here asserts
    // accuracy, so its value is otherwise arbitrary.
    double softening = 1.0e-3;

    AoSoA_t particles{ "particles", 0 };
    TreeBuilder<TEST_MS, TEST_ES> builder;
    TreePartitioner<TEST_MS, TEST_ES> partitioner;
    CommunicationPlan<TEST_MS, TEST_ES> comm_plan;
    UpwardSweep<TEST_MS, TEST_ES, kernel> upward;
    DownwardSweep<TEST_MS, TEST_ES, kernel> downward;
    int num_local = 0;

    explicit TwoScaleFixture( TwoScaleDraw draw_in = TwoScaleDraw::TwoScale,
                              double mac_theta_in = 0.3,
                              bool quantize_root_half_width = false )
        : mac_theta( mac_theta_in )
        , draw( draw_in )
        , builder( MPI_COMM_WORLD, ncrit, max_depth,
                   std::array<double, 6>{ tolerance, tolerance, tolerance, tolerance, tolerance, tolerance },
                   tolerance, quantize_root_half_width )
        , partitioner( MPI_COMM_WORLD, replication_depth )
        , comm_plan( MPI_COMM_WORLD, mac_theta )
        , upward( MPI_COMM_WORLD )
        , downward( MPI_COMM_WORLD )
    {
        int rank, nprocs;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

        // Split the global count across ranks, remainder to the low ranks,
        // so the global set is the same size at every rank count.
        const int num_particles = num_particles_global / nprocs +
                                  ( rank < num_particles_global % nprocs );

        particles = AoSoA_t( "particles", num_particles );
        if ( draw == TwoScaleDraw::Graded )
            generate_graded_particles( particles, num_particles, rank );
        else
            generate_two_scale_particles( particles, num_particles, rank );
        num_local = num_particles;

        // Both sweeps get the same parameters, the upward one before setup():
        // it builds the aux tables the downward sweep borrows.
        M2LKernelParams kernel_params;
        kernel_params.softening = softening;
        upward.set_m2l_kernel_params( kernel_params );
        downward.set_m2l_kernel_params( kernel_params );

        rebuild_with( builder );
    }

    // Build the tree over the current particles with `tree`, partition, and
    // set both sweeps up on it, keeping the downward sweep and its operator
    // cache as Solver keeps its own across a rebuild. The constructor runs
    // this with `builder`; B2's retention case runs it again with a builder
    // whose bounding-box padding differs.
    void rebuild_with( TreeBuilder<TEST_MS, TEST_ES>& tree )
    {
        auto positions = Cabana::slice<Position>( particles );
        tree.build( positions, num_local );
        partitioner.partition( tree, particles, num_local );
        num_local = partitioner.num_local_particles();

        positions = Cabana::slice<Position>( particles );
        tree.build( positions, num_local );

        comm_plan.build( tree.cells(), partitioner.ownership(),
                         partitioner.cell_owner_map(), replication_depth );

        // As Solver::_push_root_half_width: a key_needs_level basis builds
        // its operators from the per-depth widths this sets.
        downward.set_root_half_width( tree.root_half_width() );

        upward.setup( tree.cells(), partitioner.cell_owner_map(),
                      tree.particle_keys(), num_local );
        upward.execute( Cabana::slice<Charge>( particles ),
                        Cabana::slice<Position>( particles ), comm_plan );

        downward.invalidate_interaction_list();
        downward.setup( upward, num_local );
    }

    // Run one solve, which is what populates every counter below: the
    // per-reason fallback tallies are set by build_interaction_list_device(),
    // which execute() drives.
    void solve()
    {
        auto positions = Cabana::slice<Position>( particles );
        auto pot = downward.allocate_potential( num_local );
        auto grad = downward.allocate_gradient( num_local );
        Kokkos::deep_copy( pot, 0.0 );
        downward.execute( upward.multipoles(), positions, pot, grad, false,
                          comm_plan );
    }
};

// THE FIXTURE'S CONTRACT, and the reason it is an assertion and not a print:
// a later change to ncrit, to max_depth or to the draw could flatten this tree
// to a uniform one, on which the range-guard counter reads 0 everywhere and
// every measurement built on the fixture becomes a measurement of nothing that
// still passes. Risk R7 in tasks/tree-opt.md. So the geometry is asserted at
// the source.
template <class TEST_MS, class TEST_ES>
void testTwoScaleTreeHasShallowAndDeepLeaves()
{
    TwoScaleFixture<TEST_MS, TEST_ES> fix;
    fix.solve();

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    const std::vector<int> cells_at_depth = fix.downward.m2l_cells_at_depth();

    // Occupied depths, and the shallowest and deepest of them. Depth 0 is the
    // root and is occupied on every tree, so it is the depths BELOW it that
    // carry the claim.
    int n_occupied = 0;
    int deepest = -1;
    for ( std::size_t d = 0; d < cells_at_depth.size(); ++d )
    {
        if ( cells_at_depth[d] <= 0 )
            continue;
        ++n_occupied;
        deepest = static_cast<int>( d );
    }

    // The shallowest depth at which refinement STOPS for some cell: occupancy
    // falls from d to d+1, so at least one occupied cell at d has no children
    // and is a leaf. "Shallowest OCCUPIED depth" would not do -- depth 0 is
    // the root and is occupied on every tree ever built, uniform ones
    // included, so asserting on it asserts nothing.
    int shallowest_leaf = -1;
    for ( std::size_t d = 1; d + 1 < cells_at_depth.size(); ++d )
    {
        if ( cells_at_depth[d] > 0 &&
             cells_at_depth[d] > cells_at_depth[d + 1] )
        {
            shallowest_leaf = static_cast<int>( d );
            break;
        }
    }

    std::string depth_occ;
    for ( std::size_t d = 0; d < cells_at_depth.size(); ++d )
    {
        depth_occ += std::to_string( cells_at_depth[d] );
        if ( d + 1 < cells_at_depth.size() )
            depth_occ += ",";
    }

    // Step 8's record line: every number a later task reads out of this
    // fixture, on one line, per (nprocs, rank). Printed before the assertions
    // so a failing rank still contributes its numbers to the log.
    std::printf( "[two-scale] nprocs %d rank %d num_local %d "
                 "range_guard %lld count_cap %lld depth_dropped %lld "
                 "total_fallback %lld unique_ops %d demanded_ops %d "
                 "realized_keys %d occupied_depths %d shallowest_leaf %d "
                 "deepest %d cells_at_depth [%s]\n",
                 nprocs, rank, fix.num_local,
                 fix.downward.m2l_n_fallback_pairs_range_guard(),
                 fix.downward.m2l_n_fallback_pairs_count_cap(),
                 fix.downward.m2l_n_fallback_pairs_depth_dropped(),
                 fix.downward.total_fallback_pair_count(),
                 fix.downward.m2l_n_unique_ops(),
                 fix.downward.m2l_n_demanded_ops(),
                 static_cast<int>( fix.downward.m2l_realized_keys().size() ),
                 n_occupied, shallowest_leaf, deepest, depth_occ.c_str() );
    std::fflush( stdout );

    EXPECT_GE( n_occupied, 3 )
        << "the two-scale tree has fewer than three occupied depths, so it is "
           "not the shallow-beside-deep geometry every tree-opt measurement "
           "assumes (cells_at_depth = [" << depth_occ << "])";

    // A shallow leaf beside a deep subtree, stated as its two halves. Both
    // matter -- a uniformly deep tree has no shallow leaf and a uniformly
    // shallow one has no deep subtree, and neither carries a level
    // difference.
    ASSERT_GE( shallowest_leaf, 1 )
        << "occupancy never falls going deeper, so no cell below the root "
           "terminated as a leaf and this tree is uniformly refined "
           "(cells_at_depth = [" << depth_occ << "])";
    EXPECT_LE( shallowest_leaf, 4 )
        << "the shallowest leaf is deeper than level 4: the halo did not "
           "reach ncrit early, so there are no shallow leaves "
           "(cells_at_depth = [" << depth_occ << "])";
    EXPECT_GE( deepest, 6 )
        << "no deep occupied depth: the blob did not force refinement below "
           "the halo's level (cells_at_depth = [" << depth_occ << "])";
    EXPECT_GE( deepest - shallowest_leaf, 2 )
        << "a shallow leaf and the deepest cell are under 2 levels apart, so "
           "no pair in this tree carries the depth difference the fixture "
           "exists to produce (cells_at_depth = [" << depth_occ << "])";
}

// The refusal claim: on this geometry the range guard actually fires, and the
// count cap -- left at its default, so unbounded -- does not. The second half
// is what makes the first unambiguous.
template <class TEST_MS, class TEST_ES>
void testTwoScaleRefusalsAreRangeGuard()
{
    TwoScaleFixture<TEST_MS, TEST_ES> fix;
    fix.solve();

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    const long long range_guard =
        fix.downward.m2l_n_fallback_pairs_range_guard();
    const long long count_cap =
        fix.downward.m2l_n_fallback_pairs_count_cap();
    const long long depth_dropped =
        fix.downward.m2l_n_fallback_pairs_depth_dropped();
    const long long total = fix.downward.total_fallback_pair_count();

    std::printf( "[two-scale-refusals] nprocs %d rank %d range_guard %lld "
                 "count_cap %lld depth_dropped %lld total_fallback %lld "
                 "op_count_cap %d\n",
                 nprocs, rank, range_guard, count_cap, depth_dropped, total,
                 fix.downward.m2l_op_count_cap() );
    std::fflush( stdout );

#ifdef CANOPY_ENABLE_PROFILING
    // Sum over ranks: the assertion the exit criterion names is "at least one
    // rank", because the partition decides which rank carries the deep
    // subtree and that assignment is not reproducible above two ranks
    // (risk R6). A per-rank EXPECT_GT would be asserting the partition.
    long long global_range_guard = 0;
    MPI_Allreduce( &range_guard, &global_range_guard, 1, MPI_LONG_LONG,
                   MPI_SUM, MPI_COMM_WORLD );

    EXPECT_GT( global_range_guard, 0 )
        << "no pair in the two-scale tree was refused a key by the range "
           "guard, so this fixture measures nothing. Either the tree "
           "flattened (see testTwoScaleTreeHasShallowAndDeepLeaves) or "
           "M2L_KEY_OFFSET_MAX / KernelType::m2l_key_dd_max was raised past "
           "what this geometry produces";

    // The column cap is at its default here, so a budget refusal is not
    // available and every refusal above is a representability refusal.
    EXPECT_EQ( count_cap, 0 )
        << "a pair was refused a column by the count cap on a sweep whose "
           "cap was never set, which would make the range_guard reading "
           "above ambiguous";

    // A dropped pair is a contribution that is never evaluated at all -- not
    // through a column and not through the fallback.
    EXPECT_EQ( depth_dropped, 0 )
        << "a refused pair was placed in neither an operator column nor the "
           "fallback table, so its contribution is missing from the solve";

    // The two reasons partition the fallback population exactly.
    EXPECT_EQ( range_guard + count_cap, total )
        << "the per-reason counters do not sum to the fallback total, so at "
           "least one refusal path is unaccounted for";
#else
    // Built without CANOPY_ENABLE_PROFILING: all three read -1, the
    // "unavailable" sentinel, and NEVER 0 -- 0 is a legal count for each of
    // them, so a 0 here would be indistinguishable from a real measurement of
    // no refusals.
    EXPECT_EQ( range_guard, -1 );
    EXPECT_EQ( count_cap, -1 );
    EXPECT_EQ( depth_dropped, -1 );

    // The sum identity is SKIPPED, not evaluated: -1 + -1 against a real
    // total is a claim about nothing, and a check that "passed" on sentinels
    // would be a vacuous pass.
    if ( rank == 0 )
        std::printf( "[two-scale-refusals] sum identity SKIPPED: built "
                     "without CANOPY_ENABLE_PROFILING, counters are "
                     "sentinels\n" );
    std::fflush( stdout );
#endif
}

// B0 of tasks/tree-opt.md: how many admitted operator columns differ from
// another only in dd. CartesianTaylorBasis's operator ignores dd
// (Canopy_CartesianTaylorBasis.hpp, m2l_operator_block), so for it
// admitted / distinct-ignoring-dd is the factor a dd-free key would remove.
// LaplaceKernel's operator does depend on dd, so its figure is the control,
// not a saving. A measurement: the only assertion on the counts is the
// by-construction distinct <= admitted.
//
// `tag` and `context` name the counts line: B0's case prints "[b0-dd] basis
// ..." unchanged, B0b's prefixes its draw and angle. The [dd-hist] line
// (B0b step 1) histograms the admitted keys by signed dd = d_s - d_t, from
// -m2l_key_dd_max to +m2l_key_dd_max; the range guard admits nothing outside,
// so `outside` must read 0. Returns the number of admitted keys with dd != 0.
template <class FarField, class Fixture>
long long reportTwoScaleDdDuplicates( const char* basis, const Fixture& fix,
                                      int nprocs, int rank,
                                      const char* tag = "b0-dd",
                                      const std::string& context = "" )
{
    const auto& keys = fix.downward.m2l_realized_keys();
    std::set<std::array<int, 4>> distinct;
    constexpr int dd_max = FarField::m2l_key_dd_max;
    std::array<long long, 2 * dd_max + 1> hist{};
    long long outside = 0;
    for ( const auto& k : keys )
    {
        distinct.insert( { k.max_d, k.ii, k.jj, k.kk } );
        if ( std::abs( k.dd ) <= dd_max )
            ++hist[k.dd + dd_max];
        else
            ++outside;
    }

    const long long admitted = static_cast<long long>( keys.size() );
    const long long n_distinct = static_cast<long long>( distinct.size() );
    const long long bytes_per_key =
        static_cast<long long>( FarField::bytes_per_key );
    const long long cross_level = admitted - hist[dd_max];

    std::printf( "[%s] %sbasis %s nprocs %d rank %d admitted %lld "
                 "distinct_no_dd %lld factor %.4f demanded_ops %d "
                 "bytes_per_key %lld admitted_bytes %lld "
                 "distinct_bytes %lld\n",
                 tag, context.c_str(), basis, nprocs, rank, admitted,
                 n_distinct,
                 n_distinct > 0 ? double( admitted ) / double( n_distinct )
                                : 1.0,
                 fix.downward.m2l_n_demanded_ops(), bytes_per_key,
                 admitted * bytes_per_key, n_distinct * bytes_per_key );

    std::string hist_str;
    for ( int dd = -dd_max; dd <= dd_max; ++dd )
    {
        hist_str += std::to_string( dd ) + ":" +
                    std::to_string( hist[dd + dd_max] );
        if ( dd < dd_max )
            hist_str += " ";
    }
    std::printf( "[dd-hist] draw %s theta %.2f basis %s nprocs %d rank %d "
                 "admitted %lld cross_level %lld outside %lld hist [%s]\n",
                 to_string( fix.draw ), fix.mac_theta, basis, nprocs, rank,
                 admitted, cross_level, outside, hist_str.c_str() );
    std::fflush( stdout );

    EXPECT_LE( n_distinct, admitted )
        << basis << ": more distinct (max_d, ii, jj, kk) tuples than "
                    "admitted keys, so the count itself is wrong";
    return cross_level;
}

template <class TEST_MS, class TEST_ES>
void testTwoScaleDdDuplicates()
{
    using CTBasis = CartesianTaylorBasis<double, 3, 1>;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    TwoScaleFixture<TEST_MS, TEST_ES, CTBasis> ct;
    ct.solve();
    reportTwoScaleDdDuplicates<CTBasis>( "CartesianTaylor", ct, nprocs,
                                         rank );

    TwoScaleFixture<TEST_MS, TEST_ES, Kernel> laplace;
    laplace.solve();
    reportTwoScaleDdDuplicates<Kernel>( "Laplace", laplace, nprocs, rank );

    // The control means something only on the same tree.
    EXPECT_EQ( ct.downward.m2l_cells_at_depth(),
               laplace.downward.m2l_cells_at_depth() );
}

// B0b of tasks/tree-opt.md: B0's count on the graded draw, beside T1's draw,
// at the fixture's angle and the solver default. Its contract, on a basis
// whose key keeps dd (key_needs_dd): the graded tree admits more cross-level
// (dd != 0) keys, summed over ranks, than T1's at the same rank count and
// angle -- otherwise it tests nothing B0 did not. On a basis that zeroes dd
// (B1), the cross-level count must instead be 0 on every rank, which checks
// that the trait reaches the table. No threshold on the factor.
template <class TEST_MS, class TEST_ES>
void testGradedDdDuplicates()
{
    using CTBasis = CartesianTaylorBasis<double, 3, 1>;
    // The draw under test. Pointing it at TwoScaleDraw::TwoScale is the
    // failure direction: the contract below must then fail.
    const TwoScaleDraw graded = TwoScaleDraw::Graded;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    for ( const double theta : { 0.3, 0.5 } )
    {
        // Cross-level admitted keys on this rank, [draw][basis], draw 0 the
        // two-scale baseline and 1 the graded draw; basis 0 CT, 1 Laplace.
        long long cross[2][2] = {};
        for ( int di = 0; di < 2; ++di )
        {
            const TwoScaleDraw draw = di == 0 ? TwoScaleDraw::TwoScale : graded;
            char context[64];
            std::snprintf( context, sizeof( context ), "draw %s theta %.2f ",
                           to_string( draw ), theta );

            TwoScaleFixture<TEST_MS, TEST_ES, CTBasis> ct( draw, theta );
            ct.solve();
            cross[di][0] = reportTwoScaleDdDuplicates<CTBasis>(
                "CartesianTaylor", ct, nprocs, rank, "b0b-dd", context );

            TwoScaleFixture<TEST_MS, TEST_ES, Kernel> laplace( draw, theta );
            laplace.solve();
            cross[di][1] = reportTwoScaleDdDuplicates<Kernel>(
                "Laplace", laplace, nprocs, rank, "b0b-dd", context );

            EXPECT_EQ( ct.downward.m2l_cells_at_depth(),
                       laplace.downward.m2l_cells_at_depth() );

            // Leaves per depth of the global tree, so the log shows how many
            // depths the draw actually spans.
            std::vector<int> leaves( ct.max_depth + 1, 0 );
            for ( const auto& c : ct.builder.cells() )
                if ( c.is_leaf && c.depth >= 0 && c.depth <= ct.max_depth )
                    ++leaves[c.depth];
            int leaf_depths = 0;
            std::string leaf_str;
            for ( std::size_t d = 0; d < leaves.size(); ++d )
            {
                leaf_depths += leaves[d] > 0;
                leaf_str += std::to_string( leaves[d] );
                if ( d + 1 < leaves.size() )
                    leaf_str += ",";
            }
            std::printf( "[b0b-tree] %snprocs %d rank %d leaf_depths %d "
                         "leaves_at_depth [%s]\n",
                         context, nprocs, rank, leaf_depths,
                         leaf_str.c_str() );
            std::fflush( stdout );
        }

        const char* basis_name[2] = { "CartesianTaylor", "Laplace" };
        const bool needs_dd[2] = { CTBasis::key_needs_dd,
                                   Kernel::key_needs_dd };
        for ( int b = 0; b < 2; ++b )
        {
            long long sum[2] = {};
            MPI_Allreduce( &cross[0][b], &sum[0], 1, MPI_LONG_LONG, MPI_SUM,
                           MPI_COMM_WORLD );
            MPI_Allreduce( &cross[1][b], &sum[1], 1, MPI_LONG_LONG, MPI_SUM,
                           MPI_COMM_WORLD );
            if ( rank == 0 )
                std::printf( "[b0b-cross] theta %.2f basis %s nprocs %d "
                             "two-scale %lld graded %lld\n",
                             theta, basis_name[b], nprocs, sum[0], sum[1] );
            std::fflush( stdout );
            if ( !needs_dd[b] )
            {
                for ( int di = 0; di < 2; ++di )
                    EXPECT_EQ( cross[di][b], 0 )
                        << basis_name[b] << " at theta " << theta
                        << ", draw " << ( di == 0 ? "two-scale" : "graded" )
                        << ": key_needs_dd is false, yet a realized key "
                           "carries dd != 0, so canonicalize_key did not "
                           "reach the table";
                continue;
            }
            EXPECT_GT( sum[1], sum[0] )
                << basis_name[b] << " at theta " << theta
                << ": the graded draw admits no more cross-level (dd != 0) "
                   "keys, summed over ranks, than T1's two-scale draw, so it "
                   "tests nothing B0 did not";
        }
    }
}

// B2 of tasks/tree-opt.md: the operator cache survives a rebuild whose box
// stays inside one octave, knob on, and is rebuilt in full otherwise. One
// downward sweep, two builds of the SAME particles, the second by a builder
// whose symmetric bounding-box padding is larger (so the box keeps its centre
// and only its half-width grows); ncrit_tolerance_factor is unchanged. The
// per-build increment of m2l_op_keys_built_count() is read on every rank.
//
//   in-octave, knob on:   the quantized width and the key set are unchanged,
//                         so the increment is 0;
//   in-octave, knob off:  the sweep is told a new width, so the whole cache
//                         is rebuilt: increment == m2l_n_unique_ops();
//   cross-octave, knob on: the quantized width doubles, increment as knob off.
//
// Either cache assertion alone is satisfied by a cache that always reuses or
// always clears (R4); the three together are not.
template <class TEST_MS, class TEST_ES>
void testRootWidthQuantizationRetainsCache()
{
    using CTBasis = CartesianTaylorBasis<double, 3, 1>;
    static_assert( CTBasis::key_needs_level,
                   "the retention case needs a basis whose cache a width "
                   "change clears" );

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    struct Arm
    {
        const char* name;
        bool quantize;
        // Second build's per-face padding, a fraction of the axis width; the
        // first build uses the fixture's own (0.1). The expanded half-width
        // grows by the factor (1 + 2 tol2) / 1.2: 1.17 for the in-octave arms
        // (that it stays in one octave on this draw is asserted below), 2.08
        // for the cross-octave one, and a factor >= 2 always crosses one.
        double tol2;
        bool expect_retained;
    };
    const Arm arms[] = { { "in-octave", true, 0.2, true },
                         { "in-octave", false, 0.2, false },
                         { "cross-octave", true, 0.75, false } };

    for ( const Arm& arm : arms )
    {
        TwoScaleFixture<TEST_MS, TEST_ES, CTBasis> fix(
            TwoScaleDraw::TwoScale, 0.3, arm.quantize );
        fix.solve();
        const double w1 = fix.downward.root_half_width();
        const long long built1 = fix.downward.m2l_op_keys_built_count();
        const int unique1 = fix.downward.m2l_n_unique_ops();

        TreeBuilder<TEST_MS, TEST_ES> wider(
            MPI_COMM_WORLD, fix.ncrit, fix.max_depth,
            std::array<double, 6>{ arm.tol2, arm.tol2, arm.tol2, arm.tol2,
                                   arm.tol2, arm.tol2 },
            fix.tolerance, arm.quantize );
        fix.rebuild_with( wider );
        fix.solve();
        const double w2 = fix.downward.root_half_width();
        const long long inc =
            fix.downward.m2l_op_keys_built_count() - built1;
        const int unique2 = fix.downward.m2l_n_unique_ops();

        std::printf( "[b2-retain] arm %s quantize %d nprocs %d rank %d "
                     "w1 %.17g w2 %.17g keys_built_inc %lld unique1 %d "
                     "unique2 %d\n",
                     arm.name, arm.quantize ? 1 : 0, nprocs, rank, w1, w2,
                     inc, unique1, unique2 );
        std::fflush( stdout );

        // The arm's premise, so a draw whose box sits near an octave edge
        // fails here rather than as a cache defect.
        EXPECT_EQ( w1, fix.builder.root_half_width() );
        EXPECT_EQ( w2, wider.root_half_width() );
        EXPECT_EQ( w1 == w2, arm.expect_retained )
            << arm.name << " quantize " << arm.quantize << ": w1 " << w1
            << " w2 " << w2;
        // Summed over ranks: a rank can own no admitted pair at all (np 5
        // rank 2, knob on).
        int unique1_sum = 0;
        MPI_Allreduce( &unique1, &unique1_sum, 1, MPI_INT, MPI_SUM,
                       MPI_COMM_WORLD );
        EXPECT_GT( unique1_sum, 0 ) << "no column was admitted: vacuous";

        if ( arm.expect_retained )
        {
            EXPECT_EQ( inc, 0 )
                << "knob on, box inside one octave: the cache should have "
                   "survived the rebuild, rank "
                << rank;
            EXPECT_EQ( unique2, unique1 );
        }
        else
        {
            EXPECT_EQ( inc, unique2 )
                << arm.name << " quantize " << arm.quantize
                << ": a width change must rebuild every admitted column, "
                   "rank "
                << rank;
        }
    }
}

// A1 of tasks/tree-opt.md: how far apart in depth touching leaves are, and how
// many cells a balance to within `delta` levels would add. A model of the
// balance over builder.cells(), not an implementation of it.
//
// Leaves touch if their closed cubes meet: face, edge or corner. A cell is
// (depth d, integer anchor a), in units of a depth-d cell width; the bits of
// a are the key's octants, bit 0 x, bit 1 y, bit 2 z (which_octant).
struct BalanceCellCoord
{
    int depth;
    long long a[3];
};

inline BalanceCellCoord balance_decode( MortonKey k )
{
    BalanceCellCoord c{ key_depth( k ), { 0, 0, 0 } };
    for ( int l = c.depth - 1; l >= 0; --l )
    {
        const int oct = static_cast<int>( ( k >> ( 3 * l ) ) & 7 );
        for ( int ax = 0; ax < 3; ++ax )
            c.a[ax] = 2 * c.a[ax] + ( ( oct >> ax ) & 1 );
    }
    return c;
}

inline MortonKey balance_encode( int depth, const long long a[3] )
{
    MortonKey k = ROOT_KEY;
    for ( int l = depth - 1; l >= 0; --l )
    {
        int oct = 0;
        for ( int ax = 0; ax < 3; ++ax )
            oct |= static_cast<int>( ( a[ax] >> l ) & 1 ) << ax;
        k = child_key( k, oct );
    }
    return k;
}

// The tree as a key -> is_leaf map. Occupied cells only, as in cells().
using BalanceTree = std::map<MortonKey, bool>;

// The touching leaves of `leaf` at its own depth or shallower. A deeper
// neighbour is found from its own side, so over all leaves this enumerates
// every touching pair. For each of the 26 same-depth cells beside `leaf`, the
// first existing cell walking up its ancestors is either a leaf (the
// neighbour), the same-depth cell itself as an internal cell (deeper
// neighbours), or a shallower internal cell (an empty region, no neighbour).
inline std::set<MortonKey> balance_coarser_neighbours( const BalanceTree& tree,
                                                       MortonKey leaf )
{
    const BalanceCellCoord c = balance_decode( leaf );
    const long long side = 1LL << c.depth;
    std::set<MortonKey> out;
    for ( int dx = -1; dx <= 1; ++dx )
        for ( int dy = -1; dy <= 1; ++dy )
            for ( int dz = -1; dz <= 1; ++dz )
            {
                if ( dx == 0 && dy == 0 && dz == 0 )
                    continue;
                const long long n[3] = { c.a[0] + dx, c.a[1] + dy,
                                         c.a[2] + dz };
                if ( n[0] < 0 || n[1] < 0 || n[2] < 0 || n[0] >= side ||
                     n[1] >= side || n[2] >= side )
                    continue;
                for ( int d = c.depth; d >= 0; --d )
                {
                    const int s = c.depth - d;
                    const long long up[3] = { n[0] >> s, n[1] >> s,
                                              n[2] >> s };
                    auto it = tree.find( balance_encode( d, up ) );
                    if ( it == tree.end() )
                        continue;
                    if ( it->second )
                        out.insert( it->first );
                    break;
                }
            }
    return out;
}

// Touching-leaf level differences: hist[ld] counts unordered pairs.
inline std::vector<long long> balance_level_hist( const BalanceTree& tree,
                                                  int max_depth )
{
    std::vector<long long> hist( max_depth + 1, 0 );
    for ( const auto& [key, is_leaf] : tree )
    {
        if ( !is_leaf )
            continue;
        const int d = key_depth( key );
        for ( MortonKey nb : balance_coarser_neighbours( tree, key ) )
        {
            const int dn = key_depth( nb );
            // Equal-depth pairs are found from both sides; count one.
            if ( dn == d && nb > key )
                continue;
            ++hist[d - dn];
        }
    }
    return hist;
}

struct BalanceCost
{
    long long added = 0; // cells the balance creates
    int passes = 0;      // refinement passes that refined something
    long long stuck = 0; // leaves out of balance and already at max_depth
};

// Refine every leaf more than `delta` levels shallower than a touching leaf,
// until none is. Refining creates only the children `occupied` holds, so the
// balanced tree keeps the builder's rule that every cell is occupied. A leaf
// at max_depth cannot be refined; it is counted in `stuck`, not dropped.
inline BalanceCost balance_simulate( BalanceTree& tree, int delta,
                                     int max_depth,
                                     const std::set<MortonKey>& occupied )
{
    // A pass deepens every refined leaf by one level, and no leaf is deeper
    // than max_depth, so a terminating balance needs well under this.
    const int max_passes = 8 * ( max_depth + 1 );
    BalanceCost cost;
    for ( int pass = 0;; ++pass )
    {
        if ( pass == max_passes )
            throw std::runtime_error(
                "balance_simulate: no fixed point after " +
                std::to_string( max_passes ) + " passes" );
        std::set<MortonKey> refine;
        for ( const auto& [key, is_leaf] : tree )
        {
            if ( !is_leaf )
                continue;
            const int d = key_depth( key );
            for ( MortonKey nb : balance_coarser_neighbours( tree, key ) )
                if ( d - key_depth( nb ) > delta )
                    refine.insert( nb );
        }
        long long stuck = 0;
        bool refined = false;
        for ( MortonKey k : refine )
        {
            if ( key_depth( k ) >= max_depth )
            {
                ++stuck;
                continue;
            }
            tree[k] = false;
            refined = true;
            for ( int oct = 0; oct < 8; ++oct )
            {
                const MortonKey ck = child_key( k, oct );
                if ( occupied.count( ck ) )
                {
                    tree[ck] = true;
                    ++cost.added;
                }
            }
        }
        cost.stuck = stuck;
        if ( !refined )
            break;
        ++cost.passes;
    }
    return cost;
}

template <class TEST_MS, class TEST_ES>
void testTwoScaleBalanceCost()
{
    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    // The range guard's two bounds, for reading max_ld against: Laplace's
    // |dd| <= m2l_key_dd_max, and DownwardSweep::M2L_KEY_OFFSET_MAX (private),
    // in half-widths at the finer cell's depth.
    constexpr int dd_bound = Kernel::m2l_key_dd_max;
    constexpr int offset_bound = 32;
    const int deltas[3] = { 1, 2, 3 };

    for ( const TwoScaleDraw draw :
          { TwoScaleDraw::TwoScale, TwoScaleDraw::Graded } )
    {
        // Root quantization off, the fixture default: B2 showed knob-on is a
        // different tree.
        TwoScaleFixture<TEST_MS, TEST_ES> fix( draw, 0.3 );
        const auto& cells = fix.builder.cells();
        const int max_depth = fix.max_depth;

        BalanceTree tree;
        std::vector<int> per_depth( max_depth + 1, 0 );
        long long n_leaves = 0;
        const CellInfo* root = nullptr;
        for ( const auto& c : cells )
        {
            tree[c.key] = c.is_leaf;
            ++per_depth[c.depth];
            n_leaves += c.is_leaf;
            if ( c.key == ROOT_KEY )
                root = &c;
        }
        ASSERT_NE( root, nullptr );

        // Global occupancy, as build() counts it: each rank walks its local
        // particles down from the root with the builder's own arithmetic,
        // and the occupied keys are united over ranks.
        auto particles_h = Cabana::create_mirror_view_and_copy(
            Kokkos::HostSpace(), fix.particles );
        auto pos = Cabana::slice<Position>( particles_h );
        std::vector<MortonKey> local_keys;
        local_keys.reserve( static_cast<std::size_t>( fix.num_local ) *
                            ( max_depth + 1 ) );
        using TB = TreeBuilder<TEST_MS, TEST_ES>;
        for ( int i = 0; i < fix.num_local; ++i )
        {
            MortonKey k = ROOT_KEY;
            double c[3] = { root->center[0], root->center[1],
                            root->center[2] };
            double hw = root->half_width;
            local_keys.push_back( k );
            for ( int d = 1; d <= max_depth; ++d )
            {
                const int oct = TB::which_octant( pos( i, 0 ), pos( i, 1 ),
                                                  pos( i, 2 ), c[0], c[1],
                                                  c[2] );
                TB::child_center( c[0], c[1], c[2], hw, oct, c[0], c[1],
                                  c[2] );
                hw *= 0.5;
                k = child_key( k, oct );
                local_keys.push_back( k );
            }
        }
        std::sort( local_keys.begin(), local_keys.end() );
        local_keys.erase( std::unique( local_keys.begin(), local_keys.end() ),
                          local_keys.end() );
        int n_local_keys = static_cast<int>( local_keys.size() );
        std::vector<int> counts( nprocs ), displs( nprocs, 0 );
        MPI_Allgather( &n_local_keys, 1, MPI_INT, counts.data(), 1, MPI_INT,
                       MPI_COMM_WORLD );
        for ( int r = 1; r < nprocs; ++r )
            displs[r] = displs[r - 1] + counts[r - 1];
        std::vector<MortonKey> all_keys( displs[nprocs - 1] +
                                         counts[nprocs - 1] );
        static_assert( sizeof( MortonKey ) == sizeof( std::uint64_t ) );
        MPI_Allgatherv( local_keys.data(), n_local_keys, MPI_UINT64_T,
                        all_keys.data(), counts.data(), displs.data(),
                        MPI_UINT64_T, MPI_COMM_WORLD );
        const std::set<MortonKey> occupied( all_keys.begin(),
                                            all_keys.end() );

        // The occupancy model must reproduce the builder's tree: every cell
        // occupied, and an internal cell's children exactly its occupied
        // octants. Otherwise the cost model refines into the wrong cells.
        long long occupancy_mismatch = 0;
        for ( const auto& c : cells )
        {
            occupancy_mismatch += !occupied.count( c.key );
            if ( !c.is_leaf )
                for ( int oct = 0; oct < 8; ++oct )
                {
                    const MortonKey ck = child_key( c.key, oct );
                    occupancy_mismatch +=
                        tree.count( ck ) != occupied.count( ck );
                }
        }

        const std::vector<long long> hist =
            balance_level_hist( tree, max_depth );
        long long n_pairs = 0, ld_gt1 = 0;
        int ld_min = -1, ld_max = -1;
        for ( int ld = 0; ld <= max_depth; ++ld )
        {
            n_pairs += hist[ld];
            if ( ld > 1 )
                ld_gt1 += hist[ld];
            if ( hist[ld] > 0 )
            {
                if ( ld_min < 0 )
                    ld_min = ld;
                ld_max = ld;
            }
        }

        BalanceCost cost[3];
        int balanced_ld_max[3];
        long long balanced_cells[3];
        for ( int i = 0; i < 3; ++i )
        {
            BalanceTree balanced = tree;
            cost[i] = balance_simulate( balanced, deltas[i], max_depth,
                                        occupied );
            balanced_cells[i] = static_cast<long long>( balanced.size() );
            const auto h = balance_level_hist( balanced, max_depth );
            balanced_ld_max[i] = 0;
            for ( int ld = 0; ld <= max_depth; ++ld )
                if ( h[ld] > 0 )
                    balanced_ld_max[i] = ld;
        }

        std::string depth_str, hist_str, cost_str;
        for ( int d = 0; d <= max_depth; ++d )
        {
            depth_str += std::to_string( per_depth[d] );
            hist_str += std::to_string( hist[d] );
            if ( d < max_depth )
            {
                depth_str += ",";
                hist_str += ",";
            }
        }
        const long long n_cells = static_cast<long long>( cells.size() );
        for ( int i = 0; i < 3; ++i )
        {
            char buf[160];
            std::snprintf( buf, sizeof( buf ),
                           " d%d added %lld cells %lld mult %.4f passes %d "
                           "stuck %lld",
                           deltas[i], cost[i].added, balanced_cells[i],
                           double( balanced_cells[i] ) / double( n_cells ),
                           cost[i].passes, cost[i].stuck );
            cost_str += buf;
        }
        std::printf( "[a1-balance] draw %s nprocs %d rank %d cells %lld "
                     "leaves %lld cells_at_depth [%s] pairs %lld "
                     "ld_hist [%s] ld_min %d ld_max %d ld_gt1 %lld "
                     "dd_bound %d offset_bound %d%s\n",
                     to_string( draw ), nprocs, rank, n_cells, n_leaves,
                     depth_str.c_str(), n_pairs, hist_str.c_str(), ld_min,
                     ld_max, ld_gt1, dd_bound, offset_bound,
                     cost_str.c_str() );
        std::fflush( stdout );

        // The cell list is the global tree, so every figure above is the same
        // on every rank. Compare each rank's against the min and max over
        // ranks; a cell-list digest covers what the figures do not.
        std::uint64_t digest = 1469598103934665603ULL;
        auto mix = [&]( std::uint64_t v )
        {
            digest = ( digest ^ v ) * 1099511628211ULL;
        };
        for ( const auto& c : cells )
        {
            mix( c.key );
            mix( static_cast<std::uint64_t>( c.global_count ) );
            mix( c.is_leaf );
        }
        std::vector<long long> sig = { n_cells, n_leaves, n_pairs, ld_gt1,
                                       ld_min, ld_max,
                                       static_cast<long long>( digest >> 1 ),
                                       static_cast<long long>(
                                           occupied.size() ) };
        sig.insert( sig.end(), hist.begin(), hist.end() );
        for ( int i = 0; i < 3; ++i )
        {
            sig.push_back( cost[i].added );
            sig.push_back( cost[i].passes );
            sig.push_back( cost[i].stuck );
        }
        std::vector<long long> sig_min( sig.size() ), sig_max( sig.size() );
        MPI_Allreduce( sig.data(), sig_min.data(),
                       static_cast<int>( sig.size() ), MPI_LONG_LONG,
                       MPI_MIN, MPI_COMM_WORLD );
        MPI_Allreduce( sig.data(), sig_max.data(),
                       static_cast<int>( sig.size() ), MPI_LONG_LONG,
                       MPI_MAX, MPI_COMM_WORLD );
        EXPECT_EQ( sig_min, sig_max )
            << "draw " << to_string( draw ) << ": ranks disagree on the "
            << "global tree or on a figure computed from it";

        EXPECT_EQ( occupancy_mismatch, 0 )
            << "draw " << to_string( draw ) << ": the particle walk does not "
            << "reproduce builder.cells()' occupancy";

        for ( int i = 0; i < 3; ++i )
        {
            EXPECT_EQ( cost[i].added, balanced_cells[i] - n_cells );
            if ( cost[i].stuck == 0 )
            {
                EXPECT_LE( balanced_ld_max[i], deltas[i] )
                    << "draw " << to_string( draw ) << ": the simulated "
                    << "balance at delta " << deltas[i] << " left a pair "
                    << balanced_ld_max[i] << " levels apart";
            }
        }

        if ( draw != TwoScaleDraw::TwoScale )
            continue;

        // T1's tree: at np 1 the per-depth count is T1's cells_at_depth line.
        if ( nprocs == 1 )
        {
            EXPECT_EQ( depth_str, "1,8,42,2,8,11,8,30,135" )
                << "builder.cells() at np 1 is not the tree T1 recorded";
        }

        // The failure direction: a largest level difference of 1 means the
        // fixture is already balanced, and A2 would have nothing to test.
        EXPECT_GE( ld_max, 2 )
            << "T1's two-scale tree has no touching leaves 2 or more levels "
               "apart (ld_hist [" << hist_str << "])";
    }
}
} // namespace DownwardSweepTest

TEST( DownwardSweepTwoScale, treeHasShallowAndDeepLeaves )
{
    DownwardSweepTest::testTwoScaleTreeHasShallowAndDeepLeaves<
        TEST_MEMSPACE, TEST_EXECSPACE>();
}

TEST( DownwardSweepTwoScale, refusalsAreRangeGuard )
{
    DownwardSweepTest::testTwoScaleRefusalsAreRangeGuard<TEST_MEMSPACE,
                                                         TEST_EXECSPACE>();
}

TEST( DownwardSweepTwoScale, ddDuplicateColumns )
{
    DownwardSweepTest::testTwoScaleDdDuplicates<TEST_MEMSPACE,
                                                TEST_EXECSPACE>();
}

TEST( DownwardSweepTwoScale, ddDuplicateColumnsGraded )
{
    DownwardSweepTest::testGradedDdDuplicates<TEST_MEMSPACE,
                                              TEST_EXECSPACE>();
}

TEST( DownwardSweepTwoScale, rootWidthQuantizationRetainsCache )
{
    DownwardSweepTest::testRootWidthQuantizationRetainsCache<
        TEST_MEMSPACE, TEST_EXECSPACE>();
}

TEST( DownwardSweepTwoScale, balanceCost )
{
    DownwardSweepTest::testTwoScaleBalanceCost<TEST_MEMSPACE,
                                               TEST_EXECSPACE>();
}

//---------------------------------------------------------------------------//

} // end namespace Test
