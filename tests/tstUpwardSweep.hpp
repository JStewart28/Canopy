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

namespace UpwardSweepTest
{

enum FieldIdx
{
    Position = 0,
    Charge = 1
};

static constexpr int P_ORDER = 6;
using Kernel = LaplaceKernel<double, P_ORDER>;

// Charges are stored as double[NComps] so the AoSoA layout matches what
// UpwardSweep expects: particle_charges(p, comp_idx).
using DataTypes =
    Cabana::MemberTypes<double[3], double[Kernel::num_components]>;
using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
using AoSoA_ht = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;

// Generate particles with random positions in [0, 1)^3 and charges in [-1, 1].
void generate_test_particles( AoSoA_t& particles, int num_particles, int rank )
{
    AoSoA_ht particles_h( "particles_h", num_particles );
    auto h_pos = Cabana::slice<Position>( particles_h );
    auto h_q = Cabana::slice<Charge>( particles_h );

    std::mt19937 gen( 42 + rank );
    std::uniform_real_distribution<double> pos_dist( 0.0, 1.0 );
    std::uniform_real_distribution<double> q_dist( -1.0, 1.0 );

    for ( int i = 0; i < num_particles; i++ )
    {
        h_pos( i, 0 ) = pos_dist( gen );
        h_pos( i, 1 ) = pos_dist( gen );
        h_pos( i, 2 ) = pos_dist( gen );
        // Component 0 is the only component for a single Laplace solve.
        h_q( i, 0 ) = q_dist( gen );
    }

    particles.resize( num_particles );
    Cabana::deep_copy( particles, particles_h );
}

// Direct P2M: accumulate multipole coefficients from all particles in
// particles_h to the expansion center (cx, cy, cz).  Returns the result
// in M_ref, sized to Kernel::num_coeffs_per_cell.
// Only component 0 is used (single Laplace solve).
void direct_p2m_to_center( const AoSoA_ht& particles_h, int num_particles,
                           double cx, double cy, double cz,
                           std::vector<Kokkos::complex<double>>& M_ref )
{
    using complex = Kokkos::complex<double>;
    M_ref.assign( Kernel::num_coeffs_per_cell, complex( 0.0, 0.0 ) );

    auto h_pos = Cabana::slice<Position>( particles_h );
    auto h_q = Cabana::slice<Charge>( particles_h );

    for ( int p = 0; p < num_particles; p++ )
    {
        const double dx = h_pos( p, 0 ) - cx;
        const double dy = h_pos( p, 1 ) - cy;
        const double dz = h_pos( p, 2 ) - cz;

        const double rho = std::sqrt( dx * dx + dy * dy + dz * dz );
        const double theta = ( rho > 0 ) ? std::acos( dz / rho ) : 0.0;
        const double phi = std::atan2( dy, dx );
        // Component 0 only for a single Laplace solve.
        const double q = h_q( p, 0 );

        double rho_pow_n = 1.0;
        for ( int n = 0; n <= P_ORDER; n++ )
        {
            for ( int m = 0; m <= n; m++ )
            {
                const complex Y = Ynm<double>( n, -m, theta, phi );
                const int idx = coeff_index( n, m );
                M_ref[idx] += q * rho_pow_n * Y;
            }
            rho_pow_n *= rho;
        }
    }
}

} // namespace UpwardSweepTest

//---------------------------------------------------------------------------//
/**
 * Verify that the FMM root multipole produced by the full upward sweep
 * matches the reference multipole computed by direct P2M from all particles
 * directly to the root center.
 *
 * This equality is exact (up to floating-point round-off) for any tree
 * depth, any ncrit, and any particle distribution, because M2M is an exact
 * translation of the multipole expansion. The test is only meaningful with
 * one MPI rank (where each rank has the complete particle set) and is
 * silently skipped otherwise.
 *
 * Checks:
 *   1. The root cell is present in the sweep (cell_index() >= 0).
 *   2. Max relative error between FMM and direct P2M over all coefficients
 *      is below 1e-10.
 */
void testRootMultipoleMatchesDirectP2M( int num_particles_per_rank, int ncrit,
                                        int max_depth, double tolerance,
                                        int replication_depth )
{
    using namespace UpwardSweepTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    if ( nprocs != 1 )
        return;

    AoSoA_t particles( "particles", num_particles_per_rank );
    generate_test_particles( particles, num_particles_per_rank, rank );

    auto positions = Cabana::slice<Position>( particles );
    auto charges = Cabana::slice<Charge>( particles );

    // Phase 1: Build tree
    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );
    builder.build( positions, num_particles_per_rank );

    // Phase 2: Partition
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    partitioner.partition( builder, particles, num_particles_per_rank );
    int num_local = partitioner.num_local_particles();

    // Reslice after migration and rebuild tree on redistributed particles
    positions = Cabana::slice<Position>( particles );
    charges = Cabana::slice<Charge>( particles );
    builder.build( positions, num_local );

    // Phase 3: Build communication plan
    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> comm_plan(
        MPI_COMM_WORLD );
    comm_plan.build( builder.cells(), partitioner.ownership(),
                     partitioner.cell_owner_map(), replication_depth );

    // Phase 4: Setup and execute upward sweep
    UpwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> sweep( MPI_COMM_WORLD );
    sweep.setup( builder.cells(), partitioner.cell_owner_map(),
                 builder.particle_keys(), num_local );
    sweep.execute( charges, positions, comm_plan );

    // Phase 5: Compute reference via direct P2M to root center
    const BoundingBox& box = builder.root_box();
    const double cx = 0.5 * ( box.min[0] + box.max[0] );
    const double cy = 0.5 * ( box.min[1] + box.max[1] );
    const double cz = 0.5 * ( box.min[2] + box.max[2] );

    AoSoA_ht particles_h( "particles_h", num_local );
    Cabana::deep_copy( particles_h, particles );

    std::vector<Kokkos::complex<double>> M_ref;
    direct_p2m_to_center( particles_h, num_local, cx, cy, cz, M_ref );

    // Phase 6: Compare coefficient by coefficient.
    // Multipoles are stored as (cell_idx, coeff_idx, comp_idx); use comp 0.
    int root_idx = sweep.cell_index( ROOT_KEY );
    ASSERT_GE( root_idx, 0 ) << "Root cell not found in sweep index";

    auto h_M = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                    sweep.multipoles() );

    double max_rel_err = 0.0;
    for ( int idx = 0; idx < Kernel::num_coeffs_per_cell; idx++ )
    {
        auto fmm = h_M( root_idx, idx, 0 );
        auto ref = M_ref[idx];
        auto diff = fmm - ref;

        const double abs_err =
            std::sqrt( diff.real() * diff.real() + diff.imag() * diff.imag() );
        const double ref_mag =
            std::sqrt( ref.real() * ref.real() + ref.imag() * ref.imag() );
        const double rel_err =
            ( ref_mag > 1e-14 ) ? abs_err / ref_mag : abs_err;

        if ( rel_err > max_rel_err )
            max_rel_err = rel_err;
    }

    EXPECT_LT( max_rel_err, 1.0e-10 )
        << "FMM root multipole deviates from direct P2M; "
           "max relative error = "
        << max_rel_err;
}

//---------------------------------------------------------------------------//
/**
 * Verify that the root multipole is non-zero after a sweep over a particle
 * distribution with strictly positive charges.
 *
 * If the sweep incorrectly leaves coefficients zeroed or fails to accumulate
 * contributions from leaf cells upward, the root will be all-zeros even for
 * a non-trivial source distribution. Using positive-only charges ensures the
 * monopole moment (and therefore at least one coefficient) is non-zero,
 * ruling out a false pass due to accidental charge cancellation.
 *
 * Checks:
 *   1. The root cell is present in the sweep (cell_index() >= 0).
 *   2. At least one root multipole coefficient has magnitude > 0.
 */
void testMultipolesNonzeroAfterSweep( int num_particles_per_rank, int ncrit,
                                      int max_depth, double tolerance,
                                      int replication_depth )
{
    using namespace UpwardSweepTest;

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    // Build particles with strictly positive charges so the monopole
    // moment is guaranteed non-zero after an MPI_Allreduce across ranks.
    AoSoA_ht particles_h( "particles_h", num_particles_per_rank );
    {
        auto h_pos = Cabana::slice<Position>( particles_h );
        auto h_q = Cabana::slice<Charge>( particles_h );

        std::mt19937 gen( 99 + rank );
        std::uniform_real_distribution<double> pos_dist( 0.0, 1.0 );
        std::uniform_real_distribution<double> q_dist( 0.1, 1.0 );

        for ( int i = 0; i < num_particles_per_rank; i++ )
        {
            h_pos( i, 0 ) = pos_dist( gen );
            h_pos( i, 1 ) = pos_dist( gen );
            h_pos( i, 2 ) = pos_dist( gen );
            h_q( i, 0 ) = q_dist( gen );
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
        MPI_COMM_WORLD );
    comm_plan.build( builder.cells(), partitioner.ownership(),
                     partitioner.cell_owner_map(), replication_depth );

    UpwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> sweep( MPI_COMM_WORLD );
    sweep.setup( builder.cells(), partitioner.cell_owner_map(),
                 builder.particle_keys(), num_local );
    sweep.execute( charges, positions, comm_plan );

    int root_idx = sweep.cell_index( ROOT_KEY );
    ASSERT_GE( root_idx, 0 ) << "Root cell not found in sweep index";

    auto h_M = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                    sweep.multipoles() );

    double max_abs = 0.0;
    for ( int idx = 0; idx < Kernel::num_coeffs_per_cell; idx++ )
    {
        const auto c = h_M( root_idx, idx, 0 );
        const double m = std::sqrt( c.real() * c.real() + c.imag() * c.imag() );
        if ( m > max_abs )
            max_abs = m;
    }

    EXPECT_GT( max_abs, 0.0 )
        << "All root multipole coefficients are zero after sweep "
           "with non-zero charges";
}

//---------------------------------------------------------------------------//
/**
 * Verify that calling execute() a second time on the same sweep object
 * and inputs produces bit-identical multipole coefficients.
 *
 * execute() must zero the coefficient storage at the start of each call, so
 * the result must not accumulate across invocations or depend on leftover
 * state from a prior run.
 *
 * Checks:
 *   1. Real and imaginary parts of every coefficient in every cell match
 *      exactly between the first and second execute() calls.
 */
void testIdempotentExecution( int num_particles_per_rank, int ncrit,
                              int max_depth, double tolerance,
                              int replication_depth )
{
    using namespace UpwardSweepTest;

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
        MPI_COMM_WORLD );
    comm_plan.build( builder.cells(), partitioner.ownership(),
                     partitioner.cell_owner_map(), replication_depth );

    UpwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> sweep( MPI_COMM_WORLD );
    sweep.setup( builder.cells(), partitioner.cell_owner_map(),
                 builder.particle_keys(), num_local );

    // First execute — snapshot the multipoles
    sweep.execute( charges, positions, comm_plan );
    auto h_M_first = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          sweep.multipoles() );

    // Second execute — must give the same result
    sweep.execute( charges, positions, comm_plan );
    auto h_M_second = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                           sweep.multipoles() );

    const int num_cells = static_cast<int>( h_M_first.extent( 0 ) );
    for ( int c = 0; c < num_cells; c++ )
    {
        for ( int idx = 0; idx < Kernel::num_coeffs_per_cell; idx++ )
        {
            EXPECT_EQ( h_M_first( c, idx, 0 ).real(),
                       h_M_second( c, idx, 0 ).real() )
                << "Real part mismatch at cell " << c << ", coeff " << idx
                << " between first and second execute()";
            EXPECT_EQ( h_M_first( c, idx, 0 ).imag(),
                       h_M_second( c, idx, 0 ).imag() )
                << "Imaginary part mismatch at cell " << c << ", coeff " << idx
                << " between first and second execute()";
        }
    }
}

//---------------------------------------------------------------------------//
/**
 * Verify that zero-valued particle charges produce all-zero multipole
 * coefficients across the entire tree.
 *
 * If every particle carries charge 0, P2M contributes nothing to any leaf.
 * The subsequent M2M translations of all-zero leaf multipoles must remain
 * zero at every level up to the root. Any non-zero entry indicates that
 * the implementation is reading uninitialized memory or failing to zero
 * the coefficient storage at the start of execute().
 *
 * Checks:
 *   1. Every coefficient in every cell is exactly zero after execute().
 */
void testZeroChargesGiveZeroMultipoles( int num_particles_per_rank, int ncrit,
                                        int max_depth, double tolerance,
                                        int replication_depth )
{
    using namespace UpwardSweepTest;

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    // Random positions but all charges set to zero
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
        MPI_COMM_WORLD );
    comm_plan.build( builder.cells(), partitioner.ownership(),
                     partitioner.cell_owner_map(), replication_depth );

    UpwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> sweep( MPI_COMM_WORLD );
    sweep.setup( builder.cells(), partitioner.cell_owner_map(),
                 builder.particle_keys(), num_local );
    sweep.execute( charges, positions, comm_plan );

    auto h_M = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                    sweep.multipoles() );

    const int num_cells = static_cast<int>( h_M.extent( 0 ) );
    for ( int c = 0; c < num_cells; c++ )
    {
        for ( int idx = 0; idx < Kernel::num_coeffs_per_cell; idx++ )
        {
            EXPECT_EQ( h_M( c, idx, 0 ).real(), 0.0 )
                << "Non-zero real part at cell " << c << ", coeff " << idx
                << " with all-zero charges";
            EXPECT_EQ( h_M( c, idx, 0 ).imag(), 0.0 )
                << "Non-zero imaginary part at cell " << c << ", coeff " << idx
                << " with all-zero charges";
        }
    }
}

//---------------------------------------------------------------------------//
/**
 * Verify that the FMM root multipole matches a globally-gathered direct P2M
 * reference when running on multiple MPI ranks.
 *
 * After partitioning, each rank owns only a subset of the global particle
 * set. The upward sweep exchanges multipole data between ranks at layer
 * boundaries and allreduces shared cells (depth <= replication_depth),
 * so the root multipole — which is always a shared cell — must equal the
 * expansion computed by applying P2M to every particle in the simulation
 * directly to the root center.
 *
 * To form the global reference, each rank copies its local post-migration
 * particles to host and packs them into a flat buffer. MPI_Gatherv collects
 * all positions and charges onto rank 0, which then computes the reference
 * direct P2M and compares it to the FMM root multipole. Ranks other than
 * rank 0 participate in the gather but skip the comparison. The test is
 * silently skipped on a single rank (testRootMultipoleMatchesDirectP2M
 * covers that case).
 *
 * Checks (on rank 0):
 *   1. The root cell is present in the sweep (cell_index() >= 0).
 *   2. Max relative error between FMM and globally-gathered direct P2M
 *      over all coefficients is below 1e-10.
 */
void testRootMultipoleMatchesDirectP2MMultiRank( int num_particles_per_rank,
                                                 int ncrit, int max_depth,
                                                 double tolerance,
                                                 int replication_depth )
{
    using namespace UpwardSweepTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    if ( nprocs == 1 )
        return;

    AoSoA_t particles( "particles", num_particles_per_rank );
    generate_test_particles( particles, num_particles_per_rank, rank );

    auto positions = Cabana::slice<Position>( particles );
    auto charges = Cabana::slice<Charge>( particles );

    // Phase 1: Build tree
    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );
    builder.build( positions, num_particles_per_rank );

    // Phase 2: Partition
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    partitioner.partition( builder, particles, num_particles_per_rank );
    int num_local = partitioner.num_local_particles();

    // Reslice after migration and rebuild tree on redistributed particles
    positions = Cabana::slice<Position>( particles );
    charges = Cabana::slice<Charge>( particles );
    builder.build( positions, num_local );

    // Phase 3: Build communication plan
    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> comm_plan(
        MPI_COMM_WORLD );
    comm_plan.build( builder.cells(), partitioner.ownership(),
                     partitioner.cell_owner_map(), replication_depth );

    // Phase 4: Setup and execute upward sweep
    UpwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, Kernel> sweep( MPI_COMM_WORLD );
    sweep.setup( builder.cells(), partitioner.cell_owner_map(),
                 builder.particle_keys(), num_local );
    sweep.execute( charges, positions, comm_plan );

    // Phase 5: Gather all post-migration particle data to rank 0
    //
    // Each rank packs its local positions (3 doubles/particle) and charges
    // (1 double/particle, component 0) into flat host buffers, then
    // MPI_Gatherv sends them to rank 0 for the global reference computation.

    // Workaround to copy the device-side position slice into a host view
    Kokkos::View<double* [3], TEST_MEMSPACE> d_pos( "d_pos", num_local );
    Kokkos::parallel_for(
        "SlicePosToView", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_local ),
        KOKKOS_LAMBDA( int i ) {
            d_pos( i, 0 ) = positions( i, 0 );
            d_pos( i, 1 ) = positions( i, 1 );
            d_pos( i, 2 ) = positions( i, 2 );
        } );
    Kokkos::fence();
    auto h_pos =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_pos );

    // Workaround to copy the device-side charge slice (component 0) into a
    // host view
    Kokkos::View<double*, TEST_MEMSPACE> d_crg( "d_crg", num_local );
    Kokkos::parallel_for(
        "SliceCrgToView", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_local ),
        KOKKOS_LAMBDA( int i ) { d_crg( i ) = charges( i, 0 ); } );
    Kokkos::fence();
    auto h_q =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_crg );

    std::vector<double> local_pos_buf( 3 * num_local );
    std::vector<double> local_q_buf( num_local );
    for ( int i = 0; i < num_local; i++ )
    {
        local_pos_buf[3 * i + 0] = h_pos( i, 0 );
        local_pos_buf[3 * i + 1] = h_pos( i, 1 );
        local_pos_buf[3 * i + 2] = h_pos( i, 2 );
        local_q_buf[i] = h_q( i );
    }

    // Gather particle counts to rank 0
    std::vector<int> all_num_local( nprocs, 0 );
    MPI_Gather( &num_local, 1, MPI_INT, all_num_local.data(), 1, MPI_INT, 0,
                MPI_COMM_WORLD );

    // Build displacement arrays and receive buffers on rank 0
    int total_particles = 0;
    std::vector<int> pos_counts( nprocs, 0 ), pos_displs( nprocs, 0 );
    std::vector<int> q_counts( nprocs, 0 ), q_displs( nprocs, 0 );
    std::vector<double> gathered_pos, gathered_q;

    if ( rank == 0 )
    {
        for ( int r = 0; r < nprocs; r++ )
        {
            pos_counts[r] = 3 * all_num_local[r];
            q_counts[r] = all_num_local[r];
            total_particles += all_num_local[r];
        }
        for ( int r = 1; r < nprocs; r++ )
        {
            pos_displs[r] = pos_displs[r - 1] + pos_counts[r - 1];
            q_displs[r] = q_displs[r - 1] + q_counts[r - 1];
        }
        gathered_pos.resize( 3 * total_particles );
        gathered_q.resize( total_particles );
    }

    MPI_Gatherv( local_pos_buf.data(), 3 * num_local, MPI_DOUBLE,
                 gathered_pos.data(), pos_counts.data(), pos_displs.data(),
                 MPI_DOUBLE, 0, MPI_COMM_WORLD );

    MPI_Gatherv( local_q_buf.data(), num_local, MPI_DOUBLE, gathered_q.data(),
                 q_counts.data(), q_displs.data(), MPI_DOUBLE, 0,
                 MPI_COMM_WORLD );

    // Phase 6: On rank 0, compute reference and compare to FMM root multipole
    if ( rank == 0 )
    {
        // Populate a host AoSoA from the gathered flat buffers
        AoSoA_ht all_particles_h( "all_particles_h", total_particles );
        auto all_pos = Cabana::slice<Position>( all_particles_h );
        auto all_q = Cabana::slice<Charge>( all_particles_h );
        for ( int i = 0; i < total_particles; i++ )
        {
            all_pos( i, 0 ) = gathered_pos[3 * i + 0];
            all_pos( i, 1 ) = gathered_pos[3 * i + 1];
            all_pos( i, 2 ) = gathered_pos[3 * i + 2];
            all_q( i, 0 ) = gathered_q[i];
        }

        // Root cell center (the global bounding box is the same on all ranks)
        const BoundingBox& box = builder.root_box();
        const double cx = 0.5 * ( box.min[0] + box.max[0] );
        const double cy = 0.5 * ( box.min[1] + box.max[1] );
        const double cz = 0.5 * ( box.min[2] + box.max[2] );

        std::vector<Kokkos::complex<double>> M_ref;
        direct_p2m_to_center( all_particles_h, total_particles, cx, cy, cz,
                              M_ref );

        int root_idx = sweep.cell_index( ROOT_KEY );
        ASSERT_GE( root_idx, 0 ) << "Root cell not found in sweep index";

        auto h_M = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                        sweep.multipoles() );

        double max_rel_err = 0.0;
        for ( int idx = 0; idx < Kernel::num_coeffs_per_cell; idx++ )
        {
            auto fmm = h_M( root_idx, idx, 0 );
            auto ref = M_ref[idx];
            auto diff = fmm - ref;

            const double abs_err = std::sqrt( diff.real() * diff.real() +
                                              diff.imag() * diff.imag() );
            const double ref_mag =
                std::sqrt( ref.real() * ref.real() + ref.imag() * ref.imag() );
            const double rel_err =
                ( ref_mag > 1e-14 ) ? abs_err / ref_mag : abs_err;

            if ( rel_err > max_rel_err )
                max_rel_err = rel_err;
        }

        EXPECT_LT( max_rel_err, 1.0e-10 )
            << "FMM root multipole deviates from globally-gathered direct "
               "P2M reference; max relative error = "
            << max_rel_err;
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//

TEST( UpwardSweep, testRootMultipoleMatchesDirectP2MBasic )
{
    testRootMultipoleMatchesDirectP2M( 1000, 32, 6, 0.1, 2 );
}

TEST( UpwardSweep, testRootMultipoleMatchesDirectP2MSmall )
{
    testRootMultipoleMatchesDirectP2M( 200, 16, 4, 0.1, 1 );
}

TEST( UpwardSweep, testRootMultipoleMatchesDirectP2MMultiRankBasic )
{
    testRootMultipoleMatchesDirectP2MMultiRank( 1000, 32, 6, 0.1, 2 );
}

TEST( UpwardSweep, testRootMultipoleMatchesDirectP2MMultiRankSmall )
{
    testRootMultipoleMatchesDirectP2MMultiRank( 200, 16, 4, 0.1, 1 );
}

TEST( UpwardSweep, testMultipolesNonzeroAfterSweepBasic )
{
    testMultipolesNonzeroAfterSweep( 1000, 32, 6, 0.1, 2 );
}

TEST( UpwardSweep, testMultipolesNonzeroAfterSweepSmall )
{
    testMultipolesNonzeroAfterSweep( 200, 16, 4, 0.1, 1 );
}

TEST( UpwardSweep, testIdempotentExecutionBasic )
{
    testIdempotentExecution( 1000, 32, 6, 0.1, 2 );
}

TEST( UpwardSweep, testIdempotentExecutionSmall )
{
    testIdempotentExecution( 200, 16, 4, 0.1, 1 );
}

TEST( UpwardSweep, testZeroChargesGiveZeroMultipolesBasic )
{
    testZeroChargesGiveZeroMultipoles( 1000, 32, 6, 0.1, 2 );
}

TEST( UpwardSweep, testZeroChargesGiveZeroMultipolesSmall )
{
    testZeroChargesGiveZeroMultipoles( 200, 16, 4, 0.1, 1 );
}

//---------------------------------------------------------------------------//

} // end namespace Test
