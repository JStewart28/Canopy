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

#include <Canopy_TreeBuilder.hpp>
#include <Canopy_TreePartitioner.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <unordered_map>

namespace Test
{
//---------------------------------------------------------------------------//

using namespace Canopy;

namespace TreePartitionerTest
{

enum FieldIdx
{
    Position = 0
};

using DataTypes = Cabana::MemberTypes<double[3]>;
using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
using AoSoA_ht = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;

void generate_test_particles( AoSoA_t& particles, int num_particles, int rank )
{
    AoSoA_ht particles_h( "particles_h", num_particles );
    auto h_positions = Cabana::slice<Position>( particles_h );

    std::mt19937 gen( 42 + rank );
    std::uniform_real_distribution<double> uniform( 0.0, 1.0 );
    std::normal_distribution<double> clustered( 0.1, 0.02 );

    for ( int i = 0; i < num_particles; ++i )
    {
        if ( uniform( gen ) < 0.3 )
        {
            h_positions( i, 0 ) = std::clamp( clustered( gen ), 0.0, 1.0 );
            h_positions( i, 1 ) = std::clamp( clustered( gen ), 0.0, 1.0 );
            h_positions( i, 2 ) = std::clamp( clustered( gen ), 0.0, 1.0 );
        }
        else
        {
            h_positions( i, 0 ) = uniform( gen );
            h_positions( i, 1 ) = uniform( gen );
            h_positions( i, 2 ) = uniform( gen );
        }
    }

    particles.resize( num_particles );
    Cabana::deep_copy( particles, particles_h );
}

// Regenerate particles with a fresh distribution shifted to a different
// spatial region. Used to simulate a scenario where particles have moved
// enough that the tree topology should change significantly.
void regenerate_shifted_particles( AoSoA_t& particles, int num_particles,
                                   int seed )
{
    AoSoA_ht particles_h( "particles_h", num_particles );
    auto h_positions = Cabana::slice<Position>( particles_h );

    std::mt19937 gen( seed );
    std::uniform_real_distribution<double> uniform( 0.0, 1.0 );
    // New cluster centered at the OPPOSITE corner from generate_test_particles.
    std::normal_distribution<double> new_cluster( 0.85, 0.02 );

    for ( int i = 0; i < num_particles; ++i )
    {
        if ( uniform( gen ) < 0.3 )
        {
            h_positions( i, 0 ) = std::clamp( new_cluster( gen ), 0.0, 1.0 );
            h_positions( i, 1 ) = std::clamp( new_cluster( gen ), 0.0, 1.0 );
            h_positions( i, 2 ) = std::clamp( new_cluster( gen ), 0.0, 1.0 );
        }
        else
        {
            h_positions( i, 0 ) = uniform( gen );
            h_positions( i, 1 ) = uniform( gen );
            h_positions( i, 2 ) = uniform( gen );
        }
    }

    particles.resize( num_particles );
    Cabana::deep_copy( particles, particles_h );
}

// Check that every local particle's leaf key is owned by this rank (or is
// OWNER_SHARED). Rebuilds the tree on migrated particles to get fresh keys.
int count_particles_in_nonowned_leaves(
    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE>& builder,
    const TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE>& partitioner,
    AoSoA_t& particles, int num_local, int rank )
{
    auto positions = Cabana::slice<Position>( particles );
    builder.build( positions, num_local );

    auto h_keys = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), builder.particle_keys() );

    int bad = 0;
    for ( int i = 0; i < num_local; ++i )
    {
        int owner = partitioner.cell_owner( h_keys( i ) );
        if ( owner != rank && owner != OWNER_SHARED )
            bad++;
    }
    return bad;
}

// ---------------------------------------------------------------------------
// Payload-carrying particle type for the coalesced-migration integrity test.
// Position drives the partition; the payload travels with each particle so we
// can verify the generic tuple pack/unpack neither loses, duplicates, nor
// corrupts data (and does not swap fields within a tuple).
// ---------------------------------------------------------------------------
enum FieldIdxP
{
    PositionP = 0,
    PayloadP = 1
};

using DataTypesP = Cabana::MemberTypes<double[3], double[2]>;
using AoSoAP_t = Cabana::AoSoA<DataTypesP, TEST_MEMSPACE>;
using AoSoAP_ht = Cabana::AoSoA<DataTypesP, Kokkos::HostSpace>;

// payload(i,1) is a deterministic function of payload(i,0); after migration we
// re-check this relation to confirm the two payload slots stayed paired with
// each other and with their particle. Exact in double for the sizes tested.
inline double payload_check( double gid ) { return gid * 3.0 + 1.0; }

void generate_payload_particles( AoSoAP_t& particles, int num_particles,
                                 int rank, int global_base )
{
    AoSoAP_ht particles_h( "particles_h", num_particles );
    auto h_pos = Cabana::slice<PositionP>( particles_h );
    auto h_pay = Cabana::slice<PayloadP>( particles_h );

    // Uniform over the whole domain so that, after an RCB partition into
    // nprocs parts, each rank scatters particles to many peers (exercises the
    // multi-peer coalesced exchange).
    std::mt19937 gen( 1234 + rank );
    std::uniform_real_distribution<double> uniform( 0.0, 1.0 );
    for ( int i = 0; i < num_particles; ++i )
    {
        h_pos( i, 0 ) = uniform( gen );
        h_pos( i, 1 ) = uniform( gen );
        h_pos( i, 2 ) = uniform( gen );
        const double gid = static_cast<double>( global_base + i );
        h_pay( i, 0 ) = gid;
        h_pay( i, 1 ) = payload_check( gid );
    }

    particles.resize( num_particles );
    Cabana::deep_copy( particles, particles_h );
}

void generate_uniform_particles( AoSoA_t& particles, int num_particles,
                                 int rank )
{
    AoSoA_ht particles_h( "particles_h", num_particles );
    auto h_positions = Cabana::slice<Position>( particles_h );
    std::mt19937 gen( 7 + 31 * rank );
    std::uniform_real_distribution<double> uniform( 0.0, 1.0 );
    for ( int i = 0; i < num_particles; ++i )
        for ( int d = 0; d < 3; ++d )
            h_positions( i, d ) = uniform( gen );
    particles.resize( num_particles );
    Cabana::deep_copy( particles, particles_h );
}

bool owner_map_agrees_across_ranks(
    const std::unordered_map<MortonKey, int>& owners )
{
    const uint64_t h = owner_map_hash( owners );
    uint64_t hmin = h, hmax = h;
    MPI_Allreduce( MPI_IN_PLACE, &hmin, 1, MPI_UINT64_T, MPI_MIN,
                   MPI_COMM_WORLD );
    MPI_Allreduce( MPI_IN_PLACE, &hmax, 1, MPI_UINT64_T, MPI_MAX,
                   MPI_COMM_WORLD );
    return hmin == hmax;
}

// Fraction of parent-child pairs, both deeper than replication_depth or a
// leaf (the partition's vertices), whose owners differ.
double parent_child_cut_fraction(
    const std::vector<CellInfo>& cells,
    const std::unordered_map<MortonKey, int>& owners, int replication_depth )
{
    std::unordered_map<MortonKey, const CellInfo*> by_key;
    for ( const auto& c : cells )
        by_key[c.key] = &c;
    long long pairs = 0, cut = 0;
    for ( const auto& c : cells )
    {
        if ( c.depth == 0 || !( c.is_leaf || c.depth > replication_depth ) )
            continue;
        const CellInfo* p = by_key.at( parent_key( c.key ) );
        if ( p->depth <= replication_depth )
            continue;
        pairs++;
        if ( owners.at( c.key ) != owners.at( p->key ) )
            cut++;
    }
    return pairs ? static_cast<double>( cut ) / pairs : 0.0;
}

} // namespace TreePartitionerTest

//---------------------------------------------------------------------------//
/**
 * Coalesced-migration integrity test. Each particle carries a globally-unique
 * id (payload slot 0) and a derived check value (payload slot 1). After an
 * initial partition() — which runs the RegisteredBufferPool-backed coalesced
 * migrate_particles() — verify:
 *
 *   1. Total particle count is conserved.
 *   2. Every global id appears EXACTLY once across all ranks (a true bijection
 *      check: no loss, no duplication). Uses an all-reduced per-id histogram.
 *   3. For every migrated particle, payload(1) == f(payload(0)) — the tuple's
 *      two payload slots stayed paired (no field swap/corruption in the
 *      generic pack/unpack).
 *   4. Every local particle resides in a leaf owned by this rank (or shared).
 *
 * Runs the migration across many peers when launched at >= 4 ranks.
 */
void testCoalescedMigrateIntegrity( int num_particles_per_rank, int ncrit,
                                    int max_depth, double tolerance,
                                    int replication_depth )
{
    using namespace TreePartitionerTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    const int global_n = num_particles_per_rank * nprocs;
    const int global_base = rank * num_particles_per_rank;

    AoSoAP_t particles( "particles", num_particles_per_rank );
    generate_payload_particles( particles, num_particles_per_rank, rank,
                                global_base );

    auto positions = Cabana::slice<PositionP>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth,
        std::array<double, 6>{ tolerance, tolerance, tolerance, tolerance,
                               tolerance, tolerance },
        tolerance );
    builder.build( positions, num_particles_per_rank );

    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );

    // Runs the coalesced migrate_particles().
    partitioner.partition( builder, particles, num_particles_per_rank );

    const int new_local = partitioner.num_local_particles();

    // Check 1: total count conserved.
    int total_after = 0;
    MPI_Allreduce( &new_local, &total_after, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );
    EXPECT_EQ( total_after, global_n )
        << "Total particle count changed during coalesced migration";

    // Pull the migrated particles to host for payload inspection.
    AoSoAP_ht particles_h( "particles_h", new_local );
    Cabana::deep_copy( particles_h, particles );
    auto h_pay = Cabana::slice<PayloadP>( particles_h );

    // Checks 2 & 3: build a local id histogram and verify payload pairing.
    std::vector<int> local_hist( global_n > 0 ? global_n : 1, 0 );
    int payload_mismatches = 0;
    int id_out_of_range = 0;
    for ( int i = 0; i < new_local; ++i )
    {
        const double gid = h_pay( i, 0 );
        if ( h_pay( i, 1 ) != payload_check( gid ) )
            payload_mismatches++;
        const long long id = static_cast<long long>( gid );
        if ( id < 0 || id >= global_n || static_cast<double>( id ) != gid )
            id_out_of_range++;
        else
            local_hist[id]++;
    }

    EXPECT_EQ( payload_mismatches, 0 )
        << "Rank " << rank << ": " << payload_mismatches
        << " migrated particles have payload(1) != f(payload(0)) — tuple "
           "pack/unpack corrupted or swapped fields";
    EXPECT_EQ( id_out_of_range, 0 )
        << "Rank " << rank << ": " << id_out_of_range
        << " migrated particles have a corrupted global id";

    std::vector<int> global_hist( global_n > 0 ? global_n : 1, 0 );
    MPI_Allreduce( local_hist.data(), global_hist.data(), global_n, MPI_INT,
                   MPI_SUM, MPI_COMM_WORLD );

    if ( rank == 0 )
    {
        int missing = 0, duplicated = 0;
        for ( int id = 0; id < global_n; ++id )
        {
            if ( global_hist[id] == 0 )
                missing++;
            else if ( global_hist[id] > 1 )
                duplicated++;
        }
        EXPECT_EQ( missing, 0 )
            << missing << " global ids vanished during migration";
        EXPECT_EQ( duplicated, 0 )
            << duplicated << " global ids were duplicated during migration";
    }

    // Check 4: every local particle sits in a leaf owned by this rank.
    positions = Cabana::slice<PositionP>( particles );
    builder.build( positions, new_local );
    auto h_keys = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), builder.particle_keys() );
    int bad_count = 0;
    for ( int i = 0; i < new_local; ++i )
    {
        int owner = partitioner.cell_owner( h_keys( i ) );
        if ( owner != rank && owner != OWNER_SHARED )
            bad_count++;
    }
    EXPECT_EQ( bad_count, 0 )
        << "Rank " << rank << ": " << bad_count
        << " particles reside in a leaf not owned by this rank after migration";
}

//---------------------------------------------------------------------------//
/**
 * Build an adaptive octree, partition its leaves across MPI ranks with
 * TreePartitioner, then verify:
 *
 *   1. The ownership vector has one entry per cell in the tree.
 *   2. Cells at depth <= replication_depth are OWNER_SHARED; all deeper
 *      cells have an owner in [0, nprocs).
 *   3. The total particle count is conserved across ranks after migration.
 *   4. After rebuilding the tree on the new particle distribution, every
 *      particle's leaf is owned by the local rank (or is OWNER_SHARED).
 */
void testPartitioner( int num_particles_per_rank, int ncrit, int max_depth,
                      double tolerance, int replication_depth )
{
    using namespace TreePartitionerTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    AoSoA_t particles( "particles", num_particles_per_rank );
    generate_test_particles( particles, num_particles_per_rank, rank );

    auto positions = Cabana::slice<Position>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );

    builder.build( positions, num_particles_per_rank );

    ASSERT_GT( builder.cells().size(), 0u )
        << "Tree has no cells after initial build";

    // -----------------------------------------------------------------------
    // Partition
    // -----------------------------------------------------------------------
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );

    partitioner.partition( builder, particles, num_particles_per_rank );

    // -----------------------------------------------------------------------
    // Check 1: ownership vector aligns with cells vector
    // -----------------------------------------------------------------------
    ASSERT_EQ( partitioner.ownership().size(), builder.cells().size() )
        << "Ownership vector size does not match cells vector";

    // -----------------------------------------------------------------------
    // Check 2: every cell has a valid ownership assignment
    // -----------------------------------------------------------------------
    for ( std::size_t i = 0; i < builder.cells().size(); ++i )
    {
        const auto& c = builder.cells()[i];
        int owner = partitioner.ownership()[i].owner_rank;

        if ( owner == OWNER_SHARED )
        {
            // Only coarse layers may be replicated
            EXPECT_LE( c.depth, replication_depth )
                << "Cell at depth " << c.depth
                << " is OWNER_SHARED but exceeds replication_depth "
                << replication_depth;
        }
        else
        {
            EXPECT_GE( owner, 0 ) << "Cell " << i << " has negative owner rank";
            EXPECT_LT( owner, nprocs ) << "Cell " << i << " owner " << owner
                                       << " >= nprocs " << nprocs;
        }
    }

    // -----------------------------------------------------------------------
    // Check 3: total particle count is conserved after migration
    // -----------------------------------------------------------------------
    int new_local_count = partitioner.num_local_particles();
    EXPECT_GE( new_local_count, 0 );

    int total_after = 0;
    MPI_Allreduce( &new_local_count, &total_after, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );
    EXPECT_EQ( total_after, num_particles_per_rank * nprocs )
        << "Total particle count changed during migration";

    // -----------------------------------------------------------------------
    // Check 4: rebuild tree on migrated particles and confirm every local
    //          particle's leaf is owned by this rank (or is shared)
    // -----------------------------------------------------------------------
    positions = Cabana::slice<Position>( particles );
    builder.build( positions, new_local_count );

    auto h_keys = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), builder.particle_keys() );

    int bad_count = 0;
    for ( int i = 0; i < new_local_count; ++i )
    {
        MortonKey key = h_keys( i );
        int owner = partitioner.cell_owner( key );
        if ( owner != rank && owner != OWNER_SHARED )
            bad_count++;
    }

    EXPECT_EQ( bad_count, 0 )
        << "Rank " << rank << ": " << bad_count
        << " particles reside in a leaf not owned by this rank";
}

//---------------------------------------------------------------------------//
/**
 * Test TreePartitioner::redistribute() — the lightweight per-timestep migration
 * path. Immediately after an initial partition, every particle already lives
 * in a locally-owned leaf, so redistribute() should detect that and move no
 * particles.
 *
 * This specifically exercises redistribute()'s fast path — reusing the
 * existing ownership map and moving only particles whose current leaf key
 * no longer matches the local rank. With no particle motion between
 * partition() and redistribute(), the expected result is a pure no-op.
 *
 * Verifies:
 *   1. Total particle count is conserved.
 *   2. RedistributeResult reports 0 sent and 0 received globally.
 *   3. num_local_after matches on both sides.
 *   4. Every local particle still sits in a leaf owned by this rank (or
 *      OWNER_SHARED).
 */
void testRedistributeNoOp( int num_particles_per_rank, int ncrit, int max_depth,
                           double tolerance, int replication_depth )
{
    using namespace TreePartitionerTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    AoSoA_t particles( "particles", num_particles_per_rank );
    generate_test_particles( particles, num_particles_per_rank, rank );

    auto positions = Cabana::slice<Position>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );
    builder.build( positions, num_particles_per_rank );

    // Initial partition. This migrates particles onto their owning ranks and
    // establishes the ownership map used by redistribute().
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    partitioner.partition( builder, particles, num_particles_per_rank );

    int num_after_partition = partitioner.num_local_particles();

    // Rebuild the tree on the migrated local particle set so
    // builder.particle_keys() correctly describes the current particles.
    // The global tree topology is unchanged because the global particle set
    // is unchanged (just redistributed).
    positions = Cabana::slice<Position>( particles );
    builder.build( positions, num_after_partition );

    // -----------------------------------------------------------------------
    // Call redistribute() with NO particle motion. Every particle's leaf
    // key already maps to this rank, so nothing should be sent.
    // -----------------------------------------------------------------------
    auto result =
        partitioner.redistribute( builder, particles, num_after_partition );

    int new_local = partitioner.num_local_particles();

    // Check 1: total count conserved.
    int total_after = 0;
    MPI_Allreduce( &new_local, &total_after, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );
    EXPECT_EQ( total_after, num_particles_per_rank * nprocs )
        << "Total particle count changed during redistribute";

    // Check 2: no particles sent or received globally.
    int global_sent = 0, global_recv = 0;
    MPI_Allreduce( &result.particles_sent, &global_sent, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );
    MPI_Allreduce( &result.particles_received, &global_recv, 1, MPI_INT,
                   MPI_SUM, MPI_COMM_WORLD );
    EXPECT_EQ( global_sent, 0 ) << "redistribute() sent " << global_sent
                                << " particles despite no motion";
    EXPECT_EQ( global_recv, 0 ) << "redistribute() received " << global_recv
                                << " particles despite no motion";

    // Check 3: result fields consistent.
    EXPECT_EQ( result.num_local_after, new_local );
    EXPECT_EQ( result.num_local_after, num_after_partition )
        << "Local count changed despite no-op redistribute";

    // Check 4: every particle in a leaf owned by this rank (or shared).
    int bad_count = count_particles_in_nonowned_leaves(
        builder, partitioner, particles, new_local, rank );
    EXPECT_EQ( bad_count, 0 )
        << "Rank " << rank << ": " << bad_count
        << " particles reside in a non-owned leaf after redistribute";
}

//---------------------------------------------------------------------------//
/**
 * Test TreePartitioner::redistribute() with actual particle motion. Crafts a
 * scenario where some particles deliberately end up on the wrong rank by
 * swapping particles between ranks AFTER the initial partition, then calls
 * redistribute() to fix the distribution.
 *
 * Setup per rank:
 *   - Build tree and partition — each rank now holds only particles in its
 *     owned leaves.
 *   - Rebuild tree on the local particle set so particle_keys matches.
 *   - Use MPI to swap a block of particles between rank r and rank (r+1)%n.
 *     The swapped particles keep their positions (so their keys still map
 *     to the ORIGINAL owner), but they now reside on the "wrong" rank.
 *   - redistribute() should detect this and migrate them back.
 *
 * Verifies:
 *   1. Before redistribute(), the misplaced particles' keys do NOT match
 *      this rank (otherwise the test setup is broken).
 *   2. Total particle count is conserved.
 *   3. After redistribute(), all particles are back in locally-owned leaves.
 *   4. The RedistributeResult is globally balanced (sent == received).
 */
void testRedistributeWithMotion( int num_particles_per_rank, int ncrit,
                                 int max_depth, double tolerance,
                                 int replication_depth, int num_to_swap )
{
    using namespace TreePartitionerTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    // This test requires at least 2 ranks to perform swaps.
    if ( nprocs < 2 )
        return;

    AoSoA_t particles( "particles", num_particles_per_rank );
    generate_test_particles( particles, num_particles_per_rank, rank );

    auto positions = Cabana::slice<Position>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );
    builder.build( positions, num_particles_per_rank );

    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    partitioner.partition( builder, particles, num_particles_per_rank );

    int num_after_partition = partitioner.num_local_particles();

    positions = Cabana::slice<Position>( particles );
    builder.build( positions, num_after_partition );

    // -----------------------------------------------------------------------
    // Swap num_to_swap particles between rank r and rank (r+1) % nprocs.
    // After the swap, each rank holds some particles whose keys belong to
    // its neighbor's owned leaves — exactly the scenario redistribute()
    // should fix.
    // -----------------------------------------------------------------------
    int send_partner = ( rank + 1 ) % nprocs;
    int recv_partner = ( rank - 1 + nprocs ) % nprocs;

    int swap_n = std::min( num_to_swap, num_after_partition );

    // Copy current particles to host for manipulation.
    AoSoA_ht particles_h( "particles_h", num_after_partition );
    Cabana::deep_copy( particles_h, particles );
    auto h_positions = Cabana::slice<Position>( particles_h );

    // Pack the tail positions to send.
    std::vector<double> send_buf( swap_n * 3 );
    for ( int i = 0; i < swap_n; ++i )
    {
        int src_idx = num_after_partition - swap_n + i;
        send_buf[3 * i + 0] = h_positions( src_idx, 0 );
        send_buf[3 * i + 1] = h_positions( src_idx, 1 );
        send_buf[3 * i + 2] = h_positions( src_idx, 2 );
    }

    std::vector<double> recv_buf( swap_n * 3 );
    MPI_Sendrecv( send_buf.data(), swap_n * 3, MPI_DOUBLE, send_partner, 77,
                  recv_buf.data(), swap_n * 3, MPI_DOUBLE, recv_partner, 77,
                  MPI_COMM_WORLD, MPI_STATUS_IGNORE );

    // Overwrite the tail of our local particles with those received from
    // recv_partner. These positions still belong to recv_partner's owned
    // leaves, so they are "misplaced" on this rank.
    for ( int i = 0; i < swap_n; ++i )
    {
        int dst_idx = num_after_partition - swap_n + i;
        h_positions( dst_idx, 0 ) = recv_buf[3 * i + 0];
        h_positions( dst_idx, 1 ) = recv_buf[3 * i + 1];
        h_positions( dst_idx, 2 ) = recv_buf[3 * i + 2];
    }
    Cabana::deep_copy( particles, particles_h );

    // Rebuild the tree on the post-swap particle set. The global particle
    // set is unchanged (swaps conserve it), so the tree topology is the
    // same, and the existing ownership map is valid.
    std::size_t cells_size_before_swap = builder.cells().size();
    positions = Cabana::slice<Position>( particles );
    builder.build( positions, num_after_partition );
    ASSERT_EQ( builder.cells().size(), cells_size_before_swap )
        << "Tree topology changed after particle swap — test precondition "
           "broken";

    // Sanity: before redistribute, some particles should be in non-owned
    // leaves (the swapped ones).
    int bad_before = count_particles_in_nonowned_leaves(
        builder, partitioner, particles, num_after_partition, rank );
    int global_bad_before = 0;
    MPI_Allreduce( &bad_before, &global_bad_before, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );
    EXPECT_GT( global_bad_before, 0 )
        << "Swap did not produce misplaced particles — test setup broken";

    // Rebuild the tree again because count_particles_in_nonowned_leaves
    // called build() internally; nothing changed but be explicit.
    positions = Cabana::slice<Position>( particles );
    builder.build( positions, num_after_partition );

    // -----------------------------------------------------------------------
    // Run redistribute.
    // -----------------------------------------------------------------------
    auto result =
        partitioner.redistribute( builder, particles, num_after_partition );

    int new_local = partitioner.num_local_particles();

    // Check 1: total count conserved.
    int total_after = 0;
    MPI_Allreduce( &new_local, &total_after, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );
    EXPECT_EQ( total_after, num_particles_per_rank * nprocs )
        << "Total particle count changed during redistribute";

    // Check 2: result fields self-consistent.
    EXPECT_EQ( result.num_local_after, new_local );
    EXPECT_EQ( num_after_partition - result.particles_sent +
                   result.particles_received,
               new_local )
        << "RedistributeResult counts are inconsistent";

    // Check 3: global sent == global received.
    int global_sent = 0, global_recv = 0;
    MPI_Allreduce( &result.particles_sent, &global_sent, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );
    MPI_Allreduce( &result.particles_received, &global_recv, 1, MPI_INT,
                   MPI_SUM, MPI_COMM_WORLD );
    EXPECT_EQ( global_sent, global_recv );
    EXPECT_GT( global_sent, 0 )
        << "redistribute() reported no sends despite misplaced particles";

    // Check 4: every local particle now lives in a locally-owned leaf.
    int bad_after = count_particles_in_nonowned_leaves(
        builder, partitioner, particles, new_local, rank );
    EXPECT_EQ( bad_after, 0 )
        << "Rank " << rank << ": " << bad_after
        << " particles reside in a non-owned leaf after redistribute";
}

//---------------------------------------------------------------------------//
/**
 * Test TreePartitioner::repartition() — full re-partitioning after tree
 * topology changes. Builds an initial tree and partition, then regenerates
 * particles with a completely different spatial distribution and rebuilds the
 * tree from scratch so the new topology differs from the original. The old
 * ownership map is now stale, and repartition() must re-partition from the
 * stale owners and rebuild ownership for the new tree.
 *
 * Using build() (instead of update()) for the topology change keeps the tree
 * globally consistent across ranks, avoiding spurious divergence.
 *
 * Verifies:
 *   1. The new tree differs from the original (otherwise this test isn't
 *      exercising the repartition path).
 *   2. Ownership vector is sized to the new cell count with valid
 *      assignments.
 *   3. Total particle count is conserved.
 *   4. Every local particle resides in a leaf owned by this rank (or
 *      shared).
 */
void testRepartition( int num_particles_per_rank, int ncrit, int max_depth,
                      double tolerance, int replication_depth )
{
    using namespace TreePartitionerTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    AoSoA_t particles( "particles", num_particles_per_rank );
    generate_test_particles( particles, num_particles_per_rank, rank );

    auto positions = Cabana::slice<Position>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );
    builder.build( positions, num_particles_per_rank );

    // Initial partition.
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    partitioner.partition( builder, particles, num_particles_per_rank );

    int num_after_partition = partitioner.num_local_particles();

    // Record the initial cell set so we can verify the new tree differs.
    std::set<MortonKey> initial_keys;
    for ( const auto& c : builder.cells() )
        initial_keys.insert( c.key );

    // -----------------------------------------------------------------------
    // Replace local particles with a completely new distribution shifted to
    // a different cluster location. Each rank gets num_particles_per_rank
    // particles (so the global total stays the same).
    // -----------------------------------------------------------------------
    regenerate_shifted_particles( particles, num_particles_per_rank,
                                  500 + rank );

    // Build a fresh tree globally from the new particles.
    positions = Cabana::slice<Position>( particles );
    builder.build( positions, num_particles_per_rank );

    // Sanity: tree should differ from the initial one (either different
    // cells or different counts). Check the key set.
    std::set<MortonKey> new_keys;
    for ( const auto& c : builder.cells() )
        new_keys.insert( c.key );
    int keys_differ = ( new_keys != initial_keys ) ? 1 : 0;
    int any_differ = 0;
    MPI_Allreduce( &keys_differ, &any_differ, 1, MPI_INT, MPI_MAX,
                   MPI_COMM_WORLD );
    ASSERT_GE( any_differ, 1 )
        << "New particle distribution produced the same tree as the "
           "original — test parameters need tuning to exercise repartition()";

    std::size_t cells_size_after = builder.cells().size();

    // -----------------------------------------------------------------------
    // Repartition on the new tree.
    // -----------------------------------------------------------------------
    partitioner.repartition( builder, particles, num_particles_per_rank );

    // Check 1: ownership vector sized to the (new) cell count.
    EXPECT_EQ( partitioner.ownership().size(), cells_size_after )
        << "Ownership vector size does not match new cells vector";

    // Check 2: every cell has a valid ownership assignment for the NEW tree.
    for ( std::size_t i = 0; i < cells_size_after; ++i )
    {
        const auto& c = builder.cells()[i];
        int owner = partitioner.ownership()[i].owner_rank;

        if ( owner == OWNER_SHARED )
        {
            EXPECT_LE( c.depth, replication_depth )
                << "Cell at depth " << c.depth
                << " is OWNER_SHARED but exceeds replication_depth "
                << replication_depth;
        }
        else
        {
            EXPECT_GE( owner, 0 ) << "Cell " << i << " has negative owner rank";
            EXPECT_LT( owner, nprocs ) << "Cell " << i << " owner " << owner
                                       << " >= nprocs " << nprocs;
        }
    }

    // Check 3: total particle count conserved.
    int new_local = partitioner.num_local_particles();
    int total_after = 0;
    MPI_Allreduce( &new_local, &total_after, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );
    EXPECT_EQ( total_after, num_particles_per_rank * nprocs )
        << "Total particle count changed during repartition";

    // Check 4: every local particle resides in a leaf owned by this rank.
    int bad_count = count_particles_in_nonowned_leaves(
        builder, partitioner, particles, new_local, rank );
    EXPECT_EQ( bad_count, 0 )
        << "Rank " << rank << ": " << bad_count
        << " particles reside in a non-owned leaf after repartition";

    (void)num_after_partition;
}


//---------------------------------------------------------------------------//
/**
 * H2 (a): cell_owner_map() is identical on every rank, after partition() and
 * after repartition() of a changed tree.
 */
void testOwnerMapAgreement( int num_particles_per_rank, int ncrit,
                            int max_depth, double tolerance,
                            int replication_depth )
{
    using namespace TreePartitionerTest;

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    AoSoA_t particles( "particles", num_particles_per_rank );
    generate_test_particles( particles, num_particles_per_rank, rank );
    auto positions = Cabana::slice<Position>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth,
        std::array<double, 6>{ tolerance, tolerance, tolerance, tolerance,
                               tolerance, tolerance },
        tolerance );
    builder.build( positions, num_particles_per_rank );

    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    partitioner.partition( builder, particles, num_particles_per_rank );
    EXPECT_TRUE( owner_map_agrees_across_ranks( partitioner.cell_owner_map() ) )
        << "cell_owner_map() differs across ranks after partition()";

    regenerate_shifted_particles( particles, num_particles_per_rank,
                                  500 + rank );
    positions = Cabana::slice<Position>( particles );
    builder.build( positions, num_particles_per_rank );
    partitioner.repartition( builder, particles, num_particles_per_rank );
    EXPECT_TRUE( owner_map_agrees_across_ranks( partitioner.cell_owner_map() ) )
        << "cell_owner_map() differs across ranks after repartition()";
}

//---------------------------------------------------------------------------//
/**
 * H2 (b): on a fixture of 24 000 uniform global particles with the tree
 * capped at max_depth so every depth band holds many cells per rank (R7: a
 * band with fewer cells than ranks cannot balance), every balance
 * constraint's max/mean over ranks is <= 1 + imbalance_tolerance: constraint
 * 0 the leaf particle count, constraint 1 + b the number of cells in band b.
 */
void testPartitionBalance( int num_global_particles, int ncrit, int max_depth,
                           double tolerance, int replication_depth,
                           double imbalance_tolerance )
{
    using namespace TreePartitionerTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
    ASSERT_EQ( num_global_particles % nprocs, 0 );
    const int num_local = num_global_particles / nprocs;

    AoSoA_t particles( "particles", num_local );
    generate_uniform_particles( particles, num_local, rank );
    auto positions = Cabana::slice<Position>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth,
        std::array<double, 6>{ tolerance, tolerance, tolerance, tolerance,
                               tolerance, tolerance },
        tolerance );
    builder.build( positions, num_local );

    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth, imbalance_tolerance );
    partitioner.partition( builder, particles, num_local );

    const auto& cells = builder.cells();
    const auto& owners = partitioner.cell_owner_map();
    const auto& bands = partitioner.bands();
    const int ncon = 1 + static_cast<int>( bands.size() );
    ASSERT_EQ( bands.size(), 3u ) << "fixture must span three bands";

    std::vector<double> load( static_cast<size_t>( ncon ) * nprocs, 0.0 );
    std::vector<int> band_cells( bands.size(), 0 );
    for ( const auto& c : cells )
    {
        const int r = owners.at( c.key );
        if ( r == OWNER_SHARED )
            continue;
        if ( c.is_leaf )
            load[r] += c.global_count;
        for ( std::size_t b = 0; b < bands.size(); ++b )
            if ( c.depth >= bands[b].depth_lo && c.depth <= bands[b].depth_hi )
            {
                load[( 1 + b ) * nprocs + r] += 1.0;
                band_cells[b]++;
            }
    }
    for ( std::size_t b = 0; b < bands.size(); ++b )
        ASSERT_GE( band_cells[b], 50 * nprocs )
            << "fixture band " << b << " holds too few cells per rank (R7)";

    for ( int k = 0; k < ncon; ++k )
    {
        double mx = 0.0, sum = 0.0;
        for ( int r = 0; r < nprocs; ++r )
        {
            mx = std::max( mx, load[k * nprocs + r] );
            sum += load[k * nprocs + r];
        }
        const double imbalance = mx * nprocs / sum;
        if ( rank == 0 )
            std::printf( "[tree-partitioner] np %d constraint %d max/mean %.4f\n",
                         nprocs, k, imbalance );
        EXPECT_LE( imbalance, 1.0 + imbalance_tolerance )
            << "constraint " << k << " is imbalanced";
    }
}

//---------------------------------------------------------------------------//
/**
 * H2 (c): the fraction of non-shared parent-child pairs owned by different
 * ranks is no higher than under the vote rule applied to the same leaf
 * assignment. Runs on the clustered fixture, where the cut is several
 * percent; on (b)'s depth-capped uniform fixture both cuts are under 1% and
 * the vote rule breaks band balance at np 5-6
 * (fix-hang-rebalance-progress-log.md, H2 partitioner arm).
 */
void testParentChildCut( int num_particles_per_rank, int ncrit, int max_depth,
                         double tolerance, int replication_depth )
{
    using namespace TreePartitionerTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    AoSoA_t particles( "particles", num_particles_per_rank );
    generate_test_particles( particles, num_particles_per_rank, rank );
    auto positions = Cabana::slice<Position>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth,
        std::array<double, 6>{ tolerance, tolerance, tolerance, tolerance,
                               tolerance, tolerance },
        tolerance );
    builder.build( positions, num_particles_per_rank );

    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    partitioner.partition( builder, particles, num_particles_per_rank );

    const auto& cells = builder.cells();
    const auto& owners = partitioner.cell_owner_map();
    const auto vote_internal = partitioner.vote_internal_owners( cells, owners );
    std::unordered_map<MortonKey, int> vote_owners;
    for ( const auto& c : cells )
    {
        if ( c.is_leaf )
            vote_owners[c.key] = owners.at( c.key );
        else if ( c.depth > replication_depth )
            vote_owners[c.key] = vote_internal.at( c.key );
    }
    const double cut =
        parent_child_cut_fraction( cells, owners, replication_depth );
    const double cut_vote =
        parent_child_cut_fraction( cells, vote_owners, replication_depth );
    if ( rank == 0 )
        std::printf( "[tree-partitioner] np %d parent-child cut %.4f vote "
                     "rule %.4f\n",
                     nprocs, cut, cut_vote );
    EXPECT_LE( cut, cut_vote )
        << "the partition cuts more parent-child pairs than the vote rule";
}

//---------------------------------------------------------------------------//
/**
 * H2 (d): after refresh_ownership_for_current_tree(), every internal cell of
 * the current tree that the partition assigned keeps its partitioned owner.
 * Run on the post-migration tree (the same topology) and on a tree built from
 * a subset of the particles (a coarser topology with new leaves). Prints the
 * fraction of non-shared cells that fell back to a vote.
 */
void testRefreshKeepsPartition( int num_particles_per_rank, int ncrit,
                                int max_depth, double tolerance,
                                int replication_depth )
{
    using namespace TreePartitionerTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    AoSoA_t particles( "particles", num_particles_per_rank );
    generate_test_particles( particles, num_particles_per_rank, rank );
    auto positions = Cabana::slice<Position>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth,
        std::array<double, 6>{ tolerance, tolerance, tolerance, tolerance,
                               tolerance, tolerance },
        tolerance );
    builder.build( positions, num_particles_per_rank );

    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );
    partitioner.partition( builder, particles, num_particles_per_rank );
    const auto partitioned = partitioner.cell_owner_map();
    const int num_local = partitioner.num_local_particles();

    for ( const char* phase : { "post-migration", "subset" } )
    {
        const int n = ( phase[0] == 'p' ) ? num_local : num_local * 3 / 5;
        positions = Cabana::slice<Position>( particles );
        builder.build( positions, n );
        partitioner.refresh_ownership_for_current_tree( builder, particles );

        const auto& owners = partitioner.cell_owner_map();
        int non_shared = 0, fallback = 0, changed = 0;
        for ( const auto& c : builder.cells() )
        {
            if ( !c.is_leaf && c.depth <= replication_depth )
                continue;
            non_shared++;
            auto it = partitioned.find( c.key );
            if ( it == partitioned.end() || it->second == OWNER_SHARED )
            {
                fallback++;
                continue;
            }
            if ( !c.is_leaf && owners.at( c.key ) != it->second )
                changed++;
        }
        EXPECT_EQ( changed, 0 )
            << phase << ": " << changed
            << " internal cells lost their partitioned owner";
        EXPECT_TRUE( owner_map_agrees_across_ranks( owners ) ) << phase;
        if ( rank == 0 )
            std::printf( "[tree-partitioner] np %d refresh %s fallback "
                         "%d/%d = %.4f\n",
                         nprocs, phase, fallback, non_shared,
                         non_shared ? double( fallback ) / non_shared : 0.0 );
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//

TEST( TreePartitioner, testBasicPartition )
{
    testPartitioner( 10000, 128, 15, 0.1, 3 );
}

TEST( TreePartitioner, testSmallTree )
{
    testPartitioner( 500, 32, 10, 0.1, 2 );
}

TEST( TreePartitioner, testRedistributeNoOpBasic )
{
    testRedistributeNoOp( 10000, 128, 15, 0.1, 3 );
}

TEST( TreePartitioner, testRedistributeNoOpSmall )
{
    testRedistributeNoOp( 500, 32, 10, 0.1, 2 );
}

TEST( TreePartitioner, testRedistributeWithMotionBasic )
{
    testRedistributeWithMotion( 10000, 128, 15, 0.1, 3, 50 );
}

TEST( TreePartitioner, testRedistributeWithMotionSmall )
{
    testRedistributeWithMotion( 500, 32, 10, 0.1, 2, 10 );
}

TEST( TreePartitioner, testRepartitionBasic )
{
    testRepartition( 10000, 128, 15, 0.1, 3 );
}

TEST( TreePartitioner, testRepartitionSmall )
{
    testRepartition( 500, 32, 10, 0.1, 2 );
}

TEST( TreePartitioner, testCoalescedMigrateIntegrityBasic )
{
    testCoalescedMigrateIntegrity( 10000, 128, 15, 0.1, 3 );
}

TEST( TreePartitioner, testCoalescedMigrateIntegritySmall )
{
    testCoalescedMigrateIntegrity( 500, 32, 10, 0.1, 2 );
}

TEST( TreePartitioner, testOwnerMapAgreement )
{
    testOwnerMapAgreement( 10000, 128, 15, 0.1, 3 );
}

TEST( TreePartitioner, testPartitionBalance )
{
    testPartitionBalance( 24000, 4, 5, 0.1, 2, 0.05 );
}

TEST( TreePartitioner, testParentChildCut )
{
    testParentChildCut( 10000, 128, 15, 0.1, 3 );
}

TEST( TreePartitioner, testRefreshKeepsPartition )
{
    testRefreshKeepsPartition( 10000, 128, 15, 0.1, 3 );
}

//---------------------------------------------------------------------------//

} // end namespace Test
