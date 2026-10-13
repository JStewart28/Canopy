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

#include <Canopy_CommunicationPlan.hpp>
#include <Canopy_TreeBuilder.hpp>
#include <Canopy_TreePartitioner.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <random>
#include <set>
#include <unordered_set>

namespace Test
{
//---------------------------------------------------------------------------//

using namespace Canopy;

namespace CommunicationPlanTest
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

// Build a tree, partition it, and return a ready-to-use CommunicationPlan.
// Caller receives ownership of the plan; builder and partitioner are
// also returned by output parameter so callers can inspect them.
template <class MemorySpace, class ExecutionSpace>
CommunicationPlan<MemorySpace, ExecutionSpace>
build_plan( TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE>& builder,
            TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE>& partitioner,
            AoSoA_t& particles, int num_particles_per_rank, int rank )
{
    generate_test_particles( particles, num_particles_per_rank, rank );

    auto positions = Cabana::slice<Position>( particles );
    builder.build( positions, num_particles_per_rank );

    partitioner.partition( builder, particles, num_particles_per_rank );

    CommunicationPlan<MemorySpace, ExecutionSpace> plan( MPI_COMM_WORLD );
    plan.build( builder.cells(), partitioner.ownership(),
                partitioner.cell_owner_map(), partitioner.replication_depth() );
    return plan;
}

} // namespace CommunicationPlanTest

//---------------------------------------------------------------------------//
/**
 * Verify that build() marks the plan valid and that invalidate() clears it.
 * Rebuilding after invalidate() must restore validity.
 *
 * Checks:
 *   1. valid() is false before build().
 *   2. valid() is true after build().
 *   3. invalidate() makes valid() false.
 *   4. A second build() makes valid() true again.
 */
void testBuildAndValid( int num_particles_per_rank, int ncrit, int max_depth,
                        double tolerance, int replication_depth )
{
    using namespace CommunicationPlanTest;

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    AoSoA_t particles( "particles", num_particles_per_rank );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );

    generate_test_particles( particles, num_particles_per_rank, rank );
    auto positions = Cabana::slice<Position>( particles );
    builder.build( positions, num_particles_per_rank );
    partitioner.partition( builder, particles, num_particles_per_rank );

    CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE> plan( MPI_COMM_WORLD );

    // Check 1: not yet valid
    EXPECT_FALSE( plan.valid() );

    // Check 2: valid after build
    plan.build( builder.cells(), partitioner.ownership(),
                partitioner.cell_owner_map(), partitioner.replication_depth() );
    EXPECT_TRUE( plan.valid() );

    // Check 3: invalidate clears it
    plan.invalidate();
    EXPECT_FALSE( plan.valid() );

    // Check 4: rebuild restores it
    plan.build( builder.cells(), partitioner.ownership(),
                partitioner.cell_owner_map(), partitioner.replication_depth() );
    EXPECT_TRUE( plan.valid() );
}

//---------------------------------------------------------------------------//
/**
 * Verify the structural invariants of the M2M and L2L vertical plans.
 *
 * Checks:
 *   1. max_depth matches the deepest cell in the tree.
 *   2. M2M and L2L share the same max_depth.
 *   3. Every cell in shared_cells has depth <= replication_depth.
 *   4. No duplicate keys appear in shared_cells.
 */
void testVerticalPlanStructure( int num_particles_per_rank, int ncrit,
                                int max_depth, double tolerance,
                                int replication_depth )
{
    using namespace CommunicationPlanTest;

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    AoSoA_t particles( "particles", num_particles_per_rank );
    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );

    auto plan = build_plan<TEST_MEMSPACE, TEST_EXECSPACE>(
        builder, partitioner, particles, num_particles_per_rank, rank );

    const auto& cells = builder.cells();

    // Check 1 & 2: max_depth agrees with tree
    int actual_max_depth = 0;
    for ( const auto& c : cells )
        actual_max_depth = std::max( actual_max_depth, c.depth );

    EXPECT_EQ( plan.m2m_plan().max_depth, actual_max_depth );
    EXPECT_EQ( plan.l2l_plan().max_depth, actual_max_depth );

    // Check 3 & 4: shared_cells depth constraint and no duplicates
    for ( const auto& vplan : { &plan.m2m_plan(), &plan.l2l_plan() } )
    {
        std::unordered_set<MortonKey> seen;
        for ( MortonKey sk : vplan->shared_cells )
        {
            auto it = std::find_if( cells.begin(), cells.end(),
                                    [sk]( const CellInfo& c )
                                    { return c.key == sk; } );
            ASSERT_NE( it, cells.end() )
                << "shared_cells contains key not in the tree: " << sk;

            EXPECT_LE( it->depth, replication_depth )
                << "Shared cell at depth " << it->depth
                << " exceeds replication_depth " << replication_depth;

            EXPECT_TRUE( seen.insert( sk ).second )
                << "Duplicate key in shared_cells: " << sk;
        }
    }
}

//---------------------------------------------------------------------------//
/**
 * Verify that M2M and L2L send/receive counts are globally balanced.
 *
 * Every send on one rank must correspond to a receive on another. So the
 * global sum of sends must equal the global sum of receives for both plans.
 *
 * Checks:
 *   1. Global M2M sends == global M2M receives.
 *   2. Global L2L sends == global L2L receives.
 *   3. CellTransfer::remote_rank is a valid rank in [0, nprocs).
 */
void testVerticalPlanBalance( int num_particles_per_rank, int ncrit,
                              int max_depth, double tolerance,
                              int replication_depth )
{
    using namespace CommunicationPlanTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    AoSoA_t particles( "particles", num_particles_per_rank );
    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );

    auto plan = build_plan<TEST_MEMSPACE, TEST_EXECSPACE>(
        builder, partitioner, particles, num_particles_per_rank, rank );

    for ( const auto& vplan : { &plan.m2m_plan(), &plan.l2l_plan() } )
    {
        int local_sends = static_cast<int>( vplan->sends.size() );
        int local_recvs = static_cast<int>( vplan->receives.size() );

        int global_sends = 0, global_recvs = 0;
        MPI_Allreduce( &local_sends, &global_sends, 1, MPI_INT, MPI_SUM,
                       MPI_COMM_WORLD );
        MPI_Allreduce( &local_recvs, &global_recvs, 1, MPI_INT, MPI_SUM,
                       MPI_COMM_WORLD );

        // Check 1 & 2
        EXPECT_EQ( global_sends, global_recvs )
            << "Vertical plan: global sends != global receives";

        // Check 3
        for ( const auto& ct : vplan->sends )
            EXPECT_LT( ct.remote_rank, nprocs );
        for ( const auto& ct : vplan->receives )
            EXPECT_LT( ct.remote_rank, nprocs );
    }
}

//---------------------------------------------------------------------------//
/**
 * Verify structural invariants of the M2L interaction lists.
 *
 * Checks:
 *   1. No cell's interaction list contains itself (no self-interaction).
 *   2. Every key in an interaction list exists in the tree.
 *
 * Note: both leaf and internal cells are valid M2L targets — leaves
 * accumulate local expansions from well-separated cells just as
 * internal cells do.
 */
void testM2LInteractionLists( int num_particles_per_rank, int ncrit,
                              int max_depth, double tolerance,
                              int replication_depth )
{
    using namespace CommunicationPlanTest;

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    AoSoA_t particles( "particles", num_particles_per_rank );
    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );

    auto plan = build_plan<TEST_MEMSPACE, TEST_EXECSPACE>(
        builder, partitioner, particles, num_particles_per_rank, rank );

    const auto& cells = builder.cells();

    // Build a lookup set of all cell keys in the tree
    std::unordered_set<MortonKey> cell_key_set;
    std::unordered_map<MortonKey, bool> is_leaf_map;
    for ( const auto& c : cells )
    {
        cell_key_set.insert( c.key );
        is_leaf_map[c.key] = c.is_leaf;
    }

    const auto& ilists = plan.m2l_plan().interaction_lists;

    for ( const auto& [target_key, sources] : ilists )
    {
        // Check 1: no self-interaction
        for ( const auto& [src, src_idx] : sources )
        {
            (void)src_idx;
            EXPECT_NE( src, target_key )
                << "Cell " << target_key << " has itself in interaction list";
        }

        // Check 2: every source key exists in the tree
        for ( const auto& [src, src_idx] : sources )
        {
            (void)src_idx;
            EXPECT_TRUE( cell_key_set.count( src ) > 0 )
                << "Interaction list source " << src << " not in tree";
        }
    }
}

//---------------------------------------------------------------------------//
/**
 * Verify that M2L send/receive counts are globally balanced.
 *
 * Checks:
 *   1. Global M2L sends == global M2L receives.
 *   2. All remote_rank values in sends and receives are valid ranks.
 */
void testM2LBalance( int num_particles_per_rank, int ncrit, int max_depth,
                     double tolerance, int replication_depth )
{
    using namespace CommunicationPlanTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    AoSoA_t particles( "particles", num_particles_per_rank );
    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );

    auto plan = build_plan<TEST_MEMSPACE, TEST_EXECSPACE>(
        builder, partitioner, particles, num_particles_per_rank, rank );

    const auto& m2l = plan.m2l_plan();

    int local_sends = static_cast<int>( m2l.sends.size() );
    int local_recvs = static_cast<int>( m2l.receives.size() );

    int global_sends = 0, global_recvs = 0;
    MPI_Allreduce( &local_sends, &global_sends, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );
    MPI_Allreduce( &local_recvs, &global_recvs, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );

    // Check 1
    EXPECT_EQ( global_sends, global_recvs )
        << "M2L plan: global sends != global receives";

    // Check 2
    for ( const auto& ct : m2l.sends )
    {
        EXPECT_GE( ct.remote_rank, 0 );
        EXPECT_LT( ct.remote_rank, nprocs );
    }
    for ( const auto& ct : m2l.receives )
    {
        EXPECT_GE( ct.remote_rank, 0 );
        EXPECT_LT( ct.remote_rank, nprocs );
    }
}

//---------------------------------------------------------------------------//
/**
 * Verify P2P neighbor list invariants.
 *
 * Checks:
 *   1. Every locally-owned leaf has a neighbor list entry.
 *   2. Each neighbor list contains the cell itself (self-interaction).
 *   3. Every key in a neighbor list exists in the tree.
 *   4. Every key in a neighbor list maps to a leaf cell.
 */
void testP2PNeighborLists( int num_particles_per_rank, int ncrit, int max_depth,
                           double tolerance, int replication_depth )
{
    using namespace CommunicationPlanTest;

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    AoSoA_t particles( "particles", num_particles_per_rank );
    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );

    auto plan = build_plan<TEST_MEMSPACE, TEST_EXECSPACE>(
        builder, partitioner, particles, num_particles_per_rank, rank );

    const auto& cells = builder.cells();

    std::unordered_set<MortonKey> cell_key_set;
    std::unordered_map<MortonKey, bool> is_leaf_map;
    for ( const auto& c : cells )
    {
        cell_key_set.insert( c.key );
        is_leaf_map[c.key] = c.is_leaf;
    }

    const auto& nlists = plan.p2p_plan().neighbor_lists;

    for ( const auto& c : cells )
    {
        if ( !c.is_leaf )
            continue;
        int owner = partitioner.cell_owner( c.key );
        if ( owner != rank )
            continue;

        // Check 1: locally-owned leaf must have an entry
        auto it = nlists.find( c.key );
        ASSERT_NE( it, nlists.end() ) << "Locally-owned leaf " << c.key
                                      << " missing from P2P neighbor lists";

        const auto& nbrs = it->second;

        // Check 2: self-interaction present
        bool has_self =
            std::find( nbrs.begin(), nbrs.end(), c.key ) != nbrs.end();
        EXPECT_TRUE( has_self )
            << "Leaf " << c.key << " missing self-interaction in P2P list";

        // Check 3 & 4: every neighbor is a known leaf
        for ( MortonKey nk : nbrs )
        {
            EXPECT_TRUE( cell_key_set.count( nk ) > 0 )
                << "P2P neighbor " << nk << " not in tree";

            auto li = is_leaf_map.find( nk );
            if ( li != is_leaf_map.end() )
            {
                EXPECT_TRUE( li->second )
                    << "P2P neighbor " << nk << " is not a leaf";
            }
        }
    }
}

//---------------------------------------------------------------------------//
/**
 * Verify P2P ghost leaf metadata is consistent.
 *
 * Checks:
 *   1. ghost_leaf_keys and ghost_leaf_owners have the same size.
 *   2. Every ghost owner is a valid rank in [0, nprocs) (not OWNER_SHARED).
 *   3. No ghost key equals a locally-owned leaf key (ghosts are remote).
 *   4. Every ghost key exists in the tree and is a leaf.
 */
void testP2PGhostConsistency( int num_particles_per_rank, int ncrit,
                              int max_depth, double tolerance,
                              int replication_depth )
{
    using namespace CommunicationPlanTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    AoSoA_t particles( "particles", num_particles_per_rank );
    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );

    auto plan = build_plan<TEST_MEMSPACE, TEST_EXECSPACE>(
        builder, partitioner, particles, num_particles_per_rank, rank );

    const auto& cells = builder.cells();
    const auto& p2p = plan.p2p_plan();

    // Check 1
    ASSERT_EQ( p2p.ghost_leaf_keys.size(), p2p.ghost_leaf_owners.size() )
        << "ghost_leaf_keys and ghost_leaf_owners size mismatch";

    // Build sets for fast lookup
    std::unordered_map<MortonKey, bool> is_leaf_map;
    for ( const auto& c : cells )
        is_leaf_map[c.key] = c.is_leaf;

    std::unordered_set<MortonKey> local_owned_leaves;
    for ( const auto& c : cells )
    {
        if ( c.is_leaf && partitioner.cell_owner( c.key ) == rank )
            local_owned_leaves.insert( c.key );
    }

    for ( std::size_t i = 0; i < p2p.ghost_leaf_keys.size(); ++i )
    {
        MortonKey gk = p2p.ghost_leaf_keys[i];
        int gowner = p2p.ghost_leaf_owners[i];

        // Check 2
        EXPECT_NE( gowner, OWNER_SHARED )
            << "Ghost leaf " << gk
            << " has OWNER_SHARED — ghosts must have a "
               "unique owner";
        EXPECT_GE( gowner, 0 );
        EXPECT_LT( gowner, nprocs );

        // Check 3
        EXPECT_FALSE( local_owned_leaves.count( gk ) > 0 )
            << "Ghost key " << gk << " is also a locally-owned leaf";

        // Check 4
        auto li = is_leaf_map.find( gk );
        ASSERT_NE( li, is_leaf_map.end() )
            << "Ghost key " << gk << " not found in tree";
        EXPECT_TRUE( li->second ) << "Ghost key " << gk << " is not a leaf";
    }
}

//---------------------------------------------------------------------------//
/**
 * On a single MPI rank every cell is owned by rank 0 (or OWNER_SHARED for
 * shallow cells). No point-to-point communication should be planned.
 *
 * Checks:
 *   1. M2M sends and receives are empty.
 *   2. L2L sends and receives are empty.
 *   3. M2L sends and receives are empty.
 *   4. P2P ghost list is empty.
 */
void testSingleRankNoTransfers( int num_particles_per_rank, int ncrit,
                                int max_depth, double tolerance,
                                int replication_depth )
{
    using namespace CommunicationPlanTest;

    int nprocs;
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
    if ( nprocs != 1 )
        return; // only meaningful on one rank

    int rank = 0;

    AoSoA_t particles( "particles", num_particles_per_rank );
    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );

    auto plan = build_plan<TEST_MEMSPACE, TEST_EXECSPACE>(
        builder, partitioner, particles, num_particles_per_rank, rank );

    // Check 1
    EXPECT_TRUE( plan.m2m_plan().sends.empty() )
        << "M2M sends non-empty on single rank";
    EXPECT_TRUE( plan.m2m_plan().receives.empty() )
        << "M2M receives non-empty on single rank";

    // Check 2
    EXPECT_TRUE( plan.l2l_plan().sends.empty() )
        << "L2L sends non-empty on single rank";
    EXPECT_TRUE( plan.l2l_plan().receives.empty() )
        << "L2L receives non-empty on single rank";

    // Check 3
    EXPECT_TRUE( plan.m2l_plan().sends.empty() )
        << "M2L sends non-empty on single rank";
    EXPECT_TRUE( plan.m2l_plan().receives.empty() )
        << "M2L receives non-empty on single rank";

    // Check 4
    EXPECT_TRUE( plan.p2p_plan().ghost_leaf_keys.empty() )
        << "P2P ghost list non-empty on single rank";
}

//---------------------------------------------------------------------------//
/**
 * Verify that the indexed build (set_indexed_build( true )) produces exactly
 * the plan the hash-driven build produces, on one tree and one partition.
 *
 * Every public field is compared, in order where it is a vector: the M2M and
 * L2L sends, receives, shared cells and max depth; every M2L interaction
 * list (source keys and cell indices) and the M2L sends and receives; every
 * P2P neighbor list, the ghost keys and owners, and the send leaves. The
 * maps are compared by key. A second indexed build on the same plan object
 * must match too, so emission buffers kept across builds leave no residue.
 */
void testIndexedMatchesLegacy( int num_particles_per_rank, int ncrit,
                               int max_depth, double tolerance,
                               int replication_depth, double mac_theta,
                               double softening )
{
    using namespace CommunicationPlanTest;

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    AoSoA_t particles( "particles", num_particles_per_rank );
    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, std::array<double, 6>{tolerance, tolerance, tolerance, tolerance, tolerance, tolerance}, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );

    generate_test_particles( particles, num_particles_per_rank, rank );
    auto positions = Cabana::slice<Position>( particles );
    builder.build( positions, num_particles_per_rank );
    partitioner.partition( builder, particles, num_particles_per_rank );

    using Plan = CommunicationPlan<TEST_MEMSPACE, TEST_EXECSPACE>;
    auto make = [&]( bool indexed )
    {
        Plan p( MPI_COMM_WORLD, mac_theta );
        p.set_near_softening( softening, 4.0 );
        p.set_indexed_build( indexed );
        p.build( builder.cells(), partitioner.ownership(),
                 partitioner.cell_owner_map(),
                 partitioner.replication_depth() );
        return p;
    };
    Plan legacy = make( false );
    Plan indexed = make( true );
    ASSERT_TRUE( indexed.indexed_build() );
    ASSERT_FALSE( legacy.indexed_build() );

    auto same_transfers = []( const std::vector<CellTransfer>& a,
                              const std::vector<CellTransfer>& b,
                              const char* what )
    {
        ASSERT_EQ( a.size(), b.size() ) << what;
        for ( std::size_t i = 0; i < a.size(); i++ )
        {
            EXPECT_EQ( a[i].cell_key, b[i].cell_key ) << what << " [" << i << "]";
            EXPECT_EQ( a[i].remote_rank, b[i].remote_rank )
                << what << " [" << i << "]";
        }
    };

    auto compare = [&]( const Plan& x, const Plan& y )
    {
        same_transfers( x.m2m_plan().sends, y.m2m_plan().sends, "m2m sends" );
        same_transfers( x.m2m_plan().receives, y.m2m_plan().receives,
                        "m2m receives" );
        EXPECT_EQ( x.m2m_plan().shared_cells, y.m2m_plan().shared_cells );
        EXPECT_EQ( x.m2m_plan().max_depth, y.m2m_plan().max_depth );
        same_transfers( x.l2l_plan().sends, y.l2l_plan().sends, "l2l sends" );
        same_transfers( x.l2l_plan().receives, y.l2l_plan().receives,
                        "l2l receives" );
        EXPECT_EQ( x.l2l_plan().shared_cells, y.l2l_plan().shared_cells );
        EXPECT_EQ( x.l2l_plan().max_depth, y.l2l_plan().max_depth );

        const auto& xl = x.m2l_plan().interaction_lists;
        const auto& yl = y.m2l_plan().interaction_lists;
        EXPECT_EQ( xl.size(), yl.size() ) << "m2l target count";
        for ( const auto& [t, sources] : xl )
        {
            auto it = yl.find( t );
            ASSERT_TRUE( it != yl.end() ) << "m2l target " << t << " missing";
            EXPECT_EQ( sources, it->second ) << "m2l list of " << t;
        }
        same_transfers( x.m2l_plan().sends, y.m2l_plan().sends, "m2l sends" );
        same_transfers( x.m2l_plan().receives, y.m2l_plan().receives,
                        "m2l receives" );

        const auto& xn = x.p2p_plan().neighbor_lists;
        const auto& yn = y.p2p_plan().neighbor_lists;
        EXPECT_EQ( xn.size(), yn.size() ) << "p2p target count";
        for ( const auto& [t, nbrs] : xn )
        {
            auto it = yn.find( t );
            ASSERT_TRUE( it != yn.end() ) << "p2p target " << t << " missing";
            EXPECT_EQ( nbrs, it->second ) << "p2p list of " << t;
        }
        EXPECT_EQ( x.p2p_plan().ghost_leaf_keys, y.p2p_plan().ghost_leaf_keys );
        EXPECT_EQ( x.p2p_plan().ghost_leaf_owners,
                   y.p2p_plan().ghost_leaf_owners );
        same_transfers( x.p2p_plan().send_leaves, y.p2p_plan().send_leaves,
                        "p2p send leaves" );
    };

    // The fixture must exercise every part of the plan being compared.
    EXPECT_GT( legacy.m2l_plan().interaction_lists.size(), 0u );
    EXPECT_GT( legacy.p2p_plan().neighbor_lists.size(), 0u );
    int nprocs;
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
    if ( nprocs > 1 )
    {
        long long local = static_cast<long long>(
            legacy.m2l_plan().receives.size() +
            legacy.p2p_plan().ghost_leaf_keys.size() );
        long long global = 0;
        MPI_Allreduce( &local, &global, 1, MPI_LONG_LONG, MPI_SUM,
                       MPI_COMM_WORLD );
        EXPECT_GT( global, 0 ) << "no cross-rank traffic to compare";
    }

    compare( legacy, indexed );

    indexed.build( builder.cells(), partitioner.ownership(),
                   partitioner.cell_owner_map(),
                   partitioner.replication_depth() );
    compare( legacy, indexed );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//

TEST( CommunicationPlan, testBuildAndValidBasic )
{
    testBuildAndValid( 10000, 128, 15, 0.1, 3 );
}

TEST( CommunicationPlan, testBuildAndValidSmall )
{
    testBuildAndValid( 500, 32, 10, 0.1, 2 );
}

TEST( CommunicationPlan, testVerticalPlanStructureBasic )
{
    testVerticalPlanStructure( 10000, 128, 15, 0.1, 3 );
}

TEST( CommunicationPlan, testVerticalPlanStructureSmall )
{
    testVerticalPlanStructure( 500, 32, 10, 0.1, 2 );
}

TEST( CommunicationPlan, testVerticalPlanBalanceBasic )
{
    testVerticalPlanBalance( 10000, 128, 15, 0.1, 3 );
}

TEST( CommunicationPlan, testVerticalPlanBalanceSmall )
{
    testVerticalPlanBalance( 500, 32, 10, 0.1, 2 );
}

TEST( CommunicationPlan, testM2LInteractionListsBasic )
{
    testM2LInteractionLists( 10000, 128, 15, 0.1, 3 );
}

TEST( CommunicationPlan, testM2LInteractionListsSmall )
{
    testM2LInteractionLists( 500, 32, 10, 0.1, 2 );
}

TEST( CommunicationPlan, testM2LBalanceBasic )
{
    testM2LBalance( 10000, 128, 15, 0.1, 3 );
}

TEST( CommunicationPlan, testM2LBalanceSmall )
{
    testM2LBalance( 500, 32, 10, 0.1, 2 );
}

TEST( CommunicationPlan, testP2PNeighborListsBasic )
{
    testP2PNeighborLists( 10000, 128, 15, 0.1, 3 );
}

TEST( CommunicationPlan, testP2PNeighborListsSmall )
{
    testP2PNeighborLists( 500, 32, 10, 0.1, 2 );
}

TEST( CommunicationPlan, testP2PGhostConsistencyBasic )
{
    testP2PGhostConsistency( 10000, 128, 15, 0.1, 3 );
}

TEST( CommunicationPlan, testP2PGhostConsistencySmall )
{
    testP2PGhostConsistency( 500, 32, 10, 0.1, 2 );
}

TEST( CommunicationPlan, testSingleRankNoTransfers )
{
    testSingleRankNoTransfers( 10000, 128, 15, 0.1, 3 );
}

// A deep tree at Beatnik's measurement leaf size and MAC, with the
// near-softening floor active, and the default configuration.
TEST( CommunicationPlan, testIndexedMatchesLegacyDeep )
{
    testIndexedMatchesLegacy( 4000, 8, 10, 0.1, 1, 0.3, 1.0e-3 );
}

TEST( CommunicationPlan, testIndexedMatchesLegacyBasic )
{
    testIndexedMatchesLegacy( 10000, 128, 15, 0.1, 3, 0.5, 0.0 );
}

//---------------------------------------------------------------------------//

} // end namespace Test
