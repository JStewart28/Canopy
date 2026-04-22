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

#include <Canopy_Experimental_CommunicationPlan.hpp>
#include <Canopy_Experimental_TreeBuilder.hpp>
#include <Canopy_Experimental_TreePartitioner.hpp>

#include <test_helpers.hpp>

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

using namespace Canopy::Experimental;

namespace CommunicationPlanTest
{

enum FieldIdx
{
    Position = 0
};

using DataTypes = Cabana::MemberTypes<double[3]>;
using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
using AoSoA_ht = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
using DeviceType = Kokkos::Device<TEST_EXECSPACE, TEST_MEMSPACE>;

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
CommunicationPlan<DeviceType> build_plan(
    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE>& builder,
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE>& partitioner,
    AoSoA_t& particles, int num_particles_per_rank, int rank )
{
    generate_test_particles( particles, num_particles_per_rank, rank );

    auto positions = Cabana::slice<Position>( particles );
    builder.build( positions, num_particles_per_rank );

    partitioner.partition( builder, particles, num_particles_per_rank );

    CommunicationPlan<DeviceType> plan( MPI_COMM_WORLD );
    plan.build( builder.cells(), partitioner.ownership(),
                partitioner.cell_owner_map(),
                partitioner.replication_depth() );
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
        ncrit, max_depth, MPI_COMM_WORLD, tolerance, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );

    generate_test_particles( particles, num_particles_per_rank, rank );
    auto positions = Cabana::slice<Position>( particles );
    builder.build( positions, num_particles_per_rank );
    partitioner.partition( builder, particles, num_particles_per_rank );

    CommunicationPlan<DeviceType> plan( MPI_COMM_WORLD );

    // Check 1: not yet valid
    EXPECT_FALSE( plan.valid() );

    // Check 2: valid after build
    plan.build( builder.cells(), partitioner.ownership(),
                partitioner.cell_owner_map(),
                partitioner.replication_depth() );
    EXPECT_TRUE( plan.valid() );

    // Check 3: invalidate clears it
    plan.invalidate();
    EXPECT_FALSE( plan.valid() );

    // Check 4: rebuild restores it
    plan.build( builder.cells(), partitioner.ownership(),
                partitioner.cell_owner_map(),
                partitioner.replication_depth() );
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
        ncrit, max_depth, MPI_COMM_WORLD, tolerance, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );

    auto plan = build_plan( builder, partitioner, particles,
                            num_particles_per_rank, rank );

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
        ncrit, max_depth, MPI_COMM_WORLD, tolerance, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );

    auto plan = build_plan( builder, partitioner, particles,
                            num_particles_per_rank, rank );

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
 *   3. Internal (non-leaf) cells at depth > 0 that this rank processes
 *      have a non-empty interaction list (they always have at least
 *      some well-separated cells in 3-D, unless the tree is trivially small).
 *   4. Leaf cells do not appear as targets in the interaction list map
 *      (M2L is applied at internal cells).
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
        ncrit, max_depth, MPI_COMM_WORLD, tolerance, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );

    auto plan = build_plan( builder, partitioner, particles,
                            num_particles_per_rank, rank );

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
        for ( MortonKey src : sources )
        {
            EXPECT_NE( src, target_key )
                << "Cell " << target_key << " has itself in interaction list";
        }

        // Check 2: every source key exists in the tree
        for ( MortonKey src : sources )
        {
            EXPECT_TRUE( cell_key_set.count( src ) > 0 )
                << "Interaction list source " << src << " not in tree";
        }

        // Check 4: target must not be a leaf
        auto leaf_it = is_leaf_map.find( target_key );
        if ( leaf_it != is_leaf_map.end() )
        {
            EXPECT_FALSE( leaf_it->second )
                << "Leaf cell " << target_key
                << " appears as M2L target — M2L should target internal cells";
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
        ncrit, max_depth, MPI_COMM_WORLD, tolerance, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );

    auto plan = build_plan( builder, partitioner, particles,
                            num_particles_per_rank, rank );

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
        ncrit, max_depth, MPI_COMM_WORLD, tolerance, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );

    auto plan = build_plan( builder, partitioner, particles,
                            num_particles_per_rank, rank );

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
        ASSERT_NE( it, nlists.end() )
            << "Locally-owned leaf " << c.key << " missing from P2P neighbor lists";

        const auto& nbrs = it->second;

        // Check 2: self-interaction present
        bool has_self = std::find( nbrs.begin(), nbrs.end(), c.key ) !=
                        nbrs.end();
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
        ncrit, max_depth, MPI_COMM_WORLD, tolerance, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );

    auto plan = build_plan( builder, partitioner, particles,
                            num_particles_per_rank, rank );

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
            << "Ghost leaf " << gk << " has OWNER_SHARED — ghosts must have a "
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
        EXPECT_TRUE( li->second )
            << "Ghost key " << gk << " is not a leaf";
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
        ncrit, max_depth, MPI_COMM_WORLD, tolerance, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );

    auto plan = build_plan( builder, partitioner, particles,
                            num_particles_per_rank, rank );

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
 * Verify that the M2L interaction lists are symmetric: if cell B appears
 * in cell A's interaction list then A must appear in B's interaction list,
 * provided both cells are processed by this rank. In a uniform tree this
 * is a strict invariant; in an adaptive tree it holds when both cells are
 * at the same depth and both owned (or shared) by this rank.
 *
 * Checks:
 *   1. For every (target, source) pair in interaction_lists where both
 *      keys appear as targets, the reverse pair also exists.
 */
void testM2LSymmetry( int num_particles_per_rank, int ncrit, int max_depth,
                      double tolerance, int replication_depth )
{
    using namespace CommunicationPlanTest;

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    AoSoA_t particles( "particles", num_particles_per_rank );
    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        ncrit, max_depth, MPI_COMM_WORLD, tolerance, tolerance );
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );

    auto plan = build_plan( builder, partitioner, particles,
                            num_particles_per_rank, rank );

    const auto& ilists = plan.m2l_plan().interaction_lists;

    // Build a set of all (target, source) pairs for fast lookup
    std::set<std::pair<MortonKey, MortonKey>> pairs;
    for ( const auto& [target, sources] : ilists )
        for ( MortonKey src : sources )
            pairs.insert( { target, src } );

    for ( const auto& [target, sources] : ilists )
    {
        for ( MortonKey src : sources )
        {
            // Only check symmetry when the source is also a target on
            // this rank (otherwise symmetry may be satisfied on the
            // remote rank that owns the source).
            if ( ilists.count( src ) == 0 )
                continue;

            EXPECT_TRUE( pairs.count( { src, target } ) > 0 )
                << "M2L asymmetry: " << target << " lists " << src
                << " as source, but " << src << " does not list " << target;
        }
    }
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

TEST( CommunicationPlan, testM2LSymmetryBasic )
{
    testM2LSymmetry( 10000, 128, 15, 0.1, 3 );
}

TEST( CommunicationPlan, testM2LSymmetrySmall )
{
    testM2LSymmetry( 500, 32, 10, 0.1, 2 );
}

//---------------------------------------------------------------------------//

} // end namespace Test
