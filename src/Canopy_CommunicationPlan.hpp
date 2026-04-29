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

#ifndef CANOPY_COMMUNICATIONPLAN_HPP
#define CANOPY_COMMUNICATIONPLAN_HPP

#include <Canopy_TreeBuilder.hpp>
#include <Canopy_TreePartitioner.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Canopy
{

// ============================================================================
// CellTransfer — a single cell-data send or receive between ranks
// ============================================================================

struct CellTransfer
{
    MortonKey cell_key; // which cell's data is being transferred
    int remote_rank;    // the other rank involved
};

// ============================================================================
// VerticalPlan — communication plan for M2M (upward) or L2L (downward)
//
// During the upward sweep, a child cell's multipole coefficients must
// reach the parent cell's owner. During the downward sweep, a parent
// cell's local expansion must reach each child cell's owner.
//
// For shared (replicated) cells at or above the replication depth,
// no point-to-point communication is needed — all ranks compute
// partial contributions and use MPI_Allreduce instead.
// ============================================================================

struct VerticalPlan
{
    // Point-to-point transfers for cells below the replication depth.
    // Sends: cells whose data this rank must send to another rank.
    // Receives: cells whose data this rank expects from another rank.
    std::vector<CellTransfer> sends;
    std::vector<CellTransfer> receives;

    // Shared cells at or above replication depth that need allreduce
    // during the upward sweep (M2M). During the downward sweep (L2L),
    // shared cells don't need allreduce — all ranks already have
    // the complete local expansion from the parent.
    std::vector<MortonKey> shared_cells;

    // Maximum tree depth (for iterating layers in order)
    int max_depth;
};

// ============================================================================
// M2LPlan — communication plan for the M2L interaction phase
//
// Each cell needs multipole coefficients from cells in its interaction
// list. If a source cell is owned by a different rank (or is shared),
// the data must be communicated.
//
// We precompute the full interaction list for every cell this rank
// processes, then extract the cross-rank subset as sends/receives.
// ============================================================================

struct M2LPlan
{
    // The full interaction list for each cell this rank owns or shares.
    // Key: target cell that this rank will compute M2L for.
    // Value: list of source cells whose multipoles are needed.
    std::unordered_map<MortonKey, std::vector<MortonKey>> interaction_lists;

    // Cross-rank transfers: source cells this rank needs from others
    std::vector<CellTransfer> receives;

    // Source cells other ranks need from this rank
    std::vector<CellTransfer> sends;
};

// ============================================================================
// P2PPlan — communication plan for direct near-field particle interactions
//
// Each leaf cell needs particle data from neighboring leaf cells for
// direct (P2P) evaluation. If a neighbor is on a different rank,
// those particles must be halo-exchanged.
//
// The plan stores which neighbor leaves are remote (for building a
// Cabana::Halo) and the neighbor list for each local leaf.
// ============================================================================

struct P2PPlan
{
    // For each locally-owned leaf, the set of leaf cell keys that are
    // its P2P neighbors (same level or different level in adaptive case).
    // Includes the leaf itself (self-interaction).
    std::unordered_map<MortonKey, std::vector<MortonKey>> neighbor_lists;

    // Remote leaf keys whose particles this rank needs (ghost leaves)
    std::vector<MortonKey> ghost_leaf_keys;

    // Which rank owns each ghost leaf
    std::vector<int> ghost_leaf_owners;

    // Locally-owned leaves that this rank must send to other ranks, because
    // a remote rank has this leaf in its P2P neighbor list.
    // cell_key is the local leaf; remote_rank is the destination rank.
    // A single leaf may appear multiple times (once per destination rank).
    std::vector<CellTransfer> send_leaves;
};

// ============================================================================
// CommunicationPlan
//
// Precomputes and stores persistent communication plans for all four
// FMM phases. Call build() after tree construction and partitioning.
// If the tree topology or ownership changes, call build() again.
//
// The plans are independent of the expansion coefficients and particle
// data — they describe WHAT needs to be communicated and between WHICH
// ranks, not the data itself. The actual data exchange is performed by
// the FMM solver using these plans.
//
// Template Parameters:
//   DeviceType - Kokkos device type
// ============================================================================

template <class MemorySpace, class ExecutionSpace>
class CommunicationPlan
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------
    CommunicationPlan( MPI_Comm comm )
        : _comm( comm )
        , _valid( false )
    {
        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_nprocs );
    }

    // -----------------------------------------------------------------------
    // build()
    //
    // Constructs all communication plans from the tree topology and
    // ownership. Call after TopDownTreeBuilder::build() and
    // TreePartitioner::partition().
    //
    // Parameters:
    //   cells      - the global cell list from TopDownTreeBuilder::cells()
    //   ownership  - the ownership list from TreePartitioner::ownership()
    //   cell_owner_fn - function to look up a cell's owner by key
    //   replication_depth - depth at or below which cells are shared
    // -----------------------------------------------------------------------
    void build( const std::vector<CellInfo>& cells,
                const std::vector<CellOwnership>& ownership,
                const std::unordered_map<MortonKey, int>& cell_owner_map,
                int replication_depth );

    // Accessors (valid after build())
    const VerticalPlan& m2m_plan() const { return _m2m_plan; }
    const VerticalPlan& l2l_plan() const { return _l2l_plan; }
    const M2LPlan& m2l_plan() const { return _m2l_plan; }
    const P2PPlan& p2p_plan() const { return _p2p_plan; }
    bool valid() const { return _valid; }

    // Invalidate — call when tree changes
    void invalidate() { _valid = false; }

  private:
    MPI_Comm _comm;
    int _rank;
    int _nprocs;
    bool _valid;

    VerticalPlan _m2m_plan;
    VerticalPlan _l2l_plan;
    M2LPlan _m2l_plan;
    P2PPlan _p2p_plan;

    // -----------------------------------------------------------------------
    // Cell lookup helpers built during build()
    // -----------------------------------------------------------------------
    std::unordered_map<MortonKey, const CellInfo*> _cell_map;
    const std::unordered_map<MortonKey, int>* _owner_map;
    int _replication_depth;

    // -----------------------------------------------------------------------
    // Internal: look up owner of a cell, defaulting to OWNER_SHARED
    // -----------------------------------------------------------------------
    int owner_of( MortonKey key ) const
    {
        auto it = _owner_map->find( key );
        if ( it != _owner_map->end() )
            return it->second;
        return OWNER_SHARED;
    }

    // -----------------------------------------------------------------------
    // Internal: does this rank own or share a cell?
    // -----------------------------------------------------------------------
    bool rank_processes( MortonKey key ) const
    {
        int o = owner_of( key );
        return ( o == _rank || o == OWNER_SHARED );
    }

    // -----------------------------------------------------------------------
    // Internal plan builders
    // -----------------------------------------------------------------------
    void build_vertical_plans( const std::vector<CellInfo>& cells );
    void build_m2l_plan( const std::vector<CellInfo>& cells );
    void build_p2p_plan( const std::vector<CellInfo>& cells );

    // -----------------------------------------------------------------------
    // Neighbor-finding in the adaptive octree
    //
    // Two cells are "neighbors" if they are at the same depth and their
    // bounding boxes are adjacent (share a face, edge, or vertex).
    // In the Morton key scheme, neighbors at a given depth can be found
    // by examining all 3^3 - 1 = 26 spatial neighbors.
    //
    // For the adaptive case, a cell's neighbors may be at a coarser
    // depth (if a neighbor region wasn't refined). We handle this by
    // finding the leaf cell that contains each neighbor position.
    // -----------------------------------------------------------------------

    // Find all existing cells that are neighbors of the given cell.
    // Returns cells at the same depth if they exist, or their ancestor
    // leaf if they were not refined to that depth.
    std::vector<MortonKey> find_neighbors( MortonKey key,
                                           const CellInfo& cell ) const;

    // Find the leaf cell containing a given point by walking down
    // from the root.
    MortonKey find_leaf_containing_point( double px, double py,
                                          double pz ) const;

    // Build the M2L interaction list for a single cell.
    // The interaction list consists of children of the parent's
    // neighbors that are NOT neighbors of the cell itself.
    std::vector<MortonKey> build_interaction_list( MortonKey key,
                                                   const CellInfo& cell ) const;

    // Build the P2P neighbor list for a single leaf cell.
    // In an adaptive tree, this includes leaf cells at the same or
    // different depths whose spatial extents are adjacent.
    std::vector<MortonKey>
    build_p2p_neighbor_list( MortonKey key, const CellInfo& cell ) const;
};

// ============================================================================
// Implementation
// ============================================================================

// --------------------------------------------------------------------------
// find_leaf_containing_point
//
// Walk from root to leaf following the octant that contains the point.
// If we reach a cell that doesn't exist, return the last valid ancestor.
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace>
MortonKey
CommunicationPlan<MemorySpace, ExecutionSpace>::find_leaf_containing_point(
    double px, double py, double pz ) const
{
    MortonKey current = ROOT_KEY;

    while ( true )
    {
        auto it = _cell_map.find( current );
        if ( it == _cell_map.end() )
            return parent_key( current ); // safety fallback

        const CellInfo* ci = it->second;
        if ( ci->is_leaf )
            return current;

        // Determine octant
        int octant = 0;
        if ( px >= ci->center[0] )
            octant |= 1;
        if ( py >= ci->center[1] )
            octant |= 2;
        if ( pz >= ci->center[2] )
            octant |= 4;

        MortonKey child = child_key( current, octant );

        // If the child doesn't exist in the tree, the current cell's
        // subtree was pruned — but current is internal, so this
        // shouldn't happen in a well-formed tree. Return current as
        // a fallback.
        if ( _cell_map.find( child ) == _cell_map.end() )
            return current;

        current = child;
    }
}

// --------------------------------------------------------------------------
// find_neighbors
//
// Find all existing cells whose spatial extents are adjacent (share a
// face, edge, or vertex) to the query cell, returning the cell at the
// "natural" depth: same depth as the query if it exists, or a coarser
// ancestor leaf if the neighbor region was not refined that deeply.
//
// Two cells A and B are adjacent iff
//   |c_A[d] - c_B[d]| <= hw_A + hw_B   for all d in {0,1,2}.
// This is symmetric by construction.
//
// Strategy: tree traversal from the root. For each visited cell s:
//   - Prune the subtree if no descendant could be adjacent:
//       |c[d] - c_s[d]| > hw + 2*hw_s   for any d
//     (the closest a descendant center can lie is c_s[d] ± hw_s, and
//     the maximum descendant half-width is hw_s).
//   - If s is a leaf or s is at the query's depth, run the adjacency
//     test and emit s on success.
//   - Otherwise descend into s's children.
//
// We never descend below the query depth: a finer-than-query cell at
// the neighbor location would be represented in the result by its
// ancestor at query depth (if that ancestor is internal, we already
// emit it because it sits at query depth; if no ancestor at query
// depth exists, the parent leaf at coarser depth is returned).
//
// Why not the previous probe-point method? Sampling at ±2*hw and
// looking up the containing leaf only works when all leaves share the
// query's half-width. For adaptive trees the probe overshoots fine
// adjacent cells (returning a non-adjacent leaf) and undershoots
// coarse adjacent leaves whose centers lie outside the probe radius.
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace>
std::vector<MortonKey>
CommunicationPlan<MemorySpace, ExecutionSpace>::find_neighbors(
    MortonKey key, const CellInfo& cell ) const
{
    std::set<MortonKey> neighbor_set;

    const double hw  = cell.half_width;
    const double eps = 1.0e-10;
    const int qdepth = key_depth( key );

    std::vector<MortonKey> stack;
    stack.push_back( ROOT_KEY );

    while ( !stack.empty() )
    {
        MortonKey s = stack.back();
        stack.pop_back();

        auto s_it = _cell_map.find( s );
        if ( s_it == _cell_map.end() )
            continue;

        const CellInfo* s_ci = s_it->second;
        const double hw_s = s_ci->half_width;

        // FMM well-separation pruning. A descendant d of s has
        // hw_d <= hw_s and center within hw_s of s.center. For d to count
        // as "near" (excluded from M2L) we require
        //     dist(cell, d) <= 2 * max(hw_cell, hw_d) <= 2 * max(hw, hw_s),
        // so no descendant can qualify when
        //     dist(cell, s) > 2 * max(hw, hw_s) + hw_s + eps.
        const double max_hw_csci = ( hw > hw_s ) ? hw : hw_s;
        const double prune_dist = 2.0 * max_hw_csci + hw_s;
        bool can_contain_neighbor = true;
        for ( int d = 0; d < 3; d++ )
        {
            double dist = std::abs( cell.center[d] - s_ci->center[d] );
            if ( dist > prune_dist + eps )
            {
                can_contain_neighbor = false;
                break;
            }
        }
        if ( !can_contain_neighbor )
            continue;

        const bool at_query_depth = ( s_ci->depth == qdepth );

        if ( s_ci->is_leaf || at_query_depth )
        {
            if ( s == key )
                continue;

            // FMM near-list test: dist <= 2 * max(hw, hw_s). The traditional
            // adjacency rule dist <= hw + hw_s is correct only for same-size
            // cells; for a coarser-depth leaf neighbor (hw_s > hw) the M2L
            // series only converges when dist > 2*hw_s, so the wider rule is
            // required to exclude such cells from M2L and keep them in P2P.
            const double near_dist = 4.0 * max_hw_csci;
            bool adjacent = true;
            for ( int d = 0; d < 3; d++ )
            {
                double dist = std::abs( cell.center[d] - s_ci->center[d] );
                if ( dist > near_dist + eps )
                {
                    adjacent = false;
                    break;
                }
            }
            if ( adjacent )
                neighbor_set.insert( s );
        }
        else
        {
            for ( int oct = 0; oct < 8; oct++ )
            {
                MortonKey ck = child_key( s, oct );
                if ( _cell_map.count( ck ) )
                    stack.push_back( ck );
            }
        }
    }

    return std::vector<MortonKey>( neighbor_set.begin(),
                                   neighbor_set.end() );
}

// --------------------------------------------------------------------------
// build_interaction_list
//
// The M2L interaction list for cell C consists of cells that are:
//   - Children of C's parent's neighbors (i.e., "cousins")
//   - NOT neighbors of C itself (i.e., well-separated from C)
//   - At the same depth as C or are leaves at a coarser depth
//
// In the standard FMM, the interaction list has at most 189 entries
// in 3D (6^3 - 3^3 = 189). In an adaptive tree, interactions with
// coarser cells can occur when the tree isn't uniformly refined.
//
// For the adaptive case, we use the following approach:
//   1. Find the parent's neighbors (cells adjacent to the parent).
//   2. For each parent neighbor, collect its children (if internal)
//      or the cell itself (if leaf).
//   3. Exclude any cell that is a neighbor of C.
//   4. The remaining cells form the interaction list.
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace>
std::vector<MortonKey>
CommunicationPlan<MemorySpace, ExecutionSpace>::build_interaction_list(
    MortonKey key, const CellInfo& cell ) const
{
    if ( key == ROOT_KEY )
        return {}; // root has no interaction list

    MortonKey pk = parent_key( key );
    auto parent_it = _cell_map.find( pk );
    if ( parent_it == _cell_map.end() )
        return {};

    const CellInfo* parent_ci = parent_it->second;

    // Step 1: Find parent's neighbors
    auto parent_neighbors = find_neighbors( pk, *parent_ci );

    // Step 2: Find the cell's own neighbors (to exclude from
    //         interaction list)
    auto my_neighbors = find_neighbors( key, cell );
    std::unordered_set<MortonKey> my_neighbor_set( my_neighbors.begin(),
                                                   my_neighbors.end() );
    my_neighbor_set.insert( key ); // exclude self too

    // Step 3: For each parent neighbor, collect its children or itself
    std::vector<MortonKey> interaction_list;

    for ( MortonKey pn_key : parent_neighbors )
    {
        auto pn_it = _cell_map.find( pn_key );
        if ( pn_it == _cell_map.end() )
            continue;

        const CellInfo* pn_ci = pn_it->second;

        if ( pn_ci->is_leaf )
        {
            // Parent's neighbor is a leaf. It interacts with C at
            // C's level only if it's well-separated from C.
            // (This handles the adaptive case where a neighbor of the
            // parent wasn't refined as deeply as C.)
            if ( my_neighbor_set.find( pn_key ) == my_neighbor_set.end() )
            {
                interaction_list.push_back( pn_key );
            }
        }
        else
        {
            // Parent's neighbor is internal — check its children
            for ( int oct = 0; oct < 8; oct++ )
            {
                MortonKey ck = child_key( pn_key, oct );
                auto ck_it = _cell_map.find( ck );
                if ( ck_it == _cell_map.end() )
                    continue; // child was pruned (empty)

                // Exclude if this child is a neighbor of C
                if ( my_neighbor_set.find( ck ) != my_neighbor_set.end() )
                    continue;

                interaction_list.push_back( ck );
            }
        }
    }

    return interaction_list;
}

// --------------------------------------------------------------------------
// build_p2p_neighbor_list
//
// For a leaf cell, the P2P neighbor list includes all leaf cells whose
// spatial extents are adjacent (share a face, edge, or vertex). In an
// adaptive tree, these may be at different depths.
//
// Uses a tree traversal rather than fixed probe points. The probe-point
// approach is only correct when all leaves have the same half-width: a
// coarser leaf's probe at ±2*hw overshoots finer adjacent leaves whose
// centers are closer than 2*hw, making the neighbor relation asymmetric
// and breaking the symmetry of the ghost/send communication plan.
//
// Two cells A and B are adjacent iff
//   |c_A[d] - c_B[d]| <= hw_A + hw_B   for all d in {0,1,2}.
// This condition is symmetric by construction, so if A finds B, B finds A.
//
// Pruning: an internal cell X cannot contain any adjacent descendant leaf
// if, for any d, |c[d] - c_X[d]| > hw + 2*hw_X (because the closest a
// descendant leaf's center can be in dimension d is c_X[d] ± hw_X, and
// the maximum descendant hw is hw_X, so the tightest the adjacency
// condition can be satisfied is hw + hw_X vs. dist - hw_X).
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace>
std::vector<MortonKey>
CommunicationPlan<MemorySpace, ExecutionSpace>::build_p2p_neighbor_list(
    MortonKey key, const CellInfo& cell ) const
{
    std::set<MortonKey> neighbor_leaves;
    neighbor_leaves.insert( key ); // self-interaction

    const double hw  = cell.half_width;
    const double eps = 1.0e-10;

    std::vector<MortonKey> stack;
    stack.push_back( ROOT_KEY );

    while ( !stack.empty() )
    {
        MortonKey s = stack.back();
        stack.pop_back();

        auto s_it = _cell_map.find( s );
        if ( s_it == _cell_map.end() )
            continue;

        const CellInfo* s_ci = s_it->second;
        const double hw_s = s_ci->half_width;

        // FMM well-separation pruning. See find_neighbors above for the
        // derivation; we use 2*max(hw, hw_s) as the near-list threshold.
        const double max_hw_csci = ( hw > hw_s ) ? hw : hw_s;
        const double prune_dist = 2.0 * max_hw_csci + hw_s;
        bool can_contain_neighbor = true;
        for ( int d = 0; d < 3; d++ )
        {
            double dist = std::abs( cell.center[d] - s_ci->center[d] );
            if ( dist > prune_dist + eps )
            {
                can_contain_neighbor = false;
                break;
            }
        }
        if ( !can_contain_neighbor )
            continue;

        if ( s_ci->is_leaf )
        {
            if ( s == key )
                continue; // already inserted self

            // FMM near-list: dist <= 2*max(hw, hw_s) in every axis. This is
            // the same threshold used by find_neighbors so M2L and P2P
            // partition all cells consistently.
            const double near_dist = 4.0 * max_hw_csci;
            bool adjacent = true;
            for ( int d = 0; d < 3; d++ )
            {
                double dist = std::abs( cell.center[d] - s_ci->center[d] );
                if ( dist > near_dist + eps )
                {
                    adjacent = false;
                    break;
                }
            }
            if ( adjacent )
                neighbor_leaves.insert( s );
        }
        else
        {
            // Descend into children that passed the pruning check
            for ( int oct = 0; oct < 8; oct++ )
            {
                MortonKey ck = child_key( s, oct );
                if ( _cell_map.count( ck ) )
                    stack.push_back( ck );
            }
        }
    }

    return std::vector<MortonKey>( neighbor_leaves.begin(),
                                   neighbor_leaves.end() );
}

// --------------------------------------------------------------------------
// build_vertical_plans — M2M (upward) and L2L (downward)
//
// Walk through every parent-child relationship in the tree. If the
// parent and child have different owners (and neither is shared),
// a point-to-point transfer is needed.
//
// M2M: child owner sends to parent owner
// L2L: parent owner sends to child owner
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace>
void CommunicationPlan<MemorySpace, ExecutionSpace>::build_vertical_plans(
    const std::vector<CellInfo>& cells )
{
    _m2m_plan.sends.clear();
    _m2m_plan.receives.clear();
    _m2m_plan.shared_cells.clear();
    _m2m_plan.max_depth = 0;

    _l2l_plan.sends.clear();
    _l2l_plan.receives.clear();
    _l2l_plan.shared_cells.clear();
    _l2l_plan.max_depth = 0;

    for ( const auto& ci : cells )
    {
        if ( ci.depth > _m2m_plan.max_depth )
            _m2m_plan.max_depth = ci.depth;

        // Collect shared cells for allreduce during M2M
        if ( ci.depth <= _replication_depth && !ci.is_leaf )
        {
            _m2m_plan.shared_cells.push_back( ci.key );
            _l2l_plan.shared_cells.push_back( ci.key );
        }

        // Skip leaves (no children) and the root (no parent)
        if ( ci.is_leaf )
            continue;

        int parent_owner = owner_of( ci.key );

        // Check each child
        for ( int oct = 0; oct < 8; oct++ )
        {
            MortonKey ck = child_key( ci.key, oct );
            if ( _cell_map.find( ck ) == _cell_map.end() )
                continue; // child doesn't exist (pruned)

            int child_owner = owner_of( ck );

            // Skip if both are shared (allreduce handles it)
            if ( parent_owner == OWNER_SHARED && child_owner == OWNER_SHARED )
                continue;

            // Skip if same owner (no communication needed)
            if ( parent_owner == child_owner )
                continue;

            // ----- M2M: child → parent -----
            // The child's owner sends. The parent's owner receives.
            // If parent is shared, all ranks need the child data,
            // but we handle that via allreduce at shared layers.
            // So we only generate transfers for uniquely-owned parents.

            if ( parent_owner != OWNER_SHARED )
            {
                if ( child_owner == _rank || child_owner == OWNER_SHARED )
                {
                    // This rank has the child data — send to parent owner
                    _m2m_plan.sends.push_back( { ck, parent_owner } );
                }

                if ( parent_owner == _rank )
                {
                    // This rank owns the parent — expect data from
                    // child's owner
                    int source =
                        ( child_owner == OWNER_SHARED ) ? _rank : child_owner;
                    if ( source != _rank )
                    {
                        _m2m_plan.receives.push_back( { ck, source } );
                    }
                }
            }

            // ----- L2L: parent → child -----
            // When parent_owner == OWNER_SHARED, all ranks already hold
            // the parent local expansion (from the M2M allreduce), so
            // the child's owner can apply it locally — no p2p needed.
            //
            // The cell_key in L2L plan entries is the CHILD's key
            // (one entry per individual child transfer). Using child
            // keys keeps entries unique even when a parent has multiple
            // children with the same remote owner — using parent keys
            // here would generate duplicate plan entries that cause
            // exchange_locals_after_l2l_at_depth to send/receive the
            // same child's local multiple times, multiplying its value.
            if ( child_owner != OWNER_SHARED )
            {
                if ( parent_owner == _rank )
                {
                    // This rank owns the parent — send to child owner
                    if ( child_owner != _rank )
                    {
                        _l2l_plan.sends.push_back( { ck, child_owner } );
                    }
                }

                if ( child_owner == _rank )
                {
                    // This rank owns the child — expect data from
                    // parent's owner
                    if ( parent_owner != _rank && parent_owner != OWNER_SHARED )
                    {
                        _l2l_plan.receives.push_back(
                            { ck, parent_owner } );
                    }
                }
            }
        }
    }

    _l2l_plan.max_depth = _m2m_plan.max_depth;
}

// --------------------------------------------------------------------------
// build_m2l_plan — interaction list communication
//
// For each cell this rank processes, compute the interaction list.
// Identify which source cells are on remote ranks, and build the
// send/receive manifests.
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace>
void CommunicationPlan<MemorySpace, ExecutionSpace>::build_m2l_plan(
    const std::vector<CellInfo>& cells )
{
    _m2l_plan.interaction_lists.clear();
    _m2l_plan.sends.clear();
    _m2l_plan.receives.clear();

    // Track which source cells we need from each rank
    // and which cells other ranks need from us
    std::set<std::pair<MortonKey, int>> receives_set; // (key, from_rank)
    std::set<std::pair<MortonKey, int>> sends_set;    // (key, to_rank)

    for ( const auto& ci : cells )
    {
        // Only build interaction lists for cells this rank processes
        if ( !rank_processes( ci.key ) )
            continue;

        // Skip leaves at the root level (no interaction list)
        if ( ci.key == ROOT_KEY )
            continue;

        auto ilist = build_interaction_list( ci.key, ci );

        if ( !ilist.empty() )
            _m2l_plan.interaction_lists[ci.key] = ilist;

        // Shared cells use collective M2L communication across all ranks;
        // their interaction-list data is not exchanged point-to-point here.
        if ( owner_of( ci.key ) == OWNER_SHARED )
            continue;

        // Check which sources are on remote ranks
        for ( MortonKey source : ilist )
        {
            int source_owner = owner_of( source );

            if ( source_owner != _rank && source_owner != OWNER_SHARED )
            {
                // We need this cell's multipole from another rank
                receives_set.insert( { source, source_owner } );
            }
        }
    }

    // Convert receives to vector
    for ( const auto& [key, from_rank] : receives_set )
    {
        _m2l_plan.receives.push_back( { key, from_rank } );
    }

    // Now determine what this rank needs to send.
    // We need to know what other ranks need from us. Since the tree
    // and ownership are global, we can compute this symmetrically:
    // for each cell this rank owns, check if any other rank's cells
    // have it in their interaction list.
    //
    // Alternatively, we can use an MPI exchange of the receive lists
    // to determine sends. This is more robust for large rank counts.
    //
    // For now, since the tree is replicated on all ranks, we compute
    // sends by iterating over all cells and checking if their
    // interaction lists include cells we own.

    for ( const auto& ci : cells )
    {
        int target_owner = owner_of( ci.key );

        // Skip cells that this rank processes (we don't send to ourselves)
        if ( target_owner == _rank || target_owner == OWNER_SHARED )
            continue;

        // Skip root
        if ( ci.key == ROOT_KEY )
            continue;

        auto ilist = build_interaction_list( ci.key, ci );

        for ( MortonKey source : ilist )
        {
            int source_owner = owner_of( source );

            if ( source_owner == _rank )
            {
                // The target rank needs our cell's multipole
                sends_set.insert( { source, target_owner } );
            }
        }
    }

    for ( const auto& [key, to_rank] : sends_set )
    {
        _m2l_plan.sends.push_back( { key, to_rank } );
    }
}

// --------------------------------------------------------------------------
// build_p2p_plan — near-field neighbor lists and ghost identification
//
// For each locally-owned leaf, compute its P2P neighbor list (adjacent
// leaf cells). Identify which neighbors are on remote ranks — those
// will need particle halo exchange.
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace>
void CommunicationPlan<MemorySpace, ExecutionSpace>::build_p2p_plan(
    const std::vector<CellInfo>& cells )
{
    _p2p_plan.neighbor_lists.clear();
    _p2p_plan.ghost_leaf_keys.clear();
    _p2p_plan.ghost_leaf_owners.clear();
    _p2p_plan.send_leaves.clear();

    std::set<MortonKey> ghost_set;

    // Build the incoming side: for each leaf we own, compute its
    // neighbor list. Any neighbor whose owner is a remote rank is a
    // ghost we need to receive.
    for ( const auto& ci : cells )
    {
        if ( !ci.is_leaf )
            continue;

        int leaf_owner = owner_of( ci.key );
        if ( leaf_owner != _rank )
            continue;

        auto neighbors = build_p2p_neighbor_list( ci.key, ci );
        _p2p_plan.neighbor_lists[ci.key] = neighbors;

        for ( MortonKey nk : neighbors )
        {
            if ( nk == ci.key )
                continue;

            int nk_owner = owner_of( nk );
            if ( nk_owner != _rank && nk_owner != OWNER_SHARED )
            {
                ghost_set.insert( nk );
            }
        }
    }

    for ( MortonKey gk : ghost_set )
    {
        _p2p_plan.ghost_leaf_keys.push_back( gk );
        _p2p_plan.ghost_leaf_owners.push_back( owner_of( gk ) );
    }

    // --------------------------------------------------------------------
    // Build the outgoing side: for each leaf I own, for each of its
    // neighbors, if that neighbor's owner is a different rank, then that
    // rank has ME in its ghost list and needs to receive this leaf.
    //
    // Adjacency is symmetric: if N is a P2P neighbor of L, then L is a
    // P2P neighbor of N. Since the tree is replicated, we can compute
    // sends locally with no MPI.
    //
    // Duplicates: if the same remote rank owns multiple of a leaf's
    // neighbors, we'd add the send multiple times. Deduplicate via a
    // set of (leaf_key, remote_rank) pairs.
    // --------------------------------------------------------------------
    std::set<std::pair<MortonKey, int>> send_set;

    for ( const auto& ci : cells )
    {
        if ( !ci.is_leaf )
            continue;

        int leaf_owner = owner_of( ci.key );
        if ( leaf_owner != _rank )
            continue;

        // We already built the neighbor list for this leaf above;
        // look it up.
        auto nit = _p2p_plan.neighbor_lists.find( ci.key );
        if ( nit == _p2p_plan.neighbor_lists.end() )
            continue;

        for ( MortonKey nk : nit->second )
        {
            if ( nk == ci.key )
                continue;
            int nk_owner = owner_of( nk );
            if ( nk_owner == _rank || nk_owner == OWNER_SHARED )
                continue;
            // nk_owner is a remote rank that needs my leaf ci.key
            send_set.insert( { ci.key, nk_owner } );
        }
    }

    for ( const auto& [k, r] : send_set )
        _p2p_plan.send_leaves.push_back( { k, r } );
}

// --------------------------------------------------------------------------
// build() — main entry point
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace>
void CommunicationPlan<MemorySpace, ExecutionSpace>::build(
    const std::vector<CellInfo>& cells,
    const std::vector<CellOwnership>& ownership,
    const std::unordered_map<MortonKey, int>& cell_owner_map,
    int replication_depth )
{
    _owner_map = &cell_owner_map;
    _replication_depth = replication_depth;

    // Build cell lookup
    _cell_map.clear();
    _cell_map.reserve( cells.size() );
    for ( const auto& c : cells )
        _cell_map[c.key] = &c;

    // Build all four plans
    build_vertical_plans( cells );
    build_m2l_plan( cells );
    build_p2p_plan( cells );

    _valid = true;
}

} // namespace Canopy

#endif // CANOPY_COMMUNICATIONPLAN_HPP
