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

#include <Canopy_Profiling.hpp>
#include <Canopy_TreeBuilder.hpp>
#include <Canopy_TreePartitioner.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    // Value: list of (source cell key, source cell index in cells vector).
    // Carrying the index alongside the key lets the downstream consumer
    // (DownwardSweep::build_interaction_list_device) skip a hash lookup
    // per source pair — there are tens of millions of source entries at
    // production sizes.
    std::unordered_map<MortonKey, std::vector<std::pair<MortonKey, int>>>
        interaction_lists;

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
    //
    // mac_theta selects the multipole acceptance criterion used by the
    // dual-tree traversal in build_all_interaction_lists. The MAC is the
    // exafmm-style spherical test R*theta > r_T + r_S where r is the
    // circumradius of a cube cell (sqrt(3)*half_width). Smaller theta is
    // more conservative (more pairs, deeper expansions converge); larger
    // theta is less conservative (fewer pairs, requires larger P for
    // equivalent accuracy). Default 0.5 is the closest single-knob match
    // to the legacy 4*max(half_width) Chebyshev rule. exafmm's default
    // is 0.4. Must satisfy 0 < theta < 1.
    // -----------------------------------------------------------------------
    CommunicationPlan( MPI_Comm comm, double mac_theta = 0.5 )
        : _comm( comm )
        , _valid( false )
        , _theta( mac_theta )
    {
        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_nprocs );
    }

    // Configure the MAC theta after construction. Triggers re-build on next
    // build() call; callers must call invalidate() if they have already
    // built a plan.
    void set_mac_theta( double mac_theta ) { _theta = mac_theta; }
    double mac_theta() const { return _theta; }

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

    // MAC theta used by the dual-tree traversal. See constructor.
    double _theta;

    // -----------------------------------------------------------------------
    // Cell lookup helpers built during build()
    // -----------------------------------------------------------------------
    std::unordered_map<MortonKey, const CellInfo*> _cell_map;
    const std::unordered_map<MortonKey, int>* _owner_map;
    int _replication_depth;

    // Base pointer of the `cells` vector passed to build(). Used by
    // emit_m2l_pair to compute the cell index of each source via pointer
    // arithmetic, which matches the index UpwardSweep stores in
    // _key_to_cell_idx (both index into the same `cells` vector).
    const CellInfo* _cells_base = nullptr;

    // Subtree relevance: true iff the subtree rooted at this cell contains
    // at least one cell that this rank processes (owns or shares). Used to
    // prune the dual-tree traversal in build_all_interaction_lists().
    std::unordered_map<MortonKey, bool> _subtree_relevant;

    // Scratch sets populated by build_all_interaction_lists, consumed by
    // finalize_m2l_plan / finalize_p2p_plan.
    std::set<std::pair<MortonKey, int>> _m2l_receives_set;
    std::set<std::pair<MortonKey, int>> _m2l_sends_set;
    std::set<MortonKey> _p2p_ghost_set;
    std::set<std::pair<MortonKey, int>> _p2p_send_set;

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
    // is_well_separated — geometric MAC shared by M2L and P2P partition.
    //
    // Two cells A, B are well-separated iff their Chebyshev (L_inf) center
    // distance exceeds 2 * max(box_width(A), box_width(B)) =
    // 4 * max(half_width(A), half_width(B)). For same-size cells this is
    // the standard FMM "one-cell buffer" rule; for asymmetric pairs it is
    // the conservative form needed for the M2L series to converge against
    // a coarser source. The complementary (near) test is
    // is_well_separated == false → both cells touch within the buffer →
    // P2P territory. Sharing this single predicate between DTT and any
    // debug assertion guarantees the two interaction lists partition the
    // global pair set with no overlap and no gap.
    // -----------------------------------------------------------------------
    bool is_well_separated( const CellInfo& a, const CellInfo& b ) const
    {
        const double eps = 1.0e-10;
        const double max_hw =
            ( a.half_width > b.half_width ) ? a.half_width : b.half_width;
        const double near_dist = 4.0 * max_hw;
        for ( int d = 0; d < 3; d++ )
        {
            if ( std::abs( a.center[d] - b.center[d] ) > near_dist + eps )
                return true;
        }
        return false;
    }

    // -----------------------------------------------------------------------
    // mac_satisfied — exafmm-style spherical multipole acceptance criterion.
    //
    // Two cells A, B are accepted as well-separated for M2L iff
    //     R * theta > r_A + r_B
    // where R = ||center_A - center_B||_2 and r = sqrt(3) * half_width is
    // the circumradius of a cube cell. The squared form below avoids the
    // sqrt. The predicate is symmetric in A, B, which is required by the
    // symmetric pair emission in emit_m2l_pair: both ranks (target owner,
    // source owner) must independently reach the same accept/reject
    // conclusion or send/receive sets diverge.
    //
    // Selecting theta is a knob: smaller theta is more conservative (more
    // pairs, M2L converges at lower P); larger theta is less conservative
    // (fewer pairs, requires larger P for the same accuracy). exafmm's
    // default is 0.4; Canopy's default 0.5 is the closest single-knob
    // match to the legacy 4*max(hw) Chebyshev rule for same-depth pairs.
    // -----------------------------------------------------------------------
    bool mac_satisfied( const CellInfo& a, const CellInfo& b ) const
    {
        const double dx = a.center[0] - b.center[0];
        const double dy = a.center[1] - b.center[1];
        const double dz = a.center[2] - b.center[2];
        const double R2 = dx * dx + dy * dy + dz * dz;
        constexpr double SQRT3 = 1.7320508075688772;
        const double r_sum = SQRT3 * ( a.half_width + b.half_width );
        return R2 * _theta * _theta > r_sum * r_sum;
    }

    // -----------------------------------------------------------------------
    // Internal plan builders
    // -----------------------------------------------------------------------
    void build_vertical_plans( const std::vector<CellInfo>& cells );

    // Compute _subtree_relevant bottom-up.
    void compute_subtree_relevance( const std::vector<CellInfo>& cells );

    // Single dual-tree traversal that fills _m2l_plan.interaction_lists,
    // _p2p_plan.neighbor_lists, and the scratch send/receive/ghost sets.
    void build_all_interaction_lists( const std::vector<CellInfo>& cells );

    // Materialize sends/receives/ghosts from the scratch sets into the
    // exposed plan vectors.
    void finalize_m2l_plan();
    void finalize_p2p_plan();

    // Symmetric emit helpers used by the DTT pass.
    void emit_m2l_pair( const CellInfo* A, const CellInfo* B );
    void emit_p2p_pair( MortonKey a, MortonKey b );
};

// ============================================================================
// Implementation
// ============================================================================

// --------------------------------------------------------------------------
// compute_subtree_relevance
//
// Sets _subtree_relevant[k] = true iff the subtree rooted at k contains any
// cell processed by this rank (owned or shared). Used to prune the dual-tree
// traversal: a (T, S) pair where neither subtree is relevant produces only
// pairs that this rank neither computes nor communicates, so the entire
// branch can be skipped.
//
// Implementation: process cells in order of decreasing depth (leaves first),
// initializing each cell's flag to rank_processes(key) and OR'ing it into
// the parent's entry. The result is exact because every descendant has been
// processed by the time we reach an ancestor.
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace>
void CommunicationPlan<MemorySpace, ExecutionSpace>::compute_subtree_relevance(
    const std::vector<CellInfo>& cells )
{
    _subtree_relevant.clear();

    // Single-rank fast path: every cell is processed by this rank, so the
    // relevance map is universally true and the DTT pruning branch is a
    // no-op. Skip both populating the map and the upward propagation.
    if ( _nprocs == 1 )
        return;

    _subtree_relevant.reserve( cells.size() );
    for ( const auto& c : cells )
        _subtree_relevant[c.key] = rank_processes( c.key );

    std::vector<MortonKey> by_depth_desc;
    by_depth_desc.reserve( cells.size() );
    for ( const auto& c : cells )
        by_depth_desc.push_back( c.key );
    std::sort( by_depth_desc.begin(), by_depth_desc.end(),
               []( MortonKey a, MortonKey b )
               { return key_depth( a ) > key_depth( b ); } );

    for ( MortonKey k : by_depth_desc )
    {
        if ( k == ROOT_KEY )
            continue;
        if ( !_subtree_relevant[k] )
            continue;
        MortonKey pk = parent_key( k );
        auto it = _subtree_relevant.find( pk );
        if ( it != _subtree_relevant.end() )
            it->second = true;
    }
}

// --------------------------------------------------------------------------
// emit_m2l_pair / emit_p2p_pair
//
// DTT visits each unordered cell pair {a, b} once. For M2L, both directed
// pairs (a target ← b source) AND (b target ← a source) contribute; we
// emit symmetrically so that build_interaction_list_device on the
// downstream consumer sees an entry for both endpoints whenever both are
// processed by this rank, and so that send/receive inference is local
// (each rank reaches the same conclusion about its own send/receive set).
//
// Receives are skipped for shared targets: shared cells live at depth
// <= replication_depth, and the FMM ensures their M2L sources are at
// equal-or-shallower depth (also shared on every rank), so no
// point-to-point exchange is required — the existing snapshot/allreduce
// path in DownwardSweep handles them. Sends are skipped symmetrically.
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace>
void CommunicationPlan<MemorySpace, ExecutionSpace>::emit_m2l_pair(
    const CellInfo* A, const CellInfo* B )
{
    const MortonKey a = A->key;
    const MortonKey b = B->key;
    const int a_idx = static_cast<int>( A - _cells_base );
    const int b_idx = static_cast<int>( B - _cells_base );
    const int oa = owner_of( a );
    const int ob = owner_of( b );

    // Each unordered pair {a, b} contributes two directed M2Ls. For each,
    // pick the rank that actually computes it: the target's owner for
    // non-shared targets, or rank 0 for shared targets (matching the
    // shared-target filter in DownwardSweep::build_interaction_list_device).
    // Populate interaction_lists[t] only on that rank, and route the
    // receive/send for s's multipole to/from the same rank.
    //
    // Shared sources are replicated on every rank after the M2M allreduce,
    // so they never require point-to-point communication.
    auto handle = [&]( MortonKey t, int ot, MortonKey s, int s_idx, int os )
    {
        const int compute_rank = ( ot == OWNER_SHARED ) ? 0 : ot;
        if ( compute_rank == _rank )
        {
            _m2l_plan.interaction_lists[t].push_back( { s, s_idx } );
            if ( os != OWNER_SHARED && os != _rank )
                _m2l_receives_set.insert( { s, os } );
        }
        else if ( os == _rank )
        {
            _m2l_sends_set.insert( { s, compute_rank } );
        }
    };
    handle( a, oa, b, b_idx, ob );
    handle( b, ob, a, a_idx, oa );
}

template <class MemorySpace, class ExecutionSpace>
void CommunicationPlan<MemorySpace, ExecutionSpace>::emit_p2p_pair(
    MortonKey a, MortonKey b )
{
    // Both endpoints are leaves and not well-separated. P2P targets are
    // owned (not shared — leaves do not live above the replication depth).
    const int oa = owner_of( a );
    const int ob = owner_of( b );
    const bool a_owned = ( oa == _rank );
    const bool b_owned = ( ob == _rank );

    auto handle = [&]( MortonKey t, MortonKey s, int os, bool t_owned )
    {
        if ( !t_owned )
            return;
        _p2p_plan.neighbor_lists[t].push_back( s );
        if ( t == s )
            return; // self-interaction, no ghost
        if ( os != _rank && os != OWNER_SHARED )
        {
            _p2p_ghost_set.insert( s );
            _p2p_send_set.insert( { t, os } );
        }
    };
    handle( a, b, ob, a_owned );
    if ( a != b )
        handle( b, a, oa, b_owned );
}

// --------------------------------------------------------------------------
// build_all_interaction_lists
//
// Single dual-tree traversal that produces both the M2L interaction lists
// and the P2P neighbor lists in one pass. This replaces the previous
// "parent's neighbors → children, minus my neighbors" recipe, which was
// correct only for 2:1-balanced trees: in an unbalanced adaptive tree it
// silently dropped the X-list (deep target, shallow non-adjacent leaf
// source) and the W-list (target's colleague's deep descendants that are
// well-separated from the target), producing an asymmetric, incomplete
// pair set.
//
// DTT enumerates every unordered (T, S) cell pair exactly once, classifies
// it via is_well_separated, and either:
//   - records an M2L pair (well-separated, both directions),
//   - records a P2P pair (both leaves, near), or
//   - splits the larger cell and recurses.
//
// Self-pairs on internal cells are split into asymmetric (Tc_i, Tc_j) for
// i <= j only — without this, the (ROOT, ROOT) starting pair would emit
// every descendant pair twice via reflected recursion paths.
//
// Subtree-relevance pruning skips any (T, S) where neither side touches
// this rank's processed cells, so the global pair set is enumerated only
// where it intersects this rank's compute or comm responsibilities.
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace>
void CommunicationPlan<MemorySpace, ExecutionSpace>::
    build_all_interaction_lists( const std::vector<CellInfo>& cells )
{
    (void)cells;
    _m2l_plan.interaction_lists.clear();
    _p2p_plan.neighbor_lists.clear();
    _m2l_receives_set.clear();
    _m2l_sends_set.clear();
    _p2p_ghost_set.clear();
    _p2p_send_set.clear();

    auto root_it = _cell_map.find( ROOT_KEY );
    if ( root_it == _cell_map.end() )
        return;

    // Carry CellInfo* on the stack — child enumeration looks up keys in
    // _cell_map once at push time, so pops are pointer dereferences with
    // no further hashing. Saves 2 hash lookups per visited pair vs. the
    // MortonKey-on-stack variant.
    using CellPair = std::pair<const CellInfo*, const CellInfo*>;
    std::vector<CellPair> stack;
    stack.reserve( 1024 );
    stack.push_back( { root_it->second, root_it->second } );

    while ( !stack.empty() )
    {
        auto [T, S] = stack.back();
        stack.pop_back();

        // Subtree-ownership pruning — skip pairs that touch nothing this
        // rank cares about. Skipped on a single rank since every cell is
        // relevant by definition (no remote ownership to consult).
        if ( _nprocs > 1 )
        {
            auto tr_it = _subtree_relevant.find( T->key );
            auto sr_it = _subtree_relevant.find( S->key );
            const bool tr = ( tr_it != _subtree_relevant.end() ) && tr_it->second;
            const bool sr = ( sr_it != _subtree_relevant.end() ) && sr_it->second;
            if ( !tr && !sr )
                continue;
        }

        // Self-pair: at a leaf this is the P2P self-interaction; at an
        // internal cell we split asymmetrically into child pairs (i, j)
        // for i <= j to avoid double-visiting reflected pairs.
        if ( T == S )
        {
            if ( T->is_leaf )
            {
                emit_p2p_pair( T->key, T->key );
                continue;
            }
            const CellInfo* children[8] = { nullptr };
            for ( int i = 0; i < 8; i++ )
            {
                auto ci_it = _cell_map.find( child_key( T->key, i ) );
                if ( ci_it != _cell_map.end() )
                    children[i] = ci_it->second;
            }
            for ( int i = 0; i < 8; i++ )
            {
                if ( !children[i] )
                    continue;
                for ( int j = i; j < 8; j++ )
                {
                    if ( !children[j] )
                        continue;
                    stack.push_back( { children[i], children[j] } );
                }
            }
            continue;
        }

        // MAC satisfied → M2L (one unordered pair, both directed M2Ls).
        if ( mac_satisfied( *T, *S ) )
        {
            emit_m2l_pair( T, S );
            continue;
        }

        // Both leaves, not well-separated → P2P.
        if ( T->is_leaf && S->is_leaf )
        {
            emit_p2p_pair( T->key, S->key );
            continue;
        }

        // Otherwise, split the larger cell. Splitting on >= (rather than
        // strict >) gives a deterministic tie-break for equal half-widths.
        const bool split_t =
            S->is_leaf || ( !T->is_leaf && T->half_width >= S->half_width );
        if ( split_t )
        {
            for ( int oct = 0; oct < 8; oct++ )
            {
                auto ck_it = _cell_map.find( child_key( T->key, oct ) );
                if ( ck_it != _cell_map.end() )
                    stack.push_back( { ck_it->second, S } );
            }
        }
        else
        {
            for ( int oct = 0; oct < 8; oct++ )
            {
                auto ck_it = _cell_map.find( child_key( S->key, oct ) );
                if ( ck_it != _cell_map.end() )
                    stack.push_back( { T, ck_it->second } );
            }
        }
    }

    // The DTT may emit duplicates of the same (target, source) directed
    // pair only via a self-pair self-emit on a leaf — which is intended
    // to appear exactly once. All other pairs are visited exactly once,
    // so no list-side dedup is required.

#if defined( CANOPY_NAN_DEBUG )
    // TEMPORARY (debug-nan): completeness invariant. For each leaf, the union
    // of (M2L source subtrees of the leaf and ALL its ancestors — those reach
    // the leaf via L2L) plus its P2P neighbor leaves must cover every particle
    // exactly once. Sum global_count over that set and compare to N_total. A
    // sum > N => double-counting (inflated far field); < N => a gap. Reports
    // the worst-offending leaf. Single-rank only (multi-rank subsets lists).
    if ( _nprocs == 1 )
    {
        auto root_c = _cell_map.find( ROOT_KEY );
        const long N_total =
            ( root_c != _cell_map.end() ) ? root_c->second->global_count : 0;
        long worst_leaf_key = -1;
        long worst_sum = N_total;
        long worst_delta = 0;
        long n_bad = 0;
        for ( const auto& kv : _cell_map )
        {
            const CellInfo* L = kv.second;
            if ( !L->is_leaf )
                continue;
            long sum = 0;
            // M2L sources along the ancestor chain (self -> ... -> root).
            MortonKey a = L->key;
            while ( true )
            {
                auto il = _m2l_plan.interaction_lists.find( a );
                if ( il != _m2l_plan.interaction_lists.end() )
                    for ( const auto& src : il->second )
                    {
                        auto sc = _cell_map.find( src.first );
                        if ( sc != _cell_map.end() )
                            sum += sc->second->global_count;
                    }
                if ( a == ROOT_KEY )
                    break;
                a = parent_key( a );
            }
            // P2P neighbor leaves (includes self).
            auto nl = _p2p_plan.neighbor_lists.find( L->key );
            if ( nl != _p2p_plan.neighbor_lists.end() )
                for ( MortonKey nk : nl->second )
                {
                    auto nc = _cell_map.find( nk );
                    if ( nc != _cell_map.end() )
                        sum += nc->second->global_count;
                }
            if ( sum != N_total )
            {
                ++n_bad;
                if ( std::labs( sum - N_total ) > std::labs( worst_delta ) )
                {
                    worst_delta = sum - N_total;
                    worst_sum = sum;
                    worst_leaf_key = static_cast<long>( L->key );
                }
            }
        }
        if ( _rank == 0 )
            std::fprintf(
                stderr,
                "[Canopy NaN-debug] M2L/P2P completeness: N_total=%ld "
                "bad_leaves=%ld worst_leaf_key=%ld worst_sum=%ld (delta=%ld)\n",
                N_total, n_bad, worst_leaf_key, worst_sum,
                worst_sum - N_total );
    }
#endif
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
                        _l2l_plan.receives.push_back( { ck, parent_owner } );
                    }
                }
            }
        }
    }

    _l2l_plan.max_depth = _m2m_plan.max_depth;
}

// --------------------------------------------------------------------------
// finalize_m2l_plan / finalize_p2p_plan
//
// The DTT pass populates interaction_lists / neighbor_lists and the scratch
// receive/send/ghost sets directly. These finalizers transcribe the scratch
// sets into the public CellTransfer vectors and drop scratch storage.
//
// Each interaction list is sorted before being exposed so that downstream
// CSR construction is deterministic across runs. The DTT visit order
// otherwise depends on stack pop order and would shuffle entries.
// --------------------------------------------------------------------------
namespace impl
{
// Stable LSD radix sort by .first (uint64 MortonKey). Equal keys preserve
// input order. n >= 1; scratch must hold n elements.
//
// Why: std::sort on vector<pair<MortonKey,int>> moves 12/16-byte payloads
// with branchy partition; radix-256 with parallel-array layout is
// sequential-access and branch-free, ~1.5-2x faster on uniform u64 keys
// for the list sizes we see here (~thousands per target).
//
// Determinism: stable LSD over u64 is order-deterministic for any input,
// satisfying the run-to-run reproducibility the previous std::sort was
// there to provide (tiebreak-by-int is dropped — equal source MortonKey
// within a target should not occur from DTT, and even if it did, any
// deterministic order would suffice).
inline void radix_sort_pairs_by_key(
    std::pair<MortonKey, int>* data, int n,
    std::pair<MortonKey, int>* scratch )
{
    constexpr int BITS = 8;
    constexpr int RADIX = 1 << BITS;
    int hist[RADIX];
    auto* src = data;
    auto* dst = scratch;
    for ( int pass = 0; pass < 8; ++pass )
    {
        const int shift = pass * BITS;
        for ( int b = 0; b < RADIX; ++b )
            hist[b] = 0;
        for ( int i = 0; i < n; ++i )
            ++hist[( static_cast<uint64_t>( src[i].first ) >> shift )
                   & 0xFFu];
        int sum = 0;
        for ( int b = 0; b < RADIX; ++b )
        {
            const int c = hist[b];
            hist[b] = sum;
            sum += c;
        }
        for ( int i = 0; i < n; ++i )
        {
            const int b = ( static_cast<uint64_t>( src[i].first ) >> shift )
                          & 0xFFu;
            dst[hist[b]++] = src[i];
        }
        std::swap( src, dst );
    }
    // 8 (even) passes => src == data on exit; result already in `data`.
}
} // namespace impl

template <class MemorySpace, class ExecutionSpace>
void CommunicationPlan<MemorySpace, ExecutionSpace>::finalize_m2l_plan()
{
    _m2l_plan.sends.clear();
    _m2l_plan.receives.clear();

    // Sort each per-target source list for downstream determinism.
    // The sorts are independent across targets — parallelize over a
    // materialized pointer array so the host execution space (OpenMP if
    // enabled, else Serial) can sort lists concurrently.
    {
        std::vector<std::vector<std::pair<MortonKey, int>>*> list_ptrs;
        list_ptrs.reserve( _m2l_plan.interaction_lists.size() );
        for ( auto& kv : _m2l_plan.interaction_lists )
            list_ptrs.push_back( &kv.second );
        const int n_lists = static_cast<int>( list_ptrs.size() );
        Kokkos::parallel_for(
            "finalize_m2l_sort_lists",
            Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
                0, n_lists ),
            [&]( int i )
            {
                auto& list = *list_ptrs[i];
                const int n = static_cast<int>( list.size() );
                // Small lists: introsort wins on tiny n (radix's 8-pass
                // overhead dominates).
                constexpr int RADIX_THRESHOLD = 64;
                if ( n < RADIX_THRESHOLD )
                {
                    std::sort( list.begin(), list.end() );
                }
                else
                {
                    // Per-thread reusable scratch — one alloc per OS
                    // thread amortized over all lists, no per-list malloc.
                    thread_local std::vector<std::pair<MortonKey, int>>
                        scratch;
                    if ( static_cast<int>( scratch.size() ) < n )
                        scratch.resize( n );
                    impl::radix_sort_pairs_by_key(
                        list.data(), n, scratch.data() );
#if defined( CANOPY_ENABLE_DEBUG )
                    for ( int j = 1; j < n; ++j )
                    {
                        if ( static_cast<uint64_t>( list[j - 1].first ) >
                             static_cast<uint64_t>( list[j].first ) )
                        {
                            std::fprintf(
                                stderr,
                                "[CANOPY_DEBUG] finalize_m2l radix sort: "
                                "out of order at idx %d (list size %d)\n",
                                j, n );
                            std::abort();
                        }
                    }
#endif
                }
            } );
    }

    for ( const auto& [key, from_rank] : _m2l_receives_set )
        _m2l_plan.receives.push_back( { key, from_rank } );
    for ( const auto& [key, to_rank] : _m2l_sends_set )
        _m2l_plan.sends.push_back( { key, to_rank } );

    _m2l_receives_set.clear();
    _m2l_sends_set.clear();
}

template <class MemorySpace, class ExecutionSpace>
void CommunicationPlan<MemorySpace, ExecutionSpace>::finalize_p2p_plan()
{
    _p2p_plan.ghost_leaf_keys.clear();
    _p2p_plan.ghost_leaf_owners.clear();
    _p2p_plan.send_leaves.clear();

    for ( auto& kv : _p2p_plan.neighbor_lists )
        std::sort( kv.second.begin(), kv.second.end() );

    for ( MortonKey gk : _p2p_ghost_set )
    {
        _p2p_plan.ghost_leaf_keys.push_back( gk );
        _p2p_plan.ghost_leaf_owners.push_back( owner_of( gk ) );
    }
    for ( const auto& [k, r] : _p2p_send_set )
        _p2p_plan.send_leaves.push_back( { k, r } );

    _p2p_ghost_set.clear();
    _p2p_send_set.clear();
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
    _cells_base = cells.data();

    {
        CANOPY_SCOPED_TIMER_DETAILED(
            Canopy::Profiling::TIMER_COMMPLAN_CELL_MAP_FILL );
        _cell_map.clear();
        _cell_map.reserve( cells.size() );
        for ( const auto& c : cells )
            _cell_map[c.key] = &c;
    }

    // Vertical plans (M2M / L2L) follow the parent-child topology and are
    // independent of the M2L/P2P pair set.
    {
        CANOPY_SCOPED_TIMER_DETAILED(
            Canopy::Profiling::TIMER_COMMPLAN_VERTICAL_PLANS );
        build_vertical_plans( cells );
    }

    // Compute subtree-relevance flags before the DTT so the traversal can
    // skip branches that don't intersect this rank's responsibilities.
    {
        CANOPY_SCOPED_TIMER_DETAILED(
            Canopy::Profiling::TIMER_COMMPLAN_SUBTREE_RELEVANCE );
        compute_subtree_relevance( cells );
    }

    // Single dual-tree traversal builds both the M2L interaction lists
    // and the P2P neighbor lists with symmetric (target, source) emissions.
    {
        CANOPY_SCOPED_TIMER_DETAILED(
            Canopy::Profiling::TIMER_COMMPLAN_DTT_TRAVERSAL );
        build_all_interaction_lists( cells );
    }

    // Materialize the scratch send/receive/ghost sets into the public plan
    // vectors.
    {
        CANOPY_SCOPED_TIMER_DETAILED(
            Canopy::Profiling::TIMER_COMMPLAN_FINALIZE_M2L );
        finalize_m2l_plan();
    }
    {
        CANOPY_SCOPED_TIMER_DETAILED(
            Canopy::Profiling::TIMER_COMMPLAN_FINALIZE_P2P );
        finalize_p2p_plan();
    }

    _valid = true;
}

} // namespace Canopy

#endif // CANOPY_COMMUNICATIONPLAN_HPP
