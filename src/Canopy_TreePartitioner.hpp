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

#ifndef CANOPY_TREE_PARTITIONER_HPP
#define CANOPY_TREE_PARTITIONER_HPP

#include <Canopy_RegisteredBufferPool.hpp>
#include <Canopy_TreeBuilder.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <parmetis.h>

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace Canopy
{

// ============================================================================
// Ownership constants
// ============================================================================

// Cells at or above the replication cutoff depth are owned by all ranks.
// This value indicates shared (replicated) ownership.
static constexpr int OWNER_SHARED = -1;

// ============================================================================
// CellOwnership
// ============================================================================

struct CellOwnership
{
    MortonKey key;
    int owner_rank;
};

// ============================================================================
// PartitionBand
//
// A contiguous range of tree depths [depth_lo, depth_hi] (inclusive) whose
// cells share one ParMETIS balance constraint.
// ============================================================================

struct PartitionBand
{
    int depth_lo;
    int depth_hi;
};

// ============================================================================
// Lattice coordinates of a cell at its own depth, in [0, 2^depth) per axis.
// Octant bit 0 is x, bit 1 is y, bit 2 is z (TreeBuilder::which_octant).
// ============================================================================

inline void key_to_lattice( MortonKey k, int depth, uint64_t ijk[3] )
{
    ijk[0] = ijk[1] = ijk[2] = 0;
    for ( int l = depth - 1; l >= 0; --l )
    {
        const int oct = static_cast<int>( ( k >> ( 3 * l ) ) & 7 );
        for ( int a = 0; a < 3; ++a )
            ijk[a] = ( ijk[a] << 1 ) | static_cast<uint64_t>( ( oct >> a ) & 1 );
    }
}

inline MortonKey lattice_to_key( const uint64_t ijk[3], int depth )
{
    MortonKey k = ROOT_KEY;
    for ( int l = depth - 1; l >= 0; --l )
    {
        int oct = 0;
        for ( int a = 0; a < 3; ++a )
            oct |= static_cast<int>( ( ijk[a] >> l ) & 1 ) << a;
        k = child_key( k, oct );
    }
    return k;
}

// ============================================================================
// RedistributeResult
// ============================================================================

struct RedistributeResult
{
    int particles_sent;
    int particles_received;
    int num_local_after;
};

// ============================================================================
// TreePartitioner
//
// Given a globally-agreed adaptive octree (from TreeBuilder), partitions
// every non-shared cell (each leaf, and each internal cell deeper than
// replication_depth) across MPI ranks with a distributed ParMETIS graph
// partition, keeps cells at depth <= replication_depth replicated, migrates
// particles to their owning ranks, and sorts particles by leaf cell index.
// docs/design.md, "Load Balancing", describes the graph.
//
// Sort-by-leaf invariant (after partition/repartition + sort_by_leaf):
//   For each cell index i, particles in that cell live in the AoSoA at
//   contiguous indices [leaf_particle_offsets(i), leaf_particle_offsets(i+1)).
//   Cells this rank does not own have zero-length ranges.
//
// Template Parameters:
//   MemorySpace, ExecutionSpace - Kokkos memory/execution spaces
// ============================================================================

template <class MemorySpace, class ExecutionSpace>
class TreePartitioner
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;

    // -----------------------------------------------------------------------
    // Constructor
    //
    // replication_depth: cells at depth <= this value are replicated on all
    //     ranks (OWNER_SHARED). Deeper cells get unique owners. Typical
    //     values should 2-4. At depth 3 there are at most 585 cells (1 + 8 +
    //     64 + 512), so the allreduce cost at coarse layers
    //     is relatively small.
    //
    // imbalance_tolerance: allowed relative excess of each balance
    //     constraint's per-rank load over its mean (e.g., 0.05 means max/mean
    //     <= 1.05); passed to ParMETIS as ubvec
    // -----------------------------------------------------------------------
    TreePartitioner( MPI_Comm comm, int replication_depth = 3,
                     double imbalance_tolerance = 0.05 )
        : _comm( comm )
        , _replication_depth( replication_depth )
        , _imbalance_tolerance( 1.0 + imbalance_tolerance )
    {
        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    // Full ownership map, indexed parallel to tree_builder.cells()
    const std::vector<CellOwnership>& ownership() const { return _ownership; }

    // Lookup owner of a specific cell by Morton key.
    // Returns OWNER_SHARED for replicated cells, or the owning rank.
    int cell_owner( MortonKey key ) const
    {
        auto it = _cell_owner_map.find( key );
        if ( it != _cell_owner_map.end() )
            return it->second;
        return OWNER_SHARED;
    }

    // Access the full owner map (for CommunicationPlan)
    const std::unordered_map<MortonKey, int>& cell_owner_map() const
    {
        return _cell_owner_map;
    }

    // Replication depth (for CommunicationPlan)
    int replication_depth() const { return _replication_depth; }

    // Depth bands of the most recent partition: constraint 1 + b weighs the
    // cells of bands()[b]. Empty when no cell is deeper than
    // replication_depth.
    const std::vector<PartitionBand>& bands() const { return _bands; }

    int num_local_particles() const { return _num_local_after; }

    // -----------------------------------------------------------------------
    // leaf_particle_offsets()
    //
    // View of size (num_cells + 1). For cell index i, particles in that
    // leaf live at indices [offsets(i), offsets(i+1)) in the sorted AoSoA.
    // Cells that are not owned by this rank (or aren't leaves) have
    // zero-length ranges.
    //
    // Valid only after sort_particles_by_leaf() has been called following
    // partition()/repartition() and builder.build().
    // -----------------------------------------------------------------------
    const Kokkos::View<int*, memory_space>& leaf_particle_offsets() const
    {
        return _leaf_particle_offsets;
    }

    // -----------------------------------------------------------------------
    // particle_leaf_cell_idx()
    //
    // View of size num_local_particles. For particle p (in sorted order),
    // particle_leaf_cell_idx(p) is the cell index of p's containing leaf.
    // Convenient for kernels that need the leaf index of each particle.
    // -----------------------------------------------------------------------
    const Kokkos::View<int*, memory_space>& particle_leaf_cell_idx() const
    {
        return _particle_leaf_cell_idx;
    }

  private:
    // MPI
    MPI_Comm _comm;
    int _rank;
    int _comm_size;

    // Parameters
    int _replication_depth;
    double _imbalance_tolerance;

    // Output: ownership for each cell
    std::vector<CellOwnership> _ownership;

    // Fast lookup: MortonKey -> owner rank
    std::unordered_map<MortonKey, int> _cell_owner_map;

    // Cell assignment from the most recent partition_cells() call, plus the
    // leaves refresh_ownership_for_current_tree() has voted on since. Read by
    // that refresh so it keeps the partitioned owners instead of
    // re-partitioning and re-migrating particles.
    std::unordered_map<MortonKey, int> _cached_cell_owners;

    // Depth bands of the most recent partition_cells() call.
    std::vector<PartitionBand> _bands;

    // Post-migration local particle count
    int _num_local_after;

    // Sort outputs (valid after sort_particles_by_leaf)
    Kokkos::View<int*, memory_space> _leaf_particle_offsets;
    Kokkos::View<int*, memory_space> _particle_leaf_cell_idx;

    // Persistent, grow-only staging buffers for the coalesced particle
    // migration in migrate_particles(). One stable registered device region
    // per direction (send/recv tuple data), reused across every rebalance so
    // the CXI NIC registration footprint stays O(1) per direction regardless
    // of peer count — the same treatment coalesced_view_exchange and the P2P
    // ghost gather already give the solve() exchanges. The data pools are
    // byte-typed because the AoSoA tuple type is only known inside the
    // migrate_particles template method; per call they are reinterpreted to
    // an unmanaged View of that tuple type. The int pool holds the packed
    // send-index list. See Canopy_RegisteredBufferPool.hpp.
    detail::RegisteredBufferPool<char, memory_space> _migrate_send_pool;
    detail::RegisteredBufferPool<char, memory_space> _migrate_recv_pool;
    detail::RegisteredBufferPool<int, memory_space> _migrate_send_idx_pool;

  public:
    // -----------------------------------------------------------------------
    // partition_cells
    //
    // ParMETIS graph partition of every non-shared cell. adaptive selects
    // ParMETIS_V3_AdaptiveRepart starting from the current _cell_owner_map;
    // otherwise ParMETIS_V3_PartKway from a block distribution. Collective.
    // Returns non-shared cell MortonKey -> owning rank, identical on every
    // rank. Throws std::runtime_error if ranks disagree on the vertex count,
    // ParMETIS fails on any rank, or a part falls outside [0, comm_size).
    // -----------------------------------------------------------------------
    std::unordered_map<MortonKey, int>
    partition_cells( const std::vector<CellInfo>& cells, bool adaptive );

    // -----------------------------------------------------------------------
    // vote_internal_owners
    //
    // The majority-vote rule: each internal cell deeper than
    // replication_depth goes to the rank owning the most descendant particles
    // (leaf global_count), ties to the lowest rank. Reads only the leaf
    // entries of owners. Returns internal cell MortonKey -> rank for every
    // such cell with a voting descendant.
    // -----------------------------------------------------------------------
    std::unordered_map<MortonKey, int>
    vote_internal_owners( const std::vector<CellInfo>& cells,
                          const std::unordered_map<MortonKey, int>& owners ) const;

    // -----------------------------------------------------------------------
    // derive_internal_ownership
    //
    // Fills _ownership and _cell_owner_map for cells. Every leaf must have an
    // entry in owners (throws otherwise). Internal cells at depth <=
    // replication_depth are OWNER_SHARED; deeper ones take their owners entry
    // if present, else the vote rule.
    // -----------------------------------------------------------------------
    void derive_internal_ownership(
        const std::vector<CellInfo>& cells,
        const std::unordered_map<MortonKey, int>& owners );

    // -----------------------------------------------------------------------
    // migrate_particles — coalesced, registration-bounded particle migration
    // (RegisteredBufferPool-backed; one registered region per direction)
    // -----------------------------------------------------------------------
    template <class AoSoAType>
    int migrate_particles(
        const TreeBuilder<MemorySpace, ExecutionSpace>& tree_builder,
        AoSoAType& particles, int num_local_particles_before );

    // -----------------------------------------------------------------------
    // partition() — initial partitioning
    // -----------------------------------------------------------------------
    template <class AoSoAType>
    void
    partition( const TreeBuilder<MemorySpace, ExecutionSpace>& tree_builder,
               AoSoAType& particles, int num_local_particles_before );

    // -----------------------------------------------------------------------
    // redistribute() — lightweight per-timestep migration
    // -----------------------------------------------------------------------
    template <class AoSoAType>
    RedistributeResult
    redistribute( const TreeBuilder<MemorySpace, ExecutionSpace>& tree_builder,
                  AoSoAType& particles, int num_local_particles_before );

    // -----------------------------------------------------------------------
    // repartition() — full re-partitioning after tree topology change
    // -----------------------------------------------------------------------
    template <class AoSoAType>
    void
    repartition( const TreeBuilder<MemorySpace, ExecutionSpace>& tree_builder,
                 AoSoAType& particles, int num_local_particles_before );

    // -----------------------------------------------------------------------
    // refresh_ownership_for_current_tree()
    //
    // Re-populate _cell_owner_map against tree_builder.cells() WITHOUT
    // re-partitioning and WITHOUT migrating particles. Every non-shared cell
    // present in the cached assignment of the most recent partition_cells()
    // call keeps its partitioned owner. A leaf absent from it goes to the
    // rank holding the most local particles in that leaf (one Allgather of
    // per-leaf counts), and an absent internal cell to the vote rule. This
    // keeps cell_owner_map consistent with the final cells passed to
    // comm_plan.build, fixing the phantom-send M2M plan that arises when the
    // post-migration build produces a different tree than the pre-partition
    // build.
    // -----------------------------------------------------------------------
    template <class AoSoAType>
    void refresh_ownership_for_current_tree(
        const TreeBuilder<MemorySpace, ExecutionSpace>& tree_builder,
        AoSoAType& particles );

    // -----------------------------------------------------------------------
    // sort_particles_by_leaf()
    //
    // Sorts the AoSoA so that particles in the same leaf cell are contiguous.
    // Builds _leaf_particle_offsets and _particle_leaf_cell_idx.
    //
    // Preconditions:
    //   - partition() or repartition() has been called.
    //   - builder.build() has been called (so particle_keys reflect the
    //     current AoSoA order).
    //
    // Postconditions:
    //   - The AoSoA has been permuted. Particles are grouped by leaf cell.
    //   - _leaf_particle_offsets is populated.
    //   - _particle_leaf_cell_idx is populated.
    //   - builder.particle_keys() is permuted in place to match the new AoSoA
    //     order (tree topology is unchanged across the sort, so re-running a
    //     full builder.build() just to refresh key order is unnecessary).
    // -----------------------------------------------------------------------
    template <class AoSoAType>
    void sort_particles_by_leaf(
        TreeBuilder<MemorySpace, ExecutionSpace>& tree_builder,
        AoSoAType& particles );
};

// ============================================================================
// Implementation
// ============================================================================

template <class MemorySpace, class ExecutionSpace>
std::unordered_map<MortonKey, int>
TreePartitioner<MemorySpace, ExecutionSpace>::partition_cells(
    const std::vector<CellInfo>& cells, bool adaptive )
{
    // Vertices: every non-shared cell, in Morton pre-order (each key shifted
    // to the deepest depth, ancestors first on ties). cells is identical on
    // every rank (TreeBuilder builds the full topology from all-reduced
    // counts), so this list is too.
    std::vector<int> vtx; // index into cells, per Morton position m
    int deepest = 0;
    for ( int i = 0; i < static_cast<int>( cells.size() ); ++i )
    {
        const auto& c = cells[i];
        if ( c.is_leaf || c.depth > _replication_depth )
        {
            vtx.push_back( i );
            deepest = std::max( deepest, c.depth );
        }
    }
    std::sort( vtx.begin(), vtx.end(),
               [&]( int a, int b )
               {
                   const MortonKey ka = cells[a].key
                                        << ( 3 * ( deepest - cells[a].depth ) );
                   const MortonKey kb = cells[b].key
                                        << ( 3 * ( deepest - cells[b].depth ) );
                   return ka != kb ? ka < kb : cells[a].depth < cells[b].depth;
               } );
    const int n = static_cast<int>( vtx.size() );

    // Bands: the non-shared depths replication_depth+1 .. deepest, split into
    // B = min(3, count) contiguous groups of near-equal depth count.
    _bands.clear();
    const int n_depths = std::max( 0, deepest - _replication_depth );
    const int num_bands = std::min( 3, n_depths );
    for ( int b = 0, lo = _replication_depth + 1; b < num_bands; ++b )
    {
        const int width =
            n_depths / num_bands + ( b < n_depths % num_bands ? 1 : 0 );
        _bands.push_back( { lo, lo + width - 1 } );
        lo += width;
    }
    const int ncon = 1 + num_bands;

    std::unordered_map<MortonKey, int> result;
    result.reserve( n );
    if ( _comm_size == 1 )
    {
        for ( int m = 0; m < n; ++m )
            result[cells[vtx[m]].key] = 0;
        return result;
    }

    long long n_minmax[2] = { n, -static_cast<long long>( n ) };
    MPI_Allreduce( MPI_IN_PLACE, n_minmax, 2, MPI_LONG_LONG, MPI_MAX, _comm );
    if ( n_minmax[0] != -n_minmax[1] )
        throw std::runtime_error(
            "TreePartitioner::partition_cells: ranks disagree on the vertex "
            "count (max " +
            std::to_string( n_minmax[0] ) + ", min " +
            std::to_string( -n_minmax[1] ) + ")" );

    // Supplier rank of each vertex. partition: rank r supplies the r-th
    // contiguous block of the Morton list. repartition: a cell's previous
    // owner supplies it; cells new to the tree follow the block rule.
    std::vector<long long> block_start( _comm_size + 1 );
    for ( int r = 0; r <= _comm_size; ++r )
        block_start[r] = static_cast<long long>( r ) * n / _comm_size;
    std::vector<int> supplier( n );
    for ( int m = 0; m < n; ++m )
    {
        supplier[m] = static_cast<int>(
            std::upper_bound( block_start.begin(), block_start.end() - 1, m ) -
            block_start.begin() - 1 );
        if ( adaptive )
        {
            auto it = _cell_owner_map.find( cells[vtx[m]].key );
            if ( it != _cell_owner_map.end() && it->second >= 0 &&
                 it->second < _comm_size )
                supplier[m] = it->second;
        }
    }

    // ParMETIS global IDs must be contiguous per rank: number vertices by
    // (supplier, Morton position).
    std::vector<idx_t> vtxdist( _comm_size + 1, 0 );
    for ( int m = 0; m < n; ++m )
        vtxdist[supplier[m] + 1]++;
    for ( int r = 0; r < _comm_size; ++r )
        vtxdist[r + 1] += vtxdist[r];
    std::vector<int> gid_of_m( n ), m_of_gid( n );
    {
        std::vector<idx_t> cursor( vtxdist.begin(), vtxdist.end() - 1 );
        for ( int m = 0; m < n; ++m )
        {
            const int g = static_cast<int>( cursor[supplier[m]]++ );
            gid_of_m[m] = g;
            m_of_gid[g] = m;
        }
    }
    std::unordered_map<MortonKey, int> m_of_key;
    m_of_key.reserve( n );
    for ( int m = 0; m < n; ++m )
        m_of_key[cells[vtx[m]].key] = m;

    // Local rows: parent-child edges between non-shared cells (M2M and L2L
    // traffic), and same-depth face/edge/corner neighbours (near-field and
    // M2L proximity). A parent-child edge weighs 27 and a neighbour edge 1, so
    // one parent-child edge outweighs all 26 neighbours of a cell; with unit
    // weights ParMETIS cuts more parent-child pairs than the vote rule
    // (fix-hang-rebalance-progress-log.md, H2 partitioner arm). Symmetric by
    // construction.
    constexpr idx_t parent_child_weight = 27;
    const idx_t g_lo = vtxdist[_rank];
    const idx_t n_local = vtxdist[_rank + 1] - g_lo;
    std::vector<idx_t> xadj( 1, 0 ), adjncy, adjwgt, vwgt( n_local * ncon, 0 );
    auto add_edge = [&]( MortonKey k, idx_t w )
    {
        auto it = m_of_key.find( k );
        if ( it != m_of_key.end() )
        {
            adjncy.push_back( gid_of_m[it->second] );
            adjwgt.push_back( w );
        }
    };
    for ( idx_t l = 0; l < n_local; ++l )
    {
        const CellInfo& c = cells[vtx[m_of_gid[g_lo + l]]];
        if ( c.depth > 0 )
            add_edge( parent_key( c.key ), parent_child_weight );
        if ( !c.is_leaf )
            for ( int oct = 0; oct < MAX_CHILDREN; ++oct )
                add_edge( child_key( c.key, oct ), parent_child_weight );
        uint64_t ijk[3];
        key_to_lattice( c.key, c.depth, ijk );
        const int64_t side = int64_t( 1 ) << c.depth;
        for ( int dx = -1; dx <= 1; ++dx )
            for ( int dy = -1; dy <= 1; ++dy )
                for ( int dz = -1; dz <= 1; ++dz )
                {
                    const int64_t nb[3] = { static_cast<int64_t>( ijk[0] ) + dx,
                                            static_cast<int64_t>( ijk[1] ) + dy,
                                            static_cast<int64_t>( ijk[2] ) + dz };
                    if ( ( dx == 0 && dy == 0 && dz == 0 ) || nb[0] < 0 ||
                         nb[1] < 0 || nb[2] < 0 || nb[0] >= side ||
                         nb[1] >= side || nb[2] >= side )
                        continue;
                    const uint64_t nbu[3] = { static_cast<uint64_t>( nb[0] ),
                                              static_cast<uint64_t>( nb[1] ),
                                              static_cast<uint64_t>( nb[2] ) };
                    add_edge( lattice_to_key( nbu, c.depth ), 1 );
                }
        xadj.push_back( static_cast<idx_t>( adjncy.size() ) );

        // Constraint 0: particles (P2P, P2M, L2P work). Constraint 1 + b: one
        // per cell in band b (M2M, M2L, L2L work).
        vwgt[l * ncon] = c.is_leaf ? c.global_count : 0;
        for ( int b = 0; b < num_bands; ++b )
            if ( c.depth >= _bands[b].depth_lo && c.depth <= _bands[b].depth_hi )
                vwgt[l * ncon + 1 + b] = 1;
    }

    // ParMETIS rejects a rank with no vertices: solve on the ranks that have
    // some, ordered as in vtxdist.
    MPI_Comm pm_comm;
    MPI_Comm_split( _comm, n_local > 0 ? 0 : MPI_UNDEFINED, _rank, &pm_comm );
    std::vector<idx_t> pm_vtxdist( 1, 0 );
    for ( int r = 0; r < _comm_size; ++r )
        if ( vtxdist[r + 1] > vtxdist[r] )
            pm_vtxdist.push_back( vtxdist[r + 1] );
    const int n_keep = static_cast<int>( pm_vtxdist.size() ) - 1;

    std::vector<idx_t> part( n_local, _rank );
    int pm_status = METIS_OK;
    if ( pm_comm != MPI_COMM_NULL )
    {
        idx_t wgtflag = 3; // vertex and edge weights
        idx_t numflag = 0;
        idx_t pm_ncon = ncon;
        idx_t nparts = _comm_size;
        idx_t edgecut = 0;
        std::vector<real_t> tpwgts( static_cast<size_t>( ncon ) * _comm_size,
                                    real_t( 1 ) / real_t( _comm_size ) );
        std::vector<real_t> ubvec( ncon,
                                   static_cast<real_t>( _imbalance_tolerance ) );
        // use options, no debug output, fixed seed, and part[] on input is the
        // current partition (sub-domains need not match processes).
        idx_t options[4] = { 1, 0, 15, PARMETIS_PSR_UNCOUPLED };
        idx_t dummy_adj = 0, dummy_wgt = 1;
        idx_t* adj = adjncy.empty() ? &dummy_adj : adjncy.data();
        idx_t* adj_w = adjwgt.empty() ? &dummy_wgt : adjwgt.data();
        if ( adaptive && n_keep >= 2 )
        {
            std::vector<idx_t> vsize( n_local, 1 );
            real_t itr = 100.0; // inter-processor comm vs. redistribution cost
            pm_status = ParMETIS_V3_AdaptiveRepart(
                pm_vtxdist.data(), xadj.data(), adj, vwgt.data(), vsize.data(),
                adj_w, &wgtflag, &numflag, &pm_ncon, &nparts, tpwgts.data(),
                ubvec.data(), &itr, options, &edgecut, part.data(), &pm_comm );
        }
        else
        {
            pm_status = ParMETIS_V3_PartKway(
                pm_vtxdist.data(), xadj.data(), adj, vwgt.data(), adj_w,
                &wgtflag, &numflag, &pm_ncon, &nparts, tpwgts.data(),
                ubvec.data(), options, &edgecut, part.data(), &pm_comm );
        }
        MPI_Comm_free( &pm_comm );
    }
    int pm_failed = ( pm_status != METIS_OK ) ? 1 : 0;
    MPI_Allreduce( MPI_IN_PLACE, &pm_failed, 1, MPI_INT, MPI_MAX, _comm );
    if ( pm_failed )
        throw std::runtime_error(
            std::string( "TreePartitioner::partition_cells: ParMETIS_V3_" ) +
            ( adaptive && n_keep >= 2 ? "AdaptiveRepart" : "PartKway" ) +
            " did not return METIS_OK on at least one rank" );

    // Every rank gets the full key -> rank map its consumers read.
    std::vector<int> counts( _comm_size ), displs( _comm_size );
    for ( int r = 0; r < _comm_size; ++r )
    {
        displs[r] = static_cast<int>( vtxdist[r] );
        counts[r] = static_cast<int>( vtxdist[r + 1] - vtxdist[r] );
    }
    std::vector<int> local_parts( part.begin(), part.end() );
    std::vector<int> all_parts( n );
    MPI_Allgatherv( local_parts.data(), static_cast<int>( n_local ), MPI_INT,
                    all_parts.data(), counts.data(), displs.data(), MPI_INT,
                    _comm );
    for ( int g = 0; g < n; ++g )
        if ( all_parts[g] < 0 || all_parts[g] >= _comm_size )
            throw std::runtime_error(
                "TreePartitioner::partition_cells: ParMETIS returned part " +
                std::to_string( all_parts[g] ) + " outside [0, " +
                std::to_string( _comm_size ) + ")" );

    for ( int g = 0; g < n; ++g )
        result[cells[vtx[m_of_gid[g]]].key] = all_parts[g];

#if defined( CANOPY_ENABLE_PROFILING )
    if ( _rank == 0 )
    {
        // Per-constraint max/mean, the non-shared parent-child cut fraction
        // against the vote rule on the same leaf assignment, and the number
        // of ranks owning no leaf.
        std::vector<double> load( static_cast<size_t>( ncon ) * _comm_size,
                                  0.0 );
        std::vector<int> leaves_per_rank( _comm_size, 0 );
        for ( int m = 0; m < n; ++m )
        {
            const CellInfo& c = cells[vtx[m]];
            const int r = all_parts[gid_of_m[m]];
            if ( c.is_leaf )
            {
                load[r] += c.global_count;
                leaves_per_rank[r]++;
            }
            for ( int b = 0; b < num_bands; ++b )
                if ( c.depth >= _bands[b].depth_lo &&
                     c.depth <= _bands[b].depth_hi )
                    load[static_cast<size_t>( 1 + b ) * _comm_size + r] += 1.0;
        }
        std::string imb, band_str;
        for ( int k = 0; k < ncon; ++k )
        {
            double mx = 0.0, sum = 0.0;
            for ( int r = 0; r < _comm_size; ++r )
            {
                const double v = load[static_cast<size_t>( k ) * _comm_size + r];
                mx = std::max( mx, v );
                sum += v;
            }
            char buf[32];
            std::snprintf( buf, sizeof( buf ), "%s%.4f", k ? "," : "",
                           sum > 0.0 ? mx * _comm_size / sum : 0.0 );
            imb += buf;
        }
        for ( int b = 0; b < num_bands; ++b )
            band_str += ( b ? "," : "" ) + std::to_string( _bands[b].depth_lo ) +
                        "-" + std::to_string( _bands[b].depth_hi );
        const auto vote = vote_internal_owners( cells, result );
        long long pairs = 0, cut = 0, cut_vote = 0;
        for ( int m = 0; m < n; ++m )
        {
            const CellInfo& c = cells[vtx[m]];
            if ( c.depth == 0 )
                continue;
            auto pit = result.find( parent_key( c.key ) );
            if ( pit == result.end() )
                continue;
            pairs++;
            const int child_owner = result.at( c.key );
            if ( child_owner != pit->second )
                cut++;
            const int vc = c.is_leaf ? child_owner : vote.at( c.key );
            if ( vc != vote.at( pit->first ) )
                cut_vote++;
        }
        int no_leaf = 0;
        for ( int r = 0; r < _comm_size; ++r )
            no_leaf += ( leaves_per_rank[r] == 0 );
        std::fprintf(
            stderr,
            "[Canopy diag] partition method=%s np=%d nverts=%d B=%d bands=%s "
            "imbalance=%s pc_pairs=%lld pc_cut=%.4f pc_cut_vote=%.4f "
            "ranks_without_leaf=%d\n",
            adaptive && n_keep >= 2 ? "AdaptiveRepart" : "PartKway",
            _comm_size, n, num_bands, band_str.empty() ? "none" : band_str.c_str(),
            imb.c_str(), pairs, pairs ? double( cut ) / pairs : 0.0,
            pairs ? double( cut_vote ) / pairs : 0.0, no_leaf );
    }
#endif

    return result;
}

template <class MemorySpace, class ExecutionSpace>
std::unordered_map<MortonKey, int>
TreePartitioner<MemorySpace, ExecutionSpace>::vote_internal_owners(
    const std::vector<CellInfo>& cells,
    const std::unordered_map<MortonKey, int>& owners ) const
{
    std::unordered_map<MortonKey, std::unordered_map<int, int64_t>> vote_map;
    for ( const auto& c : cells )
    {
        if ( !c.is_leaf )
            continue;
        auto owner_it = owners.find( c.key );
        if ( owner_it == owners.end() )
            continue;

        const int leaf_rank = owner_it->second;
        const int64_t count = static_cast<int64_t>( c.global_count );
        MortonKey parent = parent_key( c.key );
        while ( parent >= ROOT_KEY )
        {
            if ( key_depth( parent ) <= _replication_depth )
                break;
            vote_map[parent][leaf_rank] += count;
            parent = parent_key( parent );
        }
    }

    std::unordered_map<MortonKey, int> result;
    result.reserve( vote_map.size() );
    for ( const auto& [key, votes] : vote_map )
    {
        // Lowest rank wins on equal votes: unordered_map iteration order is
        // not guaranteed identical across processes, so without a total
        // order two ranks could pick different owners for the same cell.
        int best_rank = std::numeric_limits<int>::max();
        int64_t best_count = -1;
        for ( const auto& [r, cnt] : votes )
        {
            if ( cnt > best_count || ( cnt == best_count && r < best_rank ) )
            {
                best_count = cnt;
                best_rank = r;
            }
        }
        result[key] = best_rank;
    }
    return result;
}

template <class MemorySpace, class ExecutionSpace>
void TreePartitioner<MemorySpace, ExecutionSpace>::derive_internal_ownership(
    const std::vector<CellInfo>& cells,
    const std::unordered_map<MortonKey, int>& owners )
{
    const auto vote = vote_internal_owners( cells, owners );

    _ownership.clear();
    _ownership.reserve( cells.size() );
    _cell_owner_map.clear();
    _cell_owner_map.reserve( cells.size() );

    for ( const auto& c : cells )
    {
        CellOwnership co;
        co.key = c.key;

        if ( c.is_leaf )
        {
            auto it = owners.find( c.key );
            if ( it == owners.end() )
                throw std::runtime_error(
                    "TreePartitioner::derive_internal_ownership: leaf " +
                    std::to_string( c.key ) + " has no owner" );
            co.owner_rank = it->second;
        }
        else if ( c.depth <= _replication_depth )
        {
            co.owner_rank = OWNER_SHARED;
        }
        else if ( auto it = owners.find( c.key ); it != owners.end() )
        {
            co.owner_rank = it->second;
        }
        else
        {
            // No leaf below votes only for an empty internal cell.
            auto vote_it = vote.find( c.key );
            co.owner_rank = ( vote_it != vote.end() ) ? vote_it->second : 0;
        }

        _ownership.push_back( co );
        _cell_owner_map[co.key] = co.owner_rank;
    }
}

template <class MemorySpace, class ExecutionSpace>
template <class AoSoAType>
int TreePartitioner<MemorySpace, ExecutionSpace>::migrate_particles(
    const TreeBuilder<MemorySpace, ExecutionSpace>& tree_builder,
    AoSoAType& particles, int num_local_particles_before )
{
    // ----------------------------------------------------------------------
    // Coalesced, registration-bounded particle migration.
    //
    // Replaces the former Cabana::Distributor / Cabana::migrate path, whose
    // send/recv staging buffers lived inside Cabana and handed O(peers)
    // distinct device pointers to GPU-aware MPI per migrate — exhausting the
    // CXI NIC registration cache (GTL dreg_evict NO_SPACE) during a many-way
    // Rebalance at scale.
    //
    // Instead we pack outgoing AoSoA tuples into per-peer subviews of ONE
    // persistent registered send region and post one MPI_Isend per peer; the
    // matching MPI_Irecv land in ONE persistent registered recv region. Peak
    // concurrent registrations are therefore O(1) per direction regardless of
    // peer count or rollup state — the same structure coalesced_view_exchange
    // and the P2P ghost gather already use.
    //
    // The MPI element type is one whole tuple (MPI_Type_contiguous over
    // sizeof(tuple_type) bytes) and the message count is the tuple count, so a
    // single peer's payload can exceed 2 GiB without overflowing MPI's signed
    // int count — this sidesteps the upstream 32-bit Distributor byte-count
    // overflow and removes the "Patched Cabana required" constraint here.
    //
    // Migrate semantics preserved: destinations come from host particle_keys
    // looked up in _cell_owner_map (kept on this rank for OWNER_SHARED or
    // unresolved keys); the AoSoA is resized to (kept + received); num_sent is
    // the count of particles destined for a different rank. The within-AoSoA
    // order after migration is unspecified — every caller in Canopy_Solver.hpp
    // immediately rebuilds keys and re-sorts by leaf, so no order is required
    // downstream.
    // ----------------------------------------------------------------------
    using tuple_type = typename AoSoAType::tuple_type;
    using umtuple_view = Kokkos::View<tuple_type*, memory_space,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    // Copy particle keys to host to compute each particle's destination rank.
    auto particle_keys = tree_builder.particle_keys();
    auto h_keys = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                       particle_keys );

    // dest[i] = owning rank of particle i (== _rank means keep local).
    std::vector<int> dest( num_local_particles_before, _rank );
    std::vector<int> send_counts( _comm_size, 0 );
    int num_sent = 0;
    for ( int i = 0; i < num_local_particles_before; i++ )
    {
        auto it = _cell_owner_map.find( h_keys( i ) );
        // OWNER_SHARED (-1) or unresolved key -> keep on current rank.
        if ( it != _cell_owner_map.end() && it->second >= 0 )
        {
            const int owner = it->second;
            dest[i] = owner;
            if ( owner != _rank )
            {
                send_counts[owner]++;
                num_sent++;
            }
        }
    }

    // Discover incoming counts: every rank learns how many tuples each source
    // will send it. One Alltoall of comm_size ints — bounded and collective.
    std::vector<int> recv_counts( _comm_size, 0 );
    MPI_Alltoall( send_counts.data(), 1, MPI_INT, recv_counts.data(), 1,
                  MPI_INT, _comm );

    // Ordered peer lists (ascending rank) with tuple offsets into the pools.
    std::vector<int> send_peers, send_peer_off, send_peer_n;
    std::vector<int> recv_peers, recv_peer_off, recv_peer_n;
    size_t total_send = 0, total_recv = 0;
    for ( int r = 0; r < _comm_size; r++ )
    {
        if ( r != _rank && send_counts[r] > 0 )
        {
            send_peers.push_back( r );
            send_peer_off.push_back( static_cast<int>( total_send ) );
            send_peer_n.push_back( send_counts[r] );
            total_send += static_cast<size_t>( send_counts[r] );
        }
        if ( r != _rank && recv_counts[r] > 0 )
        {
            recv_peers.push_back( r );
            recv_peer_off.push_back( static_cast<int>( total_recv ) );
            recv_peer_n.push_back( recv_counts[r] );
            total_recv += static_cast<size_t>( recv_counts[r] );
        }
    }

    // Fast path: this rank neither sends nor receives. Everything stays in
    // place in its current order; no rebuild needed.
    if ( total_send == 0 && total_recv == 0 )
    {
        _num_local_after = num_local_particles_before;
        return num_sent;
    }

    // ---- Build the packed send-index list (host), grouped by peer in the
    // same ascending-rank order used for the pool offsets. ----
    _migrate_send_idx_pool.reserve( total_send );
    auto send_idx = _migrate_send_idx_pool.subview( 0, total_send );
    if ( total_send > 0 )
    {
        std::unordered_map<int, int> peer_slot; // rank -> index in send_peers
        for ( int q = 0; q < static_cast<int>( send_peers.size() ); q++ )
            peer_slot[send_peers[q]] = q;
        std::vector<int> cursor = send_peer_off; // running write pos per peer
        auto h_send_idx = Kokkos::create_mirror_view( send_idx );
        for ( int i = 0; i < num_local_particles_before; i++ )
        {
            if ( dest[i] != _rank )
                h_send_idx( cursor[peer_slot[dest[i]]]++ ) = i;
        }
        Kokkos::deep_copy( send_idx, h_send_idx );
    }

    // ---- Size the registered tuple regions up front so every subview handed
    // out below has a stable base address. ----
    _migrate_send_pool.reserve( total_send * sizeof( tuple_type ) );
    _migrate_recv_pool.reserve( total_recv * sizeof( tuple_type ) );
    umtuple_view send_buf(
        reinterpret_cast<tuple_type*>( _migrate_send_pool.data() ),
        total_send );
    umtuple_view recv_buf(
        reinterpret_cast<tuple_type*>( _migrate_recv_pool.data() ),
        total_recv );

    // ---- Pack outgoing tuples on device into the one send region. ----
    if ( total_send > 0 )
    {
        auto src = particles;
        auto idx = send_idx;
        auto out = send_buf;
        Kokkos::parallel_for(
            "migrate_pack",
            Kokkos::RangePolicy<execution_space>(
                0, static_cast<int>( total_send ) ),
            KOKKOS_LAMBDA( const int i ) {
                out( i ) = src.getTuple( idx( i ) );
            } );
        Kokkos::fence();
    }

    // ---- Exchange. One MPI element = one whole tuple; count = tuple count,
    // so a >2 GiB per-peer payload never overflows the signed int count. ----
    MPI_Datatype tuple_dtype;
    MPI_Type_contiguous( static_cast<int>( sizeof( tuple_type ) ), MPI_BYTE,
                         &tuple_dtype );
    MPI_Type_commit( &tuple_dtype );

    std::vector<MPI_Request> recv_reqs;
    recv_reqs.reserve( recv_peers.size() );
    for ( size_t q = 0; q < recv_peers.size(); q++ )
    {
        MPI_Request req;
        MPI_Irecv( recv_buf.data() + recv_peer_off[q], recv_peer_n[q],
                   tuple_dtype, recv_peers[q], /*tag=*/0, _comm, &req );
        recv_reqs.push_back( req );
    }
    std::vector<MPI_Request> send_reqs;
    send_reqs.reserve( send_peers.size() );
    for ( size_t q = 0; q < send_peers.size(); q++ )
    {
        MPI_Request req;
        MPI_Isend( send_buf.data() + send_peer_off[q], send_peer_n[q],
                   tuple_dtype, send_peers[q], /*tag=*/0, _comm, &req );
        send_reqs.push_back( req );
    }
    if ( !recv_reqs.empty() )
        MPI_Waitall( static_cast<int>( recv_reqs.size() ), recv_reqs.data(),
                     MPI_STATUSES_IGNORE );
    if ( !send_reqs.empty() )
        MPI_Waitall( static_cast<int>( send_reqs.size() ), send_reqs.data(),
                     MPI_STATUSES_IGNORE );
    MPI_Type_free( &tuple_dtype );

    // ---- Rebuild the AoSoA as (kept particles) ++ (received particles). ----
    int num_kept = 0;
    for ( int i = 0; i < num_local_particles_before; i++ )
        if ( dest[i] == _rank )
            num_kept++;

    Kokkos::View<int*, memory_space> keep_idx(
        Kokkos::view_alloc( Kokkos::WithoutInitializing, "migrate_keep_idx" ),
        static_cast<size_t>( num_kept ) );
    {
        auto h_keep = Kokkos::create_mirror_view( keep_idx );
        int k = 0;
        for ( int i = 0; i < num_local_particles_before; i++ )
            if ( dest[i] == _rank )
                h_keep( k++ ) = i;
        Kokkos::deep_copy( keep_idx, h_keep );
    }

    const size_t new_size = static_cast<size_t>( num_kept ) + total_recv;
    AoSoAType migrated( "migrated_particles", new_size );

    if ( num_kept > 0 )
    {
        auto in = particles;
        auto out = migrated;
        auto idx = keep_idx;
        Kokkos::parallel_for(
            "migrate_keep", Kokkos::RangePolicy<execution_space>( 0, num_kept ),
            KOKKOS_LAMBDA( const int i ) {
                out.setTuple( i, in.getTuple( idx( i ) ) );
            } );
    }
    if ( total_recv > 0 )
    {
        auto out = migrated;
        auto buf = recv_buf;
        const int base = num_kept;
        Kokkos::parallel_for(
            "migrate_unpack",
            Kokkos::RangePolicy<execution_space>(
                0, static_cast<int>( total_recv ) ),
            KOKKOS_LAMBDA( const int j ) {
                out.setTuple( base + j, buf( j ) );
            } );
    }
    Kokkos::fence();

    particles = migrated;

    _num_local_after = static_cast<int>( particles.size() );

    return num_sent;
}

template <class MemorySpace, class ExecutionSpace>
template <class AoSoAType>
void TreePartitioner<MemorySpace, ExecutionSpace>::partition(
    const TreeBuilder<MemorySpace, ExecutionSpace>& tree_builder,
    AoSoAType& particles, int num_local_particles_before )
{
    const auto& cells = tree_builder.cells();

    auto cell_owners = partition_cells( cells, false );
    derive_internal_ownership( cells, cell_owners );
    _cached_cell_owners = std::move( cell_owners );

    migrate_particles( tree_builder, particles, num_local_particles_before );
}

template <class MemorySpace, class ExecutionSpace>
template <class AoSoAType>
RedistributeResult TreePartitioner<MemorySpace, ExecutionSpace>::redistribute(
    const TreeBuilder<MemorySpace, ExecutionSpace>& tree_builder,
    AoSoAType& particles, int num_local_particles_before )
{
    RedistributeResult result;

    int num_before = num_local_particles_before;
    result.particles_sent = migrate_particles( tree_builder, particles,
                                               num_local_particles_before );

    result.num_local_after = _num_local_after;

    // particles_received = new count - (old count - sent)
    // i.e., the particles we have now minus the ones we kept
    result.particles_received =
        _num_local_after - ( num_before - result.particles_sent );

    return result;
}

template <class MemorySpace, class ExecutionSpace>
template <class AoSoAType>
void TreePartitioner<MemorySpace, ExecutionSpace>::repartition(
    const TreeBuilder<MemorySpace, ExecutionSpace>& tree_builder,
    AoSoAType& particles, int num_local_particles_before )
{
    const auto& cells = tree_builder.cells();

    // Starts from the current owners, so _cell_owner_map is read before
    // derive_internal_ownership replaces it.
    auto cell_owners = partition_cells( cells, true );
    derive_internal_ownership( cells, cell_owners );
    _cached_cell_owners = std::move( cell_owners );

    migrate_particles( tree_builder, particles, num_local_particles_before );
}

// --------------------------------------------------------------------------
// refresh_ownership_for_current_tree
//
// Re-populate _cell_owner_map against the current tree using the cached
// cell assignment from the most recent partition_cells() call. Particles
// are NOT migrated; ParMETIS is NOT re-run.
//
// For cells present in the cached assignment: reuse the cached owner.
// For leaves NOT in it (e.g., a coarsened tree where a former-internal cell
// is now a leaf): vote based on local particle counts via a single Allgather
// of per-leaf vote vectors. Internal cells NOT in it take the vote rule.
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace>
template <class AoSoAType>
void TreePartitioner<MemorySpace, ExecutionSpace>::
    refresh_ownership_for_current_tree(
        const TreeBuilder<MemorySpace, ExecutionSpace>& tree_builder,
        AoSoAType& particles )
{
    const auto& cells = tree_builder.cells();
    (void)particles; // current implementation only needs particle_keys

    // Collect leaves that need a new owner (not in cache).
    std::vector<MortonKey> new_leaf_keys;
    std::unordered_map<MortonKey, int> new_leaf_idx;
    for ( const auto& c : cells )
    {
        if ( !c.is_leaf )
            continue;
        if ( _cached_cell_owners.find( c.key ) != _cached_cell_owners.end() )
            continue;
        new_leaf_idx[c.key] = static_cast<int>( new_leaf_keys.size() );
        new_leaf_keys.push_back( c.key );
    }
    const int n_new = static_cast<int>( new_leaf_keys.size() );

    // Start from the cache and add entries for new leaves. Owner of a new
    // leaf = rank with the most local particles in that leaf, broken by
    // lowest rank on ties.
    std::unordered_map<MortonKey, int> owners = _cached_cell_owners;

    if ( n_new > 0 )
    {
        // Tally local particles per new leaf.
        auto particle_keys = tree_builder.particle_keys();
        auto h_keys = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), particle_keys );
        const int N = static_cast<int>( h_keys.extent( 0 ) );

        std::vector<long long> local_counts( n_new, 0 );
        for ( int i = 0; i < N; i++ )
        {
            auto it = new_leaf_idx.find( h_keys( i ) );
            if ( it != new_leaf_idx.end() )
                local_counts[it->second]++;
        }

        // Allgather every rank's per-leaf counts so each rank can pick
        // the same owner deterministically.
        std::vector<long long> all_counts(
            static_cast<size_t>( n_new ) * _comm_size, 0 );
        MPI_Allgather( local_counts.data(), n_new, MPI_LONG_LONG,
                       all_counts.data(), n_new, MPI_LONG_LONG, _comm );

        for ( int i = 0; i < n_new; i++ )
        {
            int best_rank = 0;
            long long best_count = -1;
            for ( int r = 0; r < _comm_size; r++ )
            {
                long long c =
                    all_counts[static_cast<size_t>( r ) * n_new + i];
                if ( c > best_count )
                {
                    best_count = c;
                    best_rank = r;
                }
            }
            owners[new_leaf_keys[i]] = best_rank;
        }

        // Update the cache so subsequent refreshes are cheap.
        for ( int i = 0; i < n_new; i++ )
            _cached_cell_owners[new_leaf_keys[i]] = owners[new_leaf_keys[i]];
    }

    derive_internal_ownership( cells, owners );

#if defined( CANOPY_ENABLE_PROFILING )
    if ( _rank == 0 )
    {
        // Non-shared cells of the current tree that are absent from the
        // partitioned assignment and so fell back to a vote (R9).
        int non_shared = 0;
        int fallback = n_new;
        for ( const auto& c : cells )
        {
            if ( !c.is_leaf && c.depth <= _replication_depth )
                continue;
            non_shared++;
            if ( !c.is_leaf &&
                 _cached_cell_owners.find( c.key ) == _cached_cell_owners.end() )
                fallback++;
        }
        std::fprintf( stderr,
                      "[Canopy diag] refresh_ownership fallback=%d/%d\n",
                      fallback, non_shared );
    }
#endif
}

// --------------------------------------------------------------------------
// sort_particles_by_leaf
//
// Approach:
//   1. On host, read tree_builder.particle_keys() and map each particle's
//      key to a cell index.
//   2. Build the sort permutation that groups particles by cell index.
//   3. Apply the permutation to the AoSoA via Cabana::permute (which uses
//      an Cabana::Distributor pattern but locally).
//   4. Build leaf_particle_offsets as a prefix sum of per-cell counts.
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace>
template <class AoSoAType>
void TreePartitioner<MemorySpace, ExecutionSpace>::sort_particles_by_leaf(
    TreeBuilder<MemorySpace, ExecutionSpace>& tree_builder,
    AoSoAType& particles )
{
    const auto& cells = tree_builder.cells();
    const int num_cells = static_cast<int>( cells.size() );
    const int N = static_cast<int>( particles.size() );

    // Device counting sort, avoiding Cabana::sortByKey/binByKey (both route
    // through Kokkos::BinSort, whose 1e8-wide atomic bin-count scatter faults
    // on the MI300A unified-memory path at 4e8 particles — see MI300A
    // investigation bug 4). Uses bounded Kokkos::atomic_add into num_cells-wide
    // counters (same pattern as LaplaceKernel M2M/L2L), NOT BinSort's pattern.
    //
    // Steps (all parallel):
    //   1. Build sorted (key,cell_idx) lookup on host once (num_cells small),
    //      deep_copy to device. Binary search per particle → cell_of_d.
    //   2. Atomic per-cell count → cell_counts_d; exclusive scan → offsets_d.
    //   3. Atomic scatter into perm using offsets_d + per-cell running counter.
    //   4. Fill _particle_leaf_cell_idx on device from offsets_d.
    // Within-cell order in perm is unspecified (atomic_fetch_add ordering);
    // downstream consumers (P2P leaf iteration, P2M sums) only need the
    // [offsets(c), offsets(c+1)) grouping, not a specific within-cell order.

    Kokkos::View<MortonKey*, memory_space> sorted_keys_d(
        Kokkos::view_alloc( Kokkos::WithoutInitializing, "sorted_cell_keys" ),
        static_cast<size_t>( num_cells ) );
    Kokkos::View<int*, memory_space> sorted_cell_idx_d(
        Kokkos::view_alloc( Kokkos::WithoutInitializing, "sorted_cell_idx" ),
        static_cast<size_t>( num_cells ) );
    {
        auto sk_h = Kokkos::create_mirror_view( sorted_keys_d );
        auto sci_h = Kokkos::create_mirror_view( sorted_cell_idx_d );
        std::vector<std::pair<MortonKey, int>> pairs( num_cells );
        for ( int i = 0; i < num_cells; i++ )
            pairs[i] = { cells[i].key, i };
        std::sort( pairs.begin(), pairs.end(),
                   []( const std::pair<MortonKey, int>& a,
                       const std::pair<MortonKey, int>& b ) {
                       return a.first < b.first;
                   } );
        for ( int i = 0; i < num_cells; i++ )
        {
            sk_h( i ) = pairs[i].first;
            sci_h( i ) = pairs[i].second;
        }
        Kokkos::deep_copy( sorted_keys_d, sk_h );
        Kokkos::deep_copy( sorted_cell_idx_d, sci_h );
    }

    auto particle_keys = tree_builder.particle_keys();
    Kokkos::View<int*, memory_space> cell_of_d(
        Kokkos::view_alloc( Kokkos::WithoutInitializing, "cell_of" ),
        static_cast<size_t>( N ) );
    {
        auto sk = sorted_keys_d;
        auto sci = sorted_cell_idx_d;
        auto co = cell_of_d;
        auto pk = particle_keys;
        const int nc = num_cells;
        Kokkos::parallel_for(
            "sort_classify",
            Kokkos::RangePolicy<execution_space>( 0, N ),
            KOKKOS_LAMBDA( const int i ) {
                MortonKey k = pk( i );
                int lo = 0;
                int hi = nc;
                while ( lo < hi )
                {
                    int mid = ( lo + hi ) >> 1;
                    if ( sk( mid ) < k )
                        lo = mid + 1;
                    else
                        hi = mid;
                }
                co( i ) =
                    ( lo < nc && sk( lo ) == k ) ? sci( lo ) : 0;
            } );
    }

    Kokkos::View<int*, memory_space> cell_counts_d(
        "cell_counts", static_cast<size_t>( num_cells ) );
    {
        auto cc = cell_counts_d;
        auto co = cell_of_d;
        Kokkos::parallel_for(
            "sort_count",
            Kokkos::RangePolicy<execution_space>( 0, N ),
            KOKKOS_LAMBDA( const int i ) {
                Kokkos::atomic_add( &cc( co( i ) ), 1 );
            } );
    }

    Kokkos::View<int*, memory_space> offsets_d(
        Kokkos::view_alloc( Kokkos::WithoutInitializing,
                            "leaf_particle_offsets" ),
        static_cast<size_t>( num_cells + 1 ) );
    {
        auto cc = cell_counts_d;
        auto off = offsets_d;
        const int nc = num_cells;
        Kokkos::parallel_scan(
            "sort_prefix",
            Kokkos::RangePolicy<execution_space>( 0, num_cells + 1 ),
            KOKKOS_LAMBDA( const int c, int& upd, const bool final ) {
                if ( final )
                    off( c ) = upd;
                if ( c < nc )
                    upd += cc( c );
            } );
    }

    Kokkos::View<int*, memory_space> perm(
        Kokkos::view_alloc( Kokkos::WithoutInitializing, "sort_perm" ),
        static_cast<size_t>( N ) );
    Kokkos::View<int*, memory_space> running_d(
        "sort_running", static_cast<size_t>( num_cells ) );
    {
        auto pm = perm;
        auto co = cell_of_d;
        auto off = offsets_d;
        auto rn = running_d;
        Kokkos::parallel_for(
            "sort_scatter_perm",
            Kokkos::RangePolicy<execution_space>( 0, N ),
            KOKKOS_LAMBDA( const int i ) {
                const int c = co( i );
                const int pos =
                    off( c ) + Kokkos::atomic_fetch_add( &rn( c ), 1 );
                pm( pos ) = i;
            } );
    }

    // Manual device gather: scratch(i) = particles[perm(i)], then copy back.
    {
        Kokkos::View<typename AoSoAType::tuple_type*, memory_space> scratch(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "sort_scratch" ),
            static_cast<size_t>( N ) );
        auto aosoa = particles;
        auto perm_v = perm;
        Kokkos::parallel_for(
            "sort_gather",
            Kokkos::RangePolicy<execution_space>( 0, N ),
            KOKKOS_LAMBDA( const int i ) {
                scratch( i ) = aosoa.getTuple( perm_v( i ) );
            } );
        Kokkos::fence();
        Kokkos::parallel_for(
            "sort_scatter_back",
            Kokkos::RangePolicy<execution_space>( 0, N ),
            KOKKOS_LAMBDA( const int i ) {
                aosoa.setTuple( i, scratch( i ) );
            } );
        Kokkos::fence();
    }

    // Keep builder.particle_keys() in sync with the new AoSoA order. The
    // tree topology did not change across the sort, so a full builder.build()
    // would re-derive the same keys in this new order at a much higher cost.
    tree_builder.apply_particle_permutation( perm );

    // _leaf_particle_offsets was already built on device as offsets_d.
    _leaf_particle_offsets = offsets_d;

    // Fill _particle_leaf_cell_idx on device: thread c writes the constant c
    // into slots [offsets(c), offsets(c+1)).
    _particle_leaf_cell_idx = Kokkos::View<int*, memory_space>(
        Kokkos::view_alloc( Kokkos::WithoutInitializing,
                            "particle_leaf_cell_idx" ),
        static_cast<size_t>( N ) );
    {
        auto pli = _particle_leaf_cell_idx;
        auto off = offsets_d;
        Kokkos::parallel_for(
            "sort_fill_cell_idx",
            Kokkos::RangePolicy<execution_space>( 0, num_cells ),
            KOKKOS_LAMBDA( const int c ) {
                const int lo = off( c );
                const int hi = off( c + 1 );
                for ( int k = lo; k < hi; k++ )
                    pli( k ) = c;
            } );
    }
}

} // end namespace Canopy

#endif // CANOPY_TREE_PARTITIONER_HPP