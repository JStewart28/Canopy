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

#include <Zoltan2_BasicVectorAdapter.hpp>
#include <Zoltan2_PartitioningProblem.hpp>

#include <Teuchos_Comm.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_DefaultSerialComm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Tpetra_Map.hpp>

#include <mpi.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <typeinfo>
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
// leaf cells across MPI ranks using Zoltan2 RCB, derives internal cell
// ownership using the replicated-coarse-layers strategy, migrates
// particles to their owning ranks, and sorts particles by leaf cell index.
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
    // imbalance_tolerance: Zoltan2 imbalance tolerance (e.g., 0.05 means
    //     allow 5% imbalance)
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

    // Cached leaf assignment from the most recent partition_leaves() call.
    // Used by refresh_ownership_for_current_tree() to avoid re-running the
    // non-deterministic Zoltan2 multijagged partitioner (which would emit a
    // different assignment and trigger a multi-GB second migrate that
    // overflows MPI's signed int count at scale).
    std::unordered_map<MortonKey, int> _cached_leaf_owners;

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
    // Internal: partition leaf cells using Zoltan2 RCB
    // Returns a map: leaf MortonKey -> owning rank
    std::unordered_map<MortonKey, int>
    partition_leaves( const std::vector<CellInfo>& cells );

    // -----------------------------------------------------------------------
    // derive_internal_ownership
    // -----------------------------------------------------------------------
    void derive_internal_ownership(
        const std::vector<CellInfo>& cells,
        const std::unordered_map<MortonKey, int>& leaf_owners );

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
    // re-running Zoltan2 and WITHOUT migrating particles. Uses the leaf
    // assignment cached from the most recent partition_leaves() call. For
    // any leaf in the current tree that wasn't a leaf in the cached
    // assignment, votes on owner based on which rank holds the most
    // local particles in that leaf (single Allreduce on a per-leaf vote
    // table). This keeps cell_owner_map consistent with the final cells
    // passed to comm_plan.build, fixing the phantom-send M2M plan that
    // arises when the post-migration build produces a different tree than
    // the pre-partition build.
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
TreePartitioner<MemorySpace, ExecutionSpace>::partition_leaves(
    const std::vector<CellInfo>& cells )
{
    // Collect leaf cells
    std::vector<MortonKey> leaf_keys;
    std::vector<double> leaf_x, leaf_y, leaf_z;
    std::vector<double> leaf_weights;

    for ( const auto& c : cells )
    {
        if ( c.is_leaf )
        {
            leaf_keys.push_back( c.key );
            leaf_x.push_back( c.center[0] );
            leaf_y.push_back( c.center[1] );
            leaf_z.push_back( c.center[2] );
            leaf_weights.push_back( static_cast<double>( c.global_count ) );
        }
    }

    int num_leaves = static_cast<int>( leaf_keys.size() );

    // If only one rank, all leaves are owned by rank 0
    // and can skip partitioning.
    if ( _comm_size == 1 )
    {
        std::unordered_map<MortonKey, int> result;
        for ( int i = 0; i < num_leaves; i++ )
            result[leaf_keys[i]] = 0;
        return result;
    }

    // ---------------------------------------------------------------
    // Build Zoltan2 adapter
    //
    // Every rank has the identical set of leaves (the tree structure is
    // replciated globally), so run Zoltan2 with the full leaf set on every
    // rank. Zoltan2 RCB is deterministic given the same input, so all ranks
    // will produce the same assignment.
    //
    // Use global IDs = leaf index (0..num_leaves-1), which are the
    // same on every rank since _cells from TreeBuilder is identical on all
    // ranks.
    // ---------------------------------------------------------------

    // Create Zoltan2 adapter
    // BasicVectorAdapter needs:
    //   numIds, globalIds, coords, weights
    using adapter_t = Zoltan2::BasicVectorAdapter<Tpetra::Map<int, int64_t>>;
    // Must also use Zoltan types
    using glbl_id_t = typename adapter_t::gno_t;
    using scalar_t = typename adapter_t::scalar_t;
    using longint_t = typename adapter_t::lno_t;

    // Zoltan2 needs coordinates as an array of pointers, one per dim
    const scalar_t* coords[3] = { leaf_x.data(), leaf_y.data(), leaf_z.data() };
    // const int strides[3] = { 1, 1, 1 };

    // Global IDs for the leaves
    std::vector<glbl_id_t> global_ids( num_leaves );
    for ( int i = 0; i < num_leaves; i++ )
        global_ids[i] = static_cast<glbl_id_t>( i );

    // Build a Teuchos communicator for Zoltan2.
    // Use SerialComm (not MpiComm(MPI_COMM_SELF)) so Zoltan2's internal
    // sends/receives never enter MPICH. Every rank runs the same
    // deterministic partition on identical data, so no real MPI is needed;
    // routing self-sends through Cray MPICH's CMA single-copy path was
    // triggering process_vm_readv: Bad address on AMD/HIP builds.
    auto teuchos_comm =
        Teuchos::rcp( new Teuchos::SerialComm<int>() );

    const glbl_id_t* ids_ptr = global_ids.data();
    const scalar_t* x_ptr = leaf_x.data();
    const scalar_t* y_ptr = leaf_y.data();
    const scalar_t* z_ptr = leaf_z.data();
    const scalar_t* w_ptr = leaf_weights.data();

    adapter_t adapter( static_cast<longint_t>( num_leaves ), ids_ptr, x_ptr,
                       y_ptr, z_ptr, 1, 1, 1, // strides for x, y, z
                       true,                  // use weights
                       w_ptr,
                       1 ); // weight stride

    // Configure Zoltan2
    Teuchos::ParameterList params;
    params.set( "algorithm", "multijagged" );
    params.set( "num_global_parts", _comm_size );
    params.set( "imbalance_tolerance", _imbalance_tolerance );
    params.set( "debug_level", "no_status" );

    // Need these lines to disable Zoltan-level
    // status printouts
    Teuchos::ParameterList zoltanParams;
    zoltanParams.set( "DEBUG_LEVEL", "0" );
    params.set( "zoltan_parameters", zoltanParams );

    // Solve on rank 0 only, then broadcast. We cannot use ther deterministic "rcb"
    // algorithm because it breaks on Tuolumne. The "multijagged" algorithm is
    // non-deterministic, so only rank 0 computes, then broadcasts.
    std::vector<int> parts_storage( num_leaves );
    if ( _rank == 0 )
    {
        Zoltan2::PartitioningProblem<adapter_t> problem( &adapter, &params,
                                                            teuchos_comm );
        problem.solve();
        const auto& solution = problem.getSolution();
        const int* parts_view = solution.getPartListView();
        for ( int i = 0; i < num_leaves; i++ )
            parts_storage[i] = parts_view[i];
    }
    MPI_Bcast( parts_storage.data(), num_leaves, MPI_INT, 0, _comm );
    const int* parts = parts_storage.data();

    std::unordered_map<MortonKey, int> result;
    result.reserve( num_leaves );
    for ( int i = 0; i < num_leaves; i++ )
        result[leaf_keys[i]] = parts[i];

    return result;
}

template <class MemorySpace, class ExecutionSpace>
void TreePartitioner<MemorySpace, ExecutionSpace>::derive_internal_ownership(
    const std::vector<CellInfo>& cells,
    const std::unordered_map<MortonKey, int>& leaf_owners )
{
    // Build a key -> CellInfo lookup
    std::unordered_map<MortonKey, const CellInfo*> cell_map;
    for ( const auto& c : cells )
        cell_map[c.key] = &c;

    std::unordered_map<MortonKey, std::unordered_map<int, int64_t>> vote_map;

    for ( const auto& c : cells )
    {
        // Must start at leaf cells
        if ( !c.is_leaf )
            continue;

        auto owner_it = leaf_owners.find( c.key );
        if ( owner_it == leaf_owners.end() )
            continue;

        int leaf_rank = owner_it->second;
        // Get particle count in this leaf cell
        int64_t count = static_cast<int64_t>( c.global_count );

        // Walk up from this leaf to the root, adding votes
        MortonKey parent = parent_key( c.key );
        while ( parent >= ROOT_KEY )
        {
            // Count how many particle each rank owns in parent cells
            vote_map[parent][leaf_rank] += count;

            if ( parent == ROOT_KEY )
                break;
            parent = parent_key( parent );
        }
    }

    // Phase 2: assign ownership
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
            // Leaf ownership was determined by Zoltan2
            auto it = leaf_owners.find( c.key );
            co.owner_rank = ( it != leaf_owners.end() ) ? it->second : 0;
        }
        else if ( c.depth <= _replication_depth )
        {
            // Coarse layer — replicated on all ranks
            co.owner_rank = OWNER_SHARED;
        }
        else
        {
            // Deep internal cell — pick the rank with the most
            // descendant particles
            auto vote_it = vote_map.find( c.key );
            if ( vote_it != vote_map.end() )
            {
                // Iterate unordered_map<int, int64_t> with a deterministic
                // tiebreaker (lowest rank wins on equal votes). Without the
                // tiebreaker, two ranks with identical vote_map contents can
                // still pick different "best_rank" because unordered_map
                // iteration order is not guaranteed identical across
                // processes — which silently makes cell ownership disagree
                // across ranks and causes the comm-plan asymmetry observed
                // at ~4e8 particles.
                int best_rank = std::numeric_limits<int>::max();
                int64_t best_count = -1;
                for ( const auto& [r, cnt] : vote_it->second )
                {
                    if ( cnt > best_count ||
                         ( cnt == best_count && r < best_rank ) )
                    {
                        best_count = cnt;
                        best_rank = r;
                    }
                }
                if ( best_count < 0 )
                    best_rank = 0;
                co.owner_rank = best_rank;
            }
            else
            {
                // No votes — empty internal cell, assign to rank 0
                co.owner_rank = 0;
            }
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

    // Step 1: Partition leaf cells via Zoltan2
    auto leaf_owners = partition_leaves( cells );

    // Step 2: Derive internal cell ownership
    derive_internal_ownership( cells, leaf_owners );

    // Cache leaf assignment for refresh_ownership_for_current_tree()
    _cached_leaf_owners = leaf_owners;

    // Step 3: Migrate particles
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

    // Step 1: Re-partition leaf cells via Zoltan2
    auto leaf_owners = partition_leaves( cells );

    // Step 2: Re-derive internal cell ownership
    derive_internal_ownership( cells, leaf_owners );

    // Cache leaf assignment for refresh_ownership_for_current_tree()
    _cached_leaf_owners = leaf_owners;

    // Step 3: Migrate particles to new owners
    migrate_particles( tree_builder, particles, num_local_particles_before );
}

// --------------------------------------------------------------------------
// refresh_ownership_for_current_tree
//
// Re-populate _cell_owner_map against the current tree using the cached
// leaf assignment from the most recent partition_leaves() call. Particles
// are NOT migrated; Zoltan2 is NOT re-run.
//
// For leaves present in the cached assignment: reuse the cached owner.
// For leaves NOT in the cached assignment (e.g., a coarsened tree where
// a former-internal cell is now a leaf): vote based on local particle
// counts via a single Allgather of per-leaf vote vectors.
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
        if ( _cached_leaf_owners.find( c.key ) != _cached_leaf_owners.end() )
            continue;
        new_leaf_idx[c.key] = static_cast<int>( new_leaf_keys.size() );
        new_leaf_keys.push_back( c.key );
    }
    const int n_new = static_cast<int>( new_leaf_keys.size() );

    // Build a new leaf_owners that starts from the cache and adds entries
    // for new leaves. Owner of a new leaf = rank with the most local
    // particles in that leaf, broken by lowest rank on ties.
    std::unordered_map<MortonKey, int> leaf_owners = _cached_leaf_owners;

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
            leaf_owners[new_leaf_keys[i]] = best_rank;
        }

        // Update the cache so subsequent refreshes are cheap.
        for ( int i = 0; i < n_new; i++ )
            _cached_leaf_owners[new_leaf_keys[i]] =
                leaf_owners[new_leaf_keys[i]];
    }

    // Re-derive internal ownership for the current cells using the
    // combined leaf_owners.
    derive_internal_ownership( cells, leaf_owners );
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