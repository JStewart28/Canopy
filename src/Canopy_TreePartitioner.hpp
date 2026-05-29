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
    // migrate_particles — Cabana::Distributor-based migration
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
    //   - builder.particle_keys() is now STALE (order does not match AoSoA).
    //     The caller should call builder.build() again if particle_keys are
    //     needed in their new order.
    // -----------------------------------------------------------------------
    template <class AoSoAType>
    void sort_particles_by_leaf(
        const TreeBuilder<MemorySpace, ExecutionSpace>& tree_builder,
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
    // Copy particle keys to host to build the destination array
    auto particle_keys = tree_builder.particle_keys();
    auto h_keys = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                       particle_keys );

    Kokkos::View<int*, memory_space> dest_ranks( "dest_ranks",
                                                 num_local_particles_before );
    auto h_dest = Kokkos::create_mirror_view( dest_ranks );

    int num_sent = 0;

    for ( int i = 0; i < num_local_particles_before; i++ )
    {
        MortonKey key = h_keys( i );
        auto it = _cell_owner_map.find( key );
        if ( it != _cell_owner_map.end() )
        {
            int owner = it->second;
            // If the leaf is OWNER_SHARED (-1, shouldn't happen for leaves,
            // but guard against it), keep particle on current rank
            if ( owner >= 0 )
            {
                h_dest( i ) = owner;
                if ( owner != _rank )
                    num_sent++;
            }
            else
            {
                h_dest( i ) = _rank;
            }
        }
        else
        {
            // Key not found — keep on current rank (safety fallback)
            h_dest( i ) = _rank;
        }
    }

    Kokkos::deep_copy( dest_ranks, h_dest );

    // DEBUG-ONLY (bug 3 diagnosis): the step-1 Rebuild's pre-migrate build is
    // healthy (4.2M cells, leaf<=512) but the post-migrate build collapses
    // (868k cells, one depth-19 leaf with ~3e8 particles) AND the global
    // position mean/stddev changes — impossible from a correct migrate, which
    // only redistributes. Hypothesis: Cabana::migrate's data payload byte
    // count overflows signed int when one peer receives multi-GB of particles,
    // truncating the transfer and delivering garbage positions (count stays
    // right). Test: (a) max particles to one peer * sizeof(tuple) vs INT_MAX;
    // (b) global position checksum immediately before vs after the migrate.
    auto dbg_pos_sum = [&]( const char* tag )
    {
        auto pos = Cabana::slice<0>( particles );
        const int n = static_cast<int>( particles.size() );
        double lsum[3] = { 0, 0, 0 };
        for ( int a = 0; a < 3; ++a )
        {
            double s = 0.0;
            Kokkos::parallel_reduce(
                "dbg_pos_sum",
                Kokkos::RangePolicy<execution_space>( 0, n ),
                KOKKOS_LAMBDA( int i, double& ls ) { ls += pos( i, a ); },
                s );
            lsum[a] = s;
        }
        Kokkos::fence();
        long long ln = n, gn = 0;
        double gsum[3] = { 0, 0, 0 };
        MPI_Allreduce( lsum, gsum, 3, MPI_DOUBLE, MPI_SUM, _comm );
        MPI_Allreduce( &ln, &gn, 1, MPI_LONG_LONG, MPI_SUM, _comm );
        if ( _rank == 0 && gn > 0 )
        {
            std::fprintf( stderr,
                "[Canopy DEBUG migrate] %s global_n=%lld pos_mean=(%.6g,%.6g,"
                "%.6g)\n",
                tag, gn, gsum[0] / gn, gsum[1] / gn, gsum[2] / gn );
            std::fflush( stderr );
        }
    };
    {
        std::vector<long long> per_peer( _comm_size, 0 );
        for ( int i = 0; i < num_local_particles_before; i++ )
            per_peer[h_dest( i )]++;
        long long local_max_peer = 0;
        for ( int r = 0; r < _comm_size; r++ )
            if ( r != _rank && per_peer[r] > local_max_peer )
                local_max_peer = per_peer[r];
        long long global_max_peer = 0;
        MPI_Reduce( &local_max_peer, &global_max_peer, 1, MPI_LONG_LONG,
                    MPI_MAX, 0, _comm );
        const long long tuple_bytes =
            static_cast<long long>( sizeof( typename AoSoAType::tuple_type ) );
        if ( _rank == 0 )
        {
            const long long int_max =
                static_cast<long long>( std::numeric_limits<int>::max() );
            const long long max_bytes = global_max_peer * tuple_bytes;
            std::fprintf( stderr,
                "[Canopy DEBUG migrate] max_particles_to_one_peer=%lld "
                "tuple_bytes=%lld => max_payload_bytes=%lld INT_MAX=%lld %s\n",
                global_max_peer, tuple_bytes, max_bytes, int_max,
                ( max_bytes > int_max )
                    ? "<<< EXCEEDS INT_MAX (migrate byte-count overflow)"
                    : "(within int range)" );
            std::fflush( stderr );
        }
    }
    dbg_pos_sum( "before migrate:" );

    // Use Cabana::Distributor to migrate particles
    Cabana::Distributor<memory_space> distributor( _comm, dest_ranks );
    Cabana::migrate( distributor, particles );

    dbg_pos_sum( "after  migrate:" );

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
    const TreeBuilder<MemorySpace, ExecutionSpace>& tree_builder,
    AoSoAType& particles )
{
    const auto& cells = tree_builder.cells();
    const int num_cells = static_cast<int>( cells.size() );
    const int N = static_cast<int>( particles.size() );

    // DEBUG-ONLY (bug 3 / step-2 seg fault localization): the maintenance flow
    // faults in sort_particles_by_leaf (build C's print never appears). The
    // GPU "write to read-only page" fault is an OOB write, most likely a
    // Cabana::permute byte-count overflow at ~5.6 GB (1e8 part x 56 B) — the
    // same INT_MAX class as the migrate bug, in a different code path. Bracket
    // each device op with a fence+flush marker so the next run pinpoints it.
    auto dbg_mark = [&]( const char* msg )
    {
        Kokkos::fence();
        MPI_Barrier( _comm );
        if ( _rank == 0 )
        {
            std::fprintf( stderr, "[Canopy DEBUG sort] %s (N=%d num_cells=%d)\n",
                          msg, N, num_cells );
            std::fflush( stderr );
        }
    };
    dbg_mark( "entry" );

    // Build key -> cell_index map on host
    std::unordered_map<MortonKey, int> key_to_idx;
    key_to_idx.reserve( num_cells );
    for ( int i = 0; i < num_cells; i++ )
        key_to_idx[cells[i].key] = i;

    // Compute each particle's leaf cell index on host
    auto particle_keys = tree_builder.particle_keys();
    auto h_keys = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                       particle_keys );

    // Build a sort-key view: sort_keys(i) = cell index of particle i.
    // Cabana::sortByKey groups particles with the same integer key into bins,
    // one bin per cell, which is exactly the sorted-by-leaf layout we need.
    Kokkos::View<int*, memory_space> sort_keys( "sort_keys",
                                                static_cast<size_t>( N ) );
    {
        auto h = Kokkos::create_mirror_view( sort_keys );
        int kmin = num_cells, kmax = -1;
        for ( int i = 0; i < N; i++ )
        {
            auto it = key_to_idx.find( h_keys( i ) );
            const int k = ( it != key_to_idx.end() ) ? it->second : 0;
            h( i ) = k;
            if ( k < kmin ) kmin = k;
            if ( k > kmax ) kmax = k;
        }
        Kokkos::deep_copy( sort_keys, h );
        // OOB sort key would feed an out-of-range bin to binByKey/permute.
        if ( _rank == 0 )
        {
            std::fprintf( stderr,
                "[Canopy DEBUG sort] sort_keys built: min=%d max=%d (valid "
                "[0,%d)) %s\n",
                kmin, kmax, num_cells,
                ( kmin < 0 || kmax >= num_cells ) ? "<<< OOB KEY" : "ok" );
            std::fflush( stderr );
        }
    }
    dbg_mark( "before binByKey" );

    // Group particles by cell index. The keys ARE dense cell indices in
    // [0,num_cells), so bin by cell with exactly num_cells bins (one per cell)
    // and NO within-bin sort: Cabana::binByKey -> Kokkos BinOp1D with
    // (max-min) <= num_cells sets mul_=1, so bin == key-min and particles end
    // up contiguous in ascending cell-index order — exactly the sorted-by-leaf
    // layout, which is all Cabana::permute and _leaf_particle_offsets need.
    //
    // We deliberately do NOT use Cabana::sortByKey here: it hardcodes
    // nbin = (end-begin)/2 (~5e7 range bins at 1e8 particles) AND
    // sort_within_bins=true. That pathological config — a ~5e7-entry atomic
    // bin array pounded by 1e8 atomic increments plus a 5e7-bin within-bin
    // insertion-sort pass — triggers a GPU "write to read-only page" fault on
    // the MI300A unified-memory path at 4e8 particles (MI300A investigation
    // bug 4). binByKey avoids both the giant bin array and the within-bin sort.
    auto bin_data = Cabana::binByKey( sort_keys, num_cells, std::size_t( 0 ),
                                      std::size_t( N ) );
    dbg_mark( "after binByKey" );

    // Count particles per cell. binByKey leaves sort_keys unmodified (it only
    // builds the permutation), so this is a plain histogram of cell indices —
    // order-independent, so it does not matter that the keys are not sorted.
    std::vector<int> cell_counts( num_cells, 0 );
    {
        auto h_sk = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                         sort_keys );
        for ( int i = 0; i < N; i++ )
            cell_counts[h_sk( i )]++;
    }
    {
        long long maxc = 0;
        for ( int c = 0; c < num_cells; c++ )
            if ( cell_counts[c] > maxc ) maxc = cell_counts[c];
        if ( _rank == 0 )
        {
            std::fprintf( stderr,
                "[Canopy DEBUG sort] max_cell_count=%lld\n", maxc );
            std::fflush( stderr );
        }
    }
    dbg_mark( "before permute" );

    // Permute the AoSoA so particles are contiguous within each cell.
    Cabana::permute( bin_data, particles );
    dbg_mark( "after permute" );

    // Build leaf_particle_offsets as a prefix sum of per-cell counts.
    _leaf_particle_offsets = Kokkos::View<int*, memory_space>(
        std::string( "leaf_particle_offsets" ),
        static_cast<size_t>( num_cells + 1 ) );
    {
        auto h = Kokkos::create_mirror_view( _leaf_particle_offsets );
        h( 0 ) = 0;
        for ( int c = 0; c < num_cells; c++ )
            h( c + 1 ) = h( c ) + cell_counts[c];
        Kokkos::deep_copy( _leaf_particle_offsets, h );
    }

    // Build per-particle cell index using the offsets.
    // After permute, particles in [offsets(c), offsets(c+1)) belong to cell c.
    _particle_leaf_cell_idx = Kokkos::View<int*, memory_space>(
        std::string( "particle_leaf_cell_idx" ), static_cast<size_t>( N ) );
    {
        auto h = Kokkos::create_mirror_view( _particle_leaf_cell_idx );
        auto h_off = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), _leaf_particle_offsets );
        for ( int c = 0; c < num_cells; c++ )
            for ( int k = h_off( c ); k < h_off( c + 1 ); k++ )
                h( k ) = c;
        Kokkos::deep_copy( _particle_leaf_cell_idx, h );
    }
    dbg_mark( "done (offsets + cell_idx built)" );
}

} // end namespace Canopy

#endif // CANOPY_TREE_PARTITIONER_HPP