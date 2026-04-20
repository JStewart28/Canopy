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

#pragma once

#include <Canopy_Experimental_TreeBuilder.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <Zoltan2_BasicVectorAdapter.hpp>
#include <Zoltan2_PartitioningProblem.hpp>

#include <Tpetra_Map.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_DefaultMpiComm.hpp>

#include <mpi.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace Canopy
{

namespace Experimental
{

// ============================================================================
// Ownership constants
// ============================================================================

// Cells at or above the replication cutoff depth are owned by all ranks.
// This value indicates shared (replicated) ownership.
static constexpr int OWNER_SHARED = -1;

// ============================================================================
// CellOwnership — per-cell ownership info, maps to the
// TreeBuilder's cells() vector.
// ============================================================================

struct CellOwnership
{
    MortonKey key;
    int owner_rank; // OWNER_SHARED for replicated coarse cells,
                    // otherwise the unique owning rank
};

// ============================================================================
// TreePartitioner
//
// Given a globally-agreed adaptive octree (from TreeBuilder),
// partitions the leaf cells across MPI ranks using Zoltan2 RCB, derives
// internal cell ownership using the replicated-coarse-layers strategy,
// and migrates particles so each rank holds exactly the particles in its
// owned leaf cells.
//
// Replicated coarse layers:
//   Cells at depth <= replication_depth are owned by ALL ranks. Each rank
//   holds a copy and computes partial M2M contributions, which are then
//   summed via MPI_Allreduce.
//
//   Cells deeper than replication_depth have a single owner — the rank
//   that owns the most descendant particles. Point-to-point MPI sends
//   transfer child multipole coefficients to parent owners during the
//   upward sweep.
//
// Template Parameters:
//   DeviceType - Kokkos device type (must match TopDownTreeBuilder)
//   AoSoAType  - Cabana AoSoA type for particle data
//   PositionIndex - Integer index of the position field in the AoSoA
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
    // Accessors (valid after partition())
    // -----------------------------------------------------------------------

    // Full ownership map, indexes map to tree_builder.cells()
    const std::vector<CellOwnership>& ownership() const
    {
        return _ownership;
    }

    // Lookup owner of a specific cell by Morton key.
    // Returns OWNER_SHARED for replicated cells, or the owning rank.
    int cell_owner( MortonKey key ) const
    {
        auto it = _cell_owner_map.find( key );
        if ( it != _cell_owner_map.end() )
            return it->second;
        return OWNER_SHARED; // unknown cells default to shared
    }

    // Number of local particles after migration
    int num_local_particles() const { return _num_local_after; }

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

    // Post-migration local particle count
    int _num_local_after;

  public:
    // -----------------------------------------------------------------------
    // Internal: partition leaf cells using Zoltan2 RCB
    //
    // Returns a map: leaf MortonKey -> owning rank
    // -----------------------------------------------------------------------
    std::unordered_map<MortonKey, int> partition_leaves(
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
                // User particle count for zoltan weights
                leaf_weights.push_back(
                    static_cast<double>( c.global_count ) );
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
        // Every rank has the identical set of leaves (the tree structure is replciated globally),
        // so run Zoltan2 with the full leaf set on every rank. Zoltan2 RCB is deterministic given
        // the same input, so all ranks will produce the same assignment.
        //
        // Use global IDs = leaf index (0..num_leaves-1), which are the
        // same on every rank since _cells from TreeBuilder is identical on all ranks.
        // ---------------------------------------------------------------

        // Create Zoltan2 adapter
        // BasicVectorAdapter needs:
        //   numIds, globalIds, coords, weights
        using adapter_t = Zoltan2::BasicVectorAdapter<
            Tpetra::Map<int, int64_t>>;
        // Must also use Zoltan types
        using glbl_id_t = typename adapter_t::gno_t;
        using scalar_t = typename adapter_t::scalar_t;
        using longint_t = typename adapter_t::lno_t;

        // Zoltan2 needs coordinates as an array of pointers, one per dim
        const scalar_t* coords[3] = { leaf_x.data(), leaf_y.data(),
                                    leaf_z.data() };
        // const int strides[3] = { 1, 1, 1 };

        // Global IDs for the leaves
        std::vector<glbl_id_t> global_ids( num_leaves );
        for ( int i = 0; i < num_leaves; i++ )
            global_ids[i] = static_cast<glbl_id_t>( i );

        // Build a Teuchos communicator for Zoltan2
        // Note: Use MPI_COMM_SELF because every rank is running the same
        // deterministic partitioning on the same data. This avoids Zoltan2
        // trying to do distributed partitioning (which would fail since
        // each rank claims to have ALL leaves).
        auto teuchos_comm =
            Teuchos::rcp( new Teuchos::MpiComm<int>( MPI_COMM_SELF ) );

        const glbl_id_t* ids_ptr = global_ids.data();
        const scalar_t* x_ptr = leaf_x.data();
        const scalar_t* y_ptr = leaf_y.data();
        const scalar_t* z_ptr = leaf_z.data();
        const scalar_t* w_ptr = leaf_weights.data();

        adapter_t adapter(
            static_cast<longint_t>( num_leaves ),
            ids_ptr,
            x_ptr, y_ptr, z_ptr,
            1, 1, 1,              // strides for x, y, z
            true,                 // use weights
            w_ptr,
            1 );                  // weight stride

        // Configure Zoltan2
        Teuchos::ParameterList params;
        params.set( "algorithm", "rcb" );
        params.set( "num_global_parts", _comm_size );
        params.set( "imbalance_tolerance", _imbalance_tolerance );

        // Solve
        Zoltan2::PartitioningProblem<adapter_t> problem(
            &adapter, &params, teuchos_comm );
        problem.solve();

        // Extract assignments
        const auto& solution = problem.getSolution();
        // Array of which rank should get which particle.
        const int* parts = solution.getPartListView();

        std::unordered_map<MortonKey, int> result;
        result.reserve( num_leaves );
        for ( int i = 0; i < num_leaves; i++ )
            result[leaf_keys[i]] = parts[i];

        return result;
    }

    // --------------------------------------------------------------------------
    // derive_internal_ownership
    //
    // Two-phase approach:
    //   Phase 1: For each leaf, walk up the parent chain accumulating
    //            (parent_key -> { rank -> particle_count }) votes.
    //   Phase 2: For each internal cell, the owner is the rank with the
    //            most descendant particles, unless the cell is at or above
    //            the replication depth cutoff, in which case it's SHARED.
    // --------------------------------------------------------------------------
    void derive_internal_ownership(
        const std::vector<CellInfo>& cells,
        const std::unordered_map<MortonKey, int>& leaf_owners )
    {
        // Build a key -> CellInfo lookup
        std::unordered_map<MortonKey, const CellInfo*> cell_map;
        for ( const auto& c : cells )
            cell_map[c.key] = &c;

        // Phase 1: accumulate votes
        // For each internal cell, track how many descendant particles each
        // rank contributes.
        // vote_map[internal_key][rank] = total descendant particle count
        // from leaves owned by that rank
        std::unordered_map<MortonKey,
                        std::unordered_map<int, int64_t>> vote_map;

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
                co.owner_rank = ( it != leaf_owners.end() )
                                    ? it->second
                                    : 0; // shouldn't happen
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
                    int best_rank = 0;
                    int64_t best_count = -1;
                    for ( const auto& [r, cnt] : vote_it->second )
                    {
                        if ( cnt > best_count )
                        {
                            best_count = cnt;
                            best_rank = r;
                        }
                    }
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
    
        // -----------------------------------------------------------------------
        // partition()
        // num_local_particles_before - particles.size() if nothing is ghosted.
        //
        // Main entry point. Takes the tree builder (after build() has been
        // called) and the particle container. Performs:
        //
        //   1. Extract leaf cells and partition them via Zoltan2 RCB.
        //   2. Derive internal cell ownership from leaf assignments.
        //   3. Migrate particles to the rank that owns their leaf cell.
        //
        // After calling partition():
        //   - ownership()        returns the ownership map for all cells
        //   - leaf_owner()       looks up the owner of a specific leaf
        //   - cell_owner()       looks up the owner of any cell
        //   - The AoSoA has been redistributed so each rank holds only the
        //     particles in its owned leaves.
        //   - num_local_particles() returns the new local particle count.
        //
        // -----------------------------------------------------------------------
        template <class AoSoAType>
        void partition(
            const TreeBuilder<MemorySpace, ExecutionSpace>& tree_builder,
            AoSoAType& particles,
            int num_local_particles_before )
        {
        const auto& cells = tree_builder.cells();

        // ==================================================================
        // Step 1: Partition leaf cells via Zoltan2
        // ==================================================================
        auto leaf_owners = partition_leaves( cells );

        // ==================================================================
        // Step 2: Derive internal cell ownership
        // ==================================================================
        derive_internal_ownership( cells, leaf_owners );

        // ==================================================================
        // Step 3: Migrate particles to the rank that owns their leaf cell
        //
        // Each particle's current leaf key is stored in
        // tree_builder.particle_keys(). Look up the owner of that leaf
        // and build a destination-rank array for Cabana::Distributor.
        // ==================================================================

        // Copy particle keys to host to build the destination array.
        auto particle_keys = tree_builder.particle_keys();
        auto h_keys = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), particle_keys );
        Kokkos::View<int*, memory_space> dest_ranks(
            "dest_ranks", num_local_particles_before );
        auto h_dest = Kokkos::create_mirror_view( dest_ranks );

        // Build destination ranks on host.
        for ( int i = 0; i < num_local_particles_before; i++ )
        {
            // Get the cell this particle resides in.
            MortonKey key = h_keys( i );
            auto it = _cell_owner_map.find( key );
            if ( it != _cell_owner_map.end() )
            {
                int owner = it->second;
                // If the leaf is OWNER_SHARED (shouldn't happen for leaves,
                // but guard against it), keep particle on current rank.
                h_dest( i ) = ( owner >= 0 ) ? owner : _rank;
            }
            else
            {
                // Key not found — keep on current rank (safety fallback)
                h_dest( i ) = _rank;
            }
        }

        Kokkos::deep_copy( dest_ranks, h_dest );

        // Use Cabana::Distributor to migrate particles.
        Cabana::Distributor<memory_space> distributor( _comm, dest_ranks );
        Cabana::migrate( distributor, particles );

        _num_local_after = static_cast<int>( particles.size() );
    }
};

} // end namespace Experimental

} // end namespace Canopy

#endif // CANOPY_TREE_PARTITIONER_HPP
