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

#ifndef CANOPY_SOLVER_HPP
#define CANOPY_SOLVER_HPP


#include <ArborX.hpp>
#include <Canopy_SolverLayer.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <memory>

#include <mpi.h>

#include <limits>

namespace Canopy
{

namespace Experimental
{

// https://repositorio.unesp.br/server/api/core/bitstreams/0e824479-3128-41f7-8cd2-462e9a242c42/content

// ============================================================================
// Morton key type and helpers
// ============================================================================

// 64-bit Morton key. Convention:
//   root key = 1
//   children of key k = 8*k + 0 .. 8*k + 7
//   parent of key k   = k / 8   (integer division, k > 1)
//   depth of key k    = floor(log8(k))
//
// This supports trees up to depth 20 (8^20 fits in 63 bits with the
// leading-1 convention), which is far more than any practical FMM needs.
using MortonKey = uint64_t;

static constexpr MortonKey ROOT_KEY = 1;
static constexpr int MAX_CHILDREN = 8;

KOKKOS_INLINE_FUNCTION
MortonKey parent_key( MortonKey k )
{
    return k >> 3; // equivalent to k / 8
}

KOKKOS_INLINE_FUNCTION
MortonKey child_key( MortonKey k, int octant )
{
    return ( k << 3 ) | static_cast<MortonKey>( octant );
}

KOKKOS_INLINE_FUNCTION
int key_depth( MortonKey k )
{
    int d = 0;
    while ( k > 1 )
    {
        k >>= 3;
        ++d;
    }
    return d;
}

KOKKOS_INLINE_FUNCTION
int key_octant( MortonKey k )
{
    return static_cast<int>( k & 7 ); // last 3 bits
}

// ============================================================================
// Cell data (global tree topology)
// ============================================================================
struct CellInfo
{
    MortonKey key;
    int depth;
    double center[3];
    double half_width; // half the side length of this cell's cube
    int global_count;  // total particles across all ranks
    bool is_leaf;
};

// ============================================================================
// Axis-aligned bounding box
// ============================================================================
struct BoundingBox
{
    double min[3];
    double max[3];
};

// ============================================================================
// UpdateResult — returned by update() to report what happened
// ============================================================================
struct UpdateResult
{
    int particles_migrated;  // particles that moved to a different leaf
    int cells_refined;       // leaf cells that were split
    int cells_coarsened;     // groups of siblings collapsed to parent
    bool full_rebuild_done;  // true if the bounding box changed and a
                             // full rebuild was triggered instead
};

template <class MemorySpace, class ExecutionSpace>
class TreeBuilder
{
  public:
    // Check metadata
    // static_assert(Metadata::pos != no_id, "metadata must define position index");
    // static_assert(Metadata::in != no_id, "metadata must define in_data index");
    // static_assert(Metadata::out != no_id, "metadata must define out_data index");

    // using metadata = Metadata;

    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    
    //! Self type
    // using solver_type = Solver<MemorySpace, ExecutionSpace, Metadata, P>;

    //! Dimension number
    static constexpr int num_space_dim = 3;
    //! P-term for expansions
    static constexpr int p = P;
    //! Memory space size type
    using size_type = typename memory_space::size_type;
    //! Scalar type
    using scalar_type = typename metadata::scalar_type;
    
    // Host mirror types for tree data
    using host_execution_space = Kokkos::DefaultHostExecutionSpace;
    using host_memory_space = typename host_execution_space::memory_space;

    // The particle-to-leaf mapping lives on device for fast particle kernels
    using key_view_type = Kokkos::View<MortonKey*, memory_space>;
    using key_host_view_type =
        Kokkos::View<MortonKey*, host_memory_space>;

  private:
    // MPI
    MPI_Comm _comm;
    int _rank;
    int _comm_size;

    //! Max number of particles per cell
    int _ncrit;
    //! Maximum tree depth
    int _max_depth;
    //! Tolerance factor on global bounding box
    double _tolerance_factor

    // Tree state
    bool _tree_valid;

    //! Global tree topology (identical on every rank)
    std::vector<CellInfo> _cells;

    // Fast lookup: MortonKey -> index into _cells
    std::unordered_map<MortonKey, int> _cell_lookup;

    //! Particle -> leaf key mapping (device view)
    key_view_type _particle_keys;

    //! Global bounding box
    BoundingBox _root_box;

  public:
    // Constructor
    TreeBuilder( const int ncrit, const int max_depth, const int tolerance_factor = 0.1, MPI_Comm comm ) 
        : _ncrit( ncrit )
        , _max_depth( max_depth )
        , _tolerance_factor( tolerance_factor )
        , _comm( comm )
    {
        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );
    }

    // -----------------------------------------------------------------------
    // Getters
    // -----------------------------------------------------------------------
    const std::vector<CellInfo>& cells() const { return _cells; }
    const key_view_type& particle_keys() const { return _particle_keys; }
    const BoundingBox& root_box() const { return _root_box; }
    bool tree_valid() const { return _tree_valid; }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    // Determine which octant of a cell a point falls in based on point
    // position relative to cell center.
    KOKKOS_INLINE_FUNCTION
    static int which_octant( double px, double py, double pz, double cx,
                             double cy, double cz )
    {
        int octant = 0;
        if ( px >= cx )
            octant |= 1;
        if ( py >= cy )
            octant |= 2;
        if ( pz >= cz )
            octant |= 4;
        return octant;
    }

    // Compute child cell center given parent center, half_width, and octant
    KOKKOS_INLINE_FUNCTION
    static void child_center( double parent_cx, double parent_cy,
                              double parent_cz, double parent_hw,
                              int octant, double& cx, double& cy,
                              double& cz )
    {
        double quarter = parent_hw * 0.5;
        cx = parent_cx + ( ( octant & 1 ) ? quarter : -quarter );
        cy = parent_cy + ( ( octant & 2 ) ? quarter : -quarter );
        cz = parent_cz + ( ( octant & 4 ) ? quarter : -quarter );
    }

    // Compute global bounding box from distributed particles
    template <class PositionType>
    BoundingBox compute_global_bounding_box( PositionType positions,
                                             int num_local_particles )
    {
        double local_min[3], local_max[3];
        double inf = std::numeric_limits<double>::max();

        Kokkos::parallel_reduce(
            "ComputeLocalBBox",
            Kokkos::RangePolicy<execution_space>( 0, num_local_particles ),
            KOKKOS_LAMBDA( int i, double& lmin_x, double& lmin_y,
                        double& lmin_z, double& lmax_x, double& lmax_y,
                        double& lmax_z ) {
                double px = positions( i, 0 );
                double py = positions( i, 1 );
                double pz = positions( i, 2 );
                if ( px < lmin_x ) lmin_x = px;
                if ( py < lmin_y ) lmin_y = py;
                if ( pz < lmin_z ) lmin_z = pz;
                if ( px > lmax_x ) lmax_x = px;
                if ( py > lmax_y ) lmax_y = py;
                if ( pz > lmax_z ) lmax_z = pz;
            },
            Kokkos::Min<double>( local_min[0] ),
            Kokkos::Min<double>( local_min[1] ),
            Kokkos::Min<double>( local_min[2] ),
            Kokkos::Max<double>( local_max[0] ),
            Kokkos::Max<double>( local_max[1] ),
            Kokkos::Max<double>( local_max[2] ) );

        BoundingBox box;
        MPI_Allreduce( local_min, box.min, 3, MPI_DOUBLE, MPI_MIN, comm_ );
        MPI_Allreduce( local_max, box.max, 3, MPI_DOUBLE, MPI_MAX, comm_ );

        double pad = 1.0e-10;
        for ( int d = 0; d < 3; ++d )
        {
            double width = box.max[d] - box.min[d];
            if ( width < pad )
                width = pad;
            box.min[d] -= pad * width;
            box.max[d] += pad * width;
        }

        return box;
    }

    // Rebuild cell_lookup map of cell morton key to index into cells vector.
    void rebuild_cell_lookup()
    {
        _cell_lookup.clear();
        _cell_lookup.reserve( _cells.size() );
        for ( int i = 0; i < static_cast<int>( _cells.size() ); ++i )
            _cell_lookup[_cells[i].key] = i;
    }

    // -----------------------------------------------------------------------
    // Main functions
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // build()
    //
    // Main entry point. Takes particle positions (a Cabana slice) and the
    // number of local particles. Returns after all ranks agree on the tree
    // topology and each rank's particles are tagged with their leaf cell key.
    //
    // After calling build():
    //   - cells()           returns the global tree (vector of CellInfo)
    //   - particle_keys()   returns a device view mapping particle index
    //                       to the Morton key of its enclosing leaf cell
    //   - root_box()        returns the global bounding box
    // -----------------------------------------------------------------------
    template <class PositionType>
    void build( PositionType positions, int num_local_particles )
    {
        // Compute global bounding box
        root_box_ =
            compute_global_bounding_box( positions, num_local_particles );

        // Expand the root box by the tolerance factor so particles have room
        // to move before leaving the domain.
        if ( _tolerance_factor > 0.0 )
        {
            for ( int d = 0; d < 3; ++d )
            {
                double width = _root_box.max[d] - _root_box.min[d];
                double expansion = _tolerance_factor * width;
                _root_box.min[d] -= expansion;
                _root_box.max[d] += expansion;
            }
        }

        // Coordinates of root box center
        double root_cx =
            0.5 * ( root_box_.min[0] + root_box_.max[0] );
        double root_cy =
            0.5 * ( root_box_.min[1] + root_box_.max[1] );
        double root_cz =
            0.5 * ( root_box_.min[2] + root_box_.max[2] );
        
        // Use the max dimension to compute the half-width of the root box
        double root_hw = 0.0;
        for ( int d = 0; d < 3; ++d )
        {
            double hw = 0.5 * ( root_box_.max[d] - root_box_.min[d] );
            if ( hw > root_hw )
                root_hw = hw;
        }

        // Initialize particle-to-cell mapping on device
        _particle_keys =
            key_view_type( "particle_keys", num_local_particles );
        // All particles initially live in root box
        Kokkos::deep_copy( _particle_keys, ROOT_KEY );

        // Clear existing topology
        _cells.clear();

        struct Cell
        {
            MortonKey key;
            double center[3];
            double half_width;
        };

        std::vector<Cell> cells_to_refine;
        cells_to_refine.push_back(
            { ROOT_KEY, { root_cx, root_cy, root_cz }, root_hw } );

        // Iteratively refine the root box until all cells have less than ncrit particles
        for ( int depth = 0; depth <= _max_depth; depth++ )
        {
            if ( cells_to_refine.empty() )
                break;

            int num_candidates = static_cast<int>( cells_to_refine.size() );

            // Move candidate cells into views
            Kokkos::View<MortonKey*, memory_space> cand_keys( "cand_keys",
                                                            num_candidates );
            Kokkos::View<double* [3], memory_space> cand_centers(
                "cand_centers", num_candidates );
            Kokkos::View<double*, memory_space> cand_hw( "cand_hw",
                                                        num_candidates );

            auto h_cand_keys = Kokkos::create_mirror_view( cand_keys );
            auto h_cand_centers = Kokkos::create_mirror_view( cand_centers );
            auto h_cand_hw = Kokkos::create_mirror_view( cand_hw );

            for ( int c = 0; c < num_candidates; ++c )
            {
                h_cand_keys( c ) = cells_to_refine[c].key;
                for ( int d = 0; d < 3; ++d )
                    h_cand_centers( c, d ) = cells_to_refine[c].center[d];
                h_cand_hw( c ) = cells_to_refine[c].half_width;
            }

            Kokkos::deep_copy( cand_keys, h_cand_keys );
            Kokkos::deep_copy( cand_centers, h_cand_centers );
            Kokkos::deep_copy( cand_hw, h_cand_hw );

            Kokkos::View<int*, memory_space> local_counts( "local_counts",
                                                        num_candidates );
            Kokkos::View<int* [8], memory_space> local_octant_counts(
                "local_octant_counts", num_candidates );

            Kokkos::UnorderedMap<MortonKey, int, memory_space> key_to_cand_idx(
                num_candidates * 2 );

            Kokkos::parallel_for(
                "PopulateKeyMap",
                Kokkos::RangePolicy<execution_space>( 0, num_candidates ),
                KOKKOS_LAMBDA( int c ) {
                    key_to_cand_idx.insert( cand_keys( c ), c );
                } );
            Kokkos::fence();

            auto particle_keys = _particle_keys;

            Kokkos::View<int*, memory_space> particle_octant(
                "particle_octant", num_local_particles );
            Kokkos::deep_copy( particle_octant, -1 );

            Kokkos::parallel_for(
                "CountParticlesPerCandidate",
                Kokkos::RangePolicy<execution_space>( 0, num_local_particles ),
                KOKKOS_LAMBDA( int i ) {
                    MortonKey my_key = particle_keys( i );

                    auto idx = key_to_cand_idx.find( my_key );
                    if ( !key_to_cand_idx.valid_at( idx ) )
                        return;

                    int c = key_to_cand_idx.value_at( idx );
                    Kokkos::atomic_increment( &local_counts( c ) );

                    double cx = cand_centers( c, 0 );
                    double cy = cand_centers( c, 1 );
                    double cz = cand_centers( c, 2 );
                    double px = positions( i, 0 );
                    double py = positions( i, 1 );
                    double pz = positions( i, 2 );

                    int oct = which_octant( px, py, pz, cx, cy, cz );
                    particle_octant( i ) = oct;
                    Kokkos::atomic_increment(
                        &local_octant_counts( c, oct ) );
                } );
            Kokkos::fence();

            auto h_local_counts = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), local_counts );
            std::vector<int> global_counts( num_candidates );
            for ( int c = 0; c < num_candidates; ++c )
                send_counts[c] = h_local_counts( c );

            // All reduce the number of particles in each cell.
            MPI_Allreduce( h_local_counts.data(), global_counts.data(),
                        num_candidates, MPI_INT, MPI_SUM, _comm );

            std::vector<int> should_split( num_candidates, 0 );
            std::vector<Cell> next_cells_to_refine;

            // Loop over candidate cells. If there are globally greater than 'ncrit'
            // particles in a candidate cell, refine the cell.
            for ( int c = 0; c < num_candidates; ++c )
            {
                // No particles exist in this cell; do not add it
                if ( global_counts[c] == 0 )
                    continue;

                // Get info for this cell
                CellInfo ci;
                ci.key = cells_to_refine[c].key;
                ci.depth = depth;
                ci.center[0] = cells_to_refine[c].center[0];
                ci.center[1] = cells_to_refine[c].center[1];
                ci.center[2] = cells_to_refine[c].center[2];
                ci.half_width = cells_to_refine[c].half_width;
                ci.global_count = global_counts[c];

                // Check if the cell should be refined, and add it to
                // the list of cells.
                if ( global_counts[c] <= _ncrit || depth == _max_depth )
                {
                    ci.is_leaf = true;
                    _cells.push_back( ci );
                }
                else
                {
                    ci.is_leaf = false;
                    _cells.push_back( ci );
                    should_split[c] = 1;

                    double parent_cx = cells_to_refine[c].center[0];
                    double parent_cy = cells_to_refine[c].center[1];
                    double parent_cz = cells_to_refine[c].center[2];
                    double parent_hw = cells_to_refine[c].half_width;
                    double ch_hw = parent_hw * 0.5;

                    // Split cell into its 8 children and add them to
                    // next_cells_to_refine vector as refinement candidates.
                    for ( int oct = 0; oct < 8; oct+ )
                    {
                        Cell cell;
                        cell.key = child_key( cells_to_refine[c].key, oct );
                        child_center( parent_cx, parent_cy, parent_cz,
                                    parent_hw, oct, cell.center[0],
                                    cell.center[1], cell.center[2] );
                        cell.half_width = ch_hw;
                        next_cells_to_refine.push_back( cell );
                    }
                }
            }

            // Update particle keys to be in a child cell if the cell was split.
            Kokkos::View<int*, memory_space> splits( "splits",
                                                    num_candidates );
            auto h_splits = Kokkos::create_mirror_view( splits );
            for ( int c = 0; c < num_candidates; ++c )
                h_splits( c ) = split_decision[c];
            Kokkos::deep_copy( splits, h_splits );

            Kokkos::parallel_for(
                "UpdateParticleKeys",
                Kokkos::RangePolicy<execution_space>( 0, num_local_particles ),
                KOKKOS_LAMBDA( int i ) {
                    int oct = particle_octant( i );
                    if ( oct < 0 )
                        return;

                    MortonKey my_key = particle_keys( i );
                    auto idx = key_to_cand_idx.find( my_key );
                    if ( !key_to_cand_idx.valid_at( idx ) )
                        return;

                    int c = key_to_cand_idx.value_at( idx );

                    if ( splits( c ) )
                        particle_keys( i ) = child_key( my_key, oct );
                } );
            Kokkos::fence();

            // Move next_cells_to_refine into cells_to_refine for next iteration.
            cells_to_refine = std::move( next_cells_to_refine );

        } // end tree build loop

        // Build the host-side lookup map
        rebuild_cell_lookup();
        _tree_valid = true;
    }

    // Check if particles have left the root bounding box
    template <class PositionType>
    needs_rebuild(PositionType positions, int num_local_particles ) const
    {
        // Check against the root box
        double bmin0 = root_box_.min[0], bmin1 = root_box_.min[1],
            bmin2 = root_box_.min[2];
        double bmax0 = root_box_.max[0], bmax1 = root_box_.max[1],
            bmax2 = root_box_.max[2];

        int local_escaped = 0;
        Kokkos::parallel_reduce(
            "CheckBounds",
            Kokkos::RangePolicy<execution_space>( 0, num_local_particles ),
            KOKKOS_LAMBDA( int i, int& count ) {
                double px = positions( i, 0 );
                double py = positions( i, 1 );
                double pz = positions( i, 2 );
                if ( px < bmin0 || px > bmax0 ||
                    py < bmin1 || py > bmax1 ||
                    pz < bmin2 || pz > bmax2 )
                {
                    count++;
                }
            },
            local_escaped );

        int global_escaped = 0;
        MPI_Allreduce( &local_escaped, &global_escaped, 1, MPI_INT, MPI_SUM,
                    _comm );

        return global_escaped > 0;
    }

    // --------------------------------------------------------------------------
    // update() — incremental tree adaptation
    // --------------------------------------------------------------------------
    template <class PositionType>
    update(PositionType positions, int num_local_particles )
    {
        // Particles moved to different leaf
        // Leaf cells split
        // Cells coarsened
        // Full rebuild due to bounding box change
        UpdateResult result = { 0, 0, 0, false };

        // If the tree was never built, do a full build
        if ( !_tree_valid )
        {
            build( positions, num_local_particles );
            result.full_rebuild_done = true;
            return result;
        }

        // Check if any particles escaped the root bounding box, which
        // also requires a full rebuild
        if ( needs_rebuild( positions, num_local_particles ) )
        {
            build( positions, num_local_particles );
            result.full_rebuild_done = true;
            return result;
        }

        // --------------------------------------------------------------------------
        // Detect which particles have left their current leaf cell.
        // If so, it needs reassignment.
        // --------------------------------------------------------------------------

        // Build a device-side lookup for leaf cells.
        // We need center and half_width for each leaf on the device.

        // Collect leaf cells
        std::vector<MortonKey> leaf_keys_vec;
        std::vector<double> leaf_cx, leaf_cy, leaf_cz, leaf_hw;
        for ( const auto& ci : _cells )
        {
            if ( ci.is_leaf )
            {
                leaf_keys_vec.push_back( ci.key );
                leaf_cx.push_back( ci.center[0] );
                leaf_cy.push_back( ci.center[1] );
                leaf_cz.push_back( ci.center[2] );
                leaf_hw.push_back( ci.half_width );
            }
        }

        int num_leaves = static_cast<int>( leaf_keys_vec.size() );

        // Build a device version of _particle_keys hash map
        // for leaf cells only: leaf key -> index into leaf arrays
        Kokkos::UnorderedMap<MortonKey, int, memory_space> leaf_map(
            num_leaves * 2 );

        Kokkos::View<MortonKey*, memory_space> d_leaf_keys( "d_leaf_keys",
                                                            num_leaves );
        Kokkos::View<double*, memory_space> d_leaf_cx( "d_leaf_cx",
                                                    num_leaves );
        Kokkos::View<double*, memory_space> d_leaf_cy( "d_leaf_cy",
                                                    num_leaves );
        Kokkos::View<double*, memory_space> d_leaf_cz( "d_leaf_cz",
                                                    num_leaves );
        Kokkos::View<double*, memory_space> d_leaf_hw( "d_leaf_hw",
                                                    num_leaves );

        // Create host mirrors, copy data, move to device.
        {
            auto h_lk = Kokkos::create_mirror_view( d_leaf_keys );
            auto h_cx = Kokkos::create_mirror_view( d_leaf_cx );
            auto h_cy = Kokkos::create_mirror_view( d_leaf_cy );
            auto h_cz = Kokkos::create_mirror_view( d_leaf_cz );
            auto h_hw = Kokkos::create_mirror_view( d_leaf_hw );

            for ( int j = 0; j < num_leaves; ++j )
            {
                h_lk( j ) = leaf_keys_vec[j];
                h_cx( j ) = leaf_cx[j];
                h_cy( j ) = leaf_cy[j];
                h_cz( j ) = leaf_cz[j];
                h_hw( j ) = leaf_hw[j];
            }
            Kokkos::deep_copy( d_leaf_keys, h_lk );
            Kokkos::deep_copy( d_leaf_cx, h_cx );
            Kokkos::deep_copy( d_leaf_cy, h_cy );
            Kokkos::deep_copy( d_leaf_cz, h_cz );
            Kokkos::deep_copy( d_leaf_hw, h_hw );
        }

        Kokkos::parallel_for(
            "PopulateLeafMap",
            Kokkos::RangePolicy<execution_space>( 0, num_leaves ),
            KOKKOS_LAMBDA( int j ) {
                leaf_map.insert( d_leaf_keys( j ), j );
            } );
        Kokkos::fence();

        // Check each particle: is it still inside its leaf cell's buffered
        // region?  The buffered region is the cell expanded by
        // tolerance_factor * half_width on each side.
        //
        // A particle is "escaped" if it's outside the buffered bounds.
        // We flag these and count them.

        Kokkos::View<int*, memory_space> escaped_flag( "escaped_flag",
                                                    num_local_particles );
        double tol = tolerance_factor_;
        auto p_keys = particle_keys_;

        int local_migrated = 0;
        Kokkos::parallel_reduce(
            "DetectEscaped",
            Kokkos::RangePolicy<execution_space>( 0, num_local_particles ),
            KOKKOS_LAMBDA( int i, int& count ) {
                MortonKey my_key = p_keys( i );
                auto idx = leaf_map.find( my_key );
                if ( !leaf_map.valid_at( idx ) )
                {
                    // Particle's key doesn't correspond to a current leaf
                    // (tree structure changed). Mark as needing reassignment.
                    escaped_flag( i ) = 1;
                    count++;
                    return;
                }

                int j = leaf_map.value_at( idx );
                double cx = d_leaf_cx( j );
                double cy = d_leaf_cy( j );
                double cz = d_leaf_cz( j );
                double hw = d_leaf_hw( j );
                double buffer = hw * tol;
                double buffered_hw = hw + buffer;

                double px = positions( i, 0 );
                double py = positions( i, 1 );
                double pz = positions( i, 2 );

                // Check if outside the buffered cell
                if ( px < cx - buffered_hw || px > cx + buffered_hw ||
                    py < cy - buffered_hw || py > cy + buffered_hw ||
                    pz < cz - buffered_hw || pz > cz + buffered_hw )
                {
                    escaped_flag( i ) = 1;
                    count++;
                }
            },
            local_migrated );
        Kokkos::fence();

        // Global count of migrated particles (for reporting)
        int global_migrated = 0;
        MPI_Allreduce( &local_migrated, &global_migrated, 1, MPI_INT,
                    MPI_SUM, comm_ );
        result.particles_migrated = global_migrated;

        // ==================================================================
        // Step 3: If no particles migrated, we're done — tree is unchanged
        // ==================================================================
        if ( global_migrated == 0 )
            return result;

        // ==================================================================
        // Step 4: Reassign ALL particle keys by walking the tree
        //
        // This is simpler and more robust than trying to incrementally fix
        // only the escaped particles (which may land in cells that don't
        // exist yet, or cross multiple cell boundaries). The cost is
        // O(N * tree_depth) which is modest since tree_depth is typically
        // 5-15 levels.
        // ==================================================================
        reassign_all_particle_keys( positions, num_local_particles );

        // ==================================================================
        // Step 5: Recount particles per leaf cell (global)
        // ==================================================================

        // Local counts via device kernel
        Kokkos::View<int*, memory_space> d_leaf_local_counts(
            "d_leaf_local_counts", num_leaves );

        auto pk = particle_keys_;
        Kokkos::parallel_for(
            "RecountLocal",
            Kokkos::RangePolicy<execution_space>( 0, num_local_particles ),
            KOKKOS_LAMBDA( int i ) {
                MortonKey my_key = pk( i );
                auto idx = leaf_map.find( my_key );
                if ( leaf_map.valid_at( idx ) )
                {
                    int j = leaf_map.value_at( idx );
                    Kokkos::atomic_increment( &d_leaf_local_counts( j ) );
                }
                // Particles assigned to internal nodes (empty-child case)
                // are handled in the refinement pass below.
            } );
        Kokkos::fence();

        auto h_leaf_local =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{},
                                                d_leaf_local_counts );

        // Also count particles assigned to internal (non-leaf) nodes.
        // These are particles that landed in previously empty octants.
        // We count them per cell on the host side.
        auto h_pkeys = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, particle_keys_ );

        std::unordered_map<MortonKey, int> internal_local_counts;
        for ( int i = 0; i < num_local_particles; ++i )
        {
            MortonKey k = h_pkeys( i );
            auto it = cell_lookup_.find( k );
            if ( it != cell_lookup_.end() && !cells_[it->second].is_leaf )
            {
                internal_local_counts[k]++;
            }
        }

        // Allreduce leaf counts
        std::vector<int> leaf_send( num_leaves ), leaf_global( num_leaves );
        for ( int j = 0; j < num_leaves; ++j )
            leaf_send[j] = h_leaf_local( j );

        MPI_Allreduce( leaf_send.data(), leaf_global.data(), num_leaves,
                    MPI_INT, MPI_SUM, comm_ );

        // Update leaf global_count in cells_
        for ( int j = 0; j < num_leaves; ++j )
        {
            auto it = cell_lookup_.find( leaf_keys_vec[j] );
            if ( it != cell_lookup_.end() )
                cells_[it->second].global_count = leaf_global[j];
        }

        // Allreduce internal node counts (only the ones that have particles
        // landing in them — typically very few)
        // Gather all internal keys that have local counts > 0
        std::vector<MortonKey> internal_keys_with_particles;
        std::vector<int> internal_local_vec;
        for ( auto& [k, cnt] : internal_local_counts )
        {
            internal_keys_with_particles.push_back( k );
            internal_local_vec.push_back( cnt );
        }

        // For simplicity, broadcast the set of keys and allreduce counts.
        // In practice this is a very small set.
        // We use a simple gather-and-merge approach.
        int num_internal_local =
            static_cast<int>( internal_keys_with_particles.size() );
        int num_internal_max = 0;
        MPI_Allreduce( &num_internal_local, &num_internal_max, 1, MPI_INT,
                    MPI_MAX, comm_ );

        // If any rank has particles in internal nodes, we need to handle it.
        // For now, a simple approach: allgather the keys and counts, then
        // merge. This is fine because this set is tiny.
        if ( num_internal_max > 0 )
        {
            // Pad local arrays to uniform size for allgather
            std::vector<MortonKey> padded_keys( num_internal_max, 0 );
            std::vector<int> padded_counts( num_internal_max, 0 );
            for ( int j = 0; j < num_internal_local; ++j )
            {
                padded_keys[j] = internal_keys_with_particles[j];
                padded_counts[j] = internal_local_vec[j];
            }

            std::vector<MortonKey> all_keys(
                num_internal_max * nprocs_ );
            std::vector<int> all_counts( num_internal_max * nprocs_ );

            MPI_Allgather( padded_keys.data(), num_internal_max,
                        MPI_UINT64_T, all_keys.data(),
                        num_internal_max, MPI_UINT64_T, comm_ );
            MPI_Allgather( padded_counts.data(), num_internal_max, MPI_INT,
                        all_counts.data(), num_internal_max, MPI_INT,
                        comm_ );

            // Merge
            std::unordered_map<MortonKey, int> internal_global_counts;
            for ( int j = 0; j < num_internal_max * nprocs_; ++j )
            {
                if ( all_keys[j] != 0 )
                    internal_global_counts[all_keys[j]] += all_counts[j];
            }

            // Update cells_ with these counts
            for ( auto& [k, cnt] : internal_global_counts )
            {
                auto it = cell_lookup_.find( k );
                if ( it != cell_lookup_.end() )
                    cells_[it->second].global_count = cnt;
            }
        }

        // ==================================================================
        // Step 6: Refine leaves that now exceed ncrit
        // ==================================================================
        bool changed = true;
        while ( changed )
        {
            changed = false;

            // Collect leaves that need refinement
            std::vector<MortonKey> to_refine;
            for ( const auto& ci : cells_ )
            {
                if ( ci.is_leaf && ci.global_count > ncrit_ &&
                    ci.depth < max_depth_ )
                {
                    to_refine.push_back( ci.key );
                }
            }

            for ( auto k : to_refine )
            {
                refine_leaf( k );
                result.cells_refined++;
                changed = true;
            }

            if ( !changed )
                break;

            // After refinement, reassign particles and recount to see if
            // new children also need splitting
            rebuild_cell_lookup();
            reassign_all_particle_keys( positions, num_local_particles );

            // Recount for newly created leaves
            // Reset all leaf counts to 0
            for ( auto& ci : cells_ )
            {
                if ( ci.is_leaf )
                    ci.global_count = 0;
            }

            auto h_pk = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace{}, particle_keys_ );

            // Local counts
            std::unordered_map<MortonKey, int> local_leaf_counts;
            for ( int i = 0; i < num_local_particles; ++i )
                local_leaf_counts[h_pk( i )]++;

            // Allreduce — gather all unique keys across ranks
            std::vector<MortonKey> count_keys;
            std::vector<int> count_vals;
            for ( auto& [k, v] : local_leaf_counts )
            {
                count_keys.push_back( k );
                count_vals.push_back( v );
            }

            // Use allgather + merge (same small-set approach)
            int num_local_keys = static_cast<int>( count_keys.size() );
            int num_max_keys = 0;
            MPI_Allreduce( &num_local_keys, &num_max_keys, 1, MPI_INT,
                        MPI_MAX, comm_ );

            std::vector<MortonKey> pk_padded( num_max_keys, 0 );
            std::vector<int> pv_padded( num_max_keys, 0 );
            for ( int j = 0; j < num_local_keys; ++j )
            {
                pk_padded[j] = count_keys[j];
                pv_padded[j] = count_vals[j];
            }

            std::vector<MortonKey> all_pk( num_max_keys * nprocs_ );
            std::vector<int> all_pv( num_max_keys * nprocs_ );

            MPI_Allgather( pk_padded.data(), num_max_keys, MPI_UINT64_T,
                        all_pk.data(), num_max_keys, MPI_UINT64_T,
                        comm_ );
            MPI_Allgather( pv_padded.data(), num_max_keys, MPI_INT,
                        all_pv.data(), num_max_keys, MPI_INT, comm_ );

            std::unordered_map<MortonKey, int> global_leaf_counts;
            for ( int j = 0; j < num_max_keys * nprocs_; ++j )
            {
                if ( all_pk[j] != 0 )
                    global_leaf_counts[all_pk[j]] += all_pv[j];
            }

            for ( auto& [k, cnt] : global_leaf_counts )
            {
                auto it = cell_lookup_.find( k );
                if ( it != cell_lookup_.end() )
                    cells_[it->second].global_count = cnt;
            }
        }

        // ==================================================================
        // Step 7: Coarsen — check if sibling groups can be merged
        // ==================================================================
        // Collect unique parent keys of all leaves
        std::unordered_set<MortonKey> candidate_parents;
        for ( const auto& ci : cells_ )
        {
            if ( ci.is_leaf && ci.key != ROOT_KEY )
                candidate_parents.insert( parent_key( ci.key ) );
        }

        for ( auto pk_val : candidate_parents )
        {
            if ( try_coarsen( pk_val ) )
                result.cells_coarsened++;
        }

        if ( result.cells_coarsened > 0 )
        {
            // Remove tombstoned cells and rebuild lookup
            cells_.erase(
                std::remove_if( cells_.begin(), cells_.end(),
                                []( const CellInfo& ci ) {
                                    return ci.key == 0;
                                } ),
                cells_.end() );
            rebuild_cell_lookup();

            // Reassign particle keys one final time after coarsening
            reassign_all_particle_keys( positions, num_local_particles );
        }

        return result;
    }

};

    

template <class MemorySpace, class ExecutionSpace, class Metadata, 
          std::size_t CellPerTileDim, std::size_t ExpansionCutoff>
std::shared_ptr<Solver<MemorySpace, ExecutionSpace, Metadata, CellPerTileDim, ExpansionCutoff>>
        createSolver( const std::array<typename Metadata::scalar_type, 3>& global_low_corner,
                    const std::array<typename Metadata::scalar_type, 3>& global_high_corner,
                    const std::size_t leaf_tiles_per_dim,
                    const std::size_t tile_reduction_factor,
                    MPI_Comm comm)
{
    return std::make_shared<Solver<MemorySpace, ExecutionSpace, Metadata, CellPerTileDim, ExpansionCutoff>>(global_low_corner,
            global_high_corner, leaf_tiles_per_dim, tile_reduction_factor,
            comm);
}

} // end namespace Experimental

} // end namespace Canopy

#endif // CANOPY_SOLVER_HPP
