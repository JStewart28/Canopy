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

#ifndef CANOPY_TREE_BUILDER_HPP
#define CANOPY_TREE_BUILDER_HPP

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Canopy
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
    int global_count; // total particles across all ranks that live in this cell
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
    int particles_migrated; // particles that moved to a different leaf
    int cells_refined;      // leaf cells that were split
    int cells_coarsened;    // groups of siblings collapsed to parent
    bool full_rebuild_done; // true if the bounding box changed and a
                            // full rebuild was triggered instead
};

template <class MemorySpace, class ExecutionSpace>
class TreeBuilder
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;

    // Host mirror types for tree data
    using host_execution_space = Kokkos::DefaultHostExecutionSpace;
    using host_memory_space = typename host_execution_space::memory_space;

    // The particle-to-leaf mapping lives on device for fast particle kernels
    using key_view_type = Kokkos::View<MortonKey*, memory_space>;
    using key_host_view_type = Kokkos::View<MortonKey*, host_memory_space>;

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
    std::array<double, 3> _bb_tf;
    //! Tolerance factor on gncrit
    double _ncrit_tf;

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
    TreeBuilder( MPI_Comm comm, const int ncrit, const int max_depth,
                 const std::array<double, 3> bb_tolerance_factor,
                 const double ncrit_tolerance_factor = 0.1 )
        : _ncrit( ncrit )
        , _max_depth( max_depth )
        , _comm( comm )
        , _bb_tf( bb_tolerance_factor )
        , _ncrit_tf( ncrit_tolerance_factor )
    {
        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );

        // Check that max depth is not greater than Morton key storage size
        if ( _max_depth > 19 )
        {
            throw std::runtime_error(
                "Canopy::TreeBuilder only supports depths up to 20!" );
        }
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
                             double cy, double cz );

    // Compute child cell center given parent center, half_width, and octant
    KOKKOS_INLINE_FUNCTION
    static void child_center( double parent_cx, double parent_cy,
                              double parent_cz, double parent_hw, int octant,
                              double& cx, double& cy, double& cz );

    // Compute global bounding box from distributed particles
    template <class PositionType>
    BoundingBox compute_global_bounding_box( PositionType positions,
                                             int num_local_particles );

    // Rebuild cell_lookup map of cell morton key to index into cells vector.
    void rebuild_cell_lookup();

    // Walk each particle from the root of the tree down to its correct leaf.
    // This runs on host because the tree structure is in _cells and
    // _cell_lookup which are on the host. The resulting leaf keys are then
    // copied back to the device view.
    template <class PositionType>
    void reassign_all_particle_keys( PositionType positions,
                                     int num_local_particles );

    // Merge 8 sibling leaves back into their parent only if the parent will
    // have at least than ncrit*(1 - nc_tf) particles after merge.
    // Return true if coarsened.
    bool try_coarsen( MortonKey parent_key_val );

    // Split a leaf cell into an internal cell + 8 children
    void refine_leaf( MortonKey leaf_key );

    // -----------------------------------------------------------------------
    // Main functions
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
    template <class PositionType>
    void build( PositionType positions, int num_local_particles );

    // Check if particles have left the root bounding box
    template <class PositionType>
    bool needs_rebuild( PositionType positions, int num_local_particles ) const;

    // update() — incremental tree adaptation
    template <class PositionType>
    UpdateResult update( PositionType positions, int num_local_particles );
};

// ============================================================================
// Implementation
// ============================================================================

template <class MemorySpace, class ExecutionSpace>
KOKKOS_INLINE_FUNCTION int
TreeBuilder<MemorySpace, ExecutionSpace>::which_octant( double px, double py,
                                                        double pz, double cx,
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

template <class MemorySpace, class ExecutionSpace>
KOKKOS_INLINE_FUNCTION void
TreeBuilder<MemorySpace, ExecutionSpace>::child_center(
    double parent_cx, double parent_cy, double parent_cz, double parent_hw,
    int octant, double& cx, double& cy, double& cz )
{
    double quarter = parent_hw * 0.5;
    cx = parent_cx + ( ( octant & 1 ) ? quarter : -quarter );
    cy = parent_cy + ( ( octant & 2 ) ? quarter : -quarter );
    cz = parent_cz + ( ( octant & 4 ) ? quarter : -quarter );
}

template <class MemorySpace, class ExecutionSpace>
template <class PositionType>
BoundingBox
TreeBuilder<MemorySpace, ExecutionSpace>::compute_global_bounding_box(
    PositionType positions, int num_local_particles )
{
    Kokkos::Array<double, 3> local_min, local_max;
    double inf = std::numeric_limits<double>::max();

    Kokkos::parallel_reduce(
        "ComputeLocalBBox",
        Kokkos::RangePolicy<execution_space>( 0, num_local_particles ),
        KOKKOS_LAMBDA( int i, double& lmin_x, double& lmin_y, double& lmin_z,
                       double& lmax_x, double& lmax_y, double& lmax_z ) {
            double px = positions( i, 0 );
            double py = positions( i, 1 );
            double pz = positions( i, 2 );
            if ( px < lmin_x )
                lmin_x = px;
            if ( py < lmin_y )
                lmin_y = py;
            if ( pz < lmin_z )
                lmin_z = pz;
            if ( px > lmax_x )
                lmax_x = px;
            if ( py > lmax_y )
                lmax_y = py;
            if ( pz > lmax_z )
                lmax_z = pz;
        },
        Kokkos::Min<double>( local_min[0] ),
        Kokkos::Min<double>( local_min[1] ),
        Kokkos::Min<double>( local_min[2] ),
        Kokkos::Max<double>( local_max[0] ),
        Kokkos::Max<double>( local_max[1] ),
        Kokkos::Max<double>( local_max[2] ) );

    // Copy Kokkos arrays into c-style arrays
    double lmin[3], lmax[3];
    for ( int i = 0; i < 3; i++ )
    {
        lmin[i] = local_min[i];
        lmax[i] = local_max[i];
    }

    BoundingBox box;
    MPI_Allreduce( lmin, box.min, 3, MPI_DOUBLE, MPI_MIN, _comm );
    MPI_Allreduce( lmax, box.max, 3, MPI_DOUBLE, MPI_MAX, _comm );

    // Guard against non-finite particle coordinates (e.g. a divergent
    // gravitational acceleration producing Inf/NaN positions). A non-finite
    // bounding box silently degenerates the Morton-key / cell geometry and
    // surfaces later as an opaque out-of-bounds GPU memory fault, so fail
    // loudly here with an actionable message instead.
    for ( int d = 0; d < 3; ++d )
    {
        if ( !std::isfinite( box.min[d] ) || !std::isfinite( box.max[d] ) )
        {
            std::fprintf(
                stderr,
                "[Canopy] FATAL: non-finite bounding box on axis %d "
                "(min=%g, max=%g). A particle coordinate is Inf/NaN — "
                "check force softening and timestep.\n",
                d, box.min[d], box.max[d] );
            std::fflush( stderr );
            MPI_Abort( _comm, 1 );
        }
    }

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

template <class MemorySpace, class ExecutionSpace>
void TreeBuilder<MemorySpace, ExecutionSpace>::rebuild_cell_lookup()
{
    _cell_lookup.clear();
    _cell_lookup.reserve( _cells.size() );
    for ( int i = 0; i < static_cast<int>( _cells.size() ); ++i )
        _cell_lookup[_cells[i].key] = i;
}

template <class MemorySpace, class ExecutionSpace>
template <class PositionType>
void TreeBuilder<MemorySpace, ExecutionSpace>::reassign_all_particle_keys(
    PositionType positions, int num_local_particles )
{
    // Workaround to copy a device-side slice into a host-side view
    using value_type = typename decltype( positions )::value_type;
    Kokkos::View<value_type* [3], memory_space> d_pos( "d_pos",
                                                       num_local_particles );
    Kokkos::parallel_for(
        "SliceToView",
        Kokkos::RangePolicy<execution_space>( 0, num_local_particles ),
        KOKKOS_LAMBDA( int i ) {
            d_pos( i, 0 ) = positions( i, 0 );
            d_pos( i, 1 ) = positions( i, 1 );
            d_pos( i, 2 ) = positions( i, 2 );
        } );
    Kokkos::fence();
    auto h_positions =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_pos );

    // Allocate host keys
    key_host_view_type h_keys( "h_particle_keys", num_local_particles );

    // Find the root cell info
    auto root_it = _cell_lookup.find( ROOT_KEY );
    if ( root_it == _cell_lookup.end() )
    {
        // Tree has no root — shouldn't happen after build()
        Kokkos::deep_copy( _particle_keys, ROOT_KEY );
        return;
    }

    for ( int i = 0; i < num_local_particles; ++i )
    {
        double px = h_positions( i, 0 );
        double py = h_positions( i, 1 );
        double pz = h_positions( i, 2 );

        // Walk down from root
        MortonKey current = ROOT_KEY;
        while ( true )
        {
            auto it = _cell_lookup.find( current );
            if ( it == _cell_lookup.end() )
            {
                // Cell doesn't exist in the tree — walk back to parent.
                // This shouldn't happen after the fix below, but is a
                // safety fallback.
                current = parent_key( current );
                break;
            }

            const CellInfo& ci = _cells[it->second];
            if ( ci.is_leaf )
                break; // found the leaf

            // Determine which child octant this particle falls into
            int oct = which_octant( px, py, pz, ci.center[0], ci.center[1],
                                    ci.center[2] );
            MortonKey child = child_key( current, oct );

            // Check if this child exists
            auto child_it = _cell_lookup.find( child );
            if ( child_it == _cell_lookup.end() )
            {
                // Child was pruned (empty at build time) but a particle
                // has now moved into this octant. Create a new leaf cell
                // for it so the particle has a proper leaf assignment.
                CellInfo new_leaf;
                new_leaf.key = child;
                new_leaf.depth = ci.depth + 1;
                child_center( ci.center[0], ci.center[1], ci.center[2],
                              ci.half_width, oct, new_leaf.center[0],
                              new_leaf.center[1], new_leaf.center[2] );
                new_leaf.half_width = ci.half_width * 0.5;
                new_leaf.global_count = 0; // will be filled by recount
                new_leaf.is_leaf = true;

                _cell_lookup[child] = static_cast<int>( _cells.size() );
                _cells.push_back( new_leaf );

                current = child;
                break; // new leaf — particle goes here
            }

            current = child;
        }

        h_keys( i ) = current;
    }

    // Copy back to device
    if ( static_cast<int>( _particle_keys.extent( 0 ) ) != num_local_particles )
    {
        _particle_keys = key_view_type( "particle_keys", num_local_particles );
    }
    Kokkos::deep_copy( _particle_keys, h_keys );
}

template <class MemorySpace, class ExecutionSpace>
bool TreeBuilder<MemorySpace, ExecutionSpace>::try_coarsen(
    MortonKey parent_key_val )
{
    // Can't coarsen root cell
    if ( parent_key_val < ROOT_KEY )
        return false;

    // Cell not in _cells
    auto parent_it = _cell_lookup.find( parent_key_val );
    if ( parent_it == _cell_lookup.end() )
        return false;

    CellInfo& parent_ci = _cells[parent_it->second];

    // Parent must currently be a non-leaf cell
    if ( parent_ci.is_leaf )
        return false;

    // All 8 children must exist and be leaves
    int total_count = 0;
    std::vector<MortonKey> child_keys_to_remove;

    for ( int oct = 0; oct < 8; oct++ )
    {
        MortonKey ck = child_key( parent_key_val, oct );
        auto child_it = _cell_lookup.find( ck );
        if ( child_it == _cell_lookup.end() )
        {
            // Child doesn't exist (was pruned). It had 0
            // particles. We can still coarsen if we want.
            continue;
        }

        const CellInfo& child_ci = _cells[child_it->second];
        if ( !child_ci.is_leaf )
            return false; // can't coarsen if any child is internal

        total_count += child_ci.global_count;
        child_keys_to_remove.push_back( ck );
    }

    // Only coarsen if the combined count falls below the lower
    // hysteresis threshold. This prevents thrashing: a cell that was
    // just split won't immediately re-merge if a few particles leave.
    int coarsen_threshold = static_cast<int>( _ncrit * ( 1.0 - _ncrit_tf ) );
    if ( total_count > coarsen_threshold )
        return false;

    // Remove children from lookup (mark for lazy cleanup)
    // Don't actually erase from _cells to avoid invalidating indices.
    // Instead mark them with key=0 and rebuild the lookup later.
    for ( auto ck : child_keys_to_remove )
    {
        auto child_it = _cell_lookup.find( ck );
        if ( child_it != _cell_lookup.end() )
        {
            _cells[child_it->second].key = 0;
            _cell_lookup.erase( child_it );
        }
    }

    // Convert parent back to leaf
    parent_ci.is_leaf = true;
    parent_ci.global_count = total_count;

    return true;
}

template <class MemorySpace, class ExecutionSpace>
void TreeBuilder<MemorySpace, ExecutionSpace>::refine_leaf( MortonKey leaf_key )
{
    auto it = _cell_lookup.find( leaf_key );
    if ( it == _cell_lookup.end() )
        return;

    int idx = it->second;
    CellInfo& ci = _cells[idx];

    if ( !ci.is_leaf )
        return; // already internal

    if ( ci.depth >= _max_depth )
        return; // can't refine further

    // Convert to internal cell
    ci.is_leaf = false;

    // Create 8 children
    double parent_cx = ci.center[0];
    double parent_cy = ci.center[1];
    double parent_cz = ci.center[2];
    double parent_hw = ci.half_width;
    double ch_hw = parent_hw * 0.5;

    for ( int oct = 0; oct < 8; oct++ )
    {
        CellInfo child_ci;
        child_ci.key = child_key( leaf_key, oct );
        child_ci.depth = ci.depth + 1;
        child_center( parent_cx, parent_cy, parent_cz, parent_hw, oct,
                      child_ci.center[0], child_ci.center[1],
                      child_ci.center[2] );
        child_ci.half_width = ch_hw;
        child_ci.global_count = 0; // will be filled by recount
        child_ci.is_leaf = true;

        _cell_lookup[child_ci.key] = static_cast<int>( _cells.size() );
        _cells.push_back( child_ci );
    }
}

template <class MemorySpace, class ExecutionSpace>
template <class PositionType>
void TreeBuilder<MemorySpace, ExecutionSpace>::build( PositionType positions,
                                                      int num_local_particles )
{
    // Compute global bounding box
    _root_box = compute_global_bounding_box( positions, num_local_particles );

    // Expand the root box by the tolerance factor so particles have room
    // to move before leaving the domain.
    for ( int d = 0; d < 3; ++d )
    {
        double tol = _bb_tf[d];
        if ( tol > 0.0 )
        {
            double width = _root_box.max[d] - _root_box.min[d];
            double expansion = tol * width;
            _root_box.min[d] -= expansion;
            _root_box.max[d] += expansion;
        }
    }

    // Coordinates of root box center
    double root_cx = 0.5 * ( _root_box.min[0] + _root_box.max[0] );
    double root_cy = 0.5 * ( _root_box.min[1] + _root_box.max[1] );
    double root_cz = 0.5 * ( _root_box.min[2] + _root_box.max[2] );

    // Use the max dimension to compute the half-width of the root box
    double root_hw = 0.0;
    for ( int d = 0; d < 3; ++d )
    {
        double hw = 0.5 * ( _root_box.max[d] - _root_box.min[d] );
        if ( hw > root_hw )
            root_hw = hw;
    }

    // Initialize particle-to-cell mapping on device
    _particle_keys = key_view_type( "particle_keys", num_local_particles );
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

    // Iteratively refine the root box until all cells have less than ncrit
    // particles
    for ( int depth = 0; depth <= _max_depth; depth++ )
    {
        if ( cells_to_refine.empty() )
            break;

        int num_candidates = static_cast<int>( cells_to_refine.size() );

        // Move candidate cells into views
        Kokkos::View<MortonKey*, memory_space> cand_keys( "cand_keys",
                                                          num_candidates );
        Kokkos::View<double* [3], memory_space> cand_centers( "cand_centers",
                                                              num_candidates );
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

        Kokkos::View<int*, memory_space> particle_octant( "particle_octant",
                                                          num_local_particles );
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
                Kokkos::atomic_increment( &local_octant_counts( c, oct ) );
            } );
        Kokkos::fence();

        auto h_local_counts = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), local_counts );
        std::vector<int> global_counts( num_candidates );

        // All reduce the number of particles in each cell.
        MPI_Allreduce( h_local_counts.data(), global_counts.data(),
                       num_candidates, MPI_INT, MPI_SUM, _comm );

        std::vector<int> should_split( num_candidates, 0 );
        std::vector<Cell> next_cells_to_refine;

        // Loop over candidate cells. If there are globally greater than
        // 'ncrit' particles in a candidate cell, refine the cell.
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
                for ( int oct = 0; oct < 8; oct++ )
                {
                    Cell cell;
                    cell.key = child_key( cells_to_refine[c].key, oct );
                    child_center( parent_cx, parent_cy, parent_cz, parent_hw,
                                  oct, cell.center[0], cell.center[1],
                                  cell.center[2] );
                    cell.half_width = ch_hw;
                    next_cells_to_refine.push_back( cell );
                }
            }
        }

        // Update particle keys to be in a child cell if the cell was split.
        Kokkos::View<int*, memory_space> splits( "splits", num_candidates );
        auto h_splits = Kokkos::create_mirror_view( splits );
        for ( int c = 0; c < num_candidates; ++c )
            h_splits( c ) = should_split[c];
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

        // Move next_cells_to_refine into cells_to_refine for next
        // iteration.
        cells_to_refine = std::move( next_cells_to_refine );

    } // end tree build loop

    // Build the host-side lookup map
    rebuild_cell_lookup();

    _tree_valid = true;
}

template <class MemorySpace, class ExecutionSpace>
template <class PositionType>
bool TreeBuilder<MemorySpace, ExecutionSpace>::needs_rebuild(
    PositionType positions, int num_local_particles ) const
{
    // Check against the root box
    double bmin0 = _root_box.min[0], bmin1 = _root_box.min[1],
           bmin2 = _root_box.min[2];
    double bmax0 = _root_box.max[0], bmax1 = _root_box.max[1],
           bmax2 = _root_box.max[2];

    int local_escaped = 0;
    Kokkos::parallel_reduce(
        "CheckBounds",
        Kokkos::RangePolicy<execution_space>( 0, num_local_particles ),
        KOKKOS_LAMBDA( int i, int& count ) {
            double px = positions( i, 0 );
            double py = positions( i, 1 );
            double pz = positions( i, 2 );
            if ( px < bmin0 || px > bmax0 || py < bmin1 || py > bmax1 ||
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

template <class MemorySpace, class ExecutionSpace>
template <class PositionType>
UpdateResult
TreeBuilder<MemorySpace, ExecutionSpace>::update( PositionType positions,
                                                  int num_local_particles )
{
    // Particles moved to different leaf
    // Leaf cells split
    // Cells coarsened
    // Full rebuild due to bounding box change
    UpdateResult result = { 0, 0, 0, false };

    // Step 0: If the tree was never built, do a full build
    if ( !_tree_valid )
    {
        build( positions, num_local_particles );
        result.full_rebuild_done = true;
        return result;
    }

    // Step 1: Check if any particles escaped the root bounding box, which
    // also requires a full rebuild
    if ( needs_rebuild( positions, num_local_particles ) )
    {
        build( positions, num_local_particles );
        result.full_rebuild_done = true;
        return result;
    }

    // --------------------------------------------------------------------------
    // Step 2: Detect which particles have left their current leaf cell.
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
    Kokkos::UnorderedMap<MortonKey, int, memory_space> leaf_map( num_leaves *
                                                                 2 );

    Kokkos::View<MortonKey*, memory_space> d_leaf_keys( "d_leaf_keys",
                                                        num_leaves );
    Kokkos::View<double*, memory_space> d_leaf_cx( "d_leaf_cx", num_leaves );
    Kokkos::View<double*, memory_space> d_leaf_cy( "d_leaf_cy", num_leaves );
    Kokkos::View<double*, memory_space> d_leaf_cz( "d_leaf_cz", num_leaves );
    Kokkos::View<double*, memory_space> d_leaf_hw( "d_leaf_hw", num_leaves );

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
        KOKKOS_LAMBDA( int j ) { leaf_map.insert( d_leaf_keys( j ), j ); } );
    Kokkos::fence();

    // Check for each particle if it has moved into a new cell.
    // Flag these and count them.
    Kokkos::View<int*, memory_space> escaped_flag( "escaped_flag",
                                                   num_local_particles );
    auto particle_keys = _particle_keys;

    int local_migrated = 0;
    Kokkos::parallel_reduce(
        "DetectEscaped",
        Kokkos::RangePolicy<execution_space>( 0, num_local_particles ),
        KOKKOS_LAMBDA( int i, int& count ) {
            MortonKey my_key = particle_keys( i );
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

            double px = positions( i, 0 );
            double py = positions( i, 1 );
            double pz = positions( i, 2 );

            // Check if outside its cell
            if ( px < cx - hw || px > cx + hw || py < cy - hw || py > cy + hw ||
                 pz < cz - hw || pz > cz + hw )
            {
                escaped_flag( i ) = 1;
                count++;
            }
        },
        local_migrated );
    Kokkos::fence();

    // Global count of migrated particles (for reporting)
    int global_migrated = 0;
    MPI_Allreduce( &local_migrated, &global_migrated, 1, MPI_INT, MPI_SUM,
                   _comm );
    result.particles_migrated = global_migrated;

    // Step 3: If no particles migrated, return. The tree is unchanged.
    if ( global_migrated == 0 )
        return result;

    // Step 4: Otherwise, reassign all particle keys by walking the tree.
    // The cost is O(N * tree_depth)
    reassign_all_particle_keys( positions, num_local_particles );

    // ==================================================================
    // Step 5: Recount particles per leaf cell (global)
    //
    // Since reassign_all_particle_keys may have created new leaf cells
    // (for previously empty octants that now have particles), do the
    // recount on the host side using the authoritative _cell_lookup.
    // ==================================================================

    // Reset all leaf counts
    for ( auto& ci : _cells )
    {
        if ( ci.is_leaf )
            ci.global_count = 0;
    }

    auto h_pkeys = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                        _particle_keys );

    // Local counts per cell key
    std::unordered_map<MortonKey, int> local_counts_map;
    for ( int i = 0; i < num_local_particles; ++i )
        local_counts_map[h_pkeys( i )]++;

    // Gather all unique keys and their local counts for allreduce.
    // Use allgather + merge since the number of unique keys is
    // bounded by the (modest) number of leaf cells.
    std::vector<MortonKey> count_keys;
    std::vector<int> count_vals;
    for ( auto& [k, v] : local_counts_map )
    {
        count_keys.push_back( k );
        count_vals.push_back( v );
    }

    int num_local_keys = static_cast<int>( count_keys.size() );
    int num_max_keys = 0;
    MPI_Allreduce( &num_local_keys, &num_max_keys, 1, MPI_INT, MPI_MAX, _comm );

    std::vector<MortonKey> pk_padded( num_max_keys, 0 );
    std::vector<int> pv_padded( num_max_keys, 0 );
    for ( int j = 0; j < num_local_keys; ++j )
    {
        pk_padded[j] = count_keys[j];
        pv_padded[j] = count_vals[j];
    }

    std::vector<MortonKey> all_pk( num_max_keys * _comm_size );
    std::vector<int> all_pv( num_max_keys * _comm_size );

    MPI_Allgather( pk_padded.data(), num_max_keys, MPI_UINT64_T, all_pk.data(),
                   num_max_keys, MPI_UINT64_T, _comm );
    MPI_Allgather( pv_padded.data(), num_max_keys, MPI_INT, all_pv.data(),
                   num_max_keys, MPI_INT, _comm );

    std::unordered_map<MortonKey, int> global_counts_map;
    for ( int j = 0; j < num_max_keys * _comm_size; ++j )
    {
        if ( all_pk[j] != 0 )
            global_counts_map[all_pk[j]] += all_pv[j];
    }

    for ( auto& [k, cnt] : global_counts_map )
    {
        auto it = _cell_lookup.find( k );
        if ( it != _cell_lookup.end() )
            _cells[it->second].global_count = cnt;
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
        for ( const auto& ci : _cells )
        {
            if ( ci.is_leaf && ci.global_count > _ncrit &&
                 ci.depth < _max_depth )
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
        for ( auto& ci : _cells )
        {
            if ( ci.is_leaf )
                ci.global_count = 0;
        }

        auto h_pk = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                         _particle_keys );

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
        MPI_Allreduce( &num_local_keys, &num_max_keys, 1, MPI_INT, MPI_MAX,
                       _comm );

        std::vector<MortonKey> pk_padded( num_max_keys, 0 );
        std::vector<int> pv_padded( num_max_keys, 0 );
        for ( int j = 0; j < num_local_keys; ++j )
        {
            pk_padded[j] = count_keys[j];
            pv_padded[j] = count_vals[j];
        }

        std::vector<MortonKey> all_pk( num_max_keys * _comm_size );
        std::vector<int> all_pv( num_max_keys * _comm_size );

        MPI_Allgather( pk_padded.data(), num_max_keys, MPI_UINT64_T,
                       all_pk.data(), num_max_keys, MPI_UINT64_T, _comm );
        MPI_Allgather( pv_padded.data(), num_max_keys, MPI_INT, all_pv.data(),
                       num_max_keys, MPI_INT, _comm );

        std::unordered_map<MortonKey, int> global_leaf_counts;
        for ( int j = 0; j < num_max_keys * _comm_size; ++j )
        {
            if ( all_pk[j] != 0 )
                global_leaf_counts[all_pk[j]] += all_pv[j];
        }

        for ( auto& [k, cnt] : global_leaf_counts )
        {
            auto it = _cell_lookup.find( k );
            if ( it != _cell_lookup.end() )
                _cells[it->second].global_count = cnt;
        }
    }

    // ==================================================================
    // Coarsen — check if sibling groups can be merged into a
    // single parent cell
    // ==================================================================

    // Collect unique parent keys of all leaves
    std::unordered_set<MortonKey> candidate_parents;
    for ( const auto& ci : _cells )
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
        // Remove zero-particle cells and rebuild lookup
        _cells.erase( std::remove_if( _cells.begin(), _cells.end(),
                                      []( const CellInfo& ci )
                                      { return ci.key == 0; } ),
                      _cells.end() );
        rebuild_cell_lookup();

        // Reassign particle keys one final time after coarsening
        reassign_all_particle_keys( positions, num_local_particles );
    }

    return result;
}

} // end namespace Canopy

#endif // CANOPY_TREE_BUILDER_HPP
