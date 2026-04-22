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

#ifndef CANOPY_UPWARD_SWEEP_HPP
#define CANOPY_UPWARD_SWEEP_HPP

#include "Canopy_CommunicationPlan.hpp"
#include "Canopy_SphericalCoefficients.hpp"
#include "Canopy_TreeBuilder.hpp"
#include "Canopy_TreePartitioner.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <unordered_map>
#include <vector>

namespace Canopy
{

// ============================================================================
// UpwardSweep
//
// Performs the FMM upward sweep: P2M at leaves followed by M2M from
// leaves up to the root. Uses the communication plans precomputed by
// CommunicationPlan to exchange multipole coefficients between ranks
// at layer boundaries where parent and child have different owners.
//
// Parallelism: Kokkos hierarchical (TeamPolicy).
//   - Leagues iterate over cells at the current layer owned/shared by
//     this rank.
//   - Within each team, threads cooperate over (n, m) output coefficients.
//   - P2M parallelizes particles within each leaf at the team level.
//
// Template Parameters:
//   MemorySpace, ExecutionSpace   - Kokkos device type
//   KernelType   - e.g. LaplaceKernel<double, 10>
// ============================================================================

template <class MemorySpace, class ExecutionSpace, class KernelType>
class UpwardSweep
{
  public:
    using execution_space = ExecutionSpace;
    using memory_space = MemorySpace;

    using scalar_type = typename KernelType::scalar_type;
    using complex_type = typename KernelType::complex_type;

    static constexpr int P = KernelType::max_order;
    static constexpr int coeffs_per_cell =
        KernelType::num_coeffs_per_cell;

    // Coefficient storage:
    //   _multipoles(cell_idx, coeff_idx) for cell_idx = 0 .. num_cells - 1
    //   coeff_idx encodes (n, m) with m >= 0, size = coeffs_per_cell
    using coeff_view_type =
        Kokkos::View<complex_type**, memory_space>;

    // A_{n,m} table
    using a_view_type = Kokkos::View<scalar_type*, memory_space>;

    // Cell metadata on device (center + leaf flag + depth + index)
    struct DeviceCellInfo
    {
        MortonKey key;
        int depth;
        int cell_idx;     // index into the flat coefficient view
        int owner_rank;   // owner rank, or OWNER_SHARED
        scalar_type center[3];
        scalar_type half_width;
        bool is_leaf;
    };
    using cell_view_type =
        Kokkos::View<DeviceCellInfo*, memory_space>;

    // Particle-to-leaf mapping: for each local particle, the INDEX
    // (into _device_cells) of its leaf cell. This is a device view so
    // it can be used directly in Kokkos kernels.
    using particle_cell_idx_view_type =
        Kokkos::View<int*, memory_space>;

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------
    UpwardSweep( MPI_Comm comm )
        : _comm( comm )
    {
        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_nprocs );
    }

    // -----------------------------------------------------------------------
    // setup()
    //
    // Allocate coefficient storage and build device-side cell metadata.
    // Call after the tree is built/partitioned and before execute().
    //
    // Parameters:
    //   cells         - global cell list from TopDownTreeBuilder
    //   owner_map     - cell_key -> owner rank from TreePartitioner
    //   particle_keys - device view mapping local particle index to its
    //                   leaf cell Morton key (from TreeBuilder)
    // -----------------------------------------------------------------------
    void setup(
        const std::vector<CellInfo>& cells,
        const std::unordered_map<MortonKey, int>& owner_map,
        const Kokkos::View<MortonKey*, memory_space>& particle_keys,
        int num_local_particles );

    // -----------------------------------------------------------------------
    // execute()
    //
    // Run the full upward sweep:
    //   1. Zero out all multipole coefficients.
    //   2. P2M for each leaf cell this rank owns.
    //   3. Layer-by-layer M2M from max_depth down to root.
    //   4. Between layers, exchange multipoles per the CommunicationPlan.
    //
    // Parameters:
    //   particle_charges - device view, charge for each local particle
    //   particle_positions - Cabana slice, positions for each particle
    //   comm_plan - precomputed communication plan
    // -----------------------------------------------------------------------
    template <class ChargeView, class PositionType>
    void execute(
        const ChargeView& particle_charges,
        const PositionType& particle_positions,
        const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan );

    // Access the computed multipole coefficients (after execute())
    const coeff_view_type& multipoles() const { return _multipoles; }

    // Look up the cell index for a given Morton key (host-side)
    int cell_index( MortonKey key ) const
    {
        auto it = _key_to_cell_idx.find( key );
        return ( it != _key_to_cell_idx.end() ) ? it->second : -1;
    }

    // Access the A_{n,m} table
    const a_view_type& A_table() const { return _A_table; }

  private:
    MPI_Comm _comm;
    int _rank;
    int _nprocs;

    // Multipole coefficients: shape (num_cells, coeffs_per_cell)
    coeff_view_type _multipoles;

    // A_{n,m} normalization table
    a_view_type _A_table;

    // Device-side cell info
    cell_view_type _device_cells;

    // Host-side: key -> index into _device_cells / _multipoles
    std::unordered_map<MortonKey, int> _key_to_cell_idx;

    // Host-side: lists of cell indices per depth, for each owner type:
    //   _cells_at_depth_local[d]  = cells at depth d owned by this rank
    //                               (and not shared)
    //   _leaves_at_depth_local[d] = subset of above that are leaves
    //   _internals_at_depth_local[d] = non-leaves this rank processes
    //                                  (includes SHARED ones; those
    //                                  need allreduce)
    std::vector<std::vector<int>> _leaves_at_depth_local;
    std::vector<std::vector<int>> _internals_at_depth_local;

    // Device mirror of per-depth cell-index lists, for kernel launches.
    // One View per depth.
    std::vector<Kokkos::View<int*, memory_space>> _d_leaves_at_depth;
    std::vector<Kokkos::View<int*, memory_space>> _d_internals_at_depth;

    // Particle leaf-cell index (into _device_cells).
    // Built from particle_keys during setup().
    particle_cell_idx_view_type _particle_cell_idx;

    // Number of local particles
    int _num_local_particles;

    int _max_depth;

  public:
    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    // Build particle_cell_idx on device from particle Morton keys.
    void build_particle_cell_idx(
        const Kokkos::View<MortonKey*, memory_space>& particle_keys,
        int num_local_particles );

    // P2M kernel for a single depth (over all leaves at that depth
    // owned by this rank).
    template <class ChargeView, class PositionType>
    void run_p2m_at_depth( int depth,
                           const ChargeView& particle_charges,
                           const PositionType& particle_positions );

    // M2M kernel for a single depth (over all internals at that depth
    // this rank processes — may include shared cells).
    void run_m2m_at_depth( int depth );

    // Exchange multipoles across ranks after M2M at a given depth.
    // Handles both point-to-point sends/receives (for uniquely-owned
    // cells below replication_depth) and allreduce (for shared cells
    // at or above replication_depth).
    void exchange_multipoles_at_depth(
        int depth,
        const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan );
};

// ============================================================================
// Implementation
// ============================================================================

// --------------------------------------------------------------------------
// setup
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace, class KernelType>
void UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::setup(
    const std::vector<CellInfo>& cells,
    const std::unordered_map<MortonKey, int>& owner_map,
    const Kokkos::View<MortonKey*, memory_space>& particle_keys,
    int num_local_particles )
{
    const int num_cells = static_cast<int>( cells.size() );
    _num_local_particles = num_local_particles;

    // Allocate multipole storage (cells x coeffs_per_cell)
    _multipoles = coeff_view_type( "multipoles", num_cells,
                                   coeffs_per_cell );

    // Build A_{n,m} table on device
    _A_table =
        build_A_coefficients<scalar_type, memory_space>( P );

    // Build host-side key lookup
    _key_to_cell_idx.clear();
    _key_to_cell_idx.reserve( num_cells );
    for ( int i = 0; i < num_cells; i++ )
        _key_to_cell_idx[cells[i].key] = i;

    // Compute max depth
    _max_depth = 0;
    for ( const auto& c : cells )
        if ( c.depth > _max_depth )
            _max_depth = c.depth;

    // Build device cell info
    _device_cells = cell_view_type( "device_cells", num_cells );
    auto h_device_cells = Kokkos::create_mirror_view( _device_cells );
    for ( int i = 0; i < num_cells; i++ )
    {
        const auto& c = cells[i];
        DeviceCellInfo dci;
        dci.key = c.key;
        dci.depth = c.depth;
        dci.cell_idx = i;
        auto it = owner_map.find( c.key );
        dci.owner_rank =
            ( it != owner_map.end() ) ? it->second : OWNER_SHARED;
        for ( int d = 0; d < 3; d++ )
            dci.center[d] = static_cast<scalar_type>( c.center[d] );
        dci.half_width = static_cast<scalar_type>( c.half_width );
        dci.is_leaf = c.is_leaf;
        h_device_cells( i ) = dci;
    }
    Kokkos::deep_copy( _device_cells, h_device_cells );

    // Build per-depth cell index lists
    _leaves_at_depth_local.assign( _max_depth + 1, {} );
    _internals_at_depth_local.assign( _max_depth + 1, {} );

    for ( int i = 0; i < num_cells; i++ )
    {
        const auto& c = cells[i];
        int owner = h_device_cells( i ).owner_rank;
        bool processes =
            ( owner == _rank || owner == OWNER_SHARED );
        if ( !processes )
            continue;

        if ( c.is_leaf )
            _leaves_at_depth_local[c.depth].push_back( i );
        else
            _internals_at_depth_local[c.depth].push_back( i );
    }

    // Upload per-depth lists to device
    _d_leaves_at_depth.clear();
    _d_internals_at_depth.clear();
    _d_leaves_at_depth.resize( _max_depth + 1 );
    _d_internals_at_depth.resize( _max_depth + 1 );

    for ( int d = 0; d <= _max_depth; d++ )
    {
        const int nleaves =
            static_cast<int>( _leaves_at_depth_local[d].size() );
        _d_leaves_at_depth[d] =
            Kokkos::View<int*, memory_space>( "leaves_at_depth", nleaves );
        if ( nleaves > 0 )
        {
            auto h = Kokkos::create_mirror_view( _d_leaves_at_depth[d] );
            for ( int j = 0; j < nleaves; j++ )
                h( j ) = _leaves_at_depth_local[d][j];
            Kokkos::deep_copy( _d_leaves_at_depth[d], h );
        }

        const int ninternals =
            static_cast<int>( _internals_at_depth_local[d].size() );
        _d_internals_at_depth[d] =
            Kokkos::View<int*, memory_space>( "internals_at_depth",
                                              ninternals );
        if ( ninternals > 0 )
        {
            auto h =
                Kokkos::create_mirror_view( _d_internals_at_depth[d] );
            for ( int j = 0; j < ninternals; j++ )
                h( j ) = _internals_at_depth_local[d][j];
            Kokkos::deep_copy( _d_internals_at_depth[d], h );
        }
    }

    // Build particle cell index
    build_particle_cell_idx( particle_keys, num_local_particles );
}

// --------------------------------------------------------------------------
// build_particle_cell_idx
//
// Map each particle's Morton key to the index into _device_cells /
// _multipoles. Uses a host-side std::unordered_map for the translation,
// then copies results to the device.
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace, class KernelType>
void UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::build_particle_cell_idx(
    const Kokkos::View<MortonKey*, memory_space>& particle_keys,
    int num_local_particles )
{
    _particle_cell_idx =
        particle_cell_idx_view_type( "particle_cell_idx",
                                     num_local_particles );

    auto h_keys = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), particle_keys );
    auto h_idx = Kokkos::create_mirror_view( _particle_cell_idx );

    for ( int i = 0; i < num_local_particles; i++ )
    {
        auto it = _key_to_cell_idx.find( h_keys( i ) );
        h_idx( i ) = ( it != _key_to_cell_idx.end() ) ? it->second : -1;
    }

    Kokkos::deep_copy( _particle_cell_idx, h_idx );
}

// --------------------------------------------------------------------------
// run_p2m_at_depth
//
// P2M at a single depth: each local particle whose leaf is at this depth
// contributes to that leaf's multipole coefficients via atomics.
//
// We iterate over particles rather than leaves because particles are the
// most balanced unit of work. Each particle performs one P2M contribution
// and uses atomic adds to accumulate into the leaf's coefficients.
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace, class KernelType>
template <class ChargeView, class PositionType>
void UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::run_p2m_at_depth(
    int depth,
    const ChargeView& particle_charges,
    const PositionType& particle_positions )
{
    // Capture references for lambdas
    auto multipoles = _multipoles;
    auto device_cells = _device_cells;
    auto particle_cell_idx = _particle_cell_idx;
    const int N = _num_local_particles;

    Kokkos::parallel_for(
        "P2M",
        Kokkos::RangePolicy<execution_space>( 0, N ),
        KOKKOS_LAMBDA( int p ) {
            const int cidx = particle_cell_idx( p );
            if ( cidx < 0 )
                return;

            const auto& dci = device_cells( cidx );
            if ( dci.depth != depth )
                return;
            if ( !dci.is_leaf )
                return;

            const scalar_type px =
                static_cast<scalar_type>( particle_positions( p, 0 ) );
            const scalar_type py =
                static_cast<scalar_type>( particle_positions( p, 1 ) );
            const scalar_type pz =
                static_cast<scalar_type>( particle_positions( p, 2 ) );

            const scalar_type dx = px - dci.center[0];
            const scalar_type dy = py - dci.center[1];
            const scalar_type dz = pz - dci.center[2];

            const scalar_type q =
                static_cast<scalar_type>( particle_charges( p ) );

            // Slice M_out = multipoles(cidx, :). We pass a subview to
            // the kernel P2M routine. Using Kokkos::subview keeps the
            // layout consistent.
            auto M_out = Kokkos::subview( multipoles, cidx, Kokkos::ALL );
            KernelType::p2m_contribution( q, dx, dy, dz, M_out );
        } );

    Kokkos::fence();
}

// --------------------------------------------------------------------------
// run_m2m_at_depth
//
// M2M at a single depth: for each internal cell C at this depth that
// this rank processes, accumulate translated contributions from C's
// (up to 8) existing children, which live at depth+1.
//
// Uses hierarchical parallelism:
//   - League: one team per cell C.
//   - Team threads: cooperate over output (j, k) coefficients of M_C.
//   - Children are iterated serially within the team (no atomics needed
//     because only one team writes to each M_C).
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace, class KernelType>
void UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::run_m2m_at_depth( int depth )
{
    const int ninternals =
        static_cast<int>( _internals_at_depth_local[depth].size() );
    if ( ninternals == 0 )
        return;

    auto multipoles = _multipoles;
    auto device_cells = _device_cells;
    auto A_table = _A_table;
    auto& d_internals = _d_internals_at_depth[depth];

    using team_policy = Kokkos::TeamPolicy<execution_space>;
    using team_member_type = typename team_policy::member_type;

    // Let Kokkos pick reasonable team/vector sizes
    team_policy policy( ninternals, Kokkos::AUTO );

    Kokkos::parallel_for(
        "M2M",
        policy,
        KOKKOS_LAMBDA( const team_member_type& team ) {
            const int league = team.league_rank();
            const int parent_cell = d_internals( league );
            const auto& parent_ci = device_cells( parent_cell );

            auto M_parent = Kokkos::subview( multipoles, parent_cell,
                                             Kokkos::ALL );

            // Loop over 8 possible children; if a child doesn't exist
            // its coefficients are just zero (its slot was never filled).
            // Rather than looking up each child by Morton key (expensive
            // on device), we rely on the caller to have zeroed the
            // multipoles of nonexistent cells. However, children whose
            // cells don't exist in cells_ have no index — so we need to
            // look up each child.
            //
            // Approach: walk the cell list linearly to find which cells
            // are children of this parent. To avoid the cost, we instead
            // precompute child-cell-index tables. For now, use a simple
            // brute-force search through device_cells (small arrays).
            // This is O(num_cells) per parent, which is acceptable for
            // modest trees but could be optimized later.

            const MortonKey pk = parent_ci.key;
            const int num_all = device_cells.extent( 0 );

            // Loop over cells; pick up any whose parent key == pk.
            for ( int ci = 0; ci < num_all; ci++ )
            {
                const auto& ccell = device_cells( ci );
                // parent key of ccell
                const MortonKey ccell_parent = ccell.key >> 3;
                if ( ccell_parent != pk )
                    continue;
                if ( ccell.depth != parent_ci.depth + 1 )
                    continue;

                // Translation vector: child center - parent center
                const scalar_type dx =
                    ccell.center[0] - parent_ci.center[0];
                const scalar_type dy =
                    ccell.center[1] - parent_ci.center[1];
                const scalar_type dz =
                    ccell.center[2] - parent_ci.center[2];

                auto M_child = Kokkos::subview(
                    multipoles, ci, Kokkos::ALL );

                // The M_child subview has extent 1 in the cell dim
                // we want a 1D view here for kernel input. We pass a
                // 2D (1, N) view and let the kernel index cell=0.
                // To keep it simple, adapt the kernel's access pattern:
                //   get_coeff<Scalar>( M_child_as_2d, 0, j-n, k-m, P )
                // by giving a 2D slice. We build a 2D view here:

                // Wrap M_child as a 2D view with leading size 1
                Kokkos::View<complex_type**,
                             typename coeff_view_type::memory_space,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                    M_child_2d( M_child.data(), 1,
                                static_cast<int>( M_child.extent( 0 ) ) );

                KernelType::m2m_translate( team, M_child_2d, dx, dy, dz,
                                           M_parent );
            }
        } );

    Kokkos::fence();
}

// --------------------------------------------------------------------------
// exchange_multipoles_at_depth
//
// After computing M2M at a given depth, exchange coefficients between
// ranks per the CommunicationPlan.
//
// Two cases:
//   1. Shared cells (allreduce): each rank has computed a partial
//      multipole from children it owns. Allreduce sums contributions.
//   2. Uniquely-owned cells with cross-rank children: handled by
//      point-to-point sends/receives of already-computed child data.
//
// For simplicity in this initial implementation, we do one allreduce
// over all shared cells at this depth. The sends/receives for
// uniquely-owned cells are done by the child's owner sending their
// child M_child to the parent's owner, who accumulates via a local
// M2M translation before this step. This is handled at the depth
// below: after M2M at depth d+1, we send child cells' multipoles to
// ranks that own parent cells at depth d, so when we run M2M at
// depth d those parents already have all their child data locally.
//
// In this first implementation, we do the simplest correct thing:
// bulk-exchange any multipoles needed by remote M2M operations. The
// plan's `sends`/`receives` at a depth boundary describe which child
// cell's multipoles flow from child-owner to parent-owner.
//
// Sequencing:
//   for depth d = max_depth down to 0:
//     run M2M at depth d (consumes children at depth d+1)
//     exchange multipoles at depth d (allreduce shared cells,
//         then point-to-point sends for children at d whose parents
//         at d-1 are owned by other ranks — i.e. we prepare for the
//         next iteration)
//
// To minimize complexity, this initial implementation:
//   - Performs a single allreduce per depth over that depth's shared
//     cells (if any).
//   - Performs point-to-point sends of multipoles of cells at this
//     depth whose PARENTS are on remote ranks. These happen AFTER
//     the M2M at this depth, so the next depth's M2M has the data.
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace, class KernelType>
void UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::exchange_multipoles_at_depth(
    int depth,
    const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan )
{
    // ------------------------------------------------------------------
    // Step 1: Allreduce shared cells at this depth.
    //
    // The M2M plan's shared_cells list contains every shared (replicated)
    // internal cell, across all depths. We filter to this depth.
    // ------------------------------------------------------------------
    const auto& m2m = comm_plan.m2m_plan();

    // Find shared cells at this depth (host-side filter)
    std::vector<int> shared_cell_indices;
    for ( MortonKey k : m2m.shared_cells )
    {
        auto it = _key_to_cell_idx.find( k );
        if ( it == _key_to_cell_idx.end() )
            continue;
        const int cidx = it->second;
        // Look up depth on host (we have cells on host)
        auto h_dc = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            Kokkos::subview( _device_cells, cidx ) );
        if ( h_dc().depth == depth )
            shared_cell_indices.push_back( cidx );
    }

    if ( !shared_cell_indices.empty() )
    {
        const int nshared = static_cast<int>( shared_cell_indices.size() );
        const int stride = coeffs_per_cell;
        const int total = nshared * stride;

        // Pack shared cell coefficients into a flat buffer on host.
        std::vector<complex_type> sendbuf( total ), recvbuf( total );
        auto h_mults = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), _multipoles );

        for ( int i = 0; i < nshared; i++ )
        {
            const int cidx = shared_cell_indices[i];
            for ( int c = 0; c < stride; c++ )
                sendbuf[i * stride + c] = h_mults( cidx, c );
        }

        // Allreduce as 2 * total doubles (or floats) — interpret complex
        // as pairs of real values.
        const int real_count = 2 * total;
        MPI_Datatype mpi_scalar =
            ( sizeof( scalar_type ) == 8 ) ? MPI_DOUBLE : MPI_FLOAT;

        MPI_Allreduce( reinterpret_cast<scalar_type*>( sendbuf.data() ),
                       reinterpret_cast<scalar_type*>( recvbuf.data() ),
                       real_count, mpi_scalar, MPI_SUM, _comm );

        // Scatter back into host mirror, then copy to device
        for ( int i = 0; i < nshared; i++ )
        {
            const int cidx = shared_cell_indices[i];
            for ( int c = 0; c < stride; c++ )
                h_mults( cidx, c ) = recvbuf[i * stride + c];
        }

        Kokkos::deep_copy( _multipoles, h_mults );
    }

    // ------------------------------------------------------------------
    // Step 2: Point-to-point sends.
    //
    // For cells at `depth` whose PARENT (at depth-1) is owned by a
    // different rank, send this cell's multipole to the parent's owner.
    //
    // The comm_plan's m2m.sends list contains CellTransfer entries where
    // `cell_key` is a child that must be sent and `remote_rank` is the
    // parent's owner. We filter to entries where the child is at
    // this depth.
    //
    // For simplicity we use synchronous MPI_Sendrecv-like pairs. For
    // performance, this should use non-blocking Isend/Irecv.
    // ------------------------------------------------------------------

    // Gather sends/receives at this depth on host
    struct Transfer
    {
        int cell_idx;
        int remote_rank;
    };
    std::vector<Transfer> my_sends, my_recvs;

    auto h_dc_all = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), _device_cells );

    for ( const auto& ct : m2m.sends )
    {
        auto it = _key_to_cell_idx.find( ct.cell_key );
        if ( it == _key_to_cell_idx.end() )
            continue;
        if ( h_dc_all( it->second ).depth == depth )
            my_sends.push_back( { it->second, ct.remote_rank } );
    }

    for ( const auto& ct : m2m.receives )
    {
        auto it = _key_to_cell_idx.find( ct.cell_key );
        if ( it == _key_to_cell_idx.end() )
            continue;
        if ( h_dc_all( it->second ).depth == depth )
            my_recvs.push_back( { it->second, ct.remote_rank } );
    }

    if ( my_sends.empty() && my_recvs.empty() )
        return;

    // Copy current multipole data to host for packing
    auto h_mults = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), _multipoles );

    const int stride = coeffs_per_cell;
    const int elem_real_count = 2 * stride;
    MPI_Datatype mpi_scalar =
        ( sizeof( scalar_type ) == 8 ) ? MPI_DOUBLE : MPI_FLOAT;

    // Post non-blocking receives
    std::vector<MPI_Request> recv_reqs( my_recvs.size() );
    std::vector<std::vector<complex_type>> recv_bufs( my_recvs.size() );
    for ( size_t i = 0; i < my_recvs.size(); i++ )
    {
        recv_bufs[i].resize( stride );
        const MortonKey key = h_dc_all( my_recvs[i].cell_idx ).key;
        // Use cell key as tag (truncated to int range)
        int tag = static_cast<int>( key & 0x7fffffff );
        MPI_Irecv( reinterpret_cast<scalar_type*>(
                       recv_bufs[i].data() ),
                   elem_real_count, mpi_scalar,
                   my_recvs[i].remote_rank, tag, _comm,
                   &recv_reqs[i] );
    }

    // Post non-blocking sends
    std::vector<MPI_Request> send_reqs( my_sends.size() );
    std::vector<std::vector<complex_type>> send_bufs( my_sends.size() );
    for ( size_t i = 0; i < my_sends.size(); i++ )
    {
        send_bufs[i].resize( stride );
        const int cidx = my_sends[i].cell_idx;
        for ( int c = 0; c < stride; c++ )
            send_bufs[i][c] = h_mults( cidx, c );

        const MortonKey key = h_dc_all( cidx ).key;
        int tag = static_cast<int>( key & 0x7fffffff );
        MPI_Isend( reinterpret_cast<scalar_type*>(
                       send_bufs[i].data() ),
                   elem_real_count, mpi_scalar,
                   my_sends[i].remote_rank, tag, _comm,
                   &send_reqs[i] );
    }

    // Wait for all
    if ( !recv_reqs.empty() )
        MPI_Waitall( recv_reqs.size(), recv_reqs.data(),
                     MPI_STATUSES_IGNORE );
    if ( !send_reqs.empty() )
        MPI_Waitall( send_reqs.size(), send_reqs.data(),
                     MPI_STATUSES_IGNORE );

    // Write received data into the host mirror and push to device
    for ( size_t i = 0; i < my_recvs.size(); i++ )
    {
        const int cidx = my_recvs[i].cell_idx;
        for ( int c = 0; c < stride; c++ )
            h_mults( cidx, c ) = recv_bufs[i][c];
    }

    Kokkos::deep_copy( _multipoles, h_mults );
}

// --------------------------------------------------------------------------
// execute — run the full upward sweep
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace, class KernelType>
template <class ChargeView, class PositionType>
void UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::execute(
    const ChargeView& particle_charges,
    const PositionType& particle_positions,
    const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan )
{
    // Step 1: Zero multipoles
    Kokkos::deep_copy( _multipoles, complex_type( 0.0, 0.0 ) );

    // Step 2: P2M at every depth that has leaves
    for ( int d = 0; d <= _max_depth; d++ )
    {
        if ( !_leaves_at_depth_local[d].empty() )
            run_p2m_at_depth( d, particle_charges, particle_positions );
    }

    // Step 3: Layer-by-layer M2M from deepest to root
    //
    // Before running M2M at depth d, we need all child data (at d+1)
    // to be available locally. The child data was either:
    //   - Computed locally via P2M (if the child is a leaf we own)
    //   - Computed locally via M2M at the previous iteration (if the
    //     child is an internal cell we own)
    //   - Received from another rank during the last exchange step
    //
    // So the sequence is:
    //   exchange at d+1  (make child data available everywhere it's needed)
    //   M2M at d         (consumes child data at d+1, produces M at d)
    //
    // For the deepest layer, there is no d+1, so no exchange is needed
    // before the first M2M. But after M2M at d, we need to exchange so
    // that the next iteration's M2M (at d-1) has its children available.

    // Exchange after the deepest P2M to make leaf multipoles available
    // to remote parents.
    exchange_multipoles_at_depth( _max_depth, comm_plan );

    for ( int d = _max_depth - 1; d >= 0; d-- )
    {
        // Run M2M at depth d — children are at d+1, already exchanged
        run_m2m_at_depth( d );

        // Exchange at depth d so that M2M at d-1 has its children
        if ( d > 0 )
            exchange_multipoles_at_depth( d, comm_plan );
    }
}

} // namespace Canopy

#endif // CANOPY_UPWARD_SWEEP_HPP
