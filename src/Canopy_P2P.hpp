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

#ifndef CANOPY_P2P_HPP
#define CANOPY_P2P_HPP

#include "Canopy_CommunicationPlan.hpp"
#include "Canopy_Helpers.hpp"
#include "Canopy_TreeBuilder.hpp"
#include "Canopy_TreePartitioner.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace Canopy
{

// ============================================================================
// P2P
//
// Computes direct (near-field) particle-particle interactions between
// each leaf and its P2P neighbor leaves. Accumulates contributions into
// caller-provided potential and gradient views so it composes naturally
// with DownwardSweep's L2P output.
//
// Newton's third law is applied to intra-leaf pairs only. Each pair
// (i, j) with i < j within a leaf is computed once and updates both
// endpoints via atomic accumulation. Inter-leaf pairs are computed
// once per direction (one team writes to its own leaf's particles
// without atomics).
//
// Ghost particle data is gathered from remote ranks via a custom MPI
// exchange (not Cabana::Halo) so that ghost particles are grouped by
// source leaf for efficient kernel access.
//
// Preconditions:
//   - TreePartitioner::partition()/repartition() has been called.
//   - TreePartitioner::sort_particles_by_leaf() has been called.
//   - CommunicationPlan::build() has been called.
//
// Template Parameters:
//   MemorySpace, ExecutionSpace - Kokkos memory/execution spaces
//   KernelType                  - e.g. LaplaceKernel<double, P>
// ============================================================================

template <class MemorySpace, class ExecutionSpace, class KernelType>
class P2P
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;

    using scalar_type = typename KernelType::scalar_type;

    static constexpr int NComps = KernelType::num_components;

    // Ghost particle storage (grouped by ghost leaf)
    using position_view_type = Kokkos::View<scalar_type* [3], memory_space>;
    using charge_view_type = Kokkos::View<scalar_type* [NComps], memory_space>;
    using offset_view_type = Kokkos::View<int*, memory_space>;

    // Output views (caller-owned, passed to execute())
    using potential_view_type =
        Kokkos::View<scalar_type* [NComps], memory_space>;
    using gradient_view_type =
        Kokkos::View<scalar_type* [NComps][3], memory_space>;

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------
    P2P( MPI_Comm comm )
        : _comm( comm )
        , _num_ghost_particles( 0 )
    {
        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );
    }

    // -----------------------------------------------------------------------
    // setup()
    //
    // Build the exchange plan and allocate ghost buffers. Call once after
    // partition + sort + comm_plan build. If tree topology or ownership
    // changes, call setup() again.
    //
    // Parameters:
    //   tree_builder  - provides cell info and particle_keys
    //   partitioner   - provides ownership map and leaf_particle_offsets
    //   comm_plan     - provides P2PPlan with neighbor_lists + exchange
    // -----------------------------------------------------------------------
    void
    setup( const TreeBuilder<MemorySpace, ExecutionSpace>& tree_builder,
           const TreePartitioner<MemorySpace, ExecutionSpace>& partitioner,
           const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan );

    // -----------------------------------------------------------------------
    // execute()
    //
    // Run the P2P near-field computation:
    //   1. Gather ghost particles from remote ranks.
    //   2. Intra-leaf pairs with Newton's third law (atomic accumulation).
    //   3. Inter-leaf pairs, one-way (no atomics needed).
    //
    // Contributions are ADDED to potential_out and gradient_out.
    // compute_gradient controls whether the gradient branch runs;
    // gradient_out may have zero extent if unused.
    //
    // Parameters:
    //   positions         - Cabana slice of local particle positions
    //   charges           - Cabana slice of local particle charges, shape
    //                       (num_local, NComps) where NComps comes from
    //                       KernelType::num_components. All NComps solves
    //                       are evaluated in a single execute() call,
    //                       sharing one ghost-particle halo exchange.
    //   potential_out     - caller-owned output; P2P adds its contribution.
    //   gradient_out      - caller-owned output (x, y, z); ignored if
    //                       compute_gradient is false.
    //   compute_gradient  - true to evaluate gradient contributions.
    // -----------------------------------------------------------------------
    template <class PositionSlice, class ChargeSlice>
    void execute( const PositionSlice& positions, const ChargeSlice& charges,
                  const potential_view_type& potential_out,
                  const gradient_view_type& gradient_out,
                  bool compute_gradient );

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    int num_ghost_particles() const { return _num_ghost_particles; }
    const position_view_type& ghost_positions() const
    {
        return _ghost_positions;
    }
    const charge_view_type& ghost_charges() const { return _ghost_charges; }

  private:
    MPI_Comm _comm;
    int _rank;
    int _comm_size;

    // -----------------------------------------------------------------------
    // Borrowed references from the partitioner / tree
    // -----------------------------------------------------------------------
    Kokkos::View<int*, memory_space> _leaf_particle_offsets; // (num_cells+1)

    // -----------------------------------------------------------------------
    // Ghost particle storage (grouped by ghost leaf)
    // -----------------------------------------------------------------------
    int _num_ghost_particles;
    position_view_type _ghost_positions;  // (num_ghost_particles, 3)
    charge_view_type _ghost_charges;      // (num_ghost_particles)
    offset_view_type _ghost_leaf_offsets; // (num_ghost_leaves + 1)

    // -----------------------------------------------------------------------
    // Exchange plan metadata (host-side)
    // -----------------------------------------------------------------------

    // Incoming ghost leaves, in the order they appear in the ghost buffer
    std::vector<MortonKey> _ghost_leaf_keys;
    std::vector<int> _ghost_leaf_owners;
    std::unordered_map<MortonKey, int> _ghost_leaf_key_to_idx;

    // Outgoing sends: for each, (local cell index, destination rank)
    struct SendEntry
    {
        int cell_idx;
        int dest_rank;
        MortonKey leaf_key; // retained for MPI tag computation
    };
    std::vector<SendEntry> _send_entries;

    // -----------------------------------------------------------------------
    // Neighbor lists for each local leaf, flattened on device.
    //
    // For each local leaf with cell index c, we store separately:
    //   - local neighbor cell indices (particles accessed via
    //     leaf_particle_offsets + main AoSoA)
    //   - ghost neighbor leaf indices (particles accessed via
    //     ghost_leaf_offsets + ghost views)
    //
    // The "self" leaf is handled in the intra-leaf kernel (Phase 1) and
    // is not included in the inter-leaf neighbor list.
    // -----------------------------------------------------------------------
    Kokkos::View<int*, memory_space> _local_leaf_cells; // target leaves
    Kokkos::View<int*, memory_space>
        _local_nbr_offsets; // prefix into local nbrs
    Kokkos::View<int*, memory_space>
        _local_nbr_cell_idx; // flat local neighbor cell idxs
    Kokkos::View<int*, memory_space>
        _ghost_nbr_offsets; // prefix into ghost nbrs
    Kokkos::View<int*, memory_space>
        _ghost_nbr_leaf_idx; // flat ghost neighbor indexes

  public:
    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------
    void build_exchange_plan(
        const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan,
        const std::unordered_map<MortonKey, int>& key_to_idx );

    void build_neighbor_lists_device(
        const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan,
        const std::unordered_map<MortonKey, int>& cell_key_to_idx );

    template <class PositionSlice, class ChargeSlice>
    void gather_ghost_particles( const PositionSlice& positions,
                                 const ChargeSlice& charges );
};

// ============================================================================
// Implementation
// ============================================================================

template <class MemorySpace, class ExecutionSpace, class KernelType>
void P2P<MemorySpace, ExecutionSpace, KernelType>::setup(
    const TreeBuilder<MemorySpace, ExecutionSpace>& tree_builder,
    const TreePartitioner<MemorySpace, ExecutionSpace>& partitioner,
    const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan )
{
    // Borrow leaf offsets from the partitioner
    _leaf_particle_offsets = partitioner.leaf_particle_offsets();

    // Build a host-side key -> cell_idx map for convenience
    const auto& cells = tree_builder.cells();
    std::unordered_map<MortonKey, int> key_to_idx;
    key_to_idx.reserve( cells.size() );
    for ( int i = 0; i < static_cast<int>( cells.size() ); i++ )
        key_to_idx[cells[i].key] = i;

    // Build exchange plan (ghost recv / send lists), resolving cell indices
    build_exchange_plan( comm_plan, key_to_idx );

    // Build neighbor lists on device (maps target leaf -> local + ghost
    // neighbors). Uses key_to_idx + _ghost_leaf_key_to_idx.
    build_neighbor_lists_device( comm_plan, key_to_idx );
}

// --------------------------------------------------------------------------
// build_exchange_plan
//
// Ingest the P2PPlan from CommunicationPlan:
//   - ghost_leaf_keys + ghost_leaf_owners -> our recv list
//   - send_leaves                         -> our send list
//
// Resolves send cell indices using the provided key_to_idx map.
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace, class KernelType>
void P2P<MemorySpace, ExecutionSpace, KernelType>::build_exchange_plan(
    const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan,
    const std::unordered_map<MortonKey, int>& key_to_idx )
{
    const auto& p2p = comm_plan.p2p_plan();

    // Ghost receive list
    _ghost_leaf_keys = p2p.ghost_leaf_keys;
    _ghost_leaf_owners = p2p.ghost_leaf_owners;

    _ghost_leaf_key_to_idx.clear();
    _ghost_leaf_key_to_idx.reserve( _ghost_leaf_keys.size() );
    for ( int i = 0; i < static_cast<int>( _ghost_leaf_keys.size() ); i++ )
        _ghost_leaf_key_to_idx[_ghost_leaf_keys[i]] = i;

    // Send list — resolve cell indices now
    _send_entries.clear();
    _send_entries.reserve( p2p.send_leaves.size() );
    for ( const auto& ct : p2p.send_leaves )
    {
        auto it = key_to_idx.find( ct.cell_key );
        if ( it == key_to_idx.end() )
            continue; // shouldn't happen, but skip unresolved
        SendEntry se;
        se.cell_idx = it->second;
        se.dest_rank = ct.remote_rank;
        se.leaf_key = ct.cell_key;
        _send_entries.push_back( se );
    }
}

// --------------------------------------------------------------------------
// build_neighbor_lists_device
//
// For each leaf we own (target), build the list of its neighbors,
// partitioned into:
//   - local neighbors (owned by us): reference local cell indices
//   - ghost neighbors (owned by remote): reference indices into the
//     ghost buffer (via _ghost_leaf_key_to_idx)
//
// Self is EXCLUDED from both neighbor lists. Intra-leaf pairs are
// handled by Phase 1 of the kernel via _local_leaf_cells.
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace, class KernelType>
void P2P<MemorySpace, ExecutionSpace, KernelType>::build_neighbor_lists_device(
    const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan,
    const std::unordered_map<MortonKey, int>& cell_key_to_idx )
{
    const auto& p2p = comm_plan.p2p_plan();
    const int nleaves = static_cast<int>( p2p.neighbor_lists.size() );

    // Collect target leaves (those we own) and flatten neighbor lists
    std::vector<int> target_cells;
    std::vector<int> local_offsets;
    std::vector<int> local_flat;
    std::vector<int> ghost_offsets;
    std::vector<int> ghost_flat;

    target_cells.reserve( nleaves );
    local_offsets.reserve( nleaves + 1 );
    ghost_offsets.reserve( nleaves + 1 );

    local_offsets.push_back( 0 );
    ghost_offsets.push_back( 0 );

    for ( const auto& [target_key, neighbors] : p2p.neighbor_lists )
    {
        auto tit = cell_key_to_idx.find( target_key );
        if ( tit == cell_key_to_idx.end() )
            continue;
        target_cells.push_back( tit->second );

        for ( MortonKey nk : neighbors )
        {
            if ( nk == target_key )
                continue; // exclude self (handled by Phase 1)

            // Is nk a local cell (owned by us) or a ghost?
            auto g_it = _ghost_leaf_key_to_idx.find( nk );
            if ( g_it != _ghost_leaf_key_to_idx.end() )
            {
                ghost_flat.push_back( g_it->second );
            }
            else
            {
                auto l_it = cell_key_to_idx.find( nk );
                if ( l_it != cell_key_to_idx.end() )
                    local_flat.push_back( l_it->second );
            }
        }

        local_offsets.push_back( static_cast<int>( local_flat.size() ) );
        ghost_offsets.push_back( static_cast<int>( ghost_flat.size() ) );
    }

    const int ntargets = static_cast<int>( target_cells.size() );

    auto upload_int = [&]( const std::vector<int>& src,
                           Kokkos::View<int*, memory_space>& dest,
                           const char* name )
    {
        const size_t n = src.size();
        dest = Kokkos::View<int*, memory_space>( std::string( name ), n );
        if ( n > 0 )
        {
            auto h = Kokkos::create_mirror_view( dest );
            for ( size_t i = 0; i < n; i++ )
                h( i ) = src[i];
            Kokkos::deep_copy( dest, h );
        }
    };

    upload_int( target_cells, _local_leaf_cells, "p2p_targets" );
    upload_int( local_offsets, _local_nbr_offsets, "p2p_local_nbr_off" );
    upload_int( local_flat, _local_nbr_cell_idx, "p2p_local_nbr_idx" );
    upload_int( ghost_offsets, _ghost_nbr_offsets, "p2p_ghost_nbr_off" );
    upload_int( ghost_flat, _ghost_nbr_leaf_idx, "p2p_ghost_nbr_idx" );

    (void)ntargets; // silence unused warning
}

// --------------------------------------------------------------------------
// gather_ghost_particles
//
// Custom MPI exchange to populate ghost buffers:
//   1. Exchange particle counts per ghost leaf via MPI point-to-point.
//   2. Compute ghost buffer offsets (prefix sum).
//   3. Allocate ghost buffers.
//   4. Pack local particle data per send, MPI_Isend; post MPI_Irecv into
//      temp buffers; wait; unpack into ghost views.
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace, class KernelType>
template <class PositionSlice, class ChargeSlice>
void P2P<MemorySpace, ExecutionSpace, KernelType>::gather_ghost_particles(
    const PositionSlice& positions, const ChargeSlice& charges )
{
    auto h_offsets = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), _leaf_particle_offsets );

    const int num_ghost_leaves = static_cast<int>( _ghost_leaf_keys.size() );
    const int num_send_entries = static_cast<int>( _send_entries.size() );

    std::vector<int> recv_counts( num_ghost_leaves, 0 );
    std::vector<int> send_counts( num_send_entries, 0 );

    // Fill send counts from leaf_particle_offsets
    for ( int i = 0; i < num_send_entries; i++ )
    {
        const int cidx = _send_entries[i].cell_idx;
        send_counts[i] = h_offsets( cidx + 1 ) - h_offsets( cidx );
    }

    // --- Count exchange phase ---
    std::vector<MPI_Request> recv_count_reqs( num_ghost_leaves );
    for ( int i = 0; i < num_ghost_leaves; i++ )
    {
        int tag = static_cast<int>( _ghost_leaf_keys[i] & 0x7fffffff );
        MPI_Irecv( &recv_counts[i], 1, MPI_INT, _ghost_leaf_owners[i], tag,
                   _comm, &recv_count_reqs[i] );
    }

    std::vector<MPI_Request> send_count_reqs( num_send_entries );
    for ( int i = 0; i < num_send_entries; i++ )
    {
        int tag = static_cast<int>( _send_entries[i].leaf_key & 0x7fffffff );
        MPI_Isend( &send_counts[i], 1, MPI_INT, _send_entries[i].dest_rank, tag,
                   _comm, &send_count_reqs[i] );
    }

    if ( !recv_count_reqs.empty() )
        MPI_Waitall( recv_count_reqs.size(), recv_count_reqs.data(),
                     MPI_STATUSES_IGNORE );
    if ( !send_count_reqs.empty() )
        MPI_Waitall( send_count_reqs.size(), send_count_reqs.data(),
                     MPI_STATUSES_IGNORE );

    // Build ghost offsets and allocate ghost buffers
    _ghost_leaf_offsets = Kokkos::View<int*, memory_space>(
        "ghost_leaf_offsets", num_ghost_leaves + 1 );
    auto h_goff = Kokkos::create_mirror_view( _ghost_leaf_offsets );
    h_goff( 0 ) = 0;
    for ( int i = 0; i < num_ghost_leaves; i++ )
        h_goff( i + 1 ) = h_goff( i ) + recv_counts[i];
    _num_ghost_particles = h_goff( num_ghost_leaves );
    Kokkos::deep_copy( _ghost_leaf_offsets, h_goff );

    if ( _num_ghost_particles > 0 )
    {
        _ghost_positions =
            position_view_type( "ghost_positions", _num_ghost_particles );
        _ghost_charges =
            charge_view_type( "ghost_charges", _num_ghost_particles );
    }
    else
    {
        // Ensure views have valid zero-extent storage even when no ghosts
        _ghost_positions = position_view_type( "ghost_positions", 0 );
        _ghost_charges = charge_view_type( "ghost_charges", 0 );
    }

    constexpr int per_particle = 3 + NComps;

    // --- Particle data exchange ---
    auto h_pos =
        Canopy::create_mirror_view_and_copy( Kokkos::HostSpace(), positions );
    auto h_chg =
        Canopy::create_mirror_view_and_copy( Kokkos::HostSpace(), charges );

    auto h_gpos = Kokkos::create_mirror_view( _ghost_positions );
    auto h_gchg = Kokkos::create_mirror_view( _ghost_charges );

    MPI_Datatype mpi_scalar =
        ( sizeof( scalar_type ) == 8 ) ? MPI_DOUBLE : MPI_FLOAT;

    // Post recvs for particle data
    std::vector<std::vector<scalar_type>> recv_bufs( num_ghost_leaves );
    std::vector<MPI_Request> recv_data_reqs( num_ghost_leaves,
                                             MPI_REQUEST_NULL );
    for ( int i = 0; i < num_ghost_leaves; i++ )
    {
        const int count = recv_counts[i];
        if ( count == 0 )
            continue;
        recv_bufs[i].resize( per_particle * count );

        int tag =
            static_cast<int>( ( _ghost_leaf_keys[i] & 0x7fffffff ) ^ 0xABCD );
        MPI_Irecv( recv_bufs[i].data(), per_particle * count, mpi_scalar,
                   _ghost_leaf_owners[i], tag, _comm, &recv_data_reqs[i] );
    }

    // Post sends for particle data
    std::vector<std::vector<scalar_type>> send_bufs( num_send_entries );
    std::vector<MPI_Request> send_data_reqs( num_send_entries,
                                             MPI_REQUEST_NULL );
    for ( int i = 0; i < num_send_entries; i++ )
    {
        const int cidx = _send_entries[i].cell_idx;
        const int start = h_offsets( cidx );
        const int count = h_offsets( cidx + 1 ) - start;
        if ( count == 0 )
            continue;

        send_bufs[i].resize( per_particle * count );
        for ( int p = 0; p < count; p++ )
        {
            send_bufs[i][per_particle * p + 0] =
                static_cast<scalar_type>( h_pos( start + p, 0 ) );
            send_bufs[i][per_particle * p + 1] =
                static_cast<scalar_type>( h_pos( start + p, 1 ) );
            send_bufs[i][per_particle * p + 2] =
                static_cast<scalar_type>( h_pos( start + p, 2 ) );
            for ( int c = 0; c < NComps; c++ )
                send_bufs[i][per_particle * p + 3 + c] =
                    static_cast<scalar_type>( h_chg( start + p, c ) );
        }

        int tag = static_cast<int>( ( _send_entries[i].leaf_key & 0x7fffffff ) ^
                                    0xABCD );
        MPI_Isend( send_bufs[i].data(), per_particle * count, mpi_scalar,
                   _send_entries[i].dest_rank, tag, _comm, &send_data_reqs[i] );
    }

    if ( !recv_data_reqs.empty() )
        MPI_Waitall( recv_data_reqs.size(), recv_data_reqs.data(),
                     MPI_STATUSES_IGNORE );
    if ( !send_data_reqs.empty() )
        MPI_Waitall( send_data_reqs.size(), send_data_reqs.data(),
                     MPI_STATUSES_IGNORE );

    // Unpack recv buffers into ghost views
    for ( int i = 0; i < num_ghost_leaves; i++ )
    {
        const int count = recv_counts[i];
        const int base = h_goff( i );
        for ( int p = 0; p < count; p++ )
        {
            h_gpos( base + p, 0 ) = recv_bufs[i][per_particle * p + 0];
            h_gpos( base + p, 1 ) = recv_bufs[i][per_particle * p + 1];
            h_gpos( base + p, 2 ) = recv_bufs[i][per_particle * p + 2];
            for ( int c = 0; c < NComps; c++ )
                h_gchg( base + p, c ) = recv_bufs[i][per_particle * p + 3 + c];
        }
    }

    if ( _num_ghost_particles > 0 )
    {
        Kokkos::deep_copy( _ghost_positions, h_gpos );
        Kokkos::deep_copy( _ghost_charges, h_gchg );
    }
}

// --------------------------------------------------------------------------
// execute
// --------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace, class KernelType>
template <class PositionSlice, class ChargeSlice>
void P2P<MemorySpace, ExecutionSpace, KernelType>::execute(
    const PositionSlice& positions, const ChargeSlice& charges,
    const potential_view_type& potential_out,
    const gradient_view_type& gradient_out, bool compute_gradient )
{
    // ------------------------------------------------------------------
    // 1. Gather ghost particles
    // ------------------------------------------------------------------
    gather_ghost_particles( positions, charges );

    // ------------------------------------------------------------------
    // 2. Phase 1: intra-leaf pairs with Newton's third law
    //
    // Launch one team per local leaf. Team threads iterate over pair
    // indices (i, j) with i < j. Because team threads from one team
    // may both target the same output slot (e.g., pair (0,1) and pair
    // (0,2) both update particle 0), we use atomics within the leaf.
    // Contention is bounded by the number of particles in a single leaf.
    // ------------------------------------------------------------------
    auto leaf_offsets = _leaf_particle_offsets;
    auto local_leaf_cells = _local_leaf_cells;

    const int num_target_leaves = _local_leaf_cells.extent( 0 );

    using team_policy = Kokkos::TeamPolicy<execution_space>;
    using team_member_type = typename team_policy::member_type;

    if ( num_target_leaves > 0 )
    {
        team_policy policy( num_target_leaves, Kokkos::AUTO );

        Kokkos::parallel_for(
            "P2P_intra_leaf", policy,
            KOKKOS_LAMBDA( const team_member_type& team ) {
                const int league = team.league_rank();
                const int cidx = local_leaf_cells( league );
                const int pstart = leaf_offsets( cidx );
                const int pend = leaf_offsets( cidx + 1 );
                const int ncell = pend - pstart;
                if ( ncell < 2 )
                    return;

                const int npairs = ncell * ( ncell - 1 ) / 2;

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange( team, npairs ),
                    [&]( const int pair_idx )
                    {
                        // Convert pair_idx to (i, j) with i < j
                        // Upper-triangular indexing:
                        //   row i has (ncell-1-i) entries
                        //   we invert to find (i, j) for a given flat index.
                        // Use the formula:
                        //   i = ncell - 2 - floor(sqrt(-8*pair_idx +
                        //   4*ncell*(ncell-1) - 7)/2 - 0.5) j = pair_idx + i +
                        //   1 - ncell*(ncell-1)/2 + (ncell-i)*((ncell-i)-1)/2
                        const double x =
                            Kokkos::sqrt( -8.0 * pair_idx +
                                          4.0 * ncell * ( ncell - 1 ) - 7.0 );
                        int i = static_cast<int>(
                            static_cast<double>( ncell ) - 2.0 -
                            Kokkos::floor( x * 0.5 - 0.5 ) );
                        // Clamp i
                        if ( i < 0 )
                            i = 0;
                        if ( i > ncell - 2 )
                            i = ncell - 2;
                        int row_base = ncell * ( ncell - 1 ) / 2 -
                                       ( ncell - i ) * ( ncell - i - 1 ) / 2;
                        int j = pair_idx - row_base + i + 1;
                        // Correct for any indexing drift from floor rounding
                        while ( j <= i )
                        {
                            i--;
                            row_base = ncell * ( ncell - 1 ) / 2 -
                                       ( ncell - i ) * ( ncell - i - 1 ) / 2;
                            j = pair_idx - row_base + i + 1;
                        }
                        while ( j >= ncell )
                        {
                            i++;
                            row_base = ncell * ( ncell - 1 ) / 2 -
                                       ( ncell - i ) * ( ncell - i - 1 ) / 2;
                            j = pair_idx - row_base + i + 1;
                        }

                        const int pi = pstart + i;
                        const int pj = pstart + j;

                        const scalar_type xi =
                            static_cast<scalar_type>( positions( pi, 0 ) );
                        const scalar_type yi =
                            static_cast<scalar_type>( positions( pi, 1 ) );
                        const scalar_type zi =
                            static_cast<scalar_type>( positions( pi, 2 ) );

                        const scalar_type xj =
                            static_cast<scalar_type>( positions( pj, 0 ) );
                        const scalar_type yj =
                            static_cast<scalar_type>( positions( pj, 1 ) );
                        const scalar_type zj =
                            static_cast<scalar_type>( positions( pj, 2 ) );

                        const scalar_type dx = xi - xj;
                        const scalar_type dy = yi - yj;
                        const scalar_type dz = zi - zj;
                        const scalar_type r2 = dx * dx + dy * dy + dz * dz;
                        if ( r2 < static_cast<scalar_type>( 1.0e-24 ) )
                            return;
                        const scalar_type inv_r =
                            static_cast<scalar_type>( 1.0 ) /
                            Kokkos::sqrt( r2 );
                        const scalar_type inv_r3 = inv_r * inv_r * inv_r;

                        scalar_type qi[NComps];
                        scalar_type qj[NComps];
                        for ( int c = 0; c < NComps; c++ )
                        {
                            qi[c] =
                                static_cast<scalar_type>( charges( pi, c ) );
                            qj[c] =
                                static_cast<scalar_type>( charges( pj, c ) );
                        }

                        // Potential (Newton): +q_j/r on i, +q_i/r on j
                        for ( int c = 0; c < NComps; c++ )
                        {
                            Kokkos::atomic_add( &potential_out( pi, c ),
                                                qj[c] * inv_r );
                            Kokkos::atomic_add( &potential_out( pj, c ),
                                                qi[c] * inv_r );
                        }

                        if ( compute_gradient )
                        {
                            // grad_i(phi) from j: -q_j * (r_i - r_j) / r^3
                            // grad_j(phi) from i: -q_i * (r_j - r_i) / r^3
                            //                  = +q_i * (r_i - r_j) / r^3
                            for ( int c = 0; c < NComps; c++ )
                            {
                                Kokkos::atomic_add( &gradient_out( pi, c, 0 ),
                                                    -qj[c] * dx * inv_r3 );
                                Kokkos::atomic_add( &gradient_out( pi, c, 1 ),
                                                    -qj[c] * dy * inv_r3 );
                                Kokkos::atomic_add( &gradient_out( pi, c, 2 ),
                                                    -qj[c] * dz * inv_r3 );

                                Kokkos::atomic_add( &gradient_out( pj, c, 0 ),
                                                    qi[c] * dx * inv_r3 );
                                Kokkos::atomic_add( &gradient_out( pj, c, 1 ),
                                                    qi[c] * dy * inv_r3 );
                                Kokkos::atomic_add( &gradient_out( pj, c, 2 ),
                                                    qi[c] * dz * inv_r3 );
                            }
                        }
                    } );
            } );

        Kokkos::fence();
    }

    // ------------------------------------------------------------------
    // 3. Phase 2: inter-leaf pairs (one-way, no atomics needed)
    //
    // For each local target leaf, iterate over:
    //   - Local neighbors (other leaves on this rank)
    //   - Ghost neighbors (leaves from remote ranks)
    //
    // Team parallelism: one team per target leaf. Team threads over
    // target particles. Each team writes only to its own target's
    // output slots — no contention with other teams.
    // ------------------------------------------------------------------
    auto local_nbr_off = _local_nbr_offsets;
    auto local_nbr_idx = _local_nbr_cell_idx;
    auto ghost_nbr_off = _ghost_nbr_offsets;
    auto ghost_nbr_idx = _ghost_nbr_leaf_idx;
    auto ghost_leaf_off = _ghost_leaf_offsets;
    auto ghost_positions = _ghost_positions;
    auto ghost_charges = _ghost_charges;

    if ( num_target_leaves > 0 )
    {
        team_policy policy( num_target_leaves, Kokkos::AUTO );

        Kokkos::parallel_for(
            "P2P_inter_leaf", policy,
            KOKKOS_LAMBDA( const team_member_type& team ) {
                const int league = team.league_rank();
                const int cidx = local_leaf_cells( league );
                const int pstart = leaf_offsets( cidx );
                const int pend = leaf_offsets( cidx + 1 );
                const int ntarget = pend - pstart;
                if ( ntarget == 0 )
                    return;

                // Iterate target particles in parallel across team threads
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange( team, ntarget ),
                    [&]( const int t )
                    {
                        const int pi = pstart + t;

                        const scalar_type xi =
                            static_cast<scalar_type>( positions( pi, 0 ) );
                        const scalar_type yi =
                            static_cast<scalar_type>( positions( pi, 1 ) );
                        const scalar_type zi =
                            static_cast<scalar_type>( positions( pi, 2 ) );

                        scalar_type phi[NComps];
                        scalar_type gx[NComps], gy[NComps], gz[NComps];
                        for ( int c = 0; c < NComps; c++ )
                        {
                            phi[c] = 0.0;
                            gx[c] = 0.0;
                            gy[c] = 0.0;
                            gz[c] = 0.0;
                        }

                        // --- Local neighbors ---
                        const int l_start = local_nbr_off( league );
                        const int l_end = local_nbr_off( league + 1 );
                        for ( int nn = l_start; nn < l_end; nn++ )
                        {
                            const int n_cidx = local_nbr_idx( nn );
                            const int ns = leaf_offsets( n_cidx );
                            const int ne = leaf_offsets( n_cidx + 1 );
                            for ( int pj = ns; pj < ne; pj++ )
                            {
                                const scalar_type dx =
                                    xi - static_cast<scalar_type>(
                                             positions( pj, 0 ) );
                                const scalar_type dy =
                                    yi - static_cast<scalar_type>(
                                             positions( pj, 1 ) );
                                const scalar_type dz =
                                    zi - static_cast<scalar_type>(
                                             positions( pj, 2 ) );
                                const scalar_type r2 =
                                    dx * dx + dy * dy + dz * dz;
                                if ( r2 < static_cast<scalar_type>( 1.0e-24 ) )
                                    continue;
                                const scalar_type inv_r =
                                    static_cast<scalar_type>( 1.0 ) /
                                    Kokkos::sqrt( r2 );
                                const scalar_type inv_r3 =
                                    inv_r * inv_r * inv_r;
                                for ( int c = 0; c < NComps; c++ )
                                {
                                    const scalar_type qj =
                                        static_cast<scalar_type>(
                                            charges( pj, c ) );
                                    phi[c] += qj * inv_r;
                                    if ( compute_gradient )
                                    {
                                        gx[c] -= qj * dx * inv_r3;
                                        gy[c] -= qj * dy * inv_r3;
                                        gz[c] -= qj * dz * inv_r3;
                                    }
                                }
                            }
                        }

                        // --- Ghost neighbors ---
                        const int g_start = ghost_nbr_off( league );
                        const int g_end = ghost_nbr_off( league + 1 );
                        for ( int nn = g_start; nn < g_end; nn++ )
                        {
                            const int g_idx = ghost_nbr_idx( nn );
                            const int gs = ghost_leaf_off( g_idx );
                            const int ge = ghost_leaf_off( g_idx + 1 );
                            for ( int pj = gs; pj < ge; pj++ )
                            {
                                const scalar_type dx =
                                    xi - ghost_positions( pj, 0 );
                                const scalar_type dy =
                                    yi - ghost_positions( pj, 1 );
                                const scalar_type dz =
                                    zi - ghost_positions( pj, 2 );
                                const scalar_type r2 =
                                    dx * dx + dy * dy + dz * dz;
                                if ( r2 < static_cast<scalar_type>( 1.0e-24 ) )
                                    continue;
                                const scalar_type inv_r =
                                    static_cast<scalar_type>( 1.0 ) /
                                    Kokkos::sqrt( r2 );
                                const scalar_type inv_r3 =
                                    inv_r * inv_r * inv_r;
                                for ( int c = 0; c < NComps; c++ )
                                {
                                    const scalar_type qj =
                                        ghost_charges( pj, c );
                                    phi[c] += qj * inv_r;
                                    if ( compute_gradient )
                                    {
                                        gx[c] -= qj * dx * inv_r3;
                                        gy[c] -= qj * dy * inv_r3;
                                        gz[c] -= qj * dz * inv_r3;
                                    }
                                }
                            }
                        }

                        // Single-writer: no atomic needed
                        for ( int c = 0; c < NComps; c++ )
                        {
                            potential_out( pi, c ) += phi[c];
                            if ( compute_gradient )
                            {
                                gradient_out( pi, c, 0 ) += gx[c];
                                gradient_out( pi, c, 1 ) += gy[c];
                                gradient_out( pi, c, 2 ) += gz[c];
                            }
                        }
                    } );
            } );

        Kokkos::fence();
    }
}

} // end namespace Canopy

#endif // CANOPY_P2P_HPP
