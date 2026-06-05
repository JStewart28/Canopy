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
#include "Canopy_Profiling.hpp"
#include "Canopy_Helpers.hpp"
#include "Canopy_TreeBuilder.hpp"
#include "Canopy_TreePartitioner.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
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
    // set_softening(): set the Plummer softening length eps used by the
    // near-field kernel. The pairwise 1/r and 1/r^3 terms are evaluated
    // with r^2 -> r^2 + eps^2, which bounds the force for close encounters
    // (eps = 0 reproduces the unsoftened kernel). Stored squared so the
    // device kernel does no extra work per pair.
    // -----------------------------------------------------------------------
    void set_softening( scalar_type eps )
    {
        _softening2 = eps * eps;
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

    // Plummer softening length squared (eps^2). 0 => unsoftened kernel.
    scalar_type _softening2 = static_cast<scalar_type>( 0 );

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

    // Maps each local particle index to its league (index into
    // _local_leaf_cells). Entry is -1 for particles not owned by any local
    // leaf. Built in setup(); used by inter-leaf kernel for flat
    // thread-per-particle decomposition.
    Kokkos::View<int*, memory_space> _particle_to_league;

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

    // Build particle -> league mapping for the flat inter-leaf kernel.
    {
        auto leaf_off_h = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), _leaf_particle_offsets );
        auto local_leaves_h = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), _local_leaf_cells );
        const int num_cells =
            static_cast<int>( _leaf_particle_offsets.extent( 0 ) ) - 1;
        const int n_total = ( num_cells >= 0 ) ? leaf_off_h( num_cells ) : 0;
        _particle_to_league = Kokkos::View<int*, memory_space>(
            Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                "p2p_particle_to_league" ),
            n_total );
        Kokkos::View<int*, Kokkos::HostSpace> p2l_h(
            Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                "p2p_particle_to_league_h" ),
            n_total );
        for ( int i = 0; i < n_total; i++ )
            p2l_h( i ) = -1;
        const int num_target_leaves =
            static_cast<int>( _local_leaf_cells.extent( 0 ) );
        for ( int g = 0; g < num_target_leaves; g++ )
        {
            const int cidx = local_leaves_h( g );
            const int ps = leaf_off_h( cidx );
            const int pe = leaf_off_h( cidx + 1 );
            for ( int pi = ps; pi < pe; pi++ )
                p2l_h( pi ) = g;
        }
        Kokkos::deep_copy( _particle_to_league, p2l_h );
    }
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
    CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_P2P_GHOST_COMM );
    auto h_offsets = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), _leaf_particle_offsets );

    const int num_ghost_leaves = static_cast<int>( _ghost_leaf_keys.size() );
    const int num_send_entries = static_cast<int>( _send_entries.size() );

    // Group sends/receives by peer rank, sorted by MortonKey so that both
    // sides agree on the in-buffer order without an extra metadata exchange.
    // One MPI_Isend / MPI_Irecv per peer instead of one per leaf — the
    // per-leaf pattern exhausts the MPI request freelist at large
    // particle counts.
    std::map<int, std::vector<std::pair<MortonKey, int>>> sends_by_peer_kv;
    for ( int i = 0; i < num_send_entries; i++ )
        sends_by_peer_kv[_send_entries[i].dest_rank].emplace_back(
            _send_entries[i].leaf_key, i );
    for ( auto& kv : sends_by_peer_kv )
        std::sort( kv.second.begin(), kv.second.end(),
                   []( const std::pair<MortonKey, int>& a,
                       const std::pair<MortonKey, int>& b )
                   { return a.first < b.first; } );

    std::map<int, std::vector<std::pair<MortonKey, int>>> recvs_by_peer_kv;
    for ( int i = 0; i < num_ghost_leaves; i++ )
        recvs_by_peer_kv[_ghost_leaf_owners[i]].emplace_back(
            _ghost_leaf_keys[i], i );
    for ( auto& kv : recvs_by_peer_kv )
        std::sort( kv.second.begin(), kv.second.end(),
                   []( const std::pair<MortonKey, int>& a,
                       const std::pair<MortonKey, int>& b )
                   { return a.first < b.first; } );

    // --- Count exchange phase (one message per peer, host-side) ---
    std::map<int, std::vector<int>> recv_count_bufs;
    std::vector<MPI_Request> recv_count_reqs;
    recv_count_reqs.reserve( recvs_by_peer_kv.size() );
    for ( auto& kv : recvs_by_peer_kv )
    {
        auto& buf = recv_count_bufs[kv.first];
        buf.resize( kv.second.size() );
        MPI_Request req;
        MPI_Irecv( buf.data(), static_cast<int>( buf.size() ), MPI_INT,
                   kv.first, /*tag=*/0, _comm, &req );
        recv_count_reqs.push_back( req );
    }

    std::map<int, std::vector<int>> send_count_bufs;
    std::vector<MPI_Request> send_count_reqs;
    send_count_reqs.reserve( sends_by_peer_kv.size() );
    for ( auto& kv : sends_by_peer_kv )
    {
        auto& buf = send_count_bufs[kv.first];
        buf.reserve( kv.second.size() );
        for ( const auto& pr : kv.second )
        {
            const int cidx = _send_entries[pr.second].cell_idx;
            buf.push_back( h_offsets( cidx + 1 ) - h_offsets( cidx ) );
        }
        MPI_Request req;
        MPI_Isend( buf.data(), static_cast<int>( buf.size() ), MPI_INT,
                   kv.first, /*tag=*/0, _comm, &req );
        send_count_reqs.push_back( req );
    }

    if ( !recv_count_reqs.empty() )
        MPI_Waitall( recv_count_reqs.size(), recv_count_reqs.data(),
                     MPI_STATUSES_IGNORE );
    if ( !send_count_reqs.empty() )
        MPI_Waitall( send_count_reqs.size(), send_count_reqs.data(),
                     MPI_STATUSES_IGNORE );

    // Scatter received per-peer counts into per-ghost-leaf array
    std::vector<int> recv_counts( num_ghost_leaves, 0 );
    for ( const auto& kv : recvs_by_peer_kv )
    {
        const auto& buf = recv_count_bufs.at( kv.first );
        for ( size_t i = 0; i < kv.second.size(); i++ )
            recv_counts[kv.second[i].second] = buf[i];
    }

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
    MPI_Datatype mpi_scalar =
        ( sizeof( scalar_type ) == 8 ) ? MPI_DOUBLE : MPI_FLOAT;

    // --- Particle data exchange (one message per peer, device-side pack) ---
    //
    // Pack and unpack run on the execution space of the views; the .data()
    // pointers passed to MPI are device pointers on GPU backends. This
    // requires MPICH_GPU_SUPPORT_ENABLED=1 at runtime on GPU backends
    // (Cray MPICH on Slingshot/HSN). On host backends the pointers are
    // host pointers and no env var is needed.
    const int n_send_peers = static_cast<int>( sends_by_peer_kv.size() );
    const int n_recv_peers = static_cast<int>( recvs_by_peer_kv.size() );

    std::vector<Kokkos::View<int*, memory_space>> send_idx_views( n_send_peers );
    std::vector<Kokkos::View<scalar_type*, memory_space>> send_bufs(
        n_send_peers );
    std::vector<int> send_peer_ranks( n_send_peers );
    std::vector<size_t> send_peer_nparticles( n_send_peers );

    std::vector<Kokkos::View<int*, memory_space>> recv_idx_views( n_recv_peers );
    std::vector<Kokkos::View<scalar_type*, memory_space>> recv_bufs(
        n_recv_peers );
    std::vector<int> recv_peer_ranks( n_recv_peers );
    std::vector<size_t> recv_peer_nparticles( n_recv_peers );

    // Build per-peer send index (which local particles to pack) and post
    // recv-side index / buffer allocations.
    {
        int q = 0;
        for ( const auto& kv : sends_by_peer_kv )
        {
            send_peer_ranks[q] = kv.first;
            size_t total = 0;
            for ( const auto& pr : kv.second )
            {
                const int cidx = _send_entries[pr.second].cell_idx;
                total += static_cast<size_t>( h_offsets( cidx + 1 ) -
                                              h_offsets( cidx ) );
            }
            send_peer_nparticles[q] = total;
            send_idx_views[q] = Kokkos::View<int*, memory_space>(
                Kokkos::view_alloc( "p2p_send_idx",
                                    Kokkos::WithoutInitializing ),
                total );
            send_bufs[q] = Kokkos::View<scalar_type*, memory_space>(
                Kokkos::view_alloc( "p2p_send_buf",
                                    Kokkos::WithoutInitializing ),
                per_particle * total );
            if ( total > 0 )
            {
                auto h_idx = Kokkos::create_mirror_view( send_idx_views[q] );
                size_t pos = 0;
                for ( const auto& pr : kv.second )
                {
                    const int cidx = _send_entries[pr.second].cell_idx;
                    const int start = h_offsets( cidx );
                    const int count = h_offsets( cidx + 1 ) - start;
                    for ( int p = 0; p < count; p++ )
                        h_idx( pos++ ) = start + p;
                }
                Kokkos::deep_copy( send_idx_views[q], h_idx );
            }
            ++q;
        }
    }

    // Post receives first.
    std::vector<MPI_Request> recv_data_reqs;
    recv_data_reqs.reserve( n_recv_peers );
    {
        int q = 0;
        for ( const auto& kv : recvs_by_peer_kv )
        {
            recv_peer_ranks[q] = kv.first;
            size_t total = 0;
            for ( const auto& pr : kv.second )
                total += static_cast<size_t>( recv_counts[pr.second] );
            recv_peer_nparticles[q] = total;
            recv_idx_views[q] = Kokkos::View<int*, memory_space>(
                Kokkos::view_alloc( "p2p_recv_idx",
                                    Kokkos::WithoutInitializing ),
                total );
            recv_bufs[q] = Kokkos::View<scalar_type*, memory_space>(
                Kokkos::view_alloc( "p2p_recv_buf",
                                    Kokkos::WithoutInitializing ),
                per_particle * total );
            if ( total > 0 )
            {
                auto h_idx = Kokkos::create_mirror_view( recv_idx_views[q] );
                size_t pos = 0;
                for ( const auto& pr : kv.second )
                {
                    const int ghost_leaf_idx = pr.second;
                    const int count = recv_counts[ghost_leaf_idx];
                    const int base = h_goff( ghost_leaf_idx );
                    for ( int p = 0; p < count; p++ )
                        h_idx( pos++ ) = base + p;
                }
                Kokkos::deep_copy( recv_idx_views[q], h_idx );

                MPI_Request req;
                MPI_Irecv( recv_bufs[q].data(),
                           static_cast<int>( per_particle * total ),
                           mpi_scalar, kv.first, /*tag=*/1, _comm, &req );
                recv_data_reqs.push_back( req );
            }
            ++q;
        }
    }

    // Pack send buffers on device.
    for ( int q = 0; q < n_send_peers; q++ )
    {
        const size_t n = send_peer_nparticles[q];
        if ( n == 0 )
            continue;
        auto idx_v = send_idx_views[q];
        auto buf_v = send_bufs[q];
        auto pos_v = positions;
        auto chg_v = charges;
        const int pp = per_particle;
        const int nc = NComps;
        Kokkos::parallel_for(
            "p2p_pack",
            Kokkos::RangePolicy<execution_space>( 0, static_cast<int>( n ) ),
            KOKKOS_LAMBDA( const int i ) {
                const int src = idx_v( i );
                const int base = i * pp;
                buf_v( base + 0 ) = static_cast<scalar_type>( pos_v( src, 0 ) );
                buf_v( base + 1 ) = static_cast<scalar_type>( pos_v( src, 1 ) );
                buf_v( base + 2 ) = static_cast<scalar_type>( pos_v( src, 2 ) );
                for ( int c = 0; c < nc; c++ )
                    buf_v( base + 3 + c ) =
                        static_cast<scalar_type>( chg_v( src, c ) );
            } );
    }
    Kokkos::fence();

    // Post sends with device pointers.
    std::vector<MPI_Request> send_data_reqs;
    send_data_reqs.reserve( n_send_peers );
    for ( int q = 0; q < n_send_peers; q++ )
    {
        const size_t n = send_peer_nparticles[q];
        if ( n == 0 )
            continue;
        MPI_Request req;
        MPI_Isend( send_bufs[q].data(),
                   static_cast<int>( per_particle * n ), mpi_scalar,
                   send_peer_ranks[q], /*tag=*/1, _comm, &req );
        send_data_reqs.push_back( req );
    }

    if ( !recv_data_reqs.empty() )
        MPI_Waitall( recv_data_reqs.size(), recv_data_reqs.data(),
                     MPI_STATUSES_IGNORE );
    if ( !send_data_reqs.empty() )
        MPI_Waitall( send_data_reqs.size(), send_data_reqs.data(),
                     MPI_STATUSES_IGNORE );

    // Unpack on device.
    auto ghost_pos_v = _ghost_positions;
    auto ghost_chg_v = _ghost_charges;
    for ( int q = 0; q < n_recv_peers; q++ )
    {
        const size_t n = recv_peer_nparticles[q];
        if ( n == 0 )
            continue;
        auto idx_v = recv_idx_views[q];
        auto buf_v = recv_bufs[q];
        const int pp = per_particle;
        const int nc = NComps;
        Kokkos::parallel_for(
            "p2p_unpack",
            Kokkos::RangePolicy<execution_space>( 0, static_cast<int>( n ) ),
            KOKKOS_LAMBDA( const int i ) {
                const int dst = idx_v( i );
                const int base = i * pp;
                ghost_pos_v( dst, 0 ) = buf_v( base + 0 );
                ghost_pos_v( dst, 1 ) = buf_v( base + 1 );
                ghost_pos_v( dst, 2 ) = buf_v( base + 2 );
                for ( int c = 0; c < nc; c++ )
                    ghost_chg_v( dst, c ) = buf_v( base + 3 + c );
            } );
    }
    Kokkos::fence();
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
    CANOPY_RESET_TIMERS();
    {
    CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_P2P_TOTAL );

    // Plummer softening: r^2 -> r^2 + eps^2 in every pairwise term. With
    // eps == 0 this is the unsoftened kernel and the r2 < 1e-24
    // self-coincidence guard below still fires; with eps > 0 the guard
    // never trips because r2 + eps2 >= eps2 > 0.
    const scalar_type eps2 = _softening2;

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

    {
        CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_P2P_INTRA_KERNEL );
        if ( num_target_leaves > 0 )
        {
        // Particle-centric intra-leaf kernel. Each thread handles one
        // local particle pi: it iterates all other particles in pi's leaf
        // and accumulates into pi's own output slot (single-writer, no
        // atomics). 2x FLOPs vs the Newton-3rd-law pair scheme, but
        // avoids the unified-memory atomic-contention hang observed on
        // MI300A APUs under the previous TeamPolicy+atomic_add design.
        auto particle_to_league_v = _particle_to_league;
        const int n_local_intra = static_cast<int>( positions.size() );

        Kokkos::parallel_for(
            "P2P_intra_leaf",
            Kokkos::RangePolicy<execution_space>( 0, n_local_intra ),
            KOKKOS_LAMBDA( const int pi ) {
                const int league = particle_to_league_v( pi );
                if ( league < 0 )
                    return;
                const int cidx = local_leaf_cells( league );
                const int pstart = leaf_offsets( cidx );
                const int pend = leaf_offsets( cidx + 1 );
                if ( pend - pstart < 2 )
                    return;

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
                    phi[c] = static_cast<scalar_type>( 0 );
                    gx[c] = static_cast<scalar_type>( 0 );
                    gy[c] = static_cast<scalar_type>( 0 );
                    gz[c] = static_cast<scalar_type>( 0 );
                }

                for ( int pj = pstart; pj < pend; pj++ )
                {
                    if ( pj == pi )
                        continue;
                    const scalar_type dx =
                        xi -
                        static_cast<scalar_type>( positions( pj, 0 ) );
                    const scalar_type dy =
                        yi -
                        static_cast<scalar_type>( positions( pj, 1 ) );
                    const scalar_type dz =
                        zi -
                        static_cast<scalar_type>( positions( pj, 2 ) );
                    const scalar_type r2 = dx * dx + dy * dy + dz * dz;
                    if ( r2 < static_cast<scalar_type>( 1.0e-24 ) )
                        continue;
                    const scalar_type inv_r =
                        static_cast<scalar_type>( 1.0 ) /
                        Kokkos::sqrt( r2 + eps2 );
                    const scalar_type inv_r3 = inv_r * inv_r * inv_r;
                    for ( int c = 0; c < NComps; c++ )
                    {
                        const scalar_type qj =
                            static_cast<scalar_type>( charges( pj, c ) );
                        phi[c] += qj * inv_r;
                        if ( compute_gradient )
                        {
                            gx[c] -= qj * dx * inv_r3;
                            gy[c] -= qj * dy * inv_r3;
                            gz[c] -= qj * dz * inv_r3;
                        }
                    }
                }

                // Single-writer: no atomic needed.
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

        Kokkos::fence();
        } // if ( num_target_leaves > 0 )
    } // TIMER_P2P_INTRA_KERNEL

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

    // Hoisted so the CANOPY_ENABLE_DEBUG side-by-side verifier blocks below
    // can see them outside the TIMER_P2P_INTER_KERNEL scope.
    auto particle_to_league_v = _particle_to_league;
    const int n_local = static_cast<int>( positions.size() );

    // In release, the production kernel writes directly to potential_out /
    // gradient_out. In CANOPY_ENABLE_DEBUG builds we rebind pot_out/grad_out
    // to clean scratch views (pot_new_debug/grad_new_debug) so we can compare
    // them against an OLD-kernel reference written into a second scratch
    // (pot_old_debug/grad_old_debug). After the compare, pot_new_debug is
    // accumulated into potential_out so the rest of the pipeline sees the
    // correct inter-leaf contribution.
    auto pot_out = potential_out;
    auto grad_out = gradient_out;

#if defined( CANOPY_ENABLE_DEBUG )
    potential_view_type pot_new_debug(
        Kokkos::view_alloc( Kokkos::WithoutInitializing,
                            "p2p_pot_new_debug" ),
        n_local );
    gradient_view_type grad_new_debug(
        Kokkos::view_alloc( Kokkos::WithoutInitializing,
                            "p2p_grad_new_debug" ),
        compute_gradient ? n_local : 0 );
    Kokkos::deep_copy( pot_new_debug, static_cast<scalar_type>( 0 ) );
    if ( compute_gradient )
        Kokkos::deep_copy( grad_new_debug, static_cast<scalar_type>( 0 ) );
    pot_out = pot_new_debug;
    grad_out = grad_new_debug;

    potential_view_type pot_old_debug(
        Kokkos::view_alloc( Kokkos::WithoutInitializing,
                            "p2p_pot_old_debug" ),
        n_local );
    gradient_view_type grad_old_debug(
        Kokkos::view_alloc( Kokkos::WithoutInitializing,
                            "p2p_grad_old_debug" ),
        compute_gradient ? n_local : 0 );
    Kokkos::deep_copy( pot_old_debug, static_cast<scalar_type>( 0 ) );
    if ( compute_gradient )
        Kokkos::deep_copy( grad_old_debug, static_cast<scalar_type>( 0 ) );

    // OLD reference: byte-identical to the production thread-per-particle
    // inter-leaf kernel below. When the production kernel is also the
    // thread-per-particle path (commit 2 scaffold), the compare should
    // report zero mismatch. When the production kernel is the team-per-leaf
    // rewrite (commit 3), the compare catches divergence beyond FP-reordering
    // noise. Kernel body is duplicated rather than shared via a local lambda
    // because nvcc forbids defining an extended __device__ lambda inside
    // another lambda.
    if ( n_local > 0 )
    {
        Kokkos::parallel_for(
            "P2P_inter_leaf_debug_old",
            Kokkos::RangePolicy<execution_space>( 0, n_local ),
            KOKKOS_LAMBDA( const int pi ) {
                const int league = particle_to_league_v( pi );
                if ( league < 0 )
                    return;

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
                    phi[c] = static_cast<scalar_type>( 0 );
                    gx[c] = static_cast<scalar_type>( 0 );
                    gy[c] = static_cast<scalar_type>( 0 );
                    gz[c] = static_cast<scalar_type>( 0 );
                }

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
                            xi -
                            static_cast<scalar_type>( positions( pj, 0 ) );
                        const scalar_type dy =
                            yi -
                            static_cast<scalar_type>( positions( pj, 1 ) );
                        const scalar_type dz =
                            zi -
                            static_cast<scalar_type>( positions( pj, 2 ) );
                        const scalar_type r2 = dx * dx + dy * dy + dz * dz;
                        if ( r2 < static_cast<scalar_type>( 1.0e-24 ) )
                            continue;
                        const scalar_type inv_r =
                            static_cast<scalar_type>( 1.0 ) /
                            Kokkos::sqrt( r2 + eps2 );
                        const scalar_type inv_r3 = inv_r * inv_r * inv_r;
                        for ( int c = 0; c < NComps; c++ )
                        {
                            const scalar_type qj =
                                static_cast<scalar_type>( charges( pj, c ) );
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

                const int g_start = ghost_nbr_off( league );
                const int g_end = ghost_nbr_off( league + 1 );
                for ( int nn = g_start; nn < g_end; nn++ )
                {
                    const int g_idx = ghost_nbr_idx( nn );
                    const int gs = ghost_leaf_off( g_idx );
                    const int ge = ghost_leaf_off( g_idx + 1 );
                    for ( int pj = gs; pj < ge; pj++ )
                    {
                        const scalar_type dx = xi - ghost_positions( pj, 0 );
                        const scalar_type dy = yi - ghost_positions( pj, 1 );
                        const scalar_type dz = zi - ghost_positions( pj, 2 );
                        const scalar_type r2 = dx * dx + dy * dy + dz * dz;
                        if ( r2 < static_cast<scalar_type>( 1.0e-24 ) )
                            continue;
                        const scalar_type inv_r =
                            static_cast<scalar_type>( 1.0 ) /
                            Kokkos::sqrt( r2 + eps2 );
                        const scalar_type inv_r3 = inv_r * inv_r * inv_r;
                        for ( int c = 0; c < NComps; c++ )
                        {
                            const scalar_type qj = ghost_charges( pj, c );
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

                for ( int c = 0; c < NComps; c++ )
                {
                    pot_old_debug( pi, c ) += phi[c];
                    if ( compute_gradient )
                    {
                        grad_old_debug( pi, c, 0 ) += gx[c];
                        grad_old_debug( pi, c, 1 ) += gy[c];
                        grad_old_debug( pi, c, 2 ) += gz[c];
                    }
                }
            } );
        Kokkos::fence();
    }
#endif // CANOPY_ENABLE_DEBUG

    {
        CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_P2P_INTER_KERNEL );
        if ( n_local > 0 && num_target_leaves > 0 )
        {

        // Team-per-leaf inter-leaf kernel.
        //
        // One team per local target leaf. Team threads are strided
        // across the leaf's target particles in tiles of TILE_SIZE
        // targets each -- each thread holds its tile's (x,y,z) positions
        // and (phi, gx, gy, gz) accumulators in registers. For each
        // neighbor leaf (local then ghost), the team cooperatively loads
        // up to SMEM_SRC_CAP source particles' positions/charges into
        // LDS via TeamThreadRange; barrier; each team thread then loops
        // over its assigned target tile x all loaded source particles
        // in LDS, accumulating into its tile's registers. Single-writer
        // output (no atomics) -- distinct teams own distinct target
        // leaves, distinct team threads within a team own distinct
        // target particles. Distinguishes this design from the prior
        // pair-iteration TeamPolicy that hit the MI300A unified-memory
        // atomic-contention hang (see intra-leaf kernel comment).
        //
        // Team size is Kokkos::AUTO so the kernel ports across backends:
        // on Serial team_size=1, on HIP/CUDA it picks a wavefront
        // multiple (typically 64-256). The TILE_SIZE outer loop bounds
        // the register footprint to the same fixed size regardless of
        // team_size; on Serial that means a wider leaf takes multiple
        // tile iterations and walks the neighbor list multiple times --
        // wasteful on Serial (correctness-only backend) but correct.
        //
        // The per-target source-loop order is identical to the OLD
        // thread-per-particle kernel (target-outer, source-inner, same
        // CSR walk over neighbors then their particles), so the FP
        // addition order is bit-identical and the CANOPY_ENABLE_DEBUG
        // verifier should report zero mismatch on a healthy build.
        constexpr int TILE_SIZE = 8;      // targets per team thread per tile
        constexpr int SMEM_SRC_CAP = 512; // matches ncrit; covers p99 at depth 9

        using team_policy_t = Kokkos::TeamPolicy<execution_space>;
        using team_member_t = typename team_policy_t::member_type;
        using shmem_space = typename execution_space::scratch_memory_space;
        using s_array_t = Kokkos::View<scalar_type*, shmem_space,
                                       Kokkos::MemoryUnmanaged>;

        // LDS footprint per team: (3 + NComps) scalars per source slot.
        // 8 KB at SMEM_SRC_CAP=512, NComps=1, float (16 KB double) --
        // well under MI300A's ~64 KB / CU LDS budget.
        const size_t shmem_bytes =
            ( 3 + NComps ) * SMEM_SRC_CAP * sizeof( scalar_type );

        team_policy_t policy =
            team_policy_t( num_target_leaves, Kokkos::AUTO )
                .set_scratch_size( 0, Kokkos::PerTeam( shmem_bytes ) );

        Kokkos::parallel_for(
            "P2P_inter_leaf",
            policy,
            KOKKOS_LAMBDA( const team_member_t& tm ) {
                const int league = tm.league_rank();
                const int t = tm.team_rank();
                const int team_size = tm.team_size();

                const int tgt_cidx = local_leaf_cells( league );
                const int ts = leaf_offsets( tgt_cidx );
                const int te = leaf_offsets( tgt_cidx + 1 );
                const int n_tgt = te - ts;

                // LDS sub-views into the single team-scratch arena.
                // Each constructor call bump-allocates fresh space.
                s_array_t s_x( tm.team_scratch( 0 ), SMEM_SRC_CAP );
                s_array_t s_y( tm.team_scratch( 0 ), SMEM_SRC_CAP );
                s_array_t s_z( tm.team_scratch( 0 ), SMEM_SRC_CAP );
                s_array_t s_q( tm.team_scratch( 0 ),
                               SMEM_SRC_CAP * NComps );

                const int tile_extent = TILE_SIZE * team_size;

                for ( int tile_start = 0; tile_start < n_tgt;
                      tile_start += tile_extent )
                {
                    // Load this team thread's targets for this tile.
                    scalar_type xi_r[TILE_SIZE];
                    scalar_type yi_r[TILE_SIZE];
                    scalar_type zi_r[TILE_SIZE];
                    scalar_type phi_r[TILE_SIZE][NComps];
                    scalar_type gx_r[TILE_SIZE][NComps];
                    scalar_type gy_r[TILE_SIZE][NComps];
                    scalar_type gz_r[TILE_SIZE][NComps];

                    int n_my = 0;
                    for ( int k = 0; k < TILE_SIZE; k++ )
                    {
                        const int li = tile_start + t + k * team_size;
                        if ( li >= n_tgt )
                            break;
                        const int pi = ts + li;
                        xi_r[k] =
                            static_cast<scalar_type>( positions( pi, 0 ) );
                        yi_r[k] =
                            static_cast<scalar_type>( positions( pi, 1 ) );
                        zi_r[k] =
                            static_cast<scalar_type>( positions( pi, 2 ) );
                        for ( int c = 0; c < NComps; c++ )
                        {
                            phi_r[k][c] = static_cast<scalar_type>( 0 );
                            gx_r[k][c] = static_cast<scalar_type>( 0 );
                            gy_r[k][c] = static_cast<scalar_type>( 0 );
                            gz_r[k][c] = static_cast<scalar_type>( 0 );
                        }
                        n_my++;
                    }

                    // --- Local neighbor leaves ---
                    {
                        const int l_start = local_nbr_off( league );
                        const int l_end = local_nbr_off( league + 1 );
                        for ( int nn = l_start; nn < l_end; nn++ )
                        {
                            const int n_cidx = local_nbr_idx( nn );
                            const int ns = leaf_offsets( n_cidx );
                            const int ne = leaf_offsets( n_cidx + 1 );
                            const int n_src = ne - ns;

                            for ( int cs = 0; cs < n_src;
                                  cs += SMEM_SRC_CAP )
                            {
                                const int rem = n_src - cs;
                                const int cn = ( SMEM_SRC_CAP < rem )
                                                   ? SMEM_SRC_CAP
                                                   : rem;

                                Kokkos::parallel_for(
                                    Kokkos::TeamThreadRange( tm, cn ),
                                    [&]( const int i ) {
                                        const int pj = ns + cs + i;
                                        s_x( i ) = static_cast<scalar_type>(
                                            positions( pj, 0 ) );
                                        s_y( i ) = static_cast<scalar_type>(
                                            positions( pj, 1 ) );
                                        s_z( i ) = static_cast<scalar_type>(
                                            positions( pj, 2 ) );
                                        for ( int c = 0; c < NComps; c++ )
                                            s_q( i * NComps + c ) =
                                                static_cast<scalar_type>(
                                                    charges( pj, c ) );
                                    } );
                                tm.team_barrier();

                                for ( int k = 0; k < n_my; k++ )
                                {
                                    const scalar_type Xi = xi_r[k];
                                    const scalar_type Yi = yi_r[k];
                                    const scalar_type Zi = zi_r[k];
                                    for ( int j = 0; j < cn; j++ )
                                    {
                                        const scalar_type dx = Xi - s_x( j );
                                        const scalar_type dy = Yi - s_y( j );
                                        const scalar_type dz = Zi - s_z( j );
                                        const scalar_type r2 =
                                            dx * dx + dy * dy + dz * dz;
                                        if ( r2 < static_cast<scalar_type>(
                                                      1.0e-24 ) )
                                            continue;
                                        const scalar_type inv_r =
                                            static_cast<scalar_type>(
                                                1.0 ) /
                                            Kokkos::sqrt( r2 + eps2 );
                                        const scalar_type inv_r3 =
                                            inv_r * inv_r * inv_r;
                                        for ( int c = 0; c < NComps; c++ )
                                        {
                                            const scalar_type qj =
                                                s_q( j * NComps + c );
                                            phi_r[k][c] += qj * inv_r;
                                            if ( compute_gradient )
                                            {
                                                gx_r[k][c] -=
                                                    qj * dx * inv_r3;
                                                gy_r[k][c] -=
                                                    qj * dy * inv_r3;
                                                gz_r[k][c] -=
                                                    qj * dz * inv_r3;
                                            }
                                        }
                                    }
                                }
                                tm.team_barrier(); // before LDS overwrite
                            }
                        }
                    }

                    // --- Ghost neighbor leaves ---
                    {
                        const int g_start = ghost_nbr_off( league );
                        const int g_end = ghost_nbr_off( league + 1 );
                        for ( int nn = g_start; nn < g_end; nn++ )
                        {
                            const int g_idx = ghost_nbr_idx( nn );
                            const int gs = ghost_leaf_off( g_idx );
                            const int ge = ghost_leaf_off( g_idx + 1 );
                            const int n_src = ge - gs;

                            for ( int cs = 0; cs < n_src;
                                  cs += SMEM_SRC_CAP )
                            {
                                const int rem = n_src - cs;
                                const int cn = ( SMEM_SRC_CAP < rem )
                                                   ? SMEM_SRC_CAP
                                                   : rem;

                                Kokkos::parallel_for(
                                    Kokkos::TeamThreadRange( tm, cn ),
                                    [&]( const int i ) {
                                        const int pj = gs + cs + i;
                                        s_x( i ) = ghost_positions( pj, 0 );
                                        s_y( i ) = ghost_positions( pj, 1 );
                                        s_z( i ) = ghost_positions( pj, 2 );
                                        for ( int c = 0; c < NComps; c++ )
                                            s_q( i * NComps + c ) =
                                                ghost_charges( pj, c );
                                    } );
                                tm.team_barrier();

                                for ( int k = 0; k < n_my; k++ )
                                {
                                    const scalar_type Xi = xi_r[k];
                                    const scalar_type Yi = yi_r[k];
                                    const scalar_type Zi = zi_r[k];
                                    for ( int j = 0; j < cn; j++ )
                                    {
                                        const scalar_type dx = Xi - s_x( j );
                                        const scalar_type dy = Yi - s_y( j );
                                        const scalar_type dz = Zi - s_z( j );
                                        const scalar_type r2 =
                                            dx * dx + dy * dy + dz * dz;
                                        if ( r2 < static_cast<scalar_type>(
                                                      1.0e-24 ) )
                                            continue;
                                        const scalar_type inv_r =
                                            static_cast<scalar_type>(
                                                1.0 ) /
                                            Kokkos::sqrt( r2 + eps2 );
                                        const scalar_type inv_r3 =
                                            inv_r * inv_r * inv_r;
                                        for ( int c = 0; c < NComps; c++ )
                                        {
                                            const scalar_type qj =
                                                s_q( j * NComps + c );
                                            phi_r[k][c] += qj * inv_r;
                                            if ( compute_gradient )
                                            {
                                                gx_r[k][c] -=
                                                    qj * dx * inv_r3;
                                                gy_r[k][c] -=
                                                    qj * dy * inv_r3;
                                                gz_r[k][c] -=
                                                    qj * dz * inv_r3;
                                            }
                                        }
                                    }
                                }
                                tm.team_barrier();
                            }
                        }
                    }

                    // Single-writer output for this tile.
                    // In release, pot_out/grad_out alias potential_out/
                    // gradient_out directly. In CANOPY_ENABLE_DEBUG, they
                    // alias the new-kernel scratch (accumulated into
                    // potential_out after the verifier).
                    for ( int k = 0; k < n_my; k++ )
                    {
                        const int pi = ts + ( tile_start + t + k * team_size );
                        for ( int c = 0; c < NComps; c++ )
                        {
                            pot_out( pi, c ) += phi_r[k][c];
                            if ( compute_gradient )
                            {
                                grad_out( pi, c, 0 ) += gx_r[k][c];
                                grad_out( pi, c, 1 ) += gy_r[k][c];
                                grad_out( pi, c, 2 ) += gz_r[k][c];
                            }
                        }
                    }
                }
            } );

        Kokkos::fence();
        } // if ( n_local > 0 && num_target_leaves > 0 )
    } // TIMER_P2P_INTER_KERNEL

#if defined( CANOPY_ENABLE_DEBUG )
    // Side-by-side check: pot_new_debug vs pot_old_debug (and gradient).
    // FP-reorder tolerance: 1e-10 (double) / 5e-5 (float). Abort on mismatch.
    // After the compare, accumulate pot_new_debug into potential_out so the
    // rest of the production pipeline sees the inter-leaf contribution.
    if ( n_local > 0 )
    {
        constexpr double DEBUG_REL_TOL =
            std::is_same<scalar_type, float>::value ? 5.0e-5 : 1.0e-10;
        long n_mismatch = 0;
        double max_rel = 0.0;
        const bool cg = compute_gradient;
        Kokkos::parallel_reduce(
            "P2P_inter_leaf_debug_compare",
            Kokkos::RangePolicy<execution_space>( 0, n_local ),
            KOKKOS_LAMBDA( const int pi, long& mis, double& mxr ) {
                for ( int c = 0; c < NComps; c++ )
                {
                    const double nv =
                        static_cast<double>( pot_new_debug( pi, c ) );
                    const double ov =
                        static_cast<double>( pot_old_debug( pi, c ) );
                    const double an = ( nv < 0.0 ? -nv : nv );
                    const double ao = ( ov < 0.0 ? -ov : ov );
                    const double dv = nv - ov;
                    const double ad = ( dv < 0.0 ? -dv : dv );
                    const double rel = ad / ( an + ao + 1.0e-30 );
                    if ( rel > DEBUG_REL_TOL )
                        mis++;
                    if ( rel > mxr )
                        mxr = rel;
                    if ( cg )
                    {
                        for ( int k = 0; k < 3; k++ )
                        {
                            const double gn = static_cast<double>(
                                grad_new_debug( pi, c, k ) );
                            const double go = static_cast<double>(
                                grad_old_debug( pi, c, k ) );
                            const double agn = ( gn < 0.0 ? -gn : gn );
                            const double ago = ( go < 0.0 ? -go : go );
                            const double dg = gn - go;
                            const double adg = ( dg < 0.0 ? -dg : dg );
                            const double rg =
                                adg / ( agn + ago + 1.0e-30 );
                            if ( rg > DEBUG_REL_TOL )
                                mis++;
                            if ( rg > mxr )
                                mxr = rg;
                        }
                    }
                }
            },
            Kokkos::Sum<long>( n_mismatch ),
            Kokkos::Max<double>( max_rel ) );
        Kokkos::fence();

        if ( n_mismatch > 0 )
        {
            int dbg_rank = 0;
            MPI_Comm_rank( _comm, &dbg_rank );
            std::fprintf( stderr,
                          "[CANOPY DEBUG] P2P inter-leaf verifier FAILED "
                          "(rank=%d): n_mismatch=%ld max_rel=%g tol=%g\n",
                          dbg_rank, n_mismatch, max_rel, DEBUG_REL_TOL );
            std::abort();
        }

        // Accumulate pot_new_debug into the production output views.
        auto potential_out_acc = potential_out;
        auto gradient_out_acc = gradient_out;
        Kokkos::parallel_for(
            "P2P_inter_leaf_debug_accumulate",
            Kokkos::RangePolicy<execution_space>( 0, n_local ),
            KOKKOS_LAMBDA( const int pi ) {
                for ( int c = 0; c < NComps; c++ )
                {
                    potential_out_acc( pi, c ) += pot_new_debug( pi, c );
                    if ( cg )
                    {
                        gradient_out_acc( pi, c, 0 ) +=
                            grad_new_debug( pi, c, 0 );
                        gradient_out_acc( pi, c, 1 ) +=
                            grad_new_debug( pi, c, 1 );
                        gradient_out_acc( pi, c, 2 ) +=
                            grad_new_debug( pi, c, 2 );
                    }
                }
            } );
        Kokkos::fence();
    }
#endif // CANOPY_ENABLE_DEBUG
    } // TIMER_P2P_TOTAL

#if CANOPY_PROFILING_LEVEL >= 3
    // Diagnostic (level 3 only): re-run the inter-leaf kernel into scratch
    // outputs (warm: identical distribution and already-allocated neighbor /
    // ghost buffers) and count the inter-leaf trip pairs. Placed OUTSIDE the
    // TIMER_P2P_TOTAL scope so it never inflates the production totals; the
    // per-execute locals from that scope are therefore re-derived from members
    // here. A fast rerun vs the timed kernel implies the first launch paid a
    // one-time first-touch / allocation cost; an equally slow rerun implies
    // genuinely heavier kernel work at this step. The real potential_out /
    // gradient_out are untouched, so the solve results are unchanged.
    //
    // The kernel body is duplicated here (rather than shared via a local
    // lambda) because nvcc forbids defining an extended __device__ lambda
    // (KOKKOS_LAMBDA) inside another lambda; a direct parallel_for in this
    // member-function scope is well-formed. It compiles to nothing below
    // level 3, so production builds carry no duplication.
    {
        const int n_local_d = static_cast<int>( positions.size() );
        if ( n_local_d > 0 )
        {
            const scalar_type eps2_d = _softening2;
            auto particle_to_league_v = _particle_to_league;
            auto leaf_offsets_d = _leaf_particle_offsets;
            auto local_nbr_off_d = _local_nbr_offsets;
            auto local_nbr_idx_d = _local_nbr_cell_idx;
            auto ghost_nbr_off_d = _ghost_nbr_offsets;
            auto ghost_nbr_idx_d = _ghost_nbr_leaf_idx;
            auto ghost_leaf_off_d = _ghost_leaf_offsets;
            auto ghost_positions_d = _ghost_positions;
            auto ghost_charges_d = _ghost_charges;

            potential_view_type pot_scratch(
                Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                    "p2p_pot_scratch" ),
                n_local_d );
            gradient_view_type grad_scratch(
                Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                    "p2p_grad_scratch" ),
                compute_gradient ? n_local_d : 0 );
            Kokkos::deep_copy( pot_scratch, static_cast<scalar_type>( 0 ) );
            if ( compute_gradient )
                Kokkos::deep_copy( grad_scratch,
                                   static_cast<scalar_type>( 0 ) );

            {
                CANOPY_SCOPED_TIMER(
                    Canopy::Profiling::TIMER_P2P_INTER_KERNEL_RERUN );
                Kokkos::parallel_for(
                    "P2P_inter_leaf_rerun",
                    Kokkos::RangePolicy<execution_space>( 0, n_local_d ),
                    KOKKOS_LAMBDA( const int pi ) {
                        const int league = particle_to_league_v( pi );
                        if ( league < 0 )
                            return;

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
                            phi[c] = static_cast<scalar_type>( 0 );
                            gx[c] = static_cast<scalar_type>( 0 );
                            gy[c] = static_cast<scalar_type>( 0 );
                            gz[c] = static_cast<scalar_type>( 0 );
                        }

                        const int l_start = local_nbr_off_d( league );
                        const int l_end = local_nbr_off_d( league + 1 );
                        for ( int nn = l_start; nn < l_end; nn++ )
                        {
                            const int n_cidx = local_nbr_idx_d( nn );
                            const int ns = leaf_offsets_d( n_cidx );
                            const int ne = leaf_offsets_d( n_cidx + 1 );
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
                                    Kokkos::sqrt( r2 + eps2_d );
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

                        const int g_start = ghost_nbr_off_d( league );
                        const int g_end = ghost_nbr_off_d( league + 1 );
                        for ( int nn = g_start; nn < g_end; nn++ )
                        {
                            const int g_idx = ghost_nbr_idx_d( nn );
                            const int gs = ghost_leaf_off_d( g_idx );
                            const int ge = ghost_leaf_off_d( g_idx + 1 );
                            for ( int pj = gs; pj < ge; pj++ )
                            {
                                const scalar_type dx =
                                    xi - ghost_positions_d( pj, 0 );
                                const scalar_type dy =
                                    yi - ghost_positions_d( pj, 1 );
                                const scalar_type dz =
                                    zi - ghost_positions_d( pj, 2 );
                                const scalar_type r2 =
                                    dx * dx + dy * dy + dz * dz;
                                if ( r2 < static_cast<scalar_type>( 1.0e-24 ) )
                                    continue;
                                const scalar_type inv_r =
                                    static_cast<scalar_type>( 1.0 ) /
                                    Kokkos::sqrt( r2 + eps2_d );
                                const scalar_type inv_r3 =
                                    inv_r * inv_r * inv_r;
                                for ( int c = 0; c < NComps; c++ )
                                {
                                    const scalar_type qj =
                                        ghost_charges_d( pj, c );
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

                        for ( int c = 0; c < NComps; c++ )
                        {
                            pot_scratch( pi, c ) += phi[c];
                            if ( compute_gradient )
                            {
                                grad_scratch( pi, c, 0 ) += gx[c];
                                grad_scratch( pi, c, 1 ) += gy[c];
                                grad_scratch( pi, c, 2 ) += gz[c];
                            }
                        }
                    } );
                Kokkos::fence();
            }

            long long local_pairs = 0;
            Kokkos::parallel_reduce(
                "p2p_inter_leaf_pair_count",
                Kokkos::RangePolicy<execution_space>( 0, n_local_d ),
                KOKKOS_LAMBDA( const int pi, long long& acc ) {
                    const int league = particle_to_league_v( pi );
                    if ( league < 0 )
                        return;
                    const int ls = local_nbr_off_d( league );
                    const int le = local_nbr_off_d( league + 1 );
                    for ( int nn = ls; nn < le; nn++ )
                    {
                        const int c = local_nbr_idx_d( nn );
                        acc += leaf_offsets_d( c + 1 ) - leaf_offsets_d( c );
                    }
                    const int gst = ghost_nbr_off_d( league );
                    const int gen = ghost_nbr_off_d( league + 1 );
                    for ( int nn = gst; nn < gen; nn++ )
                    {
                        const int g = ghost_nbr_idx_d( nn );
                        acc += ghost_leaf_off_d( g + 1 ) -
                               ghost_leaf_off_d( g );
                    }
                },
                local_pairs );

            long long global_pairs = 0;
            MPI_Reduce( &local_pairs, &global_pairs, 1, MPI_LONG_LONG,
                        MPI_SUM, 0, _comm );
            int diag_rank = 0;
            MPI_Comm_rank( _comm, &diag_rank );
            if ( diag_rank == 0 )
                std::fprintf( stderr,
                              "[Canopy diag] p2p_inter_leaf_pairs=%lld\n",
                              global_pairs );

            // Leaf-size histogram (rank-0 only): distributions of target-leaf,
            // local-neighbor source-leaf (with multiplicity as encountered by
            // the inter-leaf kernel), and ghost-leaf sizes. Informs team_size
            // and SMEM_SRC_CAP for the team-per-leaf rewrite. Level-3 only.
            if ( diag_rank == 0 )
            {
                auto leaf_off_h = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), _leaf_particle_offsets );
                auto local_leaves_h = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), _local_leaf_cells );
                auto local_nbr_idx_h = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), _local_nbr_cell_idx );
                auto ghost_leaf_off_h = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), _ghost_leaf_offsets );

                const int edges[] = { 1,   64,  128, 192, 256,    320, 384,
                                      448, 512, 576, 768, 1024,   INT_MAX };
                const int n_edges = sizeof( edges ) / sizeof( edges[0] );
                const int n_buckets = n_edges - 1;

                auto report = [&]( const char* label,
                                   std::vector<int>& sizes )
                {
                    if ( sizes.empty() )
                    {
                        std::fprintf(
                            stderr,
                            "[Canopy diag] leaf_size_hist %s: (empty)\n",
                            label );
                        return;
                    }
                    std::sort( sizes.begin(), sizes.end() );
                    const size_t n = sizes.size();
                    double mean = 0.0;
                    for ( int s : sizes )
                        mean += s;
                    mean /= static_cast<double>( n );
                    auto pct = [&]( double q ) -> int
                    {
                        size_t idx = static_cast<size_t>( q * n );
                        if ( idx >= n )
                            idx = n - 1;
                        return sizes[idx];
                    };
                    std::vector<int> counts( n_buckets, 0 );
                    for ( int s : sizes )
                    {
                        for ( int b = 0; b < n_buckets; b++ )
                        {
                            if ( s >= edges[b] && s < edges[b + 1] )
                            {
                                counts[b]++;
                                break;
                            }
                        }
                    }
                    std::fprintf(
                        stderr,
                        "[Canopy diag] leaf_size_hist %s: n=%zu min=%d "
                        "max=%d mean=%.1f p50=%d p95=%d p99=%d\n",
                        label, n, sizes.front(), sizes.back(), mean,
                        pct( 0.50 ), pct( 0.95 ), pct( 0.99 ) );
                    std::fprintf( stderr,
                                  "[Canopy diag] leaf_size_hist %s buckets:",
                                  label );
                    for ( int b = 0; b < n_buckets; b++ )
                    {
                        if ( edges[b + 1] == INT_MAX )
                            std::fprintf( stderr, " [%d,inf)=%d", edges[b],
                                          counts[b] );
                        else
                            std::fprintf( stderr, " [%d,%d)=%d", edges[b],
                                          edges[b + 1], counts[b] );
                    }
                    std::fprintf( stderr, "\n" );
                };

                // Target-leaf sizes
                {
                    const int nt =
                        static_cast<int>( local_leaves_h.extent( 0 ) );
                    std::vector<int> sizes;
                    sizes.reserve( nt );
                    for ( int g = 0; g < nt; g++ )
                    {
                        const int cidx = local_leaves_h( g );
                        sizes.push_back( leaf_off_h( cidx + 1 ) -
                                         leaf_off_h( cidx ) );
                    }
                    report( "target", sizes );
                }
                // Local-neighbor source-leaf sizes (with multiplicity)
                {
                    const int nn =
                        static_cast<int>( local_nbr_idx_h.extent( 0 ) );
                    std::vector<int> sizes;
                    sizes.reserve( nn );
                    for ( int i = 0; i < nn; i++ )
                    {
                        const int c = local_nbr_idx_h( i );
                        sizes.push_back( leaf_off_h( c + 1 ) -
                                         leaf_off_h( c ) );
                    }
                    report( "local_nbr", sizes );
                }
                // Ghost-leaf sizes
                {
                    const int ng =
                        static_cast<int>( ghost_leaf_off_h.extent( 0 ) ) - 1;
                    std::vector<int> sizes;
                    sizes.reserve( std::max( 0, ng ) );
                    for ( int i = 0; i < ng; i++ )
                        sizes.push_back( ghost_leaf_off_h( i + 1 ) -
                                         ghost_leaf_off_h( i ) );
                    report( "ghost", sizes );
                }
            }
        }
    }
#endif

    CANOPY_PRINT_P2P_TIMERS( _comm );
}

} // end namespace Canopy

#endif // CANOPY_P2P_HPP
