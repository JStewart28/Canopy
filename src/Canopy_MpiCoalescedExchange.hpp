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

#ifndef CANOPY_MPI_COALESCED_EXCHANGE_HPP
#define CANOPY_MPI_COALESCED_EXCHANGE_HPP

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <map>
#include <vector>

namespace Canopy
{
namespace detail
{

// Coalesced per-(rank,call) point-to-point exchange of (cell, ci, c) data
// from a Kokkos View of shape (num_cells, coeffs_per_cell, NComps).
//
// One MPI_Isend / MPI_Irecv per peer rank instead of one per cell — needed
// because the per-cell pattern exhausts the Cray-MPICH request freelist at
// O(1e5) in-flight requests on large M2L lists.
//
// Pack/unpack run on the execution space of the view; the .data() pointer
// passed to MPI is therefore a device pointer on GPU backends. This requires
// MPICH_GPU_SUPPORT_ENABLED=1 at runtime when launching with a GPU backend
// (Cray MPICH on Slingshot/HSN). On host backends the pointer is a host
// pointer and no env var is needed.
//
// Pack order on both ends is the cell list per peer sorted by MortonKey
// (which both sides compute from their plan entries) so packing and
// unpacking align without an extra metadata exchange.
//
// If accumulate_on_recv is true, received values are += into the view
// (used for the L2L-after-L2L exchange where the child owner already has
// M2L contributions). Otherwise they overwrite.
template <class CoeffView>
void coalesced_view_exchange(
    const CoeffView& view, MPI_Comm comm,
    const std::map<int, std::vector<int>>& send_cells_by_peer_in,
    const std::map<int, std::vector<int>>& recv_cells_by_peer_in,
    bool accumulate_on_recv )
{
    using complex_type = typename CoeffView::non_const_value_type;
    using scalar_type = typename complex_type::value_type;
    using memory_space = typename CoeffView::memory_space;
    using execution_space = typename CoeffView::execution_space;

    // Drop self-peer entries before doing any MPI work. With 1 MPI rank
    // the comm plan can still emit entries with remote_rank == self; the
    // referenced cells already live in `view` at the same local indices,
    // so the exchange is a no-op semantically. Posting self-send/recv on
    // device buffers under GPU-aware Cray-MPICH on MI300A has been
    // observed to deadlock or trigger a memory-access fault.
    int self_rank = 0;
    MPI_Comm_rank( comm, &self_rank );
    std::map<int, std::vector<int>> send_cells_by_peer;
    std::map<int, std::vector<int>> recv_cells_by_peer;
    for ( const auto& kv : send_cells_by_peer_in )
        if ( kv.first != self_rank )
            send_cells_by_peer.emplace( kv.first, kv.second );
    for ( const auto& kv : recv_cells_by_peer_in )
        if ( kv.first != self_rank )
            recv_cells_by_peer.emplace( kv.first, kv.second );

    const int coeffs_per_cell = static_cast<int>( view.extent( 1 ) );
    const int NComps = static_cast<int>( view.extent( 2 ) );
    const int per_cell_complex = coeffs_per_cell * NComps;
    const int per_cell_real = 2 * per_cell_complex;
    MPI_Datatype mpi_scalar =
        ( sizeof( scalar_type ) == 8 ) ? MPI_DOUBLE : MPI_FLOAT;

    const int n_send_peers = static_cast<int>( send_cells_by_peer.size() );
    const int n_recv_peers = static_cast<int>( recv_cells_by_peer.size() );

    // ---------------------------------------------------------------------
    // DEBUG count handshake. Before exchanging any cell data, verify that for
    // every peer P the number of cells THIS rank will receive from P equals
    // the number P says it will send to this rank. A mismatch is the root
    // cause of an MPI_ERR_TRUNCATE in the data-phase MPI_Waitall below (recv
    // buffer smaller than the incoming message) and means the comm plan is
    // asymmetric across ranks.
    //
    // This is intentionally point-to-point over THIS rank's own peer set
    // (union of send and recv peers), NOT a collective: coalesced_view_exchange
    // is entered only by ranks that have peers (the M2M / L2L callers early
    // return otherwise), so a collective here would deadlock. For a symmetric
    // plan the peer set is reciprocal, so the handshake matches; for the count
    // mismatch we are hunting both sides still talk, so it is caught and
    // reported rather than truncating opaquely later.
    {
        std::map<int, char> peer_set; // union of send + recv peers
        for ( const auto& kv : send_cells_by_peer )
            peer_set[kv.first] = 1;
        for ( const auto& kv : recv_cells_by_peer )
            peer_set[kv.first] = 1;

        const int n_peers = static_cast<int>( peer_set.size() );
        std::vector<int> peers;
        peers.reserve( n_peers );
        for ( const auto& kv : peer_set )
            peers.push_back( kv.first );

        std::vector<int> my_send_counts( n_peers );
        std::vector<int> their_send_counts( n_peers, -1 );
        std::vector<MPI_Request> hs_reqs( 2 * n_peers );
        for ( int i = 0; i < n_peers; i++ )
        {
            auto sit = send_cells_by_peer.find( peers[i] );
            my_send_counts[i] = ( sit != send_cells_by_peer.end() )
                                    ? static_cast<int>( sit->second.size() )
                                    : 0;
            MPI_Irecv( &their_send_counts[i], 1, MPI_INT, peers[i], /*tag=*/7,
                       comm, &hs_reqs[i] );
        }
        for ( int i = 0; i < n_peers; i++ )
            MPI_Isend( &my_send_counts[i], 1, MPI_INT, peers[i], /*tag=*/7,
                       comm, &hs_reqs[n_peers + i] );
        if ( n_peers > 0 )
            MPI_Waitall( 2 * n_peers, hs_reqs.data(), MPI_STATUSES_IGNORE );

        bool mismatch = false;
        for ( int i = 0; i < n_peers; i++ )
        {
            auto rit = recv_cells_by_peer.find( peers[i] );
            const int my_recv_count =
                ( rit != recv_cells_by_peer.end() )
                    ? static_cast<int>( rit->second.size() )
                    : 0;
            if ( my_recv_count != their_send_counts[i] )
            {
                std::fprintf(
                    stderr,
                    "[Canopy FATAL] coalesced_view_exchange comm-plan "
                    "asymmetry: rank %d expects to RECEIVE %d cells from "
                    "peer %d, but peer %d says it will SEND %d cells "
                    "(this rank will SEND %d cells to peer %d)\n",
                    self_rank, my_recv_count, peers[i], peers[i],
                    their_send_counts[i], my_send_counts[i], peers[i] );
                mismatch = true;
            }
        }
        if ( mismatch )
            MPI_Abort( comm, 17 );
    }

    // Per-peer device-side index views and pack/unpack buffers.
    std::vector<Kokkos::View<int*, memory_space>> send_idx( n_send_peers );
    std::vector<Kokkos::View<complex_type*, memory_space>> send_bufs(
        n_send_peers );
    std::vector<int> send_peer_ranks( n_send_peers );
    std::vector<int> send_peer_ncells( n_send_peers );

    std::vector<Kokkos::View<int*, memory_space>> recv_idx( n_recv_peers );
    std::vector<Kokkos::View<complex_type*, memory_space>> recv_bufs(
        n_recv_peers );
    std::vector<int> recv_peer_ranks( n_recv_peers );
    std::vector<int> recv_peer_ncells( n_recv_peers );

    auto upload_idx = [&]( const std::vector<int>& h_idx,
                           Kokkos::View<int*, memory_space>& dst,
                           const char* label )
    {
        dst = Kokkos::View<int*, memory_space>(
            Kokkos::view_alloc( std::string( label ),
                                Kokkos::WithoutInitializing ),
            h_idx.size() );
        if ( h_idx.empty() )
            return;
        auto h = Kokkos::create_mirror_view( dst );
        for ( size_t i = 0; i < h_idx.size(); i++ )
            h( i ) = h_idx[i];
        Kokkos::deep_copy( dst, h );
    };

    // Post receives first.
    std::vector<MPI_Request> recv_reqs( n_recv_peers );
    {
        int p = 0;
        for ( const auto& kv : recv_cells_by_peer )
        {
            const int n = static_cast<int>( kv.second.size() );
            recv_peer_ranks[p] = kv.first;
            recv_peer_ncells[p] = n;
            upload_idx( kv.second, recv_idx[p], "coalesced_recv_idx" );
            recv_bufs[p] = Kokkos::View<complex_type*, memory_space>(
                Kokkos::view_alloc( "coalesced_recv_buf",
                                    Kokkos::WithoutInitializing ),
                static_cast<size_t>( n ) * per_cell_complex );
            MPI_Irecv( reinterpret_cast<scalar_type*>( recv_bufs[p].data() ),
                       n * per_cell_real, mpi_scalar, kv.first, /*tag=*/0,
                       comm, &recv_reqs[p] );
            ++p;
        }
    }

    // Build send buffers and post sends.
    std::vector<MPI_Request> send_reqs( n_send_peers );
    {
        int p = 0;
        for ( const auto& kv : send_cells_by_peer )
        {
            const int n = static_cast<int>( kv.second.size() );
            send_peer_ranks[p] = kv.first;
            send_peer_ncells[p] = n;
            upload_idx( kv.second, send_idx[p], "coalesced_send_idx" );
            send_bufs[p] = Kokkos::View<complex_type*, memory_space>(
                Kokkos::view_alloc( "coalesced_send_buf",
                                    Kokkos::WithoutInitializing ),
                static_cast<size_t>( n ) * per_cell_complex );

            auto idx_v = send_idx[p];
            auto buf_v = send_bufs[p];
            auto v = view;
            const int cpc = coeffs_per_cell;
            const int nc = NComps;
            Kokkos::parallel_for(
                "coalesced_pack",
                Kokkos::RangePolicy<execution_space>( 0, n ),
                KOKKOS_LAMBDA( const int i ) {
                    const int cidx = idx_v( i );
                    const int base = i * cpc * nc;
                    for ( int ci = 0; ci < cpc; ci++ )
                        for ( int c = 0; c < nc; c++ )
                            buf_v( base + ci * nc + c ) = v( cidx, ci, c );
                } );
            ++p;
        }
        Kokkos::fence();

        for ( int q = 0; q < n_send_peers; q++ )
        {
            const int n = send_peer_ncells[q];
            MPI_Isend( reinterpret_cast<scalar_type*>( send_bufs[q].data() ),
                       n * per_cell_real, mpi_scalar, send_peer_ranks[q],
                       /*tag=*/0, comm, &send_reqs[q] );
        }
    }

    if ( !recv_reqs.empty() )
        MPI_Waitall( recv_reqs.size(), recv_reqs.data(), MPI_STATUSES_IGNORE );
    if ( !send_reqs.empty() )
        MPI_Waitall( send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE );

    // Unpack on device.
    for ( int p = 0; p < n_recv_peers; p++ )
    {
        const int n = recv_peer_ncells[p];
        if ( n == 0 )
            continue;
        auto idx_v = recv_idx[p];
        auto buf_v = recv_bufs[p];
        auto v = view;
        const int cpc = coeffs_per_cell;
        const int nc = NComps;
        if ( accumulate_on_recv )
        {
            Kokkos::parallel_for(
                "coalesced_unpack_accum",
                Kokkos::RangePolicy<execution_space>( 0, n ),
                KOKKOS_LAMBDA( const int i ) {
                    const int cidx = idx_v( i );
                    const int base = i * cpc * nc;
                    for ( int ci = 0; ci < cpc; ci++ )
                        for ( int c = 0; c < nc; c++ )
                            v( cidx, ci, c ) += buf_v( base + ci * nc + c );
                } );
        }
        else
        {
            Kokkos::parallel_for(
                "coalesced_unpack",
                Kokkos::RangePolicy<execution_space>( 0, n ),
                KOKKOS_LAMBDA( const int i ) {
                    const int cidx = idx_v( i );
                    const int base = i * cpc * nc;
                    for ( int ci = 0; ci < cpc; ci++ )
                        for ( int c = 0; c < nc; c++ )
                            v( cidx, ci, c ) = buf_v( base + ci * nc + c );
                } );
        }
    }
    Kokkos::fence();
}

} // namespace detail
} // namespace Canopy

#endif
