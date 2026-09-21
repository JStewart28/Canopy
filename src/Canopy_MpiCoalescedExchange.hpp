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

#include "Canopy_RegisteredBufferPool.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <map>
#include <type_traits>
#include <vector>

namespace Canopy
{
namespace detail
{

// The real-scalar decomposition of a coefficient element, for MPI's benefit:
// MPI wants a (count, datatype) pair in real scalars, so it needs to know
// that one coefficient is scalars_per_coeff contiguous
// component_scalar_type.
//
// This mirrors the basis contract members of the same names
// (KernelType::component_scalar_type, KernelType::scalars_per_coeff) and
// exists separately only because coalesced_view_exchange is handed a View and
// never a basis. The two are cross-checked by a static_assert in each sweep,
// which is the only place a basis and this function meet.
//
// The primary template covers a real-coefficient basis. A basis whose
// coeff_type is neither a real scalar nor Kokkos::complex must specialize
// this alongside declaring its own traits.
template <class CoeffType>
struct coeff_traits
{
    static_assert( std::is_floating_point<CoeffType>::value,
                   "coeff_traits: no real-scalar decomposition known for this "
                   "coeff_type. Specialize Canopy::detail::coeff_traits for "
                   "it, or the MPI packing cannot size its transfers" );
    using component_scalar_type = CoeffType;
    static constexpr int scalars_per_coeff = 1;
};

template <class RealType>
struct coeff_traits<Kokkos::complex<RealType>>
{
    using component_scalar_type = RealType;
    static constexpr int scalars_per_coeff = 2;
};

// Persistent, grow-only staging buffers for coalesced_view_exchange(). One
// stable registered region per direction (coefficient data buffer + int pack
// index), reused across solve() calls so the CXI NIC registration cache does
// not churn. Hold one instance per exchanging object (UpwardSweep,
// DownwardSweep) and pass it into every coalesced_view_exchange() call.
template <class CoeffType, class MemorySpace>
struct CoalescedExchangeBuffers
{
    RegisteredBufferPool<CoeffType, MemorySpace> send_pool;
    RegisteredBufferPool<CoeffType, MemorySpace> recv_pool;
    RegisteredBufferPool<int, MemorySpace> send_idx_pool;
    RegisteredBufferPool<int, MemorySpace> recv_idx_pool;
};

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
template <class CoeffView, class ExchBuffers>
void coalesced_view_exchange(
    const CoeffView& view, MPI_Comm comm,
    const std::map<int, std::vector<int>>& send_cells_by_peer_in,
    const std::map<int, std::vector<int>>& recv_cells_by_peer_in,
    bool accumulate_on_recv, ExchBuffers& bufs )
{
    using coeff_type = typename CoeffView::non_const_value_type;
    using traits_type = coeff_traits<coeff_type>;
    using scalar_type = typename traits_type::component_scalar_type;
    static constexpr int scalars_per_coeff = traits_type::scalars_per_coeff;
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
    // Real-scalar count per cell, which is what MPI is given.
    const int per_cell_real = scalars_per_coeff * per_cell_complex;
    MPI_Datatype mpi_scalar =
        ( sizeof( scalar_type ) == 8 ) ? MPI_DOUBLE : MPI_FLOAT;

    const int n_send_peers = static_cast<int>( send_cells_by_peer.size() );
    const int n_recv_peers = static_cast<int>( recv_cells_by_peer.size() );

    // Per-peer index/data buffers are unmanaged sub-ranges of the persistent
    // grow-only pools (one registered region per direction), reused every
    // call so the CXI NIC registration footprint stays bounded.
    using umint_view = Kokkos::View<int*, memory_space,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using umcplx_view = Kokkos::View<coeff_type*, memory_space,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    std::vector<umint_view> send_idx( n_send_peers );
    std::vector<umcplx_view> send_bufs( n_send_peers );
    std::vector<int> send_peer_ranks( n_send_peers );
    std::vector<int> send_peer_ncells( n_send_peers );

    std::vector<umint_view> recv_idx( n_recv_peers );
    std::vector<umcplx_view> recv_bufs( n_recv_peers );
    std::vector<int> recv_peer_ranks( n_recv_peers );
    std::vector<int> recv_peer_ncells( n_recv_peers );

    // Host->device upload of a peer's cell-index list into a pool subview.
    auto upload_idx = [&]( const std::vector<int>& h_idx, umint_view dst )
    {
        if ( h_idx.empty() )
            return;
        auto h = Kokkos::create_mirror_view( dst );
        for ( size_t i = 0; i < h_idx.size(); i++ )
            h( i ) = h_idx[i];
        Kokkos::deep_copy( dst, h );
    };

    // Post receives first. Size the recv pools up front so every subview
    // handed out below has a stable base address.
    std::vector<MPI_Request> recv_reqs( n_recv_peers );
    {
        std::vector<size_t> recv_cell_off( n_recv_peers );
        size_t total_recv_cells = 0;
        {
            int p = 0;
            for ( const auto& kv : recv_cells_by_peer )
            {
                recv_cell_off[p] = total_recv_cells;
                total_recv_cells += kv.second.size();
                ++p;
            }
        }
        bufs.recv_idx_pool.reserve( total_recv_cells );
        bufs.recv_pool.reserve( total_recv_cells * per_cell_complex );

        int p = 0;
        for ( const auto& kv : recv_cells_by_peer )
        {
            const int n = static_cast<int>( kv.second.size() );
            recv_peer_ranks[p] = kv.first;
            recv_peer_ncells[p] = n;
            recv_idx[p] = bufs.recv_idx_pool.subview( recv_cell_off[p], n );
            recv_bufs[p] = bufs.recv_pool.subview(
                recv_cell_off[p] * per_cell_complex,
                static_cast<size_t>( n ) * per_cell_complex );
            upload_idx( kv.second, recv_idx[p] );
            MPI_Irecv( reinterpret_cast<scalar_type*>( recv_bufs[p].data() ),
                       n * per_cell_real, mpi_scalar, kv.first, /*tag=*/0,
                       comm, &recv_reqs[p] );
            ++p;
        }
    }

    // Build send buffers and post sends.
    std::vector<MPI_Request> send_reqs( n_send_peers );
    {
        std::vector<size_t> send_cell_off( n_send_peers );
        size_t total_send_cells = 0;
        {
            int p = 0;
            for ( const auto& kv : send_cells_by_peer )
            {
                send_cell_off[p] = total_send_cells;
                total_send_cells += kv.second.size();
                ++p;
            }
        }
        bufs.send_idx_pool.reserve( total_send_cells );
        bufs.send_pool.reserve( total_send_cells * per_cell_complex );

        int p = 0;
        for ( const auto& kv : send_cells_by_peer )
        {
            const int n = static_cast<int>( kv.second.size() );
            send_peer_ranks[p] = kv.first;
            send_peer_ncells[p] = n;
            send_idx[p] = bufs.send_idx_pool.subview( send_cell_off[p], n );
            send_bufs[p] = bufs.send_pool.subview(
                send_cell_off[p] * per_cell_complex,
                static_cast<size_t>( n ) * per_cell_complex );
            upload_idx( kv.second, send_idx[p] );

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
