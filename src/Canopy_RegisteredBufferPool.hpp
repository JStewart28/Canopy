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

#ifndef CANOPY_REGISTERED_BUFFER_POOL_HPP
#define CANOPY_REGISTERED_BUFFER_POOL_HPP

#include <Kokkos_Core.hpp>

#include <cstddef>

namespace Canopy
{
namespace detail
{

// ============================================================================
// RegisteredBufferPool
//
// A persistent, grow-only device buffer reused across solve() calls for the
// send/recv staging of GPU-aware MPI exchanges.
//
// Why this exists: the M2L/L2L (coalesced_view_exchange) and P2P
// (gather_ghost_particles) exchanges used to allocate a brand-new device
// Kokkos::View for every peer on every call and hand its .data() pointer to
// MPI_Isend/MPI_Irecv. On Slingshot/CXI with GPU-aware Cray-MPICH, each fresh
// device allocation triggers a fresh NIC memory registration. During a fully
// rolled-up FMM step the M2L list grows and the partitioner rebalances nearly
// every step, so the registration cache churns/accumulates until the NIC runs
// out of registration resources and aborts with `cxil_map: write error` ->
// `MPIDI_OFI_send_normal: Invalid argument`.
//
// A pool keeps ONE allocation with a stable base address that only grows (1.5x
// headroom, never shrinks). All peers for a given direction are packed into
// non-overlapping [offset, offset+n) sub-ranges of that single region, so the
// CXI registration cache sees one region registered once and reused, instead
// of a fresh registration per peer per step. The registration footprint
// becomes bounded (it stabilizes after the buffer reaches its working-set
// size) rather than growing with step count.
//
// Usage:
//   pool.reserve( total_elems );                 // once, before any subview()
//   auto v = pool.subview( peer_off, peer_n );   // unmanaged view per peer
//   MPI_Isend( v.data(), ... );                  // points into the one region
//
// Caller contract: reserve() the full total for the call BEFORE taking any
// subview(), so the base address is stable for the lifetime of every subview
// handed out that call. Distinct peers must use non-overlapping ranges.
// ============================================================================
template <class T, class MemorySpace>
class RegisteredBufferPool
{
  public:
    using memory_space = MemorySpace;
    using view_type = Kokkos::View<T*, memory_space>;
    using unmanaged_view_type =
        Kokkos::View<T*, memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    // Ensure capacity for at least n elements of T. Grow-only with 1.5x
    // headroom; never shrinks. A grow reallocates (and re-registers) once,
    // but capacity converges to the working-set size so grows are log-many
    // over a run, not per-step.
    void reserve( std::size_t n )
    {
        if ( n > _capacity )
        {
            const std::size_t new_cap = n + n / 2; // 1.5x headroom
            _buf = view_type(
                Kokkos::view_alloc( "Canopy_RegisteredBufferPool",
                                    Kokkos::WithoutInitializing ),
                new_cap );
            _capacity = new_cap;
        }
    }

    // Unmanaged view of the [offset, offset+n) element range of the pool.
    // Valid only while no intervening reserve() has grown the pool. The
    // caller guarantees offset + n <= the reserved capacity.
    unmanaged_view_type subview( std::size_t offset, std::size_t n ) const
    {
        return unmanaged_view_type( _buf.data() + offset, n );
    }

    T* data() const { return _buf.data(); }
    std::size_t capacity() const { return _capacity; }

  private:
    view_type _buf;
    std::size_t _capacity = 0;
};

} // namespace detail
} // namespace Canopy

#endif // CANOPY_REGISTERED_BUFFER_POOL_HPP
