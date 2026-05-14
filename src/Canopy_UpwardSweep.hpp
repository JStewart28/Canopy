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
#include "Canopy_MpiCoalescedExchange.hpp"
#include "Canopy_Profiling.hpp"
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
// leaves up to the root. Uses the communication plans from
// CommunicationPlan to exchange multipole coefficients between ranks
// at layer boundaries where parent and child have different owners.
//
// Supports multi-component solves (NComps > 1) for Biot-Savart and
// similar kernels that need several simultaneous Laplace solves.
//
// Parallelism: Kokkos hierarchical (TeamPolicy).
//   - Leagues iterate over cells at the current layer owned/shared by
//     this rank.
//   - Within each team, threads cooperate over (n, m) output coefficients.
//   - P2M parallelizes over particles.
//
// Template Parameters:
//   MemorySpace, ExecutionSpace   - Kokkos spaces
//   KernelType   - e.g. LaplaceKernel<double, 10, 1>
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
    static constexpr int coeffs_per_cell = KernelType::num_coeffs_per_cell;
    static constexpr int NComps = KernelType::num_components;

    // Coefficient storage: (cell_idx, coeff_idx, comp_idx).
    // LayoutRight so within-cell coefficient traversals are contiguous —
    // matches what M2L_fused needs for coalesced reads/writes.
    using coeff_view_type =
        Kokkos::View<complex_type***, Kokkos::LayoutRight, memory_space>;

    using a_view_type = Kokkos::View<scalar_type*, memory_space>;

    struct DeviceCellInfo
    {
        MortonKey key;
        int depth;
        int cell_idx;
        int owner_rank;
        scalar_type center[3];
        scalar_type half_width;
        bool is_leaf;
    };
    using cell_view_type = Kokkos::View<DeviceCellInfo*, memory_space>;

    using particle_cell_idx_view_type = Kokkos::View<int*, memory_space>;

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
    // -----------------------------------------------------------------------
    void setup( const std::vector<CellInfo>& cells,
                const std::unordered_map<MortonKey, int>& owner_map,
                const Kokkos::View<MortonKey*, memory_space>& particle_keys,
                int num_local_particles );

    // -----------------------------------------------------------------------
    // execute()
    //
    // Parameters:
    //   particle_charges - Kokkos::View<Scalar*[NComps]> charges per
    //                      particle per component. If NComps==1 use
    //                      a 2D view with trailing extent 1.
    //   particle_positions - Cabana slice
    //   comm_plan        - precomputed communication plan
    // -----------------------------------------------------------------------
    template <class ChargeView, class PositionType>
    void
    execute( const ChargeView& particle_charges,
             const PositionType& particle_positions,
             const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan );

    // Access the computed multipole coefficients after execute()
    const coeff_view_type& multipoles() const { return _multipoles; }

    // Look up the cell index for a given Morton key (host-side)
    int cell_index( MortonKey key ) const
    {
        auto it = _key_to_cell_idx.find( key );
        return ( it != _key_to_cell_idx.end() ) ? it->second : -1;
    }

    // Access the A_{n,m} table
    const a_view_type& A_table() const { return _A_table; }

    // Access device cell info (shared with DownwardSweep)
    const cell_view_type& device_cells() const { return _device_cells; }

    using children_view_type =
        Kokkos::View<int* [8], Kokkos::LayoutRight, memory_space>;

    // Per-cell child index table (shared with DownwardSweep). Row i lists the
    // device-cell indices of cell i's children, padded with -1.
    const children_view_type& cell_children() const { return _d_cell_children; }

    // Access host-side key lookup (shared with DownwardSweep)
    const std::unordered_map<MortonKey, int>& key_to_cell_idx() const
    {
        return _key_to_cell_idx;
    }

    // Access the device-side particle-to-cell-index map
    // (shared with DownwardSweep for L2P)
    const particle_cell_idx_view_type& particle_cell_idx() const
    {
        return _particle_cell_idx;
    }

    // Max tree depth
    int max_depth() const { return _max_depth; }

  private:
    MPI_Comm _comm;
    int _rank;
    int _nprocs;

    coeff_view_type _multipoles;
    a_view_type _A_table;
    cell_view_type _device_cells;
    std::unordered_map<MortonKey, int> _key_to_cell_idx;

    std::vector<std::vector<int>> _leaves_at_depth_local;
    std::vector<std::vector<int>> _internals_at_depth_local;

    std::vector<Kokkos::View<int*, memory_space>> _d_leaves_at_depth;
    std::vector<Kokkos::View<int*, memory_space>> _d_internals_at_depth;

    children_view_type _d_cell_children;

    particle_cell_idx_view_type _particle_cell_idx;
    int _num_local_particles;
    int _max_depth;

  public:
    void build_particle_cell_idx(
        const Kokkos::View<MortonKey*, memory_space>& particle_keys,
        int num_local_particles );

    template <class ChargeView, class PositionType>
    void run_p2m_at_depth( int depth, const ChargeView& particle_charges,
                           const PositionType& particle_positions );

    void run_m2m_at_depth( int depth );

    void exchange_multipoles_at_depth(
        int depth,
        const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan );
};

// ============================================================================
// Implementation
// ============================================================================

template <class MemorySpace, class ExecutionSpace, class KernelType>
void UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::setup(
    const std::vector<CellInfo>& cells,
    const std::unordered_map<MortonKey, int>& owner_map,
    const Kokkos::View<MortonKey*, memory_space>& particle_keys,
    int num_local_particles )
{
    const int num_cells = static_cast<int>( cells.size() );
    _num_local_particles = num_local_particles;

    // (cells, coeffs, components)
    _multipoles =
        coeff_view_type( "multipoles", num_cells, coeffs_per_cell, NComps );

    // M2L accesses A at degree n+j where both n and j go up to P, so the
    // table must cover up to 2*P.
    _A_table = build_A_coefficients<scalar_type, memory_space>( 2 * P );

    _key_to_cell_idx.clear();
    _key_to_cell_idx.reserve( num_cells );
    for ( int i = 0; i < num_cells; i++ )
        _key_to_cell_idx[cells[i].key] = i;

    _max_depth = 0;
    for ( const auto& c : cells )
        if ( c.depth > _max_depth )
            _max_depth = c.depth;

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
        dci.owner_rank = ( it != owner_map.end() ) ? it->second : OWNER_SHARED;
        for ( int d = 0; d < 3; d++ )
            dci.center[d] = static_cast<scalar_type>( c.center[d] );
        dci.half_width = static_cast<scalar_type>( c.half_width );
        dci.is_leaf = c.is_leaf;
        h_device_cells( i ) = dci;
    }
    Kokkos::deep_copy( _device_cells, h_device_cells );

    _leaves_at_depth_local.assign( _max_depth + 1, {} );
    _internals_at_depth_local.assign( _max_depth + 1, {} );

    for ( int i = 0; i < num_cells; i++ )
    {
        const auto& c = cells[i];
        int owner = h_device_cells( i ).owner_rank;
        bool processes = ( owner == _rank || owner == OWNER_SHARED );
        if ( !processes )
            continue;

        if ( c.is_leaf )
            _leaves_at_depth_local[c.depth].push_back( i );
        else
            _internals_at_depth_local[c.depth].push_back( i );
    }

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
        _d_internals_at_depth[d] = Kokkos::View<int*, memory_space>(
            "internals_at_depth", ninternals );
        if ( ninternals > 0 )
        {
            auto h = Kokkos::create_mirror_view( _d_internals_at_depth[d] );
            for ( int j = 0; j < ninternals; j++ )
                h( j ) = _internals_at_depth_local[d][j];
            Kokkos::deep_copy( _d_internals_at_depth[d], h );
        }
    }

    _d_cell_children =
        children_view_type( Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                                "cell_children" ),
                            num_cells );
    {
        auto h_children = Kokkos::create_mirror_view( _d_cell_children );
        for ( int i = 0; i < num_cells; i++ )
        {
            for ( int k = 0; k < 8; k++ )
                h_children( i, k ) = -1;
            if ( cells[i].is_leaf )
                continue;
            const MortonKey pk = cells[i].key;
            int slot = 0;
            for ( int oct = 0; oct < 8; oct++ )
            {
                const MortonKey ck =
                    ( pk << 3 ) | static_cast<MortonKey>( oct );
                auto it = _key_to_cell_idx.find( ck );
                if ( it != _key_to_cell_idx.end() )
                    h_children( i, slot++ ) = it->second;
            }
        }
        Kokkos::deep_copy( _d_cell_children, h_children );
    }

    build_particle_cell_idx( particle_keys, num_local_particles );
}

template <class MemorySpace, class ExecutionSpace, class KernelType>
void UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::
    build_particle_cell_idx(
        const Kokkos::View<MortonKey*, memory_space>& particle_keys,
        int num_local_particles )
{
    _particle_cell_idx =
        particle_cell_idx_view_type( "particle_cell_idx", num_local_particles );

    auto h_keys = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{},
                                                       particle_keys );
    auto h_idx = Kokkos::create_mirror_view( _particle_cell_idx );

    for ( int i = 0; i < num_local_particles; i++ )
    {
        auto it = _key_to_cell_idx.find( h_keys( i ) );
        h_idx( i ) = ( it != _key_to_cell_idx.end() ) ? it->second : -1;
    }

    Kokkos::deep_copy( _particle_cell_idx, h_idx );
}

template <class MemorySpace, class ExecutionSpace, class KernelType>
template <class ChargeView, class PositionType>
void UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::run_p2m_at_depth(
    int depth, const ChargeView& particle_charges,
    const PositionType& particle_positions )
{
    CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_P2M );
    auto multipoles = _multipoles;
    auto device_cells = _device_cells;
    auto particle_cell_idx = _particle_cell_idx;
    const int N = _num_local_particles;

    Kokkos::parallel_for(
        "P2M", Kokkos::RangePolicy<execution_space>( 0, N ),
        KOKKOS_LAMBDA( int p ) {
            const int cidx = particle_cell_idx( p );
            if ( cidx < 0 )
                return;

            const auto& dci = device_cells( cidx );
            if ( dci.depth != depth || !dci.is_leaf )
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

            // Load this particle's per-component charges
            scalar_type charges[NComps];
            for ( int c = 0; c < NComps; c++ )
                charges[c] =
                    static_cast<scalar_type>( particle_charges( p, c ) );

            // Slice M_out(coeff_idx, comp_idx) for this cell
            auto M_out =
                Kokkos::subview( multipoles, cidx, Kokkos::ALL, Kokkos::ALL );
            KernelType::p2m_contribution( charges, dx, dy, dz, M_out );
        } );

    Kokkos::fence();
}

template <class MemorySpace, class ExecutionSpace, class KernelType>
void UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::run_m2m_at_depth(
    int depth )
{
    CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_M2M );
    const int ninternals =
        static_cast<int>( _internals_at_depth_local[depth].size() );
    if ( ninternals == 0 )
        return;

    auto multipoles = _multipoles;
    auto device_cells = _device_cells;
    auto A_table = _A_table;
    auto& d_internals = _d_internals_at_depth[depth];
    auto children = _d_cell_children;
    const int this_rank = _rank;

    using team_policy = Kokkos::TeamPolicy<execution_space>;
    using team_member_type = typename team_policy::member_type;

    team_policy policy( ninternals, Kokkos::AUTO );

    Kokkos::parallel_for(
        "M2M", policy, KOKKOS_LAMBDA( const team_member_type& team ) {
            const int league = team.league_rank();
            const int parent_cell = d_internals( league );
            const auto& parent_ci = device_cells( parent_cell );

            auto M_parent = Kokkos::subview( multipoles, parent_cell,
                                             Kokkos::ALL, Kokkos::ALL );

            for ( int k = 0; k < 8; k++ )
            {
                const int ci = children( parent_cell, k );
                if ( ci < 0 )
                    break;
                const auto& ccell = device_cells( ci );

                // When both parent and child are shared (replicated), every
                // rank holds the same post-Allreduce child multipole. Only
                // rank 0 accumulates the contribution so the subsequent
                // Allreduce on the parent doesn't count it N times.
                if ( parent_ci.owner_rank == OWNER_SHARED &&
                     ccell.owner_rank == OWNER_SHARED && this_rank != 0 )
                    continue;

                const scalar_type dx = ccell.center[0] - parent_ci.center[0];
                const scalar_type dy = ccell.center[1] - parent_ci.center[1];
                const scalar_type dz = ccell.center[2] - parent_ci.center[2];

                KernelType::m2m_translate( team, multipoles, ci, dx, dy, dz,
                                           A_table, M_parent );
            }
        } );

    Kokkos::fence();
}

template <class MemorySpace, class ExecutionSpace, class KernelType>
void UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::
    exchange_multipoles_at_depth(
        int depth,
        const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan )
{
    const auto& m2m = comm_plan.m2m_plan();

    // Filter shared cells to current depth
    std::vector<int> shared_cell_indices;
    auto h_dc_all = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{},
                                                         _device_cells );
    for ( MortonKey k : m2m.shared_cells )
    {
        auto it = _key_to_cell_idx.find( k );
        if ( it == _key_to_cell_idx.end() )
            continue;
        if ( h_dc_all( it->second ).depth == depth )
            shared_cell_indices.push_back( it->second );
    }

    // Bytes per cell for all components
    const int per_cell_complex = coeffs_per_cell * NComps;
    MPI_Datatype mpi_scalar =
        ( sizeof( scalar_type ) == 8 ) ? MPI_DOUBLE : MPI_FLOAT;

    // Allreduce shared cells — single device-side buffer, GPU-direct.
    if ( !shared_cell_indices.empty() )
    {
        const int nshared = static_cast<int>( shared_cell_indices.size() );
        const size_t total_complex =
            static_cast<size_t>( nshared ) * per_cell_complex;

        Kokkos::View<int*, memory_space> d_idx(
            Kokkos::view_alloc( "m2m_allreduce_idx",
                                Kokkos::WithoutInitializing ),
            nshared );
        {
            auto h_idx = Kokkos::create_mirror_view( d_idx );
            for ( int i = 0; i < nshared; i++ )
                h_idx( i ) = shared_cell_indices[i];
            Kokkos::deep_copy( d_idx, h_idx );
        }

        Kokkos::View<complex_type*, memory_space> sendbuf(
            Kokkos::view_alloc( "m2m_allreduce_send",
                                Kokkos::WithoutInitializing ),
            total_complex );
        Kokkos::View<complex_type*, memory_space> recvbuf(
            Kokkos::view_alloc( "m2m_allreduce_recv",
                                Kokkos::WithoutInitializing ),
            total_complex );

        auto mults = _multipoles;
        const int cpc = coeffs_per_cell;
        const int nc = NComps;
        Kokkos::parallel_for(
            "m2m_allreduce_pack",
            Kokkos::RangePolicy<execution_space>( 0, nshared ),
            KOKKOS_LAMBDA( const int i ) {
                const int cidx = d_idx( i );
                const int base = i * cpc * nc;
                for ( int ci = 0; ci < cpc; ci++ )
                    for ( int c = 0; c < nc; c++ )
                        sendbuf( base + ci * nc + c ) = mults( cidx, ci, c );
            } );
        Kokkos::fence();

        {
            CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_M2M_ALLREDUCE );
            MPI_Allreduce( reinterpret_cast<scalar_type*>( sendbuf.data() ),
                           reinterpret_cast<scalar_type*>( recvbuf.data() ),
                           static_cast<int>( 2 * total_complex ), mpi_scalar,
                           MPI_SUM, _comm );
        }

        Kokkos::parallel_for(
            "m2m_allreduce_unpack",
            Kokkos::RangePolicy<execution_space>( 0, nshared ),
            KOKKOS_LAMBDA( const int i ) {
                const int cidx = d_idx( i );
                const int base = i * cpc * nc;
                for ( int ci = 0; ci < cpc; ci++ )
                    for ( int c = 0; c < nc; c++ )
                        mults( cidx, ci, c ) = recvbuf( base + ci * nc + c );
            } );
        Kokkos::fence();
    }

    // Point-to-point sends/receives for this depth — coalesced per peer
    // rank to keep MPI request counts at O(#peers) rather than O(#cells).
    std::map<int, std::vector<std::pair<MortonKey, int>>> send_by_peer_kv;
    for ( const auto& ct : m2m.sends )
    {
        auto it = _key_to_cell_idx.find( ct.cell_key );
        if ( it == _key_to_cell_idx.end() )
            continue;
        if ( h_dc_all( it->second ).depth != depth )
            continue;
        send_by_peer_kv[ct.remote_rank].emplace_back( ct.cell_key,
                                                      it->second );
    }
    std::map<int, std::vector<std::pair<MortonKey, int>>> recv_by_peer_kv;
    for ( const auto& ct : m2m.receives )
    {
        auto it = _key_to_cell_idx.find( ct.cell_key );
        if ( it == _key_to_cell_idx.end() )
            continue;
        if ( h_dc_all( it->second ).depth != depth )
            continue;
        recv_by_peer_kv[ct.remote_rank].emplace_back( ct.cell_key,
                                                      it->second );
    }

    if ( send_by_peer_kv.empty() && recv_by_peer_kv.empty() )
        return;

    auto sort_and_flatten =
        []( std::map<int, std::vector<std::pair<MortonKey, int>>>& in )
    {
        std::map<int, std::vector<int>> out;
        for ( auto& kv : in )
        {
            std::sort( kv.second.begin(), kv.second.end(),
                       []( const std::pair<MortonKey, int>& a,
                           const std::pair<MortonKey, int>& b )
                       { return a.first < b.first; } );
            auto& dst = out[kv.first];
            dst.reserve( kv.second.size() );
            for ( const auto& p : kv.second )
                dst.push_back( p.second );
        }
        return out;
    };

    auto sends_by_peer = sort_and_flatten( send_by_peer_kv );
    auto recvs_by_peer = sort_and_flatten( recv_by_peer_kv );

    detail::coalesced_view_exchange( _multipoles, _comm, sends_by_peer,
                                     recvs_by_peer,
                                     /*accumulate_on_recv=*/false );
}

template <class MemorySpace, class ExecutionSpace, class KernelType>
template <class ChargeView, class PositionType>
void UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::execute(
    const ChargeView& particle_charges, const PositionType& particle_positions,
    const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan )
{
    CANOPY_RESET_TIMERS();
    {
        CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_UPWARD_TOTAL );
        Kokkos::deep_copy( _multipoles, complex_type( 0.0, 0.0 ) );

        for ( int d = 0; d <= _max_depth; d++ )
            if ( !_leaves_at_depth_local[d].empty() )
                run_p2m_at_depth( d, particle_charges, particle_positions );

        exchange_multipoles_at_depth( _max_depth, comm_plan );

        for ( int d = _max_depth - 1; d >= 0; d-- )
        {
            run_m2m_at_depth( d );
            exchange_multipoles_at_depth( d, comm_plan );
        }
    }
    CANOPY_PRINT_UPWARD_TIMERS( _comm );
}

} // namespace Canopy

#endif // CANOPY_UPWARD_SWEEP_HPP
