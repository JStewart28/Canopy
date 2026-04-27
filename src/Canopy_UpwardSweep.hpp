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
    static constexpr int coeffs_per_cell =
        KernelType::num_coeffs_per_cell;
    static constexpr int NComps = KernelType::num_components;

    // Coefficient storage: (cell_idx, coeff_idx, comp_idx)
    using coeff_view_type =
        Kokkos::View<complex_type***, memory_space>;

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
    using cell_view_type =
        Kokkos::View<DeviceCellInfo*, memory_space>;

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
    // -----------------------------------------------------------------------
    void setup(
        const std::vector<CellInfo>& cells,
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
    void execute(
        const ChargeView& particle_charges,
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

    particle_cell_idx_view_type _particle_cell_idx;
    int _num_local_particles;
    int _max_depth;

  public:
    void build_particle_cell_idx(
        const Kokkos::View<MortonKey*, memory_space>& particle_keys,
        int num_local_particles );

    template <class ChargeView, class PositionType>
    void run_p2m_at_depth( int depth,
                           const ChargeView& particle_charges,
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
    _multipoles = coeff_view_type( "multipoles", num_cells,
                                   coeffs_per_cell, NComps );

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
        dci.owner_rank =
            ( it != owner_map.end() ) ? it->second : OWNER_SHARED;
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
        bool processes =
            ( owner == _rank || owner == OWNER_SHARED );
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
        _d_leaves_at_depth[d] = Kokkos::View<int*, memory_space>(
            "leaves_at_depth", nleaves );
        if ( nleaves > 0 )
        {
            auto h =
                Kokkos::create_mirror_view( _d_leaves_at_depth[d] );
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
            auto h =
                Kokkos::create_mirror_view( _d_internals_at_depth[d] );
            for ( int j = 0; j < ninternals; j++ )
                h( j ) = _internals_at_depth_local[d][j];
            Kokkos::deep_copy( _d_internals_at_depth[d], h );
        }
    }

    build_particle_cell_idx( particle_keys, num_local_particles );
}

template <class MemorySpace, class ExecutionSpace, class KernelType>
void UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::build_particle_cell_idx(
    const Kokkos::View<MortonKey*, memory_space>& particle_keys,
    int num_local_particles )
{
    _particle_cell_idx = particle_cell_idx_view_type(
        "particle_cell_idx", num_local_particles );

    auto h_keys = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, particle_keys );
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
            auto M_out = Kokkos::subview( multipoles, cidx, Kokkos::ALL,
                                          Kokkos::ALL );
            KernelType::p2m_contribution( charges, dx, dy, dz, M_out );
        } );

    Kokkos::fence();
}

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
    const int this_rank = _rank;

    using team_policy = Kokkos::TeamPolicy<execution_space>;
    using team_member_type = typename team_policy::member_type;

    team_policy policy( ninternals, Kokkos::AUTO );

    Kokkos::parallel_for(
        "M2M",
        policy,
        KOKKOS_LAMBDA( const team_member_type& team ) {
            const int league = team.league_rank();
            const int parent_cell = d_internals( league );
            const auto& parent_ci = device_cells( parent_cell );

            auto M_parent = Kokkos::subview( multipoles, parent_cell,
                                             Kokkos::ALL, Kokkos::ALL );

            const MortonKey pk = parent_ci.key;
            const int num_all = device_cells.extent( 0 );

            for ( int ci = 0; ci < num_all; ci++ )
            {
                const auto& ccell = device_cells( ci );
                if ( ( ccell.key >> 3 ) != pk )
                    continue;
                if ( ccell.depth != parent_ci.depth + 1 )
                    continue;

                // When both parent and child are shared (replicated), every
                // rank holds the same post-Allreduce child multipole. Only
                // rank 0 accumulates the contribution so the subsequent
                // Allreduce on the parent doesn't count it N times.
                if ( parent_ci.owner_rank == OWNER_SHARED &&
                     ccell.owner_rank == OWNER_SHARED &&
                     this_rank != 0 )
                    continue;

                const scalar_type dx =
                    ccell.center[0] - parent_ci.center[0];
                const scalar_type dy =
                    ccell.center[1] - parent_ci.center[1];
                const scalar_type dz =
                    ccell.center[2] - parent_ci.center[2];

                KernelType::m2m_translate( team, multipoles, ci, dx, dy,
                                           dz, A_table, M_parent );
            }
        } );

    Kokkos::fence();
}

template <class MemorySpace, class ExecutionSpace, class KernelType>
void UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::exchange_multipoles_at_depth(
    int depth, const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan )
{
    const auto& m2m = comm_plan.m2m_plan();

    // Filter shared cells to current depth
    std::vector<int> shared_cell_indices;
    auto h_dc_all = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, _device_cells );
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
    const int per_cell_real = 2 * per_cell_complex;
    MPI_Datatype mpi_scalar =
        ( sizeof( scalar_type ) == 8 ) ? MPI_DOUBLE : MPI_FLOAT;

    // Allreduce shared cells
    if ( !shared_cell_indices.empty() )
    {
        const int nshared = static_cast<int>( shared_cell_indices.size() );
        const int total_complex = nshared * per_cell_complex;

        std::vector<complex_type> sendbuf( total_complex );
        std::vector<complex_type> recvbuf( total_complex );

        auto h_mults = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, _multipoles );

        for ( int i = 0; i < nshared; i++ )
        {
            const int cidx = shared_cell_indices[i];
            int idx = 0;
            for ( int ci = 0; ci < coeffs_per_cell; ci++ )
                for ( int c = 0; c < NComps; c++ )
                    sendbuf[i * per_cell_complex + ( idx++ )] =
                        h_mults( cidx, ci, c );
        }

        MPI_Allreduce( reinterpret_cast<scalar_type*>( sendbuf.data() ),
                       reinterpret_cast<scalar_type*>( recvbuf.data() ),
                       2 * total_complex, mpi_scalar, MPI_SUM, _comm );

        for ( int i = 0; i < nshared; i++ )
        {
            const int cidx = shared_cell_indices[i];
            int idx = 0;
            for ( int ci = 0; ci < coeffs_per_cell; ci++ )
                for ( int c = 0; c < NComps; c++ )
                    h_mults( cidx, ci, c ) =
                        recvbuf[i * per_cell_complex + ( idx++ )];
        }

        Kokkos::deep_copy( _multipoles, h_mults );
    }

    // Point-to-point sends/receives for this depth
    struct Transfer
    {
        int cell_idx;
        int remote_rank;
    };
    std::vector<Transfer> my_sends, my_recvs;

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

    auto h_mults = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, _multipoles );

    std::vector<MPI_Request> recv_reqs( my_recvs.size() );
    std::vector<std::vector<complex_type>> recv_bufs( my_recvs.size() );
    for ( size_t i = 0; i < my_recvs.size(); i++ )
    {
        recv_bufs[i].resize( per_cell_complex );
        const MortonKey key = h_dc_all( my_recvs[i].cell_idx ).key;
        int tag = static_cast<int>( key & 0x7fffffff );
        MPI_Irecv(
            reinterpret_cast<scalar_type*>( recv_bufs[i].data() ),
            per_cell_real, mpi_scalar, my_recvs[i].remote_rank, tag,
            _comm, &recv_reqs[i] );
    }

    std::vector<MPI_Request> send_reqs( my_sends.size() );
    std::vector<std::vector<complex_type>> send_bufs( my_sends.size() );
    for ( size_t i = 0; i < my_sends.size(); i++ )
    {
        send_bufs[i].resize( per_cell_complex );
        const int cidx = my_sends[i].cell_idx;
        int idx = 0;
        for ( int ci = 0; ci < coeffs_per_cell; ci++ )
            for ( int c = 0; c < NComps; c++ )
                send_bufs[i][idx++] = h_mults( cidx, ci, c );

        const MortonKey key = h_dc_all( cidx ).key;
        int tag = static_cast<int>( key & 0x7fffffff );
        MPI_Isend(
            reinterpret_cast<scalar_type*>( send_bufs[i].data() ),
            per_cell_real, mpi_scalar, my_sends[i].remote_rank, tag,
            _comm, &send_reqs[i] );
    }

    if ( !recv_reqs.empty() )
        MPI_Waitall( recv_reqs.size(), recv_reqs.data(),
                     MPI_STATUSES_IGNORE );
    if ( !send_reqs.empty() )
        MPI_Waitall( send_reqs.size(), send_reqs.data(),
                     MPI_STATUSES_IGNORE );

    for ( size_t i = 0; i < my_recvs.size(); i++ )
    {
        const int cidx = my_recvs[i].cell_idx;
        int idx = 0;
        for ( int ci = 0; ci < coeffs_per_cell; ci++ )
            for ( int c = 0; c < NComps; c++ )
                h_mults( cidx, ci, c ) = recv_bufs[i][idx++];
    }

    Kokkos::deep_copy( _multipoles, h_mults );
}

template <class MemorySpace, class ExecutionSpace, class KernelType>
template <class ChargeView, class PositionType>
void UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::execute(
    const ChargeView& particle_charges,
    const PositionType& particle_positions,
    const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan )
{
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

} // namespace Canopy

#endif // CANOPY_UPWARD_SWEEP_HPP
