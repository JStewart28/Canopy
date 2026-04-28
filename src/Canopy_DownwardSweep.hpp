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

#ifndef CANOPY_DOWNWARD_SWEEP_HPP
#define CANOPY_DOWNWARD_SWEEP_HPP

#include "Canopy_CommunicationPlan.hpp"
#include "Canopy_LaplaceKernel.hpp"
#include "Canopy_SphericalCoefficients.hpp"
#include "Canopy_TreeBuilder.hpp"
#include "Canopy_TreePartitioner.hpp"
#include "Canopy_UpwardSweep.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <unordered_map>
#include <vector>

namespace Canopy
{

// ============================================================================
// DownwardSweep
//
// Performs the FMM downward sweep:
//   1. Pre-sweep exchange: gather remote multipoles needed for M2L.
//   2. For depth d = 0 to max_depth:
//        a. M2L: sum over interaction list into this cell's local.
//        b. Allreduce partial locals at shared cells at this depth.
//        c. L2L: translate parent locals to their children.
//        d. Point-to-point exchange of local coefficients for children
//           whose parents are on remote ranks.
//   3. L2P: evaluate local expansion at each particle, producing
//           potential and (optionally) gradient.
//
// Storage is (num_cells, coeffs_per_cell, NComps) matching UpwardSweep.
//
// Output:
//   potential(particle_idx, comp_idx)
//   gradient(particle_idx, comp_idx, dim_idx)  — only if compute_gradient
//
// Both views are caller-owned. Pass an empty/zero-extent gradient view
// and set compute_gradient=false to skip gradient evaluation.
//
// Template Parameters:
//   MemorySpace, ExecutionSpace - Kokkos device type
//   KernelType - e.g. LaplaceKernel<double, P, NComps>
// ============================================================================

template <class MemorySpace, class ExecutionSpace, class KernelType>
class DownwardSweep
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

    // Local coefficient storage
    using coeff_view_type =
        Kokkos::View<complex_type***, memory_space>;

    // Potential output: (num_particles, NComps)
    using potential_view_type =
        Kokkos::View<scalar_type* [NComps], memory_space>;

    // Gradient output: (num_particles, NComps, 3)
    using gradient_view_type =
        Kokkos::View<scalar_type* [NComps][3], memory_space>;

    using a_view_type = Kokkos::View<scalar_type*, memory_space>;

    // Gradient accessor passed to l2p_evaluate — avoids nested device lambdas
    // (CUDA does not allow extended __host__ __device__ lambdas nested inside
    // another extended lambda).
    struct GradWriter
    {
        gradient_view_type grad;
        int p;
        KOKKOS_INLINE_FUNCTION scalar_type& operator()( int c, int d ) const
        {
            return grad( p, c, d );
        }
    };

    // Match UpwardSweep's cell metadata layout
    using cell_view_type =
        typename UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::cell_view_type;
    using particle_cell_idx_view_type =
        typename UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::particle_cell_idx_view_type;

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------
    DownwardSweep( MPI_Comm comm )
        : _comm( comm )
    {
        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_nprocs );
    }

    // -----------------------------------------------------------------------
    // setup()
    //
    // Allocates local-coefficient storage and references shared cell
    // metadata from the UpwardSweep. Call after UpwardSweep::setup()
    // and before execute().
    //
    // We take the UpwardSweep by reference so we can reuse its cell
    // view, key lookup, particle-cell index, and A-table. This avoids
    // duplicating host metadata.
    // -----------------------------------------------------------------------
    void setup(
        const UpwardSweep<MemorySpace, ExecutionSpace, KernelType>& upward_sweep,
        int num_local_particles );

    // -----------------------------------------------------------------------
    // allocate_potential / allocate_gradient — helper to size outputs.
    // Callers can use these or allocate their own.
    // -----------------------------------------------------------------------
    potential_view_type allocate_potential( int num_local_particles ) const
    {
        return potential_view_type( "fmm_potential",
                                    num_local_particles );
    }

    gradient_view_type allocate_gradient( int num_local_particles ) const
    {
        return gradient_view_type( "fmm_gradient",
                                   num_local_particles );
    }

    // -----------------------------------------------------------------------
    // execute()
    //
    // Runs M2L, L2L, L2P with MPI exchange.
    //
    // Parameters:
    //   multipoles         - from UpwardSweep after its execute()
    //   particle_positions - Cabana slice of positions
    //   potential_out      - (num_particles, NComps); caller-zeroed
    //   gradient_out       - (num_particles, NComps, 3); ignored if
    //                        compute_gradient is false. May be a
    //                        zero-extent view in that case.
    //   compute_gradient   - if true, populate gradient_out
    //   comm_plan          - precomputed plan
    //
    // Local coefficients are stored internally and can be inspected
    // via locals() after execute().
    // -----------------------------------------------------------------------
    template <class PositionType>
    void execute(
        const typename UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::coeff_view_type&
            multipoles,
        const PositionType& particle_positions,
        const potential_view_type& potential_out,
        const gradient_view_type& gradient_out, bool compute_gradient,
        const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan );

    // Access the computed local coefficients after execute()
    const coeff_view_type& locals() const { return _locals; }

  private:
    MPI_Comm _comm;
    int _rank;
    int _nprocs;

    coeff_view_type _locals;

    // References borrowed from UpwardSweep — only valid while the
    // upward sweep is alive and setup() has been called.
    cell_view_type _device_cells;
    const std::unordered_map<MortonKey, int>* _key_to_cell_idx;
    particle_cell_idx_view_type _particle_cell_idx;
    a_view_type _A_table;

    // Per-depth cell-index lists (this rank processes these cells).
    // Reconstructed during setup().
    std::vector<std::vector<int>> _leaves_at_depth_local;
    std::vector<std::vector<int>>
        _internals_at_depth_local; // non-leaf cells this rank processes
    std::vector<std::vector<int>> _all_at_depth_local; // leaves + internals

    std::vector<Kokkos::View<int*, memory_space>> _d_leaves_at_depth;
    std::vector<Kokkos::View<int*, memory_space>> _d_internals_at_depth;
    std::vector<Kokkos::View<int*, memory_space>> _d_all_at_depth;

    int _num_local_particles;
    int _max_depth;

    // Interaction list stored on device for M2L launches:
    // For each target cell this rank processes, the flat list of source
    // cell indices. To make this efficient on device, we concatenate
    // all interaction lists into a single flat array with per-target
    // offsets and counts.
    Kokkos::View<int*, memory_space> _m2l_target_cells;
    Kokkos::View<int*, memory_space> _m2l_source_cells_flat;
    Kokkos::View<int*, memory_space> _m2l_offsets;  // size N+1
    Kokkos::View<int*, memory_space> _m2l_counts;   // size N

    // Scratch: reference to the multipoles view during execute().
    // The M2L kernels capture this. Set at the start of execute()
    // and used by run_m2l_at_depth().
    typename UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::coeff_view_type
        _m2l_multipoles_view;

    // Per-depth snapshot of shared-cell locals taken before M2L at that
    // depth. Used by allreduce_shared_locals_at_depth to isolate the
    // M2L delta from L2L contributions inherited from prior depths.
    std::vector<int> _shared_snapshot_indices; // shared cell indices at current depth
    std::vector<complex_type> _shared_snapshot_buf;

  public:
    void build_interaction_list_device(
        const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan );

    void run_m2l_at_depth( int depth );
    void run_l2l_at_depth( int depth );

    template <class PositionType>
    void run_l2p( const PositionType& particle_positions,
                  const potential_view_type& potential_out,
                  const gradient_view_type& gradient_out,
                  bool compute_gradient );

    // Multipole exchange for M2L: receive source multipoles from
    // remote ranks so we can run M2L locally.
    void exchange_multipoles_for_m2l(
        const typename UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::coeff_view_type&
            multipoles,
        const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan );

    // Snapshot shared-cell locals at a given depth into _shared_snapshot_buf.
    // Must be called immediately before run_m2l_at_depth(depth) so that
    // allreduce_shared_locals_at_depth can compute the per-rank M2L *delta*
    // (current - snapshot) instead of summing the cumulative locals — which
    // would over-count the L2L contribution from prior depths by nprocs.
    void snapshot_shared_locals_at_depth(
        int depth, const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan );

    // Allreduce partial M2L contributions at shared cells at a given depth.
    // Uses the snapshot from snapshot_shared_locals_at_depth(depth) to
    // isolate the M2L delta before the MPI_Allreduce.
    void allreduce_shared_locals_at_depth(
        int depth, const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan );

    // Point-to-point exchange of local coefficients after L2L at a depth:
    // children owners receive from parent owners. Direction is the
    // reverse of M2M exchange.
    void exchange_locals_after_l2l_at_depth(
        int depth, const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan );
};

// ============================================================================
// Implementation
// ============================================================================

template <class MemorySpace, class ExecutionSpace, class KernelType>
void DownwardSweep<MemorySpace, ExecutionSpace, KernelType>::setup(
    const UpwardSweep<MemorySpace, ExecutionSpace, KernelType>& upward_sweep,
    int num_local_particles )
{
    _num_local_particles = num_local_particles;

    // Borrow metadata from UpwardSweep
    _device_cells = upward_sweep.device_cells();
    _key_to_cell_idx = &upward_sweep.key_to_cell_idx();
    _A_table = upward_sweep.A_table();
    _max_depth = upward_sweep.max_depth();

    const int num_cells = _device_cells.extent( 0 );

    _locals = coeff_view_type( "locals", num_cells, coeffs_per_cell,
                               NComps );

    // We need particle_cell_idx too. It's private in UpwardSweep, but
    // we can rebuild it from the key-to-idx map and particle keys. A
    // cleaner option is to expose an accessor in UpwardSweep; for now,
    // require the caller to build it via our own helper.
    // --> We add a public getter on UpwardSweep for _particle_cell_idx
    //     if that's acceptable. See note in UpwardSweep.
    // For now we rebuild via the same pattern. We need the particle
    // key view to do that; we'll require the caller to pass it via
    // a setter, or we defer the L2P particle-cell index build to
    // execute() time.
    //
    // Decision: expect UpwardSweep's setup() and DownwardSweep's setup()
    // to be called with the same particle keys. We rebuild locally.
    // For now, placeholder empty; user must re-run setup after any
    // particle redistribution.

    // Build per-depth cell lists (same filter as UpwardSweep:
    // cells we own OR share)
    _leaves_at_depth_local.assign( _max_depth + 1, {} );
    _internals_at_depth_local.assign( _max_depth + 1, {} );
    _all_at_depth_local.assign( _max_depth + 1, {} );

    auto h_dc = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, _device_cells );
    for ( int i = 0; i < num_cells; i++ )
    {
        const auto& dci = h_dc( i );
        bool processes = ( dci.owner_rank == _rank ||
                           dci.owner_rank == OWNER_SHARED );
        if ( !processes )
            continue;

        _all_at_depth_local[dci.depth].push_back( i );
        if ( dci.is_leaf )
            _leaves_at_depth_local[dci.depth].push_back( i );
        else
            _internals_at_depth_local[dci.depth].push_back( i );
    }

    _d_leaves_at_depth.clear();
    _d_internals_at_depth.clear();
    _d_all_at_depth.clear();
    _d_leaves_at_depth.resize( _max_depth + 1 );
    _d_internals_at_depth.resize( _max_depth + 1 );
    _d_all_at_depth.resize( _max_depth + 1 );

    for ( int d = 0; d <= _max_depth; d++ )
    {
        auto vec_to_view =
            [&]( const std::vector<int>& src, Kokkos::View<int*, memory_space>& dest,
                 const char* label ) {
                const size_t n = src.size();
                dest = Kokkos::View<int*, memory_space>( std::string( label ), n );
                if ( n > 0 )
                {
                    auto h = Kokkos::create_mirror_view( dest );
                    for ( size_t i = 0; i < n; i++ )
                        h( i ) = src[i];
                    Kokkos::deep_copy( dest, h );
                }
            };

        vec_to_view( _leaves_at_depth_local[d], _d_leaves_at_depth[d],
                     "leaves_at_depth" );
        vec_to_view( _internals_at_depth_local[d], _d_internals_at_depth[d],
                     "internals_at_depth" );
        vec_to_view( _all_at_depth_local[d], _d_all_at_depth[d],
                     "all_at_depth" );
    }

    // We still need particle_cell_idx for L2P. Borrow via a friend-like
    // backdoor: re-derive from UpwardSweep. Since UpwardSweep exposes
    // its key_to_cell_idx() and the caller must pass its particle_keys,
    // we provide our own setup that accepts the same particle keys.
    //
    // Simpler approach: add an accessor on UpwardSweep to return its
    // particle_cell_idx. Doing that now:
    // (Caller sees: upward_sweep.particle_cell_idx() → we copy reference.)
    // See UpwardSweep modification below.
    _particle_cell_idx = upward_sweep.particle_cell_idx();
}

template <class MemorySpace, class ExecutionSpace, class KernelType>
void DownwardSweep<MemorySpace, ExecutionSpace, KernelType>::
    build_interaction_list_device(
        const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan )
{
    const auto& m2l = comm_plan.m2l_plan();
    const auto& ilists = m2l.interaction_lists;

    // Flatten interaction lists. Target cells are keys in ilists.
    std::vector<int> target_cells;
    std::vector<int> counts;
    std::vector<int> offsets;
    std::vector<int> sources_flat;

    // Shared cells are processed by every rank. For shared targets, all
    // sources in the interaction list are themselves at shared depths
    // (their multipoles are correctly allreduced and identical on every
    // rank), so every rank would compute the SAME M2L delta and the
    // subsequent allreduce would multiply that delta by nprocs. Avoid
    // this double-count by running M2L for shared targets on rank 0
    // only — the allreduce then sums (rank-0 contribution + zeros) to
    // the correct value on every rank. (Non-shared targets are owned by
    // exactly one rank, so this filtering is a no-op for them.)
    auto h_dc_for_filter = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, _device_cells );

    offsets.push_back( 0 );
    for ( const auto& [target_key, sources] : ilists )
    {
        auto it = _key_to_cell_idx->find( target_key );
        if ( it == _key_to_cell_idx->end() )
            continue;
        const int target_idx = it->second;
        const bool target_is_shared =
            ( h_dc_for_filter( target_idx ).owner_rank == OWNER_SHARED );
        if ( target_is_shared && _rank != 0 )
            continue;
        target_cells.push_back( target_idx );

        int count = 0;
        for ( MortonKey src_key : sources )
        {
            auto sit = _key_to_cell_idx->find( src_key );
            if ( sit == _key_to_cell_idx->end() )
                continue;
            sources_flat.push_back( sit->second );
            count++;
        }
        counts.push_back( count );
        offsets.push_back( offsets.back() + count );
    }

    // Upload to device
    const int N = static_cast<int>( target_cells.size() );
    _m2l_target_cells =
        Kokkos::View<int*, memory_space>( "m2l_targets", N );
    _m2l_counts =
        Kokkos::View<int*, memory_space>( "m2l_counts", N );
    _m2l_offsets =
        Kokkos::View<int*, memory_space>( "m2l_offsets", N + 1 );
    _m2l_source_cells_flat = Kokkos::View<int*, memory_space>(
        "m2l_sources_flat",
        static_cast<int>( sources_flat.size() ) );

    if ( N > 0 )
    {
        auto h_t = Kokkos::create_mirror_view( _m2l_target_cells );
        auto h_c = Kokkos::create_mirror_view( _m2l_counts );
        for ( int i = 0; i < N; i++ )
        {
            h_t( i ) = target_cells[i];
            h_c( i ) = counts[i];
        }
        Kokkos::deep_copy( _m2l_target_cells, h_t );
        Kokkos::deep_copy( _m2l_counts, h_c );
    }

    auto h_o = Kokkos::create_mirror_view( _m2l_offsets );
    for ( int i = 0; i <= N; i++ )
        h_o( i ) = offsets[i];
    Kokkos::deep_copy( _m2l_offsets, h_o );

    if ( !sources_flat.empty() )
    {
        auto h_s = Kokkos::create_mirror_view( _m2l_source_cells_flat );
        for ( size_t i = 0; i < sources_flat.size(); i++ )
            h_s( i ) = sources_flat[i];
        Kokkos::deep_copy( _m2l_source_cells_flat, h_s );
    }
}

template <class MemorySpace, class ExecutionSpace, class KernelType>
void DownwardSweep<MemorySpace, ExecutionSpace, KernelType>::exchange_multipoles_for_m2l(
    const typename UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::coeff_view_type&
        multipoles,
    const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan )
{
    // Pre-sweep bulk exchange: we send each of our cells whose multipole
    // is in another rank's interaction list, and receive remote source
    // multipoles we need. This overwrites multipoles at the remote
    // cell indices in our view.
    const auto& m2l = comm_plan.m2l_plan();

    const int per_cell_complex = coeffs_per_cell * NComps;
    const int per_cell_real = 2 * per_cell_complex;
    MPI_Datatype mpi_scalar =
        ( sizeof( scalar_type ) == 8 ) ? MPI_DOUBLE : MPI_FLOAT;

    auto h_mults = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, multipoles );

    // Post receives
    std::vector<MPI_Request> recv_reqs( m2l.receives.size() );
    std::vector<std::vector<complex_type>> recv_bufs( m2l.receives.size() );
    std::vector<int> recv_cell_idx( m2l.receives.size() );

    for ( size_t i = 0; i < m2l.receives.size(); i++ )
    {
        const MortonKey key = m2l.receives[i].cell_key;
        auto it = _key_to_cell_idx->find( key );
        if ( it == _key_to_cell_idx->end() )
        {
            recv_reqs[i] = MPI_REQUEST_NULL;
            recv_cell_idx[i] = -1;
            continue;
        }
        recv_cell_idx[i] = it->second;
        recv_bufs[i].resize( per_cell_complex );

        int tag = static_cast<int>( key & 0x7fffffff );
        MPI_Irecv(
            reinterpret_cast<scalar_type*>( recv_bufs[i].data() ),
            per_cell_real, mpi_scalar, m2l.receives[i].remote_rank, tag,
            _comm, &recv_reqs[i] );
    }

    // Post sends
    std::vector<MPI_Request> send_reqs( m2l.sends.size() );
    std::vector<std::vector<complex_type>> send_bufs( m2l.sends.size() );

    for ( size_t i = 0; i < m2l.sends.size(); i++ )
    {
        const MortonKey key = m2l.sends[i].cell_key;
        auto it = _key_to_cell_idx->find( key );
        if ( it == _key_to_cell_idx->end() )
        {
            send_reqs[i] = MPI_REQUEST_NULL;
            continue;
        }
        const int cidx = it->second;
        send_bufs[i].resize( per_cell_complex );

        int idx = 0;
        for ( int ci = 0; ci < coeffs_per_cell; ci++ )
            for ( int c = 0; c < NComps; c++ )
                send_bufs[i][idx++] = h_mults( cidx, ci, c );

        int tag = static_cast<int>( key & 0x7fffffff );
        MPI_Isend(
            reinterpret_cast<scalar_type*>( send_bufs[i].data() ),
            per_cell_real, mpi_scalar, m2l.sends[i].remote_rank, tag,
            _comm, &send_reqs[i] );
    }

    if ( !recv_reqs.empty() )
        MPI_Waitall( recv_reqs.size(), recv_reqs.data(),
                     MPI_STATUSES_IGNORE );
    if ( !send_reqs.empty() )
        MPI_Waitall( send_reqs.size(), send_reqs.data(),
                     MPI_STATUSES_IGNORE );

    // Unpack received multipoles into the multipole view
    for ( size_t i = 0; i < m2l.receives.size(); i++ )
    {
        if ( recv_cell_idx[i] < 0 )
            continue;
        const int cidx = recv_cell_idx[i];
        int idx = 0;
        for ( int ci = 0; ci < coeffs_per_cell; ci++ )
            for ( int c = 0; c < NComps; c++ )
                h_mults( cidx, ci, c ) = recv_bufs[i][idx++];
    }

    Kokkos::deep_copy( multipoles, h_mults );
}

template <class MemorySpace, class ExecutionSpace, class KernelType>
void DownwardSweep<MemorySpace, ExecutionSpace, KernelType>::run_m2l_at_depth( int depth )
{
    // Run M2L for all target cells at this depth.
    // _m2l_target_cells holds all target cells (across all depths); filter.
    //
    // For simplicity, build per-depth target lists on the host-side
    // interaction list during setup. For now we filter at launch.

    auto device_cells = _device_cells;
    auto m2l_targets = _m2l_target_cells;
    auto m2l_offsets = _m2l_offsets;
    auto m2l_counts = _m2l_counts;
    auto m2l_sources = _m2l_source_cells_flat;
    auto locals = _locals;
    auto A_table = _A_table;

    // We need the multipoles too. They live in the UpwardSweep's view,
    // which was passed into execute(). We capture it via a member
    // variable set at the start of execute() to avoid plumbing through
    // every internal method.
    // Approach: store multipoles pointer in a temporary scratch during
    // execute. Done via _m2l_multipoles_view below.
    auto multipoles = _m2l_multipoles_view;

    const int N = m2l_targets.extent( 0 );
    if ( N == 0 )
        return;

    using team_policy = Kokkos::TeamPolicy<execution_space>;
    using team_member_type = typename team_policy::member_type;

    // Launch one team per target cell. Teams whose target is not at
    // `depth` return early.
    team_policy policy( N, Kokkos::AUTO );

    Kokkos::parallel_for(
        "M2L",
        policy,
        KOKKOS_LAMBDA( const team_member_type& team ) {
            const int league = team.league_rank();
            const int target_cell = m2l_targets( league );
            const auto& target_ci = device_cells( target_cell );
            if ( target_ci.depth != depth )
                return;

            auto L_target = Kokkos::subview( locals, target_cell,
                                             Kokkos::ALL, Kokkos::ALL );

            const int start = m2l_offsets( league );
            const int count = m2l_counts( league );

            for ( int s = 0; s < count; s++ )
            {
                const int source_cell = m2l_sources( start + s );
                const auto& src_ci = device_cells( source_cell );

                const scalar_type dx =
                    src_ci.center[0] - target_ci.center[0];
                const scalar_type dy =
                    src_ci.center[1] - target_ci.center[1];
                const scalar_type dz =
                    src_ci.center[2] - target_ci.center[2];

                KernelType::m2l_translate( team, multipoles, source_cell,
                                           dx, dy, dz, A_table,
                                           L_target );
            }
        } );

    Kokkos::fence();
}

template <class MemorySpace, class ExecutionSpace, class KernelType>
void DownwardSweep<MemorySpace, ExecutionSpace, KernelType>::run_l2l_at_depth( int depth )
{
    // Parents at `depth` translate to their children at `depth+1`.
    // For each parent this rank processes, iterate over its children
    // and call l2l_translate. One team per parent. Each team writes
    // to its (up to 8) children; since each child has only one parent,
    // no atomics are needed.
    const int nparents =
        static_cast<int>( _internals_at_depth_local[depth].size() );
    if ( nparents == 0 )
        return;

    auto locals = _locals;
    auto device_cells = _device_cells;
    auto A_table = _A_table;
    auto& d_parents = _d_internals_at_depth[depth];

    using team_policy = Kokkos::TeamPolicy<execution_space>;
    using team_member_type = typename team_policy::member_type;

    team_policy policy( nparents, Kokkos::AUTO );

    Kokkos::parallel_for(
        "L2L",
        policy,
        KOKKOS_LAMBDA( const team_member_type& team ) {
            const int league = team.league_rank();
            const int parent_cell = d_parents( league );
            const auto& parent_ci = device_cells( parent_cell );

            const MortonKey pk = parent_ci.key;
            const int num_all = device_cells.extent( 0 );

            // Find children of this parent by linear scan (same pattern
            // as M2M; can be optimized with precomputed child tables).
            for ( int ci = 0; ci < num_all; ci++ )
            {
                const auto& ccell = device_cells( ci );
                if ( ( ccell.key >> 3 ) != pk )
                    continue;
                if ( ccell.depth != parent_ci.depth + 1 )
                    continue;

                const scalar_type dx =
                    ccell.center[0] - parent_ci.center[0];
                const scalar_type dy =
                    ccell.center[1] - parent_ci.center[1];
                const scalar_type dz =
                    ccell.center[2] - parent_ci.center[2];

                auto L_child = Kokkos::subview(
                    locals, ci, Kokkos::ALL, Kokkos::ALL );
                KernelType::l2l_translate( team, locals, parent_cell,
                                           dx, dy, dz, A_table,
                                           L_child );
            }
        } );

    Kokkos::fence();
}

template <class MemorySpace, class ExecutionSpace, class KernelType>
void DownwardSweep<MemorySpace, ExecutionSpace, KernelType>::
    snapshot_shared_locals_at_depth(
        int depth, const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan )
{
    // Capture the value of _locals at all shared cells at this depth
    // BEFORE M2L runs. After M2L, allreduce_shared_locals_at_depth will
    // use the snapshot to isolate the M2L contribution (which differs
    // between ranks and must be summed) from the L2L-inherited part
    // (which is identical on all ranks and must NOT be summed).
    const auto& m2m = comm_plan.m2m_plan();
    _shared_snapshot_indices.clear();

    auto h_dc = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, _device_cells );
    for ( MortonKey k : m2m.shared_cells )
    {
        auto it = _key_to_cell_idx->find( k );
        if ( it == _key_to_cell_idx->end() )
            continue;
        if ( h_dc( it->second ).depth == depth )
            _shared_snapshot_indices.push_back( it->second );
    }

    const int nshared = static_cast<int>( _shared_snapshot_indices.size() );
    const int per_cell_complex = coeffs_per_cell * NComps;
    _shared_snapshot_buf.assign( nshared * per_cell_complex,
                                 complex_type( 0, 0 ) );
    if ( nshared == 0 )
        return;

    auto h_locals = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, _locals );
    for ( int i = 0; i < nshared; i++ )
    {
        const int cidx = _shared_snapshot_indices[i];
        int idx = 0;
        for ( int ci = 0; ci < coeffs_per_cell; ci++ )
            for ( int c = 0; c < NComps; c++ )
                _shared_snapshot_buf[i * per_cell_complex + ( idx++ )] =
                    h_locals( cidx, ci, c );
    }
}

template <class MemorySpace, class ExecutionSpace, class KernelType>
void DownwardSweep<MemorySpace, ExecutionSpace, KernelType>::
    allreduce_shared_locals_at_depth(
        int depth, const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan )
{
    // Shared cells at this depth had M2L run by every rank with each
    // rank using the multipoles it has locally. We need to sum the
    // partial M2L contributions across ranks. The snapshot captured
    // before M2L lets us subtract out the L2L-inherited value (which
    // is the same on every rank) so the allreduce only sums the
    // genuinely-disjoint M2L deltas.
    (void)comm_plan; // shared_cell_indices already cached in snapshot
    const int nshared = static_cast<int>( _shared_snapshot_indices.size() );
    if ( nshared == 0 )
        return;

    const int per_cell_complex = coeffs_per_cell * NComps;
    const int total_complex = nshared * per_cell_complex;

    std::vector<complex_type> sendbuf( total_complex );
    std::vector<complex_type> recvbuf( total_complex );

    auto h_locals = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, _locals );

    // sendbuf = current - snapshot (M2L delta on this rank)
    for ( int i = 0; i < nshared; i++ )
    {
        const int cidx = _shared_snapshot_indices[i];
        int idx = 0;
        for ( int ci = 0; ci < coeffs_per_cell; ci++ )
            for ( int c = 0; c < NComps; c++ )
            {
                const int k = i * per_cell_complex + ( idx++ );
                sendbuf[k] = h_locals( cidx, ci, c ) -
                             _shared_snapshot_buf[k];
            }
    }

    MPI_Datatype mpi_scalar =
        ( sizeof( scalar_type ) == 8 ) ? MPI_DOUBLE : MPI_FLOAT;
    MPI_Allreduce( reinterpret_cast<scalar_type*>( sendbuf.data() ),
                   reinterpret_cast<scalar_type*>( recvbuf.data() ),
                   2 * total_complex, mpi_scalar, MPI_SUM, _comm );

    // _locals[shared] = snapshot + summed delta
    for ( int i = 0; i < nshared; i++ )
    {
        const int cidx = _shared_snapshot_indices[i];
        int idx = 0;
        for ( int ci = 0; ci < coeffs_per_cell; ci++ )
            for ( int c = 0; c < NComps; c++ )
            {
                const int k = i * per_cell_complex + ( idx++ );
                h_locals( cidx, ci, c ) =
                    _shared_snapshot_buf[k] + recvbuf[k];
            }
    }

    Kokkos::deep_copy( _locals, h_locals );
}

template <class MemorySpace, class ExecutionSpace, class KernelType>
void DownwardSweep<MemorySpace, ExecutionSpace, KernelType>::
    exchange_locals_after_l2l_at_depth(
        int depth, const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan )
{
    // L2L plan: parent-owner sends L to child-owner. The cell_key in
    // the L2L plan is the PARENT's key. We filter entries where the
    // parent is at `depth` — meaning this exchange happens after L2L
    // has completed at depth d and populated children at d+1 on the
    // parent's rank. The children on child-owner rank are at d+1 too,
    // so we send the children's resulting local coefficients.
    //
    // Wait — the L2L plan as built in CommunicationPlan encodes what
    // we need to send and receive for the parent->child transfer. The
    // cell_key is the PARENT, and remote_rank is the CHILD's owner.
    // We send the parent's local, but actually what we want to transmit
    // is the result of applying L2L to each child — which is the child's
    // local coefficients AFTER this rank (the parent owner) has added
    // the L2L contribution.
    //
    // Reconsidering: after L2L runs on the parent's rank, it writes
    // into the CHILD'S storage on the parent's rank. We then send the
    // CHILD'S updated local to the CHILD'S owner. The child's owner
    // adds the received values to its own child-local storage (which
    // may have additional M2L contributions that arrived from elsewhere).
    //
    // To implement this cleanly, we should send the children's locals,
    // not the parent's. We iterate the plan and use the parent key to
    // discover which children need to be sent.

    const auto& l2l = comm_plan.l2l_plan();

    const int per_cell_complex = coeffs_per_cell * NComps;
    const int per_cell_real = 2 * per_cell_complex;
    MPI_Datatype mpi_scalar =
        ( sizeof( scalar_type ) == 8 ) ? MPI_DOUBLE : MPI_FLOAT;

    auto h_dc = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, _device_cells );

    // The cell_key in each L2L plan entry is the CHILD's key. Filter
    // entries to those whose child is at depth+1 (i.e. whose parent is
    // at the depth we just finished L2L'ing on).
    struct PendingSend
    {
        int child_cell_idx;
        int remote_rank;
    };
    std::vector<PendingSend> sends;

    struct PendingRecv
    {
        int child_cell_idx;
        int remote_rank;
    };
    std::vector<PendingRecv> recvs;

    for ( const auto& ct : l2l.sends )
    {
        auto it = _key_to_cell_idx->find( ct.cell_key );
        if ( it == _key_to_cell_idx->end() )
            continue;
        const int cidx = it->second;
        if ( h_dc( cidx ).depth != depth + 1 )
            continue;
        sends.push_back( { cidx, ct.remote_rank } );
    }

    for ( const auto& ct : l2l.receives )
    {
        auto it = _key_to_cell_idx->find( ct.cell_key );
        if ( it == _key_to_cell_idx->end() )
            continue;
        const int cidx = it->second;
        if ( h_dc( cidx ).depth != depth + 1 )
            continue;
        recvs.push_back( { cidx, ct.remote_rank } );
    }

    if ( sends.empty() && recvs.empty() )
        return;

    auto h_locals = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, _locals );

    std::vector<MPI_Request> recv_reqs( recvs.size() );
    std::vector<std::vector<complex_type>> recv_bufs( recvs.size() );
    for ( size_t i = 0; i < recvs.size(); i++ )
    {
        recv_bufs[i].resize( per_cell_complex );
        const MortonKey key = h_dc( recvs[i].child_cell_idx ).key;
        int tag = static_cast<int>( key & 0x7fffffff );
        MPI_Irecv(
            reinterpret_cast<scalar_type*>( recv_bufs[i].data() ),
            per_cell_real, mpi_scalar, recvs[i].remote_rank, tag, _comm,
            &recv_reqs[i] );
    }

    std::vector<MPI_Request> send_reqs( sends.size() );
    std::vector<std::vector<complex_type>> send_bufs( sends.size() );
    for ( size_t i = 0; i < sends.size(); i++ )
    {
        send_bufs[i].resize( per_cell_complex );
        const int cidx = sends[i].child_cell_idx;
        int idx = 0;
        for ( int ci = 0; ci < coeffs_per_cell; ci++ )
            for ( int c = 0; c < NComps; c++ )
                send_bufs[i][idx++] = h_locals( cidx, ci, c );

        const MortonKey key = h_dc( cidx ).key;
        int tag = static_cast<int>( key & 0x7fffffff );
        MPI_Isend(
            reinterpret_cast<scalar_type*>( send_bufs[i].data() ),
            per_cell_real, mpi_scalar, sends[i].remote_rank, tag,
            _comm, &send_reqs[i] );
    }

    if ( !recv_reqs.empty() )
        MPI_Waitall( recv_reqs.size(), recv_reqs.data(),
                     MPI_STATUSES_IGNORE );
    if ( !send_reqs.empty() )
        MPI_Waitall( send_reqs.size(), send_reqs.data(),
                     MPI_STATUSES_IGNORE );

    // Accumulate received locals into the child's local (the child
    // owner may already have M2L contributions there)
    for ( size_t i = 0; i < recvs.size(); i++ )
    {
        const int cidx = recvs[i].child_cell_idx;
        int idx = 0;
        for ( int ci = 0; ci < coeffs_per_cell; ci++ )
            for ( int c = 0; c < NComps; c++ )
                h_locals( cidx, ci, c ) += recv_bufs[i][idx++];
    }

    Kokkos::deep_copy( _locals, h_locals );
}

template <class MemorySpace, class ExecutionSpace, class KernelType>
template <class PositionType>
void DownwardSweep<MemorySpace, ExecutionSpace, KernelType>::run_l2p(
    const PositionType& particle_positions,
    const potential_view_type& potential_out,
    const gradient_view_type& gradient_out, bool compute_gradient )
{
    auto locals = _locals;
    auto device_cells = _device_cells;
    auto particle_cell_idx = _particle_cell_idx;
    const int N = _num_local_particles;

    Kokkos::parallel_for(
        "L2P",
        Kokkos::RangePolicy<execution_space>( 0, N ),
        KOKKOS_LAMBDA( int p ) {
            const int cidx = particle_cell_idx( p );
            if ( cidx < 0 )
                return;

            const auto& dci = device_cells( cidx );
            if ( !dci.is_leaf )
                return; // should not happen, but be safe

            const scalar_type px =
                static_cast<scalar_type>( particle_positions( p, 0 ) );
            const scalar_type py =
                static_cast<scalar_type>( particle_positions( p, 1 ) );
            const scalar_type pz =
                static_cast<scalar_type>( particle_positions( p, 2 ) );

            const scalar_type dx = px - dci.center[0];
            const scalar_type dy = py - dci.center[1];
            const scalar_type dz = pz - dci.center[2];

            scalar_type phi[NComps];

            // GradWriter is a class-level struct (not a nested lambda) so
            // CUDA can use it inside a device lambda without the
            // "nested extended lambda" restriction firing.
            GradWriter writer{ gradient_out, p };
            KernelType::l2p_evaluate( locals, cidx, dx, dy, dz, phi,
                                      writer, compute_gradient );

            // Accumulate potential (caller has zeroed or initialized)
            for ( int c = 0; c < NComps; c++ )
                potential_out( p, c ) += phi[c];
        } );

    Kokkos::fence();
}

// -------------------------------------------------------------------------
// Private member for multipoles pointer used by M2L kernels.
// We store it at the start of execute() and reference inside
// run_m2l_at_depth. Declared here as a trailing private data member.
// -------------------------------------------------------------------------
// (Declared in a separate section below in the class to keep the
// implementation above readable. For this single-file header, we
// add the member as an 'mutable' so we can set it from execute(),
// which is a non-const method anyway.)

// Forward-declared storage for the multipoles view during execute().
// We add this member to the class.
// (Implemented by adding _m2l_multipoles_view as a member.)

template <class MemorySpace, class ExecutionSpace, class KernelType>
template <class PositionType>
void DownwardSweep<MemorySpace, ExecutionSpace, KernelType>::execute(
    const typename UpwardSweep<MemorySpace, ExecutionSpace, KernelType>::coeff_view_type&
        multipoles,
    const PositionType& particle_positions,
    const potential_view_type& potential_out,
    const gradient_view_type& gradient_out, bool compute_gradient,
    const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan )
{
    // Zero local coefficients
    Kokkos::deep_copy( _locals, complex_type( 0.0, 0.0 ) );

    // Stash the multipoles for the M2L kernel
    _m2l_multipoles_view = multipoles;

    // Build device-side interaction list if needed
    if ( _m2l_target_cells.extent( 0 ) == 0 )
        build_interaction_list_device( comm_plan );

    // Pre-sweep: exchange remote multipoles needed for M2L
    exchange_multipoles_for_m2l( multipoles, comm_plan );

    // Layer-by-layer: snapshot shared, M2L, allreduce shared M2L delta,
    // L2L, exchange children
    for ( int d = 0; d <= _max_depth; d++ )
    {
        snapshot_shared_locals_at_depth( d, comm_plan );
        run_m2l_at_depth( d );
        allreduce_shared_locals_at_depth( d, comm_plan );
        run_l2l_at_depth( d );
        if ( d < _max_depth )
            exchange_locals_after_l2l_at_depth( d, comm_plan );
    }

    // L2P: evaluate local expansion at each particle
    run_l2p( particle_positions, potential_out, gradient_out,
             compute_gradient );
}

} // namespace Canopy

#endif // CANOPY_DOWNWARD_SWEEP_HPP
