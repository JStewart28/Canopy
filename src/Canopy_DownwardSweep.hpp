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

#include "Canopy_BatchedGemm.hpp"
#include "Canopy_CommunicationPlan.hpp"
#include "Canopy_Profiling.hpp"
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
    static constexpr int coeffs_per_cell = KernelType::num_coeffs_per_cell;
    static constexpr int NComps = KernelType::num_components;

    // Local coefficient storage
    using coeff_view_type = Kokkos::View<complex_type***, memory_space>;

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
    using cell_view_type = typename UpwardSweep<MemorySpace, ExecutionSpace,
                                                KernelType>::cell_view_type;
    using particle_cell_idx_view_type =
        typename UpwardSweep<MemorySpace, ExecutionSpace,
                             KernelType>::particle_cell_idx_view_type;

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
    void setup( const UpwardSweep<MemorySpace, ExecutionSpace, KernelType>&
                    upward_sweep,
                int num_local_particles );

    // -----------------------------------------------------------------------
    // allocate_potential / allocate_gradient — helper to size outputs.
    // Callers can use these or allocate their own.
    // -----------------------------------------------------------------------
    potential_view_type allocate_potential( int num_local_particles ) const
    {
        return potential_view_type( "fmm_potential", num_local_particles );
    }

    gradient_view_type allocate_gradient( int num_local_particles ) const
    {
        return gradient_view_type( "fmm_gradient", num_local_particles );
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
        const typename UpwardSweep<MemorySpace, ExecutionSpace,
                                   KernelType>::coeff_view_type& multipoles,
        const PositionType& particle_positions,
        const potential_view_type& potential_out,
        const gradient_view_type& gradient_out, bool compute_gradient,
        const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan );

    // Access the computed local coefficients after execute()
    const coeff_view_type& locals() const { return _locals; }

    // -----------------------------------------------------------------------
    // invalidate_interaction_list()
    //
    // Marks the cached M2L interaction list (all per-tree / per-comm-plan
    // device structures populated by build_interaction_list_device) as
    // stale, so the next execute() rebuilds it. Must be called whenever
    // the tree topology or communication plan changes — e.g. after
    // TreeBuilder::build() that adds/removes cells, after
    // TreePartitioner::repartition(), or after CommunicationPlan::build().
    // setup() does this implicitly, so callers that already re-run setup()
    // do not need to call this directly.
    // -----------------------------------------------------------------------
    void invalidate_interaction_list() { _interaction_list_dirty = true; }

    // Number of times build_interaction_list_device has actually performed
    // a rebuild (i.e. did not early-return because dirty was false). Used
    // by tests to verify caching.
    int interaction_list_build_count() const
    {
        return _interaction_list_build_count;
    }

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
    //
    // Targets are sorted by depth so that run_m2l_at_depth(d) launches
    // only over the contiguous range [depth_offsets[d], depth_offsets[d+1]),
    // avoiding the per-team `if (depth != d) return` filter that ran a
    // team for every target at every depth.
    Kokkos::View<int*, memory_space> _m2l_target_cells;
    Kokkos::View<int*, memory_space> _m2l_source_cells_flat;
    Kokkos::View<int*, memory_space> _m2l_offsets; // size N+1
    Kokkos::View<int*, memory_space> _m2l_counts;  // size N
    std::vector<int> _m2l_depth_offsets;           // size max_depth + 2 (host)

    // Stage 3: hashed cross-depth M2L operator table.
    //
    // The M2L operator T is a function of the physical translation vector
    // only. Two pairs sharing a vector share the operator regardless of
    // the depths involved. We canonicalize each pair to an integer key
    //   key = (max_d, dd, ii, jj, kk)
    //   max_d = max(d_t, d_s)        // deeper of target/source depths
    //   dd    = d_s - d_t            // depth difference (signed)
    //   unit_w = w0 / 2^max_d        // smaller cell width
    //   (ii,jj,kk) = round((c_s - c_t) / unit_w)
    // and reuse one T(Nt, Ns) per unique key.
    //
    // Range guards (|dd|<=6, |offset|<=32) catch pathological pairs and
    // route them to the per-pair m2l_translate fallback. Healthy MAC
    // traversals never trip these.
    static constexpr int M2L_KEY_DD_MAX = 6;
    static constexpr int M2L_KEY_OFFSET_MAX = 32;
    static constexpr int M2L_OP_COUNT_CAP = 32768;

    static constexpr int M2L_NUM_SRC = ( P + 1 ) * ( P + 1 );

    struct M2LKey
    {
        int max_d;
        int dd;
        int ii;
        int jj;
        int kk;
        bool operator==( const M2LKey& o ) const noexcept
        {
            return max_d == o.max_d && dd == o.dd && ii == o.ii &&
                   jj == o.jj && kk == o.kk;
        }
    };
    struct M2LKeyHash
    {
        std::size_t operator()( const M2LKey& k ) const noexcept
        {
            std::uint64_t h = 1469598103934665603ull;
            auto mix = [&]( int v )
            {
                h ^= static_cast<std::uint32_t>( v );
                h *= 1099511628211ull;
            };
            mix( k.max_d );
            mix( k.dd );
            mix( k.ii );
            mix( k.jj );
            mix( k.kk );
            return static_cast<std::size_t>( h );
        }
    };

    // Operator tables: shape (Nt, Ns, n_unique_ops). LayoutLeft so a
    // subview(_m2l_op_table, ALL, ALL, op_idx) is a contiguous column-major
    // (Nt, Ns) matrix consumable by cuBLAS / hipBLAS / KokkosBlas::gemm
    // without copy or transpose.
    Kokkos::View<complex_type***, Kokkos::LayoutLeft, memory_space>
        _m2l_op_table;

    // Op-major pair layout, partitioned into "non-shared-target" and
    // "shared-target" sets:
    //
    //   nonshared: pairs whose target cell has owner_rank != OWNER_SHARED.
    //              Processed in run_m2l_all() once per solve, before the
    //              per-depth loop. M2L for these targets does not interact
    //              with the snapshot/allreduce barrier, so we batch all
    //              depths together for maximum GEMM size.
    //
    //   shared:    pairs whose target cell is shared (rank-0 only — see
    //              entry-collection filter). Sub-sliced by target depth so
    //              that run_m2l_at_depth(d) runs only the contributions
    //              landing on shared cells at depth d, preserving the
    //              snapshot/allreduce semantics.
    //
    // Within each set, pairs are grouped by op_idx (so a single GEMM per op
    // lands on a contiguous column slice of the packed scratch buffers).
    // Within shared ops, pairs are additionally sorted by target depth.
    Kokkos::View<int*, memory_space> _m2l_nonshared_pair_targets;
    Kokkos::View<int*, memory_space> _m2l_nonshared_pair_sources;
    std::vector<int> _m2l_nonshared_op_keys;     // op_idx per nonshared op
    std::vector<int> _m2l_nonshared_op_offsets;  // length n_nonshared_ops + 1

    Kokkos::View<int*, memory_space> _m2l_shared_pair_targets;
    Kokkos::View<int*, memory_space> _m2l_shared_pair_sources;
    std::vector<int> _m2l_shared_op_keys;        // op_idx per shared op
    std::vector<int> _m2l_shared_op_offsets;     // length n_shared_ops + 1
    // For shared op i, pairs at target depths < d live in slots
    //   [_m2l_shared_op_offsets[i], _m2l_shared_op_offsets[i] +
    //    _m2l_shared_op_depth_starts[i][d]).
    // Length per op: max_depth + 2.
    std::vector<std::vector<int>> _m2l_shared_op_depth_starts;

    // Fallback (per-pair m2l_translate) for pairs hitting the M2L_KEY_*
    // guardrails or the M2L_OP_COUNT_CAP overflow valve. Sliced per depth
    // so run_m2l_at_depth(d) handles the depth-d slice (snapshot/allreduce
    // for shared targets at d works the same as the GEMM-path scatter).
    Kokkos::View<int*, memory_space> _m2l_fallback_targets;
    Kokkos::View<int*, memory_space> _m2l_fallback_sources;
    std::vector<int> _m2l_fallback_offsets_host;  // length max_depth + 2

    // Per-depth fallback pair counts (= consecutive differences of
    // _m2l_fallback_offsets_host). Surfaced via total_fallback_pair_count()
    // so the bin-edge regression test can assert the fallback path fires.
    std::vector<long long> _m2l_fallback_count_per_active_depth;

    // Per-solve scratch buffers for the batched-GEMM M2L pipeline:
    //   M_packed shape (Ns, max_pairs_at_depth * NComps)
    //   L_packed shape (Nt, max_pairs_at_depth * NComps)
    // Sized after build_interaction_list_device to the largest active depth
    // and reused across all solves and across all depths within a solve.
    // LayoutLeft so the GEMM operates on contiguous column-major matrices.
    Kokkos::View<complex_type**, Kokkos::LayoutLeft, memory_space>
        _m2l_M_packed;
    Kokkos::View<complex_type**, Kokkos::LayoutLeft, memory_space>
        _m2l_L_packed;

    // Vendor / KokkosKernels GEMM wrapper. Default-constructible; a real
    // backend handle (cuBLAS / hipBLAS) is created on first solve.
    detail::M2LBatchedGemm<execution_space, scalar_type> _m2l_gemm;

    // Scratch: reference to the multipoles view during execute().
    // The M2L kernels capture this. Set at the start of execute()
    // and used by run_m2l_at_depth().
    typename UpwardSweep<MemorySpace, ExecutionSpace,
                         KernelType>::coeff_view_type _m2l_multipoles_view;

    // Caching: build_interaction_list_device early-returns when this is
    // false. setup() and invalidate_interaction_list() set it to true; the
    // builder clears it at the end of a successful rebuild.
    bool _interaction_list_dirty = true;

    // Count of actual rebuilds done by build_interaction_list_device (does
    // not increment on the early-return path). Surfaced by
    // interaction_list_build_count() for the caching tests.
    int _interaction_list_build_count = 0;

    // Per-depth snapshot of shared-cell locals taken before M2L at that
    // depth. Used by allreduce_shared_locals_at_depth to isolate the
    // M2L delta from L2L contributions inherited from prior depths.
    std::vector<int>
        _shared_snapshot_indices; // shared cell indices at current depth
    std::vector<complex_type> _shared_snapshot_buf;

  public:
    void build_interaction_list_device(
        const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan );

    // Process all M2L pairs whose target is non-shared, batched across
    // every depth. Called once per solve, before the per-depth loop.
    void run_m2l_all();

    void run_m2l_at_depth( int depth );
    void run_l2l_at_depth( int depth );

    // Total number of M2L pairs this rank carries through the per-pair
    // m2l_translate fallback path (only pairs hitting the M2L_KEY_*
    // guardrails or the M2L_OP_COUNT_CAP overflow valve). Used by the
    // bin-edge regression test to assert that a configuration intended to
    // exercise the fallback actually does.
    long long total_fallback_pair_count() const
    {
        long long s = 0;
        for ( long long x : _m2l_fallback_count_per_active_depth )
            s += x;
        return s;
    }

    // Total M2L pair count carried by this rank: GEMM path (nonshared +
    // shared) plus fallback path. Used alongside total_fallback_pair_count()
    // to compute the fraction of pairs missing the GEMM fast path.
    long long total_m2l_pair_count() const
    {
        return static_cast<long long>(
                   _m2l_nonshared_pair_targets.extent( 0 ) ) +
               static_cast<long long>(
                   _m2l_shared_pair_targets.extent( 0 ) ) +
               static_cast<long long>( _m2l_fallback_targets.extent( 0 ) );
    }

    // Per-pair fallback for out-of-range pairs at depth `depth`.
    void run_m2l_fallback_at_depth( int depth );

  public:

    template <class PositionType>
    void run_l2p( const PositionType& particle_positions,
                  const potential_view_type& potential_out,
                  const gradient_view_type& gradient_out,
                  bool compute_gradient );

    // Multipole exchange for M2L: receive source multipoles from
    // remote ranks so we can run M2L locally.
    void exchange_multipoles_for_m2l(
        const typename UpwardSweep<MemorySpace, ExecutionSpace,
                                   KernelType>::coeff_view_type& multipoles,
        const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan );

    // Snapshot shared-cell locals at a given depth into _shared_snapshot_buf.
    // Must be called immediately before run_m2l_at_depth(depth) so that
    // allreduce_shared_locals_at_depth can compute the per-rank M2L *delta*
    // (current - snapshot) instead of summing the cumulative locals — which
    // would over-count the L2L contribution from prior depths by nprocs.
    void snapshot_shared_locals_at_depth(
        int depth,
        const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan );

    // Allreduce partial M2L contributions at shared cells at a given depth.
    // Uses the snapshot from snapshot_shared_locals_at_depth(depth) to
    // isolate the M2L delta before the MPI_Allreduce.
    void allreduce_shared_locals_at_depth(
        int depth,
        const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan );

    // Point-to-point exchange of local coefficients after L2L at a depth:
    // children owners receive from parent owners. Direction is the
    // reverse of M2M exchange.
    void exchange_locals_after_l2l_at_depth(
        int depth,
        const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan );
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

    _locals = coeff_view_type( "locals", num_cells, coeffs_per_cell, NComps );

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

    auto h_dc = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{},
                                                     _device_cells );
    for ( int i = 0; i < num_cells; i++ )
    {
        const auto& dci = h_dc( i );
        bool processes =
            ( dci.owner_rank == _rank || dci.owner_rank == OWNER_SHARED );
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
        auto vec_to_view = [&]( const std::vector<int>& src,
                                Kokkos::View<int*, memory_space>& dest,
                                const char* label )
        {
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

    // Invalidate the cached M2L interaction list so the next execute()
    // rebuilds it against the current tree. Without this, a re-setup
    // after a tree-topology change would silently reuse stale cell
    // indices from the previous tree.
    _m2l_target_cells = Kokkos::View<int*, memory_space>();
    _m2l_source_cells_flat = Kokkos::View<int*, memory_space>();
    _m2l_offsets = Kokkos::View<int*, memory_space>();
    _m2l_counts = Kokkos::View<int*, memory_space>();
    _m2l_depth_offsets.clear();
    _m2l_op_table =
        Kokkos::View<complex_type***, Kokkos::LayoutLeft, memory_space>();
    _m2l_nonshared_pair_targets = Kokkos::View<int*, memory_space>();
    _m2l_nonshared_pair_sources = Kokkos::View<int*, memory_space>();
    _m2l_nonshared_op_keys.clear();
    _m2l_nonshared_op_offsets.clear();
    _m2l_shared_pair_targets = Kokkos::View<int*, memory_space>();
    _m2l_shared_pair_sources = Kokkos::View<int*, memory_space>();
    _m2l_shared_op_keys.clear();
    _m2l_shared_op_offsets.clear();
    _m2l_shared_op_depth_starts.clear();
    _m2l_fallback_targets = Kokkos::View<int*, memory_space>();
    _m2l_fallback_sources = Kokkos::View<int*, memory_space>();
    _m2l_fallback_offsets_host.clear();
    _m2l_M_packed =
        Kokkos::View<complex_type**, Kokkos::LayoutLeft, memory_space>();
    _m2l_L_packed =
        Kokkos::View<complex_type**, Kokkos::LayoutLeft, memory_space>();
    _m2l_fallback_count_per_active_depth.clear();
    _interaction_list_dirty = true;
}

template <class MemorySpace, class ExecutionSpace, class KernelType>
void DownwardSweep<MemorySpace, ExecutionSpace, KernelType>::
    build_interaction_list_device(
        const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan )
{
    if ( !_interaction_list_dirty )
        return;

    const auto& m2l = comm_plan.m2l_plan();
    const auto& ilists = m2l.interaction_lists;

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

    // First pass: collect (target_idx, sources_idx_vec, depth) so we can
    // sort by depth before flattening into the CSR. Depth-sorted target
    // order lets run_m2l_at_depth(d) launch only over the contiguous
    // [depth_offsets[d], depth_offsets[d+1]) slice — no per-team filter.
    struct TargetEntry
    {
        int target_idx;
        int depth;
        std::vector<int> sources;
    };
    std::vector<TargetEntry> entries;
    entries.reserve( ilists.size() );

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

        TargetEntry e;
        e.target_idx = target_idx;
        e.depth = h_dc_for_filter( target_idx ).depth;
        e.sources.reserve( sources.size() );
        for ( MortonKey src_key : sources )
        {
            auto sit = _key_to_cell_idx->find( src_key );
            if ( sit == _key_to_cell_idx->end() )
                continue;
            e.sources.push_back( sit->second );
        }
        entries.push_back( std::move( e ) );
    }

    std::sort( entries.begin(), entries.end(),
               []( const TargetEntry& a, const TargetEntry& b )
               {
                   if ( a.depth != b.depth )
                       return a.depth < b.depth;
                   return a.target_idx < b.target_idx;
               } );

    std::vector<int> target_cells;
    std::vector<int> counts;
    std::vector<int> offsets;
    std::vector<int> sources_flat;
    target_cells.reserve( entries.size() );
    counts.reserve( entries.size() );
    offsets.reserve( entries.size() + 1 );
    offsets.push_back( 0 );
    for ( const auto& e : entries )
    {
        target_cells.push_back( e.target_idx );
        counts.push_back( static_cast<int>( e.sources.size() ) );
        for ( int s : e.sources )
            sources_flat.push_back( s );
        offsets.push_back( offsets.back() +
                           static_cast<int>( e.sources.size() ) );
    }

    // Build host-side per-depth offsets into the sorted target list:
    // _m2l_depth_offsets[d] = first index in target_cells with depth >= d.
    _m2l_depth_offsets.assign( _max_depth + 2, 0 );
    {
        int cursor = 0;
        for ( int d = 0; d <= _max_depth + 1; d++ )
        {
            while ( cursor < static_cast<int>( entries.size() ) &&
                    entries[cursor].depth < d )
                ++cursor;
            _m2l_depth_offsets[d] = cursor;
        }
    }

    // Upload to device
    const int N = static_cast<int>( target_cells.size() );
    _m2l_target_cells = Kokkos::View<int*, memory_space>(
        Kokkos::view_alloc( Kokkos::WithoutInitializing, "m2l_targets" ), N );
    _m2l_counts = Kokkos::View<int*, memory_space>(
        Kokkos::view_alloc( Kokkos::WithoutInitializing, "m2l_counts" ), N );
    _m2l_offsets = Kokkos::View<int*, memory_space>(
        Kokkos::view_alloc( Kokkos::WithoutInitializing, "m2l_offsets" ),
        N + 1 );
    _m2l_source_cells_flat = Kokkos::View<int*, memory_space>(
        Kokkos::view_alloc( Kokkos::WithoutInitializing, "m2l_sources_flat" ),
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

    // -----------------------------------------------------------------------
    // Stage 3: hash-map every (target, source) pair to a unique M2L operator
    // key (max_d, dd, ii, jj, kk). Pairs sharing a key share an operator T,
    // regardless of which depths target / source live at.
    //
    // Unit grid: cell at depth d, index i has center (2i+1) * half_w_d, so
    // the difference of any two centers (at any depths) is an integer
    // multiple of half_w_max_d, where max_d = max(d_t, d_s). We use that as
    // the unit length.
    //
    // Range guards: pairs with |dd| > M2L_KEY_DD_MAX or |offset| component >
    // M2L_KEY_OFFSET_MAX route to per-pair m2l_translate fallback. These
    // guards exist purely to defend against pathological tree state; a
    // healthy MAC traversal does not produce out-of-range pairs.
    // -----------------------------------------------------------------------
    std::vector<double> half_width_at_depth( _max_depth + 1, 0.0 );
    {
        const int num_cells = _device_cells.extent( 0 );
        for ( int i = 0; i < num_cells; i++ )
        {
            const auto& dci = h_dc_for_filter( i );
            if ( dci.depth >= 0 && dci.depth <= _max_depth )
                half_width_at_depth[dci.depth] = dci.half_width;
        }
    }

    const int total_pairs = static_cast<int>( sources_flat.size() );

    // Per-pair information collected during the classification pass.
    // op_idx == -1 means the pair routes to the fallback path.
    std::vector<int> pair_op_idx( total_pairs, -1 );
    std::vector<int> pair_target( total_pairs );
    std::vector<int> pair_source( total_pairs );
    std::vector<int> pair_target_depth( total_pairs );
    std::vector<unsigned char> pair_target_is_shared( total_pairs, 0 );

    std::unordered_map<M2LKey, int, M2LKeyHash> key_to_op;
    std::vector<M2LKey> ops;
    bool overflow_warned = false;

    {
        size_t pair_cursor = 0;
        for ( const auto& e : entries )
        {
            const auto& tci = h_dc_for_filter( e.target_idx );
            const bool tgt_shared =
                ( tci.owner_rank == OWNER_SHARED );
            for ( int s : e.sources )
            {
                const auto& sci = h_dc_for_filter( s );
                const int max_d = std::max( e.depth, sci.depth );
                const double unit_w =
                    ( max_d >= 0 && max_d <= _max_depth )
                        ? half_width_at_depth[max_d]
                        : 0.0;
                const double inv_unit_w =
                    ( unit_w > 0.0 ) ? ( 1.0 / unit_w ) : 0.0;
                const double dx = sci.center[0] - tci.center[0];
                const double dy = sci.center[1] - tci.center[1];
                const double dz = sci.center[2] - tci.center[2];
                const int ii = static_cast<int>(
                    std::lround( dx * inv_unit_w ) );
                const int jj = static_cast<int>(
                    std::lround( dy * inv_unit_w ) );
                const int kk = static_cast<int>(
                    std::lround( dz * inv_unit_w ) );
                const int dd = sci.depth - e.depth;

                int op_idx = -1;
                if ( unit_w > 0.0 && std::abs( dd ) <= M2L_KEY_DD_MAX &&
                     std::abs( ii ) <= M2L_KEY_OFFSET_MAX &&
                     std::abs( jj ) <= M2L_KEY_OFFSET_MAX &&
                     std::abs( kk ) <= M2L_KEY_OFFSET_MAX )
                {
                    M2LKey key{ max_d, dd, ii, jj, kk };
                    auto it = key_to_op.find( key );
                    if ( it != key_to_op.end() )
                    {
                        op_idx = it->second;
                    }
                    else if ( static_cast<int>( ops.size() ) <
                              M2L_OP_COUNT_CAP )
                    {
                        op_idx = static_cast<int>( ops.size() );
                        key_to_op.emplace( key, op_idx );
                        ops.push_back( key );
                    }
                    else if ( !overflow_warned )
                    {
                        std::fprintf(
                            stderr,
                            "[Canopy] M2L op count exceeded cap %d; "
                            "remaining pairs route to fallback path.\n",
                            M2L_OP_COUNT_CAP );
                        overflow_warned = true;
                    }
                }

                pair_op_idx[pair_cursor] = op_idx;
                pair_target[pair_cursor] = e.target_idx;
                pair_source[pair_cursor] = s;
                pair_target_depth[pair_cursor] = e.depth;
                pair_target_is_shared[pair_cursor] =
                    tgt_shared ? 1u : 0u;
                ++pair_cursor;
            }
        }
    }

    const int n_unique_ops = static_cast<int>( ops.size() );

    // -----------------------------------------------------------------------
    // Stage 4: build the (Nt, Ns, n_unique_ops) operator table on host,
    // then deep_copy to device. Each op key encodes the physical translation
    // (ii, jj, kk) * unit_w, with unit_w = width_at_depth[max_d]; the
    // operator only depends on this physical vector.
    // -----------------------------------------------------------------------
    {
        const int Nt = KernelType::num_coeffs_per_cell;
        const int Ns = KernelType::m2l_num_src_coeffs;
        Kokkos::View<complex_type***, Kokkos::LayoutLeft, memory_space>
            op_table( Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                          "m2l_op_table" ),
                      Nt, Ns, n_unique_ops > 0 ? n_unique_ops : 1 );
        auto h_op = Kokkos::create_mirror_view( op_table );

        if ( n_unique_ops > 0 )
        {
            auto h_A = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace{}, _A_table );
            for ( int op_idx = 0; op_idx < n_unique_ops; op_idx++ )
            {
                const auto& k = ops[op_idx];
                const double unit_w = half_width_at_depth[k.max_d];
                auto T_slice = Kokkos::subview( h_op, Kokkos::ALL,
                                                Kokkos::ALL, op_idx );
                KernelType::m2l_build_operator(
                    static_cast<scalar_type>( k.ii * unit_w ),
                    static_cast<scalar_type>( k.jj * unit_w ),
                    static_cast<scalar_type>( k.kk * unit_w ), h_A,
                    T_slice );
            }
        }
        Kokkos::deep_copy( op_table, h_op );
        _m2l_op_table = op_table;
    }

    // -----------------------------------------------------------------------
    // Stage 5: classify ops as nonshared (every pair has a non-shared
    // target) vs shared (at least one shared-target pair). Shared ops must
    // be processed inside the per-depth loop so the snapshot/allreduce
    // barrier can isolate the M2L delta to shared cells at depth d.
    // Nonshared ops can be processed once before the depth loop, with all
    // depths batched together for maximum GEMM size — see run_m2l_all().
    // -----------------------------------------------------------------------
    std::vector<int> op_pair_count( n_unique_ops, 0 );
    std::vector<unsigned char> op_has_shared( n_unique_ops, 0 );
    for ( int p = 0; p < total_pairs; p++ )
    {
        const int oi = pair_op_idx[p];
        if ( oi < 0 )
            continue;
        ++op_pair_count[oi];
        if ( pair_target_is_shared[p] )
            op_has_shared[oi] = 1u;
    }

    _m2l_nonshared_op_keys.clear();
    _m2l_shared_op_keys.clear();
    std::vector<int> op_idx_to_nonshared( n_unique_ops, -1 );
    std::vector<int> op_idx_to_shared( n_unique_ops, -1 );
    for ( int op_idx = 0; op_idx < n_unique_ops; op_idx++ )
    {
        if ( op_pair_count[op_idx] == 0 )
            continue;
        if ( op_has_shared[op_idx] )
        {
            op_idx_to_shared[op_idx] =
                static_cast<int>( _m2l_shared_op_keys.size() );
            _m2l_shared_op_keys.push_back( op_idx );
        }
        else
        {
            op_idx_to_nonshared[op_idx] =
                static_cast<int>( _m2l_nonshared_op_keys.size() );
            _m2l_nonshared_op_keys.push_back( op_idx );
        }
    }
    const int n_nonshared_ops =
        static_cast<int>( _m2l_nonshared_op_keys.size() );
    const int n_shared_ops =
        static_cast<int>( _m2l_shared_op_keys.size() );

    // -----------------------------------------------------------------------
    // Stage 6: emit op-major pair tables.
    //
    // Nonshared layout: pairs grouped by nonshared-op-index; within an op,
    // the order is whatever the entries iteration produced (any order is
    // correct because run_m2l_all packs the whole op slice in one shot).
    //
    // Shared layout: pairs grouped by shared-op-index; within an op, sorted
    // by target depth (ascending). _m2l_shared_op_depth_starts[i][d] gives
    // the slot offset *within op i* where target-depth-d pairs begin.
    //
    // Fallback layout: pairs sorted by target depth so run_m2l_at_depth(d)
    // can extract a contiguous depth-d slice.
    // -----------------------------------------------------------------------

    // Per-op pair counts in the partitioned layout.
    _m2l_nonshared_op_offsets.assign( n_nonshared_ops + 1, 0 );
    _m2l_shared_op_offsets.assign( n_shared_ops + 1, 0 );
    _m2l_shared_op_depth_starts.assign(
        n_shared_ops, std::vector<int>( _max_depth + 2, 0 ) );
    _m2l_fallback_offsets_host.assign( _max_depth + 2, 0 );

    // First counting pass.
    for ( int p = 0; p < total_pairs; p++ )
    {
        const int oi = pair_op_idx[p];
        const int d = pair_target_depth[p];
        if ( oi < 0 )
        {
            if ( d >= 0 && d <= _max_depth )
                _m2l_fallback_offsets_host[d + 1]++;
        }
        else if ( op_idx_to_nonshared[oi] >= 0 )
        {
            _m2l_nonshared_op_offsets[op_idx_to_nonshared[oi] + 1]++;
        }
        else
        {
            const int si = op_idx_to_shared[oi];
            _m2l_shared_op_offsets[si + 1]++;
            if ( d >= 0 && d <= _max_depth )
                _m2l_shared_op_depth_starts[si][d + 1]++;
        }
    }

    // Prefix sums.
    for ( int i = 0; i < n_nonshared_ops; i++ )
        _m2l_nonshared_op_offsets[i + 1] += _m2l_nonshared_op_offsets[i];
    for ( int i = 0; i < n_shared_ops; i++ )
    {
        _m2l_shared_op_offsets[i + 1] += _m2l_shared_op_offsets[i];
        for ( int d = 0; d <= _max_depth; d++ )
            _m2l_shared_op_depth_starts[i][d + 1] +=
                _m2l_shared_op_depth_starts[i][d];
    }
    for ( int d = 0; d <= _max_depth; d++ )
        _m2l_fallback_offsets_host[d + 1] +=
            _m2l_fallback_offsets_host[d];

    const int total_nonshared =
        _m2l_nonshared_op_offsets[n_nonshared_ops];
    const int total_shared = _m2l_shared_op_offsets[n_shared_ops];
    const int total_fallback = _m2l_fallback_offsets_host[_max_depth + 1];

    std::vector<int> ns_targets_h( total_nonshared );
    std::vector<int> ns_sources_h( total_nonshared );
    std::vector<int> sh_targets_h( total_shared );
    std::vector<int> sh_sources_h( total_shared );
    std::vector<int> fb_targets_h( total_fallback );
    std::vector<int> fb_sources_h( total_fallback );

    // Cursors. For shared, we track per-(op, depth) cursors so writes
    // place pairs into the right depth sub-slice within the op.
    std::vector<int> ns_cursors = _m2l_nonshared_op_offsets;
    std::vector<int> sh_op_base_cursors = _m2l_shared_op_offsets;
    std::vector<std::vector<int>> sh_depth_cursors(
        n_shared_ops, std::vector<int>( _max_depth + 2, 0 ) );
    for ( int i = 0; i < n_shared_ops; i++ )
        sh_depth_cursors[i] = _m2l_shared_op_depth_starts[i];
    std::vector<int> fb_cursors = _m2l_fallback_offsets_host;

    // Second pass: scatter.
    for ( int p = 0; p < total_pairs; p++ )
    {
        const int oi = pair_op_idx[p];
        const int d = pair_target_depth[p];
        if ( oi < 0 )
        {
            if ( d < 0 || d > _max_depth )
                continue;
            const int slot = fb_cursors[d]++;
            fb_targets_h[slot] = pair_target[p];
            fb_sources_h[slot] = pair_source[p];
        }
        else if ( op_idx_to_nonshared[oi] >= 0 )
        {
            const int ni = op_idx_to_nonshared[oi];
            const int slot = ns_cursors[ni]++;
            ns_targets_h[slot] = pair_target[p];
            ns_sources_h[slot] = pair_source[p];
        }
        else
        {
            const int si = op_idx_to_shared[oi];
            const int dd =
                ( d >= 0 && d <= _max_depth ) ? d : _max_depth;
            const int slot = sh_op_base_cursors[si] +
                             sh_depth_cursors[si][dd]++;
            sh_targets_h[slot] = pair_target[p];
            sh_sources_h[slot] = pair_source[p];
        }
    }

    // Upload to device. WithoutInitializing because the upload helper below
    // overwrites every element before any reader touches it.
    _m2l_nonshared_pair_targets = Kokkos::View<int*, memory_space>(
        Kokkos::view_alloc( Kokkos::WithoutInitializing,
                            "m2l_nonshared_targets" ),
        total_nonshared );
    _m2l_nonshared_pair_sources = Kokkos::View<int*, memory_space>(
        Kokkos::view_alloc( Kokkos::WithoutInitializing,
                            "m2l_nonshared_sources" ),
        total_nonshared );
    _m2l_shared_pair_targets = Kokkos::View<int*, memory_space>(
        Kokkos::view_alloc( Kokkos::WithoutInitializing,
                            "m2l_shared_targets" ),
        total_shared );
    _m2l_shared_pair_sources = Kokkos::View<int*, memory_space>(
        Kokkos::view_alloc( Kokkos::WithoutInitializing,
                            "m2l_shared_sources" ),
        total_shared );
    _m2l_fallback_targets = Kokkos::View<int*, memory_space>(
        Kokkos::view_alloc( Kokkos::WithoutInitializing,
                            "m2l_fallback_targets" ),
        total_fallback );
    _m2l_fallback_sources = Kokkos::View<int*, memory_space>(
        Kokkos::view_alloc( Kokkos::WithoutInitializing,
                            "m2l_fallback_sources" ),
        total_fallback );

    auto upload = []( const std::vector<int>& src,
                      Kokkos::View<int*, memory_space>& dst )
    {
        if ( src.empty() )
            return;
        auto h = Kokkos::create_mirror_view( dst );
        for ( size_t i = 0; i < src.size(); i++ )
            h( i ) = src[i];
        Kokkos::deep_copy( dst, h );
    };
    upload( ns_targets_h, _m2l_nonshared_pair_targets );
    upload( ns_sources_h, _m2l_nonshared_pair_sources );
    upload( sh_targets_h, _m2l_shared_pair_targets );
    upload( sh_sources_h, _m2l_shared_pair_sources );
    upload( fb_targets_h, _m2l_fallback_targets );
    upload( fb_sources_h, _m2l_fallback_sources );

    // Per-depth fallback counts for total_fallback_pair_count().
    _m2l_fallback_count_per_active_depth.assign( _max_depth + 1, 0 );
    for ( int d = 0; d <= _max_depth; d++ )
        _m2l_fallback_count_per_active_depth[d] =
            static_cast<long long>( _m2l_fallback_offsets_host[d + 1] -
                                    _m2l_fallback_offsets_host[d] );

    // -----------------------------------------------------------------------
    // Stage 7: size and allocate the packed-multipole / packed-local
    // scratch buffers. We need at least:
    //   - total_nonshared * NComps columns (single all-at-once pack used
    //     by run_m2l_all)
    //   - max over (shared op i, depth d) of pair_count(i, d) * NComps
    //     (per-(op, depth) pack used inside run_m2l_at_depth)
    // -----------------------------------------------------------------------
    int max_shared_op_depth_pairs = 0;
    for ( int i = 0; i < n_shared_ops; i++ )
    {
        for ( int d = 0; d <= _max_depth; d++ )
        {
            const int n =
                _m2l_shared_op_depth_starts[i][d + 1] -
                _m2l_shared_op_depth_starts[i][d];
            if ( n > max_shared_op_depth_pairs )
                max_shared_op_depth_pairs = n;
        }
    }
    const int max_pairs_per_phase =
        std::max( total_nonshared, max_shared_op_depth_pairs );

    const int Ns = KernelType::m2l_num_src_coeffs;
    const int Nt = KernelType::num_coeffs_per_cell;
    const int n_cols = max_pairs_per_phase * NComps;
    // M_packed is fully populated by the M2L pack kernel before any GEMM
    // reader; L_packed is the GEMM `C` output with beta = 0 on every
    // backend (cuBLAS, hipBLAS, KokkosKernels — see Canopy_BatchedGemm.hpp).
    // Both are safe to allocate WithoutInitializing, eliminating the
    // complex<double> zero-fill that dominated the kernel summary.
    _m2l_M_packed =
        Kokkos::View<complex_type**, Kokkos::LayoutLeft, memory_space>(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "m2l_M_packed" ),
            Ns, n_cols > 0 ? n_cols : 1 );
    _m2l_L_packed =
        Kokkos::View<complex_type**, Kokkos::LayoutLeft, memory_space>(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "m2l_L_packed" ),
            Nt, n_cols > 0 ? n_cols : 1 );

    _interaction_list_dirty = false;
    _interaction_list_build_count++;
}

template <class MemorySpace, class ExecutionSpace, class KernelType>
void DownwardSweep<MemorySpace, ExecutionSpace, KernelType>::
    exchange_multipoles_for_m2l(
        const typename UpwardSweep<MemorySpace, ExecutionSpace,
                                   KernelType>::coeff_view_type& multipoles,
        const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan )
{
    CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_M2L_COMM );
    // Pre-sweep bulk exchange: we send each of our cells whose multipole
    // is in another rank's interaction list, and receive remote source
    // multipoles we need. This overwrites multipoles at the remote
    // cell indices in our view.
    const auto& m2l = comm_plan.m2l_plan();

    const int per_cell_complex = coeffs_per_cell * NComps;
    const int per_cell_real = 2 * per_cell_complex;
    MPI_Datatype mpi_scalar =
        ( sizeof( scalar_type ) == 8 ) ? MPI_DOUBLE : MPI_FLOAT;

    auto h_mults =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, multipoles );

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
        MPI_Irecv( reinterpret_cast<scalar_type*>( recv_bufs[i].data() ),
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
        MPI_Isend( reinterpret_cast<scalar_type*>( send_bufs[i].data() ),
                   per_cell_real, mpi_scalar, m2l.sends[i].remote_rank, tag,
                   _comm, &send_reqs[i] );
    }

    if ( !recv_reqs.empty() )
        MPI_Waitall( recv_reqs.size(), recv_reqs.data(), MPI_STATUSES_IGNORE );
    if ( !send_reqs.empty() )
        MPI_Waitall( send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE );

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

// -------------------------------------------------------------------------
// run_m2l_all: process every M2L pair whose target is non-shared. Run
// once per solve, before the per-depth loop. Pairs are pre-grouped by
// op_idx in _m2l_nonshared_pair_*; each op gets a single GEMM whose `n`
// dimension spans all that op's pairs across every depth — the largest
// possible batch the snapshot/allreduce barrier permits.
// -------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace, class KernelType>
void DownwardSweep<MemorySpace, ExecutionSpace, KernelType>::run_m2l_all()
{
    CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_M2L_KERNEL );

    const int n_nonshared_ops =
        static_cast<int>( _m2l_nonshared_op_keys.size() );
    if ( n_nonshared_ops == 0 )
        return;
    const int n_pairs =
        static_cast<int>( _m2l_nonshared_pair_targets.extent( 0 ) );
    if ( n_pairs == 0 )
        return;

    constexpr int Ns = KernelType::m2l_num_src_coeffs;
    constexpr int Nt = KernelType::num_coeffs_per_cell;
    constexpr int P_local = KernelType::max_order;

    // (a) Pack source multipoles for every nonshared pair into M_packed,
    //     column index = pair_slot * NComps + c.
    {
        auto pair_sources = _m2l_nonshared_pair_sources;
        auto multipoles = _m2l_multipoles_view;
        auto M_packed = _m2l_M_packed;
        Kokkos::parallel_for(
            "M2L_pack_nonshared",
            Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<3>>(
                { 0, 0, 0 }, { n_pairs, P_local + 1, NComps } ),
            KOKKOS_LAMBDA( int p, int n, int c ) {
                const int src_cell = pair_sources( p );
                for ( int m = -n; m <= n; ++m )
                {
                    const int src_idx = n * n + n + m;
                    const int abs_m = ( m < 0 ) ? -m : m;
                    const int storage_idx = n * ( n + 1 ) / 2 + abs_m;
                    const complex_type stored =
                        multipoles( src_cell, storage_idx, c );
                    const complex_type val =
                        ( m >= 0 )
                            ? stored
                            : complex_type( stored.real(),
                                            -stored.imag() );
                    M_packed( src_idx, p * NComps + c ) = val;
                }
            } );
    }

    // (b) One GEMM per nonshared op on its column slice of M_packed.
    static_assert(
        detail::M2LBatchedGemm<execution_space, scalar_type>::available,
        "Canopy DownwardSweep: no batched-GEMM backend is available for "
        "this execution space. Build with cuBLAS (CUDA), hipBLAS (HIP), "
        "or KokkosKernels — see CMakeLists.txt batched-gemm M2L block." );

    for ( int oi = 0; oi < n_nonshared_ops; ++oi )
    {
        const int op_idx = _m2l_nonshared_op_keys[oi];
        const int slot_lo = _m2l_nonshared_op_offsets[oi];
        const int slot_hi = _m2l_nonshared_op_offsets[oi + 1];
        const int n_op = slot_hi - slot_lo;
        if ( n_op == 0 )
            continue;

        auto T_slice = Kokkos::subview( _m2l_op_table, Kokkos::ALL,
                                        Kokkos::ALL, op_idx );
        const int col_off = slot_lo * NComps;
        const int n_cols = n_op * NComps;
        _m2l_gemm.gemm_NN(
            Nt, n_cols, Ns, T_slice.data(),
            static_cast<int>( T_slice.stride_1() ),
            _m2l_M_packed.data() +
                col_off * static_cast<std::ptrdiff_t>(
                              _m2l_M_packed.stride_1() ),
            static_cast<int>( _m2l_M_packed.stride_1() ),
            _m2l_L_packed.data() +
                col_off * static_cast<std::ptrdiff_t>(
                              _m2l_L_packed.stride_1() ),
            static_cast<int>( _m2l_L_packed.stride_1() ) );
    }

    // (c) Atomic scatter L_packed back into _locals.
    {
        auto pair_targets = _m2l_nonshared_pair_targets;
        auto L_packed = _m2l_L_packed;
        auto locals = _locals;
        Kokkos::parallel_for(
            "M2L_scatter_nonshared",
            Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<3>>(
                { 0, 0, 0 }, { n_pairs, Nt, NComps } ),
            KOKKOS_LAMBDA( int p, int out_idx, int c ) {
                const int tgt_cell = pair_targets( p );
                const complex_type val =
                    L_packed( out_idx, p * NComps + c );
                Kokkos::atomic_add( &locals( tgt_cell, out_idx, c ),
                                    val );
            } );
    }

    Kokkos::fence();
}

// -------------------------------------------------------------------------
// run_m2l_at_depth: process M2L contributions whose target sits at the
// given depth and is shared (i.e., needs the snapshot/allreduce barrier),
// plus the per-depth fallback slice. Nonshared targets are handled
// separately by run_m2l_all().
//
// Shared-op pairs are sorted by target depth within each op, so the
// depth-d sub-slice of each op is a contiguous range. We pack/GEMM/scatter
// per shared op (typically a small number, since shared cells live in the
// top of the tree).
// -------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace, class KernelType>
void DownwardSweep<MemorySpace, ExecutionSpace, KernelType>::run_m2l_at_depth(
    int depth )
{
    CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_M2L_KERNEL );
    if ( depth < 0 || depth > _max_depth )
        return;

    constexpr int Ns = KernelType::m2l_num_src_coeffs;
    constexpr int Nt = KernelType::num_coeffs_per_cell;
    constexpr int P_local = KernelType::max_order;

    const int n_shared_ops =
        static_cast<int>( _m2l_shared_op_keys.size() );

    for ( int si = 0; si < n_shared_ops; ++si )
    {
        const int op_idx = _m2l_shared_op_keys[si];
        const int op_base = _m2l_shared_op_offsets[si];
        const int sub_lo =
            op_base + _m2l_shared_op_depth_starts[si][depth];
        const int sub_hi =
            op_base + _m2l_shared_op_depth_starts[si][depth + 1];
        const int n_b = sub_hi - sub_lo;
        if ( n_b == 0 )
            continue;

        auto pair_sources = _m2l_shared_pair_sources;
        auto pair_targets = _m2l_shared_pair_targets;
        auto multipoles = _m2l_multipoles_view;
        auto M_packed = _m2l_M_packed;
        auto L_packed = _m2l_L_packed;
        auto locals = _locals;

        // Pack n_b pairs into columns [0, n_b * NComps) of M_packed.
        Kokkos::parallel_for(
            "M2L_pack_shared",
            Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<3>>(
                { 0, 0, 0 }, { n_b, P_local + 1, NComps } ),
            KOKKOS_LAMBDA( int p, int n, int c ) {
                const int src_cell = pair_sources( sub_lo + p );
                for ( int m = -n; m <= n; ++m )
                {
                    const int src_idx = n * n + n + m;
                    const int abs_m = ( m < 0 ) ? -m : m;
                    const int storage_idx = n * ( n + 1 ) / 2 + abs_m;
                    const complex_type stored =
                        multipoles( src_cell, storage_idx, c );
                    const complex_type val =
                        ( m >= 0 )
                            ? stored
                            : complex_type( stored.real(),
                                            -stored.imag() );
                    M_packed( src_idx, p * NComps + c ) = val;
                }
            } );

        auto T_slice = Kokkos::subview( _m2l_op_table, Kokkos::ALL,
                                        Kokkos::ALL, op_idx );
        const int n_cols = n_b * NComps;
        _m2l_gemm.gemm_NN(
            Nt, n_cols, Ns, T_slice.data(),
            static_cast<int>( T_slice.stride_1() ),
            _m2l_M_packed.data(),
            static_cast<int>( _m2l_M_packed.stride_1() ),
            _m2l_L_packed.data(),
            static_cast<int>( _m2l_L_packed.stride_1() ) );

        Kokkos::parallel_for(
            "M2L_scatter_shared",
            Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<3>>(
                { 0, 0, 0 }, { n_b, Nt, NComps } ),
            KOKKOS_LAMBDA( int p, int out_idx, int c ) {
                const int tgt_cell = pair_targets( sub_lo + p );
                const complex_type val =
                    L_packed( out_idx, p * NComps + c );
                Kokkos::atomic_add( &locals( tgt_cell, out_idx, c ),
                                    val );
            } );
    }

    // Per-pair fallback for guard-rail-violating pairs at this depth.
    run_m2l_fallback_at_depth( depth );

    Kokkos::fence();
}

// -------------------------------------------------------------------------
// Per-pair m2l_translate fallback for guard-rail-violating pairs at one
// depth. Reads the (target, source) slice
//   [_m2l_fallback_offsets_host[depth], _m2l_fallback_offsets_host[depth+1])
// and applies the existing m2l_translate kernel one team per pair. In
// healthy MAC traversals this list is empty.
// -------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace, class KernelType>
void DownwardSweep<MemorySpace, ExecutionSpace, KernelType>::
    run_m2l_fallback_at_depth( int depth )
{
    if ( depth < 0 || depth > _max_depth ||
         _m2l_fallback_offsets_host.empty() )
        return;
    const int fb_begin = _m2l_fallback_offsets_host[depth];
    const int fb_end = _m2l_fallback_offsets_host[depth + 1];
    const int n_fb = fb_end - fb_begin;
    if ( n_fb == 0 )
        return;

    auto device_cells = _device_cells;
    auto fb_targets = _m2l_fallback_targets;
    auto fb_sources = _m2l_fallback_sources;
    auto locals = _locals;
    auto A_table = _A_table;
    auto multipoles = _m2l_multipoles_view;

    using team_policy = Kokkos::TeamPolicy<execution_space>;
    using member_t = typename team_policy::member_type;
    team_policy policy( n_fb, Kokkos::AUTO );

    Kokkos::parallel_for(
        "M2L_fallback", policy, KOKKOS_LAMBDA( const member_t& team ) {
            const int slot = fb_begin + team.league_rank();
            const int target_cell = fb_targets( slot );
            const int source_cell = fb_sources( slot );
            const auto& target_ci = device_cells( target_cell );
            const auto& src_ci = device_cells( source_cell );
            const scalar_type dx = src_ci.center[0] - target_ci.center[0];
            const scalar_type dy = src_ci.center[1] - target_ci.center[1];
            const scalar_type dz = src_ci.center[2] - target_ci.center[2];

            auto L_target = Kokkos::subview( locals, target_cell, Kokkos::ALL,
                                             Kokkos::ALL );
            KernelType::m2l_translate( team, multipoles, source_cell, dx, dy,
                                       dz, A_table, L_target );
        } );
}

template <class MemorySpace, class ExecutionSpace, class KernelType>
void DownwardSweep<MemorySpace, ExecutionSpace, KernelType>::run_l2l_at_depth(
    int depth )
{
    CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_L2L_KERNEL );
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
        "L2L", policy, KOKKOS_LAMBDA( const team_member_type& team ) {
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

                const scalar_type dx = ccell.center[0] - parent_ci.center[0];
                const scalar_type dy = ccell.center[1] - parent_ci.center[1];
                const scalar_type dz = ccell.center[2] - parent_ci.center[2];

                auto L_child =
                    Kokkos::subview( locals, ci, Kokkos::ALL, Kokkos::ALL );
                KernelType::l2l_translate( team, locals, parent_cell, dx, dy,
                                           dz, A_table, L_child );
            }
        } );

    Kokkos::fence();
}

template <class MemorySpace, class ExecutionSpace, class KernelType>
void DownwardSweep<MemorySpace, ExecutionSpace, KernelType>::
    snapshot_shared_locals_at_depth(
        int depth,
        const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan )
{
    // Capture the value of _locals at all shared cells at this depth
    // BEFORE M2L runs. After M2L, allreduce_shared_locals_at_depth will
    // use the snapshot to isolate the M2L contribution (which differs
    // between ranks and must be summed) from the L2L-inherited part
    // (which is identical on all ranks and must NOT be summed).
    const auto& m2m = comm_plan.m2m_plan();
    _shared_snapshot_indices.clear();

    auto h_dc = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{},
                                                     _device_cells );
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

    auto h_locals =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, _locals );
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
        int depth,
        const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan )
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

    auto h_locals =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, _locals );

    // sendbuf = current - snapshot (M2L delta on this rank)
    for ( int i = 0; i < nshared; i++ )
    {
        const int cidx = _shared_snapshot_indices[i];
        int idx = 0;
        for ( int ci = 0; ci < coeffs_per_cell; ci++ )
            for ( int c = 0; c < NComps; c++ )
            {
                const int k = i * per_cell_complex + ( idx++ );
                sendbuf[k] = h_locals( cidx, ci, c ) - _shared_snapshot_buf[k];
            }
    }

    MPI_Datatype mpi_scalar =
        ( sizeof( scalar_type ) == 8 ) ? MPI_DOUBLE : MPI_FLOAT;
    {
        CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_M2L_ALLREDUCE );
        MPI_Allreduce( reinterpret_cast<scalar_type*>( sendbuf.data() ),
                       reinterpret_cast<scalar_type*>( recvbuf.data() ),
                       2 * total_complex, mpi_scalar, MPI_SUM, _comm );
    }

    // _locals[shared] = snapshot + summed delta
    for ( int i = 0; i < nshared; i++ )
    {
        const int cidx = _shared_snapshot_indices[i];
        int idx = 0;
        for ( int ci = 0; ci < coeffs_per_cell; ci++ )
            for ( int c = 0; c < NComps; c++ )
            {
                const int k = i * per_cell_complex + ( idx++ );
                h_locals( cidx, ci, c ) = _shared_snapshot_buf[k] + recvbuf[k];
            }
    }

    Kokkos::deep_copy( _locals, h_locals );
}

template <class MemorySpace, class ExecutionSpace, class KernelType>
void DownwardSweep<MemorySpace, ExecutionSpace, KernelType>::
    exchange_locals_after_l2l_at_depth(
        int depth,
        const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan )
{
    CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_L2L_COMM );
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

    auto h_dc = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{},
                                                     _device_cells );

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

    auto h_locals =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, _locals );

    std::vector<MPI_Request> recv_reqs( recvs.size() );
    std::vector<std::vector<complex_type>> recv_bufs( recvs.size() );
    for ( size_t i = 0; i < recvs.size(); i++ )
    {
        recv_bufs[i].resize( per_cell_complex );
        const MortonKey key = h_dc( recvs[i].child_cell_idx ).key;
        int tag = static_cast<int>( key & 0x7fffffff );
        MPI_Irecv( reinterpret_cast<scalar_type*>( recv_bufs[i].data() ),
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
        MPI_Isend( reinterpret_cast<scalar_type*>( send_bufs[i].data() ),
                   per_cell_real, mpi_scalar, sends[i].remote_rank, tag, _comm,
                   &send_reqs[i] );
    }

    if ( !recv_reqs.empty() )
        MPI_Waitall( recv_reqs.size(), recv_reqs.data(), MPI_STATUSES_IGNORE );
    if ( !send_reqs.empty() )
        MPI_Waitall( send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE );

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
    CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_L2P );
    auto locals = _locals;
    auto device_cells = _device_cells;
    auto particle_cell_idx = _particle_cell_idx;
    const int N = _num_local_particles;

    Kokkos::parallel_for(
        "L2P", Kokkos::RangePolicy<execution_space>( 0, N ),
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
            KernelType::l2p_evaluate( locals, cidx, dx, dy, dz, phi, writer,
                                      compute_gradient );

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
    const typename UpwardSweep<MemorySpace, ExecutionSpace,
                               KernelType>::coeff_view_type& multipoles,
    const PositionType& particle_positions,
    const potential_view_type& potential_out,
    const gradient_view_type& gradient_out, bool compute_gradient,
    const CommunicationPlan<MemorySpace, ExecutionSpace>& comm_plan )
{
    CANOPY_RESET_TIMERS();
    {
        CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_DOWNWARD_TOTAL );
        // Zero local coefficients
        {
            CANOPY_SCOPED_TIMER_DETAILED(
                Canopy::Profiling::TIMER_DN_ZERO_LOCALS );
            Kokkos::deep_copy( _locals, complex_type( 0.0, 0.0 ) );
            Kokkos::fence();
        }

        // Stash the multipoles for the M2L kernel
        _m2l_multipoles_view = multipoles;

        // Build device-side interaction list if dirty. The function
        // early-returns when clean, so the timer just measures the
        // dirty-flag check on the cached path.
        {
            CANOPY_SCOPED_TIMER_DETAILED(
                Canopy::Profiling::TIMER_DN_BUILD_ILIST );
            build_interaction_list_device( comm_plan );
        }

        // Pre-sweep: exchange remote multipoles needed for M2L
        exchange_multipoles_for_m2l( multipoles, comm_plan );

        // Process all M2L pairs whose target is non-shared in one batch,
        // pooling across depths for maximum GEMM size. Safe because the
        // snapshot/allreduce barrier in the per-depth loop only operates
        // on shared cells, which run_m2l_all does not touch.
        run_m2l_all();

        // Layer-by-layer: snapshot shared, M2L (shared targets only),
        // allreduce shared M2L delta, L2L, exchange children.
        for ( int d = 0; d <= _max_depth; d++ )
        {
            {
                CANOPY_SCOPED_TIMER_DETAILED(
                    Canopy::Profiling::TIMER_DN_PRE_M2L );
                snapshot_shared_locals_at_depth( d, comm_plan );
            }
            {
                CANOPY_SCOPED_TIMER_DETAILED(
                    Canopy::Profiling::TIMER_DN_M2L_CALL );
                run_m2l_at_depth( d );
            }
            {
                CANOPY_SCOPED_TIMER_DETAILED(
                    Canopy::Profiling::TIMER_DN_POST_M2L );
                allreduce_shared_locals_at_depth( d, comm_plan );
                run_l2l_at_depth( d );
                if ( d < _max_depth )
                    exchange_locals_after_l2l_at_depth( d, comm_plan );
            }
        }

        // L2P: evaluate local expansion at each particle
        run_l2p( particle_positions, potential_out, gradient_out,
                 compute_gradient );
    }
    CANOPY_PRINT_DOWNWARD_TIMERS( _comm );
}

} // namespace Canopy

#endif // CANOPY_DOWNWARD_SWEEP_HPP
