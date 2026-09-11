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
#include "Canopy_MpiCoalescedExchange.hpp"
#include "Canopy_Profiling.hpp"
#include "Canopy_LaplaceKernel.hpp"
#include "Canopy_SphericalCoefficients.hpp"
#include "Canopy_TreeBuilder.hpp"
#include "Canopy_TreePartitioner.hpp"
#include "Canopy_UpwardSweep.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Canopy
{

// Tunable: initial bucket reservation for each per-thread M2L key->op map
// used by the sharded S3 classify-pairs pass in
// DownwardSweep::build_interaction_list_device. Sized to mostly absorb the
// expected per-shard distinct-key count (globally ~16 k under MAC=0.5,
// shared across ~24 threads) without forcing a rehash, while keeping per-
// thread memory bounded. Promote to a CMake option if tuning becomes needed.
static constexpr int S3_PER_THREAD_KEYMAP_RESERVE = 4096;

// Decode a MortonKey into (depth, ix, iy, iz) where (ix, iy, iz) are the
// per-axis integer cell coordinates at that depth, in [0, 2^depth). The key
// layout is leading-1 sentinel at bit 3*depth, followed by `depth` 3-bit
// octant groups (bit 0 = x, bit 1 = y, bit 2 = z). See Canopy_TreeBuilder.hpp.
//
// Used by S3 classify to compute (dd, ii, jj, kk) directly from MortonKeys,
// avoiding the per-pair h_dc_for_filter gather on cell centers.
static inline void
decode_morton( MortonKey k, int& d, int& ix, int& iy, int& iz )
{
    d = key_depth( k );
    ix = 0;
    iy = 0;
    iz = 0;
    // Octant for level l (root is l=0; the chosen octant at level l lives in
    // bits [3*(d-l), 3*(d-l)+2] of k). Iterate from the deepest level (l=d,
    // bit positions 0..2) upward, accumulating axis bits as the integer index.
    for ( int l = 1; l <= d; ++l )
    {
        const int shift = 3 * ( d - l );
        const int oct = static_cast<int>( ( k >> shift ) & 0x7 );
        const int weight = 1 << ( d - l );
        ix += ( oct & 1 ) ? weight : 0;
        iy += ( oct & 2 ) ? weight : 0;
        iz += ( oct & 4 ) ? weight : 0;
    }
}

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

    // The basis's coefficient contract. This sweep stores, exchanges and
    // reduces coefficients without knowing anything about them beyond these
    // three: coeff_type is the storage element, and MPI is handed
    // scalars_per_coeff component_scalar_type per coefficient.
    using coeff_type = typename KernelType::coeff_type;
    using component_scalar_type = typename KernelType::component_scalar_type;
    static constexpr int scalars_per_coeff = KernelType::scalars_per_coeff;

    static_assert( sizeof( coeff_type ) ==
                       scalars_per_coeff * sizeof( component_scalar_type ),
                   "DownwardSweep: the basis's coeff_type is not "
                   "scalars_per_coeff contiguous component_scalar_type, so "
                   "the shared-cell Allreduce would transfer the wrong byte "
                   "count" );

    // coalesced_view_exchange is handed a View and never a basis, so it
    // recovers the same two facts from detail::coeff_traits. This is the one
    // place a basis and that function meet, so it is where the two sources
    // are checked against each other.
    static_assert(
        std::is_same<
            typename detail::coeff_traits<coeff_type>::component_scalar_type,
            component_scalar_type>::value &&
            detail::coeff_traits<coeff_type>::scalars_per_coeff ==
                scalars_per_coeff,
        "DownwardSweep: the basis's coefficient traits disagree with "
        "detail::coeff_traits for its coeff_type; the M2L and L2L exchanges "
        "and the shared-cell Allreduce would pack differently" );

    static constexpr int coeffs_per_cell = KernelType::num_coeffs_per_cell;
    static constexpr int NComps = KernelType::num_components;

    // Local coefficient storage. LayoutRight so a single thread can scan
    // within-cell coefficients contiguously and a warp writing different
    // out_idx values for one target writes consecutive bytes.
    using coeff_view_type =
        Kokkos::View<coeff_type***, Kokkos::LayoutRight, memory_space>;

    // Potential output: (num_particles, NComps)
    using potential_view_type =
        Kokkos::View<scalar_type* [NComps], memory_space>;

    // Gradient output: (num_particles, NComps, 3)
    using gradient_view_type =
        Kokkos::View<scalar_type* [NComps][3], memory_space>;

    // The basis's auxiliary tables — precomputed, order-dependent data the
    // basis's own operators need. Opaque here: this sweep borrows one from
    // the UpwardSweep at setup(), hands it to m2l_translate and
    // l2l_translate, and never looks inside. May be an empty struct.
    using aux_tables_type =
        typename KernelType::template aux_tables_type<memory_space>;

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

    // Persistent staging buffers for the M2L (multipole) and L2L (locals)
    // exchanges. Reused every solve so the CXI NIC registration cache stays
    // bounded. Shared between the two calls — they run sequentially within a
    // solve (each coalesced_view_exchange completes its MPI_Waitall + fence
    // before returning), so reusing one region is safe.
    mutable detail::CoalescedExchangeBuffers<coeff_type, memory_space>
        _exch_bufs;

    // References borrowed from UpwardSweep — only valid while the
    // upward sweep is alive and setup() has been called.
    cell_view_type _device_cells;
    const std::unordered_map<MortonKey, int>* _key_to_cell_idx;
    using children_view_type = typename UpwardSweep<
        MemorySpace, ExecutionSpace, KernelType>::children_view_type;
    const children_view_type* _d_cell_children;
    particle_cell_idx_view_type _particle_cell_idx;
    aux_tables_type _aux;

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
    // The key is built in full — max_d is always emitted, with no branch —
    // and then reduced by KernelType::canonicalize_key before it is hashed.
    // That call is what decides whether the level survives into the key: a
    // basis whose operators are scale-normalized (the solid-harmonic one)
    // zeroes max_d there and collapses every depth onto one column, while a
    // basis carrying physical operators returns the key unchanged and gets
    // one column per (level, offset). The sweep does not know or care which;
    // it canonicalizes once, at construction, and everything downstream —
    // the per-thread key maps, the global key_to_op, the realized key list
    // and the operator table's column order — sees only canonical keys.
    //
    // Range guards (|dd| <= KernelType::m2l_key_dd_max, |offset| <= 32) catch
    // pathological pairs and route them to the per-pair m2l_translate
    // fallback. Healthy MAC traversals never trip these. The |dd| bound is
    // the basis's, not the sweep's: what makes a large depth difference
    // unsafe is the basis's own width normalization, so the rationale and the
    // FP32 valve live on KernelType::m2l_key_dd_max.
    static constexpr int M2L_KEY_DD_MAX = KernelType::m2l_key_dd_max;
    static constexpr int M2L_KEY_OFFSET_MAX = 32;
    static constexpr int M2L_OP_COUNT_CAP = 32768;

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

    // The M2L operator set. Its type, shape and layout belong to the basis
    // (see KernelType::m2l_operators_type, which documents the
    // solid-harmonic one); the sweep stores it, hands it to the basis's
    // three M2L stages and never indexes it. The only thing the sweep says
    // about an operator is the integer op_idx its CSR carries.
    using m2l_operators_type =
        typename KernelType::template m2l_operators_type<memory_space>;

    // The operator set built by the last build_interaction_list_device().
    // Entry op_idx corresponds to _m2l_realized_keys[op_idx].
    m2l_operators_type _m2l_op_table;

    // The realized key set, in operator-table column order: entry i is the
    // key of _m2l_op_table(:, :, i). Retained past the end of
    // build_interaction_list_device purely as a read-only diagnostic
    // surface (see m2l_realized_keys()); nothing in the solve reads it.
    std::vector<M2LKey> _m2l_realized_keys;

    // Tier 2 fused-kernel layout: target-major CSRs partitioned by target
    // sharedness. One Kokkos team per target walks its source slice and
    // accumulates T(:, :, op_idx) @ M(source) into a scratch local, then
    // writes it into _locals(target, :, :) once. No per-target atomics.
    //
    //   nonshared: targets with owner_rank != OWNER_SHARED. Run once per
    //              solve in run_m2l_all() before the per-depth loop.
    //
    //   shared:    targets with owner_rank == OWNER_SHARED (rank 0 only).
    //              Targets are sorted by depth; _m2l_sh_csr_depth_offsets[d]
    //              gives the first target index at depth >= d so that
    //              run_m2l_at_depth(d) runs on the contiguous depth-d slice,
    //              preserving snapshot/allreduce semantics.
    //
    // Pairs with op_idx == -1 (range-guard violations) are kept in the CSR
    // and skipped by the fused kernel; they are processed by the existing
    // per-depth fallback path (_m2l_fallback_*) which has its own tables.
    Kokkos::View<int*, memory_space> _m2l_ns_csr_targets;
    Kokkos::View<int*, memory_space> _m2l_ns_csr_offsets; // size N_ns_t + 1
    Kokkos::View<int*, memory_space> _m2l_ns_csr_sources;
    Kokkos::View<int*, memory_space> _m2l_ns_csr_op_idx;

    Kokkos::View<int*, memory_space> _m2l_sh_csr_targets;
    Kokkos::View<int*, memory_space> _m2l_sh_csr_offsets; // size N_sh_t + 1
    Kokkos::View<int*, memory_space> _m2l_sh_csr_sources;
    Kokkos::View<int*, memory_space> _m2l_sh_csr_op_idx;
    std::vector<int> _m2l_sh_csr_depth_offsets;           // size max_depth + 2

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
    std::vector<coeff_type> _shared_snapshot_buf;

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
    // Total M2L pair count carried by this rank. Pairs with op_idx == -1
    // appear in both the per-target CSR (skipped by the fused kernel) and
    // the fallback table — they represent a single physical pair, so we
    // count CSR sources only and do not add the fallback total again.
    long long total_m2l_pair_count() const
    {
        return static_cast<long long>(
                   _m2l_ns_csr_sources.extent( 0 ) ) +
               static_cast<long long>( _m2l_sh_csr_sources.extent( 0 ) );
    }

    // -----------------------------------------------------------------------
    // Golden-harness diagnostic surface.
    //
    // Read-only views of internal state, exposed so a bit-for-bit test can
    // compare the artifacts of a solve across a refactor. Additive and
    // side-effect free. Like Solver::downward(), this is a diagnostic
    // surface and not part of the supported runtime API.
    // -----------------------------------------------------------------------

    // Type of the hashed M2L operator table: whatever the basis says an
    // M2L operator set is (see the _m2l_op_table declaration). For the
    // solid-harmonic basis, (Nt, Ns, n_unique_ops) and LayoutLeft.
    using m2l_op_table_view_type = m2l_operators_type;

    // The canonicalized M2L key, {dd, ii, jj, kk} — a signed depth
    // difference and an integer offset in units of the smaller cell width.
    using m2l_key_type = M2LKey;

    // The hashed M2L operator table built by the last
    // build_interaction_list_device(). Column op_idx holds the operator for
    // m2l_realized_keys()[op_idx]. Default-constructed (rank-0 extents)
    // before the first build.
    const m2l_op_table_view_type& m2l_op_table() const { return _m2l_op_table; }

    // The realized M2L key set from the last build_interaction_list_device(),
    // in operator-table column order. Empty before the first build.
    const std::vector<m2l_key_type>& m2l_realized_keys() const
    {
        return _m2l_realized_keys;
    }

    // Number of unique M2L operators realized by the last
    // build_interaction_list_device(); equals m2l_realized_keys().size(), and
    // equals m2l_op_table().extent( 2 ) whenever that count is non-zero.
    int m2l_n_unique_ops() const
    {
        return static_cast<int>( _m2l_realized_keys.size() );
    }

    // The basis's auxiliary tables, borrowed from the UpwardSweep at
    // setup(). Empty before setup(). Shared code must not name a member of
    // this struct — a basis whose aux_tables_type is empty has none.
    const aux_tables_type& aux() const { return _aux; }

    // Per-pair fallback for out-of-range pairs at depth `depth`.
    void run_m2l_fallback_at_depth( int depth );

    // Tier 2 fused team-per-target M2L kernel. For each target in
    // [target_index_lo, target_index_hi) of `csr_targets`, walks that
    // target's source slice in csr_sources/csr_op_idx and drives the
    // basis's three M2L stages over it, so that _locals(target, :, :)
    // is written once. Each target is owned by exactly one team, so no
    // atomics on _locals. Pairs with op_idx == -1 are skipped (handled
    // by the fallback path). Public because CUDA forbids extended
    // __host__ __device__ lambdas inside private member functions.
    void run_m2l_fused(
        const Kokkos::View<int*, memory_space>& csr_targets,
        const Kokkos::View<int*, memory_space>& csr_offsets,
        const Kokkos::View<int*, memory_space>& csr_sources,
        const Kokkos::View<int*, memory_space>& csr_op_idx,
        int target_index_lo, int target_index_hi );


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
    _d_cell_children = &upward_sweep.cell_children();
    _aux = upward_sweep.aux();
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
    _m2l_op_table = m2l_operators_type();
    _m2l_ns_csr_targets = Kokkos::View<int*, memory_space>();
    _m2l_ns_csr_offsets = Kokkos::View<int*, memory_space>();
    _m2l_ns_csr_sources = Kokkos::View<int*, memory_space>();
    _m2l_ns_csr_op_idx = Kokkos::View<int*, memory_space>();
    _m2l_sh_csr_targets = Kokkos::View<int*, memory_space>();
    _m2l_sh_csr_offsets = Kokkos::View<int*, memory_space>();
    _m2l_sh_csr_sources = Kokkos::View<int*, memory_space>();
    _m2l_sh_csr_op_idx = Kokkos::View<int*, memory_space>();
    _m2l_sh_csr_depth_offsets.clear();
    _m2l_fallback_targets = Kokkos::View<int*, memory_space>();
    _m2l_fallback_sources = Kokkos::View<int*, memory_space>();
    _m2l_fallback_offsets_host.clear();
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
        MortonKey target_key;
        std::vector<int> sources;
        std::vector<MortonKey> source_keys;
    };
    std::vector<TargetEntry> entries;
    entries.reserve( ilists.size() );

    {
    CANOPY_SCOPED_TIMER_DETAILED(
        Canopy::Profiling::TIMER_ILIST_S1_COLLECT_ENTRIES );
    for ( const auto& [target_key, sources] : ilists )
    {
        auto it = _key_to_cell_idx->find( target_key );
        if ( it == _key_to_cell_idx->end() )
            continue;
        const int target_idx = it->second;
        const auto& tci = h_dc_for_filter( target_idx );
        if ( tci.owner_rank == OWNER_SHARED && _rank != 0 )
            continue;

        TargetEntry e;
        e.target_idx = target_idx;
        e.depth = tci.depth;
        e.target_key = target_key;
        e.sources.reserve( sources.size() );
        e.source_keys.reserve( sources.size() );
        // sources is vector<pair<MortonKey, int>> — the second element is
        // the cell index in `cells`, populated by CP when it emitted the
        // pair. No per-source hash lookup needed. Capture src_key in parallel
        // so S3 can decode (d_s, ix_s, iy_s, iz_s) without gathering sci.
        for ( const auto& [src_key, src_idx] : sources )
        {
            e.sources.push_back( src_idx );
            e.source_keys.push_back( src_key );
        }
        entries.push_back( std::move( e ) );
    }
    } // S1

    {
    CANOPY_SCOPED_TIMER_DETAILED(
        Canopy::Profiling::TIMER_ILIST_S2_SORT_BY_DEPTH );
    std::sort( entries.begin(), entries.end(),
               []( const TargetEntry& a, const TargetEntry& b )
               {
                   if ( a.depth != b.depth )
                       return a.depth < b.depth;
                   return a.target_idx < b.target_idx;
               } );
    } // S2

    // Compute total pair count for the classification pass that follows.
    int total_pairs_count = 0;
    for ( const auto& e : entries )
        total_pairs_count += static_cast<int>( e.sources.size() );

    // -----------------------------------------------------------------------
    // Stage 3: hash-map every (target, source) pair to a unique M2L operator
    // key (max_d, dd, ii, jj, kk), reduced by KernelType::canonicalize_key.
    // Pairs sharing a canonical key share an operator T.
    //
    // Whether pairs at different depths can share one is the basis's call,
    // not the sweep's: the key is built with max_d always present, and
    // canonicalize_key either keeps it (physical operators — one column per
    // level) or zeroes it (scale-normalized operators — every level collapsed
    // onto one column, which is what the solid-harmonic basis does and what
    // keeps its table at the realized-offset count).
    //
    // Unit grid: cell at depth d, index i has center (2i+1) * half_w_d, so
    // the difference of any two centers (at any depths) is an integer
    // multiple of half_w_max_d, where max_d = max(d_t, d_s). We use that as
    // the unit length.
    //
    // Range guards: pairs with |dd| > M2L_KEY_DD_MAX (the basis's
    // m2l_key_dd_max) or |offset| component > M2L_KEY_OFFSET_MAX route to
    // per-pair m2l_translate fallback. These guards exist purely to defend
    // against pathological tree state; a healthy MAC traversal does not
    // produce out-of-range pairs.
    // -----------------------------------------------------------------------
#if defined( CANOPY_ENABLE_DEBUG )
    std::vector<double> half_width_at_depth( _max_depth + 1, 0.0 );
#endif
    const int total_pairs = total_pairs_count;
    std::vector<int> pair_op_idx( total_pairs, -1 );
    std::vector<int> pair_target( total_pairs );
    std::vector<int> pair_source( total_pairs );
    std::vector<int> pair_target_depth( total_pairs );
    std::vector<unsigned char> pair_target_is_shared( total_pairs, 0 );
    std::unordered_map<M2LKey, int, M2LKeyHash> key_to_op;
    std::vector<M2LKey> ops;
    bool overflow_warned = false;

    {
        CANOPY_SCOPED_TIMER_DETAILED(
            Canopy::Profiling::TIMER_ILIST_S3_CLASSIFY_PAIRS );
#if defined( CANOPY_ENABLE_DEBUG )
        // Debug-only: precompute inv_half_width_at_depth so the
        // CANOPY_ENABLE_DEBUG side-by-side check below can run the original
        // FP/gather path for comparison. Stripped in release builds.
        std::vector<double> inv_half_width_at_depth( _max_depth + 1, 0.0 );
        {
            const int num_cells = _device_cells.extent( 0 );
            for ( int i = 0; i < num_cells; i++ )
            {
                const auto& dci = h_dc_for_filter( i );
                if ( dci.depth >= 0 && dci.depth <= _max_depth )
                    half_width_at_depth[dci.depth] = dci.half_width;
            }
            for ( int d = 0; d <= _max_depth; d++ )
                inv_half_width_at_depth[d] =
                    ( half_width_at_depth[d] > 0.0 )
                        ? ( 1.0 / half_width_at_depth[d] )
                        : 0.0;
        }
#endif
        // S3 classify: pure-integer pipeline. For each pair (target, source)
        // we decode (d, ix, iy, iz) from both MortonKeys and compute:
        //   dd    = d_s - d_t
        //   max_d = max(d_t, d_s)
        //   ii    = (2*ix_s+1 - 2^d_s) * 2^(max_d - d_s)
        //         - (2*ix_t+1 - 2^d_t) * 2^(max_d - d_t)   (jj, kk analogous)
        // This is what the original FP path
        //   lround( (sci.center[x] - tci.center[x]) * inv_half_width_at_depth[max_d] )
        // computes in exact arithmetic, with no per-source h_dc_for_filter
        // gather and no FP rounding. Bit-identical M2LKey output by construction.
        //
        // max_d is emitted into the key unconditionally and the key is then
        // handed to KernelType::canonicalize_key, exactly once, before any
        // hash or map lookup sees it. Applying it here and nowhere else is
        // what lets the serial merge below stay unchanged: the keys in
        // local_ops[t] are already canonical.
        //
        // Sharded: per-thread hashmap dedup (O(N) cache-resident),
        // then a tiny serial merge over distinct keys (globally bounded by
        // M2L_OP_COUNT_CAP), then a parallel local->global op-idx remap.
        const size_t n_entries = entries.size();
        std::vector<int> entry_pair_offset( n_entries + 1, 0 );
        for ( size_t e = 0; e < n_entries; ++e )
            entry_pair_offset[e + 1] =
                entry_pair_offset[e] +
                static_cast<int>( entries[e].sources.size() );

        const int nthreads = std::max(
            1, Kokkos::DefaultHostExecutionSpace().concurrency() );
        std::vector<int> entry_begin( nthreads + 1, 0 );
        for ( int t = 1; t < nthreads; ++t )
        {
            const long long target =
                static_cast<long long>( t ) * total_pairs / nthreads;
            auto it = std::lower_bound(
                entry_pair_offset.begin(), entry_pair_offset.end(),
                static_cast<int>( target ) );
            entry_begin[t] =
                static_cast<int>( it - entry_pair_offset.begin() );
        }
        entry_begin[nthreads] = static_cast<int>( n_entries );

        std::vector<std::unordered_map<M2LKey, int, M2LKeyHash>>
            local_k2o( nthreads );
        std::vector<std::vector<M2LKey>> local_ops( nthreads );
        for ( int t = 0; t < nthreads; ++t )
        {
            local_k2o[t].reserve( S3_PER_THREAD_KEYMAP_RESERVE );
            local_ops[t].reserve( S3_PER_THREAD_KEYMAP_RESERVE );
        }

        Kokkos::parallel_for(
            "ilist_s3_classify_shard",
            Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
                0, nthreads ),
            [&]( const int t ) {
                auto& kmap = local_k2o[t];
                auto& ops_t = local_ops[t];
                const int ebeg = entry_begin[t];
                const int eend = entry_begin[t + 1];
                for ( int ei = ebeg; ei < eend; ++ei )
                {
                    const auto& e = entries[ei];
                    // Per-target read: still gather tci for the owner_rank /
                    // shared-flag check. This is once per entry (~1e6), not
                    // per pair (~3e8), so not the bandwidth bottleneck.
                    const auto& tci = h_dc_for_filter( e.target_idx );
                    const bool tgt_shared =
                        ( tci.owner_rank == OWNER_SHARED );

                    // Decode target geometry from MortonKey (once per entry).
                    const int d_t = e.depth;
                    int ix_t, iy_t, iz_t;
                    int d_t_decoded;
                    decode_morton( e.target_key, d_t_decoded,
                                   ix_t, iy_t, iz_t );
                    // d_t_decoded matches e.depth by construction (both
                    // derive from the same tree); used only as a debug-mode
                    // self-check below.
                    (void)d_t_decoded;
                    const int pow2_dt = 1 << d_t;
                    const int two_ix_t_off = 2 * ix_t + 1 - pow2_dt;
                    const int two_iy_t_off = 2 * iy_t + 1 - pow2_dt;
                    const int two_iz_t_off = 2 * iz_t + 1 - pow2_dt;

                    int p = entry_pair_offset[ei];
                    const int n_src =
                        static_cast<int>( e.sources.size() );
                    for ( int si = 0; si < n_src; ++si )
                    {
                        const int s = e.sources[si];
                        const MortonKey s_key = e.source_keys[si];

                        // Decode source geometry from MortonKey — no gather.
                        int d_s, ix_s, iy_s, iz_s;
                        decode_morton( s_key, d_s, ix_s, iy_s, iz_s );

                        const int dd = d_s - d_t;
                        const int max_d = ( d_s > d_t ) ? d_s : d_t;
                        const int shift_s = max_d - d_s;
                        const int shift_t = max_d - d_t;
                        const int pow2_ds = 1 << d_s;
                        const int two_ix_s_off = 2 * ix_s + 1 - pow2_ds;
                        const int two_iy_s_off = 2 * iy_s + 1 - pow2_ds;
                        const int two_iz_s_off = 2 * iz_s + 1 - pow2_ds;
                        // Use 64-bit accumulators to be safe; the magnitudes
                        // are bounded by 2^(max_d+1) which fits in int32 for
                        // max_d <= 30, but max_d is bounded by tree depth
                        // (<= 20) so this is mostly defensive.
                        const long long ii64 =
                            static_cast<long long>( two_ix_s_off )
                                * ( 1LL << shift_s )
                            - static_cast<long long>( two_ix_t_off )
                                * ( 1LL << shift_t );
                        const long long jj64 =
                            static_cast<long long>( two_iy_s_off )
                                * ( 1LL << shift_s )
                            - static_cast<long long>( two_iy_t_off )
                                * ( 1LL << shift_t );
                        const long long kk64 =
                            static_cast<long long>( two_iz_s_off )
                                * ( 1LL << shift_s )
                            - static_cast<long long>( two_iz_t_off )
                                * ( 1LL << shift_t );
                        const int ii = static_cast<int>( ii64 );
                        const int jj = static_cast<int>( jj64 );
                        const int kk = static_cast<int>( kk64 );

                        int local_op = -1;
                        if ( max_d >= 0 && max_d <= _max_depth &&
                             std::abs( dd ) <= M2L_KEY_DD_MAX &&
                             std::abs( ii ) <= M2L_KEY_OFFSET_MAX &&
                             std::abs( jj ) <= M2L_KEY_OFFSET_MAX &&
                             std::abs( kk ) <= M2L_KEY_OFFSET_MAX )
                        {
                            // Canonicalize once, here, before the key is
                            // hashed. Both hash sites downstream (this
                            // per-thread kmap and the serial merge's
                            // key_to_op) therefore see canonical keys only;
                            // do not apply it a second time.
                            const M2LKey key = KernelType::canonicalize_key(
                                M2LKey{ max_d, dd, ii, jj, kk } );
                            auto it = kmap.find( key );
                            if ( it != kmap.end() )
                            {
                                local_op = it->second;
                            }
                            else
                            {
                                local_op = static_cast<int>( ops_t.size() );
                                kmap.emplace( key, local_op );
                                ops_t.push_back( key );
                            }
                        }

#if defined( CANOPY_ENABLE_DEBUG )
                        // Side-by-side correctness check: run the original
                        // FP/gather path and assert the M2LKey + fallback
                        // routing match. Stripped in release builds.
                        {
                            const auto& sci_dbg = h_dc_for_filter( s );
                            const int max_d_dbg =
                                std::max( e.depth, sci_dbg.depth );
                            const double inv_unit_w_dbg =
                                ( max_d_dbg >= 0 &&
                                  max_d_dbg <= _max_depth )
                                    ? inv_half_width_at_depth[max_d_dbg]
                                    : 0.0;
                            const double dx_dbg =
                                sci_dbg.center[0] - tci.center[0];
                            const double dy_dbg =
                                sci_dbg.center[1] - tci.center[1];
                            const double dz_dbg =
                                sci_dbg.center[2] - tci.center[2];
                            const int ii_dbg = static_cast<int>(
                                std::lround( dx_dbg * inv_unit_w_dbg ) );
                            const int jj_dbg = static_cast<int>(
                                std::lround( dy_dbg * inv_unit_w_dbg ) );
                            const int kk_dbg = static_cast<int>(
                                std::lround( dz_dbg * inv_unit_w_dbg ) );
                            const int dd_dbg =
                                sci_dbg.depth - e.depth;
                            const bool in_range_dbg =
                                ( inv_unit_w_dbg > 0.0 &&
                                  std::abs( dd_dbg ) <= M2L_KEY_DD_MAX &&
                                  std::abs( ii_dbg ) <=
                                      M2L_KEY_OFFSET_MAX &&
                                  std::abs( jj_dbg ) <=
                                      M2L_KEY_OFFSET_MAX &&
                                  std::abs( kk_dbg ) <=
                                      M2L_KEY_OFFSET_MAX );
                            const bool in_range_new =
                                ( max_d >= 0 && max_d <= _max_depth &&
                                  std::abs( dd ) <= M2L_KEY_DD_MAX &&
                                  std::abs( ii ) <=
                                      M2L_KEY_OFFSET_MAX &&
                                  std::abs( jj ) <=
                                      M2L_KEY_OFFSET_MAX &&
                                  std::abs( kk ) <=
                                      M2L_KEY_OFFSET_MAX );
                            const bool ok =
                                ( d_t == d_t_decoded ) &&
                                ( d_s == sci_dbg.depth ) &&
                                ( dd == dd_dbg ) &&
                                ( in_range_dbg == in_range_new ) &&
                                ( !in_range_new ||
                                  ( ii == ii_dbg && jj == jj_dbg &&
                                    kk == kk_dbg ) );
                            if ( !ok )
                            {
                                std::fprintf(
                                    stderr,
                                    "[Canopy DEBUG] S3 decode mismatch "
                                    "ei=%d si=%d p=%d: "
                                    "new(d_t=%d d_s=%d dd=%d ii=%d jj=%d "
                                    "kk=%d in=%d) "
                                    "old(d_t=%d d_s=%d dd=%d ii=%d jj=%d "
                                    "kk=%d in=%d) "
                                    "tkey=%llu skey=%llu\n",
                                    ei, si, p,
                                    d_t, d_s, dd, ii, jj, kk,
                                    in_range_new ? 1 : 0,
                                    e.depth, sci_dbg.depth, dd_dbg,
                                    ii_dbg, jj_dbg, kk_dbg,
                                    in_range_dbg ? 1 : 0,
                                    static_cast<unsigned long long>(
                                        e.target_key ),
                                    static_cast<unsigned long long>(
                                        s_key ) );
                                std::abort();
                            }
                        }
#endif

                        pair_op_idx[p] = local_op;
                        pair_target[p] = e.target_idx;
                        pair_source[p] = s;
                        pair_target_depth[p] = e.depth;
                        pair_target_is_shared[p] = tgt_shared ? 1u : 0u;
                        ++p;
                    }
                }
            } );

        // Serial merge of per-thread local op tables into global ops/key_to_op,
        // building a local->global remap per thread. Total work is bounded by
        // sum of distinct keys per thread (in practice ~16 k globally), so
        // this is tiny relative to the 550 M-pair classify pass.
        std::vector<std::vector<int>> local_to_global( nthreads );
        for ( int t = 0; t < nthreads; ++t )
        {
            local_to_global[t].resize( local_ops[t].size() );
            for ( int lo = 0;
                  lo < static_cast<int>( local_ops[t].size() ); ++lo )
            {
                const M2LKey& key = local_ops[t][lo];
                auto it = key_to_op.find( key );
                int g;
                if ( it != key_to_op.end() )
                {
                    g = it->second;
                }
                else if ( static_cast<int>( ops.size() ) <
                          M2L_OP_COUNT_CAP )
                {
                    g = static_cast<int>( ops.size() );
                    key_to_op.emplace( key, g );
                    ops.push_back( key );
                }
                else
                {
                    g = -1;
                    if ( !overflow_warned )
                    {
                        std::fprintf(
                            stderr,
                            "[Canopy] M2L op count exceeded cap %d; "
                            "remaining pairs route to fallback path.\n",
                            M2L_OP_COUNT_CAP );
                        overflow_warned = true;
                    }
                }
                local_to_global[t][lo] = g;
            }
        }

        // Parallel local->global op-idx remap over the disjoint pair slices.
        Kokkos::parallel_for(
            "ilist_s3_remap",
            Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
                0, nthreads ),
            [&]( const int t ) {
                const auto& l2g = local_to_global[t];
                const int pbeg = entry_pair_offset[entry_begin[t]];
                const int pend = entry_pair_offset[entry_begin[t + 1]];
                for ( int p = pbeg; p < pend; ++p )
                {
                    const int lo = pair_op_idx[p];
                    pair_op_idx[p] =
                        ( lo >= 0 ) ? l2g[lo] : -1;
                }
            } );
    }

    const int n_unique_ops = static_cast<int>( ops.size() );

    // Retain the realized key list past the end of this function as a
    // read-only diagnostic surface (see m2l_realized_keys()). Copy rather
    // than move: `ops` is still consumed by the stage-4 table build below.
    _m2l_realized_keys = ops;

    // -----------------------------------------------------------------------
    // Stage 4: build the (Nt, Ns, n_unique_ops) operator table on host,
    // then deep_copy to device. One m2l_build_operator call per canonical
    // key. No physical width enters the builder — it is handed
    // (dd, ii, jj, kk) and nothing else — so a basis whose operator needs
    // the absolute level must carry it through canonicalize_key and read it
    // back out of the key here (supplying the physical width to the builder
    // is T9's). For the solid-harmonic basis max_d is canonicalized away,
    // so the realized key set under MAC = 0.5 is bounded and independent of
    // tree depth.
    // -----------------------------------------------------------------------
    {
        const int Nt = KernelType::num_coeffs_per_cell;
        const int Ns = KernelType::m2l_num_src_coeffs;
        m2l_operators_type op_table(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "m2l_op_table" ),
            Nt, Ns, n_unique_ops > 0 ? n_unique_ops : 1 );
        auto h_op = Kokkos::create_mirror_view( op_table );

        if ( n_unique_ops > 0 )
        {
            CANOPY_SCOPED_TIMER_DETAILED(
                Canopy::Profiling::TIMER_ILIST_S4_OP_TABLE_BUILD );
            // m2l_build_operator runs on host, so it needs a HostSpace
            // aux. Build one rather than mirroring the device aux: the
            // sweep cannot mirror a struct it is not allowed to look
            // inside. This is bit-identical, not merely equal —
            // build_aux_tables fills every entry on a host mirror from a
            // pure function of (n, m) before deep-copying, so the host
            // build and a mirror of the device build produce the same
            // bytes.
            const auto h_aux =
                KernelType::template build_aux_tables<Kokkos::HostSpace>(
                    KernelType::max_order );
            for ( int op_idx = 0; op_idx < n_unique_ops; op_idx++ )
            {
                const auto& k = ops[op_idx];
                auto T_slice = Kokkos::subview( h_op, Kokkos::ALL,
                                                Kokkos::ALL, op_idx );
                KernelType::m2l_build_operator( k.dd, k.ii, k.jj, k.kk, h_aux,
                                                T_slice );
            }
        }
        {
            CANOPY_SCOPED_TIMER_DETAILED(
                Canopy::Profiling::TIMER_ILIST_S4_OP_TABLE_COPY );
            Kokkos::deep_copy( op_table, h_op );
        }
        _m2l_op_table = op_table;
    }

    // -----------------------------------------------------------------------
    // Stage 5 (Tier 2): build target-major CSRs, partitioned by target
    // sharedness. The fused team-per-target kernel walks each target's
    // source slice and writes its local once; no atomics on _locals.
    // Pairs with op_idx == -1 stay in the CSR (skipped at runtime) and
    // are also entered into the per-depth fallback tables. Each physical
    // pair is therefore processed by exactly one path: fast pairs by the
    // fused kernel, fallback pairs by run_m2l_fallback_at_depth.
    //
    // Within each CSR, targets are sorted primarily by depth (ascending)
    // so that run_m2l_at_depth(d) operates on the contiguous depth-d
    // slice given by _m2l_sh_csr_depth_offsets.
    //
    // pair_target_depth/pair_target_is_shared/pair_op_idx/pair_target/
    // pair_source were filled in pair-of-entries iteration order, which
    // groups all pairs of one target contiguously. We use that to walk
    // by target without re-sorting pair-by-pair.
    // -----------------------------------------------------------------------
    _m2l_fallback_offsets_host.assign( _max_depth + 2, 0 );

    struct CsrTargetEntry
    {
        int target_idx;
        int depth;
        int begin; // index into pair_* arrays
        int end;
    };
    std::vector<CsrTargetEntry> ns_target_entries;
    std::vector<CsrTargetEntry> sh_target_entries;
    ns_target_entries.reserve( entries.size() );
    sh_target_entries.reserve( entries.size() );

    {
        CANOPY_SCOPED_TIMER_DETAILED(
            Canopy::Profiling::TIMER_ILIST_S5_CSR_PARTITION_SORT );
        int pair_cursor = 0;
        for ( const auto& e : entries )
        {
            const int n_src = static_cast<int>( e.sources.size() );
            const int begin = pair_cursor;
            const int end = pair_cursor + n_src;
            const bool tgt_shared =
                ( pair_target_is_shared[begin] != 0 );
            CsrTargetEntry ce{ e.target_idx, e.depth, begin, end };
            if ( tgt_shared )
                sh_target_entries.push_back( ce );
            else
                ns_target_entries.push_back( ce );
            pair_cursor = end;
        }

        auto sort_by_depth = []( std::vector<CsrTargetEntry>& v )
        {
            std::sort( v.begin(), v.end(),
                       []( const CsrTargetEntry& a, const CsrTargetEntry& b )
                       {
                           if ( a.depth != b.depth )
                               return a.depth < b.depth;
                           return a.target_idx < b.target_idx;
                       } );
        };
        sort_by_depth( ns_target_entries );
        sort_by_depth( sh_target_entries );
    }

    auto build_csr = [&]( const std::vector<CsrTargetEntry>& tgt_entries,
                          std::vector<int>& csr_targets_h,
                          std::vector<int>& csr_offsets_h,
                          std::vector<int>& csr_sources_h,
                          std::vector<int>& csr_op_idx_h )
    {
        const int n_tgt = static_cast<int>( tgt_entries.size() );
        csr_targets_h.resize( n_tgt );
        csr_offsets_h.assign( n_tgt + 1, 0 );
        int total = 0;
        for ( int i = 0; i < n_tgt; i++ )
        {
            csr_targets_h[i] = tgt_entries[i].target_idx;
            const int n_src =
                tgt_entries[i].end - tgt_entries[i].begin;
            csr_offsets_h[i + 1] = csr_offsets_h[i] + n_src;
            total += n_src;
        }
        csr_sources_h.resize( total );
        csr_op_idx_h.resize( total );
        int slot = 0;
        for ( const auto& ce : tgt_entries )
        {
            for ( int p = ce.begin; p < ce.end; p++ )
            {
                csr_sources_h[slot] = pair_source[p];
                csr_op_idx_h[slot] = pair_op_idx[p];
                slot++;
            }
        }
    };

    std::vector<int> ns_targets_h, ns_offsets_h, ns_sources_h, ns_op_idx_h;
    std::vector<int> sh_targets_h, sh_offsets_h, sh_sources_h, sh_op_idx_h;
    {
        CANOPY_SCOPED_TIMER_DETAILED(
            Canopy::Profiling::TIMER_ILIST_S5_CSR_BUILD );
        build_csr( ns_target_entries, ns_targets_h, ns_offsets_h, ns_sources_h,
                   ns_op_idx_h );
        build_csr( sh_target_entries, sh_targets_h, sh_offsets_h, sh_sources_h,
                   sh_op_idx_h );

        // Per-depth offsets into the shared-CSR target list.
        _m2l_sh_csr_depth_offsets.assign( _max_depth + 2, 0 );
        int cursor = 0;
        const int n_sh_t = static_cast<int>( sh_target_entries.size() );
        for ( int d = 0; d <= _max_depth + 1; d++ )
        {
            while ( cursor < n_sh_t &&
                    sh_target_entries[cursor].depth < d )
                ++cursor;
            _m2l_sh_csr_depth_offsets[d] = cursor;
        }
    }

    // Fallback table: collect per-depth (target, source) pairs with
    // op_idx == -1, ordered by target depth. In a healthy MAC traversal
    // no pairs are out-of-range, so do a cheap detection scan first and
    // skip the two full O(total_pairs) walks when there's nothing to
    // place. _m2l_fallback_offsets_host is already zero-initialized
    // above, which is the correct empty-table state.
    {
        CANOPY_SCOPED_TIMER_DETAILED(
            Canopy::Profiling::TIMER_ILIST_S5_FALLBACK_TABLE );

        bool any_fallback = false;
        for ( int p = 0; p < total_pairs; p++ )
        {
            if ( pair_op_idx[p] < 0 )
            {
                any_fallback = true;
                break;
            }
        }

        _m2l_fallback_count_per_active_depth.assign( _max_depth + 1, 0 );

        if ( !any_fallback )
        {
            _m2l_fallback_targets = Kokkos::View<int*, memory_space>(
                "m2l_fallback_targets", 0 );
            _m2l_fallback_sources = Kokkos::View<int*, memory_space>(
                "m2l_fallback_sources", 0 );
        }
        else
        {
            for ( int p = 0; p < total_pairs; p++ )
            {
                if ( pair_op_idx[p] >= 0 )
                    continue;
                const int d = pair_target_depth[p];
                if ( d < 0 || d > _max_depth )
                    continue;
                _m2l_fallback_offsets_host[d + 1]++;
            }
            for ( int d = 0; d <= _max_depth; d++ )
                _m2l_fallback_offsets_host[d + 1] +=
                    _m2l_fallback_offsets_host[d];

            const int total_fallback =
                _m2l_fallback_offsets_host[_max_depth + 1];
            std::vector<int> fb_targets_h( total_fallback );
            std::vector<int> fb_sources_h( total_fallback );
            std::vector<int> fb_cursors = _m2l_fallback_offsets_host;
            for ( int p = 0; p < total_pairs; p++ )
            {
                if ( pair_op_idx[p] >= 0 )
                    continue;
                const int d = pair_target_depth[p];
                if ( d < 0 || d > _max_depth )
                    continue;
                const int slot = fb_cursors[d]++;
                fb_targets_h[slot] = pair_target[p];
                fb_sources_h[slot] = pair_source[p];
            }

            _m2l_fallback_targets = Kokkos::View<int*, memory_space>(
                Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                    "m2l_fallback_targets" ),
                total_fallback );
            _m2l_fallback_sources = Kokkos::View<int*, memory_space>(
                Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                    "m2l_fallback_sources" ),
                total_fallback );
            Kokkos::View<const int*, Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                h_t( fb_targets_h.data(), total_fallback );
            Kokkos::View<const int*, Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                h_s( fb_sources_h.data(), total_fallback );
            Kokkos::deep_copy( _m2l_fallback_targets, h_t );
            Kokkos::deep_copy( _m2l_fallback_sources, h_s );

            for ( int d = 0; d <= _max_depth; d++ )
                _m2l_fallback_count_per_active_depth[d] =
                    static_cast<long long>(
                        _m2l_fallback_offsets_host[d + 1] -
                        _m2l_fallback_offsets_host[d] );
        }
    }

    // Upload the two CSRs to device.
    {
        CANOPY_SCOPED_TIMER_DETAILED(
            Canopy::Profiling::TIMER_ILIST_S5_DEVICE_UPLOAD );
        // Wrap the source std::vector in an unmanaged host view and
        // deep_copy it directly to the device — avoids the per-element
        // mirror copy loop and the temporary mirror allocation.
        auto upload_int = []( const std::vector<int>& src, const char* label,
                              Kokkos::View<int*, memory_space>& dst )
        {
            dst = Kokkos::View<int*, memory_space>(
                Kokkos::view_alloc( std::string( label ),
                                    Kokkos::WithoutInitializing ),
                src.size() );
            if ( src.empty() )
                return;
            Kokkos::View<const int*, Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                h_src( src.data(), src.size() );
            Kokkos::deep_copy( dst, h_src );
        };

        upload_int( ns_targets_h, "m2l_ns_csr_targets", _m2l_ns_csr_targets );
        upload_int( ns_offsets_h, "m2l_ns_csr_offsets", _m2l_ns_csr_offsets );
        upload_int( ns_sources_h, "m2l_ns_csr_sources", _m2l_ns_csr_sources );
        upload_int( ns_op_idx_h,  "m2l_ns_csr_op_idx",  _m2l_ns_csr_op_idx );
        upload_int( sh_targets_h, "m2l_sh_csr_targets", _m2l_sh_csr_targets );
        upload_int( sh_offsets_h, "m2l_sh_csr_offsets", _m2l_sh_csr_offsets );
        upload_int( sh_sources_h, "m2l_sh_csr_sources", _m2l_sh_csr_sources );
        upload_int( sh_op_idx_h,  "m2l_sh_csr_op_idx",  _m2l_sh_csr_op_idx );
    }

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

    // Group transfers by peer rank so we issue O(#peers) messages instead
    // of O(#cells). On Tuolumne the per-cell pattern exhausts Cray-MPICH's
    // internal request freelist at O(1e5) in-flight Isends.
    //
    // Both sender and receiver sort each peer's cell list by MortonKey so
    // pack/unpack order matches without an extra metadata exchange.
    std::map<int, std::vector<std::pair<MortonKey, int>>> send_by_peer_kv;
    for ( const auto& ct : m2l.sends )
    {
        auto it = _key_to_cell_idx->find( ct.cell_key );
        if ( it == _key_to_cell_idx->end() )
            continue;
        send_by_peer_kv[ct.remote_rank].emplace_back( ct.cell_key, it->second );
    }

    std::map<int, std::vector<std::pair<MortonKey, int>>> recv_by_peer_kv;
    for ( const auto& ct : m2l.receives )
    {
        auto it = _key_to_cell_idx->find( ct.cell_key );
        if ( it == _key_to_cell_idx->end() )
            continue;
        recv_by_peer_kv[ct.remote_rank].emplace_back( ct.cell_key, it->second );
    }

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

    detail::coalesced_view_exchange( multipoles, _comm, sends_by_peer,
                                     recvs_by_peer,
                                     /*accumulate_on_recv=*/false,
                                     _exch_bufs );
}

// -------------------------------------------------------------------------
// Tier 2 fused team-per-target M2L kernel. Each team owns one target cell
// and walks its CSR source slice, driving the basis's three M2L stages:
//
//   KernelType::m2l_pre_cell   once per source cell in the slice,
//   KernelType::m2l_core       once per (target, source) pair,
//   KernelType::m2l_post_cell  once, after the slice, to write _locals.
//
// The sweep owns the traversal, the CSR walk, the team launch, the scratch
// allocation and its zero-fill; it owns nothing about the arithmetic. The
// operator set is opaque here — it is carried into the stages untouched and
// selected only by the integer op_idx the CSR holds. Pairs with
// op_idx == -1 are skipped; they are processed by the per-depth fallback
// path.
//
// The scratch is KernelType::m2l_scratch_bytes(NComps) raw bytes of team
// scratch, zero-filled once per team before the pair loop and shared by all
// three stages. The layout inside is the basis's: the solid-harmonic basis
// splits it into separate real and imaginary scalar arrays, a choice that
// halves shared-memory bank conflicts and is arithmetic-visible, and that
// is why the sweep hands over bytes rather than a typed accumulator view.
// A basis whose accumulator identity element is not all-zero bytes must
// establish it itself.
//
// m2l_post_cell uses += into _locals (read-modify-write, no atomics) so the
// kernel is composable with both the L2L-pre-existing state on shared
// targets at per-depth time and the (zero-initialized) state on nonshared
// targets.
// -------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace, class KernelType>
void DownwardSweep<MemorySpace, ExecutionSpace, KernelType>::run_m2l_fused(
    const Kokkos::View<int*, memory_space>& csr_targets,
    const Kokkos::View<int*, memory_space>& csr_offsets,
    const Kokkos::View<int*, memory_space>& csr_sources,
    const Kokkos::View<int*, memory_space>& csr_op_idx,
    int target_index_lo, int target_index_hi )
{
    const int n_teams = target_index_hi - target_index_lo;
    if ( n_teams <= 0 )
        return;

    auto multipoles = _m2l_multipoles_view;
    auto locals = _locals;
    auto ops = _m2l_op_table;

    using team_policy = Kokkos::TeamPolicy<execution_space>;
    using member_t = typename team_policy::member_type;
    using scratch_space = typename execution_space::scratch_memory_space;

    // Raw bytes: the basis owns the layout inside (see
    // KernelType::m2l_scratch_bytes), the sweep only sizes it, zero-fills
    // it and passes it to the stages.
    using ScratchBytes = Kokkos::View<char*, scratch_space,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    // constexpr, not a runtime size: NComps is a compile-time constant and
    // m2l_scratch_bytes is constexpr, so the stages' internal extents stay
    // compile-time and the fused kernel keeps unrolling as it did before
    // the arithmetic moved into the basis.
    constexpr size_t scratch_bytes = KernelType::m2l_scratch_bytes( NComps );
    constexpr int scratch_bytes_int = static_cast<int>( scratch_bytes );

    // Kokkos::AUTO picks a sensible team_size per backend (e.g. 32-128 on
    // CUDA, 1 on Serial). Nt=28 at P=6 fits inside a warp; AUTO has been
    // observed to choose ~32 there, which gives good occupancy.
    team_policy policy( n_teams, Kokkos::AUTO );
    policy.set_scratch_size(
        0, Kokkos::PerTeam( ScratchBytes::shmem_size( scratch_bytes ) ) );

    Kokkos::parallel_for(
        "M2L_fused", policy, KOKKOS_LAMBDA( const member_t& team ) {
            const int k = target_index_lo + team.league_rank();
            const int target_cell = csr_targets( k );
            const int off_lo = csr_offsets( k );
            const int off_hi = csr_offsets( k + 1 );

            ScratchBytes scratch( team.team_scratch( 0 ), scratch_bytes );

            // Zero-fill is a byte fill because the layout is the basis's.
            // Every floating-point accumulator this can hold has an
            // all-zero-bytes representation under IEEE-754.
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange( team, scratch_bytes_int ),
                [&]( int i ) { scratch( i ) = char( 0 ); } );
            team.team_barrier();

            for ( int s = off_lo; s < off_hi; s++ )
            {
                const int op_idx = csr_op_idx( s );
                if ( op_idx < 0 )
                    continue;
                const int src_cell = csr_sources( s );

                KernelType::m2l_pre_cell( team, multipoles, src_cell, ops,
                                          scratch );
                KernelType::m2l_core( team, multipoles, src_cell, ops, op_idx,
                                      scratch );
            }

            KernelType::m2l_post_cell( team, scratch, locals, target_cell,
                                       ops );
        } );
}

// -------------------------------------------------------------------------
// run_m2l_all: process M2L for every nonshared target, batched across all
// depths in one kernel launch. Safe to do before the per-depth loop
// because the snapshot/allreduce barrier only operates on shared cells,
// which run_m2l_all does not touch.
// -------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace, class KernelType>
void DownwardSweep<MemorySpace, ExecutionSpace, KernelType>::run_m2l_all()
{
    CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_M2L_KERNEL );
    const int n_targets =
        static_cast<int>( _m2l_ns_csr_targets.extent( 0 ) );
    if ( n_targets == 0 )
        return;
    run_m2l_fused( _m2l_ns_csr_targets, _m2l_ns_csr_offsets,
                   _m2l_ns_csr_sources, _m2l_ns_csr_op_idx,
                   /*target_index_lo=*/0,
                   /*target_index_hi=*/n_targets );
    Kokkos::fence();
}

// -------------------------------------------------------------------------
// run_m2l_at_depth: process M2L contributions whose target sits at the
// given depth and is shared (i.e., needs the snapshot/allreduce barrier),
// plus the per-depth fallback slice. Nonshared targets are handled in
// one batch by run_m2l_all() before the per-depth loop.
// -------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace, class KernelType>
void DownwardSweep<MemorySpace, ExecutionSpace, KernelType>::run_m2l_at_depth(
    int depth )
{
    CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_M2L_KERNEL );
    if ( depth < 0 || depth > _max_depth )
        return;
    if ( !_m2l_sh_csr_depth_offsets.empty() )
    {
        const int lo = _m2l_sh_csr_depth_offsets[depth];
        const int hi = _m2l_sh_csr_depth_offsets[depth + 1];
        if ( hi > lo )
            run_m2l_fused( _m2l_sh_csr_targets, _m2l_sh_csr_offsets,
                           _m2l_sh_csr_sources, _m2l_sh_csr_op_idx, lo, hi );
    }
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
    auto aux = _aux;
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
                                       dz, src_ci.half_width,
                                       target_ci.half_width, aux, L_target );
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
    auto aux = _aux;
    auto& d_parents = _d_internals_at_depth[depth];
    auto children = *_d_cell_children;

    using team_policy = Kokkos::TeamPolicy<execution_space>;
    using team_member_type = typename team_policy::member_type;

    team_policy policy( nparents, Kokkos::AUTO );

    Kokkos::parallel_for(
        "L2L", policy, KOKKOS_LAMBDA( const team_member_type& team ) {
            const int league = team.league_rank();
            const int parent_cell = d_parents( league );
            const auto& parent_ci = device_cells( parent_cell );

            for ( int k = 0; k < 8; k++ )
            {
                const int ci = children( parent_cell, k );
                if ( ci < 0 )
                    break;
                const auto& ccell = device_cells( ci );

                const scalar_type dx = ccell.center[0] - parent_ci.center[0];
                const scalar_type dy = ccell.center[1] - parent_ci.center[1];
                const scalar_type dz = ccell.center[2] - parent_ci.center[2];

                auto L_child =
                    Kokkos::subview( locals, ci, Kokkos::ALL, Kokkos::ALL );
                KernelType::l2l_translate( team, locals, parent_cell, dx, dy,
                                           dz, ccell.half_width,
                                           parent_ci.half_width, aux, L_child );
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
    // coeff_type() is the basis's coefficient identity element.
    _shared_snapshot_buf.assign( nshared * per_cell_complex, coeff_type() );
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

    std::vector<coeff_type> sendbuf( total_complex );
    std::vector<coeff_type> recvbuf( total_complex );

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
        ( sizeof( component_scalar_type ) == 8 ) ? MPI_DOUBLE : MPI_FLOAT;
    {
        CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_M2L_ALLREDUCE );
        MPI_Allreduce(
            reinterpret_cast<component_scalar_type*>( sendbuf.data() ),
            reinterpret_cast<component_scalar_type*>( recvbuf.data() ),
            scalars_per_coeff * total_complex, mpi_scalar, MPI_SUM, _comm );
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

    auto h_dc = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{},
                                                     _device_cells );

    // The cell_key in each L2L plan entry is the CHILD's key. Filter to
    // entries whose child is at depth+1 (i.e. whose parent is at the depth
    // we just finished L2L'ing on), then group by peer rank and sort each
    // peer's list by MortonKey for deterministic pack/unpack alignment.
    std::map<int, std::vector<std::pair<MortonKey, int>>> send_by_peer_kv;
    for ( const auto& ct : l2l.sends )
    {
        auto it = _key_to_cell_idx->find( ct.cell_key );
        if ( it == _key_to_cell_idx->end() )
            continue;
        const int cidx = it->second;
        if ( h_dc( cidx ).depth != depth + 1 )
            continue;
        send_by_peer_kv[ct.remote_rank].emplace_back( ct.cell_key, cidx );
    }

    std::map<int, std::vector<std::pair<MortonKey, int>>> recv_by_peer_kv;
    for ( const auto& ct : l2l.receives )
    {
        auto it = _key_to_cell_idx->find( ct.cell_key );
        if ( it == _key_to_cell_idx->end() )
            continue;
        const int cidx = it->second;
        if ( h_dc( cidx ).depth != depth + 1 )
            continue;
        recv_by_peer_kv[ct.remote_rank].emplace_back( ct.cell_key, cidx );
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

    // Received locals accumulate into the child's local (child owner may
    // already have M2L contributions in place).
    detail::coalesced_view_exchange( _locals, _comm, sends_by_peer,
                                     recvs_by_peer,
                                     /*accumulate_on_recv=*/true,
                                     _exch_bufs );
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
            KernelType::l2p_evaluate( locals, cidx, dx, dy, dz,
                                      dci.half_width, phi, writer,
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
            // coeff_type() is the basis's coefficient identity element.
            Kokkos::deep_copy( _locals, coeff_type() );
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

        // L2P: evaluate local expansion at each particle. After step 5
        // every multipole/local in the pipeline is in scale-normalized
        // form, so l2p_evaluate consumes L̄ directly.
        run_l2p( particle_positions, potential_out, gradient_out,
                 compute_gradient );
    }
    CANOPY_PRINT_DOWNWARD_TIMERS( _comm );
    CANOPY_PRINT_ILIST_TIMERS( _comm );
}

} // namespace Canopy

#endif // CANOPY_DOWNWARD_SWEEP_HPP
