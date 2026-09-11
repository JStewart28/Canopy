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

// ===========================================================================
// The far-field conformance gate.
//
// Drives UpwardSweep and DownwardSweep with CanopyTest::MonopoleBasis — a
// non-harmonic basis carrying one REAL coefficient per cell per component —
// and compares downward.locals() against a host recomputation of the same
// quantity at EXPECT_DOUBLE_EQ. Nothing here is a spherical harmonic and
// nothing here is Kokkos::complex, so a pass is evidence that the
// trait-and-operator contract the sweeps describe is a real interface and not
// a rename of the solid-harmonic one.
//
// ---------------------------------------------------------------------------
// WHAT THE REFERENCE IS, AND WHY IT SUMS IN THE ORDER IT DOES
//
// For MonopoleBasis the whole downward pipeline collapses to a telescoping
// sum. Writing D(a) for the M2L delta accumulated at cell a,
//
//     D(a) = sum over s in ilist(a) of  T(key(a, s)) * M(s)
//
// L2L copies a parent's local to each child, so the value the sweep leaves in
// locals() is
//
//     L(t) = D(t) + L(parent(t)),      L(root) = D(root)
//
// and the reference below evaluates exactly that, root-downward.
//
// Three things make the comparison EXACT rather than merely close, and all
// three are load-bearing:
//
//  1. D(a) is summed in the sweep's own order. EXPECT_DOUBLE_EQ is 4 ULP, and
//     a reference that merely summed over the same SET would reassociate a few
//     hundred same-magnitude terms and drift past 4 ULP — at which point a
//     failure could not be told apart from a real defect. The sweep's CSR
//     sorts target entries by (depth, target_idx) and keeps the traversal's
//     push order within an entry (src/Canopy_DownwardSweep.hpp:1231-1290), and
//     that push order IS the vector order of
//     comm_plan.m2l_plan().interaction_lists[target_key]
//     (src/Canopy_CommunicationPlan.hpp:481). So the reference iterates that
//     vector as-is.
//
//  2. The operator value and the multiply-accumulate step are not duplicated.
//     Both go through MonopoleBasis::m2l_operator_entry and
//     MonopoleBasis::m2l_accumulate, the same functions the device kernel
//     calls.
//
//  3. The reference takes the UPWARD sweep's output as given. It mirrors
//     upward.multipoles() rather than recomputing P2M and M2M, so no
//     assumption about particle or child iteration order enters. This test
//     gates the FAR FIELD — M2L, L2L, L2P — which is what T6 is for.
//
// ---------------------------------------------------------------------------
// THE SECOND THING THIS FILE GATES: THE LEVEL REACHES THE KEY
//
// T7 made the sweep's M2L key carry max_d and gave the basis the say over
// whether it survives, through KernelType::canonicalize_key. LaplaceKernel
// zeroes it; MonopoleBasis keeps it. The zeroing branch is covered by the
// Laplace-solve gate pinning that its key set did not move, and the keeping
// branch is covered here, by levelReachesTheKey: the same tree driven through
// two sweeps whose bases differ in canonicalize_key alone must realize
// strictly more distinct keys under the one that keeps the level. See that
// test's own comment block for why it runs at np=1 and why it costs no
// operator values.
//
// ---------------------------------------------------------------------------
// THE DIAGNOSTIC SURFACE THIS USES, AND THE ONE IT DOES NOT
//
// The reference reads comm_plan.m2l_plan().interaction_lists, which is public
// (src/Canopy_CommunicationPlan.hpp:96, :219) and which
// tests/tstDownwardSweep.hpp:831 already establishes as the route a test takes
// to the interaction lists. The eight CSR views the sweep actually walks
// (_m2l_ns_csr_*, _m2l_sh_csr_*, src/Canopy_DownwardSweep.hpp:402-411) are
// private with no accessor, and no accessor was added for them: widening
// DownwardSweep's diagnostic surface is not this test's business.
//
// ---------------------------------------------------------------------------
// RANK COUNTS, AND WHY np=1 IS NOT A SPECIAL CASE
//
// Shared cells are NOT a multi-rank phenomenon. A cell is shared when
// `depth <= replication_depth && !is_leaf`
// (src/Canopy_CommunicationPlan.hpp:698) — a function of the tree alone, with
// no rank-count condition — and allreduce_shared_locals_at_depth runs
// unconditionally. At this test's configuration np=1 has shared cells at
// depths 0, 1 and 2 just as np=2 does. The reference above therefore makes no
// distinction: D(a) is defined globally (the communication plan puts
// interaction_lists[a] on exactly one rank — a's owner, or rank 0 when a is
// shared), is summed across ranks by one MPI_Allreduce of exact zeros, and the
// same telescoping runs at every rank count. Nothing here is designed on the
// premise that np=1 is insulated from the shared-cell path.
//
// ---------------------------------------------------------------------------
// THE NEGATIVE TEST — RUNNING IT BY HAND
//
// The block at the bottom of this file, guarded by
// CANOPY_TEST_EXPECT_COMPILE_FAILURE, declares a basis whose coefficient
// traits are internally inconsistent and instantiates both sweeps on it. It
// MUST NOT COMPILE. CTest cannot assert a compile failure, so this is run by
// hand; it is committed rather than applied-and-reverted so that a later
// session can re-run it, and so that it fails loudly if any of the four
// guards it targets is ever deleted.
//
//   b=build-tuolumne/tests
//   d=$b/CMakeFiles/Canopy_Test_FarFieldContract_MPI_SERIAL.dir
//   touch $b/SERIAL/tstFarFieldContract_SERIAL.cpp
//   make -C $b Canopy_Test_FarFieldContract_MPI_SERIAL \
//     CXX_DEFINES="$(sed -n 's/^CXX_DEFINES = //p' $d/flags.make) \
//                  -DCANOPY_TEST_EXPECT_COMPILE_FAILURE"
//
// (make's command-line assignment overrides the one flags.make makes, so the
// sed re-supplies the defines the build needs and appends ours. `touch` is
// there because changing a -D does not change any file timestamp.)
//
// Expect a diagnostic quoting one of the four sweep static_asserts:
//   src/Canopy_UpwardSweep.hpp:73-78    sizeof relation
//   src/Canopy_UpwardSweep.hpp:84-92    agreement with detail::coeff_traits
//   src/Canopy_DownwardSweep.hpp:117-122  sizeof relation
//   src/Canopy_DownwardSweep.hpp:128-136  agreement with detail::coeff_traits
// A build that fails with any other template error has not established that
// the guard is what rejected the basis.
// ===========================================================================

#include "CanopyTest_MonopoleBasis.hpp"

#include "Canopy_CommunicationPlan.hpp"
#include "Canopy_DownwardSweep.hpp"
#include "Canopy_TreeBuilder.hpp"
#include "Canopy_TreePartitioner.hpp"
#include "Canopy_UpwardSweep.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>

namespace Test
{
//---------------------------------------------------------------------------//

using namespace Canopy;

namespace FarFieldContractTest
{

// Read CANOPY_MAC_THETA so the test can be rerun at a different MAC without
// rebuilding, matching tests/tstDownwardSweep.hpp.
inline double get_test_mac_theta()
{
    if ( const char* s = std::getenv( "CANOPY_MAC_THETA" ) )
        return std::atof( s );
    return 0.5;
}

enum FieldIdx
{
    Position = 0,
    Charge = 1
};

// The conformance basis. Order is nominal — MonopoleBasis carries a single
// coefficient regardless, and max_order reaches nothing but build_aux_tables,
// which for this basis ignores it.
//
// NComps = 2 on purpose. With one coefficient per cell, NComps = 1 would make
// the shared-cell Allreduce's slot stride 1 and its slot expression
// degenerate, so a slot-indexing error there would be invisible. Two
// components give a stride of 2 and put the pack/unpack loops of
// allreduce_shared_locals_at_depth under a real index (see R6 in the design
// document, and T10, which changes those loops).
static constexpr int BASIS_ORDER = 0;
static constexpr int BASIS_NCOMPS = 2;
using Basis = CanopyTest::MonopoleBasis<double, BASIS_ORDER, BASIS_NCOMPS>;

using DataTypes = Cabana::MemberTypes<double[3], double[Basis::num_components]>;
using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
using AoSoA_ht = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;

// Random positions in [0,1)^3 and strictly positive per-component charges, so
// every cell's monopole is non-zero and a dropped interaction-list entry
// cannot cancel against another.
void generate_test_particles( AoSoA_t& particles, int num_particles, int rank )
{
    AoSoA_ht particles_h( "particles_h", num_particles );
    auto h_pos = Cabana::slice<Position>( particles_h );
    auto h_q = Cabana::slice<Charge>( particles_h );

    std::mt19937 gen( 42 + rank );
    std::uniform_real_distribution<double> pos_dist( 0.0, 1.0 );
    std::uniform_real_distribution<double> q_dist( 0.1, 1.0 );

    for ( int i = 0; i < num_particles; i++ )
    {
        h_pos( i, 0 ) = pos_dist( gen );
        h_pos( i, 1 ) = pos_dist( gen );
        h_pos( i, 2 ) = pos_dist( gen );
        for ( int c = 0; c < Basis::num_components; c++ )
            h_q( i, c ) = q_dist( gen );
    }

    Cabana::deep_copy( particles, particles_h );
}

// In-repo FNV-1a over the tree's key sequence, used only to assert that every
// rank built the same global tree in the same order. That is what lets the
// reference below use a rank-local cell index as a global one.
inline std::uint64_t hash_key_sequence( const std::vector<CellInfo>& cells )
{
    std::uint64_t h = 1469598103934665603ull;
    for ( const auto& ci : cells )
    {
        std::uint64_t k = static_cast<std::uint64_t>( ci.key );
        for ( int b = 0; b < 8; b++ )
        {
            h ^= ( k >> ( 8 * b ) ) & 0xffull;
            h *= 1099511628211ull;
        }
    }
    return h;
}

//---------------------------------------------------------------------------//
// The fixture: build the tree, partition, plan, and run both sweeps on
// MonopoleBasis. Mirrors tests/tstDownwardSweep.hpp:156-163 — construct
// builder, partitioner and comm plan, then both sweeps and setup() — which is
// the pattern for driving the sweeps directly without going through Solver.
//---------------------------------------------------------------------------//
template <class TEST_MS, class TEST_ES>
struct ContractFixture
{
    using basis = Basis;

    int num_particles;
    int ncrit;
    int max_depth;
    double tolerance;
    int replication_depth;

    AoSoA_t particles{ "particles", 0 };
    TreeBuilder<TEST_MS, TEST_ES> builder;
    TreePartitioner<TEST_MS, TEST_ES> partitioner;
    CommunicationPlan<TEST_MS, TEST_ES> comm_plan;
    UpwardSweep<TEST_MS, TEST_ES, basis> upward;
    DownwardSweep<TEST_MS, TEST_ES, basis> downward;
    int num_local = 0;

    typename DownwardSweep<TEST_MS, TEST_ES, basis>::potential_view_type
        potential;
    typename DownwardSweep<TEST_MS, TEST_ES, basis>::gradient_view_type
        gradient;

    ContractFixture( int num_particles_, int ncrit_, int max_depth_,
                     double tolerance_, int replication_depth_ )
        : num_particles( num_particles_ )
        , ncrit( ncrit_ )
        , max_depth( max_depth_ )
        , tolerance( tolerance_ )
        , replication_depth( replication_depth_ )
        , builder( MPI_COMM_WORLD, ncrit_, max_depth_,
                   std::array<double, 6>{ tolerance_, tolerance_, tolerance_,
                                          tolerance_, tolerance_, tolerance_ },
                   tolerance_ )
        , partitioner( MPI_COMM_WORLD, replication_depth_ )
        , comm_plan( MPI_COMM_WORLD, get_test_mac_theta() )
        , upward( MPI_COMM_WORLD )
        , downward( MPI_COMM_WORLD )
    {
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );

        particles = AoSoA_t( "particles", num_particles );
        generate_test_particles( particles, num_particles, rank );

        auto positions = Cabana::slice<Position>( particles );
        builder.build( positions, num_particles );
        partitioner.partition( builder, particles, num_particles );
        num_local = partitioner.num_local_particles();

        positions = Cabana::slice<Position>( particles );
        builder.build( positions, num_local );

        comm_plan.build( builder.cells(), partitioner.ownership(),
                         partitioner.cell_owner_map(), replication_depth );

        upward.setup( builder.cells(), partitioner.cell_owner_map(),
                      builder.particle_keys(), num_local );
        upward.execute( Cabana::slice<Charge>( particles ),
                        Cabana::slice<Position>( particles ), comm_plan );

        downward.setup( upward, num_local );

        potential = downward.allocate_potential( num_local );
        gradient = downward.allocate_gradient( num_local );
        Kokkos::deep_copy( potential, 0.0 );
        Kokkos::deep_copy( gradient, 0.0 );

        downward.execute( upward.multipoles(),
                          Cabana::slice<Position>( particles ), potential,
                          gradient, /*compute_gradient=*/true, comm_plan );
    }
};

//---------------------------------------------------------------------------//
// The gate: locals() against the host reference, at EXPECT_DOUBLE_EQ.
//---------------------------------------------------------------------------//
template <class TEST_MS, class TEST_ES>
void testLocalsMatchHostReference( int num_particles_per_rank, int ncrit,
                                   int max_depth, double tolerance,
                                   int replication_depth )
{
    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    ContractFixture<TEST_MS, TEST_ES> fix( num_particles_per_rank, ncrit,
                                           max_depth, tolerance,
                                           replication_depth );

    constexpr int NC = Basis::num_components;
    const auto& cells = fix.builder.cells();
    const int n_cells = static_cast<int>( cells.size() );

    // ----------------------------------------------------------------------
    // The reference indexes cells by their rank-local index. That is only a
    // global identity because TreeBuilder builds the same tree on every rank:
    // refinement is driven by MPI_Allreduce'd per-cell global counts
    // (src/Canopy_TreeBuilder.hpp:743), so the cell vector agrees in content
    // and in order. Assert it rather than assume it — if it ever stops being
    // true, the MPI_Allreduce below would silently combine different cells.
    // ----------------------------------------------------------------------
    {
        int n_min = n_cells, n_max = n_cells;
        MPI_Allreduce( MPI_IN_PLACE, &n_min, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD );
        MPI_Allreduce( MPI_IN_PLACE, &n_max, 1, MPI_INT, MPI_MAX,
                       MPI_COMM_WORLD );
        ASSERT_EQ( n_min, n_max ) << "ranks hold different cell counts; the "
                                     "reference's global cell index is invalid";

        std::uint64_t h = hash_key_sequence( cells );
        std::uint64_t h_min = h, h_max = h;
        MPI_Allreduce( MPI_IN_PLACE, &h_min, 1, MPI_UINT64_T, MPI_MIN,
                       MPI_COMM_WORLD );
        MPI_Allreduce( MPI_IN_PLACE, &h_max, 1, MPI_UINT64_T, MPI_MAX,
                       MPI_COMM_WORLD );
        ASSERT_EQ( h_min, h_max )
            << "ranks hold different cell key sequences; the reference's "
               "global cell index is invalid";
    }

    // Key -> global (== rank-local) cell index, and the per-depth half-width.
    std::unordered_map<MortonKey, int> key_to_idx;
    key_to_idx.reserve( n_cells * 2 );
    int tree_max_depth = 0;
    for ( int i = 0; i < n_cells; i++ )
    {
        key_to_idx[cells[i].key] = i;
        tree_max_depth = std::max( tree_max_depth, cells[i].depth );
    }
    std::vector<double> half_width_at_depth( tree_max_depth + 1, 0.0 );
    for ( const auto& ci : cells )
        half_width_at_depth[ci.depth] = ci.half_width;

    // ----------------------------------------------------------------------
    // Multipoles, mirrored AFTER downward.execute(). That ordering matters:
    // exchange_multipoles_for_m2l writes the remote source multipoles this
    // rank needs into the same view, without accumulation
    // (src/Canopy_DownwardSweep.hpp:1497), so only after execute() does the
    // mirror hold every source the interaction lists name.
    // ----------------------------------------------------------------------
    auto h_M = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                    fix.upward.multipoles() );

    // ----------------------------------------------------------------------
    // D(a): the per-cell M2L delta, in the sweep's summation order.
    //
    // The communication plan puts interaction_lists[a] on exactly one rank
    // (src/Canopy_CommunicationPlan.hpp:477-486: a's owner, or rank 0 when a
    // is shared), so every entry of this array is written by at most one rank
    // and the Allreduce that follows adds exact zeros to it. That is why a
    // SUM over ranks here is bit-exact rather than a reassociation.
    // ----------------------------------------------------------------------
    std::vector<double> D( static_cast<std::size_t>( n_cells ) * NC, 0.0 );
    long long n_pairs_local = 0;
    std::size_t n_targets_local = 0;

    for ( const auto& kv : fix.comm_plan.m2l_plan().interaction_lists )
    {
        const MortonKey target_key = kv.first;
        auto tit = key_to_idx.find( target_key );
        if ( tit == key_to_idx.end() )
            continue;
        const int t_idx = tit->second;
        const int d_t = cells[t_idx].depth;
        n_targets_local++;
        n_pairs_local += static_cast<long long>( kv.second.size() );

        for ( int c = 0; c < NC; c++ )
        {
            double acc = 0.0;
            // As-is: this vector's order is the sweep's CSR walk order.
            for ( const auto& src : kv.second )
            {
                const int s_idx = src.second;
                const int d_s = cells[s_idx].depth;
                const int dd = d_s - d_t;
                const int max_d = ( d_s > d_t ) ? d_s : d_t;
                const double inv_unit_w = 1.0 / half_width_at_depth[max_d];
                const int ii = static_cast<int>( std::lround(
                    ( cells[s_idx].center[0] - cells[t_idx].center[0] ) *
                    inv_unit_w ) );
                const int jj = static_cast<int>( std::lround(
                    ( cells[s_idx].center[1] - cells[t_idx].center[1] ) *
                    inv_unit_w ) );
                const int kk = static_cast<int>( std::lround(
                    ( cells[s_idx].center[2] - cells[t_idx].center[2] ) *
                    inv_unit_w ) );

                Basis::m2l_accumulate(
                    acc, Basis::m2l_operator_entry( dd, ii, jj, kk ),
                    h_M( s_idx, 0, c ) );
            }
            D[static_cast<std::size_t>( t_idx ) * NC + c] = acc;
        }
    }

    if ( nprocs > 1 )
        MPI_Allreduce( MPI_IN_PLACE, D.data(), static_cast<int>( D.size() ),
                       MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

    // ----------------------------------------------------------------------
    // Telescope root-downward, in the sweep's own arithmetic. Parents are at a
    // strictly shallower depth than their children, so a depth-ascending pass
    // has every parent final before its children are read.
    //
    // TWO CASES, and they are NOT the same expression. Which one applies is
    // decided by ownership, and TreePartitioner marks a cell OWNER_SHARED
    // exactly when `depth <= replication_depth && !is_leaf`
    // (src/Canopy_TreePartitioner.hpp:498-502) — which is also
    // CommunicationPlan's shared-cell predicate
    // (src/Canopy_CommunicationPlan.hpp:698), so the two agree.
    //
    // NON-SHARED target. run_m2l_all() writes D(t) into a freshly zeroed
    // local before the depth loop; L2L at depth-1 then adds the parent's
    // local, either directly or — when the parent is owned by another rank —
    // through the accumulating L2L exchange. Either way:
    //
    //     L(t) = D(t) + L(parent)
    //
    // SHARED target. Its M2L runs inside the depth loop, behind the
    // snapshot / Allreduce barrier, and that round trip is NOT the identity
    // in floating point even at np=1:
    //
    //     a     = L(parent)                 snapshot, taken after L2L(depth-1)
    //     c     = a + D(t)                  m2l_post_cell's `+=`
    //     delta = c - a                     the Allreduce send buffer
    //     L(t)  = a + delta                 snapshot + summed delta
    //
    // `a + (fl(a + D) - a)` is not `fl(a + D)` in general — the subtraction
    // re-rounds — so a reference that wrote `D + L(parent)` for shared cells
    // would be wrong in the last bit, and at four ULP that could show up as a
    // failure indistinguishable from a real defect. This is why shared cells
    // get their own branch rather than being folded into the non-shared one.
    //
    // At np >= 2 only rank 0 computes a shared target's M2L, so every other
    // rank's send buffer holds `a - a`, an exact +0.0, and MPI_SUM of one
    // value and a set of exact zeros is that value whatever order the
    // reduction takes. The expression above is therefore the same at every
    // rank count, which is the point: shared cells are a property of the
    // TREE, not of the rank count — np=1 has them at depths 0, 1 and 2 here
    // — and np=1 is not insulated from this path.
    // ----------------------------------------------------------------------
    std::vector<int> by_depth( n_cells );
    for ( int i = 0; i < n_cells; i++ )
        by_depth[i] = i;
    std::stable_sort( by_depth.begin(), by_depth.end(),
                      [&]( int a, int b )
                      { return cells[a].depth < cells[b].depth; } );

    const auto& owner_map = fix.partitioner.cell_owner_map();
    auto owner_of_cell = [&]( MortonKey k )
    {
        auto it = owner_map.find( k );
        return ( it != owner_map.end() ) ? it->second : OWNER_SHARED;
    };

    std::vector<double> L_ref( static_cast<std::size_t>( n_cells ) * NC, 0.0 );
    int n_shared_cells = 0;
    for ( int i : by_depth )
    {
        auto pit = key_to_idx.find( parent_key( cells[i].key ) );
        const bool is_shared =
            ( owner_of_cell( cells[i].key ) == OWNER_SHARED );
        if ( is_shared )
            n_shared_cells++;

        for ( int c = 0; c < NC; c++ )
        {
            const double parent_val =
                ( pit != key_to_idx.end() )
                    ? L_ref[static_cast<std::size_t>( pit->second ) * NC + c]
                    : 0.0;
            const double Dv = D[static_cast<std::size_t>( i ) * NC + c];

            double out;
            if ( !is_shared )
            {
                out = Dv + parent_val;
            }
            else
            {
                const double a = parent_val;
                const double post_m2l = a + Dv;
                const double delta = post_m2l - a;
                out = a + delta;
            }
            L_ref[static_cast<std::size_t>( i ) * NC + c] = out;
        }
    }

    // ----------------------------------------------------------------------
    // Compare, over the cells this rank actually computes: the ones it owns
    // outright and the shared ones. A cell this rank neither owns nor shares
    // carries a partial value in _locals — L2L for a shared parent writes
    // into every child's slot on every rank — and no correctness claim rests
    // on it.
    // ----------------------------------------------------------------------
    auto h_L = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                    fix.downward.locals() );
    ASSERT_EQ( static_cast<int>( h_L.extent( 0 ) ), n_cells );
    ASSERT_EQ( static_cast<int>( h_L.extent( 1 ) ),
               Basis::num_coeffs_per_cell );
    ASSERT_EQ( static_cast<int>( h_L.extent( 2 ) ), NC );

    // The verdict is EXPECT_DOUBLE_EQ (4 ULP), as T6 specifies. Bit-identity
    // is the stronger claim the reference is built to support, so it is
    // counted separately and reported in the provenance line below rather
    // than asserted: a run that passes with not_bit_identical > 0 means the
    // exactness argument in this file's header has a hole, even though the
    // gate held. Only the first few non-identical slots get a full message,
    // so a systematic failure does not bury the log.
    int n_checked = 0;
    int n_not_bit_identical = 0;
    double max_abs_ref = 0.0;
    for ( int i = 0; i < n_cells; i++ )
    {
        const int owner = owner_of_cell( cells[i].key );
        if ( owner != rank && owner != OWNER_SHARED )
            continue;

        for ( int c = 0; c < NC; c++ )
        {
            const double ref = L_ref[static_cast<std::size_t>( i ) * NC + c];
            const double got = h_L( i, 0, c );
            max_abs_ref = std::max( max_abs_ref, std::abs( ref ) );
            n_checked++;

            if ( ref == got )
            {
                EXPECT_DOUBLE_EQ( got, ref );
                continue;
            }
            n_not_bit_identical++;
            if ( n_not_bit_identical <= 8 )
                EXPECT_DOUBLE_EQ( got, ref )
                    << "locals() disagrees with the host reference at cell "
                    << i << " (key " << cells[i].key << ", depth "
                    << cells[i].depth << ", owner " << owner
                    << ") component " << c;
            else
                EXPECT_DOUBLE_EQ( got, ref );
        }
    }

    // The reference must be non-trivial, or an all-zero locals() would pass.
    double global_max_ref = 0.0;
    MPI_Allreduce( &max_abs_ref, &global_max_ref, 1, MPI_DOUBLE, MPI_MAX,
                   MPI_COMM_WORLD );
    EXPECT_GT( global_max_ref, 0.0 )
        << "the host reference is identically zero, so the comparison proves "
           "nothing; the M2L interaction lists are probably empty";
    EXPECT_GT( n_checked, 0 ) << "no cell on this rank was checked";

    // Provenance for the job log. total_fallback_pair_count() is printed, not
    // asserted: MonopoleBasis::m2l_translate reconstructs the same integer key
    // and calls the same m2l_operator_entry as the fused path, so a pair that
    // routes to the fallback lands on the same bits and the reference does not
    // need to know. A non-zero count is still worth seeing.
    std::printf( "[far-field-contract] nprocs=%d rank=%d cells=%d "
                 "targets=%zu ilist_pairs=%lld m2l_pairs=%lld "
                 "fallback_pairs=%lld shared_cells=%d checked=%d "
                 "not_bit_identical=%d max_abs_ref=%.17e\n",
                 nprocs, rank, n_cells, n_targets_local, n_pairs_local,
                 fix.downward.total_m2l_pair_count(),
                 fix.downward.total_fallback_pair_count(), n_shared_cells,
                 n_checked, n_not_bit_identical, global_max_ref );
    std::fflush( stdout );
}

//---------------------------------------------------------------------------//
// L2P: the potential is the leaf's local, and the gradient is identically
// zero. Exact, and it closes the pipeline — locals() being right is worthless
// if nothing reads it.
//---------------------------------------------------------------------------//
template <class TEST_MS, class TEST_ES>
void testL2PReturnsTheLocal( int num_particles_per_rank, int ncrit,
                             int max_depth, double tolerance,
                             int replication_depth )
{
    ContractFixture<TEST_MS, TEST_ES> fix( num_particles_per_rank, ncrit,
                                           max_depth, tolerance,
                                           replication_depth );

    constexpr int NC = Basis::num_components;

    auto h_L = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                    fix.downward.locals() );
    auto h_phi = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                      fix.potential );
    auto h_grad = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                       fix.gradient );
    auto h_pcidx = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), fix.upward.particle_cell_idx() );

    // run_l2p skips a particle whose cell index is negative or whose cell is
    // not a leaf (src/Canopy_DownwardSweep.hpp:1985-1991), leaving its
    // potential at the zero the caller wrote. Skip the same particles here
    // rather than asserting against a value the sweep never produced.
    const auto& cells = fix.builder.cells();
    int n_checked = 0;
    for ( int p = 0; p < fix.num_local; p++ )
    {
        const int cidx = h_pcidx( p );
        if ( cidx < 0 || cidx >= static_cast<int>( cells.size() ) )
            continue;
        if ( !cells[cidx].is_leaf )
            continue;
        n_checked++;
        for ( int c = 0; c < NC; c++ )
        {
            EXPECT_DOUBLE_EQ( h_phi( p, c ), h_L( cidx, 0, c ) )
                << "L2P did not return the leaf's local at particle " << p
                << " component " << c;
            for ( int d = 0; d < 3; d++ )
                EXPECT_EQ( h_grad( p, c, d ), 0.0 )
                    << "MonopoleBasis L2P must produce a zero gradient; "
                       "particle "
                    << p << " component " << c << " dim " << d;
        }
    }
    EXPECT_GT( n_checked, 0 ) << "no local particle was mapped to a leaf";
}

//---------------------------------------------------------------------------//
// THE LEVEL REACHES THE KEY — T7's positive assertion.
//
// The sweep's M2L key is (max_d, dd, ii, jj, kk) and is reduced by
// KernelType::canonicalize_key before it is hashed. LaplaceKernel zeroes
// max_d there, MonopoleBasis keeps it. Zeroing is the branch the Laplace-solve
// gate covers, by pinning that its key set did not move; the branch that KEEPS
// the level needs a positive measurement, or a sweep that silently dropped
// max_d on the floor would look identical to one that honoured it.
//
// The measurement is m2l_n_unique_ops() on the SAME TREE under two bases that
// differ in exactly one member: MonopoleBasis, which keeps max_d, against the
// derived basis below, which zeroes it. Keeping the level can only split keys,
// never merge them, so the level-carrying basis must realize STRICTLY MORE
// distinct keys — and does so only if max_d actually survived into the hash.
//
// np=1 ONLY. Above one rank the tree is partitioned by Zoltan2 multijagged,
// which src/Canopy_TreePartitioner.hpp:417-419 documents as non-deterministic,
// so each rank's realized key set moves between runs. The two sweeps here are
// driven over one fixture's tree, partition and communication plan, so even at
// np > 1 they would see the same cut — but the count they produce would not be
// reproducible, and a strict inequality on an irreproducible pair of counts is
// not worth asserting. np=1 is where the key set is reproducible.
//
// THIS TEST NEEDS ITS OWN, DEEPER TREE, and that is a measured requirement
// rather than a precaution. At the Basic configuration (ncrit 32) the np=1
// tree is 89 cells — 1 + 8 + 64 at depths 0-2 and just 16 at depth 3, i.e.
// two parents' worth of children — and BOTH bases realize exactly 468 keys.
// The strict inequality genuinely does not hold there, and the reason is a
// property of the dual-tree traversal rather than of T7: a same-depth pair is
// emitted only when its PARENT pair failed the MAC, so same-depth offsets sit
// in a narrow shell, and with only 16 cells at depth 3 (whose two parents are
// not a MAC-failing pair) depth 3 contributes no same-depth pairs at all.
// Every depth-3 pair is then cross-depth, its dd and offset distinguish it
// from every depth-2 pair, and (dd, ii, jj, kk) determines max_d by accident.
// Dropping ncrit to 4 puts real populations at depths 3 and 4, whose
// same-depth offset shells overlap depth 2's, and the collision appears. The
// key-multiplicity diagnostic below reports it, so a later configuration
// change that quietly flattens the tree again is visible in the log rather
// than only in a failure.
//
// This costs nothing in operator VALUES, which is why it can be asserted
// alongside a bit-exact gate: m2l_operator_entry ignores max_d, so the extra
// columns a level-carrying key produces are duplicates of one another and
// locals() is unchanged. The level is observable in the key COUNT alone.
//---------------------------------------------------------------------------//

// MonopoleBasis with the key contract flipped to LaplaceKernel's answer, and
// nothing else changed. Derivation is deliberate, as in the negative block
// below: the two bases are identical in every trait and every operator, so the
// only thing the count below can be measuring is canonicalize_key.
struct LevelBlindBasis : public Basis
{
    static constexpr bool key_needs_level = false;

    template <class Key>
    static Key canonicalize_key( Key k )
    {
        k.max_d = 0;
        return k;
    }
};

// A stand-in for the sweep's nested M2LKey, which a test cannot name without
// naming a DownwardSweep instantiation. canonicalize_key is a template on the
// key type precisely so that any struct with these five members will do.
struct ProbeKey
{
    int max_d;
    int dd;
    int ii;
    int jj;
    int kk;
};

// key_needs_level and canonicalize_key state the same fact twice, once for a
// reader (and for T8's byte accounting) and once for the classify pass. T7
// requires them to agree; assert it rather than trusting the declaration.
template <class B>
void expectKeyTraitsAgree( const char* basis_name )
{
    const ProbeKey a{ 3, -1, 2, -3, 4 };
    const ProbeKey b{ 5, -1, 2, -3, 4 };
    const ProbeKey ca = B::template canonicalize_key<ProbeKey>( a );
    const ProbeKey cb = B::template canonicalize_key<ProbeKey>( b );

    // The offset and depth-difference fields must survive canonicalization
    // under either answer — they are what the operator builder is handed.
    EXPECT_EQ( a.dd, ca.dd ) << basis_name << ": canonicalize_key altered dd";
    EXPECT_EQ( a.ii, ca.ii ) << basis_name << ": canonicalize_key altered ii";
    EXPECT_EQ( a.jj, ca.jj ) << basis_name << ": canonicalize_key altered jj";
    EXPECT_EQ( a.kk, ca.kk ) << basis_name << ": canonicalize_key altered kk";

    if ( B::key_needs_level )
    {
        EXPECT_EQ( a.max_d, ca.max_d )
            << basis_name
            << ": key_needs_level is true but canonicalize_key did not "
               "preserve max_d";
        EXPECT_NE( ca.max_d, cb.max_d )
            << basis_name
            << ": key_needs_level is true but canonicalize_key maps two "
               "different levels onto the same key, so the level cannot "
               "reach the operator table";
    }
    else
    {
        EXPECT_EQ( ca.max_d, cb.max_d )
            << basis_name
            << ": key_needs_level is false but canonicalize_key lets two "
               "different levels through as distinct keys, so the operator "
               "table would hold one column per (level, offset)";
    }
}

template <class TEST_MS, class TEST_ES>
void testLevelReachesTheKey( int num_particles_per_rank, int ncrit,
                             int max_depth, double tolerance,
                             int replication_depth )
{
    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    expectKeyTraitsAgree<Basis>( "MonopoleBasis" );
    expectKeyTraitsAgree<LevelBlindBasis>( "LevelBlindBasis" );
    EXPECT_TRUE( Basis::key_needs_level )
        << "MonopoleBasis must keep the level, or this test compares a basis "
           "against itself";
    EXPECT_FALSE( LevelBlindBasis::key_needs_level );

    if ( nprocs != 1 )
        GTEST_SKIP() << "the realized key set is reproducible only at np=1; "
                        "above one rank it moves with the multijagged cut";

    // The level-carrying run: the ordinary fixture, unchanged.
    ContractFixture<TEST_MS, TEST_ES> fix( num_particles_per_rank, ncrit,
                                           max_depth, tolerance,
                                           replication_depth );

    // The level-blind run, over THIS FIXTURE'S tree, partition and
    // communication plan — not over a second fixture's. All three are
    // basis-independent and are taken by const reference, so the two sweeps
    // below see the same cells, the same ownership and the same interaction
    // lists, and the only difference between the two runs is the basis.
    UpwardSweep<TEST_MS, TEST_ES, LevelBlindBasis> upward_blind(
        MPI_COMM_WORLD );
    upward_blind.setup( fix.builder.cells(), fix.partitioner.cell_owner_map(),
                        fix.builder.particle_keys(), fix.num_local );
    upward_blind.execute( Cabana::slice<Charge>( fix.particles ),
                          Cabana::slice<Position>( fix.particles ),
                          fix.comm_plan );

    DownwardSweep<TEST_MS, TEST_ES, LevelBlindBasis> downward_blind(
        MPI_COMM_WORLD );
    downward_blind.setup( upward_blind, fix.num_local );

    auto potential_blind = downward_blind.allocate_potential( fix.num_local );
    auto gradient_blind = downward_blind.allocate_gradient( fix.num_local );
    Kokkos::deep_copy( potential_blind, 0.0 );
    Kokkos::deep_copy( gradient_blind, 0.0 );
    downward_blind.execute( upward_blind.multipoles(),
                            Cabana::slice<Position>( fix.particles ),
                            potential_blind, gradient_blind,
                            /*compute_gradient=*/true, fix.comm_plan );

    const int n_with_level = fix.downward.m2l_n_unique_ops();
    const int n_without_level = downward_blind.m2l_n_unique_ops();

    EXPECT_GT( n_without_level, 0 )
        << "the level-blind run realized no operators at all, so the "
           "comparison below proves nothing";

    // How many DISTINCT levels the busiest offset key is realized at, and how
    // many offset keys are realized at more than one level. This is the
    // quantity the strict inequality is really about: n_with_level exceeds
    // n_without_level by exactly the number of extra (level, offset) pairs,
    // so a max multiplicity of 1 means no offset was ever seen at two levels
    // and the tree — not the key — is what made the counts equal.
    std::map<std::array<int, 4>, std::set<int>> levels_per_offset;
    for ( const auto& k : fix.downward.m2l_realized_keys() )
        levels_per_offset[{ k.dd, k.ii, k.jj, k.kk }].insert( k.max_d );

    std::size_t max_levels_at_one_offset = 0;
    std::size_t n_multi_level_offsets = 0;
    for ( const auto& kv : levels_per_offset )
    {
        max_levels_at_one_offset =
            std::max( max_levels_at_one_offset, kv.second.size() );
        if ( kv.second.size() > 1 )
            n_multi_level_offsets++;
    }

    // The level-blind run must realize exactly the projection of the
    // level-carrying run's key set onto (dd, ii, jj, kk). Both sweeps see the
    // same tree, the same ownership and the same interaction lists, so any
    // other relationship between the two counts means the two runs did not in
    // fact classify the same pairs and the comparison below is meaningless.
    EXPECT_EQ( levels_per_offset.size(),
               static_cast<std::size_t>( n_without_level ) )
        << "the level-blind run did not realize the projection of the "
           "level-carrying run's key set, so the two runs did not classify "
           "the same pairs";

    // The tree must actually pose the question. A tree in which no offset is
    // reached at two different depths cannot distinguish a key that carries
    // the level from one that drops it, whatever the sweep does.
    ASSERT_GT( max_levels_at_one_offset, 1u )
        << "no offset key is realized at more than one level on this tree, so "
           "a level-carrying key and a level-blind one are indistinguishable "
           "here by construction and the assertion below would prove nothing. "
           "The configuration has gone flat — see this test's comment block";

    // The assertion T7's exit criterion names.
    EXPECT_GT( n_with_level, n_without_level )
        << "MonopoleBasis keeps max_d in its key and the level-blind basis "
           "zeroes it, yet both realized the same number of distinct keys ("
        << n_with_level << "), even though " << n_multi_level_offsets
        << " offset keys occur at more than one level. The level is being "
           "dropped somewhere between the classify pass and the key map";

    // And the direct form of the same fact: the level-carrying run must
    // realize keys whose max_d is non-zero, and the level-blind run must
    // realize none. EXPECT_GT above can be satisfied by an unrelated key-set
    // difference; this cannot.
    int n_level_carrying = 0;
    for ( const auto& k : fix.downward.m2l_realized_keys() )
        if ( k.max_d != 0 )
            n_level_carrying++;
    EXPECT_GT( n_level_carrying, 0 )
        << "every key MonopoleBasis realized has max_d == 0, so its identity "
           "canonicalize_key never saw a level";

    int n_blind_level_carrying = 0;
    for ( const auto& k : downward_blind.m2l_realized_keys() )
        if ( k.max_d != 0 )
            n_blind_level_carrying++;
    EXPECT_EQ( 0, n_blind_level_carrying )
        << n_blind_level_carrying
        << " keys survived the level-blind canonicalize_key with a non-zero "
           "max_d, so canonicalization is not reaching the hash";

    // Fallback counts stay the discriminator (R4). m2l_key_dd_max became a
    // basis trait in T7, and a mistake there pushes pairs onto the per-pair
    // path rather than producing a wrong answer.
    EXPECT_EQ( 0, fix.downward.total_fallback_pair_count() );
    EXPECT_EQ( 0, downward_blind.total_fallback_pair_count() );

    std::printf( "[far-field-contract] level-reaches-key nprocs=%d rank=%d "
                 "cells=%d n_unique_ops_with_level=%d n_unique_ops_without=%d "
                 "level_carrying_keys=%d multi_level_offsets=%zu "
                 "max_levels_at_one_offset=%zu fallback_with=%lld "
                 "fallback_without=%lld\n",
                 nprocs, rank,
                 static_cast<int>( fix.builder.cells().size() ), n_with_level,
                 n_without_level, n_level_carrying, n_multi_level_offsets,
                 max_levels_at_one_offset,
                 fix.downward.total_fallback_pair_count(),
                 downward_blind.total_fallback_pair_count() );
    std::fflush( stdout );
}

} // namespace FarFieldContractTest

//---------------------------------------------------------------------------//
// Registered at two problem sizes, mirroring tests/tstDownwardSweep.hpp's
// Basic/Small pairing: the larger one gives a deeper tree and a fuller
// interaction list, the smaller one a tree shallow enough that the shared
// depths dominate.
//---------------------------------------------------------------------------//

TEST( FarFieldContract, localsMatchHostReferenceBasic )
{
    FarFieldContractTest::testLocalsMatchHostReference<TEST_MEMSPACE,
                                                       TEST_EXECSPACE>(
        /*num_particles_per_rank=*/1000, /*ncrit=*/32, /*max_depth=*/6,
        /*tolerance=*/0.1, /*replication_depth=*/2 );
}

TEST( FarFieldContract, localsMatchHostReferenceSmall )
{
    FarFieldContractTest::testLocalsMatchHostReference<TEST_MEMSPACE,
                                                       TEST_EXECSPACE>(
        /*num_particles_per_rank=*/200, /*ncrit=*/16, /*max_depth=*/4,
        /*tolerance=*/0.1, /*replication_depth=*/1 );
}

TEST( FarFieldContract, l2pReturnsTheLocal )
{
    FarFieldContractTest::testL2PReturnsTheLocal<TEST_MEMSPACE,
                                                 TEST_EXECSPACE>(
        /*num_particles_per_rank=*/1000, /*ncrit=*/32, /*max_depth=*/6,
        /*tolerance=*/0.1, /*replication_depth=*/2 );
}

// ncrit 4, NOT the Basic configuration's 32. The question this test asks is
// whether one offset can be realized at two different levels, and at ncrit 32
// the np=1 tree is too shallow for that to happen at all — measured, see the
// test's comment block. ncrit 4 puts real populations at depths 3 and 4 whose
// same-depth offset shells overlap depth 2's. The body asserts the trait
// agreement at every rank count and skips the key-count comparison above np=1.
TEST( FarFieldContract, levelReachesTheKey )
{
    FarFieldContractTest::testLevelReachesTheKey<TEST_MEMSPACE,
                                                 TEST_EXECSPACE>(
        /*num_particles_per_rank=*/1000, /*ncrit=*/4, /*max_depth=*/6,
        /*tolerance=*/0.1, /*replication_depth=*/2 );
}

//---------------------------------------------------------------------------//
// PERMANENT NEGATIVE TEST — this block must not compile.
//
// Two bases, each MonopoleBasis with one or two traits changed, and each
// instantiated on both sweeps. Between them they hit all four guards; see the
// per-case comments below for why one basis is not enough.
//
// WHAT IS BEING TESTED IS THE GUARD THE SWEEPS CARRY, not MonopoleBasis's own
// assert. The four asserts are:
//
//   src/Canopy_UpwardSweep.hpp:73-78     "UpwardSweep: the basis's coeff_type
//                                         is not scalars_per_coeff contiguous
//                                         component_scalar_type, ..."
//   src/Canopy_UpwardSweep.hpp:84-92     "UpwardSweep: the basis's coefficient
//                                         traits disagree with
//                                         detail::coeff_traits ..."
//   src/Canopy_DownwardSweep.hpp:117-122 the DownwardSweep sizeof analogue
//   src/Canopy_DownwardSweep.hpp:128-136 the DownwardSweep traits analogue
//
// and all four are at class scope, so naming a sweep type completely — which
// the sizeof() calls below do — is enough to fire them. An inconsistent basis
// cannot instantiate a sweep at all.
//
// DERIVATION IS DELIBERATE, not a shortcut. Because the ONLY thing wrong with
// each basis below is its one or two changed traits, deleting the four
// asserts would make this block COMPILE — the sweeps would instantiate
// happily and then mis-size an MPI count at runtime. A hand-rolled minimal
// bad basis would keep failing on missing members after the asserts were
// gone, and so would not detect their deletion. That property is the reason
// this block is committed rather than applied and reverted: nothing is
// perturbed in MonopoleBasis, and a later session can re-run it as written.
//
// See this file's header comment for the by-hand build command.
//---------------------------------------------------------------------------//
#ifdef CANOPY_TEST_EXPECT_COMPILE_FAILURE
namespace FarFieldContractTest
{

// Case A — breaks the sizeof relation. coeff_type stays double while
// scalars_per_coeff says 2, so sizeof(double) == 8 but
// scalars_per_coeff * sizeof(component_scalar_type) == 16. Targets
// Canopy_UpwardSweep.hpp:73-78 and Canopy_DownwardSweep.hpp:117-122.
struct InconsistentCoeffBasis
    : public CanopyTest::MonopoleBasis<double, BASIS_ORDER, BASIS_NCOMPS>
{
    // coeff_type, component_scalar_type and everything else are inherited.
    // This one line is the whole perturbation.
    static constexpr int scalars_per_coeff = 2;
};

// Case B — satisfies the sizeof relation and breaks the coeff_traits
// cross-check. sizeof(double) == 8 == 2 * sizeof(float), so the sizeof assert
// passes, but Canopy::detail::coeff_traits<double>::component_scalar_type is
// double and not float. Targets Canopy_UpwardSweep.hpp:84-92 and
// Canopy_DownwardSweep.hpp:128-136.
//
// CASE B IS NOT REDUNDANT. Clang reports only the FIRST failing class-scope
// static_assert per class instantiation, so Case A alone produces two
// diagnostics (one per sweep) and never reaches the traits cross-check —
// which means Case A alone would not notice if the two traits asserts were
// deleted. Both cases together cover all four guards.
struct InconsistentComponentBasis
    : public CanopyTest::MonopoleBasis<double, BASIS_ORDER, BASIS_NCOMPS>
{
    using component_scalar_type = float;
    static constexpr int scalars_per_coeff = 2;
};

using InconsistentUpwardA =
    UpwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, InconsistentCoeffBasis>;
using InconsistentDownwardA =
    DownwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, InconsistentCoeffBasis>;
using InconsistentUpwardB =
    UpwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, InconsistentComponentBasis>;
using InconsistentDownwardB =
    DownwardSweep<TEST_MEMSPACE, TEST_EXECSPACE, InconsistentComponentBasis>;

// sizeof() requires a complete type, which instantiates the class body and
// therefore the static_asserts in it.
static_assert( sizeof( InconsistentUpwardA ) > 0,
               "CANOPY_TEST_EXPECT_COMPILE_FAILURE: UpwardSweep accepted a "
               "basis whose coeff_type is not scalars_per_coeff contiguous "
               "component_scalar_type. Its sizeof static_assert "
               "(Canopy_UpwardSweep.hpp:73-78) has been deleted." );
static_assert( sizeof( InconsistentDownwardA ) > 0,
               "CANOPY_TEST_EXPECT_COMPILE_FAILURE: DownwardSweep accepted a "
               "basis whose coeff_type is not scalars_per_coeff contiguous "
               "component_scalar_type. Its sizeof static_assert "
               "(Canopy_DownwardSweep.hpp:117-122) has been deleted." );
static_assert( sizeof( InconsistentUpwardB ) > 0,
               "CANOPY_TEST_EXPECT_COMPILE_FAILURE: UpwardSweep accepted a "
               "basis whose traits disagree with detail::coeff_traits. Its "
               "coeff_traits cross-check static_assert "
               "(Canopy_UpwardSweep.hpp:84-92) has been deleted." );
static_assert( sizeof( InconsistentDownwardB ) > 0,
               "CANOPY_TEST_EXPECT_COMPILE_FAILURE: DownwardSweep accepted a "
               "basis whose traits disagree with detail::coeff_traits. Its "
               "coeff_traits cross-check static_assert "
               "(Canopy_DownwardSweep.hpp:128-136) has been deleted." );

} // namespace FarFieldContractTest
#endif // CANOPY_TEST_EXPECT_COMPILE_FAILURE

//---------------------------------------------------------------------------//

} // end namespace Test
