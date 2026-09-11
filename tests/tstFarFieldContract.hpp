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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
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
