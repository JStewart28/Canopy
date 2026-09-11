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
// tstLaplaceSolve — the solve-level gate for the solid-harmonic far field.
//
// One frozen FMM configuration is driven for 12 timesteps and the state
// after the 12th solve is gated three ways, split by rank count because the
// leaf partition is not reproducible run-to-run above two ranks
// (TreePartitioner::partition_leaves uses Zoltan2 multijagged; see
// src/Canopy_TreePartitioner.hpp:417-419 and README "Known Issues"):
//
//   bitForBitArtifacts  np 1-2  four internal artifacts, on their *bit
//                               patterns*, against committed reference data
//   crossRankAgreement  np 2-6  the np=k field reproduces the committed np=1
//                               field to floating-point reassociation
//   matchesDirectSum    np 1-6  the field matches a brute-force O(N^2) sum at
//                               the accuracy the method delivers
//
// The four bit-for-bit artifacts are:
//
//   locals()           — hash of every (real, imag) bit pattern, plus extents
//   M2L operator table — hash over the realized key columns, plus extents
//   A_{n,m} table      — full bit patterns, one uint64_t per entry
//   realized key list  — hash of the sorted keys — and n_unique_ops, full
//
// The key-list hash covers (dd, ii, jj, kk) and NOT the max_d field the key
// also carries (T7). Feeding a fifth field would change every committed hash
// in order to hash a constant zero; what max_d is worth asserting about is
// asserted directly instead — bitForBitArtifacts requires max_d == 0 on every
// realized key, which pins this basis's canonicalize_key rather than pinning
// a hash of its output. See collect_keys.
//
// The particle set is a *global* set of 600 from seed 1234 + P, generated
// identically on every rank and sliced contiguously, so the same physics
// problem is solved at every rank count and the np=k-versus-np=1 comparison
// is definable at all. Particles are paired across rank counts by GlobalId:
// migration scrambles the local ordering and the solve moves the particles,
// so position is not a key.
//
// The purpose of this test is to attribute a bitwise difference to exactly
// one artifact across a refactor that is supposed to change no bits.
// tstMultiSolve owns accuracy; matchesDirectSum is carried here only because
// crossRankAgreement compares the solve against itself and would pass a
// uniformly wrong answer at every rank count.
//
// Regenerating the reference data
// -------------------------------
// Set CANOPY_LAPLACE_SOLVE_REGENERATE to a directory. bitForBitArtifacts
// then writes its record to <dir>/laplace_solve_np<N>_rank<R>.part and skips
// the comparison; crossRankAgreement and matchesDirectSum skip outright,
// having nothing to compare against yet. Concatenate the three parts —
// (1,0), (2,0), (2,1) — into tests/data/laplace_solve_P6.txt under the
// existing header. With the variable unset — the default, and what CTest
// runs — the tests always compare and never write reference data, so a later
// change cannot silently re-baseline itself.
// ===========================================================================

#include "Canopy_Helpers.hpp"
#include "Canopy_Solver.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace Test
{
//---------------------------------------------------------------------------//

using namespace Canopy;

namespace LaplaceSolveTest
{

// ---------------------------------------------------------------------------
// The frozen configuration. Do not change any of these: the committed
// reference data is only meaningful for this exact configuration, and
// changing one of them means regenerating all of it.
//
// LS_NUM_PARTICLES is the GLOBAL total, not a per-rank count. A per-rank
// count would make the total — and therefore the physics problem — a
// function of the rank count, and crossRankAgreement would have nothing to
// compare. Every rank count from 1 to 6 divides 600 exactly.
//
// LS_DT is small deliberately. The np=k field differs from the np=1 field by
// reassociation at each step, that difference enters the velocity update, and
// the perturbed positions feed the next step. A large dt compounds it until
// the cross-rank deviation measures trajectory divergence instead of
// summation order, which is the one quantity crossRankAgreement bounds.
//
// LS_NUM_STEPS was CHOSEN BY MEASUREMENT and is now frozen with the rest. A
// 50-step trace of this exact configuration (see the per-step table in
// tasks/abstract-solver-backend-progress-log.md, section T1) was read against
// three degeneracy tests — n_unique_ops below half its step-0 value, the
// global position range more than 50% wider than the initial bounding box in
// any dimension, or the maximum gradient magnitude above 100x its step-0
// value. The first degenerate step is step 18, where max |grad phi| reaches
// 3.81e+06 against 8.60e+03 at step 0; nothing before it trips any test.
// LS_NUM_STEPS is two thirds of that, rounded down, so a later task that
// perturbs the trajectory is not sitting on a cliff edge. Raising it back
// toward 50 drives the tree to zero realized M2L operators and makes every
// check here pass vacuously.
// ---------------------------------------------------------------------------
static constexpr int LS_P = 6;
static constexpr int LS_NCOMPS = 1;
static constexpr int LS_NUM_PARTICLES = 600; // global total, sliced per rank
static constexpr double LS_MAC_THETA = 0.5;
static constexpr int LS_NCRIT = 16;
static constexpr int LS_MAX_DEPTH = 6;
static constexpr int LS_NUM_STEPS = 12;
static constexpr double LS_DT = 1.0e-4;
static constexpr double LS_DRIFT_MULTIPLIER = 1.0;

// AoSoA member indices. Velocity and GlobalId exist for the time loop:
// GlobalId is what pairs a particle across rank counts after migration has
// scrambled the local ordering.
enum FieldIdx
{
    Position = 0,
    Charge = 1,
    Velocity = 2,
    GlobalId = 3
};

// ---------------------------------------------------------------------------
// MEASURED, THEN PINNED. Both tolerances were measured on unmodified code at
// every rank count from 1 to 6, against the committed reference data, at the
// frozen configuration above (600 particles, num_steps = 12, charges on
// [0.5, 1.5], dt = 1.0e-4, softening = 0.0, migrate between steps) on the
// SERIAL backend. Every rank count had a live far field at the last solve:
// 103 cells, n_unique_ops 686 at np=1 and 111-386 per rank at np 2-6, with
// total_fallback_pair_count() 0 throughout. The full per-rank-count tables
// are in tasks/abstract-solver-backend-progress-log.md, section T1.
//
// LS_CROSS_RANK_TOL is 100x the worst measured cross-rank deviation, which is
// 5.5987399483706545e-12 (np=4, gradient). The potential deviations run
// 9.6e-14 to 1.1e-12 and the gradient deviations 6.4e-13 to 5.6e-12 across
// np 2-6 — reassociation level, three orders of magnitude below the 1.0e-9
// threshold above which a deviation would no longer be attributable to
// summation order (see risk R8). Do not raise this to accommodate a failure:
// re-measure at LS_NUM_STEPS = 1 and report.
//
// LS_DIRECT_SUM_TOL is 3x the worst measured direct-sum deviation, which is
// 3.2093610363952985e-07 (np=2, potential). The gradient deviations are
// 4.24e-08 at every rank count and the potential deviations agree to ten
// digits across all six, as they should: the direct-sum error is truncation,
// not partitioning.
//
// The direct-sum check is a truncation bound in any case: the solid-harmonic
// far-field error goes as theta^(P+1) = 0.5^7 ~ 8e-3 at this configuration,
// so it cannot be tightened toward 1e-10 without either raising P (~15 GB of
// operator table per rank) or lowering theta until no pair is MAC-admissible
// and the far field is never evaluated. The tight gate here is cross-rank
// agreement, which is bounded by reassociation rather than by truncation.
// ---------------------------------------------------------------------------
static constexpr double LS_CROSS_RANK_TOL = 5.6e-10;
static constexpr double LS_DIRECT_SUM_TOL = 9.63e-07;

// R8's stop-and-report threshold. Above this, the cross-rank deviation is no
// longer attributable to reassociation and the tolerance must not be raised
// to accommodate it.
static constexpr double LS_CROSS_RANK_R8_THRESHOLD = 1.0e-9;

// ---------------------------------------------------------------------------
// The operator-table byte-budget check (testOpTableByteBudget). NOT part of
// the frozen configuration: these two constants configure a SECOND solve of
// the same problem, run beside the default-budget one in the same process,
// and nothing about them reaches tests/data/laplace_solve_P6.txt.
//
// LS_BUDGET_KEYS is how many operator columns the tight-budget solve is
// allowed; the budget handed to it is that many times the basis's
// bytes_per_key, so the test states the thing it means (a column count) while
// exercising the path that matters (the byte budget dividing by
// bytes_per_key). 64 is well under the 111-686 keys every rank realizes at
// the frozen configuration, so every rank overflows and
// total_fallback_pair_count() is positive everywhere.
//
// A LARGER CAP DOES NOT WORK AT EVERY RANK COUNT, and this is measured rather
// than assumed: at 256 columns, rank 1 at np=3 realizes only 204 keys,
// overflows nothing, and the anti-vacuity assertion below fires (flux job
// f3XiTuYB2mao). Raise this only after checking the per-rank n_unique_ops
// table in tasks/abstract-solver-backend-progress-log.md, section T8.
//
// LS_BUDGET_POTENTIAL_TOL is 5e-2, the same relative bound
// tests/tstMultiSolve.hpp:929 puts on the potential, and it is a bound on a
// DIFFERENT-ARITHMETIC comparison rather than on an error: the overflowing
// pairs are evaluated by the basis's per-pair m2l_translate instead of out of
// the operator table, which is the same mathematics reassociated. The
// measured deviation is far below this; the tolerance is loose deliberately,
// because a tight one here would be pinning the difference between two
// summation orders and would fail for reasons that are not defects.
// ---------------------------------------------------------------------------
static constexpr int LS_BUDGET_KEYS = 64;
static constexpr double LS_BUDGET_POTENTIAL_TOL = 5.0e-2;

static const char* const LS_DATA_FILE =
    CANOPY_TEST_DATA_DIR "/laplace_solve_P6.txt";

static const char* const LS_REGENERATE_ENV = "CANOPY_LAPLACE_SOLVE_REGENERATE";

// ---------------------------------------------------------------------------
// FNV-1a, 64-bit. Modeled on DownwardSweep::M2LKeyHash so the committed
// hashes depend on nothing outside this repository — not a library version,
// not a standard-library implementation. Multi-byte values are fed
// big-endian so a hash means the same thing on any machine.
// ---------------------------------------------------------------------------
struct Fnv1a64
{
    std::uint64_t h = 1469598103934665603ull;

    void byte( unsigned char b )
    {
        h ^= static_cast<std::uint64_t>( b );
        h *= 1099511628211ull;
    }
    void u64( std::uint64_t v )
    {
        for ( int i = 7; i >= 0; --i )
            byte( static_cast<unsigned char>( ( v >> ( i * 8 ) ) & 0xffu ) );
    }
    void i32( std::int32_t v )
    {
        u64( static_cast<std::uint64_t>( static_cast<std::uint32_t>( v ) ) );
    }
};

// The bit pattern of a double, as a uint64_t. memcpy, not a reinterpret
// cast, and never a comparison on the double itself: EXPECT_DOUBLE_EQ has a
// tolerance and NaN != NaN, so neither can serve as a bitwise gate.
inline std::uint64_t bits_of( double x )
{
    std::uint64_t u;
    std::memcpy( &u, &x, sizeof u );
    return u;
}

// The inverse, for reading the committed np=1 field record back as values.
inline double double_of( std::uint64_t u )
{
    double x;
    std::memcpy( &x, &u, sizeof x );
    return x;
}

inline std::string hex64( std::uint64_t v )
{
    std::ostringstream os;
    os << "0x" << std::hex << std::setw( 16 ) << std::setfill( '0' ) << v;
    return os.str();
}

// Total order on M2L keys, so the key-list hash is a function of the key
// *set* and not of the order the classify pass discovered them in. The
// column order of the table is separately covered by the table hash.
//
// max_d leads. For the solid-harmonic basis canonicalize_key zeroes it, so
// it is constant across every realized key and this changes no ordering the
// committed data was generated under; it is included so the comparator is a
// total order on the whole key rather than on four of its five fields.
template <class Key>
inline bool key_less( const Key& a, const Key& b )
{
    if ( a.max_d != b.max_d )
        return a.max_d < b.max_d;
    if ( a.dd != b.dd )
        return a.dd < b.dd;
    if ( a.ii != b.ii )
        return a.ii < b.ii;
    if ( a.jj != b.jj )
        return a.jj < b.jj;
    return a.kk < b.kk;
}

// ---------------------------------------------------------------------------
// One bit-for-bit record: the four artifacts for one (nprocs, rank), taken
// from the 12th solve.
// ---------------------------------------------------------------------------
struct BitRecord
{
    int nprocs = -1;
    int rank = -1;

    std::size_t locals_ext[3] = { 0, 0, 0 };
    std::uint64_t locals_hash = 0;

    std::size_t optab_ext[3] = { 0, 0, 0 };
    std::uint64_t optab_hash = 0;

    int n_unique_ops = 0;
    std::uint64_t keys_hash = 0;

    long long fallback_pairs = 0;

    std::vector<std::uint64_t> a_bits;
};

// The np=1 field after the 12th solve: potential and gradient bit patterns
// in canonical GlobalId order. The np=1 run is bit-reproducible, which is
// what makes committing its output meaningful.
struct FieldRecord
{
    int n = 0;
    std::vector<std::uint64_t> pot_bits;  // n
    std::vector<std::uint64_t> grad_bits; // 3 * n
};

// Everything the committed file holds.
struct ReferenceData
{
    std::map<std::pair<int, int>, BitRecord> bits;
    bool has_initial = false;
    std::uint64_t initial_hash = 0;
    bool has_field = false;
    FieldRecord field;
};

// Serialize one bit-for-bit record in the committed file's format. The
// format is documented in the header of tests/data/laplace_solve_P6.txt.
inline std::string serialize_bits( const BitRecord& r )
{
    std::ostringstream os;
    os << "set " << r.nprocs << " " << r.rank << "\n";
    os << "locals " << r.locals_ext[0] << " " << r.locals_ext[1] << " "
       << r.locals_ext[2] << " " << hex64( r.locals_hash ) << "\n";
    os << "optab " << r.optab_ext[0] << " " << r.optab_ext[1] << " "
       << r.optab_ext[2] << " " << hex64( r.optab_hash ) << "\n";
    os << "nops " << r.n_unique_ops << "\n";
    os << "keys " << hex64( r.keys_hash ) << "\n";
    os << "fallback " << r.fallback_pairs << "\n";
    os << "atable " << r.a_bits.size() << "\n";
    for ( std::size_t i = 0; i < r.a_bits.size(); i += 4 )
    {
        os << "a";
        for ( std::size_t j = i; j < i + 4 && j < r.a_bits.size(); ++j )
            os << " " << hex64( r.a_bits[j] );
        os << "\n";
    }
    os << "end\n";
    return os.str();
}

inline std::string serialize_initial( std::uint64_t h )
{
    std::ostringstream os;
    os << "initial " << hex64( h ) << "\n";
    return os.str();
}

inline std::string serialize_field( const FieldRecord& f )
{
    std::ostringstream os;
    os << "field " << f.n << "\n";
    for ( int i = 0; i < f.n; ++i )
        os << "f " << hex64( f.pot_bits[i] ) << " "
           << hex64( f.grad_bits[3 * i + 0] ) << " "
           << hex64( f.grad_bits[3 * i + 1] ) << " "
           << hex64( f.grad_bits[3 * i + 2] ) << "\n";
    os << "endfield\n";
    return os.str();
}

// Parse the committed file. Returns false and fills `err` if the file cannot
// be opened or is malformed.
inline bool parse_reference_file( const std::string& path, ReferenceData& out,
                                  std::string& err )
{
    std::ifstream in( path );
    if ( !in )
    {
        err = "cannot open reference data file " + path;
        return false;
    }

    // Strip comments first so the token stream below is pure data.
    std::ostringstream stripped;
    std::string line;
    while ( std::getline( in, line ) )
    {
        const std::size_t hash = line.find( '#' );
        stripped << ( hash == std::string::npos ? line
                                                : line.substr( 0, hash ) )
                 << "\n";
    }

    std::istringstream ts( stripped.str() );
    std::string tok;

    auto expect = [&]( const char* kw )
    {
        std::string t;
        ts >> t;
        if ( t != kw )
        {
            err = std::string( "expected '" ) + kw + "', found '" + t + "'";
            return false;
        }
        return true;
    };
    auto read_hex = [&]( std::uint64_t& v )
    {
        std::string t;
        ts >> t;
        v = std::strtoull( t.c_str(), nullptr, 16 );
    };

    while ( ts >> tok )
    {
        if ( tok == "initial" )
        {
            read_hex( out.initial_hash );
            out.has_initial = true;
            continue;
        }

        if ( tok == "field" )
        {
            FieldRecord f;
            ts >> f.n;
            f.pot_bits.resize( f.n );
            f.grad_bits.resize( 3 * static_cast<std::size_t>( f.n ) );
            for ( int i = 0; i < f.n; ++i )
            {
                if ( !expect( "f" ) )
                    return false;
                read_hex( f.pot_bits[i] );
                read_hex( f.grad_bits[3 * i + 0] );
                read_hex( f.grad_bits[3 * i + 1] );
                read_hex( f.grad_bits[3 * i + 2] );
            }
            if ( !expect( "endfield" ) )
                return false;
            out.field = f;
            out.has_field = true;
            continue;
        }

        if ( tok != "set" )
        {
            err = "expected 'set', 'initial' or 'field', found '" + tok + "'";
            return false;
        }

        BitRecord r;
        ts >> r.nprocs >> r.rank;

        if ( !expect( "locals" ) )
            return false;
        ts >> r.locals_ext[0] >> r.locals_ext[1] >> r.locals_ext[2];
        read_hex( r.locals_hash );

        if ( !expect( "optab" ) )
            return false;
        ts >> r.optab_ext[0] >> r.optab_ext[1] >> r.optab_ext[2];
        read_hex( r.optab_hash );

        if ( !expect( "nops" ) )
            return false;
        ts >> r.n_unique_ops;

        if ( !expect( "keys" ) )
            return false;
        read_hex( r.keys_hash );

        if ( !expect( "fallback" ) )
            return false;
        ts >> r.fallback_pairs;

        if ( !expect( "atable" ) )
            return false;
        std::size_t n_a = 0;
        ts >> n_a;
        r.a_bits.reserve( n_a );
        while ( r.a_bits.size() < n_a )
        {
            if ( !expect( "a" ) )
                return false;
            for ( int j = 0; j < 4 && r.a_bits.size() < n_a; ++j )
            {
                std::uint64_t v;
                read_hex( v );
                r.a_bits.push_back( v );
            }
        }

        if ( !expect( "end" ) )
            return false;
        if ( !ts )
        {
            err = "truncated record";
            return false;
        }
        out.bits[{ r.nprocs, r.rank }] = r;
    }
    return true;
}

// Absolute path of a per-(nprocs, rank) dump file in the current working
// directory. CTest runs each test from the build directory, so that is
// where the dumps land.
inline std::string dump_path( int nprocs, int rank, const char* what )
{
    char cwd[4096];
    const char* base = getcwd( cwd, sizeof cwd ) ? cwd : ".";
    std::ostringstream os;
    os << base << "/canopy_laplace_solve_dump_np" << nprocs << "_rank" << rank
       << "_" << what << ".txt";
    return os.str();
}

// ---------------------------------------------------------------------------
// Artifact extraction. Each reads one artifact off the sweep and folds it
// into `r`.
// ---------------------------------------------------------------------------
template <class DS>
void collect_locals( const DS& ds, BitRecord& r )
{
    auto h =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), ds.locals() );
    r.locals_ext[0] = h.extent( 0 );
    r.locals_ext[1] = h.extent( 1 );
    r.locals_ext[2] = h.extent( 2 );
    Fnv1a64 f;
    for ( std::size_t i = 0; i < h.extent( 0 ); ++i )
        for ( std::size_t j = 0; j < h.extent( 1 ); ++j )
            for ( std::size_t k = 0; k < h.extent( 2 ); ++k )
            {
                f.u64( bits_of( h( i, j, k ).real() ) );
                f.u64( bits_of( h( i, j, k ).imag() ) );
            }
    r.locals_hash = f.h;
}

// The operator table is allocated WithoutInitializing with a third extent of
// max(n_unique_ops, 1), so the pad column that exists when nothing was
// realized holds garbage. Hash only the realized columns; the extents are
// recorded separately and compared on their own.
template <class DS>
std::size_t realized_columns( const DS& ds, std::size_t ext2 )
{
    return std::min<std::size_t>(
        static_cast<std::size_t>( ds.m2l_n_unique_ops() ), ext2 );
}

template <class DS>
void collect_optab( const DS& ds, BitRecord& r )
{
    auto h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                  ds.m2l_op_table() );
    r.optab_ext[0] = h.extent( 0 );
    r.optab_ext[1] = h.extent( 1 );
    r.optab_ext[2] = h.extent( 2 );
    const std::size_t n_ops = realized_columns( ds, h.extent( 2 ) );
    Fnv1a64 f;
    for ( std::size_t i = 0; i < h.extent( 0 ); ++i )
        for ( std::size_t j = 0; j < h.extent( 1 ); ++j )
            for ( std::size_t k = 0; k < n_ops; ++k )
            {
                f.u64( bits_of( h( i, j, k ).real() ) );
                f.u64( bits_of( h( i, j, k ).imag() ) );
            }
    r.optab_hash = f.h;
}

template <class DS>
void collect_keys( const DS& ds, BitRecord& r )
{
    r.n_unique_ops = ds.m2l_n_unique_ops();
    auto keys = ds.m2l_realized_keys();
    std::sort( keys.begin(), keys.end(), []( const auto& a, const auto& b )
               { return key_less( a, b ); } );
    // FOUR FIELDS, NOT FIVE. The key carries max_d as well, but it is
    // deliberately not fed to the hash. This function pushes each field
    // through FNV separately, so feeding a fifth would push eight more bytes
    // per key and change keys_hash for every record in
    // tests/data/laplace_solve_P6.txt — forcing a regeneration of the
    // committed reference data in order to hash a constant zero, which is
    // exactly the silent re-baselining this harness exists to prevent.
    // The stronger claim is asserted directly instead: bitForBitArtifacts
    // requires max_d == 0 on every realized key for this basis, which pins
    // canonicalize_key itself rather than pinning a hash of its output.
    Fnv1a64 f;
    for ( const auto& k : keys )
    {
        f.i32( k.dd );
        f.i32( k.ii );
        f.i32( k.jj );
        f.i32( k.kk );
    }
    r.keys_hash = f.h;
}

template <class DS>
void collect_a_table( const DS& ds, BitRecord& r )
{
    auto h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                  ds.aux().A_table );
    r.a_bits.resize( h.extent( 0 ) );
    for ( std::size_t i = 0; i < h.extent( 0 ); ++i )
        r.a_bits[i] = bits_of( static_cast<double>( h( i ) ) );
}

// ---------------------------------------------------------------------------
// Mismatch dumps. R2's procedure is to compare the realized key lists
// directly, which a hash alone does not permit, so on any mismatch the full
// array goes to the build directory and the failure message names the path.
// ---------------------------------------------------------------------------
template <class DS>
void dump_locals( const DS& ds, const std::string& path )
{
    auto h =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), ds.locals() );
    std::ofstream os( path );
    os << "# locals: cell coeff comp real_bits imag_bits\n";
    for ( std::size_t i = 0; i < h.extent( 0 ); ++i )
        for ( std::size_t j = 0; j < h.extent( 1 ); ++j )
            for ( std::size_t k = 0; k < h.extent( 2 ); ++k )
                os << i << " " << j << " " << k << " "
                   << hex64( bits_of( h( i, j, k ).real() ) ) << " "
                   << hex64( bits_of( h( i, j, k ).imag() ) ) << "\n";
}

template <class DS>
void dump_optab( const DS& ds, const std::string& path )
{
    auto h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                  ds.m2l_op_table() );
    const std::size_t n_ops = realized_columns( ds, h.extent( 2 ) );
    std::ofstream os( path );
    os << "# m2l_op_table (realized columns only): t s op real_bits "
          "imag_bits\n";
    for ( std::size_t i = 0; i < h.extent( 0 ); ++i )
        for ( std::size_t j = 0; j < h.extent( 1 ); ++j )
            for ( std::size_t k = 0; k < n_ops; ++k )
                os << i << " " << j << " " << k << " "
                   << hex64( bits_of( h( i, j, k ).real() ) ) << " "
                   << hex64( bits_of( h( i, j, k ).imag() ) ) << "\n";
}

template <class DS>
void dump_keys( const DS& ds, const std::string& path )
{
    auto keys = ds.m2l_realized_keys();
    std::sort( keys.begin(), keys.end(), []( const auto& a, const auto& b )
               { return key_less( a, b ); } );
    std::ofstream os( path );
    // max_d is printed even though it is not hashed (see collect_keys), so
    // that R2's direct list comparison can see it: a canonicalization that
    // is not reaching the hash shows up here as a non-zero max_d column.
    os << "# sorted realized M2L keys: max_d dd ii jj kk\n";
    for ( const auto& k : keys )
        os << k.max_d << " " << k.dd << " " << k.ii << " " << k.jj << " "
           << k.kk << "\n";
}

inline void dump_a_table( const BitRecord& r, const BitRecord& ref,
                          const std::string& path )
{
    std::ofstream os( path );
    os << "# A_table: index measured_bits reference_bits\n";
    const std::size_t n = std::max( r.a_bits.size(), ref.a_bits.size() );
    for ( std::size_t i = 0; i < n; ++i )
        os << i << " " << ( i < r.a_bits.size() ? hex64( r.a_bits[i] ) : "-" )
           << " " << ( i < ref.a_bits.size() ? hex64( ref.a_bits[i] ) : "-" )
           << "\n";
}

// ---------------------------------------------------------------------------
// The last-step global state, gathered to rank 0 and reordered into canonical
// GlobalId order. Only rank 0's copy is filled; every other rank sees empty
// vectors and `valid == false`.
// ---------------------------------------------------------------------------
struct GatheredState
{
    bool valid = false;
    std::string err;
    int n = 0;
    std::vector<double> pot;  // n
    std::vector<double> grad; // 3 * n
    std::vector<double> pos;  // 3 * n
    std::vector<double> chg;  // n
};

// Everything one 12-step run produces.
template <class DS>
struct SolveOutcome
{
    BitRecord bits;
    std::uint64_t initial_hash = 0;
    GatheredState gathered;
};

// ---------------------------------------------------------------------------
// Drive the frozen configuration for LS_NUM_STEPS timesteps and hand the
// filled outcome and the live sweep to `after`. The sweep stays alive for
// the callback so a mismatch can dump the full arrays without solving a
// second time.
// ---------------------------------------------------------------------------
// `m2l_op_table_byte_budget` is the ONE configuration knob this driver takes,
// and 0 means "leave FmmConfig's default in place" — which is what the three
// bodies gating the frozen configuration pass, so their solves are the same
// solve they always were. Only testOpTableByteBudget passes a non-zero value,
// to make the operator table's byte budget bind before its count cap. The
// frozen-configuration block above is untouched by it: a budget changes which
// pairs get an operator column, not the problem being solved.
template <class MemorySpace, class ExecutionSpace, class Fn>
void with_laplace_solve( Fn&& after, std::size_t m2l_op_table_byte_budget = 0 )
{
    constexpr int P = LS_P;
    using Scalar = double;
    using DataTypes = Cabana::MemberTypes<Scalar[3], // Position
                                          Scalar[1], // Charge (NComps = 1)
                                          Scalar[3], // Velocity
                                          int>;      // GlobalId
    using AoSoA_t = Cabana::AoSoA<DataTypes, MemorySpace>;
    using AoSoA_ht = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
    using Solver_t =
        Canopy::Solver<MemorySpace, ExecutionSpace, Scalar, P, LS_NCOMPS>;
    using DS = typename Solver_t::downward_type;

    // The artifact layouts the committed hashes assume. A layout flip changes
    // the memory order without changing any index-order hash, so it has to be
    // asserted separately and at compile time.
    static_assert( std::is_same<typename DS::coeff_view_type::array_layout,
                                Kokkos::LayoutRight>::value,
                   "DownwardSweep::coeff_view_type must be LayoutRight" );
    static_assert(
        std::is_same<typename DS::m2l_op_table_view_type::array_layout,
                     Kokkos::LayoutLeft>::value,
        "DownwardSweep M2L operator table must be LayoutLeft" );

    BitRecord r;
    MPI_Comm_rank( MPI_COMM_WORLD, &r.rank );
    MPI_Comm_size( MPI_COMM_WORLD, &r.nprocs );

    const int n_total = LS_NUM_PARTICLES;

    // -----------------------------------------------------------------------
    // The global particle set. Seeded once with 1234 + P and generated
    // identically on every rank at every rank count, so the same physics
    // problem is solved throughout and the initial-set hash is exact by
    // construction. Each rank keeps the contiguous slice
    // [rank * N / nprocs, (rank+1) * N / nprocs); setup() redistributes, so
    // the slicing does not affect the answer. Velocities start at zero:
    // the generator draws positions and charges only.
    //
    // Charges are uniform on [0.5, 1.5] — ONE-SIGNED, the distribution
    // tests/tstMultiSolve.hpp:200 uses for its gravity tests. With charges on
    // [-1, 1] and softening = 0.0 the closest opposite-charge pair free-falls
    // to contact inside the simulated interval, the participants are ejected,
    // the bounding box grows about thirtyfold, and with max_depth = 6 the tree
    // cannot refine into the residual cloud: no pair is MAC-admissible and
    // n_unique_ops falls to 0, at which point every check here passes
    // vacuously. A one-signed set is mutually attracting — it still collapses,
    // but as a cloud rather than as a two-body singularity.
    // -----------------------------------------------------------------------
    std::vector<double> g_pos( 3 * n_total );
    std::vector<double> g_chg( n_total );
    {
        std::mt19937 gen( 1234 + P );
        std::uniform_real_distribution<double> pos_dist( 0.05, 0.95 );
        std::uniform_real_distribution<double> q_dist( 0.5, 1.5 );
        for ( int i = 0; i < n_total; i++ )
        {
            g_pos[3 * i + 0] = pos_dist( gen );
            g_pos[3 * i + 1] = pos_dist( gen );
            g_pos[3 * i + 2] = pos_dist( gen );
            g_chg[i] = q_dist( gen );
        }
    }

    // Hash of the initial global set: all positions, then all charges, in
    // global index order. Checked before anything else, so a generator drift
    // fails with one legible message instead of 600 value mismatches.
    Fnv1a64 init_hasher;
    for ( int i = 0; i < 3 * n_total; i++ )
        init_hasher.u64( bits_of( g_pos[i] ) );
    for ( int i = 0; i < n_total; i++ )
        init_hasher.u64( bits_of( g_chg[i] ) );
    const std::uint64_t initial_hash = init_hasher.h;

    const int i_begin = r.rank * n_total / r.nprocs;
    const int i_end = ( r.rank + 1 ) * n_total / r.nprocs;
    const int n_local_initial = i_end - i_begin;

    AoSoA_ht particles_h( "particles_h", n_local_initial );
    {
        auto hp = Cabana::slice<Position>( particles_h );
        auto hq = Cabana::slice<Charge>( particles_h );
        auto hv = Cabana::slice<Velocity>( particles_h );
        auto hid = Cabana::slice<GlobalId>( particles_h );
        for ( int i = 0; i < n_local_initial; i++ )
        {
            const int g = i_begin + i;
            hp( i, 0 ) = g_pos[3 * g + 0];
            hp( i, 1 ) = g_pos[3 * g + 1];
            hp( i, 2 ) = g_pos[3 * g + 2];
            hq( i, 0 ) = g_chg[g];
            hv( i, 0 ) = 0.0;
            hv( i, 1 ) = 0.0;
            hv( i, 2 ) = 0.0;
            hid( i ) = g;
        }
    }
    AoSoA_t particles( "particles", n_local_initial );
    Cabana::deep_copy( particles, particles_h );

    Canopy::FmmConfig cfg;
    cfg.ncrit = LS_NCRIT;
    cfg.max_depth = LS_MAX_DEPTH;
    cfg.xmin_tol = cfg.xmax_tol = 0.1;
    cfg.ymin_tol = cfg.ymax_tol = 0.1;
    cfg.zmin_tol = cfg.zmax_tol = 0.1;
    cfg.ncrit_tol = 0.1;
    cfg.replication_depth = 2;
    cfg.imbalance_tolerance = 0.05;
    cfg.mac_theta = LS_MAC_THETA;
    cfg.softening = 0.0;
    if ( m2l_op_table_byte_budget > 0 )
        cfg.m2l_op_table_byte_budget = m2l_op_table_byte_budget;

    Solver_t solver( MPI_COMM_WORLD, cfg );
    solver.template setup<Position, Charge>( particles, n_local_initial );

    // -----------------------------------------------------------------------
    // Time loop. Each step is solve -> symplectic-Euler update -> migrate.
    //
    // The update and the migrate are omitted after the LAST solve: every
    // check below reads that solve's artifacts and its field, and migrate()
    // rebuilds the tree and re-runs downward.setup(), which would overwrite
    // locals() and the operator table, while the update would move the
    // particles away from the positions the field was evaluated at. So the
    // loop is 12 solves with 11 intervening maintenance steps.
    //
    // migrate(), and never rebalance(), rebuild() or auto_maintain():
    // migrate moves particles to the ranks that already own their cells and
    // never repartitions, so the np 1-2 bitwise gate faces one partition
    // rather than twelve. The partitioner is not reproducible run-to-run
    // above two ranks (src/Canopy_TreePartitioner.hpp:417-419) and every
    // further invocation is another opportunity for the np=2 cut to stop
    // coming out the same way.
    // -----------------------------------------------------------------------
    for ( int step = 0; step < LS_NUM_STEPS; step++ )
    {
        solver.template solve<Position, Charge>( particles,
                                                 /*compute_gradient=*/true );

        if ( step + 1 == LS_NUM_STEPS )
            break;

        // Symplectic Euler: v += dt * g;  r += dt * drift * v. On device,
        // written directly into the AoSoA slices, as
        // tests/tstMultiSolve.hpp:365-392 writes it.
        const int n_local = solver.num_local_particles();
        auto positions = Cabana::slice<Position>( particles );
        auto velocities = Cabana::slice<Velocity>( particles );
        auto grad = solver.gradient();
        const double dt_local = LS_DT;
        const double drift_local = LS_DRIFT_MULTIPLIER;
        Kokkos::parallel_for(
            "LaplaceSolve::integrate",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_local ),
            KOKKOS_LAMBDA( int i ) {
                const double gx = grad( i, 0, 0 );
                const double gy = grad( i, 0, 1 );
                const double gz = grad( i, 0, 2 );
                velocities( i, 0 ) += dt_local * gx;
                velocities( i, 1 ) += dt_local * gy;
                velocities( i, 2 ) += dt_local * gz;
                positions( i, 0 ) +=
                    dt_local * drift_local * velocities( i, 0 );
                positions( i, 1 ) +=
                    dt_local * drift_local * velocities( i, 1 );
                positions( i, 2 ) +=
                    dt_local * drift_local * velocities( i, 2 );
            } );
        Kokkos::fence();

        solver.template migrate<Position>( particles );
    }

    const DS& ds = solver.downward();

    collect_locals( ds, r );
    collect_optab( ds, r );
    collect_keys( ds, r );
    collect_a_table( ds, r );
    r.fallback_pairs = ds.total_fallback_pair_count();

    // -----------------------------------------------------------------------
    // Gather the last-step state to rank 0 and reorder it into canonical
    // GlobalId order, as tests/tstMultiSolve.hpp:790-852 gathers it.
    // Particles pair across rank counts by GlobalId and by nothing else:
    // migration scrambles the local ordering, and the solve has moved the
    // particles, so the last-step positions are not bit-identical between one
    // rank count and another.
    // -----------------------------------------------------------------------
    GatheredState gs;
    {
        const int n_local = solver.num_local_particles();
        auto positions = Cabana::slice<Position>( particles );
        auto charges = Cabana::slice<Charge>( particles );
        auto gids = Cabana::slice<GlobalId>( particles );
        auto h_pos = Canopy::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          positions, "h_pos" );
        auto h_chg = Canopy::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          charges, "h_chg" );
        auto h_gid = Canopy::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          gids, "h_gid" );
        auto h_pot = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          solver.potential() );
        auto h_grad = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                           solver.gradient() );

        std::vector<double> l_pos( 3 * n_local ), l_chg( n_local );
        std::vector<double> l_pot( n_local ), l_grad( 3 * n_local );
        std::vector<int> l_gid( n_local );
        for ( int i = 0; i < n_local; i++ )
        {
            l_pos[3 * i + 0] = h_pos( i, 0 );
            l_pos[3 * i + 1] = h_pos( i, 1 );
            l_pos[3 * i + 2] = h_pos( i, 2 );
            l_chg[i] = h_chg( i, 0 );
            l_pot[i] = h_pot( i, 0 );
            l_grad[3 * i + 0] = h_grad( i, 0, 0 );
            l_grad[3 * i + 1] = h_grad( i, 0, 1 );
            l_grad[3 * i + 2] = h_grad( i, 0, 2 );
            l_gid[i] = h_gid( i );
        }

        std::vector<int> all_n( r.nprocs, 0 );
        MPI_Gather( &n_local, 1, MPI_INT, all_n.data(), 1, MPI_INT, 0,
                    MPI_COMM_WORLD );
        std::vector<int> cnt1( r.nprocs, 0 ), dsp1( r.nprocs, 0 );
        std::vector<int> cnt3( r.nprocs, 0 ), dsp3( r.nprocs, 0 );
        int total = 0;
        if ( r.rank == 0 )
        {
            for ( int k = 0; k < r.nprocs; k++ )
            {
                cnt1[k] = all_n[k];
                cnt3[k] = 3 * all_n[k];
                total += all_n[k];
            }
            for ( int k = 1; k < r.nprocs; k++ )
            {
                dsp1[k] = dsp1[k - 1] + cnt1[k - 1];
                dsp3[k] = dsp3[k - 1] + cnt3[k - 1];
            }
        }

        std::vector<double> a_pos, a_chg, a_pot, a_grad;
        std::vector<int> a_gid;
        if ( r.rank == 0 )
        {
            a_pos.resize( 3 * total );
            a_chg.resize( total );
            a_pot.resize( total );
            a_grad.resize( 3 * total );
            a_gid.resize( total );
        }
        MPI_Gatherv( l_pos.data(), 3 * n_local, MPI_DOUBLE, a_pos.data(),
                     cnt3.data(), dsp3.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD );
        MPI_Gatherv( l_chg.data(), n_local, MPI_DOUBLE, a_chg.data(),
                     cnt1.data(), dsp1.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD );
        MPI_Gatherv( l_pot.data(), n_local, MPI_DOUBLE, a_pot.data(),
                     cnt1.data(), dsp1.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD );
        MPI_Gatherv( l_grad.data(), 3 * n_local, MPI_DOUBLE, a_grad.data(),
                     cnt3.data(), dsp3.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD );
        MPI_Gatherv( l_gid.data(), n_local, MPI_INT, a_gid.data(), cnt1.data(),
                     dsp1.data(), MPI_INT, 0, MPI_COMM_WORLD );

        if ( r.rank == 0 )
        {
            gs.n = n_total;
            gs.pot.assign( n_total, 0.0 );
            gs.grad.assign( 3 * n_total, 0.0 );
            gs.pos.assign( 3 * n_total, 0.0 );
            gs.chg.assign( n_total, 0.0 );
            std::vector<int> seen( n_total, 0 );
            std::ostringstream err;
            if ( total != n_total )
                err << "gathered " << total << " particles, expected "
                    << n_total << "; ";
            for ( int i = 0; i < total; i++ )
            {
                const int g = a_gid[i];
                if ( g < 0 || g >= n_total )
                {
                    err << "GlobalId " << g << " out of range; ";
                    break;
                }
                if ( seen[g]++ )
                {
                    err << "GlobalId " << g << " appears twice; ";
                    break;
                }
                gs.pot[g] = a_pot[i];
                gs.chg[g] = a_chg[i];
                for ( int d = 0; d < 3; d++ )
                {
                    gs.grad[3 * g + d] = a_grad[3 * i + d];
                    gs.pos[3 * g + d] = a_pos[3 * i + d];
                }
            }
            for ( int g = 0; g < n_total && err.str().empty(); g++ )
                if ( seen[g] != 1 )
                {
                    err << "GlobalId " << g << " missing from gathered set; ";
                    break;
                }
            gs.err = err.str();
            gs.valid = gs.err.empty();
        }
    }

    // Always echo the measurements. `ctest -V` on this test is how the
    // per-rank-count n_unique_ops and fallback counts reach the progress log.
    std::printf( "[laplace-solve] nprocs=%d rank=%d steps=%d n_unique_ops=%d "
                 "fallback_pairs=%lld locals_ext=(%zu,%zu,%zu) "
                 "optab_ext=(%zu,%zu,%zu) a_extent=%zu initial_hash=%s "
                 "op_budget=%zu op_cap=%d\n",
                 r.nprocs, r.rank, LS_NUM_STEPS, r.n_unique_ops,
                 r.fallback_pairs, r.locals_ext[0], r.locals_ext[1],
                 r.locals_ext[2], r.optab_ext[0], r.optab_ext[1],
                 r.optab_ext[2], r.a_bits.size(),
                 hex64( initial_hash ).c_str(),
                 cfg.m2l_op_table_byte_budget, ds.m2l_effective_op_cap() );
    std::fflush( stdout );

    SolveOutcome<DS> outcome;
    outcome.bits = r;
    outcome.initial_hash = initial_hash;
    outcome.gathered = std::move( gs );

    after( outcome, ds );
}

// Read the committed reference file, failing the calling test with one
// legible message if it cannot be read, and check the initial-particle-set
// hash before anything else. Returns false if the caller should stop.
inline bool load_reference( std::uint64_t measured_initial_hash,
                            ReferenceData& ref )
{
    std::string err;
    if ( !parse_reference_file( LS_DATA_FILE, ref, err ) )
    {
        ADD_FAILURE() << err << "\nRegenerate with " << LS_REGENERATE_ENV
                      << "=<dir> at ranks 1 and 2; see the comment at the top "
                         "of tstLaplaceSolve.hpp.";
        return false;
    }
    if ( !ref.has_initial )
    {
        ADD_FAILURE() << "no 'initial' record in " << LS_DATA_FILE;
        return false;
    }
    if ( ref.initial_hash != measured_initial_hash )
    {
        ADD_FAILURE()
            << "the initial global particle set has drifted: measured "
            << hex64( measured_initial_hash ) << ", reference "
            << hex64( ref.initial_hash )
            << ". The generator or the frozen configuration changed; every "
               "committed record in "
            << LS_DATA_FILE << " is meaningless until it is regenerated.";
        return false;
    }
    return true;
}

// The global normalization scales for one field: max |phi| and the maximum
// gradient magnitude, both over the whole set. Never a per-particle relative
// error. The charges are one-signed, on [0.5, 1.5], which does keep
// per-particle |phi| away from zero — but the GRADIENT components still pass
// through zero by cancellation wherever a particle's neighbours pull against
// each other, so a per-particle ratio there would measure that cancellation
// rather than accuracy. One scale rule for both fields keeps the two
// comparisons reading the same way.
inline void field_scales( const std::vector<double>& pot,
                          const std::vector<double>& grad, int n,
                          double& pot_scale, double& grad_scale )
{
    pot_scale = 0.0;
    grad_scale = 0.0;
    for ( int i = 0; i < n; i++ )
    {
        pot_scale = std::max( pot_scale, std::abs( pot[i] ) );
        const double gm = std::sqrt( grad[3 * i + 0] * grad[3 * i + 0] +
                                     grad[3 * i + 1] * grad[3 * i + 1] +
                                     grad[3 * i + 2] * grad[3 * i + 2] );
        grad_scale = std::max( grad_scale, gm );
    }
}

// ---------------------------------------------------------------------------
// Test bodies.
// ---------------------------------------------------------------------------

// np 1-2 only. The partition is not reproducible run-to-run above two ranks,
// so there is no stable baseline to compare bit patterns against there.
template <class MemorySpace, class ExecutionSpace>
void testBitForBitArtifacts()
{
    int nprocs = 0;
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
    if ( nprocs >= 3 )
        GTEST_SKIP() << "bit-for-bit artifacts are gated at np 1-2 only: "
                        "TreePartitioner::partition_leaves uses the Zoltan2 "
                        "multijagged algorithm, which is not reproducible "
                        "run-to-run above two parts, so cell ownership — and "
                        "with it locals(), the M2L operator table and the "
                        "realized key set — moves between runs at np >= 3. "
                        "See README.md \"Known Issues\" and "
                        "tasks/abstract-solver-backend.md T1.";

    with_laplace_solve<MemorySpace, ExecutionSpace>(
        []( const auto& outcome, const auto& ds )
        {
            const BitRecord& r = outcome.bits;

            const char* regen = std::getenv( LS_REGENERATE_ENV );
            if ( regen && regen[0] != '\0' )
            {
                std::ostringstream path;
                path << regen << "/laplace_solve_np" << r.nprocs << "_rank"
                     << r.rank << ".part";
                std::ofstream os( path.str() );
                ASSERT_TRUE( os.good() ) << LS_REGENERATE_ENV << " is set but "
                                         << path.str() << " cannot be written";
                os << serialize_bits( r );
                // The np=1 rank-0 run is the one that also carries the
                // initial-set hash and the np=1 field record: it is the
                // reference every cross-rank comparison is made against, and
                // it is bit-reproducible, which is what makes committing its
                // output meaningful.
                if ( r.nprocs == 1 && r.rank == 0 )
                {
                    os << serialize_initial( outcome.initial_hash );
                    ASSERT_TRUE( outcome.gathered.valid )
                        << "gathered last-step state is invalid: "
                        << outcome.gathered.err;
                    FieldRecord f;
                    f.n = outcome.gathered.n;
                    f.pot_bits.resize( f.n );
                    f.grad_bits.resize( 3 * static_cast<std::size_t>( f.n ) );
                    for ( int i = 0; i < f.n; i++ )
                    {
                        f.pot_bits[i] = bits_of( outcome.gathered.pot[i] );
                        for ( int d = 0; d < 3; d++ )
                            f.grad_bits[3 * i + d] =
                                bits_of( outcome.gathered.grad[3 * i + d] );
                    }
                    os << serialize_field( f );
                }
                os.close();
                GTEST_SKIP() << "regeneration mode: wrote " << path.str()
                             << "; comparison skipped";
            }

            const std::string tag = "(nprocs=" + std::to_string( r.nprocs ) +
                                    " rank=" + std::to_string( r.rank ) + ")";

            // A non-zero count means pairs moved onto the per-pair fallback
            // path, which is different arithmetic from the operator path.
            // R4's discriminator and T8's exit criterion both rest on this
            // being zero at P = 6 for this configuration, so it is asserted
            // to be zero rather than pinned to a measured value.
            EXPECT_EQ( 0, r.fallback_pairs )
                << "total_fallback_pair_count() must be 0 for the frozen "
                   "Laplace-solve configuration "
                << tag;

            // A tree that has degenerated realizes ZERO M2L operators, and at
            // that point every check in this harness passes without measuring
            // the far field: matchesDirectSum passes to machine precision
            // because the solve is pure P2P, this test compares an operator
            // table with zero realized columns, and the m-loop perturbation
            // the exit criterion requires to FAIL cannot fire, because the
            // loop is reached only through a CSR entry with a valid op_idx.
            // This assertion is what makes that state a failure.
            EXPECT_GT( r.n_unique_ops, 0 )
                << "m2l_n_unique_ops() is 0: the tree has degenerated and no "
                   "pair is MAC-admissible, so the far field this harness "
                   "exists to protect was never evaluated "
                << tag;

            ReferenceData ref_all;
            ASSERT_TRUE( load_reference( outcome.initial_hash, ref_all ) );

            auto it = ref_all.bits.find( { r.nprocs, r.rank } );
            ASSERT_NE( it, ref_all.bits.end() )
                << "no reference record for " << tag << " in " << LS_DATA_FILE;
            const BitRecord& ref = it->second;

            // --- A_{n,m} table: full bit patterns ------------------------
            bool a_ok = ( r.a_bits.size() == ref.a_bits.size() );
            EXPECT_EQ( ref.a_bits.size(), r.a_bits.size() )
                << "A_table() extent differs " << tag;
            if ( a_ok )
            {
                for ( std::size_t i = 0; i < r.a_bits.size(); ++i )
                {
                    if ( r.a_bits[i] != ref.a_bits[i] )
                    {
                        a_ok = false;
                        ADD_FAILURE() << "A_table() bit pattern differs " << tag
                                      << " at index " << i << ": measured "
                                      << hex64( r.a_bits[i] ) << ", reference "
                                      << hex64( ref.a_bits[i] );
                        break;
                    }
                }
            }
            if ( !a_ok )
            {
                const std::string p = dump_path( r.nprocs, r.rank, "atable" );
                dump_a_table( r, ref, p );
                ADD_FAILURE() << "A_table() bit patterns written to " << p;
            }

            // --- canonicalize_key actually zeroed the level ---------------
            // The key carries max_d (T7), and this basis's canonicalize_key
            // zeroes it because its operators are scale-normalized and
            // therefore depth-independent. keys_hash cannot see that — it
            // hashes four of the five fields, so the committed data stays
            // valid across T7 — so assert it on the realized keys directly.
            // This is the stronger check of the two: hashing a constant zero
            // would only prove the zero was hashed, while this proves every
            // key the classify pass realized came out canonical, which is
            // what keeps the operator table at one column per offset instead
            // of one per (level, offset).
            {
                int n_level_carrying = 0;
                int first_bad_max_d = 0;
                for ( const auto& k : ds.m2l_realized_keys() )
                {
                    if ( k.max_d != 0 )
                    {
                        if ( n_level_carrying == 0 )
                            first_bad_max_d = k.max_d;
                        n_level_carrying++;
                    }
                }
                EXPECT_EQ( 0, n_level_carrying )
                    << "canonicalize_key did not reach the hash: "
                    << n_level_carrying << " of "
                    << ds.m2l_realized_keys().size()
                    << " realized keys carry a non-zero max_d (first is "
                    << first_bad_max_d
                    << "). The solid-harmonic operator table would then hold "
                       "one column per (level, offset) instead of one per "
                       "offset "
                    << tag;
            }

            // --- realized key list and n_unique_ops -----------------------
            const bool keys_ok = ( r.keys_hash == ref.keys_hash &&
                                   r.n_unique_ops == ref.n_unique_ops );
            EXPECT_EQ( ref.n_unique_ops, r.n_unique_ops )
                << "n_unique_ops differs " << tag;
            EXPECT_EQ( ref.keys_hash, r.keys_hash )
                << "sorted realized key list hash differs " << tag
                << ": measured " << hex64( r.keys_hash ) << ", reference "
                << hex64( ref.keys_hash );
            if ( !keys_ok )
            {
                const std::string p = dump_path( r.nprocs, r.rank, "keys" );
                dump_keys( ds, p );
                ADD_FAILURE()
                    << "sorted realized key list written to " << p
                    << " (R2: compare the key lists directly from here)";
            }

            // --- M2L operator table ---------------------------------------
            const bool optab_ok = ( r.optab_ext[0] == ref.optab_ext[0] &&
                                    r.optab_ext[1] == ref.optab_ext[1] &&
                                    r.optab_ext[2] == ref.optab_ext[2] &&
                                    r.optab_hash == ref.optab_hash );
            EXPECT_EQ( ref.optab_ext[0], r.optab_ext[0] )
                << "M2L operator table extent 0 differs " << tag;
            EXPECT_EQ( ref.optab_ext[1], r.optab_ext[1] )
                << "M2L operator table extent 1 differs " << tag;
            EXPECT_EQ( ref.optab_ext[2], r.optab_ext[2] )
                << "M2L operator table extent 2 differs " << tag;
            EXPECT_EQ( ref.optab_hash, r.optab_hash )
                << "M2L operator table hash differs " << tag << ": measured "
                << hex64( r.optab_hash ) << ", reference "
                << hex64( ref.optab_hash );
            if ( !optab_ok )
            {
                const std::string p = dump_path( r.nprocs, r.rank, "optab" );
                dump_optab( ds, p );
                ADD_FAILURE() << "M2L operator table written to " << p;
            }

            // --- locals() --------------------------------------------------
            const bool locals_ok = ( r.locals_ext[0] == ref.locals_ext[0] &&
                                     r.locals_ext[1] == ref.locals_ext[1] &&
                                     r.locals_ext[2] == ref.locals_ext[2] &&
                                     r.locals_hash == ref.locals_hash );
            EXPECT_EQ( ref.locals_ext[0], r.locals_ext[0] )
                << "locals() extent 0 differs " << tag;
            EXPECT_EQ( ref.locals_ext[1], r.locals_ext[1] )
                << "locals() extent 1 differs " << tag;
            EXPECT_EQ( ref.locals_ext[2], r.locals_ext[2] )
                << "locals() extent 2 differs " << tag;
            EXPECT_EQ( ref.locals_hash, r.locals_hash )
                << "locals() hash differs " << tag << ": measured "
                << hex64( r.locals_hash ) << ", reference "
                << hex64( ref.locals_hash );
            if ( !locals_ok )
            {
                const std::string p = dump_path( r.nprocs, r.rank, "locals" );
                dump_locals( ds, p );
                ADD_FAILURE() << "locals() bit patterns written to " << p;
            }
        } );
}

// np 2-6. np=1 is the reference and is skipped.
//
// The FMM answer is partition-independent as mathematics: the interaction set
// is a function of the tree and not of ownership; cells at depth <=
// replication_depth are shared and their M2L runs on rank 0 alone, with the
// Allreduce summing rank 0's contribution against zeros
// (src/Canopy_DownwardSweep.hpp:701-708); every non-shared target is owned by
// exactly one rank. Only the summation order moves when the partition moves.
// What that argument does not settle is the size of the deviation after 12
// steps, because each step's difference enters the velocity update and is
// carried into the next step's positions — so LS_CROSS_RANK_TOL is measured,
// not derived.
template <class MemorySpace, class ExecutionSpace>
void testCrossRankAgreement()
{
    int nprocs = 0;
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
    if ( nprocs == 1 )
        GTEST_SKIP() << "np=1 is the committed reference this test compares "
                        "against; there is nothing to compare it to.";

    const char* regen = std::getenv( LS_REGENERATE_ENV );
    if ( regen && regen[0] != '\0' )
        GTEST_SKIP() << LS_REGENERATE_ENV
                     << " is set; the reference data is being written by "
                        "bitForBitArtifacts and cannot be compared against "
                        "in the same run.";

    with_laplace_solve<MemorySpace, ExecutionSpace>(
        []( const auto& outcome, const auto& )
        {
            const BitRecord& r = outcome.bits;
            const std::string tag = "(nprocs=" + std::to_string( r.nprocs ) +
                                    " rank=" + std::to_string( r.rank ) + ")";

            // R4's discriminator, still covered at np 3-6 where
            // bitForBitArtifacts no longer runs.
            EXPECT_EQ( 0, r.fallback_pairs )
                << "total_fallback_pair_count() must be 0 for the frozen "
                   "Laplace-solve configuration "
                << tag;

            // A tree that has degenerated realizes ZERO M2L operators, and at
            // that point every check in this harness passes without measuring
            // the far field: matchesDirectSum passes to machine precision
            // because the solve is pure P2P, this test compares an operator
            // table with zero realized columns, and the m-loop perturbation
            // the exit criterion requires to FAIL cannot fire, because the
            // loop is reached only through a CSR entry with a valid op_idx.
            // This assertion is what makes that state a failure.
            EXPECT_GT( r.n_unique_ops, 0 )
                << "m2l_n_unique_ops() is 0: the tree has degenerated and no "
                   "pair is MAC-admissible, so the far field this harness "
                   "exists to protect was never evaluated "
                << tag;

            ReferenceData ref_all;
            ASSERT_TRUE( load_reference( outcome.initial_hash, ref_all ) );
            if ( r.rank != 0 )
                return;

            ASSERT_TRUE( ref_all.has_field )
                << "no np=1 field record in " << LS_DATA_FILE;
            const GatheredState& gs = outcome.gathered;
            ASSERT_TRUE( gs.valid )
                << "gathered last-step state is invalid: " << gs.err;
            ASSERT_EQ( ref_all.field.n, gs.n )
                << "np=1 field record holds a different particle count";

            const int n = gs.n;
            std::vector<double> ref_pot( n ), ref_grad( 3 * n );
            for ( int i = 0; i < n; i++ )
            {
                ref_pot[i] = double_of( ref_all.field.pot_bits[i] );
                for ( int d = 0; d < 3; d++ )
                    ref_grad[3 * i + d] =
                        double_of( ref_all.field.grad_bits[3 * i + d] );
            }

            // Normalize by a GLOBAL scale taken from the np=1 reference,
            // never by a per-particle magnitude.
            double pot_scale = 0.0, grad_scale = 0.0;
            field_scales( ref_pot, ref_grad, n, pot_scale, grad_scale );
            ASSERT_GT( pot_scale, 0.0 );
            ASSERT_GT( grad_scale, 0.0 );

            double max_pot_dev = 0.0, max_grad_dev = 0.0;
            for ( int i = 0; i < n; i++ )
            {
                max_pot_dev =
                    std::max( max_pot_dev,
                              std::abs( gs.pot[i] - ref_pot[i] ) / pot_scale );
                const double dx = gs.grad[3 * i + 0] - ref_grad[3 * i + 0];
                const double dy = gs.grad[3 * i + 1] - ref_grad[3 * i + 1];
                const double dz = gs.grad[3 * i + 2] - ref_grad[3 * i + 2];
                max_grad_dev = std::max(
                    max_grad_dev,
                    std::sqrt( dx * dx + dy * dy + dz * dz ) / grad_scale );
            }

            std::printf( "[laplace-solve] nprocs=%d cross_rank "
                         "max_pot_dev=%.17g max_grad_dev=%.17g "
                         "tol=%.17g\n",
                         r.nprocs, max_pot_dev, max_grad_dev,
                         LS_CROSS_RANK_TOL );
            std::fflush( stdout );

            // A deviation above LS_CROSS_RANK_R8_THRESHOLD is R8: the FMM
            // answer would depend on the partition rather than only on the
            // summation order, and the tolerance must not be raised to
            // accommodate it. Re-measure the same configuration at
            // LS_NUM_STEPS = 1 before attributing it to the dataflow — a
            // one-step deviation at reassociation level with a 12-step
            // deviation above the threshold is the integrator amplifying
            // reassociation; a one-step deviation already above it is R8.
            const double worst = std::max( max_pot_dev, max_grad_dev );
            if ( worst >= LS_CROSS_RANK_R8_THRESHOLD )
                ADD_FAILURE()
                    << "R8: the np=" << r.nprocs
                    << " field deviates from the committed np=1 field by "
                    << worst << ", at or above the "
                    << LS_CROSS_RANK_R8_THRESHOLD
                    << " threshold above which the deviation is no longer "
                       "attributable to floating-point reassociation. Do not "
                       "raise LS_CROSS_RANK_TOL: re-measure at "
                       "LS_NUM_STEPS = 1, record both numbers in "
                       "tasks/abstract-solver-backend-progress-log.md, and "
                       "report. See abstract-solver-backend.md risk R8.";

            EXPECT_LT( max_pot_dev, LS_CROSS_RANK_TOL )
                << "np=" << r.nprocs
                << " potential does not reproduce the committed np=1 "
                   "potential to floating-point reassociation";
            EXPECT_LT( max_grad_dev, LS_CROSS_RANK_TOL )
                << "np=" << r.nprocs
                << " gradient does not reproduce the committed np=1 gradient "
                   "to floating-point reassociation";
        } );
}

// np 1-6. The only check in this harness that the far field is the *right*
// field: crossRankAgreement compares the solve against itself and would pass
// a uniformly wrong answer at every rank count.
template <class MemorySpace, class ExecutionSpace>
void testMatchesDirectSum()
{
    const char* regen = std::getenv( LS_REGENERATE_ENV );
    if ( regen && regen[0] != '\0' )
        GTEST_SKIP() << LS_REGENERATE_ENV
                     << " is set; reference data is being written and the "
                        "initial-set hash cannot be checked against it in "
                        "the same run.";

    with_laplace_solve<MemorySpace, ExecutionSpace>(
        []( const auto& outcome, const auto& )
        {
            const BitRecord& r = outcome.bits;

            ReferenceData ref_all;
            ASSERT_TRUE( load_reference( outcome.initial_hash, ref_all ) );
            if ( r.rank != 0 )
                return;

            const GatheredState& gs = outcome.gathered;
            ASSERT_TRUE( gs.valid )
                << "gathered last-step state is invalid: " << gs.err;

            // Brute-force O(N^2) reference in double precision over the
            // gathered last-step positions and charges, structured as
            // tests/tstMultiSolve.hpp:866-901.
            const int n = gs.n;
            std::vector<double> bf_pot( n, 0.0 ), bf_grad( 3 * n, 0.0 );
            for ( int i = 0; i < n; i++ )
            {
                double phi = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
                for ( int j = 0; j < n; j++ )
                {
                    if ( j == i )
                        continue;
                    const double dx = gs.pos[3 * i + 0] - gs.pos[3 * j + 0];
                    const double dy = gs.pos[3 * i + 1] - gs.pos[3 * j + 1];
                    const double dz = gs.pos[3 * i + 2] - gs.pos[3 * j + 2];
                    const double inv_r =
                        1.0 / std::sqrt( dx * dx + dy * dy + dz * dz );
                    const double inv_r3 = inv_r * inv_r * inv_r;
                    phi += gs.chg[j] * inv_r;
                    gx -= gs.chg[j] * dx * inv_r3;
                    gy -= gs.chg[j] * dy * inv_r3;
                    gz -= gs.chg[j] * dz * inv_r3;
                }
                bf_pot[i] = phi;
                bf_grad[3 * i + 0] = gx;
                bf_grad[3 * i + 1] = gy;
                bf_grad[3 * i + 2] = gz;
            }

            // Same global-scale normalization as crossRankAgreement, rather
            // than run_fmm_and_compare's per-particle ratio, and for the same
            // cancellation reason.
            double pot_scale = 0.0, grad_scale = 0.0;
            field_scales( bf_pot, bf_grad, n, pot_scale, grad_scale );
            ASSERT_GT( pot_scale, 0.0 );
            ASSERT_GT( grad_scale, 0.0 );

            double max_pot_dev = 0.0, max_grad_dev = 0.0;
            for ( int i = 0; i < n; i++ )
            {
                max_pot_dev =
                    std::max( max_pot_dev,
                              std::abs( gs.pot[i] - bf_pot[i] ) / pot_scale );
                const double dx = gs.grad[3 * i + 0] - bf_grad[3 * i + 0];
                const double dy = gs.grad[3 * i + 1] - bf_grad[3 * i + 1];
                const double dz = gs.grad[3 * i + 2] - bf_grad[3 * i + 2];
                max_grad_dev = std::max(
                    max_grad_dev,
                    std::sqrt( dx * dx + dy * dy + dz * dz ) / grad_scale );
            }

            std::printf( "[laplace-solve] nprocs=%d direct_sum "
                         "max_pot_dev=%.17g max_grad_dev=%.17g tol=%.17g\n",
                         r.nprocs, max_pot_dev, max_grad_dev,
                         LS_DIRECT_SUM_TOL );
            std::fflush( stdout );

            EXPECT_LT( max_pot_dev, LS_DIRECT_SUM_TOL )
                << "np=" << r.nprocs
                << " potential does not match the direct sum at the accuracy "
                   "the method delivers";
            EXPECT_LT( max_grad_dev, LS_DIRECT_SUM_TOL )
                << "np=" << r.nprocs
                << " gradient does not match the direct sum at the accuracy "
                   "the method delivers";
        } );
}

// ---------------------------------------------------------------------------
// np 1-6. The operator table's memory budget, and the overflow path it drives.
//
// Two Solvers, one process, one rank count: the first at FmmConfig's default
// 2 GB budget, where the count cap binds and no pair overflows; the second at
// LS_BUDGET_KEYS columns' worth of bytes, where the budget binds long before
// the count cap and most pairs are refused a column. The refused pairs take
// the basis's overflow path — for LaplaceKernel, M2LOverflow::PerPairTranslate
// — and the two solves must still agree on the potential.
//
// WHAT THIS ASSERTS, AND WHY EACH HALF MATTERS.
//
//   - The default-budget run has total_fallback_pair_count() == 0. That is
//     R4's discriminator: the byte budget must not change which pairs overflow
//     at the frozen configuration, and the retained count cap is what
//     guarantees it. bitForBitArtifacts and crossRankAgreement assert the same
//     thing; this body asserts it a third time, in the one test that also
//     proves a non-zero count is reachable.
//   - The tight-budget run has total_fallback_pair_count() > 0 on every rank,
//     and n_unique_ops <= LS_BUDGET_KEYS. Without both, the comparison below
//     would be comparing two identical runs and would pass vacuously — which
//     is exactly how a cap that silently stopped binding would look.
//   - The potentials agree to LS_BUDGET_POTENTIAL_TOL. This is the substantive
//     claim: the fallback path is different arithmetic, not wrong arithmetic.
//
// Rank 0 does the comparison, on the gathered GlobalId-ordered field, with the
// same global-scale normalization crossRankAgreement and matchesDirectSum use.
// Both runs are fully collective and every rank drives both.
//
// This body does not read tests/data/laplace_solve_P6.txt and does not depend
// on the two pinned tolerances. It compares one run against another run of the
// same binary at the same rank count, so the partitioner's run-to-run
// non-determinism enters only as reassociation — orders of magnitude under
// this tolerance — and no committed record constrains it.
// ---------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace>
void testOpTableByteBudget()
{
    using Kernel = Canopy::LaplaceKernel<double, LS_P, LS_NCOMPS>;

    // DERIVED, never a literal: the budget is expressed in columns and
    // converted here by the same trait the sweep divides by, so this test
    // cannot drift from the thing it is testing if a basis's coefficient
    // width changes.
    constexpr std::size_t bytes_per_key = Kernel::bytes_per_key;
    const std::size_t tight_budget =
        bytes_per_key * static_cast<std::size_t>( LS_BUDGET_KEYS );

    int rank = 0, nprocs = 1;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    GatheredState wide_field;
    long long wide_fallback = -1;
    int wide_ops = -1;
    int wide_cap = -1;

    with_laplace_solve<MemorySpace, ExecutionSpace>(
        [&]( const auto& outcome, const auto& ds )
        {
            wide_fallback = outcome.bits.fallback_pairs;
            wide_ops = outcome.bits.n_unique_ops;
            wide_cap = ds.m2l_effective_op_cap();
            wide_field = outcome.gathered;
        } );

    GatheredState tight_field;
    long long tight_fallback = -1;
    int tight_ops = -1;
    int tight_cap = -1;

    with_laplace_solve<MemorySpace, ExecutionSpace>(
        [&]( const auto& outcome, const auto& ds )
        {
            tight_fallback = outcome.bits.fallback_pairs;
            tight_ops = outcome.bits.n_unique_ops;
            tight_cap = ds.m2l_effective_op_cap();
            tight_field = outcome.gathered;
        },
        tight_budget );

    std::printf( "[laplace-solve] nprocs=%d rank=%d op_budget_check "
                 "bytes_per_key=%zu wide(budget=default cap=%d ops=%d "
                 "fallback=%lld) tight(budget=%zu cap=%d ops=%d "
                 "fallback=%lld)\n",
                 nprocs, rank, bytes_per_key, wide_cap, wide_ops,
                 wide_fallback, tight_budget, tight_cap, tight_ops,
                 tight_fallback );
    std::fflush( stdout );

    // The default budget must leave the count cap binding and no pair
    // overflowing — R4's discriminator, asserted on every rank.
    EXPECT_EQ( wide_cap, 32768 )
        << "the default FmmConfig budget no longer leaves M2L_OP_COUNT_CAP "
           "as the binding cap, so the byte budget has moved which pairs "
           "overflow at the frozen configuration";
    EXPECT_EQ( wide_fallback, 0 )
        << "np=" << nprocs << " rank=" << rank
        << " the default-budget solve routed pairs to the per-pair fallback; "
           "R4's discriminator has fired";
    EXPECT_GT( wide_ops, 0 );

    // The tight budget must actually bind, or the comparison below is
    // vacuous.
    EXPECT_EQ( tight_cap, LS_BUDGET_KEYS )
        << "the tight budget of " << tight_budget << " B at " << bytes_per_key
        << " B per key did not produce a cap of " << LS_BUDGET_KEYS
        << " columns";
    EXPECT_LE( tight_ops, LS_BUDGET_KEYS )
        << "np=" << nprocs << " rank=" << rank
        << " the tight-budget solve built more operator columns than the cap "
           "allows";
    EXPECT_GT( tight_fallback, 0 )
        << "np=" << nprocs << " rank=" << rank
        << " the tight-budget solve overflowed no pair onto the fallback "
           "path, so this test compares two identical solves and proves "
           "nothing";

    if ( rank != 0 )
        return;

    ASSERT_TRUE( wide_field.valid )
        << "default-budget gathered state is invalid: " << wide_field.err;
    ASSERT_TRUE( tight_field.valid )
        << "tight-budget gathered state is invalid: " << tight_field.err;
    ASSERT_EQ( wide_field.n, tight_field.n );

    const int n = wide_field.n;
    double pot_scale = 0.0, grad_scale = 0.0;
    field_scales( wide_field.pot, wide_field.grad, n, pot_scale, grad_scale );
    ASSERT_GT( pot_scale, 0.0 );
    ASSERT_GT( grad_scale, 0.0 );

    double max_pot_dev = 0.0, max_grad_dev = 0.0;
    for ( int i = 0; i < n; i++ )
    {
        max_pot_dev = std::max(
            max_pot_dev,
            std::abs( tight_field.pot[i] - wide_field.pot[i] ) / pot_scale );
        const double dx = tight_field.grad[3 * i + 0] -
                          wide_field.grad[3 * i + 0];
        const double dy = tight_field.grad[3 * i + 1] -
                          wide_field.grad[3 * i + 1];
        const double dz = tight_field.grad[3 * i + 2] -
                          wide_field.grad[3 * i + 2];
        max_grad_dev =
            std::max( max_grad_dev,
                      std::sqrt( dx * dx + dy * dy + dz * dz ) / grad_scale );
    }

    // The gradient deviation is PRINTED and not asserted. The exit criterion
    // this body exists for is a statement about the potential, and the
    // overflowing pairs are a different subset of the far field at every rank
    // count, so a pinned gradient bound here would be pinning how the
    // partitioner happened to cut rather than a property of the fallback path.
    std::printf( "[laplace-solve] nprocs=%d op_budget max_pot_dev=%.17g "
                 "max_grad_dev=%.17g tol=%.17g\n",
                 nprocs, max_pot_dev, max_grad_dev,
                 LS_BUDGET_POTENTIAL_TOL );
    std::fflush( stdout );

    EXPECT_LT( max_pot_dev, LS_BUDGET_POTENTIAL_TOL )
        << "np=" << nprocs
        << " the tight-budget solve does not reproduce the default-budget "
           "potential. The per-pair fallback path is different arithmetic "
           "from the operator-table path, not different mathematics, so a "
           "deviation this large is a defect in the fallback and not a "
           "consequence of the budget";
}

} // namespace LaplaceSolveTest

//---------------------------------------------------------------------------//

TEST( LaplaceSolve, bitForBitArtifacts )
{
    LaplaceSolveTest::testBitForBitArtifacts<TEST_MEMSPACE, TEST_EXECSPACE>();
}

//---------------------------------------------------------------------------//

TEST( LaplaceSolve, crossRankAgreement )
{
    LaplaceSolveTest::testCrossRankAgreement<TEST_MEMSPACE, TEST_EXECSPACE>();
}

//---------------------------------------------------------------------------//

TEST( LaplaceSolve, matchesDirectSum )
{
    LaplaceSolveTest::testMatchesDirectSum<TEST_MEMSPACE, TEST_EXECSPACE>();
}

//---------------------------------------------------------------------------//

TEST( LaplaceSolve, opTableByteBudget )
{
    LaplaceSolveTest::testOpTableByteBudget<TEST_MEMSPACE, TEST_EXECSPACE>();
}

//---------------------------------------------------------------------------//

} // end namespace Test
