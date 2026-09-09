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
// tstGolden — the bit-for-bit gate for the solid-harmonic far field.
//
// One fixed FMM configuration is solved and four internal artifacts are
// compared against committed reference data on their *bit patterns*, never
// on a floating-point tolerance:
//
//   locals()           — hash of every (real, imag) bit pattern, plus extents
//   M2L operator table — hash over the realized key columns, plus extents
//   A_{n,m} table      — full bit patterns, one uint64_t per entry
//   realized key list  — hash of the sorted keys — and n_unique_ops, full
//
// Reference data is keyed by (nprocs, rank), because the particle seed is
// `1234 + rank * 31 + P`: the particle set, and everything derived from it,
// is a function of both the rank count and the rank. Ranks 1-6 is 21 sets,
// all held in one committed file under tests/data.
//
// The purpose of this test is to attribute a bitwise difference to exactly
// one artifact across a refactor that is supposed to change no bits. It is
// not an accuracy test: tstMultiSolve owns accuracy.
//
// Regenerating the reference data
// -------------------------------
// Set CANOPY_GOLDEN_REGENERATE to a directory. Each rank then writes its own
// record to <dir>/golden_np<N>_rank<R>.part and the comparison is skipped.
// Run once at each of ranks 1-6 and concatenate the parts in (nprocs, rank)
// order into tests/data/golden_solid_harmonic_P6.txt, under the existing
// header. With the variable unset — the default, and what CTest runs — the
// test always compares and never writes reference data, so a later change
// cannot silently re-baseline itself.
// ===========================================================================

#include "Canopy_Helpers.hpp"
#include "Canopy_Solver.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <unistd.h>

#include <algorithm>
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
#include <utility>
#include <vector>

namespace Test
{
//---------------------------------------------------------------------------//

using namespace Canopy;

namespace GoldenTest
{

// ---------------------------------------------------------------------------
// The frozen configuration. Do not change any of these: the committed
// reference data is only meaningful for this exact configuration, and
// changing one of them means regenerating all 21 sets.
// ---------------------------------------------------------------------------
static constexpr int GOLDEN_P = 6;
static constexpr int GOLDEN_NCOMPS = 1;
static constexpr int GOLDEN_NUM_PARTICLES_PER_RANK = 400;
static constexpr double GOLDEN_MAC_THETA = 0.5;
static constexpr int GOLDEN_NCRIT = 16;
static constexpr int GOLDEN_MAX_DEPTH = 6;

static const char* const GOLDEN_DATA_FILE =
    CANOPY_TEST_DATA_DIR "/golden_solid_harmonic_P6.txt";

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

inline std::string hex64( std::uint64_t v )
{
    std::ostringstream os;
    os << "0x" << std::hex << std::setw( 16 ) << std::setfill( '0' ) << v;
    return os.str();
}

// Total order on M2L keys, so the key-list hash is a function of the key
// *set* and not of the order the classify pass discovered them in. The
// column order of the table is separately covered by the table hash.
template <class Key>
inline bool key_less( const Key& a, const Key& b )
{
    if ( a.dd != b.dd )
        return a.dd < b.dd;
    if ( a.ii != b.ii )
        return a.ii < b.ii;
    if ( a.jj != b.jj )
        return a.jj < b.jj;
    return a.kk < b.kk;
}

// ---------------------------------------------------------------------------
// One golden record: the four artifacts for one (nprocs, rank).
// ---------------------------------------------------------------------------
struct GoldenRecord
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

// Serialize a record in the committed file's format. The format is
// documented in the header of tests/data/golden_solid_harmonic_P6.txt.
inline std::string serialize( const GoldenRecord& r )
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

// Parse the committed file into (nprocs, rank) -> record. Returns false and
// fills `err` if the file cannot be opened or is malformed.
inline bool parse_golden_file( const std::string& path,
                               std::map<std::pair<int, int>, GoldenRecord>& out,
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
        if ( tok != "set" )
        {
            err = "expected 'set', found '" + tok + "'";
            return false;
        }
        GoldenRecord r;
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
        out[{ r.nprocs, r.rank }] = r;
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
    os << base << "/canopy_golden_dump_np" << nprocs << "_rank" << rank << "_"
       << what << ".txt";
    return os.str();
}

// ---------------------------------------------------------------------------
// Artifact extraction. Each reads one artifact off the sweep and folds it
// into `r`.
// ---------------------------------------------------------------------------
template <class DS>
void collect_locals( const DS& ds, GoldenRecord& r )
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
void collect_optab( const DS& ds, GoldenRecord& r )
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
void collect_keys( const DS& ds, GoldenRecord& r )
{
    r.n_unique_ops = ds.m2l_n_unique_ops();
    auto keys = ds.m2l_realized_keys();
    std::sort( keys.begin(), keys.end(), []( const auto& a, const auto& b )
               { return key_less( a, b ); } );
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
void collect_a_table( const DS& ds, GoldenRecord& r )
{
    auto h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                  ds.A_table() );
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
    os << "# sorted realized M2L keys: dd ii jj kk\n";
    for ( const auto& k : keys )
        os << k.dd << " " << k.ii << " " << k.jj << " " << k.kk << "\n";
}

inline void dump_a_table( const GoldenRecord& r, const GoldenRecord& ref,
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
// Run the frozen configuration once and hand the filled record and the live
// sweep to `after`. The sweep stays alive for the callback so a mismatch can
// dump the full arrays without solving a second time.
// ---------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace, class Fn>
void with_golden_solve( Fn&& after )
{
    constexpr int P = GOLDEN_P;
    using Scalar = double;
    using DataTypes = Cabana::MemberTypes<Scalar[3], Scalar[1]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, MemorySpace>;
    using AoSoA_ht = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
    using Solver_t =
        Canopy::Solver<MemorySpace, ExecutionSpace, Scalar, P, GOLDEN_NCOMPS>;
    using DS = typename Solver_t::downward_type;

    // Do step 5: the artifact layouts the committed hashes assume. A layout
    // flip changes the memory order without changing any index-order hash,
    // so it has to be asserted separately and at compile time.
    static_assert( std::is_same<typename DS::coeff_view_type::array_layout,
                                Kokkos::LayoutRight>::value,
                   "DownwardSweep::coeff_view_type must be LayoutRight" );
    static_assert(
        std::is_same<typename DS::m2l_op_table_view_type::array_layout,
                     Kokkos::LayoutLeft>::value,
        "DownwardSweep M2L operator table must be LayoutLeft" );

    GoldenRecord r;
    MPI_Comm_rank( MPI_COMM_WORLD, &r.rank );
    MPI_Comm_size( MPI_COMM_WORLD, &r.nprocs );

    const int num_particles = GOLDEN_NUM_PARTICLES_PER_RANK;

    // Particle generator copied verbatim from tstMultiSolve.hpp:753-767 —
    // seed 1234 + rank * 31 + P, positions uniform on [0.05, 0.95], charges
    // uniform on [-1, 1]. It is inlined inside run_fmm_and_compare there and
    // cannot be called on its own.
    AoSoA_ht particles_h( "particles_h", num_particles );
    {
        auto hp = Cabana::slice<0>( particles_h );
        auto hq = Cabana::slice<1>( particles_h );
        std::mt19937 gen( 1234 + r.rank * 31 + P );
        std::uniform_real_distribution<double> pos_dist( 0.05, 0.95 );
        std::uniform_real_distribution<double> q_dist( -1.0, 1.0 );
        for ( int i = 0; i < num_particles; i++ )
        {
            hp( i, 0 ) = pos_dist( gen );
            hp( i, 1 ) = pos_dist( gen );
            hp( i, 2 ) = pos_dist( gen );
            hq( i, 0 ) = q_dist( gen );
        }
    }
    AoSoA_t particles( "particles", num_particles );
    Cabana::deep_copy( particles, particles_h );

    Canopy::FmmConfig cfg;
    cfg.ncrit = GOLDEN_NCRIT;
    cfg.max_depth = GOLDEN_MAX_DEPTH;
    cfg.xmin_tol = cfg.xmax_tol = 0.1;
    cfg.ymin_tol = cfg.ymax_tol = 0.1;
    cfg.zmin_tol = cfg.zmax_tol = 0.1;
    cfg.ncrit_tol = 0.1;
    cfg.replication_depth = 2;
    cfg.imbalance_tolerance = 0.05;
    cfg.mac_theta = GOLDEN_MAC_THETA;
    cfg.softening = 0.0;

    Solver_t solver( MPI_COMM_WORLD, cfg );
    solver.template setup<0, 1>( particles, num_particles );
    solver.template solve<0, 1>( particles, /*compute_gradient=*/true );

    const DS& ds = solver.downward();

    collect_locals( ds, r );
    collect_optab( ds, r );
    collect_keys( ds, r );
    collect_a_table( ds, r );
    r.fallback_pairs = ds.total_fallback_pair_count();

    // Always echo the measurements. `ctest -V` on this test is how the
    // per-rank-count n_unique_ops and fallback counts reach the progress log.
    std::printf( "[golden] nprocs=%d rank=%d n_unique_ops=%d "
                 "fallback_pairs=%lld locals_ext=(%zu,%zu,%zu) "
                 "optab_ext=(%zu,%zu,%zu) a_extent=%zu\n",
                 r.nprocs, r.rank, r.n_unique_ops, r.fallback_pairs,
                 r.locals_ext[0], r.locals_ext[1], r.locals_ext[2],
                 r.optab_ext[0], r.optab_ext[1], r.optab_ext[2],
                 r.a_bits.size() );
    std::fflush( stdout );

    after( r, ds );
}

// ---------------------------------------------------------------------------
// The test body.
// ---------------------------------------------------------------------------
template <class MemorySpace, class ExecutionSpace>
void testGoldenBitForBit()
{
    with_golden_solve<MemorySpace, ExecutionSpace>(
        []( const GoldenRecord& r, const auto& ds )
        {
            const char* regen = std::getenv( "CANOPY_GOLDEN_REGENERATE" );
            if ( regen && regen[0] != '\0' )
            {
                std::ostringstream path;
                path << regen << "/golden_np" << r.nprocs << "_rank" << r.rank
                     << ".part";
                std::ofstream os( path.str() );
                ASSERT_TRUE( os.good() )
                    << "CANOPY_GOLDEN_REGENERATE is set but " << path.str()
                    << " cannot be written";
                os << serialize( r );
                os.close();
                GTEST_SKIP() << "regeneration mode: wrote " << path.str()
                             << "; comparison skipped";
            }

            const std::string tag = "(nprocs=" + std::to_string( r.nprocs ) +
                                    " rank=" + std::to_string( r.rank ) + ")";

            // Do step 4. A non-zero count means pairs moved onto the per-pair
            // fallback path, which is different arithmetic from the operator
            // path. R4's discriminator and T8's exit criterion both rest on
            // this being zero at P = 6 for this configuration, so it is
            // asserted to be zero rather than pinned to a measured value.
            EXPECT_EQ( 0, r.fallback_pairs )
                << "total_fallback_pair_count() must be 0 for the golden "
                   "configuration "
                << tag;

            std::map<std::pair<int, int>, GoldenRecord> ref_all;
            std::string err;
            ASSERT_TRUE( parse_golden_file( GOLDEN_DATA_FILE, ref_all, err ) )
                << err
                << "\nRegenerate with CANOPY_GOLDEN_REGENERATE=<dir> at each "
                   "of ranks 1-6; see the comment at the top of tstGolden.hpp.";

            auto it = ref_all.find( { r.nprocs, r.rank } );
            ASSERT_NE( it, ref_all.end() ) << "no reference record for " << tag
                                           << " in " << GOLDEN_DATA_FILE;
            const GoldenRecord& ref = it->second;

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

} // namespace GoldenTest

//---------------------------------------------------------------------------//

TEST( Golden, bitForBitArtifacts )
{
    GoldenTest::testGoldenBitForBit<TEST_MEMSPACE, TEST_EXECSPACE>();
}

//---------------------------------------------------------------------------//

} // end namespace Test
