// ============================================================================
// TEMPORARY (debug-nan branch) — offline replay harness for the premature
// full-rollup FMM NaN in Beatnik's FmmBRSolver.
//
// Beatnik's FmmBRSolver, built with BEATNIK_FMM_SNAPSHOT_DEBUG, dumps the exact
// (positions, charges) that each solve() sees for the steps bracketing the
// crash, as one binary file per rank per (step, substep):
//
//     fmm_snapshot_step<NNNN>_sub<B>_rank<RRRR>.bin
//     layout: int32 num_local, then num_local * {px,py,pz,qx,qy,qz} doubles
//
// This harness loads one such (step, sub) snapshot — the union of all rank
// files — into a single Canopy::Solver with the SAME P_ORDER / NComps / config
// as the run, and performs ONE solve(..., compute_gradient=true). With
// CANOPY_NAN_DEBUG compiled in, the solver prints which stage (P2M+M2M /
// M2L+L2L+L2P / P2P) first goes non-finite, reproducing the blow-up without a
// 75-minute queued run. The FMM per-particle gradient is partition-independent
// (a global N-body sum), so distributing the union of particles round-robin
// across however many replay ranks you launch yields the same result.
//
// Remove this example when the premature NaN is resolved.
// ============================================================================

#include "Canopy_ExampleSpaces.hpp"
#include "Canopy_Solver.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <glob.h>
#include <string>
#include <unistd.h>
#include <vector>

using namespace Canopy;

enum FieldIdx
{
    Position = 0,
    Charge = 1
};

// Must match Beatnik::P_ORDER and N_COMPS in src/FmmBRSolver.hpp.
// P_ORDER is overridable at compile time (-DREPLAY_P_ORDER=N) so we can sweep
// the multipole order against a fixed snapshot to test whether raising P cures
// the far-field cancellation error.
#ifndef REPLAY_P_ORDER
#define REPLAY_P_ORDER 10
#endif
static constexpr int P_ORDER = REPLAY_P_ORDER;
static constexpr int N_COMPS = 3;

using DataTypes = Cabana::MemberTypes<double[3], double[N_COMPS]>;
using MemorySpace = CanopyExample::MemorySpace;
using ExecutionSpace = CanopyExample::ExecutionSpace;
using AoSoA_t = Cabana::AoSoA<DataTypes, MemorySpace>;
using Solver_t =
    Canopy::Solver<MemorySpace, ExecutionSpace, double, P_ORDER, N_COMPS>;

void print_usage( const char* prog )
{
    std::fprintf(
        stderr,
        "Usage: %s -S <step> [options]\n"
        "\n"
        "Replays one Beatnik FMM snapshot (step, sub) through a single\n"
        "Canopy solve() to reproduce the premature full-rollup NaN.\n"
        "\n"
        "  -D dir       snapshot directory (default: .)\n"
        "  -S step      snapshot step number (REQUIRED, e.g. 1364)\n"
        "  -B sub       RK substep index 0/1/2 (default 0)\n"
        "\n"
        "Config (defaults match single_mode_debug.in):\n"
        "  -n ncrit     (default 64)\n"
        "  -d max_depth (default 19)\n"
        "  -m mac_theta (default 0.4)\n"
        "  -r repl_depth(default 3)\n"
        "  -i imbal_tol (default 0.20)\n"
        "  -c ncrit_tol (default 0.15)\n"
        "  -e softening (default sqrt(2)=1.41421356; = sqrt(epsilon=2))\n"
        "  -b bbox_tol  uniform bbox padding for all 6 faces (overrides the\n"
        "               per-face deck defaults 0.15 / zmax 0.50)\n"
        "  -h           help\n",
        prog );
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    int rank = 0, nprocs = 1;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    Kokkos::initialize( argc, argv );
    {
        std::string dir = ".";
        int step = -1;
        int sub = 0;

        Canopy::FmmConfig cfg;
        cfg.ncrit = 64;
        cfg.max_depth = 19;
        cfg.mac_theta = 0.4;
        cfg.replication_depth = 3;
        cfg.imbalance_tolerance = 0.20;
        cfg.ncrit_tol = 0.15;
        cfg.softening = std::sqrt( 2.0 );
        // Per-face bbox padding from the deck (zmax larger for rollup headroom).
        cfg.xmin_tol = cfg.xmax_tol = 0.15;
        cfg.ymin_tol = cfg.ymax_tol = 0.15;
        cfg.zmin_tol = 0.15;
        cfg.zmax_tol = 0.50;

        int opt;
        while ( ( opt = getopt( argc, argv, "D:S:B:n:d:m:r:i:c:e:b:h" ) ) != -1 )
        {
            switch ( opt )
            {
            case 'D': dir = optarg; break;
            case 'S': step = std::atoi( optarg ); break;
            case 'B': sub = std::atoi( optarg ); break;
            case 'n': cfg.ncrit = std::atoi( optarg ); break;
            case 'd': cfg.max_depth = std::atoi( optarg ); break;
            case 'm': cfg.mac_theta = std::atof( optarg ); break;
            case 'r': cfg.replication_depth = std::atoi( optarg ); break;
            case 'i': cfg.imbalance_tolerance = std::atof( optarg ); break;
            case 'c': cfg.ncrit_tol = std::atof( optarg ); break;
            case 'e': cfg.softening = std::atof( optarg ); break;
            case 'b':
            {
                const double t = std::atof( optarg );
                cfg.xmin_tol = cfg.xmax_tol = t;
                cfg.ymin_tol = cfg.ymax_tol = t;
                cfg.zmin_tol = cfg.zmax_tol = t;
                break;
            }
            case 'h':
            default:
                if ( rank == 0 )
                    print_usage( argv[0] );
                Kokkos::finalize();
                MPI_Finalize();
                return opt == 'h' ? 0 : 1;
            }
        }

        if ( step < 0 )
        {
            if ( rank == 0 )
                print_usage( argv[0] );
            Kokkos::finalize();
            MPI_Finalize();
            return 1;
        }

        // Enumerate all rank files for this (step, sub), round-robin to ranks.
        char pattern[512];
        std::snprintf( pattern, sizeof( pattern ),
                       "%s/fmm_snapshot_step%04d_sub%d_rank*.bin", dir.c_str(),
                       step, sub );
        std::vector<std::string> files;
        {
            glob_t g;
            std::memset( &g, 0, sizeof( g ) );
            if ( glob( pattern, 0, nullptr, &g ) == 0 )
                for ( size_t i = 0; i < g.gl_pathc; ++i )
                    files.emplace_back( g.gl_pathv[i] );
            globfree( &g );
        }
        std::sort( files.begin(), files.end() );

        if ( files.empty() )
        {
            if ( rank == 0 )
                std::fprintf( stderr,
                              "[nan_replay] no snapshot files match %s\n",
                              pattern );
            Kokkos::finalize();
            MPI_Finalize();
            return 1;
        }

        // Read this rank's round-robin subset into a host buffer.
        std::vector<double> host; // flattened, 6 doubles per particle
        long my_count = 0;
        for ( size_t fi = static_cast<size_t>( rank ); fi < files.size();
              fi += static_cast<size_t>( nprocs ) )
        {
            std::FILE* f = std::fopen( files[fi].c_str(), "rb" );
            if ( !f )
            {
                std::fprintf( stderr, "[nan_replay] rank %d: cannot open %s\n",
                              rank, files[fi].c_str() );
                continue;
            }
            std::int32_t n = 0;
            if ( std::fread( &n, sizeof( std::int32_t ), 1, f ) != 1 || n < 0 )
            {
                std::fclose( f );
                continue;
            }
            const size_t base = host.size();
            host.resize( base + static_cast<size_t>( n ) * 6 );
            const size_t got =
                std::fread( host.data() + base, sizeof( double ),
                            static_cast<size_t>( n ) * 6, f );
            std::fclose( f );
            if ( got != static_cast<size_t>( n ) * 6 )
                host.resize( base + got );
            my_count += static_cast<long>( got / 6 );
        }

        long total = 0;
        MPI_Allreduce( &my_count, &total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD );
        if ( rank == 0 )
            std::printf(
                "[nan_replay] step=%d sub=%d files=%zu total_particles=%ld "
                "nprocs=%d | P_ORDER=%d cfg: ncrit=%d max_depth=%d "
                "mac_theta=%.3g softening=%.6g imbal_tol=%.3g ncrit_tol=%.3g\n",
                step, sub, files.size(), total, nprocs, P_ORDER, cfg.ncrit,
                cfg.max_depth, cfg.mac_theta, cfg.softening,
                cfg.imbalance_tolerance, cfg.ncrit_tol );

        const int num_local = static_cast<int>( my_count );

        // Fill a device AoSoA from the host buffer.
        AoSoA_t particles( "replay_particles", num_local );
        {
            auto h_aosoa = Cabana::create_mirror_view_and_copy(
                Kokkos::HostSpace(), particles );
            auto pos = Cabana::slice<Position>( h_aosoa );
            auto chg = Cabana::slice<Charge>( h_aosoa );
            for ( int p = 0; p < num_local; ++p )
            {
                pos( p, 0 ) = host[6 * p + 0];
                pos( p, 1 ) = host[6 * p + 1];
                pos( p, 2 ) = host[6 * p + 2];
                chg( p, 0 ) = host[6 * p + 3];
                chg( p, 1 ) = host[6 * p + 4];
                chg( p, 2 ) = host[6 * p + 5];
            }
            Cabana::deep_copy( particles, h_aosoa );
        }

        Solver_t solver( MPI_COMM_WORLD, cfg );
        solver.setup<Position, Charge>( particles, num_local );
        Kokkos::fence();

        // One solve — CANOPY_NAN_DEBUG prints the first non-finite stage.
        solver.solve<Position, Charge>( particles, /*compute_gradient=*/true );
        Kokkos::fence();

        // Independent post-check on the returned gradient.
        auto grad = solver.gradient();
        const int nl = solver.num_local_particles();
        long bad = 0;
        Kokkos::parallel_reduce(
            "replay_check_gradient",
            Kokkos::RangePolicy<ExecutionSpace>( 0, nl ),
            KOKKOS_LAMBDA( const int i, long& acc ) {
                for ( int c = 0; c < N_COMPS; ++c )
                    for ( int d = 0; d < 3; ++d )
                        if ( !Kokkos::isfinite( grad( i, c, d ) ) )
                            acc += 1;
            },
            bad );
        Kokkos::fence();
        long bad_global = 0;
        MPI_Allreduce( &bad, &bad_global, 1, MPI_LONG, MPI_SUM,
                       MPI_COMM_WORLD );

        // Largest |FMM gradient| component and where it is — the spurious node
        // shows up here as an anomalously large but (at step 1362) still-finite
        // value, long before the 1364 NaN.
        {
            using MaxLoc = Kokkos::MaxLoc<double, int>;
            MaxLoc::value_type fmm_ml;
            Kokkos::parallel_reduce(
                "replay_fmm_maxabs",
                Kokkos::RangePolicy<ExecutionSpace>( 0, nl ),
                KOKKOS_LAMBDA( const int i, MaxLoc::value_type& lv ) {
                    double local = 0.0;
                    for ( int c = 0; c < N_COMPS; ++c )
                        for ( int d = 0; d < 3; ++d )
                        {
                            const double a = Kokkos::fabs( grad( i, c, d ) );
                            if ( a > local )
                                local = a;
                        }
                    if ( local > lv.val )
                    {
                        lv.val = local;
                        lv.loc = i;
                    }
                },
                MaxLoc( fmm_ml ) );
            Kokkos::fence();
            if ( rank == 0 )
                std::printf( "[nan_replay] FMM max|grad|=%.6g at local idx %d\n",
                             fmm_ml.val, fmm_ml.loc );
        }

        // Brute-force all-pairs exact reference (matches Canopy's softened
        // kernel: grad(i,c,d) = -sum_{j!=i} q(j,c) (x_i-x_j)_d (r^2+eps^2)^-3/2).
        // Only when single-rank so every source is local; O(N^2) but feasible
        // for the ~65k debug mesh. Reports the worst FMM-vs-exact divergence to
        // localize which node the FMM mis-evaluates.
        if ( nprocs == 1 )
        {
            auto pos = Cabana::slice<Position>( particles );
            auto chg = Cabana::slice<Charge>( particles );
            const double eps2 = cfg.softening * cfg.softening;

            Kokkos::View<double* [N_COMPS][3], MemorySpace> exact(
                Kokkos::ViewAllocateWithoutInitializing( "exact_grad" ), nl );
            Kokkos::parallel_for(
                "replay_exact_allpairs",
                Kokkos::RangePolicy<ExecutionSpace>( 0, nl ),
                KOKKOS_LAMBDA( const int i ) {
                    const double xi = pos( i, 0 ), yi = pos( i, 1 ),
                                 zi = pos( i, 2 );
                    double g[N_COMPS][3];
                    for ( int c = 0; c < N_COMPS; ++c )
                        g[c][0] = g[c][1] = g[c][2] = 0.0;
                    for ( int j = 0; j < nl; ++j )
                    {
                        if ( j == i )
                            continue;
                        const double dx = xi - pos( j, 0 );
                        const double dy = yi - pos( j, 1 );
                        const double dz = zi - pos( j, 2 );
                        const double r2 = dx * dx + dy * dy + dz * dz;
                        if ( r2 < 1.0e-24 )
                            continue;
                        const double inv_r =
                            1.0 / Kokkos::sqrt( r2 + eps2 );
                        const double inv_r3 = inv_r * inv_r * inv_r;
                        for ( int c = 0; c < N_COMPS; ++c )
                        {
                            const double qj = chg( j, c );
                            g[c][0] -= qj * dx * inv_r3;
                            g[c][1] -= qj * dy * inv_r3;
                            g[c][2] -= qj * dz * inv_r3;
                        }
                    }
                    for ( int c = 0; c < N_COMPS; ++c )
                        for ( int d = 0; d < 3; ++d )
                            exact( i, c, d ) = g[c][d];
                } );
            Kokkos::fence();

            double ex_max = 0.0;
            Kokkos::parallel_reduce(
                "replay_exact_maxabs",
                Kokkos::RangePolicy<ExecutionSpace>( 0, nl ),
                KOKKOS_LAMBDA( const int i, double& m ) {
                    for ( int c = 0; c < N_COMPS; ++c )
                        for ( int d = 0; d < 3; ++d )
                        {
                            const double a = Kokkos::fabs( exact( i, c, d ) );
                            if ( a > m )
                                m = a;
                        }
                },
                Kokkos::Max<double>( ex_max ) );

            using MaxLoc = Kokkos::MaxLoc<double, int>;
            MaxLoc::value_type diff_ml;
            Kokkos::parallel_reduce(
                "replay_fmm_vs_exact",
                Kokkos::RangePolicy<ExecutionSpace>( 0, nl ),
                KOKKOS_LAMBDA( const int i, MaxLoc::value_type& lv ) {
                    double local = 0.0;
                    for ( int c = 0; c < N_COMPS; ++c )
                        for ( int d = 0; d < 3; ++d )
                        {
                            const double a =
                                Kokkos::fabs( grad( i, c, d ) - exact( i, c, d ) );
                            if ( a > local )
                                local = a;
                        }
                    if ( local > lv.val )
                    {
                        lv.val = local;
                        lv.loc = i;
                    }
                },
                MaxLoc( diff_ml ) );
            Kokkos::fence();
            const double diff_max = diff_ml.val;
            const int diff_arg = diff_ml.loc;

            // Dump the worst-diverging node's coordinates + both gradients.
            double wp[3] = { 0, 0, 0 };
            double wf[N_COMPS][3], we[N_COMPS][3];
            if ( diff_arg >= 0 )
            {
                Kokkos::View<double[3], MemorySpace> dpos( "dpos" );
                Kokkos::View<double[N_COMPS][3], MemorySpace> dgf( "dgf" );
                Kokkos::View<double[N_COMPS][3], MemorySpace> dge( "dge" );
                Kokkos::parallel_for(
                    "replay_extract_worst",
                    Kokkos::RangePolicy<ExecutionSpace>( 0, 1 ),
                    KOKKOS_LAMBDA( const int ) {
                        for ( int d = 0; d < 3; ++d )
                            dpos( d ) = pos( diff_arg, d );
                        for ( int c = 0; c < N_COMPS; ++c )
                            for ( int d = 0; d < 3; ++d )
                            {
                                dgf( c, d ) = grad( diff_arg, c, d );
                                dge( c, d ) = exact( diff_arg, c, d );
                            }
                    } );
                Kokkos::fence();
                auto hp = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), dpos );
                auto hf = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), dgf );
                auto he = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), dge );
                for ( int d = 0; d < 3; ++d )
                    wp[d] = hp( d );
                for ( int c = 0; c < N_COMPS; ++c )
                    for ( int d = 0; d < 3; ++d )
                    {
                        wf[c][d] = hf( c, d );
                        we[c][d] = he( c, d );
                    }
            }
            std::printf(
                "[nan_replay] EXACT max|grad|=%.6g | worst FMM-vs-exact "
                "diff=%.6g at idx %d pos=(%.6g,%.6g,%.6g)\n",
                ex_max, diff_max, diff_arg, wp[0], wp[1], wp[2] );
            std::printf( "[nan_replay]   FMM   grad[worst] = "
                         "[%.6g %.6g %.6g | %.6g %.6g %.6g | %.6g %.6g %.6g]\n",
                         wf[0][0], wf[0][1], wf[0][2], wf[1][0], wf[1][1],
                         wf[1][2], wf[2][0], wf[2][1], wf[2][2] );
            std::printf( "[nan_replay]   EXACT grad[worst] = "
                         "[%.6g %.6g %.6g | %.6g %.6g %.6g | %.6g %.6g %.6g]\n",
                         we[0][0], we[0][1], we[0][2], we[1][0], we[1][1],
                         we[1][2], we[2][0], we[2][1], we[2][2] );

#if defined( CANOPY_NAN_DEBUG )
            // Far/near split at the spurious node: re-solve with the Canopy
            // stage mask and read max|grad| at diff_arg. far-only + near-only
            // == full; whichever stage carries the spurious ~1e5 (vs the exact
            // ~few-thousand) is the culprit.
            if ( diff_arg >= 0 )
            {
                auto maxabs_at = [&]( int idx ) -> double {
                    auto g = solver.gradient();
                    double v = 0.0;
                    Kokkos::parallel_reduce(
                        "replay_grad_at",
                        Kokkos::RangePolicy<ExecutionSpace>( 0, 1 ),
                        KOKKOS_LAMBDA( const int, double& m ) {
                            for ( int c = 0; c < N_COMPS; ++c )
                                for ( int d = 0; d < 3; ++d )
                                {
                                    const double a = Kokkos::fabs( g( idx, c, d ) );
                                    if ( a > m )
                                        m = a;
                                }
                        },
                        Kokkos::Max<double>( v ) );
                    Kokkos::fence();
                    return v;
                };

                solver.dbg_skip_far = false;
                solver.dbg_skip_p2p = true; // far-field (M2L+L2L+L2P) only
                solver.solve<Position, Charge>( particles, true );
                Kokkos::fence();
                const double far_only = maxabs_at( diff_arg );

                solver.dbg_skip_far = true; // near-field (P2P) only
                solver.dbg_skip_p2p = false;
                solver.solve<Position, Charge>( particles, true );
                Kokkos::fence();
                const double near_only = maxabs_at( diff_arg );

                solver.dbg_skip_far = false;
                solver.dbg_skip_p2p = false;

                double exact_at = 0.0;
                for ( int c = 0; c < N_COMPS; ++c )
                    for ( int d = 0; d < 3; ++d )
                        exact_at = std::max( exact_at, std::fabs( we[c][d] ) );

                std::printf(
                    "[nan_replay] SPLIT @idx %d: max|grad| far-only=%.6g  "
                    "near-only=%.6g  far+near=%.6g  EXACT=%.6g\n",
                    diff_arg, far_only, near_only, far_only + near_only,
                    exact_at );
                std::printf(
                    "[nan_replay] SPLIT => culprit stage: %s\n",
                    ( far_only > 10.0 * exact_at && far_only > near_only )
                        ? "FAR-FIELD (M2L/L2L/L2P)"
                        : ( near_only > 10.0 * exact_at )
                              ? "NEAR-FIELD (P2P)"
                              : "inconclusive (no single dominant stage)" );

                // Flag the worst node's leaf cell and re-solve so the M2L
                // fallback prints each source contributing to it.
                solver.dbg_skip_far = false;
                solver.dbg_skip_p2p = false;
                const int worst_cell = solver.dbg_cell_of_particle( diff_arg );
                std::printf( "[nan_replay] worst node %d is in leaf cell %d; "
                             "dumping its M2L sources:\n",
                             diff_arg, worst_cell );
                solver.dbg_set_target_cell( worst_cell );
                solver.solve<Position, Charge>( particles, true );
                Kokkos::fence();
                solver.dbg_set_target_cell( -1 );
            }
#endif
        }

        if ( rank == 0 )
            std::printf(
                "[nan_replay] RESULT step=%d sub=%d non-finite gradient "
                "entries=%ld  ==> %s\n",
                step, sub, bad_global,
                bad_global > 0 ? "REPRODUCED NaN" : "clean (no NaN)" );
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
