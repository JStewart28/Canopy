// ===========================================================================
// scaling_breakdown — per-phase timing of the LaplaceSolve fixture at scale.
//
// The physics and FMM parameters are those of tests/tstLaplaceSolve.hpp
// (P = 6, theta = 0.5, ncrit = 16, replication_depth = 2, softening = 0,
// positions U[0.05, 0.95]^3 and charges U[0.5, 1.5] drawn from
// mt19937(1234 + P) in the fixture's order, global set sliced contiguously
// by rank, symplectic Euler with dt = 1e-4). Three deviations, all forced by
// the particle count:
//
//   * charges are scaled by 600 / N, so the total charge — and with it the
//     mean-field acceleration and the per-step displacement — is the
//     fixture's at every N. Unscaled, N = 32M in the unit box collapses
//     within a few steps.
//   * max_depth defaults to 12 instead of 6: at 500k particles and ncrit 16
//     the tree needs depth ~6-7, at 32M ~8-9.
//   * maintenance is auto_maintain(), not migrate(), as the measurement asks.
//   * softening defaults to the Solver's auto value (-1: 0.1 x the mean
//     inter-particle spacing) instead of 0. Unsoftened, the closest pairs'
//     acceleration grows like N^(1/3) even with the charge scaling; from
//     N ~ 1e5 they are ejected within a few steps, the bounding box grows,
//     and auto_maintain rebuilds into a degenerate tree. -e 0 restores the
//     fixture's value.
//
// Built twice from this file (examples/05_scaling_breakdown/CMakeLists.txt):
//   scaling_breakdown     LaplaceKernel, P = 6 (the fixture's basis)
//   scaling_breakdown_ct  CartesianTaylorBasis, p = 2 (CANOPY_SCALING_CT),
//                         the order of tests/tstCartesianTaylorSolve.hpp's
//                         theta = 0.5 arm. Everything else is the same problem,
//                         except near_softening_factor = 0 as in that test:
//                         this basis expands the softened kernel, so the
//                         floor that keeps LaplaceKernel's unsoftened far
//                         field away from softened pairs is not needed.
//
// Each repetition builds a fresh Solver and records, per call: setup, the
// cold solve, the back-to-back warm solve (warmA), `steps` rounds of
// integrate + auto_maintain (maintain) + solve (tsolve, the last is warmB).
// For every call the driver resets Profiling::timer_totals(), runs the call
// between fences, and prints every key the profiling build accumulated as
// min / max / mean across ranks. Rep 0 is a warm-up; by default it alone
// also prints the library's own tables, for cross-checking.
//
// Output lines (rank 0, stdout), each carrying its full context:
//   ### CANOPY_SCALING begin <ctx>
//   CANOPY_TIMER <ctx> key=<k> min_s=<> max_s=<> mean_s=<>
//   ### CANOPY_SCALING end <ctx> action=<> ilist_builds=<> ...
// ===========================================================================

#include "Canopy_ExampleSpaces.hpp"
#include "Canopy_Solver.hpp"
#ifdef CANOPY_SCALING_CT
#include "Canopy_CartesianTaylorBasis.hpp"
#endif

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

#ifndef CANOPY_ENABLE_PROFILING
#error "scaling_breakdown needs a profiling build (Canopy_ENABLE_PROFILING=ON)"
#endif

enum FieldIdx
{
    Position = 0,
    Charge = 1,
    Velocity = 2,
    GlobalId = 3
};

#ifdef CANOPY_SCALING_CT
static constexpr int P_ORDER = 2;
static constexpr const char* BASIS_NAME = "cartesian_taylor";
template <class S, int P, int C>
using Basis = Canopy::CartesianTaylorBasis<S, P, C>;
#else
static constexpr int P_ORDER = 6;
static constexpr const char* BASIS_NAME = "laplace";
template <class S, int P, int C>
using Basis = Canopy::LaplaceKernel<S, P, C>;
#endif
static constexpr int N_COMPS = 1;
static constexpr int FIXTURE_N = 600;

using DataTypes =
    Cabana::MemberTypes<double[3], double[N_COMPS], double[3], int>;
using MemorySpace = CanopyExample::MemorySpace;
using ExecutionSpace = CanopyExample::ExecutionSpace;
using AoSoA_t = Cabana::AoSoA<DataTypes, MemorySpace>;
using AoSoA_ht = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
using Solver_t = Canopy::Solver<MemorySpace, ExecutionSpace, double, P_ORDER,
                                N_COMPS, Basis>;

struct Context
{
    std::string scaling;
    int np;
    long long n_total;
    std::string state;
    int rep;
    int step;

    std::string str() const
    {
        char buf[256];
        std::snprintf( buf, sizeof( buf ),
                       "scaling=%s np=%d N=%lld state=%s rep=%d step=%d",
                       scaling.c_str(), np, n_total, state.c_str(), rep, step );
        return buf;
    }
};

// Union of timer_totals() keys over all ranks, sorted.
static std::vector<std::string> all_timer_keys( MPI_Comm comm )
{
    std::string mine;
    for ( const auto& kv : Canopy::Profiling::timer_totals() )
        mine += kv.first + '\n';
    int nprocs;
    MPI_Comm_size( comm, &nprocs );
    int len = static_cast<int>( mine.size() );
    std::vector<int> lens( nprocs ), displs( nprocs, 0 );
    MPI_Allgather( &len, 1, MPI_INT, lens.data(), 1, MPI_INT, comm );
    for ( int k = 1; k < nprocs; k++ )
        displs[k] = displs[k - 1] + lens[k - 1];
    std::string all( displs[nprocs - 1] + lens[nprocs - 1], '\0' );
    MPI_Allgatherv( mine.data(), len, MPI_CHAR, &all[0], lens.data(),
                    displs.data(), MPI_CHAR, comm );
    std::set<std::string> keys;
    std::size_t pos = 0, nl;
    while ( ( nl = all.find( '\n', pos ) ) != std::string::npos )
    {
        keys.insert( all.substr( pos, nl - pos ) );
        pos = nl + 1;
    }
    return { keys.begin(), keys.end() };
}

// Reduce `local` to min / max / mean on rank 0.
static void reduce3( MPI_Comm comm, const std::vector<double>& local,
                     std::vector<double>& mn, std::vector<double>& mx,
                     std::vector<double>& mean )
{
    int nprocs;
    MPI_Comm_size( comm, &nprocs );
    const int n = static_cast<int>( local.size() );
    mn.resize( n );
    mx.resize( n );
    mean.resize( n );
    MPI_Reduce( local.data(), mn.data(), n, MPI_DOUBLE, MPI_MIN, 0, comm );
    MPI_Reduce( local.data(), mx.data(), n, MPI_DOUBLE, MPI_MAX, 0, comm );
    MPI_Reduce( local.data(), mean.data(), n, MPI_DOUBLE, MPI_SUM, 0, comm );
    for ( auto& v : mean )
        v /= nprocs;
}

// Run `fn` as one measured call and print its records.
template <class Fn>
static std::string measure( MPI_Comm comm, const Context& ctx,
                            Solver_t& solver, Fn&& fn )
{
    int rank;
    MPI_Comm_rank( comm, &rank );

    Kokkos::fence();
    MPI_Barrier( comm );
    Canopy::Profiling::reset_timer_totals();
    const int ilb0 = solver.downward().interaction_list_build_count();
    const long long kb0 = solver.downward().m2l_op_keys_built_count();
    if ( rank == 0 )
    {
        std::printf( "### CANOPY_SCALING begin %s\n", ctx.str().c_str() );
        std::fflush( stdout );
    }

    const double t0 = MPI_Wtime();
    const std::string action = fn();
    Kokkos::fence();
    const double wall = MPI_Wtime() - t0;

    auto keys = all_timer_keys( comm );
    std::vector<double> local;
    for ( const auto& k : keys )
    {
        auto it = Canopy::Profiling::timer_totals().find( k );
        local.push_back( it == Canopy::Profiling::timer_totals().end()
                             ? 0.0
                             : it->second );
    }
    keys.push_back( "drv_call_wall" );
    local.push_back( wall );

    // Raw per-rank values, so a later merge of several keys into one phase
    // can take its min / max over ranks exactly.
    int nprocs;
    MPI_Comm_size( comm, &nprocs );
    const int nk = static_cast<int>( keys.size() );
    std::vector<double> per_rank( rank == 0 ? nk * nprocs : 0 );
    MPI_Gather( local.data(), nk, MPI_DOUBLE, per_rank.data(), nk, MPI_DOUBLE,
                0, comm );
    if ( rank == 0 )
    {
        const std::string c = ctx.str();
        for ( int i = 0; i < nk; i++ )
        {
            std::printf( "CANOPY_RANKS %s key=%s v=", c.c_str(),
                         keys[i].c_str() );
            for ( int r = 0; r < nprocs; r++ )
                std::printf( r ? ",%.6e" : "%.6e", per_rank[r * nk + i] );
            std::putchar( '\n' );
        }
    }

    // Per-call diagnostics, reduced the same way and printed on the end line.
    const auto& ds = solver.downward();
    int max_depth = 0;
    for ( const auto& c : solver.builder().cells() )
        max_depth = std::max( max_depth, c.depth );
    std::vector<double> diag = {
        double( ds.interaction_list_build_count() - ilb0 ),
        double( ds.m2l_op_keys_built_count() - kb0 ),
        double( ds.m2l_op_cache_size() ),
        double( solver.num_local_particles() ),
        double( ds.total_fallback_pair_count() ) };
    local.insert( local.end(), diag.begin(), diag.end() );

    std::vector<double> mn, mx, mean;
    reduce3( comm, local, mn, mx, mean );
    if ( rank == 0 )
    {
        const std::string c = ctx.str();
        for ( std::size_t i = 0; i < keys.size(); i++ )
            std::printf( "CANOPY_TIMER %s key=%s min_s=%.9e max_s=%.9e "
                         "mean_s=%.9e\n",
                         c.c_str(), keys[i].c_str(), mn[i], mx[i], mean[i] );
        const std::size_t d = keys.size();
        std::printf( "### CANOPY_SCALING end %s action=%s ilist_builds=%g "
                     "op_keys_built_max=%g op_cache_size_max=%g "
                     "n_local_min=%g n_local_max=%g fallback_pairs_max=%g "
                     "cells=%zu max_cell_depth=%d wall_max_s=%.6e\n",
                     c.c_str(), action.c_str(), mx[d], mx[d + 1], mx[d + 2],
                     mn[d + 3], mx[d + 3], mx[d + 4],
                     solver.builder().cells().size(), max_depth, mx[d - 1] );
        std::fflush( stdout );
    }
    return action;
}

static const char* action_name( Solver_t::MaintenanceAction a )
{
    switch ( a )
    {
    case Solver_t::MaintenanceAction::Migrate: return "migrate";
    case Solver_t::MaintenanceAction::Rebalance: return "rebalance";
    case Solver_t::MaintenanceAction::Rebuild: return "rebuild";
    }
    return "unknown";
}

static void print_usage( const char* prog )
{
    std::printf(
        "Usage: %s -N n_total [options]\n"
        "  -N n_total   global particle count (required)\n"
        "  -l label     scaling label for the records (default: run)\n"
        "  -s steps     time steps before warmB (default 5)\n"
        "  -r reps      measured repetitions (default 3)\n"
        "  -w warmups   discarded warm-up repetitions (default 1)\n"
        "  -d max_depth tree depth limit (default 12)\n"
        "  -e soft      softening; < 0 = Solver auto (default -1)\n"
        "  -q           quantize_root_half_width = true (default false)\n"
        "  -L mode      library tables: warmup | all | none (default warmup)\n"
        "  -h           help\n",
        prog );
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );
    {
        MPI_Comm comm = MPI_COMM_WORLD;
        int rank, nprocs;
        MPI_Comm_rank( comm, &rank );
        MPI_Comm_size( comm, &nprocs );

        long long n_total = 0;
        std::string label = "run";
        int steps = 5, reps = 3, warmups = 1, max_depth = 12;
        double softening = -1.0;
        bool quantize = false;
        std::string lib_mode = "warmup";
        int opt;
        while ( ( opt = getopt( argc, argv, "N:l:s:r:w:d:e:qL:h" ) ) != -1 )
        {
            switch ( opt )
            {
            case 'N': n_total = std::atoll( optarg ); break;
            case 'l': label = optarg; break;
            case 's': steps = std::atoi( optarg ); break;
            case 'r': reps = std::atoi( optarg ); break;
            case 'w': warmups = std::atoi( optarg ); break;
            case 'd': max_depth = std::atoi( optarg ); break;
            case 'e': softening = std::atof( optarg ); break;
            case 'q': quantize = true; break;
            case 'L': lib_mode = optarg; break;
            default:
                if ( rank == 0 )
                    print_usage( argv[0] );
                Kokkos::finalize();
                MPI_Finalize();
                return opt == 'h' ? 0 : 1;
            }
        }
        if ( n_total <= 0 || steps < 1 )
        {
            if ( rank == 0 )
                print_usage( argv[0] );
            MPI_Abort( comm, 1 );
        }

        const double dt = 1.0e-4;
        const double q_scale = double( FIXTURE_N ) / double( n_total );
        Canopy::FmmConfig cfg;
        cfg.ncrit = 16;
        cfg.max_depth = max_depth;
        cfg.xmin_tol = cfg.xmax_tol = 0.1;
        cfg.ymin_tol = cfg.ymax_tol = 0.1;
        cfg.zmin_tol = cfg.zmax_tol = 0.1;
        cfg.ncrit_tol = 0.1;
        cfg.replication_depth = 2;
        cfg.imbalance_tolerance = 0.05;
        cfg.mac_theta = 0.5;
        cfg.softening = softening;
        cfg.quantize_root_half_width = quantize;
#ifdef CANOPY_SCALING_CT
        cfg.near_softening_factor = 0.0;
#endif

        if ( rank == 0 )
        {
            std::printf( "### CANOPY_SCALING config scaling=%s np=%d N=%lld "
                         "basis=%s P=%d ncrit=%d max_depth=%d theta=%g "
                         "replication_depth=%d softening=%g "
                         "near_softening_factor=%g quantize_root=%d dt=%g "
                         "q_scale=%.9e steps=%d reps=%d warmups=%d "
                         "lib_tables=%s exec_space=%s\n",
                         label.c_str(), nprocs, n_total, BASIS_NAME, P_ORDER,
                         cfg.ncrit, cfg.max_depth, cfg.mac_theta,
                         cfg.replication_depth, cfg.softening,
                         cfg.near_softening_factor,
                         int( cfg.quantize_root_half_width ), dt, q_scale,
                         steps, reps, warmups, lib_mode.c_str(),
                         ExecutionSpace::name() );
            std::fflush( stdout );
        }

        // The fixture's generator, streamed so a rank keeps only its slice.
        const long long i_begin = rank * n_total / nprocs;
        const long long i_end = ( rank + 1 ) * n_total / nprocs;
        const int n_local0 = static_cast<int>( i_end - i_begin );
        AoSoA_ht init_h( "init_h", n_local0 );
        {
            auto hp = Cabana::slice<Position>( init_h );
            auto hq = Cabana::slice<Charge>( init_h );
            auto hv = Cabana::slice<Velocity>( init_h );
            auto hid = Cabana::slice<GlobalId>( init_h );
            std::mt19937 gen( 1234 + P_ORDER );
            std::uniform_real_distribution<double> pos_dist( 0.05, 0.95 );
            std::uniform_real_distribution<double> q_dist( 0.5, 1.5 );
            for ( long long g = 0; g < i_end; g++ )
            {
                const double x = pos_dist( gen );
                const double y = pos_dist( gen );
                const double z = pos_dist( gen );
                const double q = q_dist( gen );
                if ( g < i_begin )
                    continue;
                const int i = static_cast<int>( g - i_begin );
                hp( i, 0 ) = x;
                hp( i, 1 ) = y;
                hp( i, 2 ) = z;
                hq( i, 0 ) = q * q_scale;
                for ( int d = 0; d < 3; d++ )
                    hv( i, d ) = 0.0;
                hid( i ) = static_cast<int>( g );
            }
        }

        for ( int rep = 0; rep < warmups + reps; rep++ )
        {
            const bool warm_up = rep < warmups;
            Canopy::Profiling::set_reporting(
                lib_mode == "all" || ( lib_mode == "warmup" && warm_up ) );
            // Measured reps are numbered from 1; warm-ups from 0 down.
            const int rep_id = warm_up ? -( warmups - 1 - rep ) : rep - warmups + 1;

            AoSoA_t particles( "particles", n_local0 );
            Cabana::deep_copy( particles, init_h );

            Solver_t solver( comm, cfg );
            Context ctx{ label, nprocs, n_total, "setup", rep_id, 0 };
            auto solve = [&]() {
                solver.template solve<Position, Charge>( particles, true );
                return std::string( "none" );
            };

            measure( comm, ctx, solver, [&]() {
                solver.template setup<Position, Charge>( particles, n_local0 );
                return std::string( "none" );
            } );
            ctx.state = "cold";
            measure( comm, ctx, solver, solve );
            ctx.state = "warmA";
            measure( comm, ctx, solver, solve );

            for ( int step = 1; step <= steps; step++ )
            {
                // The fixture's symplectic Euler, on the latest gradient.
                {
                    const int n_local = solver.num_local_particles();
                    auto x = Cabana::slice<Position>( particles );
                    auto v = Cabana::slice<Velocity>( particles );
                    auto grad = solver.gradient();
                    Kokkos::parallel_for(
                        "scaling_breakdown::integrate",
                        Kokkos::RangePolicy<ExecutionSpace>( 0, n_local ),
                        KOKKOS_LAMBDA( int i ) {
                            for ( int d = 0; d < 3; d++ )
                            {
                                v( i, d ) += dt * grad( i, 0, d );
                                x( i, d ) += dt * v( i, d );
                            }
                        } );
                    Kokkos::fence();
                }
                ctx.step = step;
                ctx.state = "maintain";
                measure( comm, ctx, solver, [&]() {
                    return std::string( action_name(
                        solver.template auto_maintain<Position, Charge>(
                            particles ) ) );
                } );
                ctx.state = ( step == steps ) ? "warmB" : "tsolve";
                measure( comm, ctx, solver, solve );
            }
        }
        Canopy::Profiling::set_reporting( true );

        if ( rank == 0 )
        {
            std::printf( "### CANOPY_SCALING done scaling=%s np=%d N=%lld\n",
                         label.c_str(), nprocs, n_total );
            std::fflush( stdout );
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
