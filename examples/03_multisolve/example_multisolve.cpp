#include "Canopy_Helpers.hpp"
#include "Canopy_Solver.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unistd.h>
#include <vector>

using namespace Canopy;

enum FieldIdx
{
    Position = 0,
    Charge = 1
};

static constexpr int P_ORDER = 6;
static constexpr int N_COMPS = 1;

using DataTypes = Cabana::MemberTypes<double[3], double[N_COMPS]>;

using MemorySpace = Kokkos::CudaSpace;
using ExecutionSpace = Kokkos::Cuda;

using AoSoA_t = Cabana::AoSoA<DataTypes, MemorySpace>;
using Solver_t =
    Canopy::Solver<MemorySpace, ExecutionSpace, double, P_ORDER, N_COMPS>;

void generate_particles( AoSoA_t& particles, int num_particles, int rank )
{
    using AoSoA_h = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
    AoSoA_h particles_h( "particles_h", num_particles );
    auto positions = Cabana::slice<Position>( particles_h );
    auto charges = Cabana::slice<Charge>( particles_h );

    std::mt19937 gen( 42 + rank );
    std::uniform_real_distribution<double> pos_dist( 0.0, 1.0 );
    std::uniform_real_distribution<double> q_dist( -1.0, 1.0 );

    for ( int i = 0; i < num_particles; i++ )
    {
        positions( i, 0 ) = pos_dist( gen );
        positions( i, 1 ) = pos_dist( gen );
        positions( i, 2 ) = pos_dist( gen );
        charges( i, 0 ) = q_dist( gen );
    }

    particles.resize( num_particles );
    Cabana::deep_copy( particles, particles_h );
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    Kokkos::initialize( argc, argv );
    {
        int num_particles_per_rank = 10000;
        int ncrit = 32;
        int max_depth = 15;
        double ncrit_tol = 0.10;
        double bbox_tol = 0.10;
        int replication_depth = 3;
        double imbalance_tolerance = 0.10;
        double mac_theta = 0.5;
        bool compute_gradient = true;
        int num_steps = 3;

        int opt;
        while ( ( opt = getopt( argc, argv, "p:d:r:i:n:t:b:m:" ) ) != -1 )
        {
            switch ( opt )
            {
            case 'p': num_particles_per_rank = std::atoi( optarg ); break;
            case 'd': max_depth = std::atoi( optarg ); break;
            case 'r': replication_depth = std::atoi( optarg ); break;
            case 'i': imbalance_tolerance = std::atof( optarg ); break;
            case 'n': ncrit = std::atoi( optarg ); break;
            case 't': num_steps = std::atoi( optarg ); break;
            case 'c': ncrit_tol = std::atof( optarg ); break;
            case 'b': bbox_tol = std::atof( optarg ); break;
            case 'm': mac_theta = std::atof( optarg ); break;
            default:
                if ( rank == 0 )
                    std::fprintf(
                        stderr,
                        "Usage: %s [-p N] [-d max_depth] [-r repl_depth] "
                        "[-i imbal_tol] [-n ncrit] [-t num_steps] "
                        "[-b bbox_tol] [-m mac_theta]\n",
                        argv[0] );
                MPI_Abort( MPI_COMM_WORLD, 1 );
            }
        }

        if ( num_steps < 1 )
            num_steps = 1;

        if ( rank == 0 )
            std::printf( "Multi-solve FMM (P=%d, NComps=%d): "
                         "%d ranks, %d particles/rank, ncrit=%d, "
                         "max_depth=%d, repl_depth=%d, imbal_tol=%.3g, "
                         "ncrit_tol=%.3g, bbox_tol=%.3g, mac_theta=%.3g, "
                         "num_steps=%d\n",
                         P_ORDER, N_COMPS, nprocs, num_particles_per_rank,
                         ncrit, max_depth, replication_depth,
                         imbalance_tolerance, ncrit_tol, bbox_tol, mac_theta,
                         num_steps );

        AoSoA_t particles( "particles", num_particles_per_rank );
        generate_particles( particles, num_particles_per_rank, rank );

        std::array<double, 3> bb_tol = { bbox_tol, bbox_tol, bbox_tol };

        Solver_t solver( MPI_COMM_WORLD, ncrit, max_depth, bb_tol, ncrit_tol,
                         replication_depth, imbalance_tolerance, mac_theta );

        solver.setup<Position, Charge>( particles, num_particles_per_rank );

        // Repeated solves on the same tree. No inter-step maintenance — the
        // particles are not moved, so the M2L interaction list cached after
        // solve 1 is reused exactly by solves 2..N. This is precisely the
        // workload the build_interaction_list cache is designed to amortize.
        std::vector<double> solve_times( num_steps, 0.0 );
        for ( int step = 0; step < num_steps; step++ )
        {
            MPI_Barrier( MPI_COMM_WORLD );
            const auto t0 = std::chrono::steady_clock::now();

            solver.solve<Position, Charge>( particles, compute_gradient );

            Kokkos::fence();
            MPI_Barrier( MPI_COMM_WORLD );
            const auto t1 = std::chrono::steady_clock::now();
            const double secs =
                std::chrono::duration<double>( t1 - t0 ).count();
            solve_times[step] = secs;

            if ( rank == 0 )
                std::printf( "  solve %d: %.4f s  (interaction-list rebuilds "
                             "so far: %d)\n",
                             step + 1, secs,
                             solver.downward().interaction_list_build_count() );
        }

        if ( rank == 0 && num_steps >= 2 )
        {
            double sum_rest = 0.0;
            for ( int step = 1; step < num_steps; step++ )
                sum_rest += solve_times[step];
            const double mean_rest =
                sum_rest / static_cast<double>( num_steps - 1 );
            std::printf( "\nSummary: solve 1 = %.4f s, "
                         "mean of solves 2..%d = %.4f s, "
                         "speedup = %.2fx\n",
                         solve_times[0], num_steps, mean_rest,
                         mean_rest > 0.0 ? solve_times[0] / mean_rest : 0.0 );
        }

        {
            const long long fb = solver.downward().total_fallback_pair_count();
            const long long tot = solver.downward().total_m2l_pair_count();
            long long g_fb = 0, g_tot = 0;
            MPI_Reduce( &fb, &g_fb, 1, MPI_LONG_LONG, MPI_SUM, 0,
                        MPI_COMM_WORLD );
            MPI_Reduce( &tot, &g_tot, 1, MPI_LONG_LONG, MPI_SUM, 0,
                        MPI_COMM_WORLD );
            if ( rank == 0 )
            {
                const double frac = g_tot > 0
                    ? static_cast<double>( g_fb ) / static_cast<double>( g_tot )
                    : 0.0;
                std::printf( "M2L diag: fallback=%lld total=%lld frac=%.4f\n",
                             g_fb, g_tot, frac );
            }
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
