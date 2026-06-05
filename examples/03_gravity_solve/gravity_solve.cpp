#include "Canopy_ExampleSpaces.hpp"
#include "Canopy_Helpers.hpp"
#include "Canopy_Solver.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <mpi.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>

using namespace Canopy;

enum FieldIdx
{
    Position = 0,
    Mass = 1,
    Velocity = 2
};

static constexpr int P_ORDER = 6;
static constexpr int N_COMPS = 1;

using DataTypes =
    Cabana::MemberTypes<double[3], double[N_COMPS], double[3]>;

using MemorySpace = CanopyExample::MemorySpace;
using ExecutionSpace = CanopyExample::ExecutionSpace;

using AoSoA_t = Cabana::AoSoA<DataTypes, MemorySpace>;
using Solver_t =
    Canopy::Solver<MemorySpace, ExecutionSpace, double, P_ORDER, N_COMPS>;

void generate_particles( AoSoA_t& particles, int num_particles, int rank )
{
    particles.resize( num_particles );
    auto positions = Cabana::slice<Position>( particles );
    auto masses = Cabana::slice<Mass>( particles );
    auto velocities = Cabana::slice<Velocity>( particles );

    using RandPool = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
    RandPool pool( static_cast<uint64_t>( 42 + rank ) );

    Kokkos::parallel_for(
        "generate_particles",
        Kokkos::RangePolicy<ExecutionSpace>( 0, num_particles ),
        KOKKOS_LAMBDA( const int i ) {
            auto gen = pool.get_state();
            positions( i, 0 ) = gen.drand( 0.0, 1.0 );
            positions( i, 1 ) = gen.drand( 0.0, 1.0 );
            positions( i, 2 ) = gen.drand( 0.0, 1.0 );
            masses( i, 0 ) = gen.drand( 0.5, 1.5 );
            velocities( i, 0 ) = 0.0;
            velocities( i, 1 ) = 0.0;
            velocities( i, 2 ) = 0.0;
            pool.free_state( gen );
        } );
    Kokkos::fence();
}

template <class GradView>
void integrate_particles( AoSoA_t& particles, GradView gradient,
                          int num_local, double dt, double G )
{
    auto pos = Cabana::slice<Position>( particles );
    auto vel = Cabana::slice<Velocity>( particles );

    Kokkos::parallel_for(
        "integrate",
        Kokkos::RangePolicy<ExecutionSpace>( 0, num_local ),
        KOKKOS_LAMBDA( const int i ) {
            double ax = G * gradient( i, 0, 0 );
            double ay = G * gradient( i, 0, 1 );
            double az = G * gradient( i, 0, 2 );

            // Defense-in-depth: a non-finite acceleration (e.g. from an
            // unsoftened close encounter) would propagate Inf/NaN into the
            // positions and degenerate the next tree build. Coast the
            // particle this step rather than corrupt its trajectory.
            if ( !( Kokkos::isfinite( ax ) && Kokkos::isfinite( ay ) &&
                    Kokkos::isfinite( az ) ) )
            {
                ax = 0.0;
                ay = 0.0;
                az = 0.0;
            }

            vel( i, 0 ) += ax * dt;
            vel( i, 1 ) += ay * dt;
            vel( i, 2 ) += az * dt;

            pos( i, 0 ) += vel( i, 0 ) * dt;
            pos( i, 1 ) += vel( i, 1 ) * dt;
            pos( i, 2 ) += vel( i, 2 ) * dt;
        } );
}

const char* action_name( Solver_t::MaintenanceAction a )
{
    switch ( a )
    {
    case Solver_t::MaintenanceAction::Migrate: return "Migrate";
    case Solver_t::MaintenanceAction::Rebalance: return "Rebalance";
    case Solver_t::MaintenanceAction::Rebuild: return "Rebuild";
    }
    return "?";
}

void print_usage( const char* prog )
{
    std::fprintf(
        stderr,
        "Usage: %s [options]\n"
        "\n"
        "Gravitational N-body FMM miniapp: builds an adaptive octree, runs\n"
        "FMM + near-field P2P each step, then integrates the particles.\n"
        "\n"
        "Tree / FMM parameters (shared with example_fmm):\n"
        "  -p N         particles per MPI rank            "
        "(default 10000;  rec. 1e4-1e7)\n"
        "  -n ncrit     max particles per leaf cell       "
        "(default 32;     rec. 64-512, larger on GPU)\n"
        "  -d max_depth octree depth cap                  "
        "(default 15;     rec. 10-20, deep enough to hit ncrit)\n"
        "  -r repl_depth top tree levels replicated/rank  "
        "(default 3;      rec. 2-4)\n"
        "  -i imbal_tol load-imbalance tol for repartition"
        " (default 0.10;   rec. 0.05-0.10)\n"
        "  -c ncrit_tol leaf-split tolerance on ncrit     "
        "(default 0.10;   rec. ~0.10)\n"
        "  -b bbox_tol  bounding-box padding fraction     "
        "(default 0.10;   rec. ~0.10)\n"
        "  -m mac_theta multipole acceptance (opening) angle "
        "(default 0.5; rec. 0.5; smaller=more accurate, slower)\n"
        "\n"
        "Time-integration parameters (gravity_solve only):\n"
        "  -t num_steps number of timesteps               "
        "(default 3;      rec. >=1)\n"
        "  -s dt        timestep size                     "
        "(default 1e-4;   rec. small enough for stability)\n"
        "  -g G         gravitational constant scaling    "
        "(default 1.0)\n"
        "  -e softening Plummer softening length eps; force uses r^2+eps^2\n"
        "               (default 0.0 = unsoftened;  rec. >0, ~0.1-1x the\n"
        "               mean inter-particle spacing, to bound close-\n"
        "               encounter forces and avoid runaway/NaN positions)\n"
        "  -h           show this help and exit\n",
        prog );
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
        double dt = 1e-4;
        double G = 1.0;
        double softening = 0.0;

        int opt;
        while ( ( opt = getopt( argc, argv, "p:d:r:i:n:t:b:m:s:g:c:e:h" ) ) !=
                -1 )
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
            case 's': dt = std::atof( optarg ); break;
            case 'g': G = std::atof( optarg ); break;
            case 'e': softening = std::atof( optarg ); break;
            case 'h':
                if ( rank == 0 )
                    print_usage( argv[0] );
                Kokkos::finalize();
                MPI_Finalize();
                return 0;
            default:
                if ( rank == 0 )
                    print_usage( argv[0] );
                MPI_Abort( MPI_COMM_WORLD, 1 );
            }
        }

        if ( num_steps < 1 )
            num_steps = 1;

        if ( rank == 0 )
            std::printf( "Gravitational FMM miniapp (P=%d, NComps=%d): "
                         "%d ranks, %d particles/rank, ncrit=%d, "
                         "max_depth=%d, repl_depth=%d, imbal_tol=%.3g, "
                         "ncrit_tol=%.3g, bbox_tol=%.3g, mac_theta=%.3g, "
                         "num_steps=%d, dt=%.3g, G=%.3g, softening=%.3g\n",
                         P_ORDER, N_COMPS, nprocs, num_particles_per_rank,
                         ncrit, max_depth, replication_depth,
                         imbalance_tolerance, ncrit_tol, bbox_tol, mac_theta,
                         num_steps, dt, G, softening );

        using clock = std::chrono::steady_clock;
        using sec = std::chrono::duration<double>;

        MPI_Barrier( MPI_COMM_WORLD );
        const auto t_setup0 = clock::now();

        AoSoA_t particles( "particles", num_particles_per_rank );
        generate_particles( particles, num_particles_per_rank, rank );

        std::array<double, 3> bb_tol = { bbox_tol, bbox_tol, bbox_tol };

        Solver_t solver( MPI_COMM_WORLD, ncrit, max_depth, bb_tol, ncrit_tol,
                         replication_depth, imbalance_tolerance, mac_theta,
                         softening );

        solver.setup<Position, Mass>( particles, num_particles_per_rank );

        Kokkos::fence();
        MPI_Barrier( MPI_COMM_WORLD );
        const auto t_setup1 = clock::now();
        const double setup_time = sec( t_setup1 - t_setup0 ).count();

        std::vector<double> solve_times( num_steps, 0.0 );
        std::vector<double> action_times( num_steps, 0.0 );
        std::vector<Solver_t::MaintenanceAction> actions(
            num_steps, Solver_t::MaintenanceAction::Migrate );

        MPI_Barrier( MPI_COMM_WORLD );
        const auto t_loop0 = clock::now();

        for ( int step = 0; step < num_steps; step++ )
        {
            MPI_Barrier( MPI_COMM_WORLD );
            const auto t_solve0 = clock::now();

            solver.solve<Position, Mass>( particles, compute_gradient );

            Kokkos::fence();
            MPI_Barrier( MPI_COMM_WORLD );
            const auto t_solve1 = clock::now();
            solve_times[step] = sec( t_solve1 - t_solve0 ).count();

            integrate_particles( particles, solver.gradient(),
                                 solver.num_local_particles(), dt, G );

            Kokkos::fence();

            MPI_Barrier( MPI_COMM_WORLD );
            const auto t_action0 = clock::now();

            actions[step] =
                solver.auto_maintain<Position, Mass>( particles );

            Kokkos::fence();
            MPI_Barrier( MPI_COMM_WORLD );
            action_times[step] = sec( clock::now() - t_action0 ).count();

            if ( rank == 0 )
                std::printf( "  step %3d: action=%-9s  solve=%.4f s  action=%.4f s\n",
                             step + 1, action_name( actions[step] ),
                             solve_times[step], action_times[step] );
        }

        Kokkos::fence();
        MPI_Barrier( MPI_COMM_WORLD );
        const auto t_loop1 = clock::now();
        const double loop_time = sec( t_loop1 - t_loop0 ).count();

        if ( rank == 0 )
        {
            double sum_solve = 0.0;
            for ( double s : solve_times )
                sum_solve += s;

            std::printf( "\nSetup time (particles + solver setup):    "
                         "%.4f s\n",
                         setup_time );
            std::printf( "Total solve time (sum over steps):        "
                         "%.4f s\n",
                         sum_solve );
            std::printf( "Total end-to-end timestep-loop time:      "
                         "%.4f s\n",
                         loop_time );

            std::printf( "\nPer-step share of total solve time:\n" );
            for ( int step = 0; step < num_steps; step++ )
            {
                const double pct = ( sum_solve > 0.0 )
                                       ? 100.0 * solve_times[step] / sum_solve
                                       : 0.0;
                std::printf( "  step %3d: action=%-9s  solve=%.4f s  (%.1f%% of total solve)"
                             "  action=%.4f s\n",
                             step + 1, action_name( actions[step] ),
                             solve_times[step], pct, action_times[step] );
            }
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
