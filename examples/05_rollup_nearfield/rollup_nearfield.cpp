// ===========================================================================
// 05_rollup_nearfield — near-field / P2P cost diagnostic under clustering
//
// Phase-1 diagnostic for the near-field blowup described in
// tasks/near-field-softening.md. The FMM far field is an UNSOFTENED 1/r
// expansion, so near_softening_factor (k) forces any cell pair within k*eps
// onto the softened P2P path. As a system clusters, that fixed physical radius
// captures ever more pairs and the near-field cost grows like
// density*(k*eps)^3.
//
// This miniapp reproduces the mechanism cheaply with a gravitational COLD
// COLLAPSE: a cold (zero-velocity) ball collapses under the library's own 1/r
// gradient kernel, growing density monotonically into a softened core. The
// metric (P2P particle-pairs vs. clustering) depends only on how tightly
// points pile up, not on cluster geometry, so the collapse is a faithful
// surrogate for a rolled-up vortex sheet's near-field cost.
//
// Each step it emits one CSV row of near-field counters (from
// Solver::near_field_stats()) vs. simulation time. The experiment is a sweep
// over k (-k 0,2,4 with identical IC/eps): the gap between the
// n_p2p_particle_pairs curves is the softening-attributable near-field cost.
// ===========================================================================

#include "Canopy_ExampleSpaces.hpp"
#include "Canopy_Helpers.hpp"
#include "Canopy_Solver.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <mpi.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
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

using DataTypes = Cabana::MemberTypes<double[3], double[N_COMPS], double[3]>;

using MemorySpace = CanopyExample::MemorySpace;
using ExecutionSpace = CanopyExample::ExecutionSpace;

using AoSoA_t = Cabana::AoSoA<DataTypes, MemorySpace>;
using Solver_t =
    Canopy::Solver<MemorySpace, ExecutionSpace, double, P_ORDER, N_COMPS>;

// Cold-collapse initial condition: particles uniformly distributed in a ball
// of radius R0 centered at the origin, equal masses summing to total_mass, and
// zero velocity. With self-gravity the ball contracts into a dense softened
// core, driving the clustering that stresses the near_softening floor.
void generate_collapse_ball( AoSoA_t& particles, int num_particles, double R0,
                             double total_mass, int rank )
{
    particles.resize( num_particles );
    auto positions = Cabana::slice<Position>( particles );
    auto masses = Cabana::slice<Mass>( particles );
    auto velocities = Cabana::slice<Velocity>( particles );

    const double m_each =
        total_mass / static_cast<double>( num_particles > 0 ? num_particles : 1 );

    using RandPool = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
    RandPool pool( static_cast<uint64_t>( 42 + rank ) );

    Kokkos::parallel_for(
        "generate_collapse_ball",
        Kokkos::RangePolicy<ExecutionSpace>( 0, num_particles ),
        KOKKOS_LAMBDA( const int i ) {
            auto gen = pool.get_state();
            // Uniform in the ball: r = R0 * u^(1/3), direction isotropic.
            const double r = R0 * Kokkos::cbrt( gen.drand( 0.0, 1.0 ) );
            const double cos_t = 2.0 * gen.drand( 0.0, 1.0 ) - 1.0;
            const double sin_t = Kokkos::sqrt( 1.0 - cos_t * cos_t );
            const double phi = 2.0 * M_PI * gen.drand( 0.0, 1.0 );
            positions( i, 0 ) = r * sin_t * Kokkos::cos( phi );
            positions( i, 1 ) = r * sin_t * Kokkos::sin( phi );
            positions( i, 2 ) = r * cos_t;
            masses( i, 0 ) = m_each;
            velocities( i, 0 ) = 0.0;
            velocities( i, 1 ) = 0.0;
            velocities( i, 2 ) = 0.0;
            pool.free_state( gen );
        } );
    Kokkos::fence();
}

// Leapfrog-ish explicit integration (same kick-drift form as 03_gravity_solve).
// Acceleration = G * gradient(phi). Non-finite accelerations coast (defense in
// depth) so a stray close encounter cannot degenerate the next tree build.
template <class GradView>
void integrate_particles( AoSoA_t& particles, GradView gradient, int num_local,
                          double dt, double G )
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
    Kokkos::fence();
}

void print_usage( const char* prog )
{
    std::fprintf(
        stderr,
        "Usage: %s [options]\n"
        "\n"
        "Near-field / P2P cost diagnostic via gravitational cold collapse.\n"
        "Emits one CSV row of near-field counters per timestep (see header).\n"
        "Sweep -k 0,2,4 with identical IC to isolate the softening-driven\n"
        "near-field blowup. See tasks/near-field-softening.md.\n"
        "\n"
        "Tree / FMM parameters:\n"
        "  -p N         particles per MPI rank            (default 5000)\n"
        "  -n ncrit     max particles per leaf cell       (default 32)\n"
        "  -d max_depth octree depth cap                  (default 18)\n"
        "  -r repl_depth top tree levels replicated/rank  (default 3)\n"
        "  -i imbal_tol load-imbalance tol for repartition(default 0.10)\n"
        "  -c ncrit_tol leaf-split tolerance on ncrit     (default 0.10)\n"
        "  -b bbox_tol  bounding-box padding fraction     (default 0.10)\n"
        "  -m mac_theta multipole acceptance angle        (default 0.5)\n"
        "\n"
        "Softening / near-field knobs (the focus of this study):\n"
        "  -e eps       Plummer softening length (FIXED, >0 recommended so the\n"
        "               floor is active and constant across a -k sweep)\n"
        "               (default 0.02; <0 selects auto-softening, which drifts\n"
        "               as the box shrinks and is NOT recommended here)\n"
        "  -k factor    near_softening_factor: force pairs within factor*eps to\n"
        "               P2P. The swept knob. 0 = pure geometric MAC.\n"
        "               (default 4.0)\n"
        "\n"
        "Cold-collapse / integration parameters:\n"
        "  -t num_steps number of timesteps               (default 30)\n"
        "  -s dt        timestep size                     (default 5e-3)\n"
        "  -g G         gravitational constant scaling    (default 50.0)\n"
        "  -R R0        initial ball radius               (default 1.0)\n"
        "  -M mass      total mass of the ball            (default 1.0)\n"
        "  -o file      write CSV to file (default stdout)\n"
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
        int num_particles_per_rank = 5000;
        int ncrit = 32;
        int max_depth = 18;
        double ncrit_tol = 0.10;
        double bbox_tol = 0.10;
        int replication_depth = 3;
        double imbalance_tolerance = 0.10;
        double mac_theta = 0.5;
        int num_steps = 30;
        double dt = 5e-3;
        double G = 50.0;
        double softening = 0.02;
        double near_softening_factor = 4.0;
        double R0 = 1.0;
        double total_mass = 1.0;
        std::string csv_path;

        int opt;
        while ( ( opt = getopt( argc, argv,
                                "p:d:r:i:n:t:b:m:s:g:c:e:k:R:M:o:h" ) ) != -1 )
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
            case 'k': near_softening_factor = std::atof( optarg ); break;
            case 'R': R0 = std::atof( optarg ); break;
            case 'M': total_mass = std::atof( optarg ); break;
            case 'o': csv_path = optarg; break;
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

        // Open CSV sink (rank 0 only).
        std::FILE* out = stdout;
        if ( rank == 0 && !csv_path.empty() )
        {
            out = std::fopen( csv_path.c_str(), "w" );
            if ( !out )
            {
                std::fprintf( stderr, "error: cannot open '%s' for writing\n",
                              csv_path.c_str() );
                out = stdout;
            }
        }

        if ( rank == 0 )
        {
            std::fprintf(
                stderr,
                "rollup_nearfield: %d ranks, %d particles/rank, eps=%.4g, "
                "k=%.3g, G=%.3g, dt=%.3g, steps=%d, R0=%.3g, M=%.3g\n",
                nprocs, num_particles_per_rank, softening,
                near_softening_factor, G, dt, num_steps, R0, total_mass );
            // CSV header. eps_over_min_hw and max_leaf_count are the clustering
            // x-axis; n_p2p_particle_pairs is the headline near-field cost.
            std::fprintf( out,
                          "# rollup_nearfield CSV: near-field cost vs. "
                          "clustering (eps=%.6g, k=%.6g, ranks=%d)\n",
                          softening, near_softening_factor, nprocs );
            std::fprintf(
                out,
                "step,time,n_particles,n_leaves,max_depth,max_leaf_count,"
                "min_leaf_halfwidth,eps,eps_over_min_hw,n_p2p_leaf_pairs,"
                "n_p2p_particle_pairs,n_m2l_pairs,n_softening_blocked_pairs,"
                "p2p_pairs_per_particle,solve_time_s\n" );
            std::fflush( out );
        }

        AoSoA_t particles( "particles", num_particles_per_rank );
        generate_collapse_ball( particles, num_particles_per_rank, R0,
                                total_mass, rank );

        Canopy::FmmConfig cfg;
        cfg.ncrit = ncrit;
        cfg.max_depth = max_depth;
        cfg.xmin_tol = cfg.xmax_tol = bbox_tol;
        cfg.ymin_tol = cfg.ymax_tol = bbox_tol;
        cfg.zmin_tol = cfg.zmax_tol = bbox_tol;
        cfg.ncrit_tol = ncrit_tol;
        cfg.replication_depth = replication_depth;
        cfg.imbalance_tolerance = imbalance_tolerance;
        cfg.mac_theta = mac_theta;
        cfg.softening = softening;
        cfg.near_softening_factor = near_softening_factor;

        Solver_t solver( MPI_COMM_WORLD, cfg );
        solver.setup<Position, Mass>( particles, num_particles_per_rank );

        using clock = std::chrono::steady_clock;
        using sec = std::chrono::duration<double>;

        double sim_time = 0.0;
        for ( int step = 0; step < num_steps; step++ )
        {
            MPI_Barrier( MPI_COMM_WORLD );
            const auto t0 = clock::now();
            solver.solve<Position, Mass>( particles, /*compute_gradient=*/true );
            Kokkos::fence();
            MPI_Barrier( MPI_COMM_WORLD );
            const double solve_time = sec( clock::now() - t0 ).count();

            // Near-field counters for the plan this solve used. Per-rank counts
            // are summed; global tree proxies are reduced (max/min) — they are
            // identical on every rank but reducing keeps the example robust.
            const NearFieldStats& nf = solver.near_field_stats();

            long long local_counts[5] = { nf.n_p2p_leaf_pairs,
                                          nf.n_p2p_particle_pairs,
                                          nf.n_m2l_pairs,
                                          nf.n_softening_blocked_pairs,
                                          static_cast<long long>(
                                              solver.num_local_particles() ) };
            long long sum_counts[5];
            MPI_Reduce( local_counts, sum_counts, 5, MPI_LONG_LONG, MPI_SUM, 0,
                        MPI_COMM_WORLD );

            int local_imax[3] = { nf.n_leaves, nf.max_depth, nf.max_leaf_count };
            int max_imax[3];
            MPI_Reduce( local_imax, max_imax, 3, MPI_INT, MPI_MAX, 0,
                        MPI_COMM_WORLD );

            double min_hw = nf.min_leaf_halfwidth;
            double global_min_hw;
            MPI_Reduce( &min_hw, &global_min_hw, 1, MPI_DOUBLE, MPI_MIN, 0,
                        MPI_COMM_WORLD );

            if ( rank == 0 )
            {
                const long long n_p2p_leaf = sum_counts[0];
                const long long n_p2p_part = sum_counts[1];
                const long long n_m2l = sum_counts[2];
                const long long n_blocked = sum_counts[3];
                const long long n_particles = sum_counts[4];
                const double eps_over_hw =
                    ( global_min_hw > 0.0 ) ? softening / global_min_hw : 0.0;
                const double pairs_per_particle =
                    ( n_particles > 0 )
                        ? static_cast<double>( n_p2p_part ) /
                              static_cast<double>( n_particles )
                        : 0.0;
                std::fprintf( out,
                              "%d,%.6g,%lld,%d,%d,%d,%.6g,%.6g,%.6g,%lld,%lld,"
                              "%lld,%lld,%.6g,%.6g\n",
                              step, sim_time, n_particles, max_imax[0],
                              max_imax[1], max_imax[2], global_min_hw, softening,
                              eps_over_hw, n_p2p_leaf, n_p2p_part, n_m2l,
                              n_blocked, pairs_per_particle, solve_time );
                std::fflush( out );
            }

            integrate_particles( particles, solver.gradient(),
                                 solver.num_local_particles(), dt, G );
            solver.auto_maintain<Position, Mass>( particles );
            sim_time += dt;
        }

        if ( rank == 0 && out != stdout )
            std::fclose( out );
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
