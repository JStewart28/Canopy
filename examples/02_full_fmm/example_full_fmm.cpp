#include "Canopy_Helpers.hpp"
#include "Canopy_Solver.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <array>
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

void direct_evaluate_global(
    const AoSoA_t& particles, int num_local, MPI_Comm comm,
    std::vector<double>& phi_direct,
    std::vector<std::array<double, 3>>& grad_direct, bool include_gradient )
{
    int nprocs;
    MPI_Comm_size( comm, &nprocs );

    auto positions = Cabana::slice<Position>( particles );
    auto charges = Cabana::slice<Charge>( particles );
    auto h_positions = Canopy::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, positions );
    auto h_charges = Canopy::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, charges );

    std::vector<double> local_pos( 3 * num_local );
    std::vector<double> local_q( num_local );
    for ( int i = 0; i < num_local; i++ )
    {
        local_pos[3 * i + 0] = h_positions( i, 0 );
        local_pos[3 * i + 1] = h_positions( i, 1 );
        local_pos[3 * i + 2] = h_positions( i, 2 );
        local_q[i] = h_charges( i, 0 );
    }

    std::vector<int> counts( nprocs ), displs_pos( nprocs ), displs_q( nprocs );
    MPI_Allgather( &num_local, 1, MPI_INT, counts.data(), 1, MPI_INT, comm );

    int total = 0;
    std::vector<int> counts_pos( nprocs ), counts_q( nprocs );
    for ( int r = 0; r < nprocs; r++ )
    {
        counts_q[r] = counts[r];
        counts_pos[r] = 3 * counts[r];
        displs_q[r] = total;
        displs_pos[r] = 3 * total;
        total += counts[r];
    }

    std::vector<double> all_pos( 3 * total );
    std::vector<double> all_q( total );
    MPI_Allgatherv( local_pos.data(), 3 * num_local, MPI_DOUBLE,
                    all_pos.data(), counts_pos.data(), displs_pos.data(),
                    MPI_DOUBLE, comm );
    MPI_Allgatherv( local_q.data(), num_local, MPI_DOUBLE, all_q.data(),
                    counts_q.data(), displs_q.data(), MPI_DOUBLE, comm );

    phi_direct.assign( num_local, 0.0 );
    if ( include_gradient )
        grad_direct.assign( num_local, { 0.0, 0.0, 0.0 } );

    for ( int i = 0; i < num_local; i++ )
    {
        double xi = h_positions( i, 0 );
        double yi = h_positions( i, 1 );
        double zi = h_positions( i, 2 );
        double phi_i = 0.0;
        double gx = 0.0, gy = 0.0, gz = 0.0;

        for ( int j = 0; j < total; j++ )
        {
            double dx = xi - all_pos[3 * j + 0];
            double dy = yi - all_pos[3 * j + 1];
            double dz = zi - all_pos[3 * j + 2];
            double r2 = dx * dx + dy * dy + dz * dz;
            double r = std::sqrt( r2 );
            if ( r < 1.0e-12 )
                continue;

            const double q_j = all_q[j];
            phi_i += q_j / r;

            if ( include_gradient )
            {
                double inv_r3 = 1.0 / ( r * r * r );
                gx -= q_j * dx * inv_r3;
                gy -= q_j * dy * inv_r3;
                gz -= q_j * dz * inv_r3;
            }
        }

        phi_direct[i] = phi_i;
        if ( include_gradient )
        {
            grad_direct[i][0] = gx;
            grad_direct[i][1] = gy;
            grad_direct[i][2] = gz;
        }
    }
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
            case 't': ncrit_tol = std::atof( optarg ); break;
            case 'b': bbox_tol = std::atof( optarg ); break;
            case 'm': mac_theta = std::atof( optarg ); break;
            default:
                if ( rank == 0 )
                    std::fprintf(
                        stderr,
                        "Usage: %s [-p N] [-d max_depth] [-r repl_depth] "
                        "[-i imbal_tol] [-n ncrit] [-t ncrit_tol] "
                        "[-b bbox_tol] [-m mac_theta]\n",
                        argv[0] );
                MPI_Abort( MPI_COMM_WORLD, 1 );
            }
        }

        if ( rank == 0 )
            std::printf( "Full FMM validation (P=%d, NComps=%d): "
                         "%d ranks, %d particles/rank, ncrit=%d, "
                         "max_depth=%d, repl_depth=%d, imbal_tol=%.3g, "
                         "ncrit_tol=%.3g, bbox_tol=%.3g, mac_theta=%.3g\n",
                         P_ORDER, N_COMPS, nprocs, num_particles_per_rank,
                         ncrit, max_depth, replication_depth,
                         imbalance_tolerance, ncrit_tol, bbox_tol, mac_theta );

        AoSoA_t particles( "particles", num_particles_per_rank );
        generate_particles( particles, num_particles_per_rank, rank );

        std::array<double, 3> bb_tol = { bbox_tol, bbox_tol, bbox_tol };

        Solver_t solver( MPI_COMM_WORLD, ncrit, max_depth, bb_tol, ncrit_tol,
                         replication_depth, imbalance_tolerance, mac_theta );

        solver.setup<Position, Charge>( particles, num_particles_per_rank );
        solver.solve<Position, Charge>( particles, compute_gradient );

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

        int num_local = solver.num_local_particles();

        std::vector<double> phi_direct;
        std::vector<std::array<double, 3>> grad_direct;
        direct_evaluate_global( particles, num_local, MPI_COMM_WORLD,
                                phi_direct, grad_direct, compute_gradient );

        auto h_pot = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, solver.potential() );
        auto h_grad = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, solver.gradient() );

        double l_max_phi_err = 0.0, l_sum_phi_err = 0.0, l_phi_err2 = 0.0;
        double l_max_phi_rel = 0.0, l_phi_norm2 = 0.0;
        double l_max_grad_err = 0.0, l_sum_grad_err = 0.0, l_grad_err2 = 0.0;
        double l_max_grad_rel = 0.0, l_grad_norm2 = 0.0;

        for ( int i = 0; i < num_local; i++ )
        {
            double fmm_phi = h_pot( i, 0 );
            double ref_phi = phi_direct[i];
            double err = std::abs( fmm_phi - ref_phi );
            l_max_phi_err = std::max( l_max_phi_err, err );
            l_sum_phi_err += err;
            l_phi_err2 += err * err;
            double refmag = std::abs( ref_phi );
            if ( refmag > 1e-14 )
                l_max_phi_rel = std::max( l_max_phi_rel, err / refmag );
            l_phi_norm2 += ref_phi * ref_phi;

            if ( compute_gradient )
            {
                double gx = h_grad( i, 0, 0 );
                double gy = h_grad( i, 0, 1 );
                double gz = h_grad( i, 0, 2 );
                double rx = grad_direct[i][0];
                double ry = grad_direct[i][1];
                double rz = grad_direct[i][2];

                double ge = std::sqrt( ( gx - rx ) * ( gx - rx ) +
                                       ( gy - ry ) * ( gy - ry ) +
                                       ( gz - rz ) * ( gz - rz ) );
                l_max_grad_err = std::max( l_max_grad_err, ge );
                l_sum_grad_err += ge;
                l_grad_err2 += ge * ge;

                double rmag = std::sqrt( rx * rx + ry * ry + rz * rz );
                if ( rmag > 1e-14 )
                    l_max_grad_rel = std::max( l_max_grad_rel, ge / rmag );
                l_grad_norm2 += rmag * rmag;
            }
        }

        double max_phi_err, sum_phi_err, phi_err2, max_phi_rel, phi_norm2;
        double max_grad_err, sum_grad_err, grad_err2, max_grad_rel, grad_norm2;
        long long total_n_ll = num_local, total_n;
        MPI_Reduce( &l_max_phi_err, &max_phi_err, 1, MPI_DOUBLE, MPI_MAX, 0,
                    MPI_COMM_WORLD );
        MPI_Reduce( &l_sum_phi_err, &sum_phi_err, 1, MPI_DOUBLE, MPI_SUM, 0,
                    MPI_COMM_WORLD );
        MPI_Reduce( &l_phi_err2, &phi_err2, 1, MPI_DOUBLE, MPI_SUM, 0,
                    MPI_COMM_WORLD );
        MPI_Reduce( &l_max_phi_rel, &max_phi_rel, 1, MPI_DOUBLE, MPI_MAX, 0,
                    MPI_COMM_WORLD );
        MPI_Reduce( &l_phi_norm2, &phi_norm2, 1, MPI_DOUBLE, MPI_SUM, 0,
                    MPI_COMM_WORLD );
        MPI_Reduce( &l_max_grad_err, &max_grad_err, 1, MPI_DOUBLE, MPI_MAX, 0,
                    MPI_COMM_WORLD );
        MPI_Reduce( &l_sum_grad_err, &sum_grad_err, 1, MPI_DOUBLE, MPI_SUM, 0,
                    MPI_COMM_WORLD );
        MPI_Reduce( &l_grad_err2, &grad_err2, 1, MPI_DOUBLE, MPI_SUM, 0,
                    MPI_COMM_WORLD );
        MPI_Reduce( &l_max_grad_rel, &max_grad_rel, 1, MPI_DOUBLE, MPI_MAX, 0,
                    MPI_COMM_WORLD );
        MPI_Reduce( &l_grad_norm2, &grad_norm2, 1, MPI_DOUBLE, MPI_SUM, 0,
                    MPI_COMM_WORLD );
        MPI_Reduce( &total_n_ll, &total_n, 1, MPI_LONG_LONG, MPI_SUM, 0,
                    MPI_COMM_WORLD );

        if ( rank == 0 )
        {
            double avg_phi_err = sum_phi_err / static_cast<double>( total_n );
            double phi_norm = std::sqrt( phi_norm2 );
            double phi_rel_l2 = phi_norm2 > 0.0
                ? std::sqrt( phi_err2 / phi_norm2 ) : 0.0;
            std::printf( "\nPotential (FMM vs direct):\n" );
            std::printf( "  max abs error:      %.6e\n", max_phi_err );
            std::printf( "  avg abs error:      %.6e\n", avg_phi_err );
            std::printf( "  max rel error:      %.6e\n", max_phi_rel );
            std::printf( "  rel L2 error:       %.6e\n", phi_rel_l2 );
            std::printf( "  reference norm:     %.6e\n", phi_norm );

            if ( compute_gradient )
            {
                double avg_grad_err =
                    sum_grad_err / static_cast<double>( total_n );
                double grad_norm = std::sqrt( grad_norm2 );
                double grad_rel_l2 = grad_norm2 > 0.0
                    ? std::sqrt( grad_err2 / grad_norm2 ) : 0.0;
                std::printf( "\nGradient (FMM vs direct):\n" );
                std::printf( "  max abs error:      %.6e\n", max_grad_err );
                std::printf( "  avg abs error:      %.6e\n", avg_grad_err );
                std::printf( "  max rel error:      %.6e\n", max_grad_rel );
                std::printf( "  rel L2 error:       %.6e\n", grad_rel_l2 );
                std::printf( "  reference norm:     %.6e\n", grad_norm );
            }
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
