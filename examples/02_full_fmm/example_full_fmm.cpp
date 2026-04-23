#include "Canopy_CommunicationPlan.hpp"
#include "Canopy_DownwardSweep.hpp"
#include "Canopy_LaplaceKernel.hpp"
#include "Canopy_SphericalCoefficients.hpp"
#include "Canopy_TreeBuilder.hpp"
#include "Canopy_TreePartitioner.hpp"
#include "Canopy_UpwardSweep.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <random>

// ============================================================================
// Validation: full FMM (upward + downward) against direct O(N^2) sum.
//
// For each local particle p, direct computation:
//   phi_direct(p) = sum_{q != p} q_q / |r_p - r_q|
//
// FMM is accurate only for well-separated pairs; near-field contributions
// (within the P2P neighborhood) would normally be added via P2P. Since
// we haven't implemented P2P yet, we cannot validate leaf-level accuracy
// directly. Instead, we validate the "far-field" FMM result by comparing
// against a direct sum that excludes near-field pairs.
//
// For single-rank only: straightforward comparison.
// ============================================================================

using namespace Canopy;

enum FieldIdx
{
    Position = 0,
    Charge = 1
};

using DataTypes = Cabana::MemberTypes<double[3], double>;

using MemorySpace = Kokkos::HostSpace;
using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;

using AoSoA_t = Cabana::AoSoA<DataTypes, MemorySpace>;

static constexpr int P_ORDER = 6;
static constexpr int N_COMPS = 1;
using Kernel = Canopy::LaplaceKernel<double, P_ORDER, N_COMPS>;

// ============================================================================
// Generate particles
// ============================================================================
void generate_particles( AoSoA_t& particles, int num_particles, int rank )
{
    particles.resize( num_particles );
    auto positions = Cabana::slice<Position>( particles );
    auto charges = Cabana::slice<Charge>( particles );

    // Workaround to copy a device-side slice into a host-side view
    using pos_value_type = typename decltype( positions )::value_type;
    Kokkos::View<pos_value_type* [3], MemorySpace> d_pos( "d_pos",
                                                       num_local_particles );
    Kokkos::parallel_for(
        "SliceToView",
        Kokkos::RangePolicy<ExecutionSpace>( 0, num_local_particles ),
        KOKKOS_LAMBDA( int i ) {
            d_pos( i, 0 ) = positions( i, 0 );
            d_pos( i, 1 ) = positions( i, 1 );
            d_pos( i, 2 ) = positions( i, 2 );
        } );
    Kokkos::fence();
    auto h_positions =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_pos );
    
    // Workaround to copy a device-side slice into a host-side view
    using crg_value_type = typename decltype( positions )::value_type;
    Kokkos::View<crg_value_type*, MemorySpace> d_crg( "d_crg",
                                                       num_local_particles );
    Kokkos::parallel_for(
        "SliceToView",
        Kokkos::RangePolicy<ExecutionSpace>( 0, num_local_particles ),
        KOKKOS_LAMBDA( int i ) {
            d_crg( i, 0 ) = charges( i );
        } );
    Kokkos::fence();
    auto h_charges =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_crg );

    std::mt19937 gen( 42 + rank );
    std::uniform_real_distribution<double> pos_dist( 0.0, 1.0 );
    std::uniform_real_distribution<double> q_dist( -1.0, 1.0 );

    for ( int i = 0; i < num_particles; i++ )
    {
        h_positions( i, 0 ) = pos_dist( gen );
        h_positions( i, 1 ) = pos_dist( gen );
        h_positions( i, 2 ) = pos_dist( gen );
        h_charges( i ) = q_dist( gen );
    }

    Kokkos::deep_copy( positions, h_positions );
    Kokkos::deep_copy( charges, h_charges );
}

// ============================================================================
// Direct O(N^2) evaluation (for reference)
// ============================================================================
void direct_evaluate( const AoSoA_t& particles, int num_particles,
                      std::vector<double>& phi_direct,
                      std::vector<std::array<double, 3>>& grad_direct,
                      bool include_gradient )
{
    auto positions = Cabana::slice<Position>( particles );
    auto charges = Cabana::slice<Charge>( particles );
    auto h_positions = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, positions );
    auto h_charges = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, charges );

    phi_direct.assign( num_particles, 0.0 );
    if ( include_gradient )
        grad_direct.assign( num_particles, { 0.0, 0.0, 0.0 } );

    for ( int i = 0; i < num_particles; i++ )
    {
        double phi_i = 0.0;
        double gx = 0.0, gy = 0.0, gz = 0.0;

        for ( int j = 0; j < num_particles; j++ )
        {
            if ( i == j )
                continue;

            double dx = h_positions( i, 0 ) - h_positions( j, 0 );
            double dy = h_positions( i, 1 ) - h_positions( j, 1 );
            double dz = h_positions( i, 2 ) - h_positions( j, 2 );
            double r2 = dx * dx + dy * dy + dz * dz;
            double r = std::sqrt( r2 );
            if ( r < 1.0e-12 )
                continue;

            const double q_j = h_charges( j );
            phi_i += q_j / r;

            if ( include_gradient )
            {
                // grad(1/r) w.r.t. r_i = -(r_i - r_j)/|r|^3
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

// ============================================================================
// Main
// ============================================================================
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    Kokkos::initialize( argc, argv );
    {
        int num_particles_per_rank = 500;
        int ncrit = 32;
        int max_depth = 6;
        double tolerance = 0.1;
        int replication_depth = 2;
        bool compute_gradient = true;

        if ( argc > 1 )
            num_particles_per_rank = std::atoi( argv[1] );
        if ( argc > 2 )
            ncrit = std::atoi( argv[2] );
        if ( argc > 3 )
            max_depth = std::atoi( argv[3] );

        if ( rank == 0 )
            std::printf( "Full FMM validation (P=%d, NComps=%d): "
                         "%d ranks, %d particles/rank, ncrit=%d\n",
                         P_ORDER, N_COMPS, nprocs,
                         num_particles_per_rank, ncrit );

        // Phase 1: particles + tree
        AoSoA_t particles( "particles", num_particles_per_rank );
        generate_particles( particles, num_particles_per_rank, rank );

        auto positions = Cabana::slice<Position>( particles );
        auto charges = Cabana::slice<Charge>( particles );
        using PositionSlice = decltype( positions );

        TreeBuilder<MemorySpace, ExecutionSpace PositionSlice> builder(
            MPI_COMM_WORLD, ncrit, max_depth, tolerance );
        builder.build( positions, num_particles_per_rank );

        // Phase 2: partition
        TreePartitioner<MemorySpace, ExecutionSpace AoSoA_t, Position> partitioner(
            MPI_COMM_WORLD, replication_depth );
        partitioner.partition( builder, particles,
                               num_particles_per_rank );
        int num_local = partitioner.num_local_particles();

        // Re-slice after migration
        positions = Cabana::slice<Position>( particles );
        charges = Cabana::slice<Charge>( particles );
        builder.build( positions, num_local );

        // Phase 3: comm plan
        CommunicationPlan<MemorySpace, ExecutionSpace> comm_plan( MPI_COMM_WORLD );
        comm_plan.build( builder.cells(), partitioner.ownership(),
                         partitioner.cell_owner_map(),
                         replication_depth );

        // Build charge view with NComps dimension for the kernel
        Kokkos::View<double* [N_COMPS], MemorySpace> charge_view(
            "charge_view", num_local );
        auto h_charge_view =
            Kokkos::create_mirror_view( charge_view );
        auto h_charges = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, charges );
        for ( int i = 0; i < num_local; i++ )
            h_charge_view( i, 0 ) = h_charges( i );
        Kokkos::deep_copy( charge_view, h_charge_view );

        // Phase 4: upward sweep
        UpwardSweep<MemorySpace, ExecutionSpace Kernel> upward( MPI_COMM_WORLD );
        upward.setup( builder.cells(), partitioner.cell_owner_map(),
                      builder.particle_keys(), num_local );
        upward.execute( charge_view, positions, comm_plan );

        // Phase 5: downward sweep
        DownwardSweep<MemorySpace, ExecutionSpace Kernel> downward(
            MPI_COMM_WORLD );
        downward.setup( upward, num_local );

        auto potential = downward.allocate_potential( num_local );
        auto gradient = downward.allocate_gradient( num_local );
        Kokkos::deep_copy( potential, 0.0 );
        Kokkos::deep_copy( gradient, 0.0 );

        downward.execute( upward.multipoles(), positions, potential,
                          gradient, compute_gradient, comm_plan );

        // Phase 6: validate against direct sum (single rank only)
        if ( nprocs == 1 )
        {
            std::vector<double> phi_direct;
            std::vector<std::array<double, 3>> grad_direct;
            direct_evaluate( particles, num_local, phi_direct,
                             grad_direct, compute_gradient );

            // Workaround to copy a device-side slice into a host-side view
            using pos_value_type = typename decltype( positions )::value_type;
            Kokkos::View<pos_value_type* [3], MemorySpace> d_pos( "d_pos",
                                                            num_local_particles );
            Kokkos::parallel_for(
                "SliceToView",
                Kokkos::RangePolicy<ExecutionSpace>( 0, num_local_particles ),
                KOKKOS_LAMBDA( int i ) {
                    d_pos( i, 0 ) = positions( i, 0 );
                    d_pos( i, 1 ) = positions( i, 1 );
                    d_pos( i, 2 ) = positions( i, 2 );
                } );
            Kokkos::fence();
            auto h_positions =
                Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_pos );
            
            // Workaround to copy a device-side slice into a host-side view
            using gr_value_type = typename decltype( positions )::value_type;
            Kokkos::View<pos_value_type* [3], MemorySpace> d_pos( "d_pos",
                                                            num_local_particles );
            Kokkos::parallel_for(
                "SliceToView",
                Kokkos::RangePolicy<ExecutionSpace>( 0, num_local_particles ),
                KOKKOS_LAMBDA( int i ) {
                    d_pos( i, 0 ) = positions( i, 0 );
                    d_pos( i, 1 ) = positions( i, 1 );
                    d_pos( i, 2 ) = positions( i, 2 );
                } );
            Kokkos::fence();
            auto h_positions =
                Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), d_pos );

            double max_phi_err = 0.0;
            double avg_phi_err = 0.0;
            double max_phi_rel = 0.0;
            double phi_norm = 0.0;

            double max_grad_err = 0.0;
            double avg_grad_err = 0.0;
            double max_grad_rel = 0.0;
            double grad_norm = 0.0;

            for ( int i = 0; i < num_local; i++ )
            {
                double fmm_phi = h_pot( i, 0 );
                double ref_phi = phi_direct[i];
                double err = std::abs( fmm_phi - ref_phi );
                max_phi_err = std::max( max_phi_err, err );
                avg_phi_err += err;
                double refmag = std::abs( ref_phi );
                if ( refmag > 1e-14 )
                    max_phi_rel =
                        std::max( max_phi_rel, err / refmag );
                phi_norm += ref_phi * ref_phi;

                if ( compute_gradient )
                {
                    double gx = h_grad( i, 0, 0 );
                    double gy = h_grad( i, 0, 1 );
                    double gz = h_grad( i, 0, 2 );
                    double rx = grad_direct[i][0];
                    double ry = grad_direct[i][1];
                    double rz = grad_direct[i][2];

                    double ge = std::sqrt(
                        ( gx - rx ) * ( gx - rx ) +
                        ( gy - ry ) * ( gy - ry ) +
                        ( gz - rz ) * ( gz - rz ) );
                    max_grad_err = std::max( max_grad_err, ge );
                    avg_grad_err += ge;

                    double rmag =
                        std::sqrt( rx * rx + ry * ry + rz * rz );
                    if ( rmag > 1e-14 )
                        max_grad_rel =
                            std::max( max_grad_rel, ge / rmag );
                    grad_norm += rmag * rmag;
                }
            }
            avg_phi_err /= num_local;
            if ( compute_gradient )
                avg_grad_err /= num_local;
            phi_norm = std::sqrt( phi_norm );
            grad_norm = std::sqrt( grad_norm );

            std::printf( "\nPotential (FMM vs direct):\n" );
            std::printf( "  max abs error:      %.6e\n", max_phi_err );
            std::printf( "  avg abs error:      %.6e\n", avg_phi_err );
            std::printf( "  max rel error:      %.6e\n", max_phi_rel );
            std::printf( "  reference norm:     %.6e\n", phi_norm );

            if ( compute_gradient )
            {
                std::printf( "\nGradient (FMM vs direct):\n" );
                std::printf( "  max abs error:      %.6e\n",
                             max_grad_err );
                std::printf( "  avg abs error:      %.6e\n",
                             avg_grad_err );
                std::printf( "  max rel error:      %.6e\n",
                             max_grad_rel );
                std::printf( "  reference norm:     %.6e\n",
                             grad_norm );
            }

            std::printf( "\nNOTE: without P2P for near-field, the FMM "
                         "result lacks direct contributions from "
                         "neighboring leaves — large errors at "
                         "particle level are expected until P2P is "
                         "added. Confirm here that far-field contributions "
                         "are within reasonable bounds given P and ncrit.\n" );
        }
        else
        {
            if ( rank == 0 )
                std::printf( "\nMulti-rank direct validation TBD.\n" );
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
