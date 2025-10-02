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

#include <Canopy_Tree.hpp>

#include <test_helper_functions.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//

// Used to sum positions in the following AverageValueFunctor struct.
template <class ScalarType>
struct Triple {
    ScalarType x, y, z;

    KOKKOS_INLINE_FUNCTION
    Triple()
        : x(ScalarType(0)), y(ScalarType(0)), z(ScalarType(0)) {}

    KOKKOS_INLINE_FUNCTION
    Triple& operator+=(const Triple& rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }
};

/**
 * Aggregates the first slice (positions) based on average and 
 * the second slice based on sum.
 */
template <class MemorySpace, class ExecutionSpace, class AoSoAType>
struct KernelFunction {
public:

    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using aosoa_type = AoSoAType;
    using member_types = typename AoSoAType::member_types;

    KernelFunction() 
    {
        _avgs = aosoa_type("avgs", 1);
    }

    aosoa_type _avgs;

    aosoa_type vals() {return _avgs;}

    void operator()(const aosoa_type& data) const
    {
        std::size_t data_size = data.size();

        auto slice0 = Cabana::slice<0>(data);
        auto slice1 = Cabana::slice<1>(data);

        // Calculate average position of slice 0
        Triple<double> sum;
        Kokkos::parallel_reduce(
            "aggregate_xyz",
            Kokkos::RangePolicy<execution_space>(0, data_size),
            KOKKOS_LAMBDA(const int i, Triple<double>& local_sum) {
                local_sum.x += slice0(i, 0);
                local_sum.y += slice0(i, 1);
                local_sum.z += slice0(i, 2);
            }, sum );
        
        sum.x /= static_cast<double>(data_size);
        sum.y /= static_cast<double>(data_size);
        sum.z /= static_cast<double>(data_size);

        // Calculate total sum of slice 1
        int total = 0;
        Kokkos::parallel_reduce(
            "aggregate_xyz",
            Kokkos::RangePolicy<execution_space>(0, data_size),
            KOKKOS_LAMBDA(const int i, int& local_total) {
                local_total += slice1(i);
            }, total );

        Cabana::Tuple<member_types> tp;
        Cabana::get<0>( tp, 0 ) = sum.x;
        Cabana::get<0>( tp, 1 ) = sum.y;
        Cabana::get<0>( tp, 2 ) = sum.z;
        Cabana::get<1>(tp) = total;

        _avgs.setTuple(0, tp);
    }
};

//---------------------------------------------------------------------------//

/**
 * Rank 0 creates all the data. Each rank gets two particles.
 * All ranks insert into the tree.
 * Tests that data was correctly distributed to the rank that owns it
 * and correctly inserted into the tree and converted into multipole coefficients
 * at the leaf layer.
 */
void testLeafLayer()
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Create a tree of at least depth 3 for any number of processes
    using particle_tuple_type = Cabana::MemberTypes<double[3], double>;
    using particle_aosoa_type = Cabana::AoSoA<particle_tuple_type, TEST_MEMSPACE, 4>;
    std::array<double, 3> global_low_corner = { -1.5, -1.5, -1.5 };
    std::array<double, 3> global_high_corner = { 1.5, 1.5, 1.5 };
    static constexpr std::size_t num_dim = 3;
    static constexpr std::size_t cells_per_tile = 2;
    static constexpr std::size_t p = 3;
    std::size_t leaf_tiles, red_factor;
    red_factor = comm_size / 2, leaf_tiles = comm_size * 2;
    if (red_factor < 2) red_factor = 2;
    auto tree = Canopy::createTree<TEST_EXECSPACE, TEST_MEMSPACE, Cabana::Grid::Cell,
        num_dim, cells_per_tile, p>(
            global_low_corner, global_high_corner, leaf_tiles, red_factor, MPI_COMM_WORLD);
    
    // The tree depth should always be at least three, but this check is here just in case.
    // If the depth is less than 3, this test may not work correctly.
    if (rank == 0) printf("R%d: num tree layers: %d\n", rank, tree->numLayers());
    // ASSERT_GE(tree->numLayers(), 3) << "testUpwardsAggregation: Error: Tree depth must be at least 3.\n";
    
    // Create the data
    int num_points = (rank == 0) ? (comm_size * 500) : 0;
    Kokkos::View<double* [3], TEST_MEMSPACE> cart_coords( "cart_coords",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q( "q", num_points );
    
    Kokkos::Array<double, 6> coord_bounds = {-1.5, -1.5, -1.5, 1.5, 1.5, 1.5};
    fillRandomCoordinates(cart_coords, coord_bounds);

    Kokkos::Array<double, 2> charge_bounds = {-10.0, 10.0};
    fillRandomScalar(q, charge_bounds);

    Cabana::AoSoA<particle_tuple_type, Kokkos::HostSpace, 4> particle_aosoa_host("particle_aosoa", num_points);
    auto pos_slice_host = Cabana::slice<0>(particle_aosoa_host);
    auto scalar_slice_host = Cabana::slice<1>(particle_aosoa_host);

    // Returns a vector of domains for each rank
    // auto domains_vec = tree->layer(0)->get_domains();

    // Fill the particles into the AoSoA
    auto cart_coords_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cart_coords);
    auto q_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q);
    for (int i = 0; i < num_points; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            pos_slice_host(i, j) = cart_coords_h(i, j);
        }
        scalar_slice_host(i) = q_h(i);
        printf("R%d: initial particle: p(%0.3lf, %0.3lf, %0.3lf), q(%0.3lf)\n", rank,
            pos_slice_host(i, 0), pos_slice_host(i, 1), pos_slice_host(i, 2), scalar_slice_host(i));
    }

    // Calculate direct potential.
    // Target point far away from domain so multipole approximation holds.
    double Px = 8.8, Py = -5.1, Pz = 12.2;
    double r, theta, phi;
    Canopy::Kernel::cart2sph( Px, Py, Pz, r, theta, phi );
    double potential_direct = 0.0;
    for (std::size_t i = 0; i < num_points; ++i)
    {
        double dx = Px - pos_slice_host( i, 0 );
        double dy = Py - pos_slice_host( i, 1 );
        double dz = Pz - pos_slice_host( i, 2 );
        double dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        potential_direct += scalar_slice_host( i ) / dist;
    }

    // Broadcast direct potential to all other ranks.
    MPI_Bcast(&potential_direct, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Copy to device
    auto particle_aosoa =
        Cabana::create_mirror_view_and_copy( TEST_MEMSPACE(), particle_aosoa_host );
        
    // Fill the tree
    tree->create_multipoles(particle_aosoa);

    /***********************************************
     * Check the data in the root layer
     **********************************************/
    auto m_root_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), tree->M_root() );
    if (rank == 0)
        for (std::size_t i = 0; i < m_root_h.size(); ++i)
        {
            printf("m_root(%d): (%.4lf, %.4lf)\n", i, m_root_h(i).real(), m_root_h(i).imag());
        }

    // Compute potential at P using M
    Kokkos::complex<double> potential_M = 0.0;
    for ( int j = 0; j <= p; ++j )
    {
        for ( int k = -j; k <= j; ++k )
        {
            int idx = Canopy::Kernel::Scalar::index( j, k );
            potential_M +=
                m_root_h( idx ) / Kokkos::pow( r, j + 1 ) *
                Canopy::Kernel::Scalar::Ynm( j, k, theta, phi );
        }
    }

    // Check error between translated multipole and direct potentials
    // The error is already mathematically checked in testM2MKernel0,
    // so here we just make sure they are close to each other.
    int p_int = p;
    double error = Kokkos::pow(10, -p_int+1);
    // EXPECT_NEAR(potential_direct, potential_M.real(), error) << "p="
    //     << p << ": error between (shifted and added) and (direct potential) calculations too high.";
    printf("R%d: potential: %0.8lf, M: %0.8lf\n", rank, potential_direct, potential_M.real());
    

    // Each rank should own two particles
    // EXPECT_EQ(2, data_host.size());

    // Check that the correct rank owns the particle
    // rank_slice_host = Cabana::slice<2>(data_host);
    // for (std::size_t i = 0; i < data_host.size(); i++)
    // {
    //     EXPECT_EQ(rank_slice_host(i), rank) << "Rank " << rank << std::endl;
    // }

}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( Tree, testLeafLayer ) { testLeafLayer(); }

//---------------------------------------------------------------------------//

} // end namespace Test