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
    using particle_tuple_type = Cabana::MemberTypes<double[3], double, int>;
    using particle_aosoa_type = Cabana::AoSoA<particle_tuple_type, TEST_MEMSPACE, 4>;
    std::array<double, 3> global_low_corner = { -1.5, -1.5, -1.5 };
    std::array<double, 3> global_high_corner = { 1.5, 1.5, 1.5 };
    static constexpr std::size_t num_dim = 3;
    static constexpr std::size_t cells_per_tile = 4;
    static constexpr std::size_t p = 3;
    std::size_t leaf_tiles, red_factor;
    red_factor = comm_size / 2, leaf_tiles = comm_size * 4;
    if (red_factor < 2) red_factor = 2;
    auto tree = Canopy::createTree<TEST_EXECSPACE, TEST_MEMSPACE, Cabana::Grid::Cell,
        num_dim, cells_per_tile, p>(
            global_low_corner, global_high_corner, leaf_tiles, red_factor, MPI_COMM_WORLD);
    
    // The tree depth should always be at least three, but this check is here just in case.
    // If the depth is less than 3, this test may not work correctly.
    ASSERT_GE(tree->numLayers(), 3) << "testUpwardsAggregation: Error: Tree depth must be at least 3.\n";
    
    // Create the data
    int num_particles = (rank == 0) ? (comm_size * 2) : 0;
    Cabana::AoSoA<particle_tuple_type, Kokkos::HostSpace, 4> particle_aosoa_host("particle_aosoa", num_particles);
    auto pos_slice_host = Cabana::slice<0>(particle_aosoa_host);
    auto scalar_slice_host = Cabana::slice<1>(particle_aosoa_host);
    auto rank_slice_host = Cabana::slice<2>(particle_aosoa_host);

    // Returns a vector of domains for each rank
    auto domains_vec = tree->layer(0)->get_domains();

    // Fill the particles. Each rank gets one particle at the center of its domain
    // and one particle very close to it. The purpose of the second particle
    // is for it to be close enough that it shares the same cell in a higher layer
    // and it aggregated.
    // Slice 1 contains the rank it should be sent to
    auto layer_0_cell_size = tree->layer(0)->cellSize();
    for (std::size_t i = 0; i < num_particles; i += 2)
    {
        auto darray = domains_vec[i / 2];

        // First particle: center of the domain
        double x = (darray[0] + darray[3]) / 2.0;
        double y = (darray[1] + darray[4]) / 2.0;
        double z = (darray[2] + darray[5]) / 2.0;

        pos_slice_host(i, 0) = x;
        pos_slice_host(i, 1) = y;
        pos_slice_host(i, 2) = z;
        scalar_slice_host(i) = static_cast<double>(i) * 2.2;
        rank_slice_host(i) = i / 2;

        // Second particle: exists one cell away in leaf layer
        double x2 = x + layer_0_cell_size[0] *  1.1;
        double y2 = y + layer_0_cell_size[1] *  1.1;
        double z2 = z + layer_0_cell_size[2] *  1.1;

        pos_slice_host(i+1, 0) = x2;
        pos_slice_host(i+1, 1) = y2;
        pos_slice_host(i+1, 2) = z2;
        scalar_slice_host(i+1) = static_cast<double>(i+1) * -0.9;
        rank_slice_host(i+1) = i / 2;

        // printf("R%d: domain: (%0.2lf, %0.2lf, %0.2lf) to (%0.2lf, %0.2lf, %0.2lf), particle pos: (%0.2lf, %0.2lf, %0.2lf)\n",
        //         i, darray[0], darray[1], darray[2], darray[3], darray[4], darray[5], x, y, z);
        // printf("R%d: rank_slice(%d): %d\n", rank, i, rank_slice_host(i));
    }

    // Calculate direct potential.
    // Target point far away from domain so multipole approximation holds.
    double Px = 8.8, Py = -5.1, Pz = 12.2;
    // double r, theta, phi;
    double potential_direct = 0.0;
    for (std::size_t i = 0; i < num_particles; ++i)
    {
        double dx = Px - pos_slice_host( i, 0 );
        double dy = Py - pos_slice_host( i, 1 );
        double dz = Pz - pos_slice_host( i, 2 );
        double dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        potential_direct += scalar_slice_host( i ) / dist;
    }

    // Copy to device
    auto particle_aosoa =
        Cabana::create_mirror_view_and_copy( TEST_MEMSPACE(), particle_aosoa_host );
        
    // Fill the tree
    tree->create_multipoles(particle_aosoa);

    /***********************************************
     * Check the data in the root layer
     **********************************************/
    
    

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