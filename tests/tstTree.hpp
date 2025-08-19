/****************************************************************************
 * Copyright (c) 2018-2023 by the Canopy authors                            *
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
 * Functor that given a dataset of x/y/z positions, calcuates the average 
 * x/y/z position and stores it in _avgs.
 */
template <class MemorySpace, class ExecutionSpace, class AoSoAType>
struct AverageValueFunctor {
public:

    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using aosoa_type = AoSoAType;
    using member_types = typename AoSoAType::member_types;

    AverageValueFunctor(std::vector<int>& dof) 
        : _dof( dof )
    {
        _avgs = aosoa_type("avgs", 1);
    }

    std::vector<int> _dof; // Vector with the depth of frame of each element in the tuple
    aosoa_type _avgs;

    aosoa_type vals() {return _avgs;}

    void operator()(const aosoa_type& data) const
    {
        std::size_t data_size = data.size();

        // We only care about positions right now, the first tuple
        int dof = _dof[0];

        auto slice0 = Cabana::slice<0>(data); // double[3] pos
        auto slice1 = Cabana::slice<1>(data); // double[2] vort

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

        Cabana::Tuple<member_types> tp;
        Cabana::get<0>( tp, 0 ) = sum.x;
        Cabana::get<0>( tp, 1 ) = sum.y;
        Cabana::get<0>( tp, 2 ) = sum.z;

        _avgs.setTuple(0, tp);

        // auto avgs = _avgs;
        // Kokkos::parallel_for("set_avg",
        //     Kokkos::RangePolicy<execution_space>(0, 1),
        //     KOKKOS_LAMBDA(const int i) {
            
            
        //     auto tp = avgs.getTuple(i);

        // });
    }
};

//---------------------------------------------------------------------------//

/**
 * Rank 0 creates all the data. Each rank gets one particle.
 * All ranks insert into the tree.
 * Tests that data was correctly distributed to the rank that owns it
 * and correctly inserted into the tree at the leaf layer.
 */
void testOnePerCellLeaf()
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Create the tree
    using particle_tuple_type = Cabana::MemberTypes<double[3]>;
    using particle_aosoa_type = Cabana::AoSoA<particle_tuple_type, TEST_MEMSPACE, 4>;
    std::array<double, 3> global_low_corner = { -1.5, -1.5, -1.5 };
    std::array<double, 3> global_high_corner = { 1.5, 1.5, 1.5 };
    static constexpr std::size_t num_dim = 3;
    static constexpr std::size_t cells_per_tile = 4;
    static constexpr std::size_t cell_slice_id = 0; 
    std::size_t leaf_tiles, root_tiles, red_factor;
    root_tiles = 4, red_factor = 2, leaf_tiles = 16;
    auto tree_ptr = Canopy::createTree<TEST_EXECSPACE, TEST_MEMSPACE, particle_tuple_type, Cabana::Grid::Cell,
        num_dim, cells_per_tile, cell_slice_id>(
            global_low_corner, global_high_corner, leaf_tiles, red_factor, root_tiles, MPI_COMM_WORLD);

    // Create the data
    int num_particles = (rank == 0) ? comm_size : 0;
    particle_aosoa_type particle_aosoa("particle_aosoa", num_particles);
    auto pos_slice = Cabana::slice<0>(particle_aosoa);

    // Returns a vector of domains for each rank
    auto domains_vec = tree_ptr->layer(0)->get_domains();

    // Convert vector to view so we can access it in the parallel for
    Kokkos::View<double*[6], Kokkos::HostSpace> domains_host("domains", num_particles);
    for (std::size_t i = 0; i < num_particles; i++)
        for (std::size_t j = 0; j < 6; j++)
        {
            auto darray = domains_vec[i];
            domains_host(i, j) = darray[j];
        }
    auto domains = Kokkos::create_mirror_view_and_copy(TEST_MEMSPACE(), domains_host);

    Kokkos::parallel_for(
        "raw_data_fill",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_particles ),
        KOKKOS_LAMBDA( const int p ) {
            // Only rank 0 will be in this loop
            // Set a particle at the center of each rank's domain
            
            
        } );
    Kokkos::fence();



}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( Tree, testOnePerCellLeaf ) { testOnePerCellLeaf(); }

//---------------------------------------------------------------------------//

} // end namespace Test