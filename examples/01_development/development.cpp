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

#include <helpers.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

// vortex sheet methods
// BR rott vortex sheet methods
//

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

/**
 * User-defined struct to collect data from the Octree cells
 */
struct DataGather
{
    template<class SrcAoSoA, class DstAoSoA>
    KOKKOS_INLINE_FUNCTION
    void operator()(const SrcAoSoA& src,
                    const int tid,
                    const int cid,
                    DstAoSoA& dst,
                    const int dstIndex) const
    {
        for (int d = 0; d < 3; d++)
            dst.template get<0>(dstIndex, d) = src.template get<0>(tid, cid, d);
        for (int d = 0; d < 2; d++)
            dst.template get<1>(dstIndex, d) = src.template get<1>(tid, cid, d);
    }
};


template <class MemorySpace, class ExecutionSpace>
void octreeExperiments( std::string view_size )
{
    using execution_space = ExecutionSpace;
    using memory_space = MemorySpace;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    std::cout << "view_size: " << view_size << std::endl;
    std::string z_path = "../view_data/" + view_size + "/" + std::to_string(comm_size) + "/";
    std::string w_path = "../view_data/" + view_size + "/" + std::to_string(comm_size) + "/";
    int mesh_size = 64;
    if (view_size == "small") mesh_size = 16;
    int periodic_flag = 0;
    std::string z_name = Utils::get_filename(rank, comm_size, mesh_size, periodic_flag, 'z');
    std::string w_name = Utils::get_filename(rank, comm_size, mesh_size, periodic_flag, 'w');
    z_path += z_name;
    w_path += w_name;

    auto positions = Utils::read_z<memory_space>(z_path);
    auto vorticities = Utils::read_w<memory_space>(w_path);

    // using T = double;
    // Create global mesh
    // global low corners: random numbers
    std::array<double, 3> global_low_corner = { -1.5, -1.5, -1.5 };
    std::array<double, 3> global_high_corner = { 1.5, 1.5, 1.5 };

    /*
    small size (16x16 mesh), comm_size 4 view extents and owned index space from rocketrig:

    R0: extents: z: 13, 13, w: 13, 13
    R0: d1: [2, 10), d2: [2, 10)

    R1: extents: z: 13, 12, w: 13, 12
    R1: d1: [2, 10), d2: [2, 10)

    R2: extents: z: 12, 13, w: 12, 13
    R2: d1: [2, 10), d2: [2, 10)

    R3: extents: z: 12, 12, w: 12, 12
    R3: d1: [2, 10), d2: [2, 10)

    ------------------------------------
    large size (64x64 mesh), comm_size 4:

    R0: extents: z: 37, 37, w: 37, 37
    R0: d1: [2, 34), d2: [2, 34)

    R1: extents: z: 37, 36, w: 37, 36
    R1: d1: [2, 34), d2: [2, 34)

    R2: extents: z: 36, 37, w: 36, 37
    R2: d1: [2, 34), d2: [2, 34)

    R3: extents: z: 36, 36, w: 36, 36
    R3: d1: [2, 34), d2: [2, 34)

    ------------------------------------
    large size (64x64 mesh), comm_size 16:

    R0: extents: z: 21, 21, w: 21, 21
    R0: d1: [2, 18), d2: [2, 18)

    R1: extents: z: 21, 21, w: 21, 21
    R1: d1: [2, 18), d2: [2, 18)

    R2: extents: z: 21, 21, w: 21, 21
    R2: d1: [2, 18), d2: [2, 18)

    R3: extents: z: 21, 20, w: 21, 20
    R3: d1: [2, 18), d2: [2, 18)

    R4: extents: z: 21, 21, w: 21, 21
    R4: d1: [2, 18), d2: [2, 18)

    R5: extents: z: 21, 21, w: 21, 21
    R5: d1: [2, 18), d2: [2, 18)

    R6: extents: z: 21, 21, w: 21, 21
    R6: d1: [2, 18), d2: [2, 18)

    R7: extents: z: 21, 20, w: 21, 20
    R7: d1: [2, 18), d2: [2, 18)

    R8: extents: z: 21, 21, w: 21, 21
    R8: d1: [2, 18), d2: [2, 18)

    R9: extents: z: 21, 21, w: 21, 21
    R9: d1: [2, 18), d2: [2, 18)

    R10: extents: z: 21, 21, w: 21, 21
    R10: d1: [2, 18), d2: [2, 18)

    R11: extents: z: 21, 20, w: 21, 20
    R11: d1: [2, 18), d2: [2, 18)

    R12: extents: z: 20, 21, w: 20, 21
    R12: d1: [2, 18), d2: [2, 18)

    R13: extents: z: 20, 21, w: 20, 21
    R13: d1: [2, 18), d2: [2, 18)

    R14: extents: z: 20, 21, w: 20, 21
    R14: d1: [2, 18), d2: [2, 18)

    R15: extents: z: 20, 20, w: 20, 20
    R15: d1: [2, 18), d2: [2, 18)

    */
    std::size_t num_particles;
    int istart = 2, jstart = 2;
    int iend, jend;
    if ((view_size == "small") && (comm_size == 4))
    {
        iend = 10; jend = 10;
    }
    else if ((view_size == "large") && (comm_size == 4))
    {
        iend = 34; jend = 34;
    }
    else if ((view_size == "large") && (comm_size == 16))
    {
        iend = 18; jend = 18;
    }
    else
    {
        throw std::runtime_error("Unsuported comm_size and mesh size combo.\n");
    }
    int ni = iend - istart;
    int nj = jend - jstart;
    num_particles = ni * nj;
    
    // Distribute points to the correct 3D rank of owwnership
    // Step 1: Move data from 2D Kokkos views to Canopy AoSoAs
    using particle_tuple_type = Cabana::MemberTypes<double[3], // xyz position
                                                    double[2], // vorticity
                                                    >;
    using particle_aosoa_type = Cabana::AoSoA<particle_tuple_type, memory_space, 4>;
    particle_aosoa_type particle_aosoa("particle_aosoa", num_particles);
    auto pos_slice = Cabana::slice<0>(particle_aosoa);
    auto vort_slice = Cabana::slice<1>(particle_aosoa);

    // Adjust start/end for ghost values from read-in views.
    Kokkos::parallel_for("populate_particles", Kokkos::RangePolicy<execution_space>(0, num_particles),
        KOKKOS_LAMBDA(int particle_id) {
            int i = particle_id / nj + istart;
            int j = particle_id % nj + jstart;

            for (int dim = 0; dim < 3; ++dim) {
                pos_slice(particle_id, dim) = positions(i, j, dim);
                if (dim < 2)
                    vort_slice(particle_id, dim) = vorticities(i, j, dim);
            }
    });

    printf("R%d: num_particles: %d\n", rank, num_particles);
    
    using entity_type = Cabana::Grid::Cell;
    static constexpr std::size_t num_dim = 3;
    static constexpr std::size_t cells_per_tile = 4; // Why does this not compile when != 4
    // The slice that us used to determine which cell the particle belongs in
    // Here, slice 0 is the x/y/z position.
    static constexpr std::size_t cell_slice_id = 0; 
    std::size_t root_tiles, red_factor;
    root_tiles = 4, red_factor = 2;
    Canopy::Tree<execution_space, memory_space, particle_tuple_type, entity_type,
        num_dim, cells_per_tile, cell_slice_id>
            tree(global_low_corner, global_high_corner, num_particles, red_factor, root_tiles, MPI_COMM_WORLD);

    
    Kokkos::View<int*, memory_space> owner3D("owner3D", num_particles);
    tree.mapParticles(pos_slice, owner3D, num_particles, 0);
    // for (size_t i = 0; i < num_particles; i++)
    // {
    //     if (rank == 0) printf("R%d: p(%0.2lf, %0.2lf, %0.2lf) -> R%d\n", rank, pos_slice(i, 0), pos_slice(i, 1), pos_slice(i, 2), owner3D(i));
    // }

    Cabana::Distributor<MemorySpace> distributor(MPI_COMM_WORLD, owner3D);
    Cabana::migrate( distributor, particle_aosoa );
    num_particles = particle_aosoa.size();
    pos_slice = Cabana::slice<0>(particle_aosoa);
    // vort_slice = Cabana::slice<1>(particle_aosoa);
    // for (size_t i = 0; i < particle_aosoa.size(); i++)
    // {
    //     if (rank == 0) printf("R%d: p(%0.2lf, %0.2lf, %0.2lf)\n", rank, pos_slice(i, 0), pos_slice(i, 1), pos_slice(i, 2));
    // }
    
    std::vector<int> dof = {3, 2};
    AverageValueFunctor<memory_space, execution_space, particle_aosoa_type> avg_val_functor(dof);

    tree.aggregateDataUp(particle_aosoa, avg_val_functor);

    
}

//---------------------------------------------------------------------------//
// main
int main( int argc, char* argv[] )
{
    using exec_space = Kokkos::DefaultHostExecutionSpace;
    using memory_space = Kokkos::HostSpace;

    std::string mesh_size;
    
    if (argc > 1) {
        // Access the first command-line argument (index 1) as a C-style string
        // char* mesh_size = argv[1];
        // std::cout << "Mesh size: " << mesh_size << std::endl;

        // Convert the C-style string to a std::string
        mesh_size = argv[1];
        // std::cout << "std::string argument: " << mesh_size << std::endl;
    } else {
        std::cout << "No command-line argument provided." << std::endl;
        std::cout << "Usage: " << argv[0] << " <mesh_size (small or large)>" << std::endl;
        return 0;
    }
    
    // Initialize environment
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    // sparseExperiments<memory_space, exec_space>(Cabana::Grid::Cell());

    octreeExperiments<memory_space, exec_space>(mesh_size);

    // Finalize
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
