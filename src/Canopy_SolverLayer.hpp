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

#ifndef CANOPY_TREELAYER_HPP
#define CANOPY_TREELAYER_HPP

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <Canopy_Operators.hpp>

#include <memory>

#include <mpi.h>

#include <limits>
#include <climits>

namespace Canopy
{

// https://repositorio.unesp.br/server/api/core/bitstreams/0e824479-3128-41f7-8cd2-462e9a242c42/content

/**
 * Convert a std::vector to a Kokkos::View
 */
template <class MemorySpace, class ElementType>
Kokkos::View<typename ElementType::value_type*[
                 std::tuple_size<ElementType>::value],
             MemorySpace>
vec2view(const std::vector<ElementType>& vec, const std::string& label)
{
    using value_type = typename ElementType::value_type;
    constexpr std::size_t N = std::tuple_size<ElementType>::value;
    const std::size_t num = vec.size();

    // Allocate the destination view in the target memory space.
    Kokkos::View<value_type*[N], MemorySpace> device_view(label, num);

    auto host_mirror = Kokkos::create_mirror_view(device_view);

    // Fill mirror from std::vector
    for (std::size_t i = 0; i < num; ++i)
        for (std::size_t j = 0; j < N; ++j)
            host_mirror(i, j) = vec[i][j];

    // Copy to device
    Kokkos::deep_copy(device_view, host_mirror);

    return device_view;
}

/**
 * Return the center of a cell given its ijk location
 */
template <class Scalar>
KOKKOS_INLINE_FUNCTION
Kokkos::Array<Scalar, 3>
cellCenter(int i, int j, int k,
            const Kokkos::Array<Scalar, 3>& global_low_corner,
            const Kokkos::Array<Scalar, 3>& cell_size)
{
    Kokkos::Array<Scalar, 3> center;
    center[0] = global_low_corner[0] + (static_cast<Scalar>(i) + 0.5) * cell_size[0];
    center[1] = global_low_corner[1] + (static_cast<Scalar>(j) + 0.5) * cell_size[1];
    center[2] = global_low_corner[2] + (static_cast<Scalar>(k) + 0.5) * cell_size[2];
    return center;
}

/**
 * Given an x/y/z position, return the global ijk location of the cell
 * that owns this position.
 */
template <class Scalar>
KOKKOS_INLINE_FUNCTION
Kokkos::Array<std::size_t, 3>
position2ijk(Scalar x, Scalar y, Scalar z,
            const Kokkos::Array<Scalar, 3>& global_low_corner,
            const Kokkos::Array<Scalar, 3>& cell_size)
{
    Kokkos::Array<Scalar, 3> dx_inv = {
        (Scalar)1.0 / cell_size[0], (Scalar)1.0 / cell_size[1],
        (Scalar)1.0 / cell_size[2] };
    
    Scalar pos[3] = { x - global_low_corner[0],
                      y - global_low_corner[1],
                      z - global_low_corner[2] };
    return Kokkos::Array<std::size_t, 3>{
        static_cast<std::size_t>( std::floor( pos[0] * dx_inv[0] ) ),
        static_cast<std::size_t>( std::floor( pos[1] * dx_inv[1] ) ),
        static_cast<std::size_t>( std::floor( pos[2] * dx_inv[2] ) ) };
}

/**
 * Given a cell ijk location, inner local cutoff from a more coarse layer
 * in terms of the cell ijk of this layer, and the cells per dimension,
 * return the new outer bounds and inner bounds for the local cutoff.
 */
KOKKOS_INLINE_FUNCTION
Kokkos::pair<Kokkos::Array<int, 6>, Kokkos::Array<int, 6>>
cell2Bound(Kokkos::Array<int, 3> cell_ijk,
            const int layer,
            const int start_layer,
            const int start_cpd,
            const int cell_incr_factor)
{
    // Number of refinement steps
    const int num_layers = start_layer - layer;

    // Parent ijk at start_layer
    Kokkos::Array<int, 3> parent = cell_ijk;
    for (int l = 0; l < num_layers; ++l)
    {
        for (int d = 0; d < 3; ++d)
        {
            parent[d] /= cell_incr_factor;
        }
    }

    int cpd = start_cpd;

    Kokkos::Array<int, 6> include;
    Kokkos::Array<int, 6> exclude;

    // Initial include: full domain at start layer
    for (int d = 0; d < 3; ++d)
    {
        include[d]     = 0;
        include[d + 3] = cpd;
    }

    // Initial exclude
    for (int d = 0; d < 3; ++d)
    {
        exclude[d]     = Kokkos::max(parent[d] - 2, 0);
        exclude[d + 3] = Kokkos::min(parent[d] + 3, cpd);
    }

    // -----------------------------
    // Walk down to 'layer'
    // -----------------------------
    for (int L = start_layer - 1; L >= layer; --L)
    {
        // Project include from previous exclude
        for (int d = 0; d < 3; ++d)
        {
            include[d]     = exclude[d]     * cell_incr_factor;
            include[d + 3] = exclude[d + 3] * cell_incr_factor;
        }

        // Update cpd
        cpd *= cell_incr_factor;

        // Recompute parent at this layer
        parent = cell_ijk;
        for (int l = 0; l < L - layer; ++l)
        {
            for (int d = 0; d < 3; ++d)
            {
                parent[d] /= cell_incr_factor;
            }
        }

        // Compute exclude at this layer
        for (int d = 0; d < 3; ++d)
        {
            exclude[d]     = Kokkos::max(parent[d] - 2, 0);
            exclude[d + 3] = Kokkos::min(parent[d] + 3, cpd);
        }
    }

    return Kokkos::pair{include, exclude};
}

// Reduction struct used to create multipole halo outer bounds 
// POD accumulator
struct MinMax6
{
  int v[6];
};

// Reduction functor
template <class ViewType>
struct HaloBoundsReduce
{
  ViewType m2l_bounds;

  using value_type = MinMax6;

  KOKKOS_INLINE_FUNCTION
  void init(value_type& dst) const
  {
    dst.v[0] = dst.v[1] = dst.v[2] = INT_MAX;
    dst.v[3] = dst.v[4] = dst.v[5] = INT_MIN;
  }

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dst, const value_type& src) const
  {
    for (int d = 0; d < 3; ++d)
      dst.v[d] = dst.v[d] < src.v[d] ? dst.v[d] : src.v[d];

    for (int d = 3; d < 6; ++d)
      dst.v[d] = dst.v[d] > src.v[d] ? dst.v[d] : src.v[d];
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, value_type& local) const
  {
    for (int d = 0; d < 3; ++d)
      local.v[d] = local.v[d] < m2l_bounds(i,d) ? local.v[d] : m2l_bounds(i,d);

    for (int d = 3; d < 6; ++d)
      local.v[d] = local.v[d] > m2l_bounds(i,d) ? local.v[d] : m2l_bounds(i,d);
  }
};


template <class SolverType, std::size_t CellPerTileDim>
class SolverLayer
{
  public:
    //! Self type. All TreeLayers in the Octree are of the same type. 
    using tree_layer_type = SolverLayer<SolverType, CellPerTileDim>;

    //! Execution space
    using execution_space = typename SolverType::execution_space;
    //! Memory space.
    using memory_space = typename SolverType::memory_space;
    //! Metadata
    using metadata = typename SolverType::metadata;
    //! Number of dimensions
    static constexpr std::size_t num_space_dim = SolverType::num_space_dim;

    //! Multipole/local expansion cutoff
    static constexpr std::size_t p = SolverType::p;

    //! Sparse partitioner type
    using sparse_partitioner_type = Cabana::Grid::SparseDimPartitioner<memory_space, CellPerTileDim, num_space_dim>;

    //! DataTypes Data types (Cabana::MemberTypes).
    using scalar_type = typename SolverType::scalar_type;
    using complex = typename SolverType::complex;
    using multipole_tuple_type = typename SolverType::multipole_tuple_type;
    using local_tuple_type = typename SolverType::local_tuple_type;
    using multipole_member_types = typename SolverType::multipole_member_types;
    using local_member_types = typename SolverType::local_member_types;
    using multipole_aosoa_type = typename SolverType::multipole_aosoa_type;
    using local_aosoa_type = typename SolverType::local_aosoa_type;

    //! Particle data
    using particle_aosoa_type = typename SolverType::particle_aosoa_type;

    using mesh_type = typename SolverType::mesh_type;

    static constexpr std::size_t cell_per_tile_dim = CellPerTileDim;

    using sparse_map_type = Cabana::Grid::SparseMap<memory_space, cell_per_tile_dim>;

    static constexpr unsigned long long cell_bits_per_tile =
        sparse_map_type::cell_bits_per_tile;
    //! Cell ID mask inside a tile
    static constexpr unsigned long long cell_mask_per_tile =
        sparse_map_type::cell_mask_per_tile;

   using sparse_layout_type =
        Cabana::Grid::Experimental::SparseArrayLayout<multipole_member_types, Cabana::Grid::Node, mesh_type, sparse_map_type>;
    
    using index_map_type = Kokkos::UnorderedMap<Kokkos::Array<std::size_t, 3>, std::size_t, memory_space>;
    
    SolverLayer(const std::array<scalar_type, 3>& global_low_corner,
            const std::array<scalar_type, 3>& global_high_corner,
	        const int tiles_per_dim, const int tile_reduction_factor,
            const int halo_width,
            const int layer_number,
            MPI_Comm comm )
        : _global_low_corner( global_low_corner )
        , _global_high_corner( global_high_corner )
        , _tiles_per_dim( tiles_per_dim )
        , _tile_reduction_factor( tile_reduction_factor )
        , _halo_width( halo_width )
        , _layer_number( layer_number )
        , _cells_per_dim( _tiles_per_dim * cell_per_tile_dim )
        , _comm( comm )
    {
        MPI_Comm_rank( comm, &_rank );
        MPI_Comm_size( comm, &_comm_size );

        _global_num_cell = {
            _cells_per_dim,
            _cells_per_dim,
            _cells_per_dim
            };
        
        // sparse partitioner
        float max_workload_coeff = 1.5;
        int workload_num = _cells_per_dim * _cells_per_dim * _cells_per_dim;
        _num_step_rebalance = 200;
        _max_optimize_iteration = 10;
        _partitioner_ptr = std::make_shared<sparse_partitioner_type>(
            _comm, max_workload_coeff, workload_num, _num_step_rebalance,
            _global_num_cell, _max_optimize_iteration );
        auto ranks_per_dim =
            _partitioner_ptr->ranksPerDimension( comm, _global_num_cell );
        std::array<int, 3> periodic_dims = { 0, 0, 0 };

        // rank-related information
        // Kokkos::Array<int, 3> cart_rank;
        int reordered_cart_ranks = 0;
        // int linear_rank;

        MPI_Cart_create( comm, 3, ranks_per_dim.data(),
                        periodic_dims.data(), reordered_cart_ranks, &_cart_comm );
        
        // Get the Cartesian dimensions (number of ranks in each direction)
        int dims[3];
        int periods[3];
        int cart_coords[3];
        MPI_Cart_get(_cart_comm, 3, dims, periods, cart_coords);

        // Function to compute 1D tile partitioning
        auto compute_partition = [](int total_tiles, int num_parts) {
            std::vector<int> partitions(num_parts + 1);
            for (int i = 0; i <= num_parts; ++i)
                partitions[i] = (i * total_tiles) / num_parts;
            return partitions;
        };

        // Compute tile partitions in each direction
        std::vector<int> x_partition = compute_partition(_tiles_per_dim, dims[0]);
        std::vector<int> y_partition = compute_partition(_tiles_per_dim, dims[1]);
        std::vector<int> z_partition = compute_partition(_tiles_per_dim, dims[2]);

        _partitioner_ptr->initializeRecPartition(x_partition, y_partition, z_partition);

        initialize();
    }

    /**
     * Use the sparse partitioner to initialize the global and local grids, sparse map,
     * and sparse array objects.
     */
    void initialize()
    {
        Kokkos::Profiling::ScopedRegion region("Canopy::SolverLayer::initialize");

        // mesh/grid related initialization
        auto global_mesh = Cabana::Grid::createSparseGlobalMesh(
            _global_low_corner, _global_high_corner, _global_num_cell );
        
        std::array<bool, 3> is_dim_periodic = { false, false, false };
        auto& partitioner_ref = *_partitioner_ptr;
        auto global_grid = Cabana::Grid::createGlobalGrid( _comm, global_mesh,
                                            is_dim_periodic, partitioner_ref );
        auto local_grid =
            Cabana::Grid::Experimental::createSparseLocalGrid( global_grid, _halo_width, cell_per_tile_dim );
        sparse_map_type sparse_map =
            Cabana::Grid::createSparseMap<memory_space, scalar_type, cell_per_tile_dim>( global_mesh, 1.2 );
        // Save sparse map as shared pointer
        _map_ptr = std::make_shared<sparse_map_type>(sparse_map);
        
        // initializeRecPartition(sparse_map);
        _layout_ptr =
            Cabana::Grid::Experimental::createSparseArrayLayout<multipole_member_types>( local_grid, *_map_ptr, Cabana::Grid::Node() );
        
        // Store cell size
        updateCellSize();

        // Get the owned number of cells and the global cell offset
        // each MPI rank on this layer.
        computeCellInfo();

        // Set coefficient view index to 0
        _coefficient_view_index = Kokkos::View<std::size_t, memory_space>("_coefficient_view_index");
        Kokkos::deep_copy(_coefficient_view_index, 0);
    }

    void updateCellSize()
    {
        auto local_grid = _layout_ptr->localGrid();
        auto sparse_mesh = local_grid->globalGrid().globalMesh();
        _cell_size = {sparse_mesh.cellSize( 0 ), sparse_mesh.cellSize( 1 ), sparse_mesh.cellSize( 2 )};
    }

    template <class ParticlePositions>
    void optimizePartition(ParticlePositions positions, std::size_t num_particles)
    {
        Kokkos::Profiling::ScopedRegion region("Canopy::SolverLayer::optimizePartition");

        _partitioner_ptr->optimizePartition( positions, num_particles, _global_low_corner,
            _cell_size[0], _comm);

        // Reinitialize sparse data structures after updating the partition.
        initialize();
    }

     /*!
      \brief Populate _domains, _num_owned_cell_view, and _cell_offsets_view using the current
      partition.
    */
    void computeCellInfo()
    {
        // Get x/y/z domains. The domains are also needed to correctly filter invalid
        // cell counts and offsets
        auto current_partition = _partitioner_ptr->getCurrentPartition();

        // Allocate vectors
        std::vector<Kokkos::Array<int, 3>> cell_offsets_vec(_comm_size);
        std::vector<Kokkos::Array<int, 3>> num_owned_cell_vec(_comm_size);
        std::vector<Kokkos::Array<scalar_type, 6>> domains_vec(_comm_size);

        for (int rank = 0; rank < _comm_size; ++rank)
        {
            int coords[3];
            MPI_Cart_coords(_cart_comm, rank, 3, coords);

            Kokkos::Array<scalar_type, 6> domain;
            Kokkos::Array<int, 3> cell_offsets;
            Kokkos::Array<int, 3> cells_owned;
            for (int d = 0; d < 3; ++d)
            {
                int tile_start = current_partition[d][coords[d]];
                int tile_end   = current_partition[d][coords[d] + 1];

                scalar_type global_min = _global_low_corner[d];
                scalar_type global_max = _global_high_corner[d];
                scalar_type tile_width = (global_max - global_min) / _tiles_per_dim;
                
                // Set domain lower and upper bound for this rank
                domain[d]     = global_min + tile_start * tile_width;
                domain[d + 3] = global_min + tile_end   * tile_width;

                // Set cells owned: (cells per tile) * (tiles owned) 
                cells_owned[d] = (tile_end - tile_start) * cell_per_tile_dim;
                // Set cell offset: (tile_start) * (cells per tile)
                // No owned cells in this dimension = offset is invalid, set to -1
                if (cells_owned[d] == 0)
                {
                    // Set all offsets to -1 and cells owned to 0
                    for (int j = 0; j < 3; ++j)
                    {
                        cell_offsets[j] = -1;
                        cells_owned[j] = 0;
                    }
                    break;
                }
                else
                    cell_offsets[d] = tile_start * cell_per_tile_dim;

            }

            domains_vec[rank] = domain;
            num_owned_cell_vec[rank] = cells_owned;
            cell_offsets_vec[rank] = cell_offsets;
        }

        // Convert vectors to views and save
        _cell_offsets_view = vec2view<memory_space>(cell_offsets_vec, "_cell_offsets_view");
        _num_owned_cell_view = vec2view<memory_space>(num_owned_cell_vec, "_num_owned_cell_view");
        _domains = vec2view<memory_space>(domains_vec, "_domains");
    }

    /**
     * Populate a Kokkos::View that maps to the passed-in AoSoA to the rank
     * each particle should be migrated to based on its x/y/z position.
     */
    template <class ViewType, class PositionSliceType>
    void mapParticles(const PositionSliceType& positions, ViewType& particle_ranks, const int particle_num)
    {
        Kokkos::Profiling::ScopedRegion region("Canopy::SolverLayer::mapParticles");

        using mem_space = typename ViewType::memory_space;
        using exec_space = typename ViewType::execution_space;

        // Get all rank domains on host
        auto domains_host = _domains;
        int num_ranks = domains_host.size();

        // Copy domains to device
        Kokkos::View<scalar_type*[6], mem_space> domain_bounds("domain_bounds", num_ranks);
        auto domain_bounds_host = Kokkos::create_mirror_view(domain_bounds);
        for (int r = 0; r < num_ranks; ++r)
            for (int j = 0; j < 6; ++j)
                domain_bounds_host(r, j) = domains_host[r][j];
        Kokkos::deep_copy(domain_bounds, domain_bounds_host);

        Kokkos::parallel_for(
            "mapParticles",
            Kokkos::RangePolicy<exec_space>(0, particle_num),
            KOKKOS_LAMBDA(const int i) {
                scalar_type xpos = positions(i, 0);
                scalar_type ypos = positions(i, 1);
                scalar_type zpos = positions(i, 2);

                // Linear search: check each rank domain
                for (int r = 0; r < num_ranks; ++r)
                {
                    scalar_type x_lo = domain_bounds(r, 0);
                    scalar_type y_lo = domain_bounds(r, 1);
                    scalar_type z_lo = domain_bounds(r, 2);
                    scalar_type x_hi = domain_bounds(r, 3);
                    scalar_type y_hi = domain_bounds(r, 4);
                    scalar_type z_hi = domain_bounds(r, 5);

                    // Non-inclusive upper bound
                    if (xpos >= x_lo && xpos < x_hi &&
                        ypos >= y_lo && ypos < y_hi &&
                        zpos >= z_lo && zpos < z_hi)
                    {
                        particle_ranks(i) = r;
                        return;
                    }
                }

                // If no domain was found, mark as invalid
                particle_ranks(i) = -1;
            });
    }

    /**
     * Takes an AoSoA of particle data where positions is the first tuple.
     * Saves the cell each particle falls in into cell_map
     *  1.
     *  2. Initialize the correct cells based on particle locations. Track which particles belong
     *      to which cells.
     *  3. Aggregate the data for each cell using the KernelFunction.
     *  4. Inserts the aggregated data into the correct cell on this layer.
     * 
     *  @param data_aosoa: either an AoSoA of particles (for layer 0), or
     *  an AoSoA of cell data (layers > 0). Either way, the first tuple element
     *  must be the position.
     */
    template <class ParticleAoSoA>
    void populateCells(const ParticleAoSoA data_aosoa, const std::size_t start, const std::size_t end)
    {
        Kokkos::Profiling::ScopedRegion region("Canopy::SolverLayer::populateCells");

        // int rank = _rank;
        // int layer_number = _layer_number;

        // printf("L%d: start/end: %d, %d\n", _layer_number, start, end);

        updateCellSize();

        std::size_t num_particles = end - start;

        // Size _ijk2index to hold the number of incoming particles. This is an
        // overestimate. Assuming the particles are evenly distributed, size
        // _ijk2index to hold the number of incoming particles * the
        // communicator size on layers > 0. This is the number of activated
        // cells for layers > 0. At layer 0, assume the number of incoming
        // particles >> the number of incoming particles, and size to the number
        // of incoming particles. We need this overestimate to hold ijk2index
        // maps of ghosted cells.
        // XXX - size this correctly
        _ijk2index.clear();
        if ( _layer_number == 0 )
            _ijk2index.rehash( _cells_per_dim * _cells_per_dim * _cells_per_dim );
        else
            _ijk2index.rehash( num_particles * _comm_size );
        auto ijk2index = _ijk2index;

        // If ParticleAoSoA type is data_aosoa_type, then the positions are the second tuple element.
        // Otherwise they are the first.
        // If positions are the 2nd element, this is not layer 0 and the first element are multipole coefficients.
        // Otherwise the values at each particle are the 2nd coefficient. 
        static constexpr bool is_coeff =
            std::is_same_v<ParticleAoSoA, multipole_aosoa_type>;

        static constexpr std::size_t position_index = is_coeff ? 1 : metadata::pos;
        static constexpr std::size_t data_index = is_coeff ? 0 : metadata::in;
        auto positions = Cabana::slice<position_index>(data_aosoa);
        auto data_slice = Cabana::slice<data_index>(data_aosoa);

        auto map = *_map_ptr;

        auto cell_size = _cell_size;

        // Convert std::array to Kokkos::Array
        Kokkos::Array<scalar_type, 3> low_corner = {_global_low_corner[0], _global_low_corner[1], _global_low_corner[2]};
        
        // Register cells in the sparse map and count the number of cells that will
        // be activated in this layer for sizing data structures. Use the _ijk2index
        // map as a temporary counter.
        Kokkos::parallel_for(
            "registerSparseMap",
            Kokkos::RangePolicy<execution_space>( 0, num_particles ),
            KOKKOS_LAMBDA( const std::size_t index ) {

                auto pid = start + index;
                
                auto cell_activated_ijk =
                    position2ijk(positions( pid, 0 ), positions( pid, 1 ), positions( pid, 2 ),
                                 low_corner, cell_size);                   

                // Register cell in sparse map for load balancing
                map.insertCell( cell_activated_ijk[0], cell_activated_ijk[1],
                                cell_activated_ijk[2] );
                                        
                // Local cell id
                auto cell_id = map.queryCell(cell_activated_ijk[0],
                                         cell_activated_ijk[1],
                                         cell_activated_ijk[2]);

                // Insert into map to count cells activated. Use dummy values
                // because we will clear the map after this.
                auto result = ijk2index.insert(cell_activated_ijk, 0);
                if (!result.success())
                {
                    // Getting here means some particles activate the same cell.
                }
                if (result.success())
                {
                    // Getting here means the cell has not been activated.
                }
            } );

        Kokkos::fence();
        
        const auto num_cells_activated = ijk2index.size();

        // Clear map to remove dummy values
        _ijk2index.clear();

        // Size data structures to hold the number of cells activated
        _multipoles = multipole_aosoa_type("_multipoles", num_cells_activated);
        _locals = local_aosoa_type("_locals", num_cells_activated);
        _m2l_bounds = Kokkos::View<int*[6], memory_space>("_m2l_bounds", num_cells_activated);
        Kokkos::deep_copy(_m2l_bounds, 0);
        
        // Now use the incoming data to compute p2m, if layer 0, or m2m, if layer > 0.
        auto coefficient_view_index = _coefficient_view_index;
        auto multipole_coefficients_slice = Cabana::slice<0>(_multipoles);
        auto cell_center_slice = Cabana::slice<1>(_multipoles);
        auto local_coefficients_slice = Cabana::slice<0>(_locals);
        auto local_cell_ijk_slice = Cabana::slice<1>(_locals);

        // Zero newly-initialized values. Cabana does not guarantee initialization to 0
        Cabana::deep_copy(multipole_coefficients_slice, 0.0);
        Cabana::deep_copy(cell_center_slice, 0.0);
        Cabana::deep_copy(local_coefficients_slice, 0.0);
        Cabana::deep_copy(local_cell_ijk_slice, 0);

        // Define value conflict operator
        // using value_view_type = Kokkos::View<typename index_map_type::value_type*, memory_space>;
        using value_view_type = Kokkos::View<std::size_t*, memory_space>;
        using map_op_type = Kokkos::UnorderedMapInsertOpTypes<value_view_type, std::size_t>;
        using atomic_add_type = typename map_op_type::AtomicAdd;
        atomic_add_type atomic_add;

        Kokkos::parallel_for( "set_cell_keys",
            Kokkos::RangePolicy<execution_space>( 0, num_particles ),
            KOKKOS_LAMBDA( const std::size_t pnum ) {

                auto pid = start + pnum;
                
                auto cell_activated_ijk =
                    position2ijk(positions( pid, 0 ), positions( pid, 1 ), positions( pid, 2 ),
                                 low_corner, cell_size);

                // Try to insert into map with dummy key
                auto result = ijk2index.insert(cell_activated_ijk, 0);

                // If the cell is not in the map, update the key with its index into
                // local/multipole views
                if (result.success())
                {
                    auto index = Kokkos::atomic_fetch_add(&coefficient_view_index(), 1);
                    ijk2index.insert(cell_activated_ijk, index, atomic_add);

                    // Save cell center
                    auto cell_center_array = cellCenter(cell_activated_ijk[0], cell_activated_ijk[1], cell_activated_ijk[2], low_corner, cell_size);
                    for (int i = 0; i < 3; i++)
                        cell_center_slice(index, i) = cell_center_array[i];
                }
            });

        Kokkos::fence();

        // Now that cell keys are set, we can populate the multipoles
        static constexpr std::size_t num_coefficients = (p+1) * (p+1);

        // We need to separate parallel for loops to correctly lambda capture
        // the data_slice, which changes depending on if layer 0 or not.
        if constexpr (position_index == 0)
        {
            Kokkos::parallel_for( "set_multipoles_layer0",
                Kokkos::RangePolicy<execution_space>( 0, num_particles ),
                KOKKOS_LAMBDA( const std::size_t pnum ) {

                    auto pid = start + pnum;
                    
                    auto cell_activated_ijk =
                        position2ijk(positions( pid, 0 ), positions( pid, 1 ), positions( pid, 2 ),
                                    low_corner, cell_size);

                    // Get the cell key
                    auto map_index = ijk2index.find(cell_activated_ijk);
                    auto cell_index = ijk2index.value_at(map_index);

                    // Get cell center
                    Kokkos::Array<scalar_type, 3> cell_center;
                    for (int i = 0; i < 3; i++)
                        cell_center[i] = cell_center_slice(cell_index, i);
                    
                    // Get position
                    Kokkos::Array<scalar_type, 3> pos;
                    for (int i = 0; i < 3; i++)
                        pos[i] = positions( pid, i );
                    
                    // This means we are layer 0 and incoming data must be converted to multipoles
                    scalar_type scalar = data_slice(pid);    

                    // Create multipole array
                    Kokkos::Array<complex, num_coefficients> M;
                    for (std::size_t i = 0; i < num_coefficients; i++)
                        M[i] = complex(0.0, 0.0);

                    // Compute multipoles
                    Canopy::Operator::Scalar::p2m<p>(pos, scalar, cell_center, M);

                    // Add this particle's contribution to the total multipoles
                    for (std::size_t i = 0; i < num_coefficients; i++)
                    {
                        Kokkos::atomic_add(&multipole_coefficients_slice(cell_index, i, 0), M[i].real());
                        Kokkos::atomic_add(&multipole_coefficients_slice(cell_index, i, 1), M[i].imag());
                    }
                });
        }

        else if constexpr (position_index == 1)
        {
            // This means we are not layer 0 and incoming data are multipoles to be translated
            Kokkos::parallel_for( "set_multipoles",
                Kokkos::RangePolicy<execution_space>( 0, num_particles ),
                KOKKOS_LAMBDA( const std::size_t pnum ) {

                    auto pid = start + pnum;
                    
                    auto cell_activated_ijk =
                        position2ijk(positions( pid, 0 ), positions( pid, 1 ), positions( pid, 2 ),
                                    low_corner, cell_size);

                    // Get the cell key
                    auto map_index = ijk2index.find(cell_activated_ijk);
                    auto cell_index = ijk2index.value_at(map_index);

                    // Get cell center
                    Kokkos::Array<scalar_type, 3> cell_center;
                    for (int i = 0; i < 3; i++)
                        cell_center[i] = cell_center_slice(cell_index, i);
                    
                    // Get position
                    Kokkos::Array<scalar_type, 3> pos;
                    for (int i = 0; i < 3; i++)
                        pos[i] = positions( pid, i );

                    // Create arrays
                    Kokkos::Array<complex, num_coefficients> M_orig_array;
                    Kokkos::Array<complex, num_coefficients> M_trans_array;
                    for (std::size_t i = 0; i < num_coefficients; i++)
                    {
                        M_trans_array[i] = complex(0.0, 0.0);
                        M_orig_array[i] = complex(data_slice(pid, i, 0), data_slice(pid, i, 1));
                    }

                    // Create Kokkos:Array of vector pointing from child cell center to cell center.
                    Kokkos::Array<scalar_type, 3> vector_to_center;
                    
                    for (int i = 0; i < 3; i++)
                        vector_to_center[i] = (cell_center[i] - pos[i]) * -1;

                    Canopy::Operator::Scalar::m2m<p>(M_orig_array, vector_to_center, M_trans_array);

                    // Add this multipole's contribution to the total multipoles
                    for (std::size_t i = 0; i < num_coefficients; i++)
                    {
                        Kokkos::atomic_add(&multipole_coefficients_slice(cell_index, i, 0), M_trans_array[i].real());
                        Kokkos::atomic_add(&multipole_coefficients_slice(cell_index, i, 1), M_trans_array[i].imag());
                    }
                });
        }
    }


    /**
     * Figure out which local coefficients from the cells in the layer above we need,
     * and which ranks we need them from.
     * Then send these ranks the ijk indices of the local coefficients we need
     * Then get those local coefficients sent to us.
     * 
     * Steps:
     *  1. Iterate over our cells in ijk2index map:
     *      a) Save which ranks own child cells we need to send locals to
     *      b) Pack (ijk, locals) pairs into an aosoa for haloing 
     *      c) Fill another aosoa that was which pairs go to which ranks
     *  2. Gather locals - each rank now has the locals of all its cells' parents.
     *  3. Iterate over parent locals. Shift and add parent locals to all its
     *      child cells on this layer. 
     */
    void sendCoarseLocals(local_aosoa_type& halo_aosoa, const Kokkos::View<scalar_type*[6], memory_space>& child_domain)
    {
        Kokkos::Profiling::ScopedRegion region("Canopy::SolverLayer::sendCoarseLocals");

        const int comm_size = _comm_size;
        const int num_cells = static_cast<int>(numCells());

        auto ijk2index = _ijk2index;
        auto locals = _locals;

        // For cell center calculations
        Kokkos::Array<scalar_type, 3> low_corner = {_global_low_corner[0], _global_low_corner[1], _global_low_corner[2]};
        auto factor = _tile_reduction_factor;
        auto cell_size = _cell_size;
        Kokkos::Array<scalar_type, 3> child_size;
        for (int i = 0; i < 3; i++)
        {
            child_size[i] = cell_size[i] / factor;
        }

        const int children_per_cell = factor * factor * factor;

        // Map locals to ranks
        Kokkos::View<int**, memory_space> id2rank("id2rank", _locals.size(), _comm_size);
        Kokkos::deep_copy(id2rank, 0);

        auto cell_ijk_slice = Cabana::slice<1>(_locals);
        auto coefficient_slice = Cabana::slice<0>(_locals);

        using md_policy = Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<2>>;
        Kokkos::parallel_for("fill_vert_halo_data",
            md_policy({{0, 0}}, {{num_cells, children_per_cell}}),
            KOKKOS_LAMBDA(const int index, const int c)
        {
            const int di =  c % factor;
            const int dj = (c / factor) % factor;
            const int dk =  c / (factor*factor);
            
            Kokkos::Array<std::size_t, 3> child_ijk = {
                static_cast<std::size_t>(cell_ijk_slice(index, 0) * factor + di),
                static_cast<std::size_t>(cell_ijk_slice(index, 1) * factor + dj),
                static_cast<std::size_t>(cell_ijk_slice(index, 2) * factor + dk)
            };

            auto child_center = cellCenter(child_ijk[0], child_ijk[1], child_ijk[2], low_corner, child_size);

            for (int r = 0; r < comm_size; ++r)
            {
                if ( child_center[0] >= child_domain(r, 0) &&
                    child_center[0] <  child_domain(r, 3) &&
                    child_center[1] >= child_domain(r, 1) &&
                    child_center[1] <  child_domain(r, 4) &&
                    child_center[2] >= child_domain(r, 2) &&
                    child_center[2] <  child_domain(r, 5) )
                {
                    Kokkos::atomic_store(&id2rank(index, r), 1);
                    return;
                }
            }
        });

        // Count the number of exports (non-zero values in id2rank)
        std::size_t num_exports = 0;
        Kokkos::parallel_reduce("count_nonzero_id2rank",
            md_policy({{0, 0}}, {{num_cells, comm_size}}),
            KOKKOS_LAMBDA(const int i, const int j, std::size_t& lsum) {
            lsum += (id2rank(i, j) != 0);
            },
            num_exports
        );

        Cabana::AoSoA<Cabana::MemberTypes<int, int>, memory_space, 4> ids_ranks("ids_ranks", num_exports);
        auto id_slice = Cabana::slice<0>(ids_ranks);
        auto rank_slice = Cabana::slice<1>(ids_ranks);
       
        Kokkos::parallel_scan(
        "pack_exports_scan",
        Kokkos::RangePolicy<execution_space>(0, id2rank.size()),
        KOKKOS_LAMBDA(const int idx, std::size_t& update, const bool final_pass)
        {
            const int a = idx / comm_size;
            const int b = idx - a * comm_size;

            const int flag = id2rank(a, b); // 0 or 1
            const std::size_t inc = (flag != 0);

            const std::size_t pos = update;

            update += inc;

            if (final_pass && inc)
            {
                rank_slice(pos) = b;
                id_slice(pos) = a;
            }
        });

        // Vertical halo for getting local coefficients from more coarse cells
        Cabana::Halo<memory_space> halo( _comm, num_cells, id_slice,
                                    rank_slice );
        std::size_t num_local = halo.numLocal();
        _locals.resize(halo.numLocal() + halo.numGhost());
        _num_local_locals = halo.numLocal();
        _num_ghost_locals = halo.numGhost();
        Cabana::gather(halo, _locals);

        halo_aosoa.resize(halo.numGhost());
        cell_ijk_slice = Cabana::slice<1>(_locals);
        coefficient_slice = Cabana::slice<0>(_locals);

        auto gathered_ijk_slice = Cabana::slice<1>(halo_aosoa);
        auto gathered_coefficient_slice = Cabana::slice<0>(halo_aosoa);

        const auto num_coefficients = (p+1) * (p+1);
        Kokkos::parallel_for("fill_haloed_locals",
        Kokkos::RangePolicy<execution_space>(num_local, num_local + halo.numGhost()),
        KOKKOS_LAMBDA(const int hi)
        {
            for (std::size_t i = 0; i < 3; i++)
                gathered_ijk_slice(hi - num_local, i) = cell_ijk_slice(hi, i);

            for (std::size_t i = 0; i < num_coefficients; i++)
            {
                gathered_coefficient_slice(hi - num_local, i, 0) = coefficient_slice(hi, i, 0);
                gathered_coefficient_slice(hi - num_local, i, 1) = coefficient_slice(hi, i, 1);
            }
        });
    }

    void addCoarseLocals(local_aosoa_type& parent_locals)
    {
        Kokkos::Profiling::ScopedRegion region("Canopy::SolverLayer::addCoarseLocals");

        static constexpr std::size_t num_coefficients = ( p + 1 ) * ( p + 1 );
        auto ijk2index = _ijk2index;

        auto parent_ijk_slice = Cabana::slice<1>(parent_locals);
        auto parent_coefficient_slice = Cabana::slice<0>(parent_locals);
        auto ijk_slice = Cabana::slice<1>(_locals);
        auto coefficient_slice = Cabana::slice<0>(_locals);

        Kokkos::Array<scalar_type, 3> low_corner = {_global_low_corner[0], _global_low_corner[1], _global_low_corner[2]};
        auto factor = _tile_reduction_factor;
        auto cell_size = _cell_size;
        Kokkos::Array<scalar_type, 3> parent_cell_size;
        for (int i = 0; i < 3; i++)
            parent_cell_size[i] = cell_size[i] * factor;

        // Iterate through received locals. Translate and add to the correct child cells
        Kokkos::parallel_for("fill_locals_cells",
        Kokkos::RangePolicy<execution_space>(0, parent_locals.size()),
        KOKKOS_LAMBDA(const int hi)
        {
            auto parent_cell_center = cellCenter(parent_ijk_slice(hi, 0), parent_ijk_slice(hi, 1), parent_ijk_slice(hi, 2),
                low_corner, parent_cell_size);

            // Iterate over all children of this parent
            for (int c = 0; c < factor*factor*factor; ++c)
            {
                const int di =  c % factor;
                const int dj = (c / factor) % factor;
                const int dk =  c / (factor*factor);
                
                Kokkos::Array<std::size_t, 3> cell_ijk = {
                    static_cast<std::size_t>(parent_ijk_slice(hi, 0) * factor + di),
                    static_cast<std::size_t>(parent_ijk_slice(hi, 1) * factor + dj),
                    static_cast<std::size_t>(parent_ijk_slice(hi, 2) * factor + dk)
                };

                // Check if this cell is activated
                auto cell_ijk_exists = ijk2index.exists( cell_ijk );
                if (cell_ijk_exists)
                {
                    auto index = ijk2index.find(cell_ijk);
                    auto local_index = ijk2index.value_at(index);
                    
                    auto child_cell_center = cellCenter(cell_ijk[0], cell_ijk[1], cell_ijk[2],
                        low_corner, cell_size);
                    
                    // Shift and add the parent cell's locals to this cell's locals.
                    // Start by getting vector from new local center (child cell center, A)
                    // to old local center (parent cell center, B), which is B - A
                    Kokkos::Array<scalar_type, 3> X_0;
                    for (int i = 0; i < 3; i++)
                        X_0[i] = parent_cell_center[i] - child_cell_center[i];
                    
                    // Translate locals
                    Kokkos::Array<complex, num_coefficients> L_orig;
                    Kokkos::Array<complex, num_coefficients> L_trans;
                    for (std::size_t i = 0; i < num_coefficients; i++)
                    {
                        L_orig[i].real() = parent_coefficient_slice(hi, i, 0);
                        L_orig[i].imag() = parent_coefficient_slice(hi, i, 1);
                    }
                    Operator::Scalar::l2l<p>(L_orig, L_trans, X_0);

                    // Add translated locals to cell
                    for (std::size_t i = 0; i < num_coefficients; i++)
                    {
                        coefficient_slice(local_index, i, 0) += L_trans[i].real();
                        coefficient_slice(local_index, i, 1) += L_trans[i].imag();
                    }
                }
            }
        });
    }

    /**
     * Computes the interaction list for each cell in the layer.
     * 
     * The interaction list of cell0 is the set of all cells such that:
     *  1) cell0 and cell_other are on the same layer of the tree.
     *  2) cell0 and cell_other do not touch.
     *  3) The parent cells of cell0 and cell_other do touch.
     */
    void computeInteractionBounds(int starting_cells_per_dimension, int start_layer)
    {
        Kokkos::Profiling::ScopedRegion region("Canopy::SolverLayer::computeInteractionBounds");

        // Initialize _mhalo_outer_bound
        _mhalo_outer_bound = Kokkos::View<int[6], memory_space>("_mhalo_outer_bound");
        
        auto m2l_bounds = _m2l_bounds;
        auto ijk2index = _ijk2index;
        auto mhalo_outer_bound = _mhalo_outer_bound;

        // Set outermost bounds to a high value
        auto min_subview = Kokkos::subview( mhalo_outer_bound, Kokkos::make_pair( 0, 3 ) );
        auto max_subview = Kokkos::subview( mhalo_outer_bound, Kokkos::make_pair( 3, 6 ) );
        Kokkos::deep_copy(min_subview, INT_MAX);
        Kokkos::deep_copy(max_subview, INT_MIN);

        const int layer_number = _layer_number;
        const int cell_incr_factor = _tile_reduction_factor;

        // Per-cell calculation
        Kokkos::parallel_for("interaction bounds",
        Kokkos::RangePolicy<execution_space>(0, ijk2index.capacity()),
        KOKKOS_LAMBDA(const int ijk2index_index)
        {
            if (ijk2index.valid_at(ijk2index_index))
            {
                // Cell ijk
                auto cell_ijk = ijk2index.key_at( ijk2index_index );

                // Cell index in local and multipole structures
                auto index = ijk2index.value_at(ijk2index_index);

                // Cast cell_ijk to ints
                Kokkos::Array<int, 3> cell_ijk_int;
                for (int i = 0; i < 3; i++)
                    cell_ijk_int[i] = static_cast<int>(cell_ijk[i]);

                auto bounds = cell2Bound(cell_ijk_int, layer_number, start_layer, starting_cells_per_dimension,
                    cell_incr_factor);
                
                // Save include bounds
                for (int i = 0; i < 6; i++)
                    m2l_bounds(index, i) = bounds.first[i];
            }
        });
      
        MinMax6 result;

        Kokkos::parallel_reduce(
        "compute_halo_bounds",
        Kokkos::RangePolicy<execution_space>(0, m2l_bounds.extent(0)),
        HaloBoundsReduce<decltype(m2l_bounds)>{m2l_bounds},
        result
        );

        auto host_bounds = Kokkos::create_mirror_view(mhalo_outer_bound);
        for (int d = 0; d < 6; ++d)
            host_bounds(d) = result.v[d];

        Kokkos::deep_copy(mhalo_outer_bound, host_bounds);
    }

    /**
     * Halo multipoles across the same layer based on the absolute outer multipole bounds.
     * Add halo multipole coefficients to _multipoles and add haloed cell ijks to _ijk2index.
     */
    void haloMultipoles()
    {
        Kokkos::Profiling::ScopedRegion region("Canopy::SolverLayer::haloMultipoles");

        // Communicate _mhalo_outer_bound to each rank so we know who we need to send to.
        // Allocate view to store data from other ranks
        Kokkos::View<int*[6], Kokkos::HostSpace> all_mhalo_outer_bounds_h("all_mhalo_outer_bounds", _comm_size);

        // Host buffer 
        auto h_send = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), _mhalo_outer_bound);

        // MPI on host data
        MPI_Allgather(h_send.data(), 6, MPI_INT,
                      all_mhalo_outer_bounds_h.data(), 6, MPI_INT, _comm);

        // Copy back to device
        auto all_mhalo_outer_bounds = Kokkos::create_mirror_view_and_copy(memory_space(), all_mhalo_outer_bounds_h);

        const int rank = _rank;
        const int comm_size = _comm_size;
        const int layer_number = _layer_number;

        const auto num_cells = numCells();

        auto ijk2index = _ijk2index;

        // For cell center calculations
        Kokkos::Array<scalar_type, 3> low_corner = {_global_low_corner[0], _global_low_corner[1], _global_low_corner[2]};
        auto cell_size = _cell_size;
            
        std::size_t max_num_exports = num_cells * _comm_size;
        Cabana::AoSoA<Cabana::MemberTypes<int, int>, memory_space, 4> ids_ranks("ids_ranks", max_num_exports);
        auto id_slice = Cabana::slice<0>(ids_ranks);
        auto rank_slice = Cabana::slice<1>(ids_ranks);
        auto cell_center_slice = Cabana::slice<1>(_multipoles);
        auto coefficient_slice = Cabana::slice<0>(_multipoles);

        // Hold size of exports
        Kokkos::View<int, memory_space> counter("counter");
        Kokkos::deep_copy(counter, 0);

        Kokkos::parallel_for("fill_horz_multipole_halo_data",
        Kokkos::RangePolicy<execution_space>(0, num_cells),
        KOKKOS_LAMBDA(const int m_index)
        {
            // Convert cell center to cell ijk
            auto cell_ijk = position2ijk(cell_center_slice( m_index, 0 ), cell_center_slice( m_index, 1 ), cell_center_slice( m_index, 2 ),
                low_corner, cell_size);

            // Check if this cell center is within a rank's halo bound
            for (int r = 0; r < comm_size; ++r)
            {
                if (rank == r)
                    continue;

                if ( cell_ijk[0] >= all_mhalo_outer_bounds(r, 0) &&
                    cell_ijk[0] <  all_mhalo_outer_bounds(r, 3) &&
                    cell_ijk[1] >= all_mhalo_outer_bounds(r, 1) &&
                    cell_ijk[1] <  all_mhalo_outer_bounds(r, 4) &&
                    cell_ijk[2] >= all_mhalo_outer_bounds(r, 2) &&
                    cell_ijk[2] <  all_mhalo_outer_bounds(r, 5) )
                {
                    auto index = Kokkos::atomic_fetch_add(&counter(), 1);
                    id_slice(index) = m_index;
                    rank_slice(index) = r;
                }
            }
        });
        Kokkos::fence();

        int num_exports;
        Kokkos::deep_copy(num_exports, counter);
        ids_ranks.resize(num_exports);
        id_slice = Cabana::slice<0>(ids_ranks);
        rank_slice = Cabana::slice<1>(ids_ranks);

        // Create halo
        Cabana::Halo<memory_space> halo( _comm, num_cells, id_slice,
                                    rank_slice );
        _num_local_multipoles = halo.numLocal();
        _num_ghost_multipoles = halo.numGhost();
        _multipoles.resize(_num_local_multipoles + _num_ghost_multipoles);
        Cabana::gather(halo, _multipoles);
        
        // Add haloed multipole data to _ijk2index map.
        cell_center_slice = Cabana::slice<1>(_multipoles);
        coefficient_slice = Cabana::slice<0>(_multipoles);
        Kokkos::parallel_for("add_haloed_multipoles_to_ijk2index",
        Kokkos::RangePolicy<execution_space>(_num_local_multipoles, _num_local_multipoles + _num_ghost_multipoles),
        KOKKOS_LAMBDA(const int m_index)
        {
            // Convert cell center to cell ijk
            auto cell_ijk = position2ijk(cell_center_slice( m_index, 0 ), cell_center_slice( m_index, 1 ), cell_center_slice( m_index, 2 ),
                low_corner, cell_size);

            // Insert into map
            auto result = ijk2index.insert(cell_ijk, m_index);
            
            if (!result.success())
            {
                printf("L%d: R%d: Error inserting haloed multipole!\n", layer_number, rank);
            }
        });
        Kokkos::fence();
    }

    /**
     * For each cell, iterate over all cells in its interaction list. These are cells that
     * are at least two cells away from the cell in question and have not been accounted
     * for in more coarse layers.
     * Convert the multipole coefficients centered around the other cell to local coefficients
     * centered around this cell.
     */
    void multipole_to_local(int starting_cells_per_dimension, int start_layer)
    {
        Kokkos::Profiling::ScopedRegion region("Canopy::SolverLayer::multipole_to_local");

        computeInteractionBounds(starting_cells_per_dimension, start_layer);

        haloMultipoles();

        auto locals = _locals;
        auto m2l_bounds = _m2l_bounds;
        auto map = *_map_ptr;
        auto multipoles = _multipoles;
        auto ijk2index = _ijk2index;

        auto m_slice = Cabana::slice<0>(_multipoles);
        auto m_cell_center_slice = Cabana::slice<1>(_multipoles);
        auto l_slice = Cabana::slice<0>(_locals);
        auto l_cell_ijk_slice = Cabana::slice<1>(_locals);

        const int cells_per_dim = _cells_per_dim;
        Kokkos::Array<scalar_type, 3> low_corner = {_global_low_corner[0], _global_low_corner[1], _global_low_corner[2]};
        auto cell_size = _cell_size;

        // Compute neighbor list of cells within the outer cutoff
        // We look at most 10 cells in each dimension
        scalar_type neighborhood_radius = Kokkos::sqrt(
            Kokkos::pow(5.0*_cell_size[0], 2) +
            Kokkos::pow(5.0*_cell_size[1], 2) +
            Kokkos::pow(5.0*_cell_size[2], 2)) + 0.00001;
        auto neighbor_list = Cabana::Experimental::makeNeighborList(
            Cabana::FullNeighborTag{}, m_cell_center_slice, 0, _multipoles.size(),
            neighborhood_radius );
        using list_type = decltype(neighbor_list);
        using team_policy = Kokkos::TeamPolicy<execution_space>;
        using member_type = team_policy::member_type;

        static constexpr int num_coefficients = (p+1)*(p+1);

        // Scratch: store (real, imag) as doubles for each coefficient
        const int scratch_bytes =
            Kokkos::View<scalar_type*, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                        Kokkos::MemoryUnmanaged>::shmem_size(2 * num_coefficients);

        Kokkos::parallel_for(
            "Canopy::SolverLayer::multipole_to_local team",
            team_policy(_num_local_multipoles, Kokkos::AUTO)
                .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
            KOKKOS_LAMBDA(const member_type& team)
            {
                const int index = team.league_rank();

                // Team scratch accumulation buffer: [0..num_coeff-1]=real, [num_coeff..2*num_coeff-1]=imag
                using scratch_space = typename member_type::scratch_memory_space;
                Kokkos::View<scalar_type*, scratch_space, Kokkos::MemoryUnmanaged> accum(
                    team.team_scratch(0), 2 * num_coefficients);

                // Zero scratch
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 2 * num_coefficients),
                                    [&](const int t) { accum(t) = 0.0; });
                team.team_barrier();

                // Get cell ijk from cell center (same as before)
                auto cell_ijk = position2ijk(m_cell_center_slice(index,0),
                                            m_cell_center_slice(index,1),
                                            m_cell_center_slice(index,2),
                                            low_corner, cell_size);

                // Inner bounds
                Kokkos::Array<std::size_t, 3> inner_lower_bound;
                Kokkos::Array<std::size_t, 3> inner_upper_bound;
                for (int d = 0; d < 3; d++)
                {
                    inner_upper_bound[d] = Kokkos::min(static_cast<int>(cell_ijk[d]) + 3, cells_per_dim);
                    inner_lower_bound[d] = Kokkos::max(static_cast<int>(cell_ijk[d]) - 2, 0);
                }

                // Write l_cell_ijk_slice once
                Kokkos::single(Kokkos::PerTeam(team), [&](){
                    for (int d = 0; d < 3; d++)
                        l_cell_ijk_slice(index, d) = cell_ijk[d];
                });

                // Team-parallel over neighbors
                const int num_neighbors =
                    Cabana::NeighborList<list_type>::numNeighbor(neighbor_list, index);

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, num_neighbors),
                                    [&](const int j)
                {
                    const int neighbor_id =
                        Cabana::NeighborList<list_type>::getNeighbor(neighbor_list, index, j);

                    auto neighbor_cell_ijk = position2ijk(m_cell_center_slice(neighbor_id,0),
                                                        m_cell_center_slice(neighbor_id,1),
                                                        m_cell_center_slice(neighbor_id,2),
                                                        low_corner, cell_size);

                    const bool in_outer =
                        (neighbor_cell_ijk[0] >= m2l_bounds(index,0) && neighbor_cell_ijk[0] < m2l_bounds(index,3)) &&
                        (neighbor_cell_ijk[1] >= m2l_bounds(index,1) && neighbor_cell_ijk[1] < m2l_bounds(index,4)) &&
                        (neighbor_cell_ijk[2] >= m2l_bounds(index,2) && neighbor_cell_ijk[2] < m2l_bounds(index,5));

                    const bool in_inner =
                        (neighbor_cell_ijk[0] >= inner_lower_bound[0] && neighbor_cell_ijk[0] < inner_upper_bound[0]) &&
                        (neighbor_cell_ijk[1] >= inner_lower_bound[1] && neighbor_cell_ijk[1] < inner_upper_bound[1]) &&
                        (neighbor_cell_ijk[2] >= inner_lower_bound[2] && neighbor_cell_ijk[2] < inner_upper_bound[2]);

                    if (!in_outer || in_inner)
                        return;

                    // m2l vector
                    Kokkos::Array<scalar_type, 3> m2l_vec;
                    for (int d = 0; d < 3; ++d)
                        m2l_vec[d] = m_cell_center_slice(neighbor_id, d) - m_cell_center_slice(index, d);

                    // Load multipole coefficients
                    Kokkos::Array<complex, num_coefficients> M;
                    for (int i = 0; i < num_coefficients; i++)
                    {
                        M[i].real() = m_slice(neighbor_id, i, 0);
                        M[i].imag() = m_slice(neighbor_id, i, 1);
                    }

                    // Compute local contribution
                    Kokkos::Array<complex, num_coefficients> L;
                    Operator::Scalar::m2l<p>(M, L, m2l_vec);

                    // Accumulate into team scratch
                    for (int i = 0; i < num_coefficients; i++)
                    {
                        Kokkos::atomic_add(&accum(i), L[i].real());
                        Kokkos::atomic_add(&accum(i + num_coefficients), L[i].imag());
                    }
                });

                team.team_barrier();

                // Add from scratch
                Kokkos::single(Kokkos::PerTeam(team), [&](){
                    for (int i = 0; i < num_coefficients; i++)
                    {
                        l_slice(index, i, 0) += accum(i);
                        l_slice(index, i, 1) += accum(i + num_coefficients);
                    }
                });
            });
    }

    int rank() const { return _rank; }
    int layerNumber() const { return _layer_number; }

    /**
     * Get the domain in 3D space that each rank owns with the upper value being non-inclusive
     * Each entry in the returned vector is (x_start, y_start, z_start, x_end, y_end, z_end)
     */
    auto domains() const {return _domains;}
    auto cell_offsets() const {return _cell_offsets_view;}
    auto num_owned_cell() const {return _num_owned_cell_view;}

    std::shared_ptr<sparse_layout_type> layout() {return _layout_ptr;}
    std::shared_ptr<sparse_map_type> map() {return _map_ptr;}
    int cellsPerDim() const {return _cells_per_dim;}
    int tilesPerDim() const {return _tiles_per_dim;}
    Kokkos::Array<scalar_type, 3> cellSize() const {return _cell_size;}
    Kokkos::UnorderedMap<int, Kokkos::Array<std::size_t, 3>, memory_space>& cid2ijk() {return _cid2ijk;}

    // Get the multipole coefficients
    auto multipoles() {return _multipoles;}

    // Get the local coefficients
    auto locals() {return _locals;}

    // Get the m2l bounds
    auto m2l_bounds() {return _m2l_bounds;}

    // Return cell_ijk to index into local view map
    auto cellijk2i() {return _ijk2index;}

    // The number of cells activated in this layer. Can't use the size of local or multipole views
    // because they may contain ghost elements
    auto numCells()
    {
        std::size_t val;
        Kokkos::deep_copy(val, _coefficient_view_index);
        return val;
    }

  private:
    const std::array<scalar_type, 3> _global_high_corner;
    const std::array<scalar_type, 3> _global_low_corner;
    std::array<int, 3> _global_num_cell;
	const int _tiles_per_dim;
    const int _tile_reduction_factor;
    const int _halo_width;
    const int _cells_per_dim;
    const int _layer_number;
    int _rank, _comm_size;

    // Cell size in the x, y, and z dimensions.
    Kokkos::Array<scalar_type, 3> _cell_size;

    // Information about which processes own which other
    // section of the sparse mesh
    Kokkos::View<int*[3], memory_space> _cell_offsets_view;
    Kokkos::View<int*[3], memory_space> _num_owned_cell_view;
    Kokkos::View<scalar_type*[6], memory_space> _domains;

    // Partitioner parameters
    int _num_step_rebalance, _max_optimize_iteration;

    // MPI communicators
    const MPI_Comm _comm;
    MPI_Comm _cart_comm;
    
    std::shared_ptr<sparse_partitioner_type> _partitioner_ptr;
    std::shared_ptr<sparse_layout_type> _layout_ptr;
    std::shared_ptr<sparse_map_type> _map_ptr;

    // Map of cell id (cid, unique per process) to cell ijk position.
    // Cabana supports ijk -> cid but not the inverse.
    Kokkos::UnorderedMap<int, Kokkos::Array<std::size_t, 3>, memory_space> _cid2ijk;

    // Map of cell_ijk to its location in the coefficient data structures.
    // Cell ijks are replicated so they do not need to be haloed
    // separately and be re-mapped to coefficients.
    Kokkos::View<std::size_t, memory_space> _coefficient_view_index;
    index_map_type _ijk2index;

    // Multipole coefficients for each cell.
    multipole_aosoa_type _multipoles;
    std::size_t _num_local_multipoles;
    std::size_t _num_ghost_multipoles;

    // Local coefficients for each cell.
    local_aosoa_type _locals;
    std::size_t _num_local_locals;
    std::size_t _num_ghost_locals;
    // For each cell, the subset of the domain, in i/j/k indices for cells in this layer,
    // where the contribution from cells outside of these bounds have already been
    // accounted for in more coarse layers.
    // Indices 0, 1, 2 = lower bound, inclusive
    // Indices 3, 4, 5 = upper bound, exclusive
    Kokkos::View<int*[6], memory_space> _m2l_bounds;

    // The outer bound for haloing multipoles within the same layer.
    // This is the outermost of the outer bounds in _m2l_bounds.
    Kokkos::View<int[6], memory_space> _mhalo_outer_bound;
};

template <class SolverType, std::size_t CellPerTileDim, class ScalarType>
std::shared_ptr<SolverLayer<SolverType, CellPerTileDim>> createSolverLayer(const std::array<ScalarType, 3>& global_low_corner,
            const std::array<ScalarType, 3>& global_high_corner,
	        const int tiles_per_dim, const int tile_reduction_factor,
            const int halo_width,
            const int layer_number,
            MPI_Comm comm)
{
    return std::make_shared<SolverLayer<SolverType, CellPerTileDim>>(global_low_corner,
            global_high_corner,
	        tiles_per_dim, tile_reduction_factor,
            halo_width, layer_number, comm);
}

} // end namespace Canopy

#endif // CANOPY_TREELAYER_HPP
