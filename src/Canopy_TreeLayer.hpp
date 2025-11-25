#ifndef CANOPY_TREELAYER_HPP
#define CANOPY_TREELAYER_HPP

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <Canopy_Kernels.hpp>

#include <memory>

#include <mpi.h>

#include <limits>

namespace Canopy
{

// https://repositorio.unesp.br/server/api/core/bitstreams/0e824479-3128-41f7-8cd2-462e9a242c42/content

/**
 * Convert a std::vector to a Kokkos::View
 */
template <class MemorySpace, class ElementType>
Kokkos::View<typename ElementType::value_type**, MemorySpace>
vec2view(const std::vector<ElementType>& vector, const std::string& label)
{
    using value_type = typename ElementType::value_type;
    const std::size_t num_elements = vector.size();
    constexpr std::size_t element_size = std::tuple_size<ElementType>::value;

    // Create a host view
    Kokkos::View<value_type**, Kokkos::HostSpace> host_view(label, num_elements, element_size);

    // Copy vector data into the host view
    for (std::size_t i = 0; i < num_elements; ++i)
    {
        for (std::size_t j = 0; j < element_size; ++j)
        {
            host_view(i, j) = vector[i][j];
        }
    }

    // Copy to device
    auto device_view = Kokkos::create_mirror_view_and_copy(MemorySpace(), host_view);

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

template <class TreeType, std::size_t CellPerTileDim>
class TreeLayer
{
  public:
    //! Self type. All TreeLayers in the Octree are of the same type. 
    using tree_layer_type = TreeLayer<TreeType, CellPerTileDim>;

    //! Execution space
    using execution_space = typename TreeType::execution_space;
    //! Memory space.
    using memory_space = typename TreeType::memory_space;
    //! Number of dimensions
    static constexpr std::size_t num_space_dim = TreeType::num_space_dim;

    //! Sparse partitioner type
    using sparse_partitioner_type = Cabana::Grid::SparseDimPartitioner<memory_space, CellPerTileDim, num_space_dim>;

    //! DataTypes Data types (Cabana::MemberTypes).
    using cdouble = typename TreeType::cdouble;
    using tuple_type = typename TreeType::tuple_type;
    using member_types = typename TreeType::member_types;
    static constexpr std::size_t p = TreeType::p;
    using data_aosoa_type = typename TreeType::data_aosoa_type; // AoSoA type that holds cell data


    using entity_type = typename TreeType::entity_type;

    using mesh_type = typename TreeType::mesh_type;

    static constexpr std::size_t cell_per_tile_dim = CellPerTileDim;

    using sparse_map_type = Cabana::Grid::SparseMap<memory_space, cell_per_tile_dim>;

    static constexpr unsigned long long cell_bits_per_tile =
        sparse_map_type::cell_bits_per_tile;
    //! Cell ID mask inside a tile
    static constexpr unsigned long long cell_mask_per_tile =
        sparse_map_type::cell_mask_per_tile;

    using sparse_layout_type =
        Cabana::Grid::Experimental::SparseArrayLayout<member_types, entity_type, mesh_type, sparse_map_type>;

    using sparse_array_type = Cabana::Grid::Experimental::SparseArray<member_types, memory_space, entity_type,
                                          mesh_type, sparse_map_type>;

    //! AoSoA type
    // using aosoa_type = typename sparse_array_type::aosoa_type;
    
    TreeLayer(const std::array<double, 3>& global_low_corner,
            const std::array<double, 3>& global_high_corner,
	        const int tiles_per_dim, const int halo_width,
            const int layer_number,
            MPI_Comm comm )
        : _global_low_corner( global_low_corner )
        , _global_high_corner( global_high_corner )
        , _tiles_per_dim( tiles_per_dim )
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
        // printf("L%d: R%d: high-low: %0.2lf, %0.2lf, %0.2lf, _tiles_per_dim: %d\n", _layer_number, _rank,
        //     _global_high_corner[0] - _global_low_corner[0],
        //     _global_high_corner[1] - _global_low_corner[1],
        //     _global_high_corner[2] - _global_low_corner[2],
        //     _tiles_per_dim);
        // printf("L%d: R%d: global_num_cell: %d, %d, %d\n",  _layer_number, _rank, _global_num_cell[0], _global_num_cell[1], _global_num_cell[2]);
    
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
        // if (_rank == 0) printf("R%d: ranks per dim: %d, %d, %d\n", rank, ranks_per_dim[0], ranks_per_dim[1], ranks_per_dim[2]);
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
        MPI_Cart_get(_cart_comm, 3, dims, periods, cart_coords); // dims will contain [x, y, z] counts

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

        /*!
        \brief From Cabana docs: Initialize the tile partition; partition in each dimension
        has the form [0, p_1, ..., p_n, total_tile_num], so the partition
        would be [0, p_1), [p_1, p_2) ... [p_n, total_tile_num]
        \param rec_partition_i partition array in dimension i
        \param rec_partition_j partition array in dimension j
        \param rec_partition_k partition array in dimension k
        */
        _partitioner_ptr->initializeRecPartition(x_partition, y_partition, z_partition);

        initialize();
        /*
        Steps:
        1. Initially partition based on the 2D partition of the surface.
        2. Register sparse grid using positions.
        3. Optimize partitioner.
        4. Re-register sparse grid.
        5. Use Distributor to send particles to their rank of ownership in the new partition.
        6. Aggregate data (vorticities) into cells based on particles that reside in the cell.
        */
    }

    /**
     * Use the sparse partitioner to initialize the global and local grids, sparse map,
     * and sparse array objects.
     */
    void initialize()
    {
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
            Cabana::Grid::createSparseMap<memory_space, double, cell_per_tile_dim>( global_mesh, 1.2 );
        // Save sparse map as shared pointer
        _map_ptr = std::make_shared<sparse_map_type>(sparse_map);

        // printf("R%d: global num cell x/y/z: %d, %d, %d\n", _rank, global_mesh->globalNumCell( Cabana::Grid::Dim::I ),
        //   global_mesh->globalNumCell( Cabana::Grid::Dim::J ),
        //   global_mesh->globalNumCell( Cabana::Grid::Dim::K ));
        
        // initializeRecPartition(sparse_map);
        _layout_ptr =
            Cabana::Grid::Experimental::createSparseArrayLayout<member_types>( local_grid, *_map_ptr, entity_type() );
        _cells_ptr = Cabana::Grid::Experimental::createSparseArray<memory_space>(
            std::string( "cell_array" ), *_layout_ptr );
        
        // Store cell size
        updateCellSize();

        // Get the owned number of cells and the global cell offset
        // each MPI rank on this layer.
        computeCellInfo();

        // Set local view index to 0
        _local_view_index = Kokkos::View<std::size_t, memory_space>("local_view_index");
        Kokkos::deep_copy(_local_view_index, 0);
    }

    void updateCellSize()
    {
        auto local_grid = _cells_ptr->layout().localGrid();
        auto sparse_mesh = local_grid->globalGrid().globalMesh();
        _cell_size = {sparse_mesh.cellSize( 0 ), sparse_mesh.cellSize( 1 ), sparse_mesh.cellSize( 2 )};
    }

    template <class ParticlePositions>
    void optimizePartition(ParticlePositions positions, std::size_t num_particles)
    {
        _partitioner_ptr->optimizePartition( positions, num_particles, _global_low_corner,
            _cell_size[0], _comm);

        // Reinitialize sparse data structures after updating the partition.
        initialize();
    }

     /*!
      \brief Populate _domains, _num_owned_tile_view, and _tile_offsets_view using the current
      partition.
    */
    void computeCellInfo()
    {
        // Get x/y/z domains. The domains are also needed to correctly filter invalid
        // cell counts and offsets
        auto current_partition = _partitioner_ptr->getCurrentPartition();

        // Allocate vectors
        std::vector<Kokkos::Array<int, 3>> tile_offsets_vec(_comm_size);
        std::vector<Kokkos::Array<int, 3>> num_owned_tile_vec(_comm_size);
        std::vector<Kokkos::Array<double, 6>> domains_vec(_comm_size);

        for (int rank = 0; rank < _comm_size; ++rank)
        {
            int coords[3];
            MPI_Cart_coords(_cart_comm, rank, 3, coords);

            Kokkos::Array<double, 6> domain;
            Kokkos::Array<int, 3> tile_offsets;
            Kokkos::Array<int, 3> tiles_owned;
            for (int d = 0; d < 3; ++d)
            {
                int tile_start = current_partition[d][coords[d]];
                int tile_end   = current_partition[d][coords[d] + 1];

                double global_min = _global_low_corner[d];
                double global_max = _global_high_corner[d];
                double tile_width = (global_max - global_min) / _tiles_per_dim;
                
                // Set domain lower and upper bound for this rank
                domain[d]     = global_min + tile_start * tile_width;
                domain[d + 3] = global_min + tile_end   * tile_width;

                // Set cells owned: (cells per tile) * (tiles owned) 
                tiles_owned[d] = (tile_end - tile_start);
                // Set cell offset: (tile_start) * (cells per tile)
                // No owned cells in this dimension = offset is invalid, set to -1
                if (tiles_owned[d] == 0)
                {
                    // Set all offsets to -1 and cells owned to 0
                    for (int j = 0; j < 3; ++j)
                    {
                        tile_offsets[j] = -1;
                        tiles_owned[j] = 0;
                    }
                    break;
                }
                else
                    tile_offsets[d] = tile_start;

            }
            // if (_rank == 0) printf("L%d: R%d: i(%d, %d), j(%d, %d), k(%d, %d)\n", _layer_number, rank, 
            //     current_partition[0][coords[0]], current_partition[0][coords[0] + 1],
            //     current_partition[1][coords[1]], current_partition[1][coords[1] + 1],
            //     current_partition[2][coords[2]], current_partition[2][coords[2] + 1]);

            domains_vec[rank] = domain;
            num_owned_tile_vec[rank] = tiles_owned;
            tile_offsets_vec[rank] = tile_offsets;
            // if (_rank == 0) printf("L%d: R%d: tiles owned: (%d, %d, %d), offset: (%d, %d, %d)\n",
            //     _layer_number, rank, tiles_owned[0], tiles_owned[1], tiles_owned[2],
            //     tile_offsets[0], tile_offsets[1], tile_offsets[2]); 
        }

        // Convert vectors to views and save
        _tile_offsets_view = vec2view<memory_space>(tile_offsets_vec, "_tile_offsets_view");
        _num_owned_tile_view = vec2view<memory_space>(num_owned_tile_vec, "_num_owned_tile_view");
        _domains = vec2view<memory_space>(domains_vec, "_domains");

        // for (std::size_t i = 0; i < _domains.size(); ++i)
        // {
        //     if (_rank == 0)
        //         printf("L%d: R%d: [%0.3lf, %0.3lf, %0.3lf] to [%0.3lf, %0.3lf, %0.3lf]\n", _layer_number,
        //             i, _domains(i, 0), _domains(i, 1), _domains(i, 2), _domains(i, 3),
        //             _domains(i, 4), _domains(i, 5));
        // }
    }

    /**
     * Populate a Kokkos::View that maps to the passed-in AoSoA to the rank
     * each particle should be migrated to based on its x/y/z position.
     */
    template <class ViewType, class PositionSliceType>
    void mapParticles(const PositionSliceType& positions, ViewType& particle_ranks, const int particle_num)
    {
        using mem_space = typename ViewType::memory_space;
        using exec_space = typename ViewType::execution_space;

        // Get all rank domains on host
        auto domains_host = _domains;
        int num_ranks = domains_host.size();

        // Copy domains to device
        Kokkos::View<double*[6], mem_space> domain_bounds("domain_bounds", num_ranks);
        auto domain_bounds_host = Kokkos::create_mirror_view(domain_bounds);
        for (int r = 0; r < num_ranks; ++r)
            for (int j = 0; j < 6; ++j)
                domain_bounds_host(r, j) = domains_host[r][j];
        Kokkos::deep_copy(domain_bounds, domain_bounds_host);

        Kokkos::parallel_for(
            "mapParticles",
            Kokkos::RangePolicy<exec_space>(0, particle_num),
            KOKKOS_LAMBDA(const int i) {
                double xpos = positions(i, 0);
                double ypos = positions(i, 1);
                double zpos = positions(i, 2);

                // Linear search: check each rank domain
                for (int r = 0; r < num_ranks; ++r)
                {
                    double x_lo = domain_bounds(r, 0);
                    double y_lo = domain_bounds(r, 1);
                    double z_lo = domain_bounds(r, 2);
                    double x_hi = domain_bounds(r, 3);
                    double y_hi = domain_bounds(r, 4);
                    double z_hi = domain_bounds(r, 5);

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
     * Return an AoSoA of cell data, indexed by cell id
     */
    data_aosoa_type data()
    {
        // int rank = _rank;
        // int layer_number = _layer_number;

        data_aosoa_type cell_data("cell_data", _cid2ijk.size());
        Kokkos::View<int, memory_space> idx("idx");
        Kokkos::deep_copy(idx, 0);

        auto map = *_map_ptr;
        auto aosoa = _cells_ptr->aosoa();
        auto cid2ijk = _cid2ijk;

        // printf("R%d: map size: %d\n", rank, cell_ids_map.size());

        // XXX - can this be made more efficient by extracting my SoAs rather than tuples?
        // int layer_number = _layer_number;
        Kokkos::parallel_for(
            "get_data",
            Kokkos::RangePolicy<execution_space>(0, cid2ijk.capacity()),
            KOKKOS_LAMBDA(const int index) {
                if (cid2ijk.valid_at(index))
                {
                    auto cell_ijk = cid2ijk.value_at( index );
                    auto tid = map.queryTile(cell_ijk[0],
                                            cell_ijk[1],
                                            cell_ijk[2]);
                    auto ctid = map.cell_local_id(cell_ijk[0],
                                            cell_ijk[1],
                                            cell_ijk[2]);       
                    auto tp = aosoa.getTuple(( tid << cell_bits_per_tile ) |
                                            ( ctid & cell_mask_per_tile ) );
                    int offset = Kokkos::atomic_fetch_add(&idx(), 1);
                    cell_data.setTuple(offset, tp);

                    // if (layer_number == 0)
                    //     printf("L%d: R%d: extracting c(%.3lf, %.3lf, %.3lf) into i%d\n", layer_number, rank,
                    //         Cabana::get<1>(tp, 0), Cabana::get<1>(tp, 1), Cabana::get<1>(tp, 2), offset);
                }
            }
        );

        return cell_data;
    }

    /**
     * Initialize the leaf layer. This requires different initialization than other layers
     * because data must be converted to multipole coefficients. For all other layers,
     * the data has already been converted.
     */
    template <class ParticleAoSoA, class In2OutMap>
    void initializeLeafCell(const ParticleAoSoA particle_aosoa, In2OutMap in2out,
        const std::size_t start, const std::size_t end)
    {
        if (_layer_number != 0)
        {
            throw std::runtime_error("TreeLayer::initializeLeafCell: must be called with _layer_number = 0");
        }

        int rank = _rank;
        int layer_number = _layer_number;
        // printf("R%d: leaf cell data from [%d, %d)\n", rank, start, end);

        std::size_t view_size = end - start;
        ParticleAoSoA cell_data("cell_data", view_size);

        // The following is needed to retrieve and save cell center
        auto in_id_slice = Cabana::slice<0>(in2out);
        auto out_id_slice = Cabana::slice<1>(in2out);
        Kokkos::View<double[3], memory_space> cell_center("cell_center");
        Kokkos::Array<double, 3> low_corner = {_global_low_corner[0], _global_low_corner[1], _global_low_corner[2]};
        auto cell_size = _cell_size;
        auto cid2ijk = _cid2ijk;

        // Save the tile id and cell tile id to set the approportiate index
        // in the sparse mesh AoSoA.
        auto map = *_map_ptr;
        Kokkos::View<std::size_t, memory_space> tid("tid");
        Kokkos::View<std::size_t, memory_space> ctid("ctid");

        // Move all tuples in this cell into a contiguous AoSoA
        Kokkos::parallel_for(
            "populate_cell_data",
            Kokkos::RangePolicy<execution_space>( 0, view_size ),
            KOKKOS_LAMBDA( const int i ) {

                std::size_t index = i + start;
                std::size_t in_id = in_id_slice(index);
                auto data_tuple = particle_aosoa.getTuple(in_id);
                cell_data.setTuple(i, data_tuple);

                if (i == 0)
                {
                    // Since all incoming data are in the same cell, these values
                    // will be the same for all threads so only one thread needs to
                    // compute them.
                    auto cid = out_id_slice(in_id);
                    auto index = cid2ijk.find(cid);

                    // Get the cell center for multipole calculations using the cid
                    auto cell_ijk = cid2ijk.value_at(index);
                    auto cell_center_array = cellCenter(cell_ijk[0], cell_ijk[1], cell_ijk[2], low_corner, cell_size);
                    // printf("L0: R%d: cid: %llu, ijk: %llu, %llu, %llu\n",
                    //     rank,
                    //     (unsigned long long)cid,
                    //     (unsigned long long)cell_ijk[0],
                    //     (unsigned long long)cell_ijk[1],
                    //     (unsigned long long)cell_ijk[2]);
                    for (int j = 0; j < 3; ++j)
                    {
                        cell_center(j) = cell_center_array[j];
                    }

                    // Save the tile id and cell tile id
                    // auto tid = map.queryTile(cell_ijk[0],
                    //                         cell_ijk[1],
                    //                         cell_ijk[2]);
                    // auto ctid = map.cell_local_id(cell_ijk[0],
                    //                         cell_ijk[1],
                    //                         cell_ijk[2]); 
                    // (-0.469, 0.094, -0.469)
                    // printf("L%d: R%d: cid: %d, c(%0.3lf, %0.3lf, %0.3lf), c_ijk(%d, %d, %d), tid: %d, ctid: %d, in_pos(%0.3lf, %0.3lf, %0.3lf)\n",
                    //     layer_number, rank, cid(),
                    //     cell_ijk[0], cell_ijk[1], cell_ijk[2], tid(), ctid(),
                    //     cell_center(0), cell_center(1), cell_center(2), x, y, z);
                }
                
            });
            
        Kokkos::fence();

        // Copy cell center to host and move to a Kokkos::Array
        auto cell_center_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cell_center);
        Kokkos::Array<double, 3> cell_center_array;
        for (std::size_t i = 0; i < 3; ++i)
            cell_center_array[i] = cell_center_h(i);
        
        // Use the center of the activated cell, the positions of the incoming data, and the 
        // scalar values attached to the incoming data to compute multipole coefficients.
        Kernel::Scalar::P2M<memory_space, execution_space> p2m( p );
        auto positions = Cabana::slice<0>(cell_data);
        auto scalars = Cabana::slice<1>(cell_data);
        p2m(positions, scalars, view_size, cell_center_array);
        auto M_coefficients = p2m.coefficients();

        /**
         * Set the cell data for this cell:
         *  1. The multipole coefficients calculated from child data.
         *  2. The center of this cell.
         *  3. The local id of this cell.
         *  4. The rank that owns this cell.
         *  5. Where this cell indexes into the _locals view (via ijk2l map)
         */
        auto local_view_index = _local_view_index;
        auto ijk2l = _ijk2l;
        auto aosoa = _cells_ptr->aosoa();
        // aosoa.resize(aosoa.capacity());
        // printf("L%d: R%d: cell aosoa capacity: %d, size: %d\n", _layer_number, _rank, _cells_ptr->capacity(), _cells_ptr->size());
        Kokkos::parallel_for(
            "set_cell_data",
            Kokkos::RangePolicy<execution_space>( 0, 1 ),
            KOKKOS_LAMBDA( const int i ) {

                tuple_type tp;

                // Multipole coefficients
                for (std::size_t j = 0; j < ((p+1)*(p+1)); ++j)
                {
                    Cabana::get<0>(tp, j, 0) = M_coefficients(j).real();
                    Cabana::get<0>(tp, j, 1) = M_coefficients(j).imag();
                    // printf("R%d: cid: %d: P2M(%d): (%0.4lf, %0.4lf)\n", rank, out_id_slice(start),
                    //     j, Cabana::get<0>(tp, j, 0), Cabana::get<0>(tp, j, 1));
                }

                // Cell center
                for (std::size_t j = 0; j < 3; ++j)
                    Cabana::get<1>(tp, j) = cell_center(j);

                // Cell id
                Cabana::get<2>(tp) = out_id_slice(start);

                // Rank
                Cabana::get<3>(tp) = rank;

                auto cid = out_id_slice(start);
                auto index = cid2ijk.find(cid);
                auto cell_ijk = cid2ijk.value_at(index);

                auto tid = map.queryTile(cell_ijk[0],
                                            cell_ijk[1],
                                            cell_ijk[2]);
                auto ctid = map.cell_local_id(cell_ijk[0],
                                            cell_ijk[1],
                                            cell_ijk[2]);
                aosoa.setTuple(( tid << cell_bits_per_tile ) |
                               ( ctid & cell_mask_per_tile ), tp );

                // Index into local view
                auto idx = Kokkos::atomic_fetch_add(&local_view_index(), 1);
                auto result = ijk2l.insert( Kokkos::Array<std::size_t, 3>{
                                                cell_ijk[0],
                                                cell_ijk[1],
                                                cell_ijk[2]}, idx);
                if (!result.success())
                {
                    // printf("Did not add local index\n");
                }
                // else
                // {
                //     printf("L%d: R%d: cell (%d, %d, %d), lid: %d\n", layer_number, rank, cell_ijk[0],
                //                                 cell_ijk[1],
                //                                 cell_ijk[2], idx);
                // }

                // printf("R%d: setting leaf c(%.3lf, %.3lf, %.3lf)\n",
                //     rank, cell_center(0), cell_center(1), cell_center(2));
                // printf("R%d: setting leaf cid %d, c(%.3lf, %.3lf, %.3lf)\n",
                //     rank, out_id_slice(start), tid2, ctid2,
                //     ( tid2 << cell_bits_per_tile ) | ( ctid2 & cell_mask_per_tile ),
                //     cell_center(0), cell_center(1), cell_center(2));
            });
            // printf("L%d: R%d: cell aosoa capacity: %d, size: %d\n", _layer_number, _rank, aosoa.capacity(), aosoa.size());
    }

    /**
     * xxx
     */
    template <class In2OutMap>
    void initializeCell(const data_aosoa_type incoming_data, In2OutMap in2out,
        const std::size_t start, const std::size_t end)
    {
        if (!(_layer_number > 0))
        {
            throw std::runtime_error("TreeLayer::initializeCell: must be called with _layer_number > 0");
        }

        int rank = _rank;
        // int layer_number = _layer_number;
        // printf("R%d: leaf cell data from [%d, %d)\n", rank, start, end);

        std::size_t view_size = end - start;
        data_aosoa_type cell_data("cell_data", view_size);

        // The following is needed to retrieve and save cell center
        auto in_id_slice = Cabana::slice<0>(in2out);
        auto out_id_slice = Cabana::slice<1>(in2out);
        Kokkos::View<double[3], memory_space> cell_center("cell_center");
        Kokkos::Array<double, 3> low_corner = {_global_low_corner[0], _global_low_corner[1], _global_low_corner[2]};
        auto cell_size = _cell_size;
        auto cid2ijk = _cid2ijk;

        // Save the tile id and cell tile id to set the approportiate index
        // in the sparse mesh AoSoA.
        auto map = *_map_ptr;
        Kokkos::View<std::size_t, memory_space> tid("tid");
        Kokkos::View<std::size_t, memory_space> ctid("ctid");

        // Save cell centers for multipole translations
        Kokkos::View<double*[3], memory_space> incoming_cell_centers("incoming_cell_centers", view_size);

        Kokkos::parallel_for(
            "populate_cell_data",
            Kokkos::RangePolicy<execution_space>( 0, view_size ),
            KOKKOS_LAMBDA( const std::size_t i ) {

                std::size_t index = i + start;
                std::size_t in_id = in_id_slice(index);
                auto data_tuple = incoming_data.getTuple(in_id);
                cell_data.setTuple(i, data_tuple);

                // printf("L%d: R%d: getting in cell c(%.3lf, %.3lf, %.3lf)\n",
                //     layer_number, rank,
                //     Cabana::get<1>(data_tuple, 0), Cabana::get<1>(data_tuple, 1), Cabana::get<1>(data_tuple, 2));

                // Save incoming positions, which are needed for multipole translation.
                incoming_cell_centers(i, 0) = Cabana::get<1>(data_tuple, 0);
                incoming_cell_centers(i, 1) = Cabana::get<1>(data_tuple, 1);
                incoming_cell_centers(i, 2) = Cabana::get<1>(data_tuple, 2);

                // Set data specific to this cell
                if (i == 0)
                {
                    // Since all incoming data are in the same cell, these values
                    // will be the same for all threads so only one thread needs to
                    // compute them.
                    auto cid = out_id_slice(in_id);
                    auto index = cid2ijk.find(cid);

                    // Get the cell center for multipole calculations using the cid
                    auto cell_ijk = cid2ijk.value_at(index);
                    auto cell_center_array = cellCenter(cell_ijk[0], cell_ijk[1], cell_ijk[2], low_corner, cell_size);
                    // printf("R%d: cid: %llu, ijk: %llu, %llu, %llu\n",
                    //     rank,
                    //     (unsigned long long)cid,
                    //     (unsigned long long)cell_ijk[0],
                    //     (unsigned long long)cell_ijk[1],
                    //     (unsigned long long)cell_ijk[2]);
                    for (int j = 0; j < 3; ++j)
                    {
                        cell_center(j) = cell_center_array[j];
                    }

                    // Save the tile id and cell tile id
                    // tid() = map.queryTile(cell_ijk[0],
                    //                         cell_ijk[1],
                    //                         cell_ijk[2]);
                    // ctid() = map.cell_local_id(cell_ijk[0],
                    //                         cell_ijk[1],
                    //                         cell_ijk[2]);
                }
            });
            
        Kokkos::fence();

        // Copy cell centers to host
        auto cell_center_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cell_center);
        auto incoming_cell_centers_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), incoming_cell_centers);

        // Create objects needed for translation of multipole coefficients.
        Canopy::Kernel::Scalar::M2M<memory_space, execution_space> m2m( p );
        auto M_slice = Cabana::slice<0>(cell_data);
        std::size_t M_size = (p+1)*(p+1);
        Kokkos::View<cdouble*, memory_space> M("M", M_size);

        // Iterate over each incoming data.
        for (std::size_t i = 0; i < view_size; ++i)
        {
            // Fill multipole view from coefficients of incoming data.
            Kokkos::parallel_for("set M",
                Kokkos::RangePolicy<execution_space>( 0, M_size ),
                KOKKOS_LAMBDA( const std::size_t j ) {
                    double real_part = M_slice(i, j, 0);
                    double imag_part = M_slice(i, j, 1);
                    M(j) = cdouble(real_part, imag_part);
                    // auto cid = out_id_slice(start);
                    // printf("L%d: R%d: cid: %d, i%d: M(%d): (%.3lf, %.3lf)\n", layer_number, rank, cid,
                    //     i, j, real_part, imag_part);
            });

            // Create Kokkos:Array of vector pointing from child cell center to cell center.
            Kokkos::Array<double, 3> vector_to_center;
            Kokkos::Array<double, 3> child_center = {incoming_cell_centers_h(i, 0),
                incoming_cell_centers_h(i, 1), incoming_cell_centers_h(i, 2)};
            
            for (int j = 0; j < 3; ++j)
                vector_to_center[j] = (cell_center_h(j) - child_center[j]) * -1;
            
            // Translate and add coefficients.
            m2m(M, vector_to_center);
        }

        // Retrieve coefficients
        auto M_coefficients = m2m.coefficients();

        // Set cell data for this cell
        auto local_view_index = _local_view_index;
        auto ijk2l = _ijk2l;
        auto aosoa = _cells_ptr->aosoa();
        // printf("L%d: R%d: cell aosoa capacity: %d, size: %d\n", _layer_number, _rank, aosoa.capacity(), _cells_ptr->size());
        Kokkos::parallel_for(
            "set_cell_data",
            Kokkos::RangePolicy<execution_space>( 0, 1 ),
            KOKKOS_LAMBDA( const int i ) {

                tuple_type tp;

                // Multipole coefficients
                for (std::size_t j = 0; j < ((p+1)*(p+1)); ++j)
                {
                    Cabana::get<0>(tp, j, 0) = M_coefficients(j).real();
                    Cabana::get<0>(tp, j, 1) = M_coefficients(j).imag();
                    // printf("L%d: R%d: cid: %d, i%d: M(%d): (%.3lf, %.3lf)\n", layer_number, rank, out_id_slice(start),
                    //     i, j, Cabana::get<0>(tp, j, 0), Cabana::get<0>(tp, j, 1));
                }

                // Cell center
                for (std::size_t j = 0; j < 3; ++j)
                    Cabana::get<1>(tp, j) = cell_center(j);

                // Cell id. All incoming data have the same out cell id
                auto cid = out_id_slice(start);
                Cabana::get<2>(tp) = cid;

                // Rank
                Cabana::get<3>(tp) = rank;

                auto index = cid2ijk.find(cid);
                auto cell_ijk = cid2ijk.value_at(index);

                auto tid = map.queryTile(cell_ijk[0],
                                            cell_ijk[1],
                                            cell_ijk[2]);
                auto ctid = map.cell_local_id(cell_ijk[0],
                                            cell_ijk[1],
                                            cell_ijk[2]);
                aosoa.setTuple(( tid << cell_bits_per_tile ) |
                               ( ctid & cell_mask_per_tile ), tp );

                // Index into local view
                auto idx = Kokkos::atomic_fetch_add(&local_view_index(), 1);
                auto result = ijk2l.insert( Kokkos::Array<std::size_t, 3>{
                                                cell_ijk[0],
                                                cell_ijk[1],
                                                cell_ijk[2]}, idx);
                if (!result.success())
                {
                    // Shouldn't get here as all cells activated are unique
                }

                // printf("L%d: R%d: cid: %d, tid: %d, ctid: %d, tuple %d, c_ijk(%d, %d, %d)\n", layer_number, rank, cid,
                //     tid, ctid,
                //     ( tid << cell_bits_per_tile ) | ( ctid & cell_mask_per_tile ),
                //     cell_ijk[0], cell_ijk[1], cell_ijk[2]);
                // printf("L%d: R%d: setting out cell c(%.3lf, %.3lf, %.3lf)\n",
                //     layer_number, rank,
                //     Cabana::get<1>(tp, 0), Cabana::get<1>(tp, 1), Cabana::get<1>(tp, 2));
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
    void populateCells(const ParticleAoSoA data_aosoa)
    {
        int rank = _rank;
        int layer_number = _layer_number;

        updateCellSize();

        std::size_t num_particles = data_aosoa.size();

        // Initialize _cid2ijk
        _cid2ijk.clear();
        _cid2ijk.rehash(num_particles);

        // If ParticleAoSoA type is data_aosoa_type, then the positions are the second tuple element.
        // Otherwise they are the first
        static constexpr std::size_t position_index =
            std::is_same_v<ParticleAoSoA, data_aosoa_type> ? 1 : 0;
        auto positions = Cabana::slice<position_index>(data_aosoa);

        auto map = *_map_ptr;
        auto cid2ijk = _cid2ijk;

        auto cell_size = _cell_size;

        // Convert std::array to Kokkos::Array
        Kokkos::Array<double, 3> low_corner = {_global_low_corner[0], _global_low_corner[1], _global_low_corner[2]};
        
        // Map for incoming ids (iid) -> the local id cell they are a part of (cid)
        using map_tuple_type = Cabana::MemberTypes<std::size_t, std::size_t>; 
        using map_aosoa_type = Cabana::AoSoA<map_tuple_type, memory_space, cell_per_tile_dim>;
        map_aosoa_type in2out("in2out", num_particles);
        auto in_id_slice = Cabana::slice<0>(in2out);
        auto out_id_slice = Cabana::slice<1>(in2out);

        Kokkos::parallel_for(
            "registerSparseMap",
            Kokkos::RangePolicy<execution_space>( 0, num_particles ),
            KOKKOS_LAMBDA( const int pid ) {
                auto cell_activated_ijk = position2ijk(positions( pid, 0 ), positions( pid, 1 ), positions( pid, 2 ),
                    low_corner, cell_size);

                // if (layer_number == 1)
                // {
                //     printf("L%d: R%d: pos: %.1lf, %.1lf, %.1lf, ijk: %llu, %llu, %llu, cell size: %0.3lf\n",
                //         layer_number, rank,
                //         positions( pid, 0 ), positions( pid, 1 ), positions( pid, 2 ),
                //         (unsigned long long)cell_activated_ijk[0],
                //         (unsigned long long)cell_activated_ijk[1],
                //         (unsigned long long)cell_activated_ijk[2], cell_size[0]);
                // }

                // register grids that will have data transfer with the particle
                map.insertCell( cell_activated_ijk[0], cell_activated_ijk[1],
                                cell_activated_ijk[2] );
                                        
                // Local cell id
                auto cell_id = map.queryCell(cell_activated_ijk[0],
                                         cell_activated_ijk[1],
                                         cell_activated_ijk[2]);
                // Local tile id
                auto tile_id = map.queryTile(cell_activated_ijk[0],
                                         cell_activated_ijk[1],
                                         cell_activated_ijk[2]);
                // Cell id within the tile
                auto ctid = map.cell_local_id(cell_activated_ijk[0],
                                         cell_activated_ijk[1],
                                         cell_activated_ijk[2]);

                // Save cell activated ijk
                auto result = cid2ijk.insert(cell_id,
                                            Kokkos::Array<std::size_t, 3>{
                                                cell_activated_ijk[0],
                                                cell_activated_ijk[1],
                                                cell_activated_ijk[2]});
                // if (rank == 0)
                //     printf("R%d: vgid_parent %d, vowner: %d, result: %d key: %" PRIu64 "\n", rank,
                //         vgid_parent, vert_owner, result.success(), hash_key);
                if (!result.success())
                {
                    // Getting here means some particles activate the same cell.
                    // printf("Rank %d: incoming tuple %d activates already activated cell at cell id %d\n", rank, pid, cell_id);
                }
                // if (result.success() && layer_number == 0)
                // {
                //     printf("Insert: L%d: R%d: cid: %llu, ijk: %llu, %llu, %llu, cell size: %0.3lf\n",
                //         layer_number, rank,
                //         (unsigned long long)cell_id,
                //         (unsigned long long)cell_activated_ijk[0],
                //         (unsigned long long)cell_activated_ijk[1],
                //         (unsigned long long)cell_activated_ijk[2], cell_size[0]);
                // }

                // Save the cell the incoming data activates.
                // The following line appears redundant, but later this
                // AoSoA is sorted by cell_id.
                in_id_slice(pid) = static_cast<std::size_t>(pid);
                out_id_slice(pid) = static_cast<std::size_t>(cell_id);
            } );

        Kokkos::fence();
        
        // if (_layer_number == 1)
        //     for (int i = 0; i < num_particles; ++i)
        //     {
        //         printf("L%d: R%d: num_p: %d, in2out cell: %d -> %d\n", _layer_number, _rank, num_particles,
        //             in_id_slice(i), out_id_slice(i));
        //     }
        

        // Allocate memory for the AoSoA which stores cell data
        _cells_ptr->reserveFromMap( 1.1 );

        // Size the AoSoA based on how many cells have been activated.
        _cells_ptr->resize( map.sizeCell() );

         // Initialize _ijk2l and _locals to slightly larger than the number of cells activated.
        // XXX - Do we need to do this overallocation?
        std::size_t allocation_size = static_cast<std::size_t>(map.sizeCell() * 1.2);
        _ijk2l.clear();
        _ijk2l.rehash(allocation_size);
        _locals = Kokkos::View<cdouble*[(p+1)*(p+1)], memory_space>("_locals", allocation_size);
        
        // Sort the in2out array and by increasing cell_id
        auto sort_data = Cabana::sortByKey( out_id_slice );
        Cabana::permute( sort_data, in2out );

        // Now, all incoming data that is mapped to the same cell in this layer appears next to
        // each other in the AoSOA.
        using host_aosoa_type = Cabana::AoSoA<map_tuple_type, Kokkos::HostSpace, cell_per_tile_dim>; // XXX - Set vector size?
        host_aosoa_type in2out_h("host_cid_pid_map", num_particles);
        Cabana::deep_copy(in2out_h, in2out);
        auto out_id_h = Cabana::slice<1>(in2out_h);
        std::size_t index = 0;
        while (index < num_particles)
        {
            // Find the start and end indices of each group of incoming data
            // that activate the same cell.
            std::size_t cid = out_id_h(index);
            std::size_t start = index;
            while ((cid == out_id_h(index)) && (index < num_particles))
            {
                index++;
            }
            std::size_t end = index;

            // Now, initialize each cell in this layer one at a time, passing the incoming data
            // AoSoA, the cell id they activate, and the start and end indicies of all
            // incoming data within the cell.

            // Leaf data
            if constexpr (position_index == 0) initializeLeafCell(data_aosoa, in2out, start, end);
            // Non-leaf data
            if constexpr (position_index == 1) initializeCell(data_aosoa, in2out, start, end);
        }

        // Test optimizing the partition after all cells initialized.
        // printf("R%d: L%d: sparse map size: %d\n", _rank, _layer_number, map.size());
        // printf("R%d: sparse map size: %d\n", _rank, map.size());
        // auto imbalance_factor = _partitioner_ptr->computeImbalanceFactor( _cart_comm );
        // printf("R%d: L%d: imbalance factor: %0.4lf\n", _rank, _layer_number, imbalance_factor);
        // if (_layer_number == 0) optimizePartition();
        // auto imbalance_factor = partitioner_ptr->computeImbalanceFactor( _cart_comm );
        // printf("R%d: L%d: imbalance factor: %0.4lf\n", imbalance_factor);
    }

    /**
     * Computes the interaction list for each cell in the layer.
     * 
     * The interaction list of cell0 is the set of all cells such that:
     *  1) cell0 and cell_other are on the same layer of the tree.
     *  2) cell0 and cell_other do not touch.
     *  3) The parent cells of cell0 and cell_other do touch.
     */
    void computeInteractionList(Kokkos::Array<double, 3> parent_cell_size)
    {
        
    }

    /**
     * For each cell, iterate over all cells in its interaction list. These are cells that
     * are at least two cells away from the cell in question and have not been accounted
     * for in more coarse layers.
     * Convert the multipole coefficients centered around the other cell to local coefficients
     * centered around this cell.
     */
    void multipole_to_local(int outer_cell_cutoff)
    {
        // This only works for one process right now.
        if (_comm_size != 1)
        {
            throw std::runtime_error("multipole_to_local only works for comm_size 1");
        }
        printf("L%d: R%d: cells per dim: %d, cutoff: %d\n",  _layer_number, _rank, _cells_per_dim, outer_cell_cutoff);
        auto locals = _locals;
        auto map = *_map_ptr;
        auto aosoa = _cells_ptr->aosoa();
        auto cid2ijk = _cid2ijk;
        auto ijk2l = _ijk2l;

        auto m_slice = Cabana::slice<0>(aosoa);
        auto cell_center_slice = Cabana::slice<1>(aosoa);

        int cells_per_dim = _cells_per_dim;
        int rank = _rank;
        int layer_number = _layer_number;

        // Per-cell calculation
        Kokkos::parallel_for("multipole_to_local",
        Kokkos::RangePolicy<execution_space>(0, cid2ijk.capacity()),
        KOKKOS_LAMBDA(const int cid2ijk_index)
        {
            if (cid2ijk.valid_at(cid2ijk_index))
            {
                // Cell ijk
                auto cell_ijk = cid2ijk.value_at( cid2ijk_index );

                // Cell local index
                auto ijk2l_index = ijk2l.find(cell_ijk);
                auto local_index = ijk2l.value_at(ijk2l_index);
                
                // Index of this cell into sparse array AoSoA
                auto tid = map.queryTile(cell_ijk[0], cell_ijk[1], cell_ijk[2]);
                auto ctid = map.cell_local_id(cell_ijk[0], cell_ijk[1], cell_ijk[2]);
                auto this_cell_index = ( tid << cell_bits_per_tile ) | ( ctid & cell_mask_per_tile );

                // Center of this cell, which is the local center
                Kokkos::Array<double, 3> local_center;

                // Set outer bound - where cells have been accounted for in
                // more coarse layers.
                // Set inner bound - where cells are too close for the local
                // approximation to be accurate. Inclusive on lower end,
                // exclusive on upper end
                Kokkos::Array<int, 3> outer_lower_bound;
                Kokkos::Array<int, 3> outer_upper_bound;
                Kokkos::Array<int, 3> inner_lower_bound;
                Kokkos::Array<int, 3> inner_upper_bound;
                for (int i = 0; i < 3; i++)
                {
                    outer_upper_bound[i] = Kokkos::min(static_cast<int>(cell_ijk[i]) + outer_cell_cutoff, cells_per_dim);
                    outer_lower_bound[i] = Kokkos::max(static_cast<int>(cell_ijk[i]) - outer_cell_cutoff, 0);

                    inner_upper_bound[i] = Kokkos::min(static_cast<int>(cell_ijk[i]) + 3, cells_per_dim);
                    inner_lower_bound[i] = Kokkos::max(static_cast<int>(cell_ijk[i]) - 2, 0);

                    local_center[i] = cell_center_slice(this_cell_index, i);
                }
                
                // if (rank == 0 && layer_number == 0)
                // printf("L%d: R%d: considering cell %d, %d, %d, o: (%d, %d, %d), (%d, %d, %d), i: (%d, %d, %d), (%d, %d, %d)\n",
                //     layer_number, rank,
                //     cell_ijk[0], cell_ijk[1], cell_ijk[2],
                //     outer_lower_bound[0], outer_lower_bound[1], outer_lower_bound[2],
                //     outer_upper_bound[0], outer_upper_bound[1], outer_upper_bound[2],
                //     inner_lower_bound[0], inner_lower_bound[1], inner_lower_bound[2],
                //     inner_upper_bound[0], inner_upper_bound[1], inner_upper_bound[2]);
                // if (rank == 0 && layer_number == 0)
                // printf("L%d: R%d: considering cell %d, %d, %d, li: %d\n",
                //     layer_number, rank,
                //     cell_ijk[0], cell_ijk[1], cell_ijk[2],
                //     local_index);

                // Iterate over all cells whose multipoles we must consider.
                // XXX - Make this a team policy nested for loop
                for (int ci = outer_lower_bound[0]; ci < outer_upper_bound[0]; ci++)
                    for (int cj = outer_lower_bound[1]; cj < outer_upper_bound[1]; cj++)
                        for (int ck = outer_lower_bound[2]; ck < outer_upper_bound[2]; ck++)
                        {
                            // Only consider cells between our outer lower and inner lower
                            // or inner upper and outer upper bounds. If inside these bounds,
                            // skip.
                            if ((ci >= inner_lower_bound[0] && ci < inner_upper_bound[0]) &&
                            (cj >= inner_lower_bound[1] && cj < inner_upper_bound[1]) &&
                            (ck >= inner_lower_bound[2] && ck < inner_upper_bound[2]))
                            {
                                continue;
                            }

                            // Check if this cell is activated by checking if it's recorded
                            // in the cell id to local index map.
                            // XXX - for now, we assume this cell is haloed if necessary and
                            // activated in the sparse map.
                            Kokkos::Array<std::size_t, 3> neighbor_ijk = {ci, cj, ck};                   
                            auto neighbor_activated = ijk2l.exists(neighbor_ijk);
                            // auto neighbor_lid = ijk2l.value_at(neighbor_i);
                            if (!neighbor_activated)
                            {
                                // Cell not activated; do not consider
                                continue;
                            }
                            // if (rank == 0 && layer_number == 0 && cell_ijk[0] == 10 && cell_ijk[1] == 10 && cell_ijk[2] == 3)
                            // printf("L%d: R%d: cell %d, %d, %d, neighbor %d, %d, %d\n", layer_number, rank,
                            //     cell_ijk[0], cell_ijk[1], cell_ijk[2], ci, cj, ck);

                            // Otherwise get the data
                            auto n_tid = map.queryTile(ci, cj, ck);
                            auto n_ctid = map.cell_local_id(ci, cj, ck);
                            auto neighbor_index = ( n_tid << cell_bits_per_tile ) | ( n_ctid & cell_mask_per_tile );
                            
                            // For multipole to local conversion we need the multipole
                            // center relative to the local center
                            Kokkos::Array<double, 3> m2l_vec;
                            for (int i = 0; i < 3; ++i)
                                m2l_vec[i] = cell_center_slice(neighbor_index, i) - cell_center_slice(this_cell_index, i);
                            
                            // Multipole coefficients
                            constexpr std::size_t num_coefficients = (p+1)*(p+1);
                            Kokkos::Array<cdouble, num_coefficients> M;
                            for (std::size_t i = 0; i < num_coefficients; i++)
                            {
                                M[i].real() = m_slice(neighbor_index, i, 0);
                                M[i].imag() = m_slice(neighbor_index, i, 1);
                            }

                            // Convert to locals
                            Kokkos::Array<cdouble, num_coefficients> L;
                            Kernel::Scalar::m2l<p>(M, L, m2l_vec);
                            
                            // Add contribution to locals for this cell
                            for (std::size_t i = 0; i < num_coefficients; i++)
                                locals(local_index, i) += L[i];
                        }
                // if (cell_ijk[0] == 10 && cell_ijk[1] == 10 && cell_ijk[2] == 3)
                // printf("L%d: R%d: cell(%d, %d, %d): locals: %.1lf, %.1lf, %.1lf\n", layer_number, rank,
                //     cell_ijk[0], cell_ijk[1], cell_ijk[2],
                //     locals(local_index, 0).real(), locals(local_index, 1).real(),
                //     locals(local_index, 2).real(), locals(local_index, 3).real());
            }
        });
    }

    void printOwnedCells()
    {
        // Test to iterate over call data
        int rank = _rank;
        auto array = *_cells_ptr;
        auto cell_ids_map = _cid2ijk;
        // printf("R%d: amp size: %d, capacity: %d\n", rank, cell_ids_map.size(), cell_ids_map.capacity());
        Kokkos::View<int, memory_space> valid("valid");
        Kokkos::deep_copy(valid, 0);
        Kokkos::parallel_for(
        "iterate cell data",
        Kokkos::RangePolicy<execution_space>( 0, cell_ids_map.capacity() ),
        KOKKOS_LAMBDA( const int index ) {
            if ( cell_ids_map.valid_at( index ) )
            {
                auto ids = cell_ids_map.value_at( index ); // pair(tid, cid)
                auto tkey = cell_ids_map.key_at( index ); // cglid
                // if (rank == 0) printf("R%d: valid tid, key: %d, %d\n", rank, tid, tkey);
                
                double x = array.template get<0>( ids[0], ids[1], 0 );
                double y = array.template get<0>( ids[0], ids[1],  1 );
                double z = array.template get<0>( ids[0], ids[1],  2 );
                int val = array.template get<1>( ids[0], ids[1]);
                printf("R%d: val: %d, x/y/z: %0.3lf, %0.3lf, %0.3lf\n", rank, val, x, y, z);
                Kokkos::atomic_fetch_add(&valid(), 1);
            }
        } );
        int v;
        Kokkos::deep_copy(v, valid);
        if (v == 0) printf("R%d: No cells to print.\n", rank);
    }

    int rank() const { return _rank; }
    int layerNumber() const { return _layer_number; }

    /**
     * Get the domain in 3D space that each rank owns with the upper value being non-inclusive
     * Each entry in the returned vector is (x_start, y_start, z_start, x_end, y_end, z_end)
     */
    auto domains() const {return _domains;}
    auto tile_offsets() const {return _tile_offsets_view;}
    auto num_owned_tile() const {return _num_owned_tile_view;}

    std::shared_ptr<sparse_layout_type> layout() {return _layout_ptr;}
    std::shared_ptr<sparse_array_type> array() {return _cells_ptr;}
    std::shared_ptr<sparse_map_type> map() {return _map_ptr;}
    int cellsPerDim() const {return _cells_per_dim;}
    int tilesPerDim() const {return _tiles_per_dim;}
    Kokkos::Array<double, 3> cellSize() const {return _cell_size;}
    Kokkos::UnorderedMap<int, Kokkos::Array<std::size_t, 3>, memory_space>& cid2ijk() {return _cid2ijk;}

    // Get the local coefficients
    auto locals() {return _locals;}

    // Return cell_ijk to index into local view map
    auto cellijk2l() {return _ijk2l;}

  private:
    const std::array<double, 3> _global_high_corner;
    const std::array<double, 3> _global_low_corner;
    std::array<int, 3> _global_num_cell;
	const int _tiles_per_dim;
    const int _halo_width;
    const int _cells_per_dim;
    const int _layer_number;
    int _rank, _comm_size;

    // Cell size in the x, y, and z dimensions.
    Kokkos::Array<double, 3> _cell_size;

    // Information about which processes own which other
    // section of the sparse mesh
    Kokkos::View<int*[3], memory_space> _tile_offsets_view;
    Kokkos::View<int*[3], memory_space> _num_owned_tile_view;
    Kokkos::View<double*[6], memory_space> _domains;

    // Partitioner parameters
    int _num_step_rebalance, _max_optimize_iteration;

    // MPI communicators
    const MPI_Comm _comm;
    MPI_Comm _cart_comm;
    
    std::shared_ptr<sparse_partitioner_type> _partitioner_ptr;
    std::shared_ptr<sparse_layout_type> _layout_ptr;
    std::shared_ptr<sparse_map_type> _map_ptr;
    std::shared_ptr<sparse_array_type> _cells_ptr;

    // Map of cell id (cid, unique per process) to cell ijk position.
    // Cabana supports ijk -> cid but not the inverse.
    Kokkos::UnorderedMap<int, Kokkos::Array<std::size_t, 3>, memory_space> _cid2ijk;

    // Map of cell_ijk to its location in the Local view.
    Kokkos::View<std::size_t, memory_space> _local_view_index;
    Kokkos::UnorderedMap<Kokkos::Array<std::size_t, 3>, std::size_t, memory_space> _ijk2l;

    // Locals coefficients for each cell. This data is haloed differently than
    // multipole coefficients so it is stored outside of the sparse mesh.
    Kokkos::View<cdouble*[(p+1)*(p+1)], memory_space> _locals;
};

template <class TreeType, std::size_t CellPerTileDim>
std::shared_ptr<TreeLayer<TreeType, CellPerTileDim>> createTreeLayer(const std::array<double, 3>& global_low_corner,
            const std::array<double, 3>& global_high_corner,
	        const int tiles_per_dim, const int halo_width,
            const int layer_number,
            MPI_Comm comm)
{
    return std::make_shared<TreeLayer<TreeType, CellPerTileDim>>(global_low_corner,
            global_high_corner,
	        tiles_per_dim, halo_width, layer_number, comm);
}

} // end namespace Canopy

#endif // CANOPY_TREELAYER_HPP
