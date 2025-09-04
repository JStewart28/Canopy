#ifndef CANOPY_TREELAYER_HPP
#define CANOPY_TREELAYER_HPP

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <memory>

#include <mpi.h>

#include <limits>

namespace Canopy
{

// https://repositorio.unesp.br/server/api/core/bitstreams/0e824479-3128-41f7-8cd2-462e9a242c42/content

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

    using sparse_partitioner_type = typename TreeType::sparse_partitioner_type;

    //! DataTypes Data types (Cabana::MemberTypes).
    using member_types = typename TreeType::member_types;

    using entity_type = typename TreeType::entity_type;

    using mesh_type = typename TreeType::mesh_type;

    static constexpr std::size_t position_slice_id = TreeType::position_slice_id;

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
    using data_aosoa_type = typename TreeType::data_aosoa_type; // AoSoA type that holds cell data
    
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

        std::array<int, 3> global_num_cell({
            _cells_per_dim,
            _cells_per_dim,
            _cells_per_dim
            });
        // printf("R%d: high-low: %0.2lf, %0.2lf, %0.2lf, _tiles_per_dim: %d\n", _rank,
        //     _global_high_corner[0] - _global_low_corner[0],
        //     _global_high_corner[1] - _global_low_corner[1],
        //     _global_high_corner[2] - _global_low_corner[2],
        //     _tiles_per_dim);
        // printf("R%d: global_num_cell: %d, %d, %d\n", _rank, global_num_cell[0], global_num_cell[1], global_num_cell[2]);
                
        // sparse partitioner
        float max_workload_coeff = 1.5;
        int workload_num = _tiles_per_dim * _tiles_per_dim * _tiles_per_dim;
        _num_step_rebalance = 200;
        _max_optimize_iteration = 10;
        _partitioner_ptr = std::make_shared<sparse_partitioner_type>(
            _comm, max_workload_coeff, workload_num, _num_step_rebalance,
            global_num_cell, _max_optimize_iteration );
        
        auto ranks_per_dim =
            _partitioner_ptr->ranksPerDimension( comm, global_num_cell );
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

        initializeRecPartition(x_partition, y_partition, z_partition);

        // mesh/grid related initialization
        auto global_mesh = Cabana::Grid::createSparseGlobalMesh(
            global_low_corner, global_high_corner, global_num_cell );
        
        std::array<bool, 3> is_dim_periodic = { false, false, false };
        auto& partitioner_ref = *_partitioner_ptr;
        auto global_grid = Cabana::Grid::createGlobalGrid( _comm, global_mesh,
                                            is_dim_periodic, partitioner_ref );
        auto local_grid =
            Cabana::Grid::Experimental::createSparseLocalGrid( global_grid, _halo_width, cell_per_tile_dim );
        sparse_map_type sparse_map =
            Cabana::Grid::createSparseMap<memory_space>( global_mesh, 1.2 );
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

            // Where do you store the persistent gathers and scatters? 
            // How do you tell a halo to create perssitent gathers and scatters
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

    void updateCellSize()
    {
        auto local_grid = _cells_ptr->layout().localGrid();
        auto sparse_mesh = local_grid->globalGrid().globalMesh();
        _cell_size = {sparse_mesh.cellSize( 0 ), sparse_mesh.cellSize( 1 ), sparse_mesh.cellSize( 2 )};
    }

    /*!
      \brief Initialize the tile partition; partition in each dimension
      has the form [0, p_1, ..., p_n, total_tile_num], so the partition
      would be [0, p_1), [p_1, p_2) ... [p_n, total_tile_num]
      \param rec_partition_i partition array in dimension i
      \param rec_partition_j partition array in dimension j
      \param rec_partition_k partition array in dimension k
    */
    void initializeRecPartition( std::vector<int>& rec_partition_i,
                                 std::vector<int>& rec_partition_j,
                                 std::vector<int>& rec_partition_k )
    {
        _partitioner_ptr->initializeRecPartition(rec_partition_i, rec_partition_j, rec_partition_k);
        // auto current_partition = _partitioner_ptr->getCurrentPartition();
        // if (_rank == 0)
        // for (std::size_t d = 0; d < 3; ++d)
        // {
        //     std::cout << "Dimension " << d << ": ";
        //     for (std::size_t i = 0; i < current_partition[d].size(); ++i)
        //     {
        //         std::cout << current_partition[d][i] << " ";
        //     }
        //     std::cout << std::endl;
        // }
    }

    /**
     * Initialize tiles in the layer based on particle locations
     */
    // template <class PositionSliceType>
    // void initializeLayer(int layer, PositionSliceType position_slice, std::size_t num_particles)
    // {
    //     auto array = _tree[layer]->array();
    //     array->registerSparseGrid( position_slice, num_particles );
    //     array->reserveFromMap( 1.2 );
    //     printf("R%d: array size: %d\n", _rank, (int)array->size());
    // }

    /**
     * Get the domain in 3D space that each rank owns with the upper value being non-inclusive
     * Each entry in the returned vector is (x_start, y_start, z_start, x_end, y_end, z_end)
     */
    std::vector<std::array<double, 6>> get_domains()
    {
        auto current_partition = _partitioner_ptr->getCurrentPartition();

        // Get the number of tiles per dimension
        std::array<int, 3> num_tiles_per_dim;
        for (int d = 0; d < 3; ++d)
            num_tiles_per_dim[d] = current_partition[d].back();

        // Get total number of ranks
        int size;
        MPI_Comm_size(_cart_comm, &size);

        // Allocate the result vector
        std::vector<std::array<double, 6>> domains(size);

        for (int rank = 0; rank < size; ++rank)
        {
            int coords[3];
            MPI_Cart_coords(_cart_comm, rank, 3, coords);

            std::array<double, 6> domain;
            for (int d = 0; d < 3; ++d)
            {
                int tile_start = current_partition[d][coords[d]];
                int tile_end   = current_partition[d][coords[d] + 1];

                double global_min = _global_low_corner[d];
                double global_max = _global_high_corner[d];
                double tile_width = (global_max - global_min) / num_tiles_per_dim[d];

                domain[d]     = global_min + tile_start * tile_width;  // lower bound
                domain[d + 3] = global_min + tile_end   * tile_width;  // upper bound
            }

            domains[rank] = domain;
        }

        return domains;
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
        auto domains_host = get_domains();
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

    template <class PositionSliceType>
    bool loadBalance(PositionSliceType position_slice, std::size_t num_particles)
    {
        // if (_rank == 0) for (size_t i = 0; i < num_particles; i++)
        // {
        //     printf("R%d: i%d: %0.3lf, %0.3lf, %0.3lf\n", _rank, i, position_slice(i, 0), position_slice(i, 1), position_slice(i, 2));
        // }
        float dx = (_global_high_corner[0] - _global_low_corner[0]) / _cells_per_dim;
        _partitioner_ptr->optimizePartition( position_slice, num_particles,
                                            _global_low_corner,
                                            dx, _cart_comm );

        // compute prefix sum matrix
        // _partitioner_ptr->computeFullPrefixSum( _cart_comm );

        // // optimization
        // bool is_changed = false;
        // for ( int i = 0; i < _max_optimize_iteration; ++i )
        // {
        //     _partitioner_ptr->optimizePartition( is_changed,
        //                                     std::rand() % 3 );
        //     if ( !is_changed )
        //         break;
        // }

        // return is_changed;
        return false;
    }

    /**
     * Return an AoSoA of cell data, indexed by cell id
     */
    data_aosoa_type data()
    {
        int rank = _rank;

        data_aosoa_type cell_data("cell_data", _cid_tid_map.size());
        Kokkos::View<int, memory_space> idx("idx");
        Kokkos::deep_copy(idx, 0);

        auto aosoa = _cells_ptr->aosoa();
        auto cid_tid_map = _cid_tid_map;

        // printf("R%d: map size: %d\n", rank, cid_tid_map.size());

        // XXX - can this be made more efficient by extracting my SoAs rather than tuples?
        int layer_number = _layer_number;
        Kokkos::parallel_for(
            "get_data",
            Kokkos::RangePolicy<execution_space>(0, cid_tid_map.capacity()),
            KOKKOS_LAMBDA(const int index) {
                if (cid_tid_map.valid_at(index))
                {
                    auto ids = cid_tid_map.value_at( index ); // pair(tid, cid)
                    // auto cglid = cid_tid_map.key_at( index ); // cglid
                    auto tp = aosoa.getTuple(( ids.first << cell_bits_per_tile ) |
                                            ( ids.second & cell_mask_per_tile ) );
                    int offset = Kokkos::atomic_fetch_add(&idx(), 1);
                    // printf("R%d: index %d: cell glid: %d\n", rank, index, cglid);
                    //auto tp = array.getTuple(ids.first, ids.second);
                    // printf("R%d: setting tp %d...\n", rank, offset);
                    // int r = Cabana::get<1>(tp);
                    // double x = Cabana::get<0>(tp, 0);
                    // double y = Cabana::get<0>(tp, 1);
                    // double z = Cabana::get<0>(tp, 2);
                    // if (layer_number == 2) printf("R%d: get_data: L%d: int: %d, pos: %0.3lf, %0.3lf, %0.3lf\n",
                    //     rank, layer_number, r, x, y, z);
                    cell_data.setTuple(offset, tp);
                }
            }
        );

        return cell_data;
    }


    /*******************************************************************************
     * Set by AoSoA
     ******************************************************************************/
    /**
     * Particle to multipole function. Used at the leaf layer to convert 
     */
    template <class MapAoSoAType, class AggregationFunctor>
    void p2m(const data_aosoa_type aosoa_data, const MapAoSoAType data_map, AggregationFunctor agg_functor,
        int start, int end)
    {
        int rank = _rank;
        // printf("R%d: agg data from [%d, %d)\n", rank, start, end);

        // Create an aosoa of the data we want to aggregate
        int view_size = end - start;
        auto cglid_slice = Cabana::slice<0>(data_map);
        auto tid_slice = Cabana::slice<1>(data_map);
        auto clid_slice = Cabana::slice<2>(data_map);
        auto plid_slice = Cabana::slice<3>(data_map);
        data_aosoa_type cell_data("cell_data", view_size);

        // Save the cell ID
        Kokkos::View<int, memory_space> tid("tid");
        Kokkos::View<int, memory_space> clid("clid");

        // printf("R%d: AoSoA data size: %d\n", _rank, aosoa_data.size());

        Kokkos::parallel_for(
            "populate_cell_data",
            Kokkos::RangePolicy<execution_space>( 0, view_size ),
            KOKKOS_LAMBDA( const int i ) {

                int index = i + start;
                int pid = plid_slice(index);
                if (i == 0)
                {
                    // Same values for all threads
                    tid() = tid_slice(index);
                    clid() = clid_slice(index);
                }
                // printf("R%d: getting tuple %d from index %d\n", rank, pid, index);
                auto data_tuple = aosoa_data.getTuple(pid);
                cell_data.setTuple(i, data_tuple);
            });
            
        Kokkos::fence();
        // int tid_h, clid_h;
        // Kokkos::deep_copy(tid_h, tid);
        // Kokkos::deep_copy(clid_h, clid);

        // printf("R%d: tid: %d, cid: %d\n", rank, tid_h, clid_h);
        
        // Aggregate cell data
        agg_functor(cell_data);
        auto vals = agg_functor.vals();
        auto pslice = Cabana::slice<0>(vals);
        auto islice = Cabana::slice<1>(vals);
        int layer_number = _layer_number;
        // printf("R%d: vals size: %d\n", rank, vals.size());
        // if (_layer_number == 2)
        //     for (size_t i = 0; i < vals.size(); i++)
        //     {
        //         printf("R%d: agg_data0: L%d: int: %d, pos: %0.3lf, %0.3lf, %0.3lf\n",
        //             rank, _layer_number, islice(i), pslice(i, 0), pslice(i, 1), pslice(i, 2));
        //     }

        // Set cell data for cell cid
        auto aosoa = _cells_ptr->aosoa();
        // printf("R%d: #2: array capacity: %d, size: %d\n", rank, array.capacity(), array.size());
        Kokkos::parallel_for(
            "set_cell_data",
            Kokkos::RangePolicy<execution_space>( 0, 1 ),
            KOKKOS_LAMBDA( const int i ) {
                auto tp = vals.getTuple(i);
                // double x, y, z;
                // int r;
                // x = Cabana::get<0>(tp, 0);
                // y = Cabana::get<0>(tp, 1);
                // z = Cabana::get<0>(tp, 2);
                // r = Cabana::get<1>(tp);
                // printf("R%d: agg_data1: L%d: int: %d, pos: %0.3lf, %0.3lf, %0.3lf\n",
                //     rank, layer_number, r, x, y, z);
                aosoa.setTuple(( tid() << cell_bits_per_tile ) |
                               ( clid() & cell_mask_per_tile ), tp );
            }); 
    }

    template <class MapAoSoAType, class AggregationFunctor>
    void aggregate_and_add_cell_data(const data_aosoa_type aosoa_data, const MapAoSoAType data_map, AggregationFunctor agg_functor,
        int start, int end)
    {
        int rank = _rank;
        // printf("R%d: agg data from [%d, %d)\n", rank, start, end);

        // Create an aosoa of the data we want to aggregate
        int view_size = end - start;
        auto cglid_slice = Cabana::slice<0>(data_map);
        auto tid_slice = Cabana::slice<1>(data_map);
        auto clid_slice = Cabana::slice<2>(data_map);
        auto plid_slice = Cabana::slice<3>(data_map);
        data_aosoa_type cell_data("cell_data", view_size);

        // Save the cell ID
        Kokkos::View<int, memory_space> tid("tid");
        Kokkos::View<int, memory_space> clid("clid");

        // printf("R%d: AoSoA data size: %d\n", _rank, aosoa_data.size());

        Kokkos::parallel_for(
            "populate_cell_data",
            Kokkos::RangePolicy<execution_space>( 0, view_size ),
            KOKKOS_LAMBDA( const int i ) {

                int index = i + start;
                int pid = plid_slice(index);
                if (i == 0)
                {
                    // Same values for all threads
                    tid() = tid_slice(index);
                    clid() = clid_slice(index);
                }
                // printf("R%d: getting tuple %d from index %d\n", rank, pid, index);
                auto data_tuple = aosoa_data.getTuple(pid);
                cell_data.setTuple(i, data_tuple);
            });
            
        Kokkos::fence();
        // int tid_h, clid_h;
        // Kokkos::deep_copy(tid_h, tid);
        // Kokkos::deep_copy(clid_h, clid);

        // printf("R%d: tid: %d, cid: %d\n", rank, tid_h, clid_h);
        
        // Aggregate cell data
        agg_functor(cell_data);
        auto vals = agg_functor.vals();
        auto pslice = Cabana::slice<0>(vals);
        auto islice = Cabana::slice<1>(vals);
        int layer_number = _layer_number;
        // printf("R%d: vals size: %d\n", rank, vals.size());
        // if (_layer_number == 2)
        //     for (size_t i = 0; i < vals.size(); i++)
        //     {
        //         printf("R%d: agg_data0: L%d: int: %d, pos: %0.3lf, %0.3lf, %0.3lf\n",
        //             rank, _layer_number, islice(i), pslice(i, 0), pslice(i, 1), pslice(i, 2));
        //     }

        // Set cell data for cell cid
        auto aosoa = _cells_ptr->aosoa();
        // printf("R%d: #2: array capacity: %d, size: %d\n", rank, array.capacity(), array.size());
        Kokkos::parallel_for(
            "set_cell_data",
            Kokkos::RangePolicy<execution_space>( 0, 1 ),
            KOKKOS_LAMBDA( const int i ) {
                auto tp = vals.getTuple(i);
                // double x, y, z;
                // int r;
                // x = Cabana::get<0>(tp, 0);
                // y = Cabana::get<0>(tp, 1);
                // z = Cabana::get<0>(tp, 2);
                // r = Cabana::get<1>(tp);
                // printf("R%d: agg_data1: L%d: int: %d, pos: %0.3lf, %0.3lf, %0.3lf\n",
                //     rank, layer_number, r, x, y, z);
                aosoa.setTuple(( tid() << cell_bits_per_tile ) |
                               ( clid() & cell_mask_per_tile ), tp );
            }); 
    }
    /**
     * Takes an AoSoA of particle data where positions is the first tuple.
     * Saves the cell each particle falls in into cell_map
     *  1.
     *  2. Initialize the correct cells based on particle locations. Track which particles belong
     *      to which cells.
     *  3. Aggregate the data for each cell using the AggregationFunctor.
     *  4. Inserts the aggregated data into the correct cell on this layer.
     */
    template <class AggregationFunctor>
    void populateCells(const data_aosoa_type data_aosoa, AggregationFunctor functor)
    {
        int rank = _rank;

        updateCellSize();

        std::size_t num_particles = data_aosoa.size();

        // Initialize _cid_tid_map
        _cid_tid_map.clear();
        _cid_tid_map.rehash(num_particles);

        printf("R%d: L%d: tpd: %d, cpd: %d\n", rank, _layer_number, _tiles_per_dim, _cells_per_dim);

        auto positions = Cabana::slice<position_slice_id>(data_aosoa);

        auto map = *_map_ptr;
        auto cid_tid_map = _cid_tid_map;

        auto array = _cells_ptr;

        auto cells_per_dim = this->cellsPerDim();
        auto tiles_per_dim = this->tilesPerDim();
        Kokkos::Array<double, 3> dx_inv = {
            (double)1.0 / _cell_size[0], (double)1.0 / _cell_size[1],
            (double)1.0 / _cell_size[2] };

        Kokkos::Array<double, 3> low_corner = {_global_low_corner[0], _global_low_corner[1], _global_low_corner[2]};
        
        // 0. global local cell id (unique per tile but not per process)
        // 1. tile id
        // 2. cell local id (unique per tile)
        // 3. particle local id (unique per process but not globally)
        using map_tuple_type = Cabana::MemberTypes<int, int, int, int>; 
        using map_aosoa_type = Cabana::AoSoA<map_tuple_type, memory_space, cell_per_tile_dim>;
        map_aosoa_type cell_id_particle_id_map("cell_id_particle_id_map", num_particles);

        auto cglid_slice = Cabana::slice<0>(cell_id_particle_id_map);
        auto tid_slice = Cabana::slice<1>(cell_id_particle_id_map);
        auto clid_slice = Cabana::slice<2>(cell_id_particle_id_map);
        auto plid_slice = Cabana::slice<3>(cell_id_particle_id_map);
        Kokkos::parallel_for(
            "registerSparseMap",
            Kokkos::RangePolicy<execution_space>( 0, num_particles ),
            KOKKOS_LAMBDA( const int pid ) {
                double pos[3] = { positions( pid, 0 ) - low_corner[0],
                                       positions( pid, 1 ) - low_corner[1],
                                       positions( pid, 2 ) - low_corner[2] };
                int cell_activated_ijk[3] = {
                    static_cast<int>( std::lround( pos[0] * dx_inv[0] ) ),
                    static_cast<int>( std::lround( pos[1] * dx_inv[1] ) ),
                    static_cast<int>( std::lround( pos[2] * dx_inv[2] ) ) };
                // printf("R%d: cell activated: %d, %d, %d\n", rank, cell_activated_ijk[0], cell_activated_ijk[1], cell_activated_ijk[2]);
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
                auto cell_local_id = map.cell_local_id(cell_activated_ijk[0],
                                         cell_activated_ijk[1],
                                         cell_activated_ijk[2]);
                // if (rank == 0) printf("tid: %d, cid: %d, pid: %d, clid: %d, R%d: particle loc %0.3lf, %0.3lf, %0.3lf, ijk: %d, %d, %d\n",
                //     tile_id, cell_id, pid, cell_local_id, rank,
                //     positions( pid, 0 ), positions( pid, 1 ), positions( pid, 2 ),
                //     cell_activated_ijk[0], cell_activated_ijk[1], cell_activated_ijk[2]);
                cglid_slice(pid) = static_cast<int>(cell_id);
                clid_slice(pid) = static_cast<int>(cell_id);
                tid_slice(pid) = static_cast<int>(tile_id);
                clid_slice(pid) = static_cast<int>(cell_local_id);
                plid_slice(pid) = pid;

                auto result = cid_tid_map.insert(cell_id, Kokkos::make_pair(tile_id, cell_local_id));
                // if (rank == 0)
                //     printf("R%d: vgid_parent %d, vowner: %d, result: %d key: %" PRIu64 "\n", rank,
                //         vgid_parent, vert_owner, result.success(), hash_key);
                if (!result.success())
                {
                    // Getting here means some particles activate the same cell.
                    // printf("Rank %d: Particle activates already activated cell at cell id %d\n", rank, cell_id);
                }
            } );
        
        // Allocate memory for the AoSoA which stores cell data
        array->reserveFromMap( 1.1 );
        // printf("R%d: array capacity: %d, size: %d\n", rank, array->capacity(), array->size());
        
        // Sort the cell_id_particle_id_map array and by increasing cell_id
        auto sort_data = Cabana::sortByKey( cglid_slice );
        Cabana::permute( sort_data, cell_id_particle_id_map );

        // Aggregate and insert data into the mesh
        using host_aosoa_type = Cabana::AoSoA<map_tuple_type, Kokkos::HostSpace, 4>; // XXX - Set vector size?
        host_aosoa_type host_cid_pid_map("host_cid_pid_map", num_particles);
        Cabana::deep_copy(host_cid_pid_map, cell_id_particle_id_map);
        auto h_cid_slice = Cabana::slice<0>(host_cid_pid_map);
        auto h_pid_slice = Cabana::slice<1>(host_cid_pid_map);

        // if (rank == 0)
        // {
        //     std::size_t size = host_cid_pid_map.size();
        //     for (std::size_t i = 0; i < size; i++)
        //     {
        //         printf("R%d: cid: %d, pid: %d\n", rank, h_cid_slice(i), h_pid_slice(i));
        //     }
        // }


        std::size_t index = 0;
        while (index < num_particles)
        {
            int cid = h_cid_slice(index);
            int start = index;
            while ((cid == h_cid_slice(index)) && (index < num_particles))
            {
                index++;
            }
            int end = index;
            // printf("R%d: agg data from [%d, %d), cid: %d\n", rank, start, end, cid);
            aggregate_and_add_cell_data(data_aosoa, cell_id_particle_id_map, functor, start, end);
        }
        // printf("R%d: aosoa size: %d\n", _rank, _cells_ptr->size());

        // Kokkos::parallel_for(
        //     "print_cell_data",
        //     Kokkos::RangePolicy<execution_space>( 0, 1 ),
        //     KOKKOS_LAMBDA( const int i ) {
        //         // printf("R%d: Setting tid %d, clid %d...\n", rank, tid(), clid());
        //         if (rank == 0)
        //         {
        //             auto xpos = array->template get<0>( 9, 15, 0 );
        //             auto ypos = array->template get<0>( 9, 15, 1 );
        //             auto zpos = array->template get<0>( 9, 15, 2 );
        //             printf("R%d: 9, 15: %0.3lf, %0.3lf, %0.3lf\n", rank, xpos, ypos, zpos);
        //         }
        //     });
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

    void printOwnedCells()
    {
        // Test to iterate over call data
        int rank = _rank;
        auto array = *_cells_ptr;
        auto cid_tid_map = _cid_tid_map;
        // printf("R%d: amp size: %d, capacity: %d\n", rank, cid_tid_map.size(), cid_tid_map.capacity());
        Kokkos::View<int, memory_space> valid("valid");
        Kokkos::deep_copy(valid, 0);
        Kokkos::parallel_for(
        "iterate cell data",
        Kokkos::RangePolicy<execution_space>( 0, cid_tid_map.capacity() ),
        KOKKOS_LAMBDA( const int index ) {
            if ( cid_tid_map.valid_at( index ) )
            {
                auto ids = cid_tid_map.value_at( index ); // pair(tid, cid)
                auto tkey = cid_tid_map.key_at( index ); // cglid
                // if (rank == 0) printf("R%d: valid tid, key: %d, %d\n", rank, tid, tkey);
                
                double x = array.template get<0>( ids.first, ids.second, 0 );
                double y = array.template get<0>( ids.first, ids.second,  1 );
                double z = array.template get<0>( ids.first, ids.second,  2 );
                int val = array.template get<1>( ids.first, ids.second);
                printf("R%d: val: %d, x/y/z: %0.3lf, %0.3lf, %0.3lf\n", rank, val, x, y, z);
                Kokkos::atomic_fetch_add(&valid(), 1);
            }
        } );
        int v;
        Kokkos::deep_copy(v, valid);
        if (v == 0) printf("R%d: No cells to print.\n", rank);
    }

    /**
     * Return the center of a cell given its ijk location
     */
    KOKKOS_INLINE_FUNCTION
    template <class Scalar>
    static Kokkos::Array<Scalar, 3>
    cellCenter(int i, int j, int k,
               const std::array<Scalar, 3>& global_low_corner,
               const Kokkos::Array<Scalar, 3>& cell_size)
    {
        Kokkos::Array<Scalar, 3> center;
        center[0] = global_low_corner[0] + ((Scalar)i + 0.5) * cell_size[0];
        center[1] = global_low_corner[1] + ((Scalar)j + 0.5) * cell_size[1];
        center[2] = global_low_corner[2] + ((Scalar)k + 0.5) * cell_size[2];
        return center;
    }

    int rank() const { return _rank; }
    int layerNumber() const { return _layer_number; }

    std::shared_ptr<sparse_layout_type> layout() {return _layout_ptr;}
    std::shared_ptr<sparse_array_type> array() {return _cells_ptr;}
    std::shared_ptr<sparse_map_type> map() {return _map_ptr;}
    int cellsPerDim() const {return _cells_per_dim;}
    int tilesPerDim() const {return _tiles_per_dim;}
    Kokkos::Array<double, 3> cellSize() const {return _cell_size;}
    

  private:
    const std::array<double, 3> _global_high_corner;
    const std::array<double, 3> _global_low_corner;
	const int _tiles_per_dim;
    const int _halo_width;
    const int _cells_per_dim;
    const int _layer_number;
    int _rank, _comm_size;

    // Cell size in the x, y, and z dimensions.
    Kokkos::Array<double, 3> _cell_size;

    // Partitioner parameters
    int _num_step_rebalance, _max_optimize_iteration;

    // MPI communicators
    const MPI_Comm _comm;
    MPI_Comm _cart_comm;
    
    std::shared_ptr<sparse_partitioner_type> _partitioner_ptr;
    std::shared_ptr<sparse_layout_type> _layout_ptr;
    std::shared_ptr<sparse_map_type> _map_ptr;
    std::shared_ptr<sparse_array_type> _cells_ptr;

    // Map to store which cells are activated in the mesh
    // tid, cid pair:
    //  tid: The local tile id.
    //  cid: The local cell id within a tile.
    Kokkos::UnorderedMap<int, Kokkos::pair<int, int>, memory_space> _cid_tid_map;
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
