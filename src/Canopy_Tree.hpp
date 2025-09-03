#ifndef CANOPY_TREE_HPP
#define CANOPY_TREE_HPP


#include <Canopy_TreeLayer.hpp>

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

template <class ExecutionSpace, class MemorySpace, class DataTypes, class EntityType,
          std::size_t NumSpaceDim, std::size_t CellPerTileDim, std::size_t CellSliceId>
class Tree
{
  public:
    using execution_space = ExecutionSpace;
    
    using memory_space = MemorySpace;

    //! Self type
    using tree_type = Tree<ExecutionSpace, MemorySpace, DataTypes, EntityType,
        NumSpaceDim, CellPerTileDim, CellSliceId>;

    //! Memory space size type
    using size_type = typename memory_space::size_type;
    //! Array entity type (node, cell, face, edge).
    using entity_type = EntityType;
    //! Dimension number
    static constexpr std::size_t num_space_dim = NumSpaceDim;
    //! Mesh type
    using mesh_type = Cabana::Grid::SparseMesh<double, num_space_dim>;

    static constexpr std::size_t cell_per_tile_dim = CellPerTileDim;

    // The AoSoA slice to use to determine which cell the particle resides in
    static constexpr std::size_t cell_slice_id = CellSliceId;

    // AoSoA related types
    //! DataTypes Data types (Cabana::MemberTypes).
    using member_types = DataTypes;
    using tuple_type = Cabana::Tuple<member_types>;
    using data_aosoa_type = Cabana::AoSoA<member_types, memory_space, cell_per_tile_dim>;

    //! Sparse partitioner type
    using sparse_partitioner_type = Cabana::Grid::SparseDimPartitioner<memory_space, num_space_dim>;
    
    Tree( const std::array<double, 3>& global_low_corner,
            const std::array<double, 3>& global_high_corner,
            const std::size_t leaf_tiles_per_dim,
            const std::size_t tile_reduction_factor,
            const std::size_t root_tiles_per_dim,
            MPI_Comm comm )
        : _global_low_corner( global_low_corner )
        , _global_high_corner( global_high_corner )
        , _leaf_tiles_per_dim( leaf_tiles_per_dim )
        , _tile_reduction_factor( tile_reduction_factor )
        , _root_tiles_per_dim( root_tiles_per_dim )
        , _comm( comm )
    {
        MPI_Comm_rank( comm, &_rank );
        MPI_Comm_size( comm, &_comm_size );

        // Reserve space for 10 layers
        _tree.reserve(10);

        build();
        
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
    
    void add_layer(const int tiles_per_dim, const int halo_width, const int layer_num)
    {
        // printf("R%d: cell_per_tile: %d\n", _rank, cell_per_tile_dim);
        auto layer = createTreeLayer<tree_type, cell_per_tile_dim>(
            _global_low_corner, _global_high_corner, tiles_per_dim, halo_width, layer_num, _comm);
        _tree.push_back(layer);
    }

    void build()
    {
        if (_tile_reduction_factor < 2)
            throw std::runtime_error("Canopy::Tree::build: _tile_reduction_factor must be greater than 1.\n");

        int layer_num = 0;

        std::size_t next_layer_tiles_per_dim = _leaf_tiles_per_dim;
        add_layer(next_layer_tiles_per_dim, 2, layer_num++);

        // auto leaf_tiles_per_dim = _next_layer_tiles_per_dim;

        // Calculate the depth of the tree
        int depth = 0;
        // if (_rank == 0) printf("R%d: Layer %d: tiles: %d, root tiles: %d\n", _rank, depth, next_layer_tiles_per_dim, _root_tiles_per_dim);
        while (next_layer_tiles_per_dim > _root_tiles_per_dim)
        {
            // printf("R%d: next: %d, root: %d\n", _rank, next_layer_tiles_per_dim, _root_tiles_per_dim);
            depth++;
            next_layer_tiles_per_dim = static_cast<std::size_t>(next_layer_tiles_per_dim / _tile_reduction_factor);
            if (next_layer_tiles_per_dim == 0) next_layer_tiles_per_dim = 1;
            // if (_rank == 0) printf("R%d: Layer %d: tiles: %d\n", _rank, layer_num, next_layer_tiles_per_dim);
            add_layer(next_layer_tiles_per_dim, 2, layer_num++);
            
        }
        // printf("R%d: created tree of depth %d\n", _rank, _tree.size());
        // if (_rank == 0) printf("R%d: num_p: %d, reduct fac: %d, input root: %d, leaf_t: %d, root_t: %d, depth: %d\n",
        //     _rank, _num_particles, _tile_reduction_factor, _root_tiles_per_dim, leaf_tiles_per_dim, _next_layer_tiles_per_dim, depth);


    }

    /**
     * Populate a Kokkos::View that maps to the passed-in AoSoA to the rank
     * each particle should be migrated to based on its x/y/z position.
     * Maps particles according to a specific layer of the tree
     */
    template <class ViewType, class PositionSliceType>
    void mapParticles(const PositionSliceType& positions, ViewType& particle_ranks,
                      const int particle_num, const int layer)
    {
        using mem_space = typename ViewType::memory_space;
        using exec_space = typename ViewType::execution_space;

        // Get all rank domains on host
        auto tree_layer = _tree[layer];
        auto domains_host = tree_layer->get_domains();
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
     * Initialize tiles in the leaf layer based on particle locations
     */
    template <class PositionSliceType>
    void initializeLayer(int layer, PositionSliceType position_slice, std::size_t num_particles)
    {
        auto array = _tree[layer]->array();
        array->registerSparseGrid( position_slice, num_particles );
        array->reserveFromMap( 1.2 );
        // printf("R%d: array size: %d\n", _rank, (int)array->size());
    }

    template <class PositionSliceType>
    bool loadBalanceLayer(int layer, PositionSliceType position_slice, std::size_t num_particles)
    {
        return _tree[layer]->loadBalance(position_slice, num_particles);
    }


    /**
     * Assumes all particles in 'data' are owned by this rank; i.e., particles have already been
     * distributed to their correct owner rank
     * 
     * Assumes x/y/z coordinates are the first tuple element in "data"
     */
    template <class AggregationFunctor>
    void aggregateDataUp(data_aosoa_type external_data, AggregationFunctor functor)
    {
        // Data comes from externally to populate leaf layer (layer 0)
        migrateData(external_data, 0);
        _tree[0]->populateCells(external_data, functor);
        // auto data = _tree[0]->data();
        for (std::size_t i = 1; i < _tree.size(); i++)
        {
            // if (_rank == 0) printf("Starting layer %d...\n", i);
            migrateAndSetLayer(i-1, i, functor);
        }



        // Initialize mesh to 0
        // auto num_particles = data.size();
        // initializeLayer(0, position_slice, num_particles);
        // _tree[0]->populateCells(data, functor);
        // _tree[0]->printOwnedCells();
        // auto data1 = _tree[0]->data();
        // auto positions = Cabana::slice<cell_slice_id>(data1);
        // int rank = _rank;
        // for (size_t i = 0; i < data1.size(); i++)
        // {
        //     if (rank == 0) printf("R%d: x/y/z: %0.3lf, %0.3lf, %0.3lf\n", rank, positions(i, 0), positions(i, 1),
        //         positions(i, 2));
        // }
    }

    /**
     * Migrate AoSoA data to the correct rank of ownership for a given layer.
     * Use cell_slice_id slice for positions.
     */
    void migrateData(data_aosoa_type& external_data, int to_layer)
    {
        auto positions = Cabana::slice<cell_slice_id>(external_data);
        Kokkos::View<int*, memory_space> layer_owner("layer_owner", external_data.size());
        mapParticles(positions, layer_owner, external_data.size(), to_layer);
        Cabana::Distributor<MemorySpace> distributor(_comm, layer_owner);
        Cabana::migrate( distributor, external_data );
    }

    /**
     * Used to internally migrate and aggregate data from one layer to the next.
     * Use cell_slice_id slice for positions.
     */
    template <class AggregationFunctor>
    void migrateAndSetLayer(int from_layer, int to_layer, AggregationFunctor functor)
    {
        auto data = _tree[from_layer]->data();
        auto positions = Cabana::slice<cell_slice_id>(data);
        Kokkos::View<int*, memory_space> layer_owner("layer_owner", data.size());
        mapParticles(positions, layer_owner, data.size(), to_layer);
        Cabana::Distributor<MemorySpace> distributor(_comm, layer_owner);
        Cabana::migrate( distributor, data );
        _tree[to_layer]->populateCells(data, functor);
    }

    /**
     * Computes the interaction list for each cell in the tree.
     * 
     * The interaction list of cell0 is the set of all cells such that:
     *  1) cell0 and cell_other are on the same layer of the tree.
     *  2) cell0 and cell_other do not touch.
     *  3) The parent cells of cell0 and cell_other do touch.
     */
    void computeInteractionList()
    {
        // At the root layer, we assume all cells touch all other cells.
        // In other words, these cells are all in each other's neighbor
        // list, not interaction list.
        
    }
    /**
     * Computes the neighbor list for each cell in the tree.
     * 
     * The neighbor list of cell0 is the set of all cells such that:
     *  1) cell0 and cell_other are on the same layer of the tree.
     *  2) cell0 and cell_other directly border each other.
     */
    void computeNeighborList()
    {
        // At the root layer, all cells are neighbors with one another
        // Otherwise, neighbor cells are cells that are +-1 in each
        // dimension in cell_ijk locations.
    } 


    int rank() const { return _rank; }
    std::size_t numLayers() const { return _tree.size(); }
    std::array<double, 3> globalLowCorner() const { return _global_low_corner; }
    std::array<double, 3> globalHighCorner() const { return _global_high_corner; }

    /**
     * Get a layer of the tree
     */
    auto layer(int layer)
    {
        if (layer >= _tree.size())
            throw std::runtime_error("Canopy::Tree:layer: Requested layer larger than tree depth!\n");
        return _tree[layer];
    }

  private:
    std::array<double, 3> _global_high_corner;
    std::array<double, 3> _global_low_corner;
    const MPI_Comm _comm;
    int _rank, _comm_size;

    // Tree layers.
    std::vector<std::shared_ptr<TreeLayer<tree_type, cell_per_tile_dim>>> _tree;

    // How many tiles per dimension in the leaf layer.
    std::size_t _leaf_tiles_per_dim;

    // Factor for how many tiles the mesh should be reduced by for each layer
    std::size_t _tile_reduction_factor;

    // Maxmimum tiles per dimension at the root layer
    std::size_t _root_tiles_per_dim;
};

template <class ExecutionSpace, class MemorySpace, class DataTypes, class EntityType,
          std::size_t NumSpaceDim, std::size_t CellPerTileDim, std::size_t CellSliceId>
std::shared_ptr<Tree<ExecutionSpace, MemorySpace, DataTypes, EntityType,
    NumSpaceDim, CellPerTileDim, CellSliceId>>
        createTree( const std::array<double, 3>& global_low_corner,
                    const std::array<double, 3>& global_high_corner,
                    const std::size_t leaf_tiles_per_dim,
                    const std::size_t tile_reduction_factor,
                    const std::size_t root_tiles_per_dim,
                    MPI_Comm comm)
{
    return std::make_shared<Tree<ExecutionSpace, MemorySpace, DataTypes, EntityType,
        NumSpaceDim, CellPerTileDim, CellSliceId>>(global_low_corner,
            global_high_corner, leaf_tiles_per_dim, tile_reduction_factor, root_tiles_per_dim,
            comm);
}

} // end namespace Canopy

#endif // CANOPY_TREE_HPP
