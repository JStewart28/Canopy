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

template <class ExecutionSpace, class MemorySpace, class EntityType,
          std::size_t NumSpaceDim, std::size_t CellPerTileDim, std::size_t ExpansionCutoff>
class Tree
{
  public:
    using execution_space = ExecutionSpace;
    
    using memory_space = MemorySpace;

    //! Self type
    using tree_type = Tree<ExecutionSpace, MemorySpace, EntityType,
        NumSpaceDim, CellPerTileDim, ExpansionCutoff>;

    //! Memory space size type
    using size_type = typename memory_space::size_type;
    //! Array entity type (node, cell, face, edge).
    using entity_type = EntityType;
    //! Dimension number
    static constexpr std::size_t num_space_dim = NumSpaceDim;
    //! Mesh type
    using mesh_type = Cabana::Grid::SparseMesh<double, num_space_dim>;

    static constexpr std::size_t cell_per_tile_dim = CellPerTileDim;

    //! AoSoA related types
    //! MemberType Data types
    //! Cell x/y/z center
    //! cell ID
    //! Rank
    static constexpr std::size_t p = ExpansionCutoff;
    using cdouble = Kokkos::complex<double>;
    // MemberType must be trivially copyable, so we cannot use cdouble.
    // Instead, store as two doubles
    using member_types = Cabana::MemberTypes<double[(p+1)*(p+1)][2], double[3], std::size_t, int>;
    //! AoSoA Tuple type
    using tuple_type = Cabana::Tuple<member_types>;
    using data_aosoa_type = Cabana::AoSoA<member_types, memory_space, cell_per_tile_dim>;

    //! Sparse partitioner type
    using sparse_partitioner_type = Cabana::Grid::SparseDimPartitioner<memory_space, num_space_dim>;
    
    Tree( const std::array<double, 3>& global_low_corner,
            const std::array<double, 3>& global_high_corner,
            const std::size_t leaf_tiles_per_dim,
            const std::size_t tile_reduction_factor,
            MPI_Comm comm )
        : _global_low_corner( global_low_corner )
        , _global_high_corner( global_high_corner )
        , _leaf_tiles_per_dim( leaf_tiles_per_dim )
        , _tile_reduction_factor( tile_reduction_factor )
        , _root_tiles_per_dim( 1 )
        , _comm( comm )
    {
        MPI_Comm_rank( comm, &_rank );
        MPI_Comm_size( comm, &_comm_size );

        // Reserve space for 10 layers
        _tree.reserve(10);

        // Reserve space for root layer
        _M_root = Kokkos::View<cdouble*, memory_space>("_M_root", (p+1) * (p+1));

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
        // for (std::size_t i = 0; i < domains_host.size(); ++i)
        // {
        //     if (_rank == 0) printf("L%d: R%d: [%0.3lf, %0.3lf, %0.3lf] to [%0.3lf, %0.3lf, %0.3lf]\n", layer,
        //         i, domains_host[i][0], domains_host[i][1], domains_host[i][2], domains_host[i][3],
        //         domains_host[i][4], domains_host[i][5]);
        // }
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

    /*
     Set the root layer. At (root layer - 1) there one tile per dimension,
     but since there are still multiple cells per tile, there must be one
     final aggregation step to translate and add multipoles into a single
     set of coefficients at the root. Since the root layer is a single set of
     multipole coefficients that is not distributed, store the root layer data
     in this object instyead of a TreeLayer.
    */
    void initializeRootLayer()
    {
        // One rank holds all the data in the layer below the root because
        // there is only one tile per dimensions and therefore no
        // distributed partitioning.
        if(_tree.empty())
        {
            throw std::runtime_error("Canopy::Tree::initializeRootLayer: function called with an empty tree.");
        }


        // DEBUG: Set top layer to first layer
        // auto top_layer = _tree[0];
        auto top_layer = _tree.back();

        // auto domains = top_layer->get_domains();
        // for (std::size_t i = 0; i < domains.size(); ++i)
        // {
        //     if (_rank == 0) printf("R%d: [%d, %d, %d] to [%d, %d, %d]\n",
        //         i, domains[i][0], domains[i][1], domains[i][2], domains[i][3],
        //         domains[i][4], domains[i][5]);
        // }

        auto cid2ijk = top_layer->cid2ijk();
        auto map = *top_layer->map();
        auto aosoa = top_layer->array()->aosoa();
        auto cells_activated = cid2ijk.size();
        auto map_size = cid2ijk.size();
        
        printf("R%d: aosoa size: %d, map size: %d\n", _rank, aosoa.size(), map_size);

        // Save cell centers for multipole translations
        Kokkos::View<double*[3], memory_space> incoming_cell_centers("incoming_cell_centers", cells_activated);

        // Save multipole coefficients.
        std::size_t num_M = (p+1) * (p+1);
        Kokkos::View<cdouble*, memory_space> M_children("M_children", num_M * cells_activated);

        // Offset for filling M_children.
        Kokkos::View<std::size_t, memory_space> idx("idx");
        Kokkos::deep_copy(idx, 0);

        // The center of expansion at the root layer is the center of the domain.
        Kokkos::Array<double, 3> domain_center;
        for (int d = 0; d < 3; ++d)
            domain_center[d] = _global_low_corner[d] + 0.5 * (_global_high_corner[d] - _global_low_corner[d]);

        // Properties of top layer
        using top_layer_type = typename decltype(top_layer)::element_type;
        static constexpr std::size_t cell_bits_per_tile =
            top_layer_type::cell_bits_per_tile;
        static constexpr std::size_t cell_mask_per_tile =
            top_layer_type::cell_mask_per_tile;

        // Iterate over all activiated cells
        int rank = _rank;
        Kokkos::parallel_for(
        "iterate_top_layer",
        Kokkos::RangePolicy<execution_space>( 0, cid2ijk.capacity() ),
        KOKKOS_LAMBDA( const int index ) {
            // printf("R%d: checking index %d\n", rank, index);
            if ( cid2ijk.valid_at( index ) )
            {
                auto cid = cid2ijk.key_at( index );
                auto cell_ijk = cid2ijk.value_at(index);
                auto offset = Kokkos::atomic_fetch_add(&idx(), 1);
                
                // Get the data tuple from the mesh
                auto tid = map.queryTile(cell_ijk[0],
                                        cell_ijk[1],
                                        cell_ijk[2]);
                auto ctid = map.cell_local_id(cell_ijk[0],
                                        cell_ijk[1],
                                        cell_ijk[2]);       
                auto tp = aosoa.getTuple(( tid << cell_bits_per_tile ) |
                                        ( ctid & cell_mask_per_tile ) );
                
                // Save the incoming cell center.
                for (int j = 0; j < 3; ++j)
                    incoming_cell_centers(offset, j) = Cabana::get<1>(tp, j);
                
                // printf("Root: R%d: cid: %d, tid: %d, ctid: %d, tuple %d, c_ijk(%d, %d, %d)\n", rank, cid,
                //     tid, ctid,
                //     ( tid << cell_bits_per_tile ) | ( ctid & cell_mask_per_tile ),
                //     cell_ijk[0], cell_ijk[1], cell_ijk[2]);

                // Save multipole coefficients
                auto offset_M_base = offset * num_M;
                for (std::size_t j = 0; j < num_M; ++j)
                {
                    double real_part = Cabana::get<0>(tp, j, 0);
                    double imag_part = Cabana::get<0>(tp, j, 1);
                    M_children(offset_M_base + j) = cdouble(real_part, imag_part);
                    // printf("Root: R%d: cid: %d, M_notrans(%d): (%0.4lf, %0.4lf)\n",
                    //     rank, cid,
                    //     j, Cabana::get<0>(tp, j, 0), Cabana::get<0>(tp, j, 1));
                }
            }
        } );

        Kokkos::fence();

        // Copy cell centers and multipole coefficients to host
        auto incoming_cell_centers_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), incoming_cell_centers);
        auto M_children_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), M_children);

        // Create objects needed for translation of multipole coefficients.
        Canopy::Kernel::Scalar::M2M<memory_space, execution_space> m2m( p );

        // Iterate over each incoming data.
        for (std::size_t i = 0; i < cells_activated; ++i)
        {
            // Create subview of correct multipole coefficients
            auto sub_M = Kokkos::subview(M_children_h, Kokkos::make_pair(i * num_M, (i+1)*num_M));

            // Create Kokkos:Array of vector pointing from child cell center to cell center.
            Kokkos::Array<double, 3> vector_to_center;
            Kokkos::Array<double, 3> child_center = {incoming_cell_centers_h(i, 0),
                incoming_cell_centers_h(i, 1), incoming_cell_centers_h(i, 2)};
            
            for (int j = 0; j < 3; ++j)
                vector_to_center[j] = (domain_center[j] - child_center[j])*-1;

            // printf("R%d: center: %0.3lf, %0.3lf, %0.3lf, vec to center: %0.3lf, %0.3lf, %0.3lf\n", rank,
            //     domain_center[0], domain_center[1], domain_center[2],
            //     vector_to_center[0], vector_to_center[1], vector_to_center[2]);
            
            // Translate and add coefficients.
            m2m(sub_M, vector_to_center);
        }

        // Set _M_root
        Kokkos::deep_copy(_M_root, m2m.coefficients());

        // Determine which rank owns the (root layer - 1) tiles
        std::vector<std::size_t> sendbuf(_comm_size, map_size);
        std::vector<std::size_t> recvbuf(_comm_size, 0);
        MPI_Alltoall(sendbuf.data(), 1, MPI_UNSIGNED_LONG_LONG,
                    recvbuf.data(), 1, MPI_UNSIGNED_LONG_LONG,
                    _comm);

        // Now recvbuf[r] contains map_size for rank r.
        // Find the rank with a non-zero value.
        int root = -1;
        for (int r = 0; r < _comm_size; ++r)
        {
            if (recvbuf[r] != 0)
            {
                root = r;
                break;
            }
        }
        if (root == -1)
        {
            throw std::runtime_error("Canopy::Tree::initializeRootLayer: No rank has non-empty map size!");
        }

        // Now broadcast the data from the root.
        MPI_Bcast(reinterpret_cast<double*>(_M_root.data()), 2 * num_M, MPI_DOUBLE, root, _comm);
    }


    /**
     * Assumes all particles in 'data' are owned by this rank; i.e., particles have already been
     * distributed to their correct owner rank
     * 
     * Assumes x/y/z coordinates are the first tuple element in "data"
     */
    template <class ParticleAoSoA>
    void create_multipoles(ParticleAoSoA external_data)
    {
        // Data comes from externally to populate leaf layer (layer 0)
        migrateParticleData(external_data);
        if (_rank == 0) printf("Starting layer 0...\n");
        _tree[0]->populateCells(external_data);
        for (std::size_t i = 1; i < _tree.size(); i++)
        {
            if (_rank == 0) printf("Starting layer %d...\n", i);
            migrateAndSetLayer(i-1, i);
        }
        if (_rank == 0) printf("Starting root layer (%d)...\n", _tree.size());
        initializeRootLayer();
    }

    /**
     * Migrate particle data to the rank that owns them at the leaf layer.
     * Positions must be the first AoSoA slice.
     */
    template <class ParticleAoSoA>
    void migrateParticleData(ParticleAoSoA& external_data)
    {
        auto positions = Cabana::slice<0>(external_data);
        Kokkos::View<int*, memory_space> layer_owner("layer_owner", external_data.size());
        mapParticles(positions, layer_owner, external_data.size(), 0);
        Cabana::Distributor<MemorySpace> distributor(_comm, layer_owner);
        Cabana::migrate( distributor, external_data );
    }

    /**
     * Used to internally migrate and aggregate data from one layer to the next.
     * Use position_slice_id slice for positions.
     */
    void migrateAndSetLayer(int from_layer, int to_layer)
    {
        // Communicate cell data
        auto data = _tree[from_layer]->data();
        auto positions = Cabana::slice<1>(data);
        Kokkos::View<int*, memory_space> to_layer_owner("to_layer_owner", data.size());
        mapParticles(positions, to_layer_owner, data.size(), to_layer);
        Cabana::Distributor<memory_space> distributor(_comm, to_layer_owner);
        Cabana::migrate( distributor, data );
        _tree[to_layer]->populateCells(data);
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

    /**
     * Returns the number of layers with the root layer included in the
     * count, which is stored outside of the tree.
     */
    std::size_t numLayers() const { return _tree.size() + 1; }

    Kokkos::View<cdouble*, memory_space>& M_root() {return _M_root;}
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

    // Maxmimum tiles per dimension at the (root layer -1) layer
    std::size_t _root_tiles_per_dim;

    // Root data
    Kokkos::View<cdouble*, memory_space> _M_root;

};

template <class ExecutionSpace, class MemorySpace, class EntityType,
          std::size_t NumSpaceDim, std::size_t CellPerTileDim, std::size_t ExpansionCutoff>
std::shared_ptr<Tree<ExecutionSpace, MemorySpace, EntityType,
    NumSpaceDim, CellPerTileDim, ExpansionCutoff>>
        createTree( const std::array<double, 3>& global_low_corner,
                    const std::array<double, 3>& global_high_corner,
                    const std::size_t leaf_tiles_per_dim,
                    const std::size_t tile_reduction_factor,
                    MPI_Comm comm)
{
    return std::make_shared<Tree<ExecutionSpace, MemorySpace, EntityType,
        NumSpaceDim, CellPerTileDim, ExpansionCutoff>>(global_low_corner,
            global_high_corner, leaf_tiles_per_dim, tile_reduction_factor,
            comm);
}

} // end namespace Canopy

#endif // CANOPY_TREE_HPP
