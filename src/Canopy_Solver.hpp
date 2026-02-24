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

#ifndef CANOPY_TREE_HPP
#define CANOPY_TREE_HPP


#include <Canopy_SolverLayer.hpp>

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

// Value for no field given
inline constexpr std::size_t no_id = static_cast<std::size_t>(-1);

// Container for Particle input data and mapping from slice Id to the
// correct data unit
template<class AoSoAType,
         std::size_t PositionId,
         std::size_t InDataId,
         std::size_t OutDataId,
         std::size_t ForceId = no_id>
struct ParticleMetadata
{
  using aosoa_type = AoSoAType;
  static constexpr std::size_t pos = PositionId;
  static constexpr std::size_t in  = InDataId;
  static constexpr std::size_t out = OutDataId;
  static constexpr std::size_t force = ForceId;
};

template <class MemorySpace, class ExecutionSpace, class Metadata, 
          std::size_t CellPerTileDim, std::size_t ExpansionCutoff>
class Solver
{
  public:
    // Check metadata
    static_assert(Metadata::pos != no_id, "Metadata must define position index");
    static_assert(Metadata::in != no_id, "Metadata must define in_data index");
    static_assert(Metadata::out != no_id, "Metadata must define out_data index");

    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    
    //! Self type
    using solver_type = Solver<MemorySpace, ExecutionSpace, Metadata, CellPerTileDim, ExpansionCutoff>;

    //! Memory space size type
    using size_type = typename memory_space::size_type;
    //! Dimension number
    static constexpr std::size_t num_space_dim = 3;
    //! Mesh type
    using mesh_type = Cabana::Grid::SparseMesh<double, num_space_dim>;

    static constexpr std::size_t cell_per_tile_dim = CellPerTileDim;

    //! AoSoA related types
    //! MemberType Data types
    //! Cell x/y/z center
    static constexpr std::size_t p = ExpansionCutoff;
    using cdouble = Kokkos::complex<double>;
    // MemberType must be trivially copyable, so we cannot use cdouble.
    // Instead, store as two doubles
    // Multipoles are stored with cell center position, locals are stored with cell ijk position
    using multipole_member_types = Cabana::MemberTypes<double[(p+1)*(p+1)][2], double[3]>;
    using local_member_types = Cabana::MemberTypes<double[(p+1)*(p+1)][2], int[3]>;
    //! AoSoA Tuple type
    using multipole_tuple_type = Cabana::Tuple<multipole_member_types>;
    using local_tuple_type = Cabana::Tuple<local_member_types>;
    using multipole_aosoa_type = Cabana::AoSoA<multipole_member_types, memory_space, cell_per_tile_dim>;
    using local_aosoa_type = Cabana::AoSoA<local_member_types, memory_space, cell_per_tile_dim>;

    //! Particle data
    using particle_aosoa_type = Metadata::aosoa_type;
    
    Solver( const std::array<double, 3>& global_low_corner,
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
        // printf("L%d: cell_per_dim: %d\n", layer_num, cell_per_tile_dim * tiles_per_dim);
        auto layer = createSolverLayer<solver_type, cell_per_tile_dim>(
            _global_low_corner, _global_high_corner, tiles_per_dim, _tile_reduction_factor, halo_width, layer_num, _comm);
        _tree.push_back(layer);
    }

    void build()
    {
        if (_tile_reduction_factor < 2)
            throw std::runtime_error("Canopy::Solver::build: _tile_reduction_factor must be greater than 1.\n");

        int layer_num = 0;

        std::size_t next_layer_tiles_per_dim = _leaf_tiles_per_dim;
        add_layer(next_layer_tiles_per_dim, 2, layer_num++);

        // Calculate the depth of the tree
        int depth = 0;
        while (next_layer_tiles_per_dim > _root_tiles_per_dim)
        {
            depth++;
            next_layer_tiles_per_dim = static_cast<std::size_t>(next_layer_tiles_per_dim / _tile_reduction_factor);
            if (next_layer_tiles_per_dim == 0) next_layer_tiles_per_dim = 1;
            add_layer(next_layer_tiles_per_dim, 2, layer_num++);
            
        }
    }

    /**
     * Populate a Kokkos::View that maps to the passed-in AoSoA to the rank
     * each particle should be migrated to based on its x/y/z position.
     * Maps particles according to a specific layer of the tree
     */
    template <class ViewType, class PositionSliceType>
    void mapParticles(const PositionSliceType& positions, ViewType& particle_ranks,
                      const std::size_t particle_num, const int layer, const bool run_load_balance)
    {
        Kokkos::Profiling::ScopedRegion region("Canopy::Solver::mapParticles");

        using mem_space = typename ViewType::memory_space;
        using exec_space = typename ViewType::execution_space;

        // Load balance the partition if requested.
        auto tree_layer = _tree[layer];
        if (run_load_balance)
        {
            tree_layer->optimizePartition(positions, particle_num);
        }

        // Get all rank domains on host
        auto domain_bounds = tree_layer->domains();
        const int comm_size = _comm_size;

        // Flag for cell centers that may be outside of the domain.
        // This will happen if the domain does not have integer-value
        // high and low points.
        Kokkos::View<int, memory_space> is_out_of_bounds("is_out_of_bounds");
        Kokkos::deep_copy(is_out_of_bounds, 0);

        Kokkos::parallel_for(
            "Canopy::Solver::mapParticles loop",
            Kokkos::RangePolicy<exec_space>(0, particle_num),
            KOKKOS_LAMBDA(const int i) {
                const double xpos = positions(i, 0);
                const double ypos = positions(i, 1);
                const double zpos = positions(i, 2);

                // Linear search: check each rank domain
                for (int r = 0; r < comm_size; ++r)
                {
                    const double x_lo = domain_bounds(r, 0);
                    const double y_lo = domain_bounds(r, 1);
                    const double z_lo = domain_bounds(r, 2);
                    const double x_hi = domain_bounds(r, 3);
                    const double y_hi = domain_bounds(r, 4);
                    const double z_hi = domain_bounds(r, 5);

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
                Kokkos::atomic_store(&is_out_of_bounds(), 1);
            });

            int out_of_bounds;
            Kokkos::deep_copy(out_of_bounds, is_out_of_bounds);
            if (out_of_bounds)
            {
                throw std::runtime_error("Canopy::Solver:MapParticles: particle or cell center is out of bounds.");
            }
    }

    /*
     Set the root layer. At (root layer - 1) there one tile per dimension,
     but since there are still multiple cells per tile, there must be one
     final aggregation step to translate and add multipoles into a single
     set of coefficients at the root. Since the root layer is a single set of
     multipole coefficients that is not distributed, store the root layer data
     in this object instyead of a SolverLayer.
    */
    void initializeRootLayer()
    {
        Kokkos::Profiling::ScopedRegion region("Canopy::Solver::initializeRootLayer");

        // One rank holds all the data in the layer below the root because
        // there is only one tile per dimensions and therefore no
        // distributed partitioning.
        if(_tree.empty())
        {
            throw std::runtime_error("Canopy::Solver::initializeRootLayer: function called with an empty tree.");
        }

        // Initialize _M_root
        _M_root = Kokkos::View<cdouble[(p+1)*(p+1)], memory_space>("_M_root");

        // DEBUG: Set top layer to first layer
        auto top_layer = _tree.back();

        auto multipoles = top_layer->multipoles();
        std::size_t cells_activated = top_layer->numCells();
        auto multipole_coefficients_slice = Cabana::slice<0>(multipoles);
        auto cell_center_slice = Cabana::slice<1>(multipoles);
        
        // Save cell centers for multipole translations
        Kokkos::View<double*[3], memory_space> incoming_cell_centers("incoming_cell_centers", cells_activated);

        // Save multipole coefficients.
        static constexpr std::size_t num_coefficients = (p+1) * (p+1);
        Kokkos::View<cdouble*, memory_space> M_children("M_children", num_coefficients * cells_activated);

        // Offset for filling M_children.
        Kokkos::View<std::size_t, memory_space> idx("idx");
        Kokkos::deep_copy(idx, 0);

        // The center of expansion at the root layer is the center of the domain.
        Kokkos::Array<double, 3> domain_center;
        for (int d = 0; d < 3; ++d)
            domain_center[d] = _global_low_corner[d] + 0.5 * (_global_high_corner[d] - _global_low_corner[d]);

        // Iterate over all activated cells
        Kokkos::parallel_for(
        "Canopy::Solver::initializeRootLayer loop",
        Kokkos::RangePolicy<execution_space>( 0, cells_activated ),
        KOKKOS_LAMBDA( const int index ) {
            // printf("R%d: checking index %d\n", rank, index);
            
            // Save the incoming cell center.
            for (int j = 0; j < 3; ++j)
                incoming_cell_centers(index, j) = cell_center_slice(index, j);

            // Save multipole coefficients
            auto offset_M_base = index * num_coefficients;
            for (std::size_t j = 0; j < num_coefficients; ++j)
            {
                double real_part = multipole_coefficients_slice(index, j, 0);
                double imag_part = multipole_coefficients_slice(index, j, 1);
                M_children(offset_M_base + j) = cdouble(real_part, imag_part);
            }
        
        } );

        Kokkos::fence();

        // Copy cell centers and multipole coefficients to host
        auto incoming_cell_centers_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), incoming_cell_centers);
        auto M_children_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), M_children);

        // Create objects needed for translation of multipole coefficients.
        Canopy::Operator::Scalar::M2M<Kokkos::HostSpace, execution_space> m2m( p );

        // Iterate over each incoming data.
        for (std::size_t i = 0; i < cells_activated; ++i)
        {
            // Create subview of correct multipole coefficients
            auto sub_M = Kokkos::subview(M_children_h, Kokkos::make_pair(i * num_coefficients, (i+1)*num_coefficients));

            // Create Kokkos:Array of vector pointing from child cell center to cell center.
            Kokkos::Array<double, 3> vector_to_center;
            Kokkos::Array<double, 3> child_center = {incoming_cell_centers_h(i, 0),
                incoming_cell_centers_h(i, 1), incoming_cell_centers_h(i, 2)};
            
            for (int j = 0; j < 3; ++j)
                vector_to_center[j] = (domain_center[j] - child_center[j])*-1;

            // Translate and add coefficients.
            m2m(sub_M, vector_to_center);
        }

        // Set _M_root
        Kokkos::deep_copy(_M_root, m2m.coefficients());

        // Determine which rank owns the (root layer - 1) tiles
        std::vector<std::size_t> sendbuf(_comm_size, cells_activated);
        std::vector<std::size_t> recvbuf(_comm_size, 0);
        MPI_Alltoall(sendbuf.data(), 1, MPI_UNSIGNED_LONG_LONG,
                    recvbuf.data(), 1, MPI_UNSIGNED_LONG_LONG,
                    _comm);

        // Now recvbuf[r] contains cells_activated for rank r.
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
            throw std::runtime_error("Canopy::Solver::initializeRootLayer: No rank has non-empty map size!");
        }

        // Now broadcast the data from the root.
        MPI_Bcast(reinterpret_cast<double*>(_M_root.data()), 2 * num_coefficients, MPI_DOUBLE, root, _comm);
    }


    /**
     * Assumes all particles in 'data' are owned by this rank; i.e., particles have already been
     * distributed to their correct owner rank
     * 
     * Assumes x/y/z coordinates are the first tuple element in "data"
     */
    void create_multipoles(particle_aosoa_type& external_data, bool run_load_balance)
    {
        Kokkos::Profiling::ScopedRegion region("Canopy::Solver::create_multipoles");

        // Data comes from externally to populate leaf layer (layer 0)
        _leaf_particles = external_data;
        migrateParticleData(_leaf_particles, run_load_balance);

        // Set out data to 0
        auto out_data_slice = Cabana::slice<Metadata::out>(_leaf_particles);
        Cabana::deep_copy(out_data_slice, 0.0);

        // Owned particles are the number of leaf particles
        _owned_particles = _leaf_particles.size();

        _tree[0]->populateCells(_leaf_particles, 0, _leaf_particles.size());
        for (std::size_t i = 1; i < _tree.size(); i++)
        {
            migrateAndSetLayer(i-1, i, run_load_balance);
        }
        initializeRootLayer();
    }

    /**
     * Migrate particle data to the rank that owns them at the leaf layer.
     */
    void migrateParticleData(particle_aosoa_type& external_data, bool run_load_balance)
    {
        Kokkos::Profiling::ScopedRegion region("Canopy::Solver::migrateParticleData");

        auto positions = Cabana::slice<Metadata::pos>(external_data);
        Kokkos::View<int*, memory_space> layer_owner("layer_owner", external_data.size());
        mapParticles(positions, layer_owner, external_data.size(), 0, run_load_balance);
        Cabana::Distributor<MemorySpace> distributor(_comm, layer_owner);
        Cabana::migrate( distributor, external_data );
    }

    /**
     * Used to internally migrate and aggregate multipoles from one layer to the next.
     * Use position_slice_id slice for positions.
     */
    void migrateAndSetLayer(int from_layer, int to_layer, bool run_load_balance)
    {
        Kokkos::Profiling::ScopedRegion region("Canopy::Solver::migrateAndSetLayer");

        // Communicate cell data
        auto f_layer = _tree[from_layer];
        auto num_cells = f_layer->numCells();
        auto multipoles = f_layer->multipoles();
        auto positions = Cabana::slice<1>(multipoles);
        Kokkos::View<int*, memory_space> export_ranks("export_ranks", num_cells);

        // printf("From layer %d: num cells: %d\n", from_layer, num_cells);

        // All coefficients are haloed, so ids is just the index
        Kokkos::View<int*, memory_space> export_ids("ids", num_cells);
        Kokkos::parallel_for(
            "fill_export_ids",
            Kokkos::RangePolicy<execution_space>(0, export_ids.extent(0)),
            KOKKOS_LAMBDA(const int i)
            {
                export_ids(i) = i;
            }
        );

        mapParticles(positions, export_ranks, num_cells, to_layer, run_load_balance);

        // Create halo
        Cabana::Halo<memory_space> halo( _comm, num_cells, export_ids,
                                    export_ranks );

        // Resize multipole AoSoA for gather
        multipoles.resize(halo.numLocal() + halo.numGhost());

        // Gather
        Cabana::gather( halo, multipoles );

        _tree[to_layer]->populateCells(multipoles, halo.numLocal(), halo.numLocal() + halo.numGhost());
    }

    /**
     * For each layer, convert multipole coefficients to local coefficients centered
     * around each cell.
     */
    void multipole_to_local()
    {
        Kokkos::Profiling::ScopedRegion region("Canopy::Solver::multipole_to_local");

        const int starting_layer = static_cast<int>(_tree.size()) - 1;

        // Find the first valid layer
        int first_valid_layer = -1;
        std::size_t starting_cells_per_dimension;
        for (int L = starting_layer; L >= 0; --L)
        {
            auto cpd = _tree[L]->cellsPerDim();
            if (cpd >= 4)
            {
                first_valid_layer = L;
                starting_cells_per_dimension = cpd;
                break;
            }
        }

        if (first_valid_layer < 0)
        {
            printf("No valid multipole layers (need >= 4 cells per dimension)\n");
            return;
        }

        // Data structures for haloing and translating locals vertically
        local_aosoa_type halo_data("halo_data", 0);

        // Compute locals at first valid layer
        _tree[first_valid_layer]->multipole_to_local(starting_cells_per_dimension, first_valid_layer);

        for (int L = first_valid_layer - 1; L >= 0; --L)
        {
            // Get the computed locals at the layer above L (more coarse layer)
            _tree[L + 1]->sendCoarseLocals(halo_data, _tree[L]->domains());

            // Add these locals to layer L
            _tree[L]->addCoarseLocals(halo_data);

            // Compute locals at layer L
            _tree[L]->multipole_to_local(starting_cells_per_dimension, first_valid_layer);
        }
    }

    void haloParticles()
    {
        Kokkos::Profiling::ScopedRegion region("Canopy::Solver::haloParticles");

        auto leaf_cell_size = _tree[0]->cellSize();
        auto leaf_cell_per_dim = _tree[0]->cellsPerDim();
        auto cell_base = _tree[0]->cell_offsets();
        auto cell_offsets = _tree[0]->num_owned_cell();
        auto positions = Cabana::slice<Metadata::pos>(_leaf_particles);

        auto particle_ids = Cabana::slice<3>(_leaf_particles);

        const int rank = _rank;
        const int comm_size = _comm_size;

        Kokkos::Array<double, 3> low_corner = {_global_low_corner[0], _global_low_corner[1], _global_low_corner[2]};

        // For particle-to-particle calculations, we need to halo all particles within
        // two cells in each direction.
        using domain_type = Kokkos::View<int*[6], memory_space>;
        domain_type halo_domains("halo_domains", _comm_size);
        Kokkos::parallel_for("Canopy::Solver::compute halo domains",
            Kokkos::RangePolicy<execution_space>(0, _comm_size),
            KOKKOS_LAMBDA(const int r)
            {
                for (int i = 0; i < 3; i++)
                {
                    int rank_min = Kokkos::max(cell_base(r, i) - 2, 0);
                    int rank_max = Kokkos::min(cell_base(r, i) + cell_offsets(r, i) + 3, leaf_cell_per_dim);

                    halo_domains(r, i) = rank_min;
                    halo_domains(r, i+3) = rank_max;
                }
            }
        );

        // Iterate over particles. If we have a particle that falls within another ranks' halo
        // domain, we must halo it.

        // First, count the number of particles that must be haloed.
        Kokkos::View<std::size_t, memory_space> num_halos("num_halos");
        Kokkos::deep_copy(num_halos, 0);
        Kokkos::parallel_for("Canopy::Solver::count halo particles",
            Kokkos::RangePolicy<execution_space>(0, _leaf_particles.size()),
            KOKKOS_LAMBDA(const int pid)
            {
                const double x = positions(pid, 0);
                const double y = positions(pid, 1);
                const double z = positions(pid, 2);

                auto cell_ijk = position2ijk(x, y, z, low_corner, leaf_cell_size);

                std::size_t local_count = 0;

                for (std::size_t r = 0; r < comm_size; r++)
                {
                    if (r == rank)
                        continue;

                    const bool inside =
                        (cell_ijk[0] >= halo_domains(r, 0) && cell_ijk[0] < halo_domains(r, 3)) &&
                        (cell_ijk[1] >= halo_domains(r, 1) && cell_ijk[1] < halo_domains(r, 4)) &&
                        (cell_ijk[2] >= halo_domains(r, 2) && cell_ijk[2] < halo_domains(r, 5));

                    if (inside)
                        local_count++;        
                }

                if (local_count > 0)
                    Kokkos::atomic_add(&num_halos(), local_count);
            });

        std::size_t num_halos_h;
        Kokkos::deep_copy(num_halos_h, num_halos);

        // Now save which particles go to which ranks
        // XXX - optimize this to reduce atomics
        Kokkos::deep_copy(num_halos, 0);
        Cabana::AoSoA<Cabana::MemberTypes<int, int>, memory_space, 4> ids_ranks("ids_ranks", num_halos_h);
        auto id_slice = Cabana::slice<0>(ids_ranks);
        auto rank_slice = Cabana::slice<1>(ids_ranks);
        Kokkos::parallel_for("Canopy::Solver::fill halo particles",
            Kokkos::RangePolicy<execution_space>(0, _leaf_particles.size()),
            KOKKOS_LAMBDA(const int pid)
            {
                const double x = positions(pid, 0);
                const double y = positions(pid, 1);
                const double z = positions(pid, 2);
                auto cell_ijk = position2ijk(x, y, z, low_corner, leaf_cell_size);

                for (std::size_t r = 0; r < comm_size; r++)
                {
                    if (r == rank)
                        continue;

                    const bool inside =
                        (cell_ijk[0] >= halo_domains(r, 0) && cell_ijk[0] < halo_domains(r, 3)) &&
                        (cell_ijk[1] >= halo_domains(r, 1) && cell_ijk[1] < halo_domains(r, 4)) &&
                        (cell_ijk[2] >= halo_domains(r, 2) && cell_ijk[2] < halo_domains(r, 5));

                    if (inside)
                    {
                        auto index = Kokkos::atomic_fetch_add(&num_halos(), 1);
                        id_slice(index) = pid;
                        rank_slice(index) = r;
                    }   
                }
            });
        
        // Now halo the particles
        Cabana::Halo<memory_space> halo( _comm, _leaf_particles.size(), id_slice,
                                    rank_slice );
        std::size_t num_local = halo.numLocal();
        _leaf_particles.resize(halo.numLocal() + halo.numGhost());
        Cabana::gather(halo, _leaf_particles);

        // Save owned and ghost information
        _owned_particles = halo.numLocal();
        _ghost_particles = halo.numGhost();
    }

    template<class PosSlice, class PotSlice, class LocalsSlice, class ForceSlice, class IJK2Index, class CellSize, class LowCorner>
    struct ComputeWithLocals
    {
        PosSlice particle_positions;
        PotSlice particle_potentials;
        LocalsSlice locals_slice;
        ForceSlice particle_force;   // Dummy type when HasForce=false
        IJK2Index ijk2index;
        CellSize cell_size;
        LowCorner low_corner;
        int p;

        KOKKOS_INLINE_FUNCTION
        void operator()(const int tpi) const
        {
            // Get the cell this point falls into
            Kokkos::Array<std::size_t, 3> target_cell_ijk;
            for (int dim = 0; dim < 3; ++dim)
            {
                target_cell_ijk[dim] = static_cast<std::size_t>(
                    Kokkos::floor((particle_positions(tpi, dim) - low_corner[dim]) / cell_size[dim]) );
            }

            // Only continue if this cell exists in the mesh.
            // It always should.
            auto cell_exists = ijk2index.exists(target_cell_ijk);
            if (!cell_exists)
                return;

            // Center of local expansion is the cell center
            Kokkos::Array<double, 3> l_center;
            for (int i = 0; i < 3; i++)
                l_center[i] = low_corner[i] + (static_cast<double>(target_cell_ijk[i]) + 0.5) * cell_size[i];

            // Convert target point to spherical coordinates relative to local center
            double r, theta, phi;
            Canopy::Operator::cart2sph( particle_positions(tpi, 0) - l_center[0],
                                    particle_positions(tpi, 1) - l_center[1],
                                    particle_positions(tpi, 2) - l_center[2],
                                    r, theta, phi );

            auto ijk2l_index = ijk2index.find(target_cell_ijk);
            auto local_index = ijk2index.value_at(ijk2l_index);

            // Calculate potential using locals, and forces if enabled
            cdouble potential_accumulator(0.0, 0.0);
            Kokkos::Array<cdouble, 3> force_accumulator = {cdouble(0.0, 0.0), cdouble(0.0, 0.0), cdouble(0.0, 0.0)};
            for ( int j = 0; j <= p; ++j )
            {
                for ( int k = -j; k <= j; ++k )
                {
                    int idx = Operator::Scalar::index( j, k );

                    /* Target point 1 calculations */
                    // Greengard eq. 3.59
                    cdouble val = cdouble(locals_slice(local_index, idx, 0), locals_slice(local_index, idx, 1));

                    // Potential accumulator
                    potential_accumulator +=
                        val * Kokkos::pow( r, j ) *
                            Operator::Scalar::Ynm( j, k, theta, phi );

                    // Force accumulator
                    if constexpr (Metadata::force != no_id)
                    {
                        auto force_part = Operator::Scalar::partial2gradient(r, theta, phi, j, k, val);
                        for (int d = 0; d < 3; d++)
                            force_accumulator[d] += force_part[d];
                    }
                }
            }
            particle_potentials(tpi) += potential_accumulator.real();
            if constexpr (Metadata::force != no_id)
                for (int d = 0; d < 3; d++)
                    particle_force(tpi, d) += force_accumulator[d].real();
        }
    };

    /**
     * Take the locals from the leaf layer and convert them into potentials at each particle.
     */
    void computeL2P()
    {
        Kokkos::Profiling::ScopedRegion region("Canopy::Solver::computeL2P");

        // int rank = _rank;

        auto particle_positions = Cabana::slice<Metadata::pos>(_leaf_particles);
        auto particle_potentials = Cabana::slice<Metadata::out>(_leaf_particles);
        auto cell_size = _tree[0]->cellSize();
        auto cells_per_dim = _tree[0]->cellsPerDim();
        Kokkos::Array<double, 3> low_corner = {_global_low_corner[0], _global_low_corner[1], _global_low_corner[2]};

        auto ijk2index = _tree[0]->cellijk2i();

        auto locals_slice = Cabana::slice<0>(_tree[0]->locals());

        if constexpr (Metadata::force != no_id)
        {
            auto particle_force = Cabana::slice<Metadata::force>(_leaf_particles);

            Kokkos::parallel_for(
                "Canopy::Solver::populate_local_potential",
                Kokkos::RangePolicy<TEST_EXECSPACE>(0, _leaf_particles.size()),
                ComputeWithLocals{particle_positions, particle_potentials, locals_slice, particle_force,
                    ijk2index, cell_size, low_corner, p});
        }
        else
        {
            // Use a dummy force slice object
            struct NoForceSlice {
                KOKKOS_INLINE_FUNCTION void operator()(int,int) const {}
            } no_force;

            Kokkos::parallel_for(
                "Canopy::Solver::populate_local_potential",
                Kokkos::RangePolicy<TEST_EXECSPACE>(0, _leaf_particles.size()),
                ComputeWithLocals{particle_positions, particle_potentials, locals_slice, no_force,
                    ijk2index, cell_size, low_corner, p});
        }
    }

    void computeP2P()
    {
        Kokkos::Profiling::ScopedRegion region("Canopy::Solver::computeP2P");

        haloParticles();

        auto positions = Cabana::slice<Metadata::pos>(_leaf_particles);
        auto scalars = Cabana::slice<Metadata::in>(_leaf_particles);
        auto potentials = Cabana::slice<Metadata::out>(_leaf_particles);

        auto cell_size = _tree[0]->cellSize();
        auto cells_per_dim = _tree[0]->cellsPerDim();
        Kokkos::Array<double, 3> low_corner = {_global_low_corner[0], _global_low_corner[1], _global_low_corner[2]};

        auto total_particles = _leaf_particles.size();
        auto owned_particles = _owned_particles;
        
        // Find neighbor particles that are within 3 cells width of each other. 
        // We need to use 3 cell width to ensure that for a particle in any given cell,
        // all particles within cells up to 2 cells away are considered.
        const double neighborhood_radius = Kokkos::max(Kokkos::max(cell_size[0], cell_size[1]), cell_size[2]) * 3 * Kokkos::sqrt(3.0);
        auto neighbor_list = Cabana::Experimental::makeNeighborList(
            Cabana::FullNeighborTag{}, positions, 0, total_particles,
            neighborhood_radius );

        using list_type = decltype(neighbor_list);

        Kokkos::parallel_for("Canopy::Solver::compute_P2P loop", Kokkos::RangePolicy<execution_space>(0, owned_particles), 
            KOKKOS_LAMBDA(int my_id) {

            const double xi = positions(my_id,0);
            const double yi = positions(my_id,1);
            const double zi = positions(my_id,2);

            auto ijk_i = position2ijk(xi, yi, zi, low_corner, cell_size);

            // Cell bounds for this particle
            Kokkos::Array<int,3> lower, upper;
            for (int d = 0; d < 3; ++d) {
                lower[d] = Kokkos::max(int(ijk_i[d]) - 2, 0);
                upper[d] = Kokkos::min(int(ijk_i[d]) + 3, cells_per_dim);
            }

            int num_neighbors = Cabana::NeighborList<list_type>::numNeighbor(neighbor_list, my_id);
            double phi = 0.0;
            for (int j = 0; j < num_neighbors; j++) {
                int neighbor_id = Cabana::NeighborList<list_type>::getNeighbor(neighbor_list, my_id, j);

                const double xn = positions(neighbor_id,0);
                const double yn = positions(neighbor_id,1);
                const double zn = positions(neighbor_id,2);

                auto ijk_n = position2ijk(xn, yn, zn, low_corner, cell_size);

                // Check that our neighbor particle's cell is within 2 cells
                if (ijk_n[0] < lower[0] || ijk_n[0] >= upper[0] ||
                    ijk_n[1] < lower[1] || ijk_n[1] >= upper[1] ||
                    ijk_n[2] < lower[2] || ijk_n[2] >= upper[2])
                        continue;
                
                const double dx = xi - xn;
                const double dy = yi - yn;
                const double dz = zi - zn;
                const double r  = sqrt(dx*dx + dy*dy + dz*dz);

                phi += scalars(neighbor_id) / r;
            }

            potentials(my_id) += phi;
        });
        Kokkos::fence();
    }

    /**
     * Perform the fast multipole method.
     */

    void solve(particle_aosoa_type aosoa, bool run_load_balance)
    {
        create_multipoles(aosoa, run_load_balance);
        multipole_to_local();
        computeL2P();
        computeP2P();
    }

    int rank() const { return _rank; }

    /**
     * Returns the number of layers with the root layer included in the
     * count, which is stored outside of the tree.
     */
    std::size_t numLayers() const { return _tree.size() + 1; }

    auto M_root() {return _M_root;}
    auto particles() {return _leaf_particles;}
    auto numOwnedParticles() {return _owned_particles;}
    auto numGhostParticles() {return _ghost_particles;}
    std::array<double, 3> globalLowCorner() const { return _global_low_corner; }
    std::array<double, 3> globalHighCorner() const { return _global_high_corner; }

    /**
     * Get a layer of the tree
     */
    auto layer(int layer)
    {
        if (layer >= _tree.size())
            throw std::runtime_error("Canopy::Solver:layer: Requested layer larger than tree depth!\n");
        return _tree[layer];
    }

  private:
    std::array<double, 3> _global_high_corner;
    std::array<double, 3> _global_low_corner;
    const MPI_Comm _comm;
    int _rank, _comm_size;

    // Solver layers.
    std::vector<std::shared_ptr<SolverLayer<solver_type, cell_per_tile_dim>>> _tree;

    // How many tiles per dimension in the leaf layer.
    std::size_t _leaf_tiles_per_dim;

    // Factor for how many tiles the mesh should be reduced by for each layer
    std::size_t _tile_reduction_factor;

    // Maxmimum tiles per dimension at the (root layer -1) layer
    std::size_t _root_tiles_per_dim;

    // Root data
    Kokkos::View<cdouble[(p+1)*(p+1)], memory_space> _M_root;

    // Leaf particles
    particle_aosoa_type _leaf_particles;
    std::size_t _owned_particles = 0;
    std::size_t _ghost_particles = 0;
};

template <class MemorySpace, class ExecutionSpace, class Metadata, 
          std::size_t CellPerTileDim, std::size_t ExpansionCutoff>
std::shared_ptr<Solver<MemorySpace, ExecutionSpace, Metadata, CellPerTileDim, ExpansionCutoff>>
        createSolver( const std::array<double, 3>& global_low_corner,
                    const std::array<double, 3>& global_high_corner,
                    const std::size_t leaf_tiles_per_dim,
                    const std::size_t tile_reduction_factor,
                    MPI_Comm comm)
{
    return std::make_shared<Solver<MemorySpace, ExecutionSpace, Metadata, CellPerTileDim, ExpansionCutoff>>(global_low_corner,
            global_high_corner, leaf_tiles_per_dim, tile_reduction_factor,
            comm);
}

} // end namespace Canopy

#endif // CANOPY_TREE_HPP
