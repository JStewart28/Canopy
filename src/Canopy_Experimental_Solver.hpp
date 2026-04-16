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

#ifndef CANOPY_SOLVER_HPP
#define CANOPY_SOLVER_HPP


#include <ArborX.hpp>
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

namespace Experimental
{

// https://repositorio.unesp.br/server/api/core/bitstreams/0e824479-3128-41f7-8cd2-462e9a242c42/content

// ============================================================================
// Morton key type and helpers
// ============================================================================

// 64-bit Morton key. Convention:
//   root key = 1
//   children of key k = 8*k + 0 .. 8*k + 7
//   parent of key k   = k / 8   (integer division, k > 1)
//   depth of key k    = floor(log8(k))
//
// This supports trees up to depth 20 (8^20 fits in 63 bits with the
// leading-1 convention), which is far more than any practical FMM needs.
using MortonKey = uint64_t;

static constexpr MortonKey ROOT_KEY = 1;
static constexpr int MAX_CHILDREN = 8;

KOKKOS_INLINE_FUNCTION
MortonKey parent_key( MortonKey k )
{
    return k >> 3; // equivalent to k / 8
}

KOKKOS_INLINE_FUNCTION
MortonKey child_key( MortonKey k, int octant )
{
    return ( k << 3 ) | static_cast<MortonKey>( octant );
}

KOKKOS_INLINE_FUNCTION
int key_depth( MortonKey k )
{
    int d = 0;
    while ( k > 1 )
    {
        k >>= 3;
        ++d;
    }
    return d;
}

KOKKOS_INLINE_FUNCTION
int key_octant( MortonKey k )
{
    return static_cast<int>( k & 7 ); // last 3 bits
}

// ============================================================================
// Cell data stored on host (global tree topology)
// ============================================================================
struct CellInfo
{
    MortonKey key;
    int depth;
    double center[3];
    double half_width; // half the side length of this cell's cube
    int global_count;  // total particles across all ranks
    bool is_leaf;
};

// ============================================================================
// Axis-aligned bounding box
// ============================================================================
struct BoundingBox
{
    double min[3];
    double max[3];
};

// Value for no field given
inline constexpr std::size_t no_id = static_cast<std::size_t>(-1);

// Container for Particle input data and mapping from slice Id to the
// correct data unit
template<class AoSoAType, class Scalar,
         std::size_t PositionId,
         std::size_t InDataId,
         std::size_t OutDataId,
         std::size_t ForceId = no_id>
struct ParticleMetadata
{   
  using aosoa_type = AoSoAType;
  using scalar_type = Scalar;
  static constexpr std::size_t pos = PositionId;
  static constexpr std::size_t in  = InDataId;
  static constexpr std::size_t out = OutDataId;
  static constexpr std::size_t force = ForceId;
};

template <class MemorySpace, class ExecutionSpace, class Metadata, int p>
class Solver
{
  public:
    // Check metadata
    static_assert(Metadata::pos != no_id, "metadata must define position index");
    static_assert(Metadata::in != no_id, "metadata must define in_data index");
    static_assert(Metadata::out != no_id, "metadata must define out_data index");

    using metadata = Metadata;

    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    
    //! Self type
    using solver_type = Solver<MemorySpace, ExecutionSpace, Metadata, P>;

    //! Dimension number
    static constexpr int num_space_dim = 3;
    //! P-term for expansions
    static constexpr int p = P;
    //! Memory space size type
    using size_type = typename memory_space::size_type;
    //! Scalar type
    using scalar_type = typename metadata::scalar_type;
    
    // Host mirror types for tree data
    using host_execution_space = Kokkos::DefaultHostExecutionSpace;
    using host_memory_space = typename host_execution_space::memory_space;

    // The particle-to-leaf mapping lives on device for fast particle kernels
    using key_view_type = Kokkos::View<MortonKey*, memory_space>;
    using key_host_view_type =
        Kokkos::View<MortonKey*, host_memory_space>;

  private:
    // MPI
    MPI_Comm _comm;
    int _rank;
    int _comm_size;

    //! Min number of particles per cell
    int _ncrit;
    //! Maximum tree depth
    int _max_depth;

    // Global tree topology (identical on every rank)
    std::vector<CellInfo> _cells;

    // Particle -> leaf key mapping (device view)
    key_view_type _particle_keys;

    // Global bounding box
    BoundingBox _root_box;

  public:
    // Constructor
    Solver( const int ncrit, const int max_depth, MPI_Comm comm ) 
        : _ncrit( ncrit )
        , _max_depth( max_depth )
        , _comm( comm )
    {
        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );
    }

    // Compute global bounding box from distributed particles
    template <class PositionSlice>
    BoundingBox compute_global_bounding_box( PositionSlice positions,
                                    int num_local_particles )
    {
        // Local bounding box via Kokkos parallel_reduce
        double local_min[3], local_max[3];

        // Initialize with extreme values
        double inf = std::numeric_limits<double>::max();

        Kokkos::parallel_reduce(
            "ComputeLocalBBox",
            Kokkos::RangePolicy<execution_space>( 0, num_local_particles ),
            KOKKOS_LAMBDA( int i, double& lmin_x, double& lmin_y, double& lmin_z,
                        double& lmax_x, double& lmax_y, double& lmax_z ) {
                double px = positions( i, 0 );
                double py = positions( i, 1 );
                double pz = positions( i, 2 );
                if ( px < lmin_x ) lmin_x = px;
                if ( py < lmin_y ) lmin_y = py;
                if ( pz < lmin_z ) lmin_z = pz;
                if ( px > lmax_x ) lmax_x = px;
                if ( py > lmax_y ) lmax_y = py;
                if ( pz > lmax_z ) lmax_z = pz;
            },
            Kokkos::Min<double>( local_min[0] ),
            Kokkos::Min<double>( local_min[1] ),
            Kokkos::Min<double>( local_min[2] ),
            Kokkos::Max<double>( local_max[0] ),
            Kokkos::Max<double>( local_max[1] ),
            Kokkos::Max<double>( local_max[2] ) );

        // Global bounding box via MPI
        BoundingBox box;
        MPI_Allreduce( local_min, box.min, 3, MPI_DOUBLE, MPI_MIN, comm_ );
        MPI_Allreduce( local_max, box.max, 3, MPI_DOUBLE, MPI_MAX, comm_ );

        // Pad by a small epsilon so no particle sits exactly on the boundary
        double pad = 1.0e-10;
        for ( int d = 0; d < 3; ++d )
        {
            double width = box.max[d] - box.min[d];
            if ( width < pad )
                width = pad; // degenerate case
            box.min[d] -= pad * width;
            box.max[d] += pad * width;
        }

        return box;
    }

    // -----------------------------------------------------------------------
    // build()
    //
    // Main entry point. Takes particle positions (a Cabana slice) and the
    // number of local particles. Returns after all ranks agree on the tree
    // topology and each rank's particles are tagged with their leaf cell key.
    //
    // After calling build():
    //   - cells()           returns the global tree (vector of CellInfo)
    //   - particle_keys()   returns a device view mapping particle index
    //                       to the Morton key of its enclosing leaf cell
    //   - root_box()        returns the global bounding box
    // -----------------------------------------------------------------------
    template <class PositionSlice>
    void build( PositionSlice positions, int num_local_particles )
    {

    }

};

    

template <class MemorySpace, class ExecutionSpace, class Metadata, 
          std::size_t CellPerTileDim, std::size_t ExpansionCutoff>
std::shared_ptr<Solver<MemorySpace, ExecutionSpace, Metadata, CellPerTileDim, ExpansionCutoff>>
        createSolver( const std::array<typename Metadata::scalar_type, 3>& global_low_corner,
                    const std::array<typename Metadata::scalar_type, 3>& global_high_corner,
                    const std::size_t leaf_tiles_per_dim,
                    const std::size_t tile_reduction_factor,
                    MPI_Comm comm)
{
    return std::make_shared<Solver<MemorySpace, ExecutionSpace, Metadata, CellPerTileDim, ExpansionCutoff>>(global_low_corner,
            global_high_corner, leaf_tiles_per_dim, tile_reduction_factor,
            comm);
}

} // end namespace Experimental

} // end namespace Canopy

#endif // CANOPY_SOLVER_HPP
