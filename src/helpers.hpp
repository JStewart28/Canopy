#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <Utils.hpp>

// generate a random partition, to mimic a random simulation status
std::array<std::vector<int>, 3>
generate_random_partition( std::array<int, 3> ranks_per_dim,
                           int size_tile_per_dim )
{
    std::array<std::set<int>, 3> gt_partition_set;
    std::array<std::vector<int>, 3> gt_partition;
    int world_rank, world_size;
    MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );
    for ( int d = 0; d < 3; ++d )
    {
        gt_partition[d].resize( ranks_per_dim[d] + 1 );
    }

    if ( world_rank == 0 )
    {
        for ( int d = 0; d < 3; ++d )
        {
            gt_partition_set[d].insert( 0 );
            while ( static_cast<int>( gt_partition_set[d].size() ) <
                    ranks_per_dim[d] )
            {
                int rand_num = std::rand() % size_tile_per_dim;
                gt_partition_set[d].insert( rand_num );
            }
            gt_partition_set[d].insert( size_tile_per_dim );
            int i = 0;
            for ( auto it = gt_partition_set[d].begin();
                  it != gt_partition_set[d].end(); ++it )
            {
                gt_partition[d][i++] = *it;
            }
        }
    }

    // broadcast the ground truth partition to all ranks
    for ( int d = 0; d < 3; ++d )
    {
        MPI_Barrier( MPI_COMM_WORLD );
        MPI_Bcast( gt_partition[d].data(), gt_partition[d].size(), MPI_INT, 0,
                   MPI_COMM_WORLD );
        MPI_Barrier( MPI_COMM_WORLD );
    }

    return gt_partition;
}

// generate a static partition
std::array<std::vector<int>, 3>
generate_static_partition( std::array<int, 3> ranks_per_dim,
                           int size_tile_per_dim )
{
    std::array<std::set<int>, 3> gt_partition_set;
    std::array<std::vector<int>, 3> gt_partition;
    int world_rank, world_size;
    MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );
    for ( int d = 0; d < 3; ++d )
    {
        gt_partition[d].resize( ranks_per_dim[d] + 1 );
    }

    if ( world_rank == 0 )
    {
        for ( int d = 0; d < 3; ++d )
        {
            gt_partition_set[d].insert( 0 );
            while ( static_cast<int>( gt_partition_set[d].size() ) <
                    ranks_per_dim[d] )
            {
                // int rand_num = std::rand() % size_tile_per_dim;
                int rand_num = 1;
                gt_partition_set[d].insert( rand_num );
            }
            gt_partition_set[d].insert( size_tile_per_dim );
            int i = 0;
            for ( auto it = gt_partition_set[d].begin();
                  it != gt_partition_set[d].end(); ++it )
            {
                gt_partition[d][i++] = *it;
            }
        }
    }

    // broadcast the ground truth partition to all ranks
    for ( int d = 0; d < 3; ++d )
    {
        MPI_Barrier( MPI_COMM_WORLD );
        MPI_Bcast( gt_partition[d].data(), gt_partition[d].size(), MPI_INT, 0,
                   MPI_COMM_WORLD );
        MPI_Barrier( MPI_COMM_WORLD );
    }

    return gt_partition;
}

// convert std::set to device-side view
template <typename MemorySpace, typename ExecutionSpace, typename T>
auto set2view( const std::set<std::array<T, 3>>& in_set )
    -> Kokkos::View<T* [3], MemorySpace>
{
    // set => view (host)
    typedef typename ExecutionSpace::array_layout layout;
    Kokkos::View<T* [3], layout, Kokkos::HostSpace> host_view( "view_host",
                                                               in_set.size() );
    int i = 0;
    for ( auto it = in_set.begin(); it != in_set.end(); ++it )
    {
        for ( int d = 0; d < 3; ++d )
            host_view( i, d ) = ( *it )[d];
        ++i;
    }

    // create tiles view on device
    Kokkos::View<T* [3], MemorySpace> dev_view =
        Kokkos::create_mirror_view_and_copy( MemorySpace(), host_view );
    return dev_view;
}

// return random generated particles and occupied tile numbers (last two params)
template <typename T>
void generate_random_particles( const int particle_number,
                                const std::array<int, 3>& part_start,
                                const std::array<int, 3>& part_end,
                                const int cell_per_tile_dim,
                                const std::array<T, 3> global_low_corner,
                                const T cell_size,
                                std::set<std::array<int, 3>>& tile_set,
                                std::set<std::array<T, 3>>& par_pos_set )
{
    // range of particle positions
    T start[3], size[3];
    for ( int d = 0; d < 3; ++d )
    {
        // because each particle will activate three around tiles, we apply
        // 1.01 cell_size offset compared to the real partition to ensure
        // all the activated tiles sit inside the valid partition range
        start[d] = global_low_corner[d] +
                   cell_size * ( 2.01f + cell_per_tile_dim * (T)part_start[d] );
        size[d] =
            cell_size *
            ( cell_per_tile_dim * (T)( part_end[d] - part_start[d] ) - 4.02f );
    }

    // insert random particles to the set
    while ( static_cast<int>( par_pos_set.size() ) < particle_number )
    {
        T rand_offset[3];
        for ( int d = 0; d < 3; ++d )
            rand_offset[d] = (T)std::rand() / (T)RAND_MAX * size[d];
        std::array<T, 3> new_pos = { start[0] + rand_offset[0],
                                     start[1] + rand_offset[1],
                                     start[2] + rand_offset[2] };
        par_pos_set.insert( new_pos );

        std::array<int, 3> grid_base;
        for ( int d = 0; d < 3; ++d )
        {
            grid_base[d] =
                int( std::lround( ( new_pos[d] - global_low_corner[d] ) /
                                  cell_size ) ) -
                1;
        }

        for ( int i = 0; i <= 2; i++ )
            for ( int j = 0; j <= 2; j++ )
                for ( int k = 0; k <= 2; k++ )
                {
                    tile_set.insert( {
                        ( grid_base[0] + i ) / cell_per_tile_dim,
                        ( grid_base[1] + j ) / cell_per_tile_dim,
                        ( grid_base[2] + k ) / cell_per_tile_dim,
                    } );
                }
    }
}

// return statically generated particles and occupied tile numbers (last two params)
template <typename T>
void generate_static_particles( const int particle_number,
                                const std::array<int, 3>& part_start,
                                const std::array<int, 3>& part_end,
                                const int cell_per_tile_dim,
                                const std::array<T, 3> global_low_corner,
                                const T cell_size,
                                std::set<std::array<int, 3>>& tile_set,
                                std::set<std::array<T, 3>>& par_pos_set )
{
    // range of particle positions
    T start[3], size[3];
    for ( int d = 0; d < 3; ++d )
    {
        // because each particle will activate three around tiles, we apply
        // 1.01 cell_size offset compared to the real partition to ensure
        // all the activated tiles sit inside the valid partition range
        start[d] = global_low_corner[d] +
                   cell_size * ( 2.01f + cell_per_tile_dim * (T)part_start[d] );
        size[d] =
            cell_size *
            ( cell_per_tile_dim * (T)( part_end[d] - part_start[d] ) - 4.02f );
    }

    // insert random particles to the set
    while ( static_cast<int>( par_pos_set.size() ) < 1 )
    {
        T offset[3];
        for ( int d = 0; d < 3; ++d )
        {
            // rand_offset[d] = (T)std::rand() / (T)RAND_MAX * size[d];
            offset[d] = (T) (size[d]);  
        }
            
        // std::array<T, 3> new_pos = { start[0] + offset[0],
        //                              start[1] + offset[1],
        //                              start[2] + offset[2] };
        std::array<T, 3> new_pos = { 0,
                                     0,
                                     0 };
        par_pos_set.insert( new_pos );

        std::array<int, 3> grid_base;
        for ( int d = 0; d < 3; ++d )
        {
            grid_base[d] =
                int( std::lround( ( new_pos[d] - global_low_corner[d] ) /
                                  cell_size ) ) -
                1;
        }

        for ( int i = 0; i <= 2; i++ )
            for ( int j = 0; j <= 2; j++ )
                for ( int k = 0; k <= 2; k++ )
                {
                    tile_set.insert( {
                        ( grid_base[0] + i ) / cell_per_tile_dim,
                        ( grid_base[1] + j ) / cell_per_tile_dim,
                        ( grid_base[2] + k ) / cell_per_tile_dim,
                    } );
                }
    }
}

// generate a random tile sequence
int current = 0;
int uniqueNumber() { return current++; }

Kokkos::View<int* [3], Kokkos::HostSpace>
generateRandomTileSequence( int tiles_per_dim )
{
    Kokkos::View<int* [3], Kokkos::HostSpace> tiles_host(
        "random_tile_sequence_host",
        tiles_per_dim * tiles_per_dim * tiles_per_dim );

    std::vector<int> random_seq( tiles_per_dim );
    std::generate_n( random_seq.data(), tiles_per_dim, uniqueNumber );
    for ( int d = 0; d < 3; ++d )
    {
        std::shuffle( random_seq.begin(), random_seq.end(),
                      std::default_random_engine( 3439203991 ) );
        for ( int n = 0; n < tiles_per_dim; ++n )
        {
            tiles_host( n, d ) = random_seq[n];
        }
    }
    return tiles_host;
}

Kokkos::View<int* [3], Kokkos::HostSpace>
generateBiasedTileSequence(int tiles_per_dim, double bias_fraction = 0.3)
{
    Kokkos::View<int* [3], Kokkos::HostSpace> tiles_host(
        "biased_tile_sequence_host",
        tiles_per_dim * tiles_per_dim * tiles_per_dim );

    // Create a biased region size in each dimension
    int biased_dim = static_cast<int>(tiles_per_dim * bias_fraction);
    if (biased_dim < 1) biased_dim = 1;

    // Generate tiles: put 90% in the biased region, 10% in the rest
    int total_tiles = tiles_per_dim * tiles_per_dim * tiles_per_dim;
    int biased_tiles = static_cast<int>(0.99 * total_tiles);
    // int remaining_tiles = total_tiles - biased_tiles;

    int idx = 0;

    // 1. Fill from the biased region (e.g., lower corner of the domain)
    for (int x = 0; x < biased_dim && idx < biased_tiles; ++x)
        for (int y = 0; y < biased_dim && idx < biased_tiles; ++y)
            for (int z = 0; z < biased_dim && idx < biased_tiles; ++z)
            {
                tiles_host(idx, 0) = x;
                tiles_host(idx, 1) = y;
                tiles_host(idx, 2) = z;
                ++idx;
            }

    // 2. Fill the rest uniformly across the domain
    for (int x = 0; x < tiles_per_dim && idx < total_tiles; ++x)
        for (int y = 0; y < tiles_per_dim && idx < total_tiles; ++y)
            for (int z = 0; z < tiles_per_dim && idx < total_tiles; ++z)
            {
                if (x < biased_dim && y < biased_dim && z < biased_dim)
                    continue; // skip already inserted
                tiles_host(idx, 0) = x;
                tiles_host(idx, 1) = y;
                tiles_host(idx, 2) = z;
                ++idx;
            }

    // Optional: shuffle if desired
    std::vector<int> order(total_tiles);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), std::default_random_engine(3439203991));

    // Apply shuffle to the tiles
    Kokkos::View<int* [3], Kokkos::HostSpace> shuffled("shuffled_tiles", total_tiles);
    for (int i = 0; i < total_tiles; ++i)
    {
        shuffled(i, 0) = tiles_host(order[i], 0);
        shuffled(i, 1) = tiles_host(order[i], 1);
        shuffled(i, 2) = tiles_host(order[i], 2);
    }

    return shuffled;
}

// generate average partitioner
std::array<std::vector<int>, 3> computeAveragePartition(
    const int tile_per_dim, const std::array<int, 3>& ranks_per_dim )
{
    std::array<std::vector<int>, 3> rec_partitions;
    for ( int d = 0; d < 3; ++d )
    {
        int ele = tile_per_dim / ranks_per_dim[d];
        int part = 0;
        for ( int i = 0; i < ranks_per_dim[d]; ++i )
        {
            rec_partitions[d].push_back( part );
            part += ele;
        }
        rec_partitions[d].push_back( tile_per_dim );
    }
    return rec_partitions;
}

template <class l2g_type, class View>
void printView(l2g_type local_L2G, int rank, View z, int option, int DEBUG_X, int DEBUG_Y)
{
    
    int dims = z.extent(2);

    std::array<long, 2> rmin, rmax;
    for (int d = 0; d < 2; d++) {
        rmin[d] = local_L2G.local_own_min[d];
        rmax[d] = local_L2G.local_own_max[d];
    }
    Cabana::Grid::IndexSpace<2> remote_space(rmin, rmax);

    Kokkos::parallel_for("print views",
        Cabana::Grid::createExecutionPolicy(remote_space, Kokkos::DefaultHostExecutionSpace()),
        KOKKOS_LAMBDA(int i, int j) {
        
        int local_li[2] = {i, j};
        int local_gi[2] = {0, 0};   // global i, j
        local_L2G(local_li, local_gi);
        if (option == 1){
            if (dims == 3) {
                printf("R%d %d %d %d %d %.12lf %.12lf %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1), z(i, j, 2));
            }
            else if (dims == 2) {
                printf("R%d %d %d %d %d %.12lf %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1));
            }
        }
        else if (option == 2) {
            if (local_gi[0] == DEBUG_X && local_gi[1] == DEBUG_Y) {
                if (dims == 3) {
                    printf("R%d: %d: %d: %d: %d: %.12lf: %.12lf: %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1), z(i, j, 2));
                }   
                else if (dims == 2) {
                    printf("R%d: %d: %d: %d: %d: %.12lf: %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1));
                }
            }
        }
    });
} 

#endif // HELPERS_HPP
