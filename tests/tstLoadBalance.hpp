#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

// Zoltan2 headers
#include <Zoltan2_BasicVectorAdapter.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_PartitioningSolution.hpp>

#include <gtest/gtest.h>

#include <mpi.h>
#include <random>
#include <vector>

// ---------------------------------------------------------------
// 1. Define the AoSoA layout
// ---------------------------------------------------------------
// Field tags
struct Position {};   // x, y, z
struct Velocity {};
struct Mass {};

// AoSoA type aliases
using DataTypes = Cabana::MemberTypes<double[3],   // Position
                                      double[3],   // Velocity
                                      double>;     // Mass
enum Fields : int { POS = 0, VEL = 1, MASS = 2 };

// Memory / execution space – adjust for your backend
using MemorySpace  = Kokkos::HostSpace;
using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
constexpr int VectorLength = 8;

using AoSoA_t = Cabana::AoSoA<DataTypes, MemorySpace, VectorLength>;

// ---------------------------------------------------------------
// 2. Build a Zoltan2 adapter from the AoSoA positions
// ---------------------------------------------------------------
// Zoltan2's BasicVectorAdapter wants contiguous coordinate arrays,
// so we extract x/y/z into separate views.
struct ParticleCoords
{
    std::vector<double> x, y, z;
    std::vector<Zoltan2::default_gno_t> global_ids;

    void extractFromAoSoA(const AoSoA_t& aosoa,
                          Zoltan2::default_gno_t gid_offset)
    {
        const auto n = aosoa.size();
        x.resize(n);
        y.resize(n);
        z.resize(n);
        global_ids.resize(n);

        auto pos = Cabana::slice<POS>(aosoa, "position");
        for (std::size_t i = 0; i < n; ++i)
        {
            x[i] = pos(i, 0);
            y[i] = pos(i, 1);
            z[i] = pos(i, 2);
            global_ids[i] = gid_offset + static_cast<Zoltan2::default_gno_t>(i);
        }
    }
};

// The adapter is templated on a "User" type;
// Tpetra::Map<> is the common lightweight choice.
Zoltan2::BasicVectorAdapter<Tpetra::Map<>>
buildAdapter(const ParticleCoords& coords)
{
    // const int dim = 3;
    const auto n  = static_cast<int>(coords.global_ids.size());

    // Pointers to coordinate arrays – one per dimension
    const double* coordPtrs[3] = { coords.x.data(),
                                    coords.y.data(),
                                    coords.z.data() };
    // Strides (contiguous within each array)
    const int strides[3] = {1, 1, 1};

    // Adapter takes: numIds, globalIds, coords, strides, dim
    return Zoltan2::BasicVectorAdapter<Tpetra::Map<>>(
        n,
        coords.global_ids.data(),
        coordPtrs[0], coordPtrs[1], coordPtrs[2],
        strides[0],   strides[1],   strides[2]);
}

// ---------------------------------------------------------------
// 3. Run Zoltan2 partitioning
// ---------------------------------------------------------------
void loadBalanceParticles(AoSoA_t& particles, MPI_Comm comm)
{
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    // Compute a global-ID offset so every particle has a unique GID.
    long long local_count = particles.size();
    long long gid_offset  = 0;
    MPI_Exscan(&local_count, &gid_offset, 1,
                MPI_LONG_LONG, MPI_SUM, comm);

    // Extract coordinates into contiguous arrays for the adapter.
    ParticleCoords coords;
    coords.extractFromAoSoA(particles, gid_offset);

    // Build the Zoltan2 adapter.
    auto adapter = buildAdapter(coords);

    // Set Zoltan2 parameters (RCB geometric partitioning is
    // natural for particle codes).
    Teuchos::ParameterList params;
    params.set("algorithm",           "rcb");
    params.set("num_global_parts",    nprocs);
    params.set("imbalance_tolerance",  1.05);

    // Create and solve the partitioning problem.
    Zoltan2::PartitioningProblem<Zoltan2::BasicVectorAdapter<Tpetra::Map<>>>
        problem(&adapter, &params, comm);

    problem.solve();

    // ---------------------------------------------------------------
    // 4. Migrate particles according to the solution
    // ---------------------------------------------------------------
    const auto& solution = problem.getSolution();
    const auto* partAssign = solution.getPartListView();  // new part for each local particle

    // Bin particles by destination rank.
    // partAssign[i] gives the new owning part (== MPI rank) for local particle i.
    std::vector<std::vector<std::size_t>> sendLists(nprocs);
    for (std::size_t i = 0; i < particles.size(); ++i)
        sendLists[partAssign[i]].push_back(i);

    // --- From here you would use Cabana::Distributor or your own
    //     MPI exchange to move particle data. A sketch:
    //
    //   Cabana::Distributor<MemorySpace> distributor(
    //       comm, destinationRanks );           // Kokkos::View of dest rank per particle
    //   Cabana::migrate( distributor, particles );
    //
    // Building the destinationRanks view from partAssign:
    Kokkos::View<int*, MemorySpace> dest("dest_ranks", particles.size());
    auto dest_h = Kokkos::create_mirror_view(dest);
    for (std::size_t i = 0; i < particles.size(); ++i)
        dest_h(i) = static_cast<int>(partAssign[i]);
    Kokkos::deep_copy(dest, dest_h);

    Cabana::Distributor<MemorySpace> distributor(comm, dest);
    Cabana::migrate(distributor, particles);
    // After migrate(), `particles` is resized and contains only
    // the particles now owned by this rank.
}

// ---------------------------------------------------------------
// 5. Example main
// ---------------------------------------------------------------
void run()
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Create some particles
    const std::size_t num_local = 10;
    AoSoA_t particles("particles", num_local);


    // std::random_device rd; 
    std::mt19937 gen(1000 + rank); 
    std::uniform_real_distribution<double> dis(0.0, 5.0);

    // 4. Generate the number
    double randomNum = dis(gen);

    auto pos = Cabana::slice<POS>(particles, "position");
    for (std::size_t i = 0; i < num_local; i++)
    {
        auto x = dis(gen);
        auto y = dis(gen);
        auto z = dis(gen);
        pos(i, 0) = x;
        pos(i, 1) = y;
        pos(i, 2) = z;
        printf("Before (%.2lf, %.2lf, %.2lf), R%d\n",
            pos(i, 0), pos(i, 1), pos(i, 2), rank);
    }

    loadBalanceParticles(particles, comm);

    for (std::size_t i = 0; i < particles.size(); i++)
    {
        printf("After (%.2lf, %.2lf, %.2lf), R%d\n",
            pos(i, 0), pos(i, 1), pos(i, 2), rank);
    }

    // particles now holds the load-balanced subset for this rank.
    return;
}

TEST( LoadBalance, LoadBalance1 )
{ 
    run();
}