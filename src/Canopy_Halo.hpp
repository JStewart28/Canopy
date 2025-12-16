#ifndef CANOPY_HALO_HPP
#define CANOPY_HALO_HPP

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <memory>

#include <mpi.h>

#include <limits>

/**
 * Helper class for horizontal and vertical halos
 */

namespace Canopy
{

template <class ExecutionSpace, class MemorySpace, std::size_t p>
class Halo
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;

    static constexpr std::size_t num_coefficients = (p + 1) * (p + 1);

    static constexpr std::size_t vector_size = 4;

    // Locals, cell ijk index
    using halo_tuple_type = Cabana::MemberTypes<double[num_coefficients][2], int[3]>;
    using halo_aosoa_type = Cabana::AoSoA<halo_tuple_type, memory_space, vector_size>;

    // Local map
    using local_map_type = Kokkos::UnorderedMap<Kokkos::Array<std::size_t, 3>, std::size_t, memory_space>;
    // Local coefficients
    using cdouble = Kokkos::complex<double>;
    using local_view_type = Kokkos::View<cdouble*[num_coefficients], memory_space>
    
    Halo(local_map_type local_map, local_view_type locals, MPI_Comm comm)
    {
        // Allocate aosoa
        _data = halo_aosoa_type("_halo_data", locals.extent(0));
    }

  private:
    halo_aosoa_type _data;
    Kokkos::
};

template <class ExecutionSpace, class MemorySpace, std::size_t p>
std::shared_ptr<Halo<ExecutionSpace, MemorySpace, p>>
        createHalo( const local_map_type& map,
                    const local_view_type& view,
                    MPI_Comm comm)
{
    return std::make_shared<Halo<ExecutionSpace, MemorySpace, p>>(map, view,
            comm);
}

} // end namespace Canopy

#endif // CANOPY_HALO_HPP
