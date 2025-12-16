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
    using local_view_type = Kokkos::View<cdouble*[num_coefficients], memory_space>;

    using view_type = Kokkos::View<int*, memory_space>;

    // Cabana halo type
    using cabana_halo_type = Cabana::Halo<memory_space>;
    
    Halo(local_map_type local_map, local_view_type locals, view_type export_ids,
         view_type export_ranks, MPI_Comm comm)
        : _num_tuple( locals.extent(0) )
        , _comm( comm )
    {
        // Allocate aosoa
        _data = halo_aosoa_type("_halo_data", _num_tuple);

        // Fill aosoa
        auto ijk_slice = Cabana::slice<1>(_data);
        auto coefficient_slice = Cabana::slice<0>(_data);
        Kokkos::parallel_for("fill_halo_aosoa",
        Kokkos::RangePolicy<execution_space>(0, local_map.capacity()),
        KOKKOS_LAMBDA(const int index)
        {
            if (cid2ijk.valid_at(index))
            {
                // Cell ijk
                auto cell_ijk = cid2ijk.key_at( cid2ijk_index );

                // Cell local index
                auto local_index = ijk2l.value_at(ijk2l_index);
                
                // Set cell ijk
                for (int i = 0; i < 3; i++)
                    ijk_slice(local_index, i) = cell_ijk[i];
                
                // Set local coefficients
                for (std::size_t i = 0; i < num_coefficients; i++)
                {
                    coefficient_slice(local_index, i).real() = locals(local_index, i).real();
                    coefficient_slice(local_index, i).imag() = locals(local_index, i).imag();
                }    
            }
        });

        _halo = cabana_halo_type(_comm, _num_tuple, export_ids, export_ranks);

        _data.resize(_halo.numLocal() + _halo.numGhost());
    }

    void gather()
    {
        Cabana::gather( _halo, _data );
    }

    auto data() const { return _data; }

  private:
    std::size_t _num_tuple;
    MPI_Comm _comm;
    halo_aosoa_type _data;
    cabana_halo_type _halo;
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
