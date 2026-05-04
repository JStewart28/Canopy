#ifndef CANOPY_HELPERS_HPP
#define CANOPY_HELPERS_HPP

#include <Kokkos_Core.hpp>

#include <string>
#include <type_traits>

namespace Canopy
{

template <class SliceType, class MemorySpace>
struct MirrorViewType;

template <class SliceType, class MemorySpace>
struct MirrorViewType
{
    using value_type = typename SliceType::value_type;
    static constexpr int rank = SliceType::rank;

    using type =
        std::conditional_t<rank == 1, Kokkos::View<value_type*, MemorySpace>,
                           Kokkos::View<value_type**, MemorySpace>>;
};

template <class SliceType, class MemorySpace>
typename MirrorViewType<SliceType, MemorySpace>::type
create_mirror_view_and_copy( MemorySpace, const SliceType& slice,
                             const std::string& label = "slice_mirror" )
{
    constexpr int rank = SliceType::rank;

    using src_memory_space = typename SliceType::memory_space;
    using src_execution_space = typename SliceType::execution_space;

    using dst_view_type =
        typename MirrorViewType<SliceType, MemorySpace>::type;

    if constexpr ( rank == 1 )
    {
        // slice.extent(0) = num_soa, not particle count; use size() instead.
        auto ext0 = static_cast<int>( slice.size() );

        // Allocate the destination first, in the target memory space with
        // its native layout.
        dst_view_type dst_view( label, ext0 );

        // Mirror of dst_view living in the slice's memory space. The mirror
        // inherits dst_view's layout, so the eventual cross-space deep_copy
        // is a plain memcpy with no transposition.
        auto src_mirror =
            Kokkos::create_mirror_view( src_memory_space(), dst_view );

        // Fill the mirror element-wise from the slice. This must run in the
        // slice's execution space because that's the only space that can
        // dereference the slice.
        Kokkos::parallel_for(
            "SliceToView_1D",
            Kokkos::RangePolicy<src_execution_space>( 0, ext0 ),
            KOKKOS_LAMBDA( int i ) { src_mirror( i ) = slice( i ); } );
        Kokkos::fence();

        // Same-layout, cross-space copy. No-op when the spaces coincide.
        Kokkos::deep_copy( dst_view, src_mirror );

        return dst_view;
    }
    else if constexpr ( rank == 2 )
    {
        // Internal AoSoA layout is [num_soa, vector_length, num_components].
        // Use size() for particle count and extent(2) for component count.
        auto ext0 = static_cast<int>( slice.size() );
        auto ext1 = static_cast<int>( slice.extent( 2 ) );

        dst_view_type dst_view( label, ext0, ext1 );

        auto src_mirror =
            Kokkos::create_mirror_view( src_memory_space(), dst_view );

        Kokkos::parallel_for(
            "SliceToView_2D",
            Kokkos::MDRangePolicy<src_execution_space, Kokkos::Rank<2>>(
                { 0, 0 }, { ext0, ext1 } ),
            KOKKOS_LAMBDA( int i, int d ) {
                src_mirror( i, d ) = slice( i, d );
            } );
        Kokkos::fence();

        Kokkos::deep_copy( dst_view, src_mirror );

        return dst_view;
    }
    else
    {
        static_assert( rank <= 2, "Unsupported slice rank" );
    }
}

} // namespace Canopy

#endif // CANOPY_HELPERS_HPP