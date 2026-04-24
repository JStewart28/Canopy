#ifndef CANOPY_HELPERS_HPP
#define CANOPY_HELPERS_HPP

namespace Canopy {

template<class SliceType, class MemorySpace>
struct MirrorViewType;

template<class SliceType, class MemorySpace>
struct MirrorViewType
{
    using value_type = typename SliceType::value_type;
    static constexpr int rank = SliceType::rank;

    using type = std::conditional_t<
        rank == 1,
        Kokkos::View<value_type*, MemorySpace>,
        Kokkos::View<value_type**, MemorySpace>
    >;
};

template<class SliceType, class MemorySpace>
typename MirrorViewType<SliceType, MemorySpace>::type
create_mirror_view_and_copy( MemorySpace,
                             const SliceType& slice,
                             const std::string& label = "slice_mirror" )
{
    using value_type = typename SliceType::value_type;
    constexpr int rank = SliceType::rank;

    using src_memory_space    = typename SliceType::memory_space;
    using src_execution_space = typename SliceType::execution_space;

    if constexpr ( rank == 1 )
    {
        auto ext0 = slice.extent(0);

        Kokkos::View<value_type*, src_memory_space>
            src_view( label + "_src", ext0 );

        Kokkos::parallel_for(
            "SliceToView_1D",
            Kokkos::RangePolicy<src_execution_space>( 0, ext0 ),
            KOKKOS_LAMBDA( int i ) {
                src_view(i) = slice(i);
            } );
        Kokkos::fence();

        typename MirrorViewType<SliceType, MemorySpace>::type
            dst_view( label, ext0 );

        Kokkos::deep_copy( dst_view, src_view );

        return dst_view;
    }
    else if constexpr ( rank == 2 )
    {
        auto ext0 = static_cast<int>(slice.extent(0));
        auto ext1 = static_cast<int>(slice.extent(1));

        Kokkos::View<value_type**, src_memory_space>
            src_view( label + "_src", ext0, ext1 );

        // 2D parallelization
        Kokkos::parallel_for(
            "SliceToView_2D",
            Kokkos::MDRangePolicy<
                src_execution_space,
                Kokkos::Rank<2>
            >( {0, 0}, {ext0, ext1} ),
            KOKKOS_LAMBDA( int i, int d ) {
                src_view(i,d) = slice(i,d);
            } );
        Kokkos::fence();

        typename MirrorViewType<SliceType, MemorySpace>::type
            dst_view( label, ext0, ext1 );

        Kokkos::deep_copy( dst_view, src_view );

        return dst_view;
    }
    else
    {
        static_assert( rank <= 2, "Unsupported slice rank" );
    }
}

} // namespace Canopy

#endif // CANOPY_HELPERS_HPP