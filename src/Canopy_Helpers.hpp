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
    using dst_memory_space    = MemorySpace;
    using src_layout          = typename SliceType::array_layout;

    // Helper lambda to perform the copy with explicit execution space
    // This handles cross-memory-space copies (e.g., CudaSpace to HostSpace)
    auto perform_copy = [&]( auto& src_view, auto& dst_view ) {
        // deep_copy(exec_space, dst, src) - destination first, then source
        Kokkos::deep_copy( src_execution_space{}, dst_view, src_view );
        Kokkos::fence();
    };

    if constexpr ( rank == 1 )
    {
        // slice.extent(0) = num_soa, not particle count; use size() instead
        const auto ext0 = static_cast<int>( slice.size() );

        // Create destination view in the target memory space with matching layout
        // Explicitly construct with target memory space and source layout
        // Use view_alloc to specify memory space, then construct with layout
        auto dst_alloc = Kokkos::view_alloc( Kokkos::WithoutInitializing, label );
        typename MirrorViewType<SliceType, MemorySpace>::type
            dst_view( dst_alloc, ext0 );

        // Create a temporary view in the source memory space with matching layout
        // to avoid temporary allocation during deep_copy
        Kokkos::View<value_type*, src_memory_space, src_layout>
            src_view( label + "_src", ext0 );

        // Copy data from slice to src_view using the source execution space
        Kokkos::parallel_for(
            "SliceToView_1D",
            Kokkos::RangePolicy<src_execution_space>( 0, ext0 ),
            KOKKOS_LAMBDA( const int i ) {
                src_view(i) = slice(i);
            } );
        Kokkos::fence();

        // Perform deep_copy with explicit execution space to handle cross-space copies
        // Views now have matching layouts, avoiding temporary allocation
        perform_copy( src_view, dst_view );

        return dst_view;
    }
    else if constexpr ( rank == 2 )
    {
        // Internal AoSoA layout is [num_soa, vector_length, num_components].
        // extent(0)=num_soa, extent(1)=vector_length, extent(2)=num_components.
        // Use size() for particle count and extent(2) for component count.
        const auto ext0 = static_cast<int>( slice.size() );
        const auto ext1 = static_cast<int>( slice.extent( 2 ) );

        // Create destination view in the target memory space with matching layout
        // Explicitly construct with target memory space and source layout
        auto dst_alloc = Kokkos::view_alloc( Kokkos::WithoutInitializing, label );
        typename MirrorViewType<SliceType, MemorySpace>::type
            dst_view( dst_alloc, ext0, ext1 );

        // Create a temporary view in the source memory space with matching layout
        Kokkos::View<value_type**, src_memory_space, src_layout>
            src_view( label + "_src", ext0, ext1 );

        // 2D parallelization - copy from slice to src_view
        Kokkos::parallel_for(
            "SliceToView_2D",
            Kokkos::MDRangePolicy<
                src_execution_space,
                Kokkos::Rank<2>
            >( {0, 0}, {ext0, ext1} ),
            KOKKOS_LAMBDA( const int i, const int d ) {
                src_view(i,d) = slice(i,d);
            } );
        Kokkos::fence();

        // Perform deep_copy with explicit execution space to handle cross-space copies
        // Views now have matching layouts, avoiding temporary allocation
        perform_copy( src_view, dst_view );

        return dst_view;
    }
    else
    {
        static_assert( rank <= 2, "Unsupported slice rank" );
    }
}

} // namespace Canopy

#endif // CANOPY_HELPERS_HPP