#include "Canopy_Helpers.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

namespace HelpersTest
{

void testCopyScalarSlice()
{
    using DataTypes = Cabana::MemberTypes<double>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;

    const int num_particles = 137;
    AoSoA_t aosoa( "aosoa", num_particles );
    auto slice = Cabana::slice<0>( aosoa );

    Kokkos::parallel_for(
        "init_scalar",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_particles ),
        KOKKOS_LAMBDA( int i ) {
            slice( i ) = static_cast<double>( i ) * 2.5 + 1.0;
        } );
    Kokkos::fence();

    auto host_view = Canopy::create_mirror_view_and_copy(
        Kokkos::HostSpace(), slice, "scalar_mirror" );

    static_assert(
        std::is_same_v<typename decltype( host_view )::memory_space,
                       Kokkos::HostSpace>,
        "result must live in HostSpace" );

    EXPECT_EQ( static_cast<int>( host_view.extent( 0 ) ), num_particles );

    for ( int i = 0; i < num_particles; i++ )
    {
        EXPECT_DOUBLE_EQ( host_view( i ),
                          static_cast<double>( i ) * 2.5 + 1.0 );
    }
}

void testCopyVectorSlice()
{
    constexpr int num_components = 3;
    using DataTypes = Cabana::MemberTypes<double[num_components]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;

    const int num_particles = 64;
    AoSoA_t aosoa( "aosoa", num_particles );
    auto slice = Cabana::slice<0>( aosoa );

    Kokkos::parallel_for(
        "init_vector",
        Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>(
            { 0, 0 }, { num_particles, num_components } ),
        KOKKOS_LAMBDA( int i, int d ) {
            slice( i, d ) =
                static_cast<double>( i ) + 0.125 * static_cast<double>( d );
        } );
    Kokkos::fence();

    auto host_view = Canopy::create_mirror_view_and_copy(
        Kokkos::HostSpace(), slice, "vector_mirror" );

    static_assert(
        std::is_same_v<typename decltype( host_view )::memory_space,
                       Kokkos::HostSpace>,
        "result must live in HostSpace" );

    EXPECT_EQ( static_cast<int>( host_view.extent( 0 ) ), num_particles );
    EXPECT_EQ( static_cast<int>( host_view.extent( 1 ) ), num_components );

    for ( int i = 0; i < num_particles; i++ )
    {
        for ( int d = 0; d < num_components; d++ )
        {
            EXPECT_DOUBLE_EQ( host_view( i, d ),
                              static_cast<double>( i ) +
                                  0.125 * static_cast<double>( d ) );
        }
    }
}

void testEmptyScalarSlice()
{
    using DataTypes = Cabana::MemberTypes<float>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;

    AoSoA_t aosoa( "empty_aosoa", 0 );
    auto slice = Cabana::slice<0>( aosoa );

    auto host_view = Canopy::create_mirror_view_and_copy(
        Kokkos::HostSpace(), slice, "empty_mirror" );

    EXPECT_EQ( static_cast<int>( host_view.extent( 0 ) ), 0 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//

TEST( canopy_helpers, copy_scalar_slice ) { testCopyScalarSlice(); }

TEST( canopy_helpers, copy_vector_slice ) { testCopyVectorSlice(); }

TEST( canopy_helpers, copy_empty_slice ) { testEmptyScalarSlice(); }

} // namespace HelpersTest
