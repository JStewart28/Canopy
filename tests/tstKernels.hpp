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

#include <Canopy_Kernels.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//

/**
 * Fill a view with random (x, y, z) coordinates within the specified bounds,
 * where bounds is (x_min, y_min, z_min, x_max, y_max, z_max)
 */
template <class PosView>
void fillRandomCoordinates(PosView& cart_coords, Kokkos::Array<double, 6> bounds)
{
    using RandomPool = Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE>;
    RandomPool rand_pool( 12345 ); // Seed the random number generator
    Kokkos::parallel_for(
        "populate_cart_coords",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, cart_coords.extent(0) ),
        KOKKOS_LAMBDA( const int i ) {
            auto rand_gen = rand_pool.get_state();
            
            // X-coordinate
            double min_x = bounds[0];
            double max_x = bounds[3];
            cart_coords( i, 0 ) = (max_x - min_x) * rand_gen.drand() + min_x;

            // Y-coordinate
            double min_y = bounds[1];
            double max_y = bounds[4];
            cart_coords( i, 1 ) = (max_y - min_y) * rand_gen.drand() + min_y;
            
            // Z-coordinate
            double min_z = bounds[2];
            double max_z = bounds[5];
            cart_coords( i, 2 ) = (max_z - min_z) * rand_gen.drand() + min_z;
            
            rand_pool.free_state( rand_gen );
        } );
    Kokkos::fence();
}

/**
 * Fill a view with random scalar values within the specified (min, max) bound.
 */
template <class View>
void fillRandomScalar(View& q, Kokkos::Array<double, 2> bounds)
{
    using RandomPool = Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE>;
    RandomPool rand_pool( 12345 ); // Seed the random number generator
    Kokkos::parallel_for(
        "populate_cart_coords",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, q.extent(0) ),
        KOKKOS_LAMBDA( const int i ) {
            auto rand_gen = rand_pool.get_state();
            // X-coordinate
            double min = bounds[0];
            double max = bounds[1];
            q( i ) = (max - min) * rand_gen.drand() + min;
            rand_pool.free_state( rand_gen );
        } );
    Kokkos::fence();
}

/**
 * Test that scalar kernels are correctly calculated
 */
void testScalarP2MKernel()
{
    // Create points and q (scalar value)
    const int num_points = 1;
    Kokkos::View<double* [3], TEST_MEMSPACE> cart_coords( "cart_coords",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q( "q", num_points );
    
    Kokkos::Array<double, 6> coord_bounds = {-1.0, -1.0, -1.0, 1.0, 1.0, 1.0};
        Kokkos::View<double* [3], TEST_MEMSPACE> coords0( "coords0",
                                                          num_points );
    cart_coords(0, 0) = 1.0;
    cart_coords(0, 1) = 0.0;
    cart_coords(0, 2) = 0.0;
    Kokkos::Array<double, 2> charge_bounds = {-10.0, 10.0};
    q(0) = 1.0;

    // Expansion center
    Kokkos::Array<double, 3> expansion_center = { 0.0, 0.0, 0.0 };

    // Target point
    double Px = 10.0, Py = 0.0, Pz = 0.0;
    double r, theta, phi;
    Canopy::Kernel::cart2sph( Px - expansion_center[0],
                              Py - expansion_center[1],
                              Pz - expansion_center[2], r, theta, phi );

    // Direct potential
    auto cart_coords_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), cart_coords );
    auto q_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), q );
    double phi_direct = 0.0;
    double max_rho = 0.0; // for error bound
    for ( int i = 0; i < num_points; ++i )
    {
        double dx = Px - cart_coords_host( i, 0 );
        double dy = Py - cart_coords_host( i, 1 );
        double dz = Pz - cart_coords_host( i, 2 );
        double dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        phi_direct += q_host( i ) / dist;

        // distance from expansion center for error estimate
        double ddx = cart_coords_host( i, 0 ) - expansion_center[0];
        double ddy = cart_coords_host( i, 1 ) - expansion_center[1];
        double ddz = cart_coords_host( i, 2 ) - expansion_center[2];
        double rho = std::sqrt( ddx * ddx + ddy * ddy + ddz * ddz );
        max_rho = std::max( max_rho, rho );
    }

    constexpr auto pi = Kokkos::numbers::pi_v<double>;

    // Loop over truncation degree
    for ( int p = 2; p <= 2; ++p )
    {
        Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> kernel( p );

        // Particle to multipole calcuation performed in operator
        kernel( cart_coords, q, num_points, expansion_center );
        auto M = kernel.coefficients();
        auto M_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M );
        for (int i = 0; i < M_host.extent(0); i++)
        {
            printf("M-%d: (%0.4lf, %0.4lf)\n", i, M_host(i).real(), M_host(i).imag());
        }


        // Perform multipole to particle conversion to calculate potential at
        // target. Equation 3.37 in source 4
        using cdouble = Kokkos::complex<double>;
        cdouble phi_multipole = 0.0;
        for ( int n = 0; n <= p; ++n )
        {
            // double norm = Kokkos::sqrt( ( ( 2.0 * n + 1 ) / ( 4.0 * pi ) ));
            double norm = 4 * pi / double( 2 * n + 1 );
            for ( int m = -n; m <= n; ++m )
            {
                int idx = Canopy::Kernel::Scalar::index( n, m );
                phi_multipole +=
                    M_host( idx ) / Kokkos::pow( r, n + 1 ) *
                    Canopy::Kernel::Scalar::Ynm( n, m, theta, phi ) / norm;
            }
        }

        double error = std::abs( phi_multipole.real() - phi_direct );
        double bound = std::pow( max_rho / r, p + 1 ) * std::abs( phi_direct );

        // Check that the error is within 5*bound, which accounts
        // for imprecision due to imtermediate rounding.
        // EXPECT_NEAR( phi_multipole.real(), phi_direct, 5 * bound )
        std::cout    << "p=" << p << " multipole=" << phi_multipole.real()
            << " direct=" << phi_direct << " error=" << error << " bound~"
            << bound << std::endl;
    }
}

/**
 * "Base case":
 * Tests multipole-to-multipole base case where neither translation
 * nor addition of coefficients is performed.
 */
void testM2MKernel0()
{
    const int num_points = 20;

    // Domain 0
    Kokkos::View<double* [3], TEST_MEMSPACE> coords0( "coords0",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q0( "q", num_points );
    Kokkos::Array<double, 6> bounds0 = {-2.0, -2.0, -2.0, 1.0, 2.0, 1.0};
    fillRandomCoordinates(coords0, bounds0);
    Kokkos::Array<double, 2> qbounds0 = {-10.0, 10.0};
    fillRandomScalar(q0, qbounds0);
    Kokkos::Array<double, 3> center0 = { -1.0, 0.5, 0.5 };

    // Aggregated expansion center - same as domain 0 center
    Kokkos::Array<double, 3> expansion_center = { -1.0, 0.5, 0.5 };

    // Target point
    double Px = 10.0, Py = 0.0, Pz = 0.0;
    double r, theta, phi;
    Canopy::Kernel::cart2sph( Px - expansion_center[0],
                              Py - expansion_center[1],
                              Pz - expansion_center[2], r, theta, phi );

    // Direct potential
    auto coords0_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), coords0 );
    auto q0_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), q0 );
    double phi_direct = 0.0;
    double max_rho = 0.0; // for error bound
    for ( int i = 0; i < num_points; ++i )
    {
        double dx, dy, dz, dist;
        double ddx, ddy, ddz, rho;

        // Domain 0
        dx = Px - coords0_host( i, 0 );
        dy = Py - coords0_host( i, 1 );
        dz = Pz - coords0_host( i, 2 );
        dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        phi_direct += q0_host( i ) / dist;

        // distance from expansion center for error estimate
        ddx = coords0_host( i, 0 ) - expansion_center[0];
        ddy = coords0_host( i, 1 ) - expansion_center[1];
        ddz = coords0_host( i, 2 ) - expansion_center[2];
        rho = std::sqrt( ddx * ddx + ddy * ddy + ddz * ddz );
        max_rho = std::max( max_rho, rho );
    }

    constexpr auto pi = Kokkos::numbers::pi_v<double>;

    // Loop over truncation degree
    for ( int p = 1; p <= 5; ++p )
    {
        Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );

        // Compute multipoles for domain 0
        p2m( coords0, q0, num_points, center0 );
        auto M0 = p2m.coefficients();
        auto M0_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M0 );
        
        // Create M2M kernel to shift and add multipoles 
        Canopy::Kernel::Scalar::M2M<TEST_MEMSPACE, TEST_EXECSPACE> m2m( p );

        // Add M0
        m2m(M0, expansion_center, center0);

        // Get new multipole coefficients
        auto M = m2m.coefficients();
        auto M_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M );
        
        // Perform multipole to particle conversion to calculate potential at
        // target. Equation 3.36 in source 4
        using cdouble = Kokkos::complex<double>;
        cdouble phi_multipole = 0.0;
        for ( int n = 0; n <= p; ++n )
        {
            double pref = 4 * pi / double( 2 * n + 1 );
            for ( int m = -n; m <= n; ++m )
            {
                int idx = Canopy::Kernel::Scalar::index( n, m );
                phi_multipole +=
                    pref * M_host( idx ) / Kokkos::pow( r, n + 1 ) *
                    Canopy::Kernel::Scalar::Ynm( n, m, theta, phi );
            }
        }

        double error = std::abs( phi_multipole.real() - phi_direct );
        double bound = std::pow( max_rho / r, p + 1 ) * std::abs( phi_direct );

        // Check that the error is within 5*bound, which accounts
        // for imprecision due to imtermediate rounding.
        EXPECT_NEAR( phi_multipole.real(), phi_direct, 5 * bound )
            << "p=" << p << " multipole=" << phi_multipole.real()
            << " direct=" << phi_direct << " error=" << error << " bound~"
            << bound << std::endl;
    }
}

/**
 * Tests multipole-to-multipole calculations, specifically:
 *  - Translation of multipole expansions
 *  - Addition of multipole expansions
 * Creates two multipole expansions around centers with charges
 * disjunct, well-seperated domains. Translations these expansions
 * to center around a new center and then adds these expansions together.
 * Converts the aggregated multipole expansions back to potentials at
 * a target point and compares the result to the directly calculated potential
 * at the target point.
 */
void testM2MKernel1()
{
    const int num_points = 1;

    // Domain 0
    Kokkos::View<double* [3], TEST_MEMSPACE> coords0( "coords0",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q0( "q", num_points );
    // Kokkos::Array<double, 6> bounds0 = {-1.0, -1.0, -1.0, -3.0, -3.0, -3.0};
    // fillRandomCoordinates(coords0, bounds0);
    // Kokkos::Array<double, 2> qbounds0 = {-10.0, 10.0};
    // fillRandomScalar(q0, qbounds0);
    Kokkos::Array<double, 3> center0 = { 0.0, 1.0, 0.0 };
    coords0(0, 0) = 1.0;
    coords0(0, 1) = 0.0;
    coords0(0, 2) = 0.0;
    q0(0) = 1.0;

    // Aggregated expansion center
    Kokkos::Array<double, 3> expansion_center = { 0.0, -10.0, 0.0 };

    // Target point
    double Px = 10.0, Py = 0.0, Pz = 0.0;
    double r, theta, phi;
    Canopy::Kernel::cart2sph( Px - expansion_center[0],
                              Py - expansion_center[1],
                              Pz - expansion_center[2], r, theta, phi );

    // Direct potential
    auto coords0_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), coords0 );
    auto q0_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), q0 );
    double phi_direct = 0.0;
    double max_rho = 0.0; // for error bound
    for ( int i = 0; i < num_points; ++i )
    {
        double dx, dy, dz, dist;
        double ddx, ddy, ddz, rho;

        // Domain 0
        dx = Px - coords0_host( i, 0 );
        dy = Py - coords0_host( i, 1 );
        dz = Pz - coords0_host( i, 2 );
        dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        phi_direct += q0_host( i ) / dist;

        // distance from expansion center for error estimate
        ddx = coords0_host( i, 0 ) - expansion_center[0];
        ddy = coords0_host( i, 1 ) - expansion_center[1];
        ddz = coords0_host( i, 2 ) - expansion_center[2];
        rho = std::sqrt( ddx * ddx + ddy * ddy + ddz * ddz );
        max_rho = std::max( max_rho, rho );
    }

    constexpr auto pi = Kokkos::numbers::pi_v<double>;

    // Loop over truncation degree
    for ( int p = 3; p <= 3; ++p )
    {
        Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );

        // Compute multipoles for domain 0
        p2m( coords0, q0, num_points, center0 );
        auto M0 = p2m.coefficients();
        auto M0_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M0 );
        // for (int i = 0; i < M0_host.extent(0); i++)
        // {
        //     printf("M0-%d: (%0.4lf, %0.4lf)\n", i, M0_host(i).real(), M0_host(i).imag());
        // }

        // Compute multipoles directly for center
        Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m1( p );
        p2m1( coords0, q0, num_points, expansion_center );
        auto M01 = p2m1.coefficients();
        auto M01_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M01 );
        for (int i = 0; i < M01_host.extent(0); i++)
        {
            printf("Center: M0-%d: (%0.4lf, %0.4lf)\n", i, M01_host(i).real(), M01_host(i).imag());
        }
        
        // Create M2M kernel to shift and add multipoles 
        Canopy::Kernel::Scalar::M2M<TEST_MEMSPACE, TEST_EXECSPACE> m2m( p );

        // Add M0 and M1
        m2m(M0, expansion_center, center0);

        // Get new multipole coefficients
        auto M = m2m.coefficients();
        auto M_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M );
        
        for (int i = 0; i < M_host.extent(0); i++)
        {
            printf("M2M-%d: (%0.4lf, %0.4lf)\n", i, M_host(i).real(), M_host(i).imag());
        }
        
        // Perform multipole to particle conversion to calculate potential at
        // target. Equation 3.36 in source 4
        using cdouble = Kokkos::complex<double>;
        cdouble phi_multipole = 0.0;
        for ( int n = 0; n <= p; ++n )
        {
            double pref = 4 * pi / double( 2 * n + 1 );
            for ( int m = -n; m <= n; ++m )
            {
                int idx = Canopy::Kernel::Scalar::index( n, m );
                phi_multipole +=
                    pref * M_host( idx ) / Kokkos::pow( r, n + 1 ) *
                    Canopy::Kernel::Scalar::Ynm( n, m, theta, phi );
            }
        }

        double error = std::abs( phi_multipole.real() - phi_direct );
        double bound = std::pow( max_rho / r, p + 1 ) * std::abs( phi_direct );

        // Check that the error is within 10*bound, which accounts
        // for imprecision due to imtermediate rounding.
        // EXPECT_NEAR( phi_multipole.real(), phi_direct, 10 * bound )
        std::cout    << "p=" << p << " multipole=" << phi_multipole.real()
            << " direct=" << phi_direct << " error=" << error << " bound~"
            << bound << std::endl;
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( Kernel, testScalarP2MKernel ) { testScalarP2MKernel(); }

// TEST( Kernel, testM2MKernel0 ) { testM2MKernel0(); }

// TEST( Kernel, testM2MKernel1 ) { testM2MKernel1(); }

//---------------------------------------------------------------------------//

} // end namespace Test


/*
void testM2MKernel()
{
    const int num_points = 20;

    // Domain 0
    Kokkos::View<double* [3], TEST_MEMSPACE> coords0( "coords0",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q0( "q", num_points );
    Kokkos::Array<double, 6> bounds0 = {-5.0, -4.0, -4.0, -4.0, -3.0, -3.0};
    fillRandomCoordinates(coords0, bounds0);
    Kokkos::Array<double, 2> qbounds0 = {-10.0, 10.0};
    fillRandomScalar(q0, qbounds0);
    Kokkos::Array<double, 3> center0 = { -4.5, -3.6, 3.7 };


    // Domain 1
    Kokkos::View<double* [3], TEST_MEMSPACE> coords1( "coords1", num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q1( "q", num_points );
    Kokkos::Array<double, 6> bounds1 = {-1.0, 3.0, 1.0, 1.0, 5.0, 2.0};
    fillRandomCoordinates(coords1, bounds1);
    Kokkos::Array<double, 2> qbounds1 = {-10.0, 10.0};
    fillRandomScalar(q1, qbounds1);
    Kokkos::Array<double, 3> center1 = { 0.2, 4.2, 1.5 };

    // Aggregated expansion center
    Kokkos::Array<double, 3> expansion_center = { 0.1, -0.6, 0.3 };

    // Target point
    double Px = 13.6, Py = -7.1, Pz = 10.0;
    double r, theta, phi;
    Canopy::Kernel::cart2sph( Px - expansion_center[0],
                              Py - expansion_center[1],
                              Pz - expansion_center[2], r, theta, phi );

    // Direct potential
    auto coords0_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), coords0 );
    auto coords1_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), coords1 );
    auto q0_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), q0 );
    auto q1_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), q1 );
    double phi_direct = 0.0;
    double max_rho = 0.0; // for error bound
    for ( int i = 0; i < num_points; ++i )
    {
        double dx, dy, dz, dist;
        double ddx, ddy, ddz, rho;

        // Domain 0
        dx = Px - coords0_host( i, 0 );
        dy = Py - coords0_host( i, 1 );
        dz = Pz - coords0_host( i, 2 );
        dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        phi_direct += q0_host( i ) / dist;

        // distance from expansion center for error estimate
        ddx = coords0_host( i, 0 ) - expansion_center[0];
        ddy = coords0_host( i, 1 ) - expansion_center[1];
        ddz = coords0_host( i, 2 ) - expansion_center[2];
        rho = std::sqrt( ddx * ddx + ddy * ddy + ddz * ddz );
        max_rho = std::max( max_rho, rho );

         // Domain 1
        dx = Px - coords1_host( i, 0 );
        dy = Py - coords1_host( i, 1 );
        dz = Pz - coords1_host( i, 2 );
        dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        phi_direct += q1_host( i ) / dist;

        // distance from expansion center for error estimate
        ddx = coords1_host( i, 0 ) - expansion_center[0];
        ddy = coords1_host( i, 1 ) - expansion_center[1];
        ddz = coords1_host( i, 2 ) - expansion_center[2];
        rho = std::sqrt( ddx * ddx + ddy * ddy + ddz * ddz );
        max_rho = std::max( max_rho, rho );
    }

    constexpr auto pi = Kokkos::numbers::pi_v<double>;

    // Loop over truncation degree
    for ( int p = 2; p <= 2; ++p )
    {
        Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );

        // Compute multipoles for domain 0
        p2m( coords0, q0, num_points, center0 );
        auto M0 = p2m.coefficients();
        auto M0_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M0 );
        
        // Compute multipoles for domain 1
        p2m.clear();
        p2m( coords1, q1, num_points, center1 );
        auto M1 = p2m.coefficients();
        auto M1_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M1 );

        // Create M2M kernel to shift and add multipoles 
        Canopy::Kernel::Scalar::M2M<TEST_MEMSPACE, TEST_EXECSPACE> m2m( p );

        // Add M0 and M1
        m2m(M0, expansion_center, center0);
        m2m(M1, expansion_center, center1);

        // Get new multipole coefficients
        auto M = m2m.coefficients();
        auto M_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M );
        
        for (int i = 0; i < M_host.extent(0); i++)
        {
            printf("M%d: (%0.4lf, %0.4lf)\n", i, M_host(i).real(), M_host(i).imag());
        }
        

        // Perform multipole to particle conversion to calculate potential at
        // target. Equation 3.36 in source 4
        using cdouble = Kokkos::complex<double>;
        cdouble phi_multipole = 0.0;
        for ( int n = 0; n <= p; ++n )
        {
            double pref = 4 * pi / double( 2 * n + 1 );
            for ( int m = -n; m <= n; ++m )
            {
                int idx = Canopy::Kernel::Scalar::index( n, m );
                phi_multipole +=
                    pref * M_host( idx ) / Kokkos::pow( r, n + 1 ) *
                    Canopy::Kernel::Scalar::Ynm( n, m, theta, phi );
            }
        }

        double error = std::abs( phi_multipole.real() - phi_direct );
        double bound = std::pow( max_rho / r, p + 1 ) * std::abs( phi_direct );

        // Check that the error is within 10*bound, which accounts
        // for imprecision due to imtermediate rounding.
        // EXPECT_NEAR( phi_multipole.real(), phi_direct, 10 * bound )
        std::cout    << "p=" << p << " multipole=" << phi_multipole.real()
            << " direct=" << phi_direct << " error=" << error << " bound~"
            << bound << std::endl;
    }
}
*/
