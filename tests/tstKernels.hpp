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
    const int num_points = 20;
    Kokkos::View<double* [3], TEST_MEMSPACE> cart_coords( "cart_coords",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q( "q", num_points );
    
    Kokkos::Array<double, 6> coord_bounds = {-1.0, -1.0, -1.0, 1.0, 1.0, 1.0};
    fillRandomCoordinates(cart_coords, coord_bounds);

    Kokkos::Array<double, 2> charge_bounds = {-10.0, 10.0};
    fillRandomScalar(q, charge_bounds);

    // Expansion center
    Kokkos::Array<double, 3> expansion_center = { 0.1, -0.6, 0.3 };

    // Target point
    double Px = 6.6, Py = -5.1, Pz = 1.9;
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

    // constexpr auto pi = Kokkos::numbers::pi_v<double>;

    // Loop over truncation degree
    for ( int p = 2; p <= 5; ++p )
    {
        Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> kernel( p );

        // Particle to multipole calculation performed in operator
        kernel( cart_coords, q, num_points, expansion_center );
        auto M = kernel.coefficients();
        auto M_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M );

        // Perform multipole to particle conversion to calculate potential at
        // target. Equation 3.36 in source 4
        using cdouble = Kokkos::complex<double>;
        cdouble phi_multipole = 0.0;
        for ( int n = 0; n <= p; ++n )
        {
            // double norm = 4 * pi / double( 2 * n + 1 );
            // auto norm = Kokkos::sqrt(( ( 2.0 * n + 1 ) / ( 4.0 * pi ) ));
            for ( int m = -n; m <= n; ++m )
            {
                int idx = Canopy::Kernel::Scalar::index( n, m );
                phi_multipole +=
                    M_host( idx ) / Kokkos::pow( r, n + 1 ) *
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
 * "Base case":
 * Tests multipole-to-multipole base case where neither translation
 * nor addition of coefficients is performed.
 */
void testM2MKernel0()
{
    using cdouble = Kokkos::complex<double>;

    const int num_points = 20;

    // Domain 0
    Kokkos::View<double* [3], TEST_MEMSPACE> coords0( "coords0",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q0( "q", num_points );
    Kokkos::Array<double, 6> bounds0 = {-1.0, -1.0, -1.0, 1.0, 1.0, 1.0};
    fillRandomCoordinates(coords0, bounds0);
    Kokkos::Array<double, 2> qbounds0 = {-10.0, 10.0};
    fillRandomScalar(q0, qbounds0);
    // { -1.0, 0.5, 0.5 };
    // { 0.0, 0.0, 0.0 };
    // coords0(0, 0) = 1.0;
    // coords0(0, 1) = 0.0;
    // coords0(0, 2) = 0.0;
    // q0(0) = 1.0;

    // Center of Q coefficients
    Kokkos::Array<double, 3> q_center = { -0.1, 0.3, 0.2 };

    // Expansion center in polar coordinates - rho, alpha, beta
    double rho, alpha, beta;
    Canopy::Kernel::cart2sph( q_center[0], q_center[1], q_center[2], rho, alpha, beta );

    // Target point - rho, theta, phi
    double Px = 10.0, Py = 0.0, Pz = 0.0;
    double r, theta, phi;
    Canopy::Kernel::cart2sph( Px, Py, Pz, r, theta, phi );
    
    // (Target point - q_center) - r_p, theta_p, phi_p
    double r_p, theta_p, phi_p;
    Canopy::Kernel::cart2sph( Px - q_center[0],
                              Py - q_center[1],
                              Pz - q_center[2], r_p, theta_p, phi_p );

    // Direct potential at target point
    auto coords0_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), coords0 );
    auto q0_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), q0 );
    double potential_direct = 0.0;
    double a = 0.0;
    double total_q = 0.0;
    for ( int i = 0; i < num_points; ++i )
    {
        double dx, dy, dz, dist;
        double ddx, ddy, ddz, rho_tmp;

        // Q points
        dx = Px - coords0_host( i, 0 );
        dy = Py - coords0_host( i, 1 );
        dz = Pz - coords0_host( i, 2 );
        dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        potential_direct += q0_host( i ) / dist;

        // Add total charge for error bound
        total_q += q0_host(i);

        // Get max distance from center for error estimate.
        ddx = coords0_host( i, 0 ) - q_center[0];
        ddy = coords0_host( i, 1 ) - q_center[1];
        ddz = coords0_host( i, 2 ) - q_center[2];
        rho_tmp = std::sqrt( ddx * ddx + ddy * ddy + ddz * ddz );
        a = std::max( a, rho_tmp );
    }

    // P should be far enough away from the expansion center
    EXPECT_GT(r, (a + rho)) << "Point P is not far enough away from expansion center";

    // constexpr auto pi = Kokkos::numbers::pi_v<double>;

    // Loop over truncation degree
    for ( int p = 1; p <= 5; ++p )
    {
        Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );

        // Compute multipoles at center Q
        p2m( coords0, q0, num_points, q_center );
        auto tmp =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), p2m.coefficients() );
        Kokkos::View<cdouble*,Kokkos::HostSpace> O0_host("O0_host", tmp.extent(0));
        Kokkos::deep_copy(O0_host, tmp);
        
        // Compute potential at P using O
        cdouble potential_Q = 0.0;
        for ( int n = 0; n <= p; ++n )
        {
            // double norm = 4 * pi / double( 2 * n + 1 );
            // auto norm = Kokkos::sqrt(( ( 2.0 * n + 1 ) / ( 4.0 * pi ) ));
            for ( int m = -n; m <= n; ++m )
            {
                int idx = Canopy::Kernel::Scalar::index( n, m );
                potential_Q +=
                    O0_host( idx ) / Kokkos::pow( r_p, n + 1 ) *
                    Canopy::Kernel::Scalar::Ynm( n, m, theta_p, phi_p );
            }
        }

        // Compute M_kj coefficients
        Canopy::Kernel::Scalar::M2M<TEST_MEMSPACE, TEST_EXECSPACE> m2m( p );

        // Add M0 - use coefficients centered around Q.
        m2m(p2m.coefficients(), q_center);

        // Get new multipole coefficients
        tmp = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), m2m.coefficients() );
        Kokkos::View<cdouble*,Kokkos::HostSpace> M_host("M_host", tmp.extent(0));
        Kokkos::deep_copy(M_host, tmp);

        // Compute potential at P using M
        cdouble potential_M = 0.0;
        for ( int j = 0; j <= p; ++j )
        {
            // double norm = 4 * pi / double( 2 * n + 1 );
            // auto norm = Kokkos::sqrt(( ( 2.0 * n + 1 ) / ( 4.0 * pi ) ));
            for ( int k = -j; k <= j; ++k )
            {
                int idx = Canopy::Kernel::Scalar::index( j, k );
                potential_M +=
                    M_host( idx ) / Kokkos::pow( r, j + 1 ) *
                    Canopy::Kernel::Scalar::Ynm( j, k, theta, phi );
            }
        }

        // Check the error bounds from eq. 3.58
        auto bound = Kokkos::abs((total_q / (r - (a + rho))) * Kokkos::pow((a + rho)/r, p + 1));
        auto error = Kokkos::abs(potential_direct - potential_M);
        EXPECT_LE(error, bound) << "p="
            << p << ": error between shifted and direct potentials too high.";

        // Confirm the difference between the potential calcuated with the shifted coefficients
        // is within the error bound of the potential calculated with the original coefficients
        // Slightly increase the error bound due to intermediate rounding errors.
        EXPECT_NEAR(potential_M.real(), potential_Q.real(), 2*error) << "p="
            << p << ": error between shifted and un-shfted potential calculations too high.";
    }
}

/**
 * Tests translation of multipole coefficients
 *  - Translation of multipole expansions
 *  - Addition of multipole expansions
 * Creates a multipole expansions around one center and translates
 * it to another center. Tests against the exact calculation 
 * for potential at the translated center.
 */
void testM2MKernel1()
{
    const int num_points = 20;

    // Domain 0
    Kokkos::View<double* [3], TEST_MEMSPACE> coords0( "coords0",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q0( "q", num_points );
    Kokkos::Array<double, 6> bounds0 = {-1.0, -1.0, -1.0, -3.0, -3.0, -3.0};
    fillRandomCoordinates(coords0, bounds0);
    Kokkos::Array<double, 2> qbounds0 = {-10.0, 10.0};
    fillRandomScalar(q0, qbounds0);
    Kokkos::Array<double, 3> center0 = { 0.1, -0.4, 0.2 };
    // coords0(0, 0) = 1.0;
    // coords0(0, 1) = 0.0;
    // coords0(0, 2) = 0.0;
    // q0(0) = 1.0;

    // Aggregated expansion center
    Kokkos::Array<double, 3> expansion_center = { 1.0, 1.1, 0.4 };

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

    // constexpr auto pi = Kokkos::numbers::pi_v<double>;

    // Loop over truncation degree
    for ( int p = 1; p <= 1; ++p )
    {
        // Calculate the multipole coefficients driectly at the expansion center to debug
        // Known to be correct
        Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m_center( p );
        p2m_center( coords0, q0, num_points, expansion_center );
        auto O_center = p2m_center.coefficients();
        auto O_center_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), O_center );

        Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );

        // Compute multipoles for domain 0
        p2m( coords0, q0, num_points, center0 );
        auto O = p2m.coefficients();
        auto O_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), O );
        // for (int i = 0; i < M0_host.extent(0); i++)
        // {
        //     printf("M0-%d: (%0.4lf, %0.4lf)\n", i, M0_host(i).real(), M0_host(i).imag());
        // }
        
        // Create M2M kernel to shift and add multipoles 
        Canopy::Kernel::Scalar::M2M<TEST_MEMSPACE, TEST_EXECSPACE> m2m( p );

        // Translate coefficients
        m2m(O, center0);

        // Get new multipole coefficients
        auto M = m2m.coefficients();
        auto M_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M );
        
        // Compare translated coefficients to coefficients created directly around 
        // the expansion center
        for (std::size_t i = 0; i < M_host.extent(0); ++i)
        {
            EXPECT_DOUBLE_EQ(M_host(i).real(), O_center_host(i).real()) << "at i = " << i << std::endl;
            EXPECT_DOUBLE_EQ(M_host(i).imag(), O_center_host(i).imag()) << "at i = " << i << std::endl;
        }

        for (int i = 0; i < M_host.extent(0); i++)
        {
            printf("M-%d: (%0.4lf, %0.4lf), O_c-%d: (%0.4lf, %0.4lf), O-%d: (%0.4lf, %0.4lf)\n", i, M_host(i).real(),
                M_host(i).imag(), i, O_center_host(i).real(), O_center_host(i).imag(), i, O_host(i).real(), O_host(i).imag());
        }

        
        // for (int i = 0; i < M_host.extent(0); i++)
        // {
        //     printf("M2M-%d: (%0.4lf, %0.4lf)\n", i, M_host(i).real(), M_host(i).imag());
        // }
        
        // Perform multipole to particle conversion to calculate potential at
        // target. Equation 3.36 in source 4
        using cdouble = Kokkos::complex<double>;
        cdouble phi_multipole = 0.0;
        for ( int n = 0; n <= p; ++n )
        {
            // double norm = 4 * pi / double( 2 * n + 1 );
            // auto norm = Kokkos::sqrt(( ( 2.0 * n + 1 ) / ( 4.0 * pi ) ));
            for ( int m = -n; m <= n; ++m )
            {
                int idx = Canopy::Kernel::Scalar::index( n, m );
                phi_multipole +=
                    M_host( idx ) / Kokkos::pow( r, n + 1 ) *
                    Canopy::Kernel::Scalar::Ynm( n, m, theta, phi );
            }
        }

        double error = std::abs( phi_multipole.real() - phi_direct );
        double bound = std::pow( max_rho / r, p + 1 ) * std::abs( phi_direct );

        // Check that the error is within 10*bound, which accounts
        // for imprecision due to imtermediate rounding.
        EXPECT_NEAR( phi_multipole.real(), phi_direct, 10 * bound )
            << "p=" << p << " multipole=" << phi_multipole.real()
            << " direct=" << phi_direct << " error=" << error << " bound~"
            << bound << std::endl;
    }
}

/**
 * Tests addition and translation of multipole coefficients.
 * Creates two multipole expansions around centers with charges
 * disjunct, well-seperated domains. Translations these expansions
 * to center around a new center and then adds these expansions together.
 * Converts the aggregated multipole expansions back to potentials at
 * a target point and compares the result to the directly calculated potential
 * at the target point.
 */
void testM2MKernel2()
{
    using cdouble = Kokkos::complex<double>;

    const int points_per_section = 20;

    // 
    Kokkos::View<double* [3], TEST_MEMSPACE> coords( "coords",
                                                          points_per_section * 2 );
    Kokkos::View<double*, TEST_MEMSPACE> q( "q", points_per_section * 2);
    
    auto c0 = Kokkos::subview(coords, Kokkos::make_pair(0, points_per_section), Kokkos::ALL);
    auto c1 = Kokkos::subview(coords, Kokkos::make_pair(points_per_section, points_per_section * 2), Kokkos::ALL);
    auto q0 = Kokkos::subview(q, Kokkos::make_pair(0, points_per_section));
    auto q1 = Kokkos::subview(q, Kokkos::make_pair(points_per_section, points_per_section * 2));
    
    Kokkos::Array<double, 6> cbounds0 = {-3.0, -3.0, -3.0, -2.0, -2.0, -2.0};
    Kokkos::Array<double, 6> cbounds1 = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0};
    // Kokkos::Array<double, 3> center0 = { -2.5, -2.2, -2.7 };
    // Kokkos::Array<double, 3> center1 = { 1.4, 1.2, 1.8 };
    Kokkos::Array<double, 3> center0 = { 0.1, 0.0, 0.0 };
    Kokkos::Array<double, 3> center1 = { 0.0, 0.0, 0.0 };
    Kokkos::Array<double, 2> qbounds = {-10.0, 10.0};

    fillRandomCoordinates(c0, cbounds0);
    fillRandomCoordinates(c1, cbounds1);
    fillRandomScalar(q, qbounds);
    // coords0(0, 0) = 1.0;
    // coords0(0, 1) = 0.0;
    // coords0(0, 2) = 0.0;
    // q0(0) = 1.0;

    // Aggregated expansion center
    Kokkos::Array<double, 3> expansion_center = { 0.0, 0.0, 0.0 };

    // Target point
    double Px = 10.0, Py = 0.0, Pz = 0.0;
    double r, theta, phi;
    Canopy::Kernel::cart2sph( Px - expansion_center[0],
                              Py - expansion_center[1],
                              Pz - expansion_center[2], r, theta, phi );

    // Direct potential
    auto coords_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), coords );
    auto q_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), q );
    double phi_direct = 0.0;
    double max_rho = 0.0; // for error bound
    for ( int i = 0; i < points_per_section*2; ++i )
    {
        double dx, dy, dz, dist;
        double ddx, ddy, ddz, rho;

        // Domain 0
        dx = Px - coords_host( i, 0 );
        dy = Py - coords_host( i, 1 );
        dz = Pz - coords_host( i, 2 );
        dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        phi_direct += q_host( i ) / dist;

        // distance from expansion center for error estimate
        ddx = coords_host( i, 0 ) - expansion_center[0];
        ddy = coords_host( i, 1 ) - expansion_center[1];
        ddz = coords_host( i, 2 ) - expansion_center[2];
        rho = std::sqrt( ddx * ddx + ddy * ddy + ddz * ddz );
        max_rho = std::max( max_rho, rho );
    }

    // constexpr auto pi = Kokkos::numbers::pi_v<double>;

    // Loop over truncation degree
    for ( int p = 2; p <= 2; ++p )
    {
        // Calculate the multipole coefficients driectly at the expansion center to debug
        // Known to be correct
        Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m_center( p );
        p2m_center( coords, q, points_per_section * 2, expansion_center );
        auto O_all_center_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), p2m_center.coefficients() );
        
        Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m_center0( p );
        p2m_center0( c0, q0, points_per_section, expansion_center );
        auto O_center0_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), p2m_center0.coefficients() );
        
        Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m_center1( p );
        p2m_center1( c1, q1, points_per_section, expansion_center );
        auto O_center1_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), p2m_center1.coefficients() );

        // Create P2M kernel to create multipoles
        Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );

        // Create M2M kernel to shift and add multipoles 
        Canopy::Kernel::Scalar::M2M<TEST_MEMSPACE, TEST_EXECSPACE> m2m( p );

        // Compute multipoles for domain 0
        p2m( c0, q0, points_per_section, center0 );

        // Add multipole to M2M
        m2m(p2m.coefficients(), center0);
        for (int i = 0; i < m2m.coefficients().extent(0); i++)
        {
            printf("M0-%d: (%0.4lf, %0.4lf), O0-%d: (%0.4lf, %0.4lf)\n", i,  m2m.coefficients()(i).real(),  m2m.coefficients()(i).imag(),
                i, O_center0_host(i).real(), O_center0_host(i).imag());
        }

        auto tmp =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), p2m.coefficients() );
        Kokkos::View<cdouble*,Kokkos::HostSpace> O0_host("O0_host", tmp.extent(0));
        Kokkos::deep_copy(O0_host, tmp);


        // Compute multipoles for domain 1
        // Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m1( p );
        p2m.clear();
        p2m( c1, q1, points_per_section, center1 );

        // Add multipole to M2M
        m2m(p2m.coefficients(), center1);
        tmp = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), p2m.coefficients() );
        Kokkos::View<cdouble*,Kokkos::HostSpace> O1_host("O1_host", tmp.extent(0));
        Kokkos::deep_copy(O1_host, tmp);
        // for (int i = 0; i < M0_host.extent(0); i++)
        // {
        //     printf("M0-%d: (%0.4lf, %0.4lf)\n", i, M0_host(i).real(), M0_host(i).imag());
        // }

        // Get new multipole coefficients
        auto M = m2m.coefficients();
        // for (int i = 0; i < m2m.coefficients().extent(0); i++)
        // {
        //     printf("M1-%d: (%0.4lf, %0.4lf), O1-%d: (%0.4lf, %0.4lf)\n", i,  m2m.coefficients()(i).real(),  m2m.coefficients()(i).imag(),
        //         i, O_center1_host(i).real(), O_center1_host(i).imag());
        // }
        auto M_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M );
        // for (int i = 0; i < M_host.extent(0); i++)
        // {
        //     printf("M-%d: (%0.4lf, %0.4lf), O-c-%d: (%0.4lf, %0.4lf), O0-%d: (%0.4lf, %0.4lf), O1-%d: (%0.4lf, %0.4lf)\n",
        //         i, M_host(i).real(), M_host(i).imag(),
        //         i, O_all_center_host(i).real(), O_all_center_host(i).imag(),
        //         i, O0_host(i).real(), O0_host(i).imag(),
        //         i, O1_host(i).real(), O1_host(i).imag()
        //         );
        // }

        
        // for (int i = 0; i < M_host.extent(0); i++)
        // {
        //     printf("M2M-%d: (%0.4lf, %0.4lf)\n", i, M_host(i).real(), M_host(i).imag());
        // }
        
        // Perform multipole to particle conversion to calculate potential at
        // target. Equation 3.36 in source 4
        using cdouble = Kokkos::complex<double>;
        cdouble phi_multipole = 0.0;
        for ( int n = 0; n <= p; ++n )
        {
            // double norm = 4 * pi / double( 2 * n + 1 );
            // auto norm = Kokkos::sqrt(( ( 2.0 * n + 1 ) / ( 4.0 * pi ) ));
            for ( int m = -n; m <= n; ++m )
            {
                int idx = Canopy::Kernel::Scalar::index( n, m );
                phi_multipole +=
                    M_host( idx ) / Kokkos::pow( r, n + 1 ) *
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

void testM2LKernel0()
{
    // Create points and q (scalar value)
    const int num_points = 20;
    Kokkos::View<double* [3], TEST_MEMSPACE> cart_coords( "cart_coords",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q( "q", num_points );
    
    Kokkos::Array<double, 6> coord_bounds = {-4.0, -4.0, -4.0, -3.0, -3.0, -3.0};
    fillRandomCoordinates(cart_coords, coord_bounds);

    Kokkos::Array<double, 2> charge_bounds = {-10.0, 10.0};
    fillRandomScalar(q, charge_bounds);

    // Expansion center
    Kokkos::Array<double, 3> multipole_center = { -3.1, -3.6, -3.7 };
    Kokkos::Array<double, 3> local_center = { 0.1, -0.6, 0.3 };

    // Target point near local center
    double Px = 0.2, Py = -0.7, Pz = 0.4;
    double r, theta, phi;
    Canopy::Kernel::cart2sph( Px - local_center[0],
                              Py - local_center[1],
                              Pz - local_center[2], r, theta, phi );
    
    // radius of smallest sphere that encloses all sources
    auto cart_coords_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cart_coords);
    double a = 0.0;
    for (int i = 0; i < num_points; ++i) {
        double dx = cart_coords_host(i,0) - multipole_center[0];
        double dy = cart_coords_host(i,1) - multipole_center[1];
        double dz = cart_coords_host(i,2) - multipole_center[2];
        double dist_src = std::sqrt(dx*dx + dy*dy + dz*dz);
        a = std::max(a, dist_src);                     // <-- sphere radius a
    }

    // compute rho = distance between multipole_center and local_center
    double ddx = multipole_center[0] - local_center[0];
    double ddy = multipole_center[1] - local_center[1];
    double ddz = multipole_center[2] - local_center[2];
    double rho = std::sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
    
    double c = rho / a;   // theorem requires c > 1
    EXPECT_GT(c, 1.0) << "Error: c is not greater than 1.0" << std::endl;

    // Total charge
     auto q_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q);
    double q_sum = 0.0;
    for (int i = 0; i < num_points; ++i)
        q_sum += std::abs(q_host(i));
    
   

    // Direct potential
    // auto cart_coords_host =
    //     Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), cart_coords );
    // auto q_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), q );
    double phi_direct = 0.0;
    for ( int i = 0; i < num_points; ++i )
    {
        double dx = Px - cart_coords_host( i, 0 );
        double dy = Py - cart_coords_host( i, 1 );
        double dz = Pz - cart_coords_host( i, 2 );
        double dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        phi_direct += q_host( i ) / dist;
    }

    // constexpr auto pi = Kokkos::numbers::pi_v<double>;

    // Loop over truncation degree
    for ( int p = 2; p <= 2; ++p )
    {
        double bound = ( q_sum / ( c * a - a ) ) * std::pow( 1.0 / c, p + 1 );

        Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );
        p2m( cart_coords, q, num_points, multipole_center );
        auto M = p2m.coefficients();
        auto M_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M );

        // Convert multipoles to locals
        Canopy::Kernel::Scalar::M2L<TEST_MEMSPACE, TEST_EXECSPACE> m2l( p );
        m2l( M, local_center );
        auto L_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), m2l.coefficients() );

        // Perform local to potential conversion to calculate potential at
        // target. Equation 3.59 in source 4
        using cdouble = Kokkos::complex<double>;
        cdouble phi_multipole = 0.0;
        for ( int j = 0; j <= p; ++j )
        {
            // double norm = 4 * pi / double( 2 * n + 1 );
            // auto norm = Kokkos::sqrt(( ( 2.0 * n + 1 ) / ( 4.0 * pi ) ));
            for ( int k = -j; k <= j; ++k )
            {
                int idx = Canopy::Kernel::Scalar::index( j, k );
                phi_multipole +=
                    L_host( idx ) * Kokkos::pow( r, j ) *
                    Canopy::Kernel::Scalar::Ynm( j, k, theta, phi );
            }
        }

        double error = std::abs( phi_multipole.real() - phi_direct );

        // Check that the error is within 5*bound, which accounts
        // for imprecision due to imtermediate rounding.
        // EXPECT_NEAR( phi_multipole.real(), phi_direct, 5 * bound )
        std::cout    << "p=" << p << " multipole=" << phi_multipole.real()
            << " direct=" << phi_direct << " error=" << error << " bound~"
            << bound << std::endl;
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( Kernel, testScalarP2MKernel ) { testScalarP2MKernel(); }

TEST( Kernel, testM2MKernel0 ) { testM2MKernel0(); }

// TEST( Kernel, testM2MKernel1 ) { testM2MKernel1(); }

// TEST( Kernel, testM2MKernel2 ) { testM2MKernel2(); }

// TEST( Kernel, testM2LKernel0 ) { testM2LKernel0(); }

//---------------------------------------------------------------------------//

} // end namespace Test


/*

*/
