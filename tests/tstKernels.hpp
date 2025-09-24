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

    // Target point P
    double Px = 6.6, Py = -5.1, Pz = 1.9;
    // P relative to the multipole expansion center.
    double r, theta, phi;
    Canopy::Kernel::cart2sph( Px - expansion_center[0],
                              Py - expansion_center[1],
                              Pz - expansion_center[2], r, theta, phi );

    // Direct potential
    auto cart_coords_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), cart_coords );
    auto q_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), q );
    double potential_direct = 0.0;
    double max_rho = 0.0; // for error bound
    for ( int i = 0; i < num_points; ++i )
    {
        double dx = Px - cart_coords_host( i, 0 );
        double dy = Py - cart_coords_host( i, 1 );
        double dz = Pz - cart_coords_host( i, 2 );
        double dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        potential_direct += q_host( i ) / dist;

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
        Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );

        // Particle to multipole calculation performed in operator
        p2m( cart_coords, q, num_points, expansion_center );
        auto M = p2m.coefficients();
        auto M_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M );

        // Perform multipole to particle conversion to calculate potential at
        // target. Equation 3.36 in source 4
        using cdouble = Kokkos::complex<double>;
        cdouble potential_multipole = 0.0;
        for ( int n = 0; n <= p; ++n )
        {
            // double norm = 4 * pi / double( 2 * n + 1 );
            // auto norm = Kokkos::sqrt(( ( 2.0 * n + 1 ) / ( 4.0 * pi ) ));
            for ( int m = -n; m <= n; ++m )
            {
                int idx = Canopy::Kernel::Scalar::index( n, m );
                potential_multipole +=
                    M_host( idx ) / Kokkos::pow( r, n + 1 ) *
                    Canopy::Kernel::Scalar::Ynm( n, m, theta, phi );
            }
        }

        double error = std::abs( potential_multipole.real() - potential_direct );
        double bound = std::pow( max_rho / r, p + 1 ) * std::abs( potential_direct );

        // Check that the error is within 5*bound, which accounts
        // for imprecision due to imtermediate rounding.
        EXPECT_NEAR( potential_multipole.real(), potential_direct, 5 * bound )
            << "p=" << p << " multipole=" << potential_multipole.real()
            << " direct=" << potential_direct << " error=" << error << " bound~"
            << bound << std::endl;
    }
}

/**
 * Tests translation of multipole coefficients
 * Creates a multipole expansions around one center and translates
 * it to another center. Tests against the exact calculation 
 * for potential at the translated center.
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

        // Calculate potential directly from P relative to the origin.
        dx = Px - coords0_host( i, 0 );
        dy = Py - coords0_host( i, 1 );
        dz = Pz - coords0_host( i, 2 );
        dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        potential_direct += q0_host( i ) / dist;

        // Add total charge for error bound
        total_q += Kokkos::abs(q0_host(i));

        // Get max distance of each coordinate from the 
        // multipole center for error estimate.
        ddx = coords0_host( i, 0 ) - q_center[0];
        ddy = coords0_host( i, 1 ) - q_center[1];
        ddz = coords0_host( i, 2 ) - q_center[2];
        rho_tmp = std::sqrt( ddx * ddx + ddy * ddy + ddz * ddz );
        a = std::max( a, rho_tmp );
    }

    // P should be far enough away from the expansion center
    EXPECT_GT(r, (a + rho)) << "Point P is not far enough away from expansion center";

    // Loop over truncation degree
    for ( int p = 1; p <= 5; ++p )
    {
        Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );
        Canopy::Kernel::Scalar::M2M<TEST_MEMSPACE, TEST_EXECSPACE> m2m( p );

        // Compute multipoles O_nm at center Q
        p2m( coords0, q0, num_points, q_center );        

        // Compute multipoles M_kj coefficients - which are the 
        // multipoles O_nm translated to be centered around
        // the origin.
        m2m(p2m.coefficients(), q_center);

        // Get translated multipole coefficients
        auto tmp = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), m2m.coefficients() );
        Kokkos::View<cdouble*,Kokkos::HostSpace> M_host("M_host", tmp.extent(0));
        Kokkos::deep_copy(M_host, tmp);

        // Compute potential at P using M
        cdouble potential_M = 0.0;
        for ( int j = 0; j <= p; ++j )
        {
            for ( int k = -j; k <= j; ++k )
            {
                int idx = Canopy::Kernel::Scalar::index( j, k );
                potential_M +=
                    M_host( idx ) / Kokkos::pow( r, j + 1 ) *
                    Canopy::Kernel::Scalar::Ynm( j, k, theta, phi );
            }
        }

        // Check the error bounds from eq. 3.58
        auto bound = (total_q / (r - (a + rho))) * Kokkos::pow((a + rho)/r, p + 1);
        auto error = Kokkos::abs(potential_direct - potential_M);
        EXPECT_LE(error, bound) << "p="
            << p << ": error between shifted and direct potentials too high.";
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
void testM2MKernel1()
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
    Kokkos::Array<double, 3> q0_center = { -2.5, -2.6, -2.7 };
    Kokkos::Array<double, 3> q1_center = { 1.3, 1.5, 1.6 };
    Kokkos::Array<double, 2> qbounds = {-10.0, 10.0};

    fillRandomCoordinates(c0, cbounds0);
    fillRandomCoordinates(c1, cbounds1);
    fillRandomScalar(q, qbounds);

    // Expansion center of Q0 in polar coordinates - rho0, alpha0, beta0
    double rho0, alpha0, beta0;
    Canopy::Kernel::cart2sph( q0_center[0], q0_center[1], q0_center[2], rho0, alpha0, beta0 );

    // Expansion center of Q1 in polar coordinates - rho1, alpha1, beta1
    double rho1, alpha1, beta1;
    Canopy::Kernel::cart2sph( q1_center[0], q1_center[1], q1_center[2], rho1, alpha1, beta1 );

    // Target point - rho0, theta, phi
    double Px = 15.0, Py = -10.0, Pz = 7.0;
    double r, theta, phi;
    Canopy::Kernel::cart2sph( Px, Py, Pz, r, theta, phi );
    
    // (Target point - q0_center) - r_p, theta_p, phi_p
    // double r_p, theta_p, phi_p;
    // Canopy::Kernel::cart2sph( Px - q0_center[0],
    //                           Py - q0_center[1],
    //                           Pz - q0_center[2], r_p, theta_p, phi_p );

    // Direct potential using P relative to the origin.
    auto coords_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), coords );
    auto q_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), q );
    double potential_direct = 0.0;
    double a0 = 0.0, a1 = 0.0;
    for ( int i = 0; i < points_per_section*2; ++i )
    {
        double dx, dy, dz, dist;
        double ddx, ddy, ddz, rho_tmp;

        dx = Px - coords_host( i, 0 );
        dy = Py - coords_host( i, 1 );
        dz = Pz - coords_host( i, 2 );
        dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        potential_direct += q_host( i ) / dist;

        // Get max distance from center for error estimate.
        if (i < points_per_section)
        {
            // Use q0_center
            ddx = coords_host( i, 0 ) - q0_center[0];
            ddy = coords_host( i, 1 ) - q0_center[1];
            ddz = coords_host( i, 2 ) - q0_center[2];
            rho_tmp = std::sqrt( ddx * ddx + ddy * ddy + ddz * ddz );
            a0 = std::max( a0, rho_tmp );
        }
        else 
        {
            // Use q1_center
            ddx = coords_host( i, 0 ) - q1_center[0];
            ddy = coords_host( i, 1 ) - q1_center[1];
            ddz = coords_host( i, 2 ) - q1_center[2];
            rho_tmp = std::sqrt( ddx * ddx + ddy * ddy + ddz * ddz );
            a1 = std::max( a1, rho_tmp );
        }
    }
    
    // P should be far enough away from both centers
    ASSERT_GT(r, (a0 + rho0)) << "Point P is not far enough away from q0_center";
    ASSERT_GT(r, (a1 + rho1)) << "Point P is not far enough away from q1_center";

    // Loop over truncation degree
    for ( int p = 1; p <= 5; ++p )
    {
        Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );
        Canopy::Kernel::Scalar::M2M<TEST_MEMSPACE, TEST_EXECSPACE> m2m( p );

        // Compute multipoles O0 at center Q0
        p2m( c0, q0, points_per_section, q0_center );        

        // Translate O0 to be centered around the origin.
        m2m(p2m.coefficients(), q0_center);

        // Compute multipoles O1 at center Q1
        p2m.clear();
        p2m(c1, q1, points_per_section, q1_center);

        // Translate O1 to be centered around the origin.
        // Now that O1 and O0 have the same center, they can be added.
        // The m2m kernel shifts and then adds multipoles with
        // subsequent calls.
        m2m(p2m.coefficients(), q1_center);

        // Get translated multipole coefficients
        auto tmp = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), m2m.coefficients() );
        Kokkos::View<cdouble*,Kokkos::HostSpace> M_host("M_host", tmp.extent(0));
        Kokkos::deep_copy(M_host, tmp);

        // Compute potential at P using M
        cdouble potential_M = 0.0;
        for ( int j = 0; j <= p; ++j )
        {
            for ( int k = -j; k <= j; ++k )
            {
                int idx = Canopy::Kernel::Scalar::index( j, k );
                potential_M +=
                    M_host( idx ) / Kokkos::pow( r, j + 1 ) *
                    Canopy::Kernel::Scalar::Ynm( j, k, theta, phi );
            }
        }

        // Check error between translated multipole and direct potentials
        // The error is already mathematically checked in testM2MKernel0,
        // so here we just make sure they are close to each other.
        double error = Kokkos::pow(10, -p+1);
        EXPECT_NEAR(potential_direct, potential_M.real(), error) << "p="
            << p << ": error between (shifted and added) and (direct potential) calculations too high.";
    }
}

void testM2LKernel0()
{
    // Create points and q (scalar value)
    const int num_points = 1;
    Kokkos::View<double* [3], TEST_MEMSPACE> cart_coords( "cart_coords",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q( "q", num_points );
    
    Kokkos::Array<double, 6> coord_bounds = {-5.0, -5.0, -5.0, -4.0, -4.0, -4.0};
    fillRandomCoordinates(cart_coords, coord_bounds);

    Kokkos::Array<double, 2> charge_bounds = {-10.0, 10.0};
    fillRandomScalar(q, charge_bounds);

    // cart_coords(0, 0) = 10.0;
    // cart_coords(0, 0) = 0.0;
    // cart_coords(0, 0) = 0.0; 
    // q(0) = 1.0;
    // Kokkos::Array<double, 3> center = { 11.0, 0.0, 0.0 };

    // Expansion center
    Kokkos::Array<double, 3> center = { -4.5, -4.4, -4.3 };
    double rho, alpha, beta;
    Canopy::Kernel::cart2sph( center[0], center[1], center[2], rho, alpha, beta );

    // Target point near origin (within radius 'a' of origin)
    double Px = 0.1, Py = 0.0, Pz = 0.0;
    double r, theta, phi, r_d, theta_d, phi_d;
    Canopy::Kernel::cart2sph( Px - center[0], Py- center[1], Pz - center[2], r_d, theta_d, phi_d );
    Canopy::Kernel::cart2sph( Px, Py, Pz, r, theta, phi);
    
    // Compute a and total charge for error bound. (See figure 3.3)
    // Also compute direct potential
    auto cart_coords_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cart_coords);
    auto q_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q);
    double q_total = 0.0;
    double a = 0.0;
    double potential_direct = 0.0;
    for (int i = 0; i < num_points; ++i)
    {
        double dx, dy, dz, dist;

        // Radius a
        dx = cart_coords_host(i,0) - center[0];
        dy = cart_coords_host(i,1) - center[1];
        dz = cart_coords_host(i,2) - center[2];
        dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        a = std::max(a, dist);

        // Total sum
        q_total += std::abs(q_host(i));

        // Direct potential using coordinates relative to origin.
        dx = Px - cart_coords_host( i, 0 );
        dy = Py - cart_coords_host( i, 1 );
        dz = Pz - cart_coords_host( i, 2 );
        dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        potential_direct += q_host( i ) / dist;
    }
    
    // Theorem 3.5.5 requires c > 1 and rho > (c+1)*a.
    // Solve for c, getting c < (rho - a) / a for a > 0.
    EXPECT_GT(a, 0.0);
    double c = (rho - a) / a;
    EXPECT_GT(c, 1.0) << "Error: rho must be greater than (c+1)*a for theory to be valid." << std::endl;

    printf("a=%0.1lf, c=%0.1lf, rho=%0.1lf, (c+1)a=%0.1lf\n", a, c, rho, (c+1)*a);

    // Loop over truncation degree
    for ( int p = 1; p <= 7; ++p )
    {
        Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );
        p2m( cart_coords, q, num_points, center );

        // Translate multipoles to be centered around the origin
        // Canopy::Kernel::Scalar::M2L<TEST_MEMSPACE, TEST_EXECSPACE> m2m( p );
        // m2m( p2m.coefficients(), center );

        // Convert translated multipoles to locals
        Canopy::Kernel::Scalar::M2L<TEST_MEMSPACE, TEST_EXECSPACE> m2l( p );
        m2l( p2m.coefficients(), center );
        auto L_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), m2l.coefficients() );
        auto O_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), p2m.coefficients() );

        // Perform local to potential conversion to calculate potential at
        // target. Equation 3.59 in source 4
        using cdouble = Kokkos::complex<double>;
        cdouble potential_L = 0.0;
        cdouble potential_O = 0.0;
        for ( int j = 0; j <= p; ++j )
        {
            for ( int k = -j; k <= j; ++k )
            {
                int idx = Canopy::Kernel::Scalar::index( j, k );
                potential_L +=
                    L_host( idx ) * Kokkos::pow( r, j ) *
                    Canopy::Kernel::Scalar::Ynm( j, k, theta, phi );
                potential_O +=
                    O_host( idx ) / Kokkos::pow( r_d, j + 1) *
                    Canopy::Kernel::Scalar::Ynm( j, k, theta_d, phi_d );
            }
        }

        // Check the error bounds from eq. 3.61
        double bound = ( q_total / ( c * a - a ) ) * std::pow( 1.0 / c, p + 1 );
        double error = std::abs( potential_L.real() - potential_direct );
        // EXPECT_LE(error, bound) << "p="
        //     << p << ": error between local and direct potentials too high.";
        printf("p=%d: Direct: %0.7lf, local: %0.7lf, multi: %0.7lf\n", p,
            potential_direct, potential_L.real(), potential_O.real());
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( Kernel, testScalarP2MKernel ) { testScalarP2MKernel(); }

TEST( Kernel, testM2MKernel0 ) { testM2MKernel0(); }

TEST( Kernel, testM2MKernel1 ) { testM2MKernel1(); }

TEST( Kernel, testM2LKernel0 ) { testM2LKernel0(); }

//---------------------------------------------------------------------------//

} // end namespace Test


/*

*/
