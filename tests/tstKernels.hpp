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

#include <test_helper_functions.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//

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

    Kokkos::Array<double, 6> coord_bounds = { -1.0, -1.0, -1.0, 1.0, 1.0, 1.0 };
    fillRandomCoordinates( cart_coords, coord_bounds );

    Kokkos::Array<double, 2> charge_bounds = { -10.0, 10.0 };
    fillRandomScalar( q, charge_bounds );

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
        // target. Equation 3.36 in Greengard
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

        double error =
            std::abs( potential_multipole.real() - potential_direct );
        double bound =
            std::pow( max_rho / r, p + 1 ) * std::abs( potential_direct );

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
    Kokkos::View<double* [3], TEST_MEMSPACE> coords0( "coords0", num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q0( "q", num_points );
    Kokkos::Array<double, 6> bounds0 = { -1.0, -1.0, -1.0, 1.0, 1.0, 1.0 };
    fillRandomCoordinates( coords0, bounds0 );
    Kokkos::Array<double, 2> qbounds0 = { -10.0, 10.0 };
    fillRandomScalar( q0, qbounds0 );

    // Center of Q coefficients
    Kokkos::Array<double, 3> q_center = { -0.1, 0.3, 0.2 };

    // Expansion center in polar coordinates - rho, alpha, beta
    double rho, alpha, beta;
    Canopy::Kernel::cart2sph( q_center[0], q_center[1], q_center[2], rho, alpha,
                              beta );

    // Target point - rho, theta, phi
    double Px = 10.0, Py = 0.0, Pz = 0.0;
    double r, theta, phi;
    Canopy::Kernel::cart2sph( Px, Py, Pz, r, theta, phi );

    // (Target point - q_center) - r_p, theta_p, phi_p
    double r_p, theta_p, phi_p;
    Canopy::Kernel::cart2sph( Px - q_center[0], Py - q_center[1],
                              Pz - q_center[2], r_p, theta_p, phi_p );

    // Direct potential at target point
    auto coords0_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), coords0 );
    auto q0_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), q0 );
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
        total_q += Kokkos::abs( q0_host( i ) );

        // Get max distance of each coordinate from the
        // multipole center for error estimate.
        ddx = coords0_host( i, 0 ) - q_center[0];
        ddy = coords0_host( i, 1 ) - q_center[1];
        ddz = coords0_host( i, 2 ) - q_center[2];
        rho_tmp = std::sqrt( ddx * ddx + ddy * ddy + ddz * ddz );
        a = std::max( a, rho_tmp );
    }

    // P should be far enough away from the expansion center
    EXPECT_GT( r, ( a + rho ) )
        << "Point P is not far enough away from expansion center";

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
        m2m( p2m.coefficients(), q_center );

        // Get translated multipole coefficients
        auto tmp = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                        m2m.coefficients() );
        Kokkos::View<cdouble*, Kokkos::HostSpace> M_host( "M_host",
                                                          tmp.extent( 0 ) );
        Kokkos::deep_copy( M_host, tmp );

        // Compute potential at P using M
        cdouble potential_M = 0.0;
        for ( int j = 0; j <= p; ++j )
        {
            for ( int k = -j; k <= j; ++k )
            {
                int idx = Canopy::Kernel::Scalar::index( j, k );
                potential_M += M_host( idx ) / Kokkos::pow( r, j + 1 ) *
                               Canopy::Kernel::Scalar::Ynm( j, k, theta, phi );
            }
        }

        // Check the error bounds from eq. 3.58
        auto bound = ( total_q / ( r - ( a + rho ) ) ) *
                     Kokkos::pow( ( a + rho ) / r, p + 1 );
        auto error = Kokkos::abs( potential_direct - potential_M );
        EXPECT_LE( error, bound )
            << "p=" << p
            << ": error between shifted and direct potentials too high.";
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

    const int points_per_section = 200;

    //
    Kokkos::View<double* [3], TEST_MEMSPACE> coords( "coords",
                                                     points_per_section * 2 );
    Kokkos::View<double*, TEST_MEMSPACE> q( "q", points_per_section * 2 );

    auto c0 = Kokkos::subview(
        coords, Kokkos::make_pair( 0, points_per_section ), Kokkos::ALL );
    auto c1 = Kokkos::subview(
        coords, Kokkos::make_pair( points_per_section, points_per_section * 2 ),
        Kokkos::ALL );
    auto q0 = Kokkos::subview( q, Kokkos::make_pair( 0, points_per_section ) );
    auto q1 = Kokkos::subview(
        q, Kokkos::make_pair( points_per_section, points_per_section * 2 ) );

    Kokkos::Array<double, 6> cbounds0 = { -3.0, -3.0, -3.0, -2.0, -2.0, -2.0 };
    Kokkos::Array<double, 6> cbounds1 = { 1.0, 1.0, 1.0, 2.0, 2.0, 2.0 };
    Kokkos::Array<double, 3> q0_center = { -2.5, -2.6, -2.7 };
    Kokkos::Array<double, 3> q1_center = { 1.3, 1.5, 1.6 };
    Kokkos::Array<double, 2> qbounds = { -10.0, 10.0 };

    fillRandomCoordinates( c0, cbounds0 );
    fillRandomCoordinates( c1, cbounds1 );
    fillRandomScalar( q, qbounds );

    // Expansion center of Q0 in polar coordinates - rho0, alpha0, beta0
    double rho0, alpha0, beta0;
    Canopy::Kernel::cart2sph( -q0_center[0], -q0_center[1], -q0_center[2], rho0,
                              alpha0, beta0 );

    // Expansion center of Q1 in polar coordinates - rho1, alpha1, beta1
    double rho1, alpha1, beta1;
    Canopy::Kernel::cart2sph( -q1_center[0], -q1_center[1], -q1_center[2], rho1,
                              alpha1, beta1 );

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
    for ( int i = 0; i < points_per_section * 2; ++i )
    {
        double dx, dy, dz, dist;
        double ddx, ddy, ddz, rho_tmp;

        dx = Px - coords_host( i, 0 );
        dy = Py - coords_host( i, 1 );
        dz = Pz - coords_host( i, 2 );
        dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        potential_direct += q_host( i ) / dist;

        // Get max distance from center for error estimate.
        if ( i < points_per_section )
        {
            // Use q0_center
            ddx = coords_host( i, 0 ) + q0_center[0];
            ddy = coords_host( i, 1 ) + q0_center[1];
            ddz = coords_host( i, 2 ) + q0_center[2];
            rho_tmp = std::sqrt( ddx * ddx + ddy * ddy + ddz * ddz );
            a0 = std::max( a0, rho_tmp );
        }
        else
        {
            // Use q1_center
            ddx = coords_host( i, 0 ) + q1_center[0];
            ddy = coords_host( i, 1 ) + q1_center[1];
            ddz = coords_host( i, 2 ) + q1_center[2];
            rho_tmp = std::sqrt( ddx * ddx + ddy * ddy + ddz * ddz );
            a1 = std::max( a1, rho_tmp );
        }
    }

    // P should be far enough away from both centers
    ASSERT_GT( r, ( a0 + rho0 ) )
        << "Point P is not far enough away from q0_center";
    ASSERT_GT( r, ( a1 + rho1 ) )
        << "Point P is not far enough away from q1_center";

    // Loop over truncation degree
    for ( int p = 1; p <= 5; ++p )
    {
        Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );
        Canopy::Kernel::Scalar::M2M<TEST_MEMSPACE, TEST_EXECSPACE> m2m( p );

        // Compute multipoles O0 at center Q0
        p2m( c0, q0, points_per_section, q0_center );

        // Translate O0 to be centered around the origin.
        m2m( p2m.coefficients(), q0_center );

        // Compute multipoles O1 at center Q1
        p2m.clear();
        p2m( c1, q1, points_per_section, q1_center );

        // Translate O1 to be centered around the origin.
        // Now that O1 and O0 have the same center, they can be added.
        // The m2m kernel shifts and then adds multipoles with
        // subsequent calls.
        m2m( p2m.coefficients(), q1_center );

        // Get translated multipole coefficients
        auto tmp = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                        m2m.coefficients() );
        Kokkos::View<cdouble*, Kokkos::HostSpace> M_host( "M_host",
                                                          tmp.extent( 0 ) );
        Kokkos::deep_copy( M_host, tmp );

        // Compute potential at P using M
        cdouble potential_M = 0.0;
        for ( int j = 0; j <= p; ++j )
        {
            for ( int k = -j; k <= j; ++k )
            {
                int idx = Canopy::Kernel::Scalar::index( j, k );
                potential_M += M_host( idx ) / Kokkos::pow( r, j + 1 ) *
                               Canopy::Kernel::Scalar::Ynm( j, k, theta, phi );
            }
        }

        // Check error between translated multipole and direct potentials
        // The error is already mathematically checked in testM2MKernel0,
        // so here we just make sure they are close to each other.
        double error = Kokkos::pow( 10, -p + 1 );
        EXPECT_NEAR( potential_direct, potential_M.real(), error )
            << "p=" << p
            << ": error between (shifted and added) and (direct potential) "
               "calculations too high.";
    }
}

void testM2LKernel()
{
    // Create points and q (scalar value)
    const int num_points = 500;
    Kokkos::View<double* [3], TEST_MEMSPACE> cart_coords( "cart_coords",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q( "q", num_points );

    Kokkos::Array<double, 6> coord_bounds = { -8.0, -8.0, -8.0,
                                              -5.0, -5.0, -5.0 };
    fillRandomCoordinates( cart_coords, coord_bounds );

    Kokkos::Array<double, 2> charge_bounds = { -3.0, 2.0 };
    fillRandomScalar( q, charge_bounds );

    // Expansion center
    Kokkos::Array<double, 3> center = { -5.5, -5.4, -5.3 };
    double rho, alpha, beta;
    Canopy::Kernel::cart2sph( center[0], center[1], center[2], rho, alpha,
                              beta );

    // First target point near origin (within radius 'a' of origin)
    double Px1 = 0.2, Py1 = -0.1, Pz1 = -0.5;
    double r1, theta1, phi1, r_d1, theta_d1, phi_d1;
    Canopy::Kernel::cart2sph( Px1 - center[0], Py1 - center[1], Pz1 - center[2],
                              r_d1, theta_d1, phi_d1 );
    Canopy::Kernel::cart2sph( Px1, Py1, Pz1, r1, theta1, phi1 );

    // Second target point near origin (within radius 'a' of origin)
    double Px2 = -0.4, Py2 = 0.3, Pz2 = -0.2;
    double r2, theta2, phi2, r_d2, theta_d2, phi_d2;
    Canopy::Kernel::cart2sph( Px2 - center[0], Py2 - center[1], Pz2 - center[2],
                              r_d2, theta_d2, phi_d2 );
    Canopy::Kernel::cart2sph( Px2, Py2, Pz2, r2, theta2, phi2 );

    // Compute a and total charge for error bound. (See figure 3.3)
    // Also compute direct potential
    auto cart_coords_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), cart_coords );
    auto q_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), q );
    double q_total = 0.0;
    double a = 0.0;
    double potential_direct1 = 0.0;
    double potential_direct2 = 0.0;
    for ( int i = 0; i < num_points; ++i )
    {
        double dx, dy, dz, dist;

        // Radius a
        dx = cart_coords_host( i, 0 ) - center[0];
        dy = cart_coords_host( i, 1 ) - center[1];
        dz = cart_coords_host( i, 2 ) - center[2];
        dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        a = std::max( a, dist );

        // Total sum
        q_total += std::abs( q_host( i ) );

        // Direct potential at first target point using coordinates relative to
        // origin.
        dx = Px1 - cart_coords_host( i, 0 );
        dy = Py1 - cart_coords_host( i, 1 );
        dz = Pz1 - cart_coords_host( i, 2 );
        dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        potential_direct1 += q_host( i ) / dist;

        // Direct potential at second target point using coordinates relative to
        // origin.
        dx = Px2 - cart_coords_host( i, 0 );
        dy = Py2 - cart_coords_host( i, 1 );
        dz = Pz2 - cart_coords_host( i, 2 );
        dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        potential_direct2 += q_host( i ) / dist;
    }

    // Theorem 3.5.5 requires c > 1 and rho > (c+1)*a.
    // Solve for c, getting c < (rho - a) / a for a > 0.
    ASSERT_GT( a, 0.0 );
    double c = ( rho - a ) / a;
    ASSERT_GT( c, 1.0 )
        << "Error: rho must be greater than (c+1)*a for theory to be valid.";

    // Target points must be within radius a of origin
    ASSERT_LT( r1, a ) << "Error: Target point 1 must be within distance 'a' "
                          "from origin for theory to be valid.";
    ASSERT_LT( r2, a ) << "Error: Target point 2 must be within distance 'a' "
                          "from origin for theory to be valid.";

    // Loop over truncation degree
    for ( int p = 1; p <= 9; ++p )
    {
        Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );
        p2m( cart_coords, q, num_points, center );

        // Convert multipoles to locals
        Canopy::Kernel::Scalar::M2L<TEST_MEMSPACE, TEST_EXECSPACE> m2l( p );
        m2l( p2m.coefficients(), center );

        // Copy to host
        auto L_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                           m2l.coefficients() );
        auto O_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                           p2m.coefficients() );

        // Perform local to potential conversion to calculate potential at
        // target. Equation 3.59 in Greengard
        using cdouble = Kokkos::complex<double>;
        cdouble potential_L1 = 0.0;
        cdouble potential_O1 = 0.0;
        cdouble potential_L2 = 0.0;
        cdouble potential_O2 = 0.0;
        for ( int j = 0; j <= p; ++j )
        {
            for ( int k = -j; k <= j; ++k )
            {
                int idx = Canopy::Kernel::Scalar::index( j, k );

                /* Target point 1 calculations */
                // Greengard eq. 3.59
                potential_L1 +=
                    L_host( idx ) * Kokkos::pow( r1, j ) *
                    Canopy::Kernel::Scalar::Ynm( j, k, theta1, phi1 );
                // Greengard eq. 3.36
                potential_O1 +=
                    O_host( idx ) / Kokkos::pow( r_d1, j + 1 ) *
                    Canopy::Kernel::Scalar::Ynm( j, k, theta_d1, phi_d1 );

                /* Target point 2 calculations */
                // Greengard eq. 3.59
                potential_L2 +=
                    L_host( idx ) * Kokkos::pow( r2, j ) *
                    Canopy::Kernel::Scalar::Ynm( j, k, theta2, phi2 );
                // Greengard eq. 3.36
                potential_O2 +=
                    O_host( idx ) / Kokkos::pow( r_d2, j + 1 ) *
                    Canopy::Kernel::Scalar::Ynm( j, k, theta_d2, phi_d2 );
            }
        }

        // Check the error bounds from eq. 3.61
        double bound = ( q_total / ( c * a - a ) ) * std::pow( 1.0 / c, p + 1 );
        double error1 = std::abs( potential_L1.real() - potential_direct1 );
        double error2 = std::abs( potential_L2.real() - potential_direct2 );
        EXPECT_LE( error1, bound ) << "p=" << p
                                   << ": error between local and direct "
                                      "potentials at target point 1 too high.";
        EXPECT_LE( error2, bound ) << "p=" << p
                                   << ": error between local and direct "
                                      "potentials at target point 2 too high.";
        // printf("p=%d: D1: %0.5lf, L1: %0.5lf, M1: %0.5lf, b1: %0.5lf e1:
        // %0.5lf\n", p,
        //     potential_direct1, potential_L1.real(), potential_O1.real(),
        //     bound, error1);
        // printf("p=%d: D2: %0.5lf, L2: %0.5lf, M2: %0.5lf, b2: %0.5lf e2:
        // %0.5lf\n", p,
        //     potential_direct2, potential_L2.real(), potential_O2.real(),
        //     bound, error2);
    }
}

void testL2LKernel()
{
    // Create points and q (scalar value)
    const int num_points = 500;
    Kokkos::View<double* [3], TEST_MEMSPACE> cart_coords( "cart_coords",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q( "q", num_points );

    Kokkos::Array<double, 6> coord_bounds = { -8.0, -8.0, -8.0,
                                              -5.0, -5.0, -5.0 };
    fillRandomCoordinates( cart_coords, coord_bounds );

    Kokkos::Array<double, 2> charge_bounds = { -3.0, 2.0 };
    fillRandomScalar( q, charge_bounds );

    // Multipole expansion center
    Kokkos::Array<double, 3> m_center = { -5.5, -5.4, -5.3 };
    double rho_m, alpha_m, beta_m;
    Canopy::Kernel::cart2sph( m_center[0], m_center[1], m_center[2], rho_m, alpha_m,
                              beta_m );
    
    // Center of local expansion X_0: A vector from the origin to the
    // center of local expansion.
    // Kokkos::Array<double, 3> X_0 = { 3.5, 6.9, 5.1 };
    Kokkos::Array<double, 3> X_0 = { 0.0, 0.0, 0.0 };
    double rho, alpha, beta;
    Canopy::Kernel::cart2sph( X_0[0], X_0[1], X_0[2], rho, alpha,
                              beta );

    // Target point X
    double X_x = 0.7, X_y = 0.6, X_z = 0.9;
    double r, theta, phi;
    Canopy::Kernel::cart2sph( X_x, X_y, X_z,
                              r, theta, phi );
    
    // Vector X - X_0
    double r_p, theta_p, phi_p;
    Canopy::Kernel::cart2sph( X_x - X_0[0], X_y - X_0[1], X_z - X_0[2],
                              r_p, theta_p, phi_p );

    // Compute a and total charge for error bound. (See figure 3.3)
    // Also compute direct potential
    auto cart_coords_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), cart_coords );
    auto q_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), q );
    double q_total = 0.0;
    double a = 0.0;
    double potential_direct = 0.0;
    for ( int i = 0; i < num_points; ++i )
    {
        double dx, dy, dz, dist;

        // Radius a
        dx = cart_coords_host( i, 0 ) - m_center[0];
        dy = cart_coords_host( i, 1 ) - m_center[1];
        dz = cart_coords_host( i, 2 ) - m_center[2];
        dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        a = std::max( a, dist );

        // Total sum
        q_total += std::abs( q_host( i ) );

        // Direct potential at target point using coordinates relative to
        // origin.
        dx = X_x - cart_coords_host( i, 0 );
        dy = X_y - cart_coords_host( i, 1 );
        dz = X_z - cart_coords_host( i, 2 );
        dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        potential_direct += q_host( i ) / dist;
    }

    // Theorem 3.5.5 requires c > 1 and rho > (c+1)*a.
    // Solve for c, getting c < (rho - a) / a for a > 0.
    ASSERT_GT( a, 0.0 );
    double c = ( rho_m - a ) / a;
    EXPECT_GT( c, 1.0 )
        << "Error: rho must be greater than (c+1)*a for theory to be valid. a = " << a;

    // Target point must be within radius a of local center
    EXPECT_LT( r_p, a ) << "Error: Target point must be within distance 'a' "
                          "from local center for theory to be valid.";

    // Loop over truncation degree
    for ( int p = 1; p <= 5; ++p )
    {
        // Create multipole coefficients.
        Canopy::Kernel::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );
        p2m( cart_coords, q, num_points, m_center );

        // Convert multipoles to locals. By Theorem 2.4 in Cheng, these local
        // coefficients are centered around the origin.
        Canopy::Kernel::Scalar::M2L<TEST_MEMSPACE, TEST_EXECSPACE> m2l( p );
        m2l( p2m.coefficients(), m_center );

        // Translate locals to X_0
        Canopy::Kernel::Scalar::L2L<TEST_MEMSPACE, TEST_EXECSPACE> l2l( p );
        l2l( m2l.coefficients(), X_0 );

        // Copy to host
        auto O_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                           m2l.coefficients() );
        auto L_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                           l2l.coefficients() );

        // Perform local to potential conversion to calculate potential at
        // target. Equation 3.59 in Greengard
        using cdouble = Kokkos::complex<double>;
        cdouble potential_O = 0.0;
        cdouble potential_L = 0.0;
        for ( int j = 0; j <= p; ++j )
        {
            for ( int k = -j; k <= j; ++k )
            {
                int idx = Canopy::Kernel::Scalar::index( j, k );

                // Chang eq. 19
                potential_O +=
                    O_host( idx ) * Kokkos::pow( r_p, j) *
                    Canopy::Kernel::Scalar::Ynm( j, k, theta_p, phi_p );

                // Cheng eq. 20
                potential_L +=
                    L_host( idx ) * Kokkos::pow( r, j ) *
                    Canopy::Kernel::Scalar::Ynm( j, k, theta, phi );
            }
        }

        // Check the error bounds from eq. 3.61
        double bound = ( q_total / ( c * a - a ) ) * std::pow( 1.0 / c, p + 1 );
        double error = std::abs( potential_L.real() - potential_direct );
        // EXPECT_LE( error, bound ) << "p=" << p
        //                            << ": error between local and direct "
        //                               "potentials at target point too high.";
        printf("p=%d: D: %0.5lf, L: %0.5lf, O: %0.5lf, b: %0.5lf e: %0.5lf\n", p,
            potential_direct, potential_L.real(), potential_O.real(),
            bound, error);
        // printf("p=%d: D2: %0.5lf, L2: %0.5lf, M2: %0.5lf, b2: %0.5lf e2:
        // %0.5lf\n", p,
        //     potential_direct2, potential_L2.real(), potential_O2.real(),
        //     bound, error2);
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
// TEST( Kernel, testScalarP2MKernel ) { testScalarP2MKernel(); }

// TEST( Kernel, testM2MKernel0 ) { testM2MKernel0(); }

// TEST( Kernel, testM2MKernel1 ) { testM2MKernel1(); }

// TEST( Kernel, testM2LKernel ) { testM2LKernel(); }

TEST( Kernel, testL2LKernel ) { testL2LKernel(); }

//---------------------------------------------------------------------------//

} // end namespace Test
