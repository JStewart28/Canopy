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

#include <Canopy_Operators.hpp>

#include <test_helpers.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//

using cdouble = Kokkos::complex<double>;
constexpr auto pi = Kokkos::numbers::pi_v<double>;

/**
 * Test that scalar structs are correctly calculated
 */
void testP2MStruct0()
{
    // Create points and q (scalar value)
    const int num_points = 20;
    Kokkos::View<double* [3], TEST_MEMSPACE> cart_coords( "cart_coords",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q( "q", num_points );

    Kokkos::Array<double, 6> coord_bounds = { -1.0, -1.0, -1.0, 1.0, 1.0, 1.0 };
    fillRandomCoordinates( cart_coords, coord_bounds, 321 );

    Kokkos::Array<double, 2> charge_bounds = { -10.0, 10.0 };
    fillRandomScalar( q, charge_bounds, 234 );

    // Expansion center
    Kokkos::Array<double, 3> expansion_center = { 0.1, -0.6, 0.3 };

    // Target point P
    double Px = 6.6, Py = -5.1, Pz = 1.9;
    // P relative to the multipole expansion center.
    double r, theta, phi;
    Canopy::Operator::cart2sph( Px - expansion_center[0],
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
        Canopy::Operator::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );

        // Particle to multipole calculation performed in operator
        p2m( cart_coords, q, num_points, expansion_center );
        auto M = p2m.coefficients();
        auto M_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M );

        // Perform multipole to particle conversion to calculate potential at
        // target. Equation 3.36 in Greengard
        cdouble potential_multipole = 0.0;
        for ( int n = 0; n <= p; ++n )
        {
            // double norm = 4 * pi / double( 2 * n + 1 );
            // auto norm = Kokkos::sqrt(( ( 2.0 * n + 1 ) / ( 4.0 * pi ) ));
            for ( int m = -n; m <= n; ++m )
            {
                int idx = Canopy::Operator::Scalar::index( n, m );
                potential_multipole +=
                    M_host( idx ) / Kokkos::pow( r, n + 1 ) *
                    Canopy::Operator::Scalar::Ynm( n, m, theta, phi );
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
void testM2MStruct0()
{
    const int num_points = 20;

    // Domain 0
    Kokkos::View<double* [3], TEST_MEMSPACE> coords0( "coords0", num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q0( "q", num_points );
    Kokkos::Array<double, 6> bounds0 = { -1.0, -1.0, -1.0, 1.0, 1.0, 1.0 };
    fillRandomCoordinates( coords0, bounds0, 789 );
    Kokkos::Array<double, 2> qbounds0 = { -10.0, 10.0 };
    fillRandomScalar( q0, qbounds0, 987 );

    // Center of Q coefficients
    Kokkos::Array<double, 3> q_center = { -0.1, 0.3, 0.2 };

    // Expansion center in polar coordinates - rho, alpha, beta
    double rho, alpha, beta;
    Canopy::Operator::cart2sph( q_center[0], q_center[1], q_center[2], rho, alpha,
                              beta );

    // Target point - rho, theta, phi
    double Px = 10.0, Py = 0.0, Pz = 0.0;
    double r, theta, phi;
    Canopy::Operator::cart2sph( Px, Py, Pz, r, theta, phi );

    // (Target point - q_center) - r_p, theta_p, phi_p
    double r_p, theta_p, phi_p;
    Canopy::Operator::cart2sph( Px - q_center[0], Py - q_center[1],
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
        Canopy::Operator::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );
        Canopy::Operator::Scalar::M2M<TEST_MEMSPACE, TEST_EXECSPACE> m2m( p );

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
                int idx = Canopy::Operator::Scalar::index( j, k );
                potential_M += M_host( idx ) / Kokkos::pow( r, j + 1 ) *
                               Canopy::Operator::Scalar::Ynm( j, k, theta, phi );
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
 * disjunct, well-separated domains. Translations these expansions
 * to center around a new center and then adds these expansions together.
 * Converts the aggregated multipole expansions back to potentials at
 * a target point and compares the result to the directly calculated potential
 * at the target point.
 */
void testM2MStruct1()
{
    const int points_per_section = 200;

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

    fillRandomCoordinates( c0, cbounds0, 765 );
    fillRandomCoordinates( c1, cbounds1, 567 );
    fillRandomScalar( q, qbounds, 444 );

    // Expansion center of Q0 in polar coordinates - rho0, alpha0, beta0
    double rho0, alpha0, beta0;
    Canopy::Operator::cart2sph( -q0_center[0], -q0_center[1], -q0_center[2], rho0,
                              alpha0, beta0 );

    // Expansion center of Q1 in polar coordinates - rho1, alpha1, beta1
    double rho1, alpha1, beta1;
    Canopy::Operator::cart2sph( -q1_center[0], -q1_center[1], -q1_center[2], rho1,
                              alpha1, beta1 );

    // Target point - rho0, theta, phi
    double Px = 15.0, Py = -10.0, Pz = 7.0;
    double r, theta, phi;
    Canopy::Operator::cart2sph( Px, Py, Pz, r, theta, phi );

    // (Target point - q0_center) - r_p, theta_p, phi_p
    // double r_p, theta_p, phi_p;
    // Canopy::Operator::cart2sph( Px - q0_center[0],
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
        Canopy::Operator::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );
        Canopy::Operator::Scalar::M2M<TEST_MEMSPACE, TEST_EXECSPACE> m2m( p );

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
                int idx = Canopy::Operator::Scalar::index( j, k );
                potential_M += M_host( idx ) / Kokkos::pow( r, j + 1 ) *
                               Canopy::Operator::Scalar::Ynm( j, k, theta, phi );
            }
        }

        // Check error between translated multipole and direct potentials
        // The error is already mathematically checked in testM2MStruct0,
        // so here we just make sure they are close to each other.
        double error = Kokkos::pow( 10, -p + 2 );
        EXPECT_NEAR( potential_direct, potential_M.real(), error )
            << "p=" << p
            << ": error between (shifted and added) and (direct potential) "
               "calculations too high.";
    }
}

void testM2LStruct0()
{
    // Create points and q (scalar value)
    const int num_points = 500;
    Kokkos::View<double* [3], TEST_MEMSPACE> cart_coords( "cart_coords",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q( "q", num_points );

    Kokkos::Array<double, 6> coord_bounds = { -8.0, -8.0, -8.0,
                                              -5.0, -5.0, -5.0 };
    fillRandomCoordinates( cart_coords, coord_bounds, 563 );

    Kokkos::Array<double, 2> charge_bounds = { -3.0, 2.0 };
    fillRandomScalar( q, charge_bounds, 742 );

    // Expansion center
    Kokkos::Array<double, 3> center = { -5.5, -5.4, -5.3 };
    double rho, alpha, beta;
    Canopy::Operator::cart2sph( center[0], center[1], center[2], rho, alpha,
                              beta );

    // First target point near origin (within radius 'a' of origin)
    double Px1 = 0.2, Py1 = -0.1, Pz1 = -0.5;
    double r1, theta1, phi1, r_d1, theta_d1, phi_d1;
    Canopy::Operator::cart2sph( Px1 - center[0], Py1 - center[1], Pz1 - center[2],
                              r_d1, theta_d1, phi_d1 );
    Canopy::Operator::cart2sph( Px1, Py1, Pz1, r1, theta1, phi1 );

    // Second target point near origin (within radius 'a' of origin)
    double Px2 = -0.4, Py2 = 0.3, Pz2 = -0.2;
    double r2, theta2, phi2, r_d2, theta_d2, phi_d2;
    Canopy::Operator::cart2sph( Px2 - center[0], Py2 - center[1], Pz2 - center[2],
                              r_d2, theta_d2, phi_d2 );
    Canopy::Operator::cart2sph( Px2, Py2, Pz2, r2, theta2, phi2 );

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

        // Total charge
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
        Canopy::Operator::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );
        p2m( cart_coords, q, num_points, center );

        // Convert multipoles to locals
        Canopy::Operator::Scalar::M2L<TEST_MEMSPACE, TEST_EXECSPACE> m2l( p );
        m2l( p2m.coefficients(), center );

        // Copy to host
        auto L_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                           m2l.coefficients() );
        auto O_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                           p2m.coefficients() );

        // Perform local to potential conversion to calculate potential at
        // target. Equation 3.59 in Greengard

        cdouble potential_L1 = 0.0;
        cdouble potential_O1 = 0.0;
        cdouble potential_L2 = 0.0;
        cdouble potential_O2 = 0.0;
        for ( int j = 0; j <= p; ++j )
        {
            for ( int k = -j; k <= j; ++k )
            {
                int idx = Canopy::Operator::Scalar::index( j, k );

                /* Target point 1 calculations */
                // Greengard eq. 3.59
                potential_L1 +=
                    L_host( idx ) * Kokkos::pow( r1, j ) *
                    Canopy::Operator::Scalar::Ynm( j, k, theta1, phi1 );
                // Greengard eq. 3.36
                potential_O1 +=
                    O_host( idx ) / Kokkos::pow( r_d1, j + 1 ) *
                    Canopy::Operator::Scalar::Ynm( j, k, theta_d1, phi_d1 );

                /* Target point 2 calculations */
                // Greengard eq. 3.59
                potential_L2 +=
                    L_host( idx ) * Kokkos::pow( r2, j ) *
                    Canopy::Operator::Scalar::Ynm( j, k, theta2, phi2 );
                // Greengard eq. 3.36
                potential_O2 +=
                    O_host( idx ) / Kokkos::pow( r_d2, j + 1 ) *
                    Canopy::Operator::Scalar::Ynm( j, k, theta_d2, phi_d2 );
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

/**
 * Convert multipoles from two separate expansions to a single set
 * of local coefficients.
 */
void testM2LStruct1()
{
    const int points_per_section = 200;

    Kokkos::View<double* [3], TEST_MEMSPACE> coords( "coords",
                                                     points_per_section * 2 );
    Kokkos::View<double*, TEST_MEMSPACE> q( "q", points_per_section * 2 );

    auto coord0 = Kokkos::subview(
        coords, Kokkos::make_pair( 0, points_per_section ), Kokkos::ALL );
    auto coord1 = Kokkos::subview(
        coords, Kokkos::make_pair( points_per_section, points_per_section * 2 ),
        Kokkos::ALL );
    auto q0 = Kokkos::subview( q, Kokkos::make_pair( 0, points_per_section ) );
    auto q1 = Kokkos::subview(
        q, Kokkos::make_pair( points_per_section, points_per_section * 2 ) );

    Kokkos::Array<double, 6> cbounds0 = { -13.0, -13.0, -13.0, -12.0, -12.0, -12.0 };
    Kokkos::Array<double, 6> cbounds1 = { 11.0, 11.0, 11.0, 12.0, 12.0, 12.0 };
    Kokkos::Array<double, 3> q0_center = { -12.5, -12.6, -12.7 };
    Kokkos::Array<double, 3> q1_center = { 11.3, 11.5, 11.6 };
    Kokkos::Array<double, 2> qbounds = { -10.0, 10.0 };

    fillRandomCoordinates( coord0, cbounds0, 123 );
    fillRandomCoordinates( coord1, cbounds1, 321 );
    fillRandomScalar( q, qbounds, 111 );

    // Center of local expansion
    Kokkos::Array<double, 3> l_center = { 1.3, 0.5, -0.6 };

    // Target point near local center - rho, theta, phi
    // Convert to spherical coordinates relative to local center
    double Px = 1.0, Py = 0.1, Pz = -0.3;
    double r, theta, phi;
    Canopy::Operator::cart2sph( Px - l_center[0],
                              Py - l_center[1],
                              Pz - l_center[2],
                              r, theta, phi );

    // Direct potential at P
    auto coords_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), coords );
    auto q_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), q );
    double q_total = 0.0;
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

        // Total charge
        q_total += std::abs( q_host( i ) );

        // Get max distance from center for error estimate.
        if ( i < points_per_section )
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

    // Theorem 3.5.5 requires the distance from the local center to the
    // multipole center to be at least 3x (farthest distance from a charge
    // to the multipole center)
    double dist0 = distance(l_center, q0_center);
    ASSERT_GT( dist0, 3 * a0 )
        << "Error: Local center too close to multipole 0 center.";
    double dist1 = distance(l_center, q1_center);
    ASSERT_GT( dist1, 3 * a1 )
        << "Error: Local center too close to multipole 1 center.";

    // Target point must be within radius a of local center
    auto a_min = Kokkos::min(a0, a1);
    ASSERT_LT( r, a_min ) << "Error: Target point must be within distance 'a' "
                          "from local center for theory to be valid.";

    // Calculate c
    double dist_min = Kokkos::min(dist0, dist1);
    double c = ( dist_min - a_min ) / a_min;
    ASSERT_GT( c, 1.0 )
        << "Error: rho must be greater than (c+1)*a for theory to be valid.";

    // Loop over truncation degree
    for ( int p = 1; p <= 7; ++p )
    {
        Canopy::Operator::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m0( p );
        p2m0( coord0, q0, points_per_section, q0_center );

        Canopy::Operator::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m1( p );
        p2m1( coord1, q1, points_per_section, q1_center );

        // Convert multipoles to locals
        Canopy::Operator::Scalar::M2L<TEST_MEMSPACE, TEST_EXECSPACE> m2l( p );
        m2l( p2m0.coefficients(), Kokkos::Array<double, 3>{q0_center[0]- l_center[0],
                                                            q0_center[1]- l_center[1], 
                                                            q0_center[2]- l_center[2]} );
        m2l( p2m1.coefficients(), Kokkos::Array<double, 3>{q1_center[0]- l_center[0],
                                                            q1_center[1]- l_center[1], 
                                                            q1_center[2]- l_center[2]} );

        // Copy to host
        auto L_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                           m2l.coefficients() );

        // Perform local to potential conversion to calculate potential at
        // target. Equation 3.59 in Greengard
        cdouble potential_L = 0.0;
        for ( int j = 0; j <= p; ++j )
        {
            for ( int k = -j; k <= j; ++k )
            {
                int idx = Canopy::Operator::Scalar::index( j, k );

                /* Target point 1 calculations */
                // Greengard eq. 3.59
                potential_L +=
                    L_host( idx ) * Kokkos::pow( r, j ) *
                    Canopy::Operator::Scalar::Ynm( j, k, theta, phi );
            }
        }

        // Check the error bounds from eq. 3.61
        double bound = ( q_total / ( c * a_min - a_min ) ) * std::pow( 1.0 / c, p + 1 );
        double error = std::abs( potential_L.real() - potential_direct );
        EXPECT_LE( error, bound ) << "p=" << p
                                   << ": error between local and direct "
                                      "potentials at target point too high.";
    }
}

void testL2LStruct0()
{
    // Create points and q (scalar value)
    const int num_points = 500;
    Kokkos::View<double* [3], TEST_MEMSPACE> cart_coords( "cart_coords",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q( "q", num_points );

    Kokkos::Array<double, 6> coord_bounds = { -8.0, -8.0, -8.0,
                                              -5.0, -5.0, -5.0 };
    fillRandomCoordinates( cart_coords, coord_bounds, 123 );

    Kokkos::Array<double, 2> charge_bounds = { -3.0, 2.0 };
    fillRandomScalar( q, charge_bounds, 321 );

    // Expansion center
    Kokkos::Array<double, 3> m_center = { -5.5, -5.4, -5.3 };
    double rho_m, alpha_m, beta_m; 
    Canopy::Operator::cart2sph( m_center[0], m_center[1], m_center[2], rho_m, alpha_m,
                              beta_m );
        
    // Shifted local center (within radius 'a' of origin): From new local center
    // to old local center.
    Kokkos::Array<double, 3> X_0 = {1.1, -0.8, -1.0};
    double rho, alpha, beta; 
    Canopy::Operator::cart2sph( X_0[0], X_0[1], X_0[2], rho, alpha,
                              beta );

    // Target point P near origin (within radius 'a' of origin)
    double Px = 0.7, Py = -1.9, Pz = -1.1;
    double r, theta, phi;
    Canopy::Operator::cart2sph( Px, Py, Pz, r, theta, phi );

    // P + X_0
    double r_p, theta_p, phi_p;
    Canopy::Operator::cart2sph( Px + X_0[0], Py + X_0[1], Pz + X_0[2], r_p, theta_p, phi_p );

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
        dx = Px - cart_coords_host( i, 0 );
        dy = Py - cart_coords_host( i, 1 );
        dz = Pz - cart_coords_host( i, 2 );
        dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        potential_direct += q_host( i ) / dist;
    }

    // Theorem 3.5.5 requires c > 1 and rho > (c+1)*a.
    // Solve for c, getting c < (rho - a) / a for a > 0.
    ASSERT_GT( a, 0.0 );
    double c = ( rho_m - a ) / a;
    ASSERT_GT( c, 1.0 )
        << "Error: rho must be greater than (c+1)*a for theory to be valid.";

    // Target point and shifted local center must be within radius a of origin.
    ASSERT_LT( r, a ) << "Error: Target point 1 must be within distance 'a' "
                          "from local center for theory to be valid.";
    ASSERT_LT( rho, a ) << "Error: Shifted local center must be within distance 'a' "
                          "from original local center for theory to be valid.";

    // Loop over truncation degree
    for ( int p = 1; p <= 9; ++p )
    {
        Canopy::Operator::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );
        p2m( cart_coords, q, num_points, m_center );

        // Convert multipoles to locals
        Canopy::Operator::Scalar::M2L<TEST_MEMSPACE, TEST_EXECSPACE> m2l( p );
        m2l( p2m.coefficients(), m_center );
    
        // Translate locals
        Canopy::Operator::Scalar::L2L<TEST_MEMSPACE, TEST_EXECSPACE> l2l( p );
        l2l( m2l.coefficients(), X_0 );

        // Copy to host
        auto L_orig_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                           m2l.coefficients() );
        auto L_shift_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                           l2l.coefficients() );

        // Perform local to potential conversion to calculate potential at
        // target. Equation 3.59 in Greengard
        using cdouble = Kokkos::complex<double>;
        cdouble potential_L = 0.0;
        cdouble potential_shift = 0.0;
        for ( int j = 0; j <= p; ++j )
        {
            for ( int k = -j; k <= j; ++k )
            {
                int idx = Canopy::Operator::Scalar::index( j, k );

                /* Target point 1 calculations */
                // Greengard eq. 3.59
                potential_L +=
                    L_orig_host( idx ) * Kokkos::pow( r, j ) *
                    Canopy::Operator::Scalar::Ynm( j, k, theta, phi );
                potential_shift +=
                    L_shift_host( idx ) * Kokkos::pow( r_p, j ) *
                    Canopy::Operator::Scalar::Ynm( j, k, theta_p, phi_p );
            }
        }

        // Check the error bounds from eq. 3.61
        double bound = ( q_total / ( c * a - a ) ) * std::pow( 1.0 / c, p + 1 );
        double error = std::abs( potential_L.real() - potential_direct );
        EXPECT_LE( error, bound ) << "p=" << p
                                   << ": error between local and direct "
                                      "potentials at target point 1 too high.";
        // printf("p=%d: D1: %0.5lf, L: %0.5lf, LS: %0.5lf\n", p,
        //     potential_direct, potential_L.real(), potential_shift.real());
    }
}

/**
 * Mimic FMM usage of locals:
 * Translate multipoles from a far source to locals centered around a box.
 * Translate locals to a child box inside of the box.
 * Test that the direct potential at a target point inside the box matches the
 * potential calculated with the translated locals.
 */
void testL2LStruct1()
{
    // Create points and q (scalar value)
    const int num_points = 500;
    Kokkos::View<double* [3], TEST_MEMSPACE> cart_coords( "cart_coords",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q( "q", num_points );

    Kokkos::Array<double, 6> coord_bounds = { -8.0, -8.0, -8.0,
                                              -5.0, -5.0, -5.0 };
    fillRandomCoordinates( cart_coords, coord_bounds, 546 );

    Kokkos::Array<double, 2> charge_bounds = { -3.0, 2.0 };
    fillRandomScalar( q, charge_bounds, 923 );

    // Expansion center
    Kokkos::Array<double, 3> m_center = { -5.5, -5.4, -5.3 };
    double rho_m, alpha_m, beta_m; 
    Canopy::Operator::cart2sph( m_center[0], m_center[1], m_center[2], rho_m, alpha_m,
                              beta_m );
        
    // Shifted local center (within radius 'a' of origin): From new local center
    // to old local center.
    Kokkos::Array<double, 3> X_0 = {1.1, -0.8, -1.0};
    double rho, alpha, beta; 
    Canopy::Operator::cart2sph( X_0[0], X_0[1], X_0[2], rho, alpha,
                              beta );

    // Target point P near origin (within radius 'a' of origin)
    double Px = 0.7, Py = -1.9, Pz = -1.1;
    double r, theta, phi;
    Canopy::Operator::cart2sph( Px, Py, Pz, r, theta, phi );

    // P + X_0
    double r_p, theta_p, phi_p;
    Canopy::Operator::cart2sph( Px + X_0[0], Py + X_0[1], Pz + X_0[2], r_p, theta_p, phi_p );

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
        dx = Px - cart_coords_host( i, 0 );
        dy = Py - cart_coords_host( i, 1 );
        dz = Pz - cart_coords_host( i, 2 );
        dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        potential_direct += q_host( i ) / dist;
    }

    // Theorem 3.5.5 requires c > 1 and rho > (c+1)*a.
    // Solve for c, getting c < (rho - a) / a for a > 0.
    ASSERT_GT( a, 0.0 );
    double c = ( rho_m - a ) / a;
    ASSERT_GT( c, 1.0 )
        << "Error: rho must be greater than (c+1)*a for theory to be valid.";

    // Target point and shifted local center must be within radius a of origin.
    ASSERT_LT( r, a ) << "Error: Target point 1 must be within distance 'a' "
                          "from local center for theory to be valid.";
    ASSERT_LT( rho, a ) << "Error: Shifted local center must be within distance 'a' "
                          "from original local center for theory to be valid.";

    // Loop over truncation degree
    for ( int p = 1; p <= 9; ++p )
    {
        Canopy::Operator::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );
        p2m( cart_coords, q, num_points, m_center );

        // Convert multipoles to locals
        Canopy::Operator::Scalar::M2L<TEST_MEMSPACE, TEST_EXECSPACE> m2l( p );
        m2l( p2m.coefficients(), m_center );
    
        // Translate locals
        Canopy::Operator::Scalar::L2L<TEST_MEMSPACE, TEST_EXECSPACE> l2l( p );
        l2l( m2l.coefficients(), X_0 );

        // Copy to host
        auto L_orig_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                           m2l.coefficients() );
        auto L_shift_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                           l2l.coefficients() );

        // Perform local to potential conversion to calculate potential at
        // target. Equation 3.59 in Greengard
        using cdouble = Kokkos::complex<double>;
        cdouble potential_L = 0.0;
        cdouble potential_shift = 0.0;
        for ( int j = 0; j <= p; ++j )
        {
            for ( int k = -j; k <= j; ++k )
            {
                int idx = Canopy::Operator::Scalar::index( j, k );

                /* Target point 1 calculations */
                // Greengard eq. 3.59
                potential_L +=
                    L_orig_host( idx ) * Kokkos::pow( r, j ) *
                    Canopy::Operator::Scalar::Ynm( j, k, theta, phi );
                potential_shift +=
                    L_shift_host( idx ) * Kokkos::pow( r_p, j ) *
                    Canopy::Operator::Scalar::Ynm( j, k, theta_p, phi_p );
            }
        }

        // Check the error bounds from eq. 3.61
        double bound = ( q_total / ( c * a - a ) ) * std::pow( 1.0 / c, p + 1 );
        double error = std::abs( potential_shift.real() - potential_direct );
        EXPECT_LE( error, bound ) << "p=" << p
                                   << ": error between local and direct "
                                      "potentials at target point 1 too high.";
        
        // Check that potential calculated with translated and original locals
        // match.
        EXPECT_NEAR(potential_L.real(), potential_shift.real(), 0.0000000001);

        // printf("p=%d: D1: %0.5lf, L: %0.5lf, LS: %0.5lf\n", p,
        //     potential_direct, potential_L.real(), potential_shift.real());
    }
}

/**********************
 * Function tests
 *********************/
template <int p>
void testP2MFunc()
{
    // Create points and q (scalar value)
    const int num_points = 50;
    Kokkos::View<double* [3], TEST_MEMSPACE> cart_coords( "cart_coords",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q( "q", num_points );

    Kokkos::Array<double, 6> coord_bounds = { -1.0, -1.0, -1.0, 1.0, 1.0, 1.0 };
    fillRandomCoordinates( cart_coords, coord_bounds, 321 );

    Kokkos::Array<double, 2> charge_bounds = { -10.0, 10.0 };
    fillRandomScalar( q, charge_bounds, 234 );

    // Expansion center
    Kokkos::Array<double, 3> expansion_center = { 0.1, -0.6, 0.3 };

    // Target point P
    double Px = 6.6, Py = -5.1, Pz = 1.9;
    // P relative to the multipole expansion center.
    double r, theta, phi;
    Canopy::Operator::cart2sph( Px - expansion_center[0],
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

    // Multipole view
    Kokkos::View<cdouble[( p + 1 ) * ( p + 1 )], TEST_MEMSPACE> M_view("M_view");
    Kokkos::deep_copy(M_view, cdouble(0.0, 0.0));

    // Convert scalar values to multipoles
    Kokkos::parallel_for(
        "testp2mfunc",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_points ),
        KOKKOS_LAMBDA( const int pid ) {

        // Get position
        Kokkos::Array<double, 3> pos;
        for (int i = 0; i < 3; i++)
            pos[i] = cart_coords(pid, i);
        
        // Get Scalar
        double scalar = q(pid);    

        // Create multipole array
        Kokkos::Array<cdouble, ( p + 1 ) * ( p + 1 )> M;
        for (int i = 0; i < ( p + 1 ) * ( p + 1 ); i++)
            M[i] = cdouble(0.0, 0.0);

        // Compute multipoles
        Canopy::Operator::Scalar::p2m<p>(pos, scalar, expansion_center, M);

        // Add this particle's contribution to the total multipoles
        for (int i = 0; i < ( p + 1 ) * ( p + 1 ); i++)
            Kokkos::atomic_add(&M_view(i), M[i]);
    } );

    auto M_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M_view );

    // Perform multipole to particle conversion to calculate potential at
    // target. Equation 3.36 in Greengard
    cdouble potential_multipole = 0.0;
    for ( int n = 0; n <= p; ++n )
    {
        // double norm = 4 * pi / double( 2 * n + 1 );
        // auto norm = Kokkos::sqrt(( ( 2.0 * n + 1 ) / ( 4.0 * pi ) ));
        for ( int m = -n; m <= n; ++m )
        {
            int idx = Canopy::Operator::Scalar::index( n, m );
            potential_multipole +=
                M_host( idx ) / Kokkos::pow( r, n + 1 ) *
                Canopy::Operator::Scalar::Ynm( n, m, theta, phi );
        }
    }

    double error =
        std::abs( potential_multipole.real() - potential_direct );
    double bound =
        std::pow( max_rho / r, p + 1 ) * std::abs( potential_direct );

    // Check that the error is within 5*bound, which accounts
    // for imprecision due to intermediate rounding.
    EXPECT_NEAR( potential_multipole.real(), potential_direct, 5 * bound )
        << "p=" << p << " multipole=" << potential_multipole.real()
        << " direct=" << potential_direct << " error=" << error << " bound~"
        << bound << std::endl;
}

/**
 * Tests addition and translation of multipole coefficients.
 * Creates two multipole expansions around centers with charges
 * disjunct, well-separated domains. Translations these expansions
 * to center around a new center and then adds these expansions together.
 * Converts the aggregated multipole expansions back to potentials at
 * a target point and compares the result to the directly calculated potential
 * at the target point.
 */
template <int p>
void testM2MFunc()
{
    const int points_per_section = 200;

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

    fillRandomCoordinates( c0, cbounds0, 765 );
    fillRandomCoordinates( c1, cbounds1, 567 );
    fillRandomScalar( q, qbounds, 444 );

    // Expansion center of Q0 in polar coordinates - rho0, alpha0, beta0
    double rho0, alpha0, beta0;
    Canopy::Operator::cart2sph( -q0_center[0], -q0_center[1], -q0_center[2], rho0,
                              alpha0, beta0 );

    // Expansion center of Q1 in polar coordinates - rho1, alpha1, beta1
    double rho1, alpha1, beta1;
    Canopy::Operator::cart2sph( -q1_center[0], -q1_center[1], -q1_center[2], rho1,
                              alpha1, beta1 );

    // Target point - rho0, theta, phi
    double Px = 15.0, Py = -10.0, Pz = 7.0;
    double r, theta, phi;
    Canopy::Operator::cart2sph( Px, Py, Pz, r, theta, phi );

    // (Target point - q0_center) - r_p, theta_p, phi_p
    // double r_p, theta_p, phi_p;
    // Canopy::Operator::cart2sph( Px - q0_center[0],
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

    // Multipole views
    Kokkos::View<cdouble[2][( p + 1 ) * ( p + 1 )], TEST_MEMSPACE> M_orig("M_orig");
    Kokkos::deep_copy(M_orig, cdouble(0.0, 0.0));
    Kokkos::View<cdouble[( p + 1 ) * ( p + 1 )], TEST_MEMSPACE> M_trans("M_trans");
    Kokkos::deep_copy(M_trans, cdouble(0.0, 0.0));

    // Convert scalar values to multipoles and translate multipoles for first section
    Kokkos::parallel_for(
        "testm2mfunc",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, points_per_section * 2 ),
        KOKKOS_LAMBDA( const int pid ) {

        // Get position
        Kokkos::Array<double, 3> pos;
        for (int i = 0; i < 3; i++)
            pos[i] = c0(pid, i);
        
        // Get Scalar
        double scalar = q0(pid);    

        // Create multipole array
        Kokkos::Array<cdouble, ( p + 1 ) * ( p + 1 )> M;
        for (int i = 0; i < ( p + 1 ) * ( p + 1 ); i++)
            M[i] = cdouble(0.0, 0.0);

        // Compute multipoles
        int index = (pid < points_per_section) ? 0 : 1;
        if (index == 0)
            Canopy::Operator::Scalar::p2m<p>(pos, scalar, q0_center, M);
        else if (index == 1)
            Canopy::Operator::Scalar::p2m<p>(pos, scalar, q1_center, M);

        // Add this particle's contribution to the total multipoles
        for (int i = 0; i < ( p + 1 ) * ( p + 1 ); i++)
            Kokkos::atomic_add(&M_orig(index, i), M[i]);
    } );

    // Translate multipoles from O0 to be centered around the origin.
    const int num_sections = 2;
    Kokkos::parallel_for(
        "testm2mfunc1",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_sections ),
        KOKKOS_LAMBDA( const int index ) {
                
        // Create multipoles arrays
        Kokkos::Array<cdouble, ( p + 1 ) * ( p + 1 )> M_orig_array;
        Kokkos::Array<cdouble, ( p + 1 ) * ( p + 1 )> M_trans_array;
        for (int i = 0; i < ( p + 1 ) * ( p + 1 ); i++)
        {
            M_trans_array[i] = cdouble(0.0, 0.0);
            M_orig_array[i] = M_orig(index, i);
        }
            
        // Compute multipoles
        if (index == 0)
            Canopy::Operator::Scalar::m2m<p>(M_orig_array, q0_center, M_trans_array);
        else if (index == 1)
            Canopy::Operator::Scalar::m2m<p>(M_orig_array, q1_center, M_trans_array);

        // Add this multipole's contribution to the total multipoles
        for (int i = 0; i < ( p + 1 ) * ( p + 1 ); i++)
            Kokkos::atomic_add(&M_trans(i), M_trans_array[i]);
    } );

    // Get translated multipole coefficients
    auto M_trans_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                    M_trans );

    // Compute potential at P using M_trans
    cdouble potential_M = 0.0;
    for ( int j = 0; j <= p; ++j )
    {
        for ( int k = -j; k <= j; ++k )
        {
            int idx = Canopy::Operator::Scalar::index( j, k );
            potential_M += M_trans_h( idx ) / Kokkos::pow( r, j + 1 ) *
                            Canopy::Operator::Scalar::Ynm( j, k, theta, phi );
        }
    }

    // Check error between translated multipole and direct potentials
    // The error is already mathematically checked in testM2MStruct0,
    // so here we just make sure they are close to each other.
    double error = Kokkos::pow( 10, -p + 3 );
    EXPECT_NEAR( potential_direct, potential_M.real(), error )
        << "p=" << p
        << ": error between (shifted and added) and (direct potential) "
            "calculations too high.";
}

template <int p>
void testM2LFunc()
{
    // Create points and q (scalar value)
    const int num_points = 500;
    Kokkos::View<double* [3], TEST_MEMSPACE> cart_coords( "cart_coords",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q( "q", num_points );

    Kokkos::Array<double, 6> coord_bounds = { -8.0, -8.0, -8.0,
                                              -5.0, -5.0, -5.0 };
    fillRandomCoordinates( cart_coords, coord_bounds, 123 );

    Kokkos::Array<double, 2> charge_bounds = { -3.0, 2.0 };
    fillRandomScalar( q, charge_bounds, 999 );

    // Expansion center
    Kokkos::Array<double, 3> center = { -5.5, -5.4, -5.3 };
    double rho, alpha, beta;
    Canopy::Operator::cart2sph( center[0], center[1], center[2], rho, alpha,
                              beta );

    // First target point near origin (within radius 'a' of origin)
    double Px1 = 0.2, Py1 = -0.1, Pz1 = -0.5;
    double r1, theta1, phi1, r_d1, theta_d1, phi_d1;
    Canopy::Operator::cart2sph( Px1 - center[0], Py1 - center[1], Pz1 - center[2],
                              r_d1, theta_d1, phi_d1 );
    Canopy::Operator::cart2sph( Px1, Py1, Pz1, r1, theta1, phi1 );

    // Second target point near origin (within radius 'a' of origin)
    double Px2 = -0.4, Py2 = 0.3, Pz2 = -0.2;
    double r2, theta2, phi2, r_d2, theta_d2, phi_d2;
    Canopy::Operator::cart2sph( Px2 - center[0], Py2 - center[1], Pz2 - center[2],
                              r_d2, theta_d2, phi_d2 );
    Canopy::Operator::cart2sph( Px2, Py2, Pz2, r2, theta2, phi2 );

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

    // Since we use compile-time sized arrays here, p must given at compile
    // time.
    Canopy::Operator::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );
    p2m( cart_coords, q, num_points, center );
    auto O_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                       p2m.coefficients() );

    // Create O and L arrays
    Kokkos::Array<cdouble, ( p + 1 ) * ( p + 1 )> O;
    Kokkos::Array<cdouble, ( p + 1 ) * ( p + 1 )> L;

    // Copy values into array
    for ( std::size_t i = 0; i < O_host.extent( 0 ); i++ )
        O[i] = O_host( i );

    // Convert multipoles to locals using function
    Canopy::Operator::Scalar::m2l<p>( O, L, center );

    // Perform local to potential conversion to calculate potential at
    // target. Equation 3.59 in Greengard
    cdouble potential_L1 = 0.0;
    cdouble potential_O1 = 0.0;
    cdouble potential_L2 = 0.0;
    cdouble potential_O2 = 0.0;
    for ( int j = 0; j <= p; ++j )
    {
        for ( int k = -j; k <= j; ++k )
        {
            int idx = Canopy::Operator::Scalar::index( j, k );

            /* Target point 1 calculations */
            // Greengard eq. 3.59
            potential_L1 += L[idx] * Kokkos::pow( r1, j ) *
                            Canopy::Operator::Scalar::Ynm( j, k, theta1, phi1 );
            // Greengard eq. 3.36
            potential_O1 +=
                O_host( idx ) / Kokkos::pow( r_d1, j + 1 ) *
                Canopy::Operator::Scalar::Ynm( j, k, theta_d1, phi_d1 );

            /* Target point 2 calculations */
            // Greengard eq. 3.59
            potential_L2 += L[idx] * Kokkos::pow( r2, j ) *
                            Canopy::Operator::Scalar::Ynm( j, k, theta2, phi2 );
            // Greengard eq. 3.36
            potential_O2 +=
                O_host( idx ) / Kokkos::pow( r_d2, j + 1 ) *
                Canopy::Operator::Scalar::Ynm( j, k, theta_d2, phi_d2 );
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

/**
 * Mimic FMM usage of locals:
 * Translate multipoles from a far source to locals centered around a box.
 * Translate locals to a child box inside of the box.
 * Test that the direct potential at a target point inside the box matches the
 * potential calculated with the translated locals.
 * Uses device-callable L2L function.
 */
template <int p>
void testL2LFunc()
{
    // Create points and q (scalar value)
    const int num_points = 500;
    Kokkos::View<double* [3], TEST_MEMSPACE> cart_coords( "cart_coords",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q( "q", num_points );

    Kokkos::Array<double, 6> coord_bounds = { -8.0, -8.0, -8.0,
                                              -5.0, -5.0, -5.0 };
    fillRandomCoordinates( cart_coords, coord_bounds, 546 );

    Kokkos::Array<double, 2> charge_bounds = { -3.0, 2.0 };
    fillRandomScalar( q, charge_bounds, 923 );

    // Expansion center
    Kokkos::Array<double, 3> m_center = { -5.5, -5.4, -5.3 };
    double rho_m, alpha_m, beta_m; 
    Canopy::Operator::cart2sph( m_center[0], m_center[1], m_center[2], rho_m, alpha_m,
                              beta_m );
        
    // Shifted local center (within radius 'a' of origin): From new local center
    // to old local center.
    Kokkos::Array<double, 3> X_0 = {1.1, -0.8, -1.0};
    double rho, alpha, beta; 
    Canopy::Operator::cart2sph( X_0[0], X_0[1], X_0[2], rho, alpha,
                              beta );

    // Target point P near origin (within radius 'a' of origin)
    double Px = 0.7, Py = -1.9, Pz = -1.1;
    double r, theta, phi;
    Canopy::Operator::cart2sph( Px, Py, Pz, r, theta, phi );

    // P + X_0
    double r_p, theta_p, phi_p;
    Canopy::Operator::cart2sph( Px + X_0[0], Py + X_0[1], Pz + X_0[2], r_p, theta_p, phi_p );

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
        dx = Px - cart_coords_host( i, 0 );
        dy = Py - cart_coords_host( i, 1 );
        dz = Pz - cart_coords_host( i, 2 );
        dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        potential_direct += q_host( i ) / dist;
    }

    // Theorem 3.5.5 requires c > 1 and rho > (c+1)*a.
    // Solve for c, getting c < (rho - a) / a for a > 0.
    ASSERT_GT( a, 0.0 );
    double c = ( rho_m - a ) / a;
    ASSERT_GT( c, 1.0 )
        << "Error: rho must be greater than (c+1)*a for theory to be valid.";

    // Target point and shifted local center must be within radius a of origin.
    ASSERT_LT( r, a ) << "Error: Target point 1 must be within distance 'a' "
                          "from local center for theory to be valid.";
    ASSERT_LT( rho, a ) << "Error: Shifted local center must be within distance 'a' "
                          "from original local center for theory to be valid.";

    Canopy::Operator::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );
    p2m( cart_coords, q, num_points, m_center );

    // Convert multipoles to locals
    Canopy::Operator::Scalar::M2L<TEST_MEMSPACE, TEST_EXECSPACE> m2l( p );
    m2l( p2m.coefficients(), m_center );

    // Copy to host
    auto L_orig_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                        m2l.coefficients() );

    // Create L_orig and L_trans arrays
    Kokkos::Array<cdouble, ( p + 1 ) * ( p + 1 )> L_orig;
    Kokkos::Array<cdouble, ( p + 1 ) * ( p + 1 )> L_trans;

    // Copy values into array
    for ( std::size_t i = 0; i < L_orig_host.extent( 0 ); i++ )
        L_orig[i] = L_orig_host( i );

    // Translate locals
    Canopy::Operator::Scalar::l2l<p>(L_orig, L_trans, X_0);

    // Perform local to potential conversion to calculate potential at
    // target. Equation 3.59 in Greengard
    using cdouble = Kokkos::complex<double>;
    cdouble potential_L = 0.0;
    cdouble potential_shift = 0.0;
    for ( int j = 0; j <= p; ++j )
    {
        for ( int k = -j; k <= j; ++k )
        {
            int idx = Canopy::Operator::Scalar::index( j, k );

            /* Target point 1 calculations */
            // Greengard eq. 3.59
            potential_L +=
                L_orig_host( idx ) * Kokkos::pow( r, j ) *
                Canopy::Operator::Scalar::Ynm( j, k, theta, phi );
            potential_shift +=
                L_trans[ idx ] * Kokkos::pow( r_p, j ) *
                Canopy::Operator::Scalar::Ynm( j, k, theta_p, phi_p );
        }
    }

    // Check the error bounds from eq. 3.61
    double bound = ( q_total / ( c * a - a ) ) * std::pow( 1.0 / c, p + 1 );
    double error = std::abs( potential_shift.real() - potential_direct );
    EXPECT_LE( error, bound ) << "p=" << p
                                << ": error between local and direct "
                                    "potentials at target point 1 too high.";
    
    // Check that potential calculated with translated and original locals
    // match.
    EXPECT_NEAR(potential_L.real(), potential_shift.real(), 0.0000000001);

    // printf("p=%d: D: %0.5lf, L: %0.5lf, LS: %0.5lf\n", p,
    //     potential_direct, potential_L.real(), potential_shift.real());
}

// Gradient test
void testGrad()
{
    constexpr double h = 1e-7;
    constexpr double tol = 1e-8;

    auto check_close = [&](const char* name, double got, double ref)
    {
        double err = Kokkos::abs(got - ref);
        if (err > tol)
            printf("FAIL %s got=%.15e ref=%.15e err=%.3e\n", name, got, ref, err);
        else
            printf("OK   %s got=%.15e ref=%.15e err=%.3e\n", name, got, ref, err);
    };

    // ---------- Case 1: Phi = z (n=1,m=0), check analytic partials at generic angle ----------
    {
        const double r = 0.7;
        const double theta = 1.1;
        const double phi = 0.9;

        const int n = 1, m = 0;
        const cdouble L_10(1.0, 0.0);
        const cdouble Y = Canopy::Operator::Scalar::Ynm(n, m, theta, phi);
        const cdouble val = Kokkos::pow(r, n) * Y;

        const double dr = (L_10 * Canopy::Operator::Scalar::d_dr(r, n, val)).real();
        const double dtheta = (L_10 * Kokkos::pow(r,n) *
                               Canopy::Operator::Scalar::d_dtheta(r, theta, phi, n, m)).real();
        const double dphi = (L_10 * Canopy::Operator::Scalar::d_dphi(m, val)).real();

        // analytic for Phi=z=r cos(theta)
        check_close("Phi=z: d/dr",     dr,     Kokkos::cos(theta));
        check_close("Phi=z: d/dtheta", dtheta, -r * Kokkos::sin(theta));
        check_close("Phi=z: d/dphi",   dphi,   0.0);

        // optional: check Cartesian gradient equals (0,0,1)
        Kokkos::Array<double,3> partials = {dr, dtheta, dphi};
        auto grad = Canopy::Operator::partials_to_cartesian_gradient(partials, r, theta, phi);
        check_close("Phi=z: gx", grad[0], 0.0);
        check_close("Phi=z: gy", grad[1], 0.0);
        check_close("Phi=z: gz", grad[2], 1.0);
    }

    // ---------- Case 2: finite-difference check for d/dphi and d/dtheta on general (n,m) ----------
    auto fd_check = [&](int n, int m, double r, double theta, double phi)
    {
        const cdouble Y0  = Canopy::Operator::Scalar::Ynm(n, m, theta, phi);
        const cdouble F0  = Kokkos::pow(r,n) * Y0;

        // d/dphi reference
        const cdouble Yp = Canopy::Operator::Scalar::Ynm(n, m, theta, phi + h);
        const cdouble Ym = Canopy::Operator::Scalar::Ynm(n, m, theta, phi - h);
        const cdouble Fp = Kokkos::pow(r,n) * Yp;
        const cdouble Fm = Kokkos::pow(r,n) * Ym;
        const cdouble dphi_fd = (Fp - Fm) * (0.5 / h);

        const cdouble dphi_op = Canopy::Operator::Scalar::d_dphi(m, F0);

        // d/dtheta reference
        const cdouble Ytp = Canopy::Operator::Scalar::Ynm(n, m, theta + h, phi);
        const cdouble Ytm = Canopy::Operator::Scalar::Ynm(n, m, theta - h, phi);
        const cdouble Ftp = Kokkos::pow(r,n) * Ytp;
        const cdouble Ftm = Kokkos::pow(r,n) * Ytm;
        const cdouble dtheta_fd = (Ftp - Ftm) * (0.5 / h);

        // Option 1: operator returns dY/dtheta, so d/dtheta (r^n Y)= r^n * dY/dtheta
        const cdouble dtheta_op = Kokkos::pow(r,n) *
            Canopy::Operator::Scalar::d_dtheta(r, theta, phi, n, m);

        auto err = [](cdouble a, cdouble b){
            return Kokkos::abs(a - b);
        };

        printf("FD (n=%d,m=%d): |dphi_op-dphi_fd|=%.3e  |dtheta_op-dtheta_fd|=%.3e\n",
               n, m, err(dphi_op, dphi_fd), err(dtheta_op, dtheta_fd));
    };

    {
        const double r = 0.8;
        const double theta = 1.0;  // avoid poles
        const double phi = 2.0;

        fd_check(1,  1, r, theta, phi);
        fd_check(2,  1, r, theta, phi);
        fd_check(2, -1, r, theta, phi);
        fd_check(3,  2, r, theta, phi);
    }
}

// Force test
template <int p>
void testForce()
{
    static_assert(p > 0);

    // Create points and q (scalar value)
    const int num_points = 1;
    Kokkos::View<double* [3], TEST_MEMSPACE> cart_coords( "cart_coords",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q( "q", num_points );

    Kokkos::Array<double, 6> coord_bounds = { -8.0, -8.0, -8.0,
                                              -5.0, -5.0, -5.0 };
    fillRandomCoordinates( cart_coords, coord_bounds, 123 );

    Kokkos::Array<double, 2> charge_bounds = { -3.0, 2.0 };
    fillRandomScalar( q, charge_bounds, 999 );

    cart_coords(0, 0) = -7.5;
    cart_coords(0, 1) = 0.0;
    cart_coords(0, 2) = 0.0;
    q(0) = 1.0;

    // Expansion center
    Kokkos::Array<double, 3> center = { -6.5, 0.0, 0.0 };
    double rho, alpha, beta;
    Canopy::Operator::cart2sph( center[0], center[1], center[2], rho, alpha,
                              beta );

    // First target point near origin (within radius 'a' of origin)
    double Px1 = 0.2, Py1 = 0.0, Pz1 = 0.0;
    double charge1 = 1.0;
    double r1, theta1, phi1, r_d1, theta_d1, phi_d1;
    Canopy::Operator::cart2sph( Px1 - center[0], Py1 - center[1], Pz1 - center[2],
                              r_d1, theta_d1, phi_d1 );
    Canopy::Operator::cart2sph( Px1, Py1, Pz1, r1, theta1, phi1 );

    // Second target point near origin (within radius 'a' of origin)
    double Px2 = -0.4, Py2 = 0.3, Pz2 = -0.2;
    double charge2 = -3.3;
    double r2, theta2, phi2, r_d2, theta_d2, phi_d2;
    Canopy::Operator::cart2sph( Px2 - center[0], Py2 - center[1], Pz2 - center[2],
                              r_d2, theta_d2, phi_d2 );
    Canopy::Operator::cart2sph( Px2, Py2, Pz2, r2, theta2, phi2 );

    // Compute a and total charge for error bound. (See figure 3.3)
    // Also compute direct potential
    auto cart_coords_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), cart_coords );
    auto q_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), q );
    double q_total = 0.0;
    double a = 0.0;
    Kokkos::Array<double, 3> force_direct1 = {0.0, 0.0, 0.0};
    Kokkos::Array<double, 3> force_direct2 = {0.0, 0.0, 0.0};
    for ( int i = 0; i < num_points; ++i )
    {
        double dx, dy, dz, dist, dist2, dist_inv, dist_inv3, fp;

        // Radius a
        dx = cart_coords_host( i, 0 ) - center[0];
        dy = cart_coords_host( i, 1 ) - center[1];
        dz = cart_coords_host( i, 2 ) - center[2];
        dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        a = std::max( a, dist );

        // Total sum
        q_total += std::abs( q_host( i ) );

        // Direct force at first target point using coordinates relative to
        // origin.
        dx = Px1 - cart_coords_host( i, 0 );
        dy = Py1 - cart_coords_host( i, 1 );
        dz = Pz1 - cart_coords_host( i, 2 );
        dist2 = dx*dx + dy*dy + dz*dz;
        dist_inv  = 1.0 / Kokkos::sqrt(dist2);
        dist_inv3 = dist_inv * dist_inv * dist_inv;
        fp = charge1 * q(i) * dist_inv3;
        force_direct1[0] += fp * dx;
        force_direct1[1] += fp * dy;
        force_direct1[2] += fp * dz;

        // Direct potential at second target point using coordinates relative to
        // origin.
        dx = Px2 - cart_coords_host( i, 0 );
        dy = Py2 - cart_coords_host( i, 1 );
        dz = Pz2 - cart_coords_host( i, 2 );
        dist2 = dx*dx + dy*dy + dz*dz;
        dist_inv  = 1.0 / Kokkos::sqrt(dist2);
        dist_inv3 = dist_inv * dist_inv * dist_inv;
        fp = charge1 * q(i) * dist_inv3;
        force_direct2[0] += fp * dx;
        force_direct2[1] += fp * dy;
        force_direct2[2] += fp * dz;
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
    // ASSERT_LT( r2, a ) << "Error: Target point 2 must be within distance 'a' "
    //                       "from origin for theory to be valid.";

    // Since we use compile-time sized arrays here, p must given at compile
    // time.
    Canopy::Operator::Scalar::P2M<TEST_MEMSPACE, TEST_EXECSPACE> p2m( p );
    p2m( cart_coords, q, num_points, center );
    auto O_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                       p2m.coefficients() );

    // Create O and L arrays
    Kokkos::Array<cdouble, ( p + 1 ) * ( p + 1 )> O;
    Kokkos::Array<cdouble, ( p + 1 ) * ( p + 1 )> L;

    // Copy values into array
    for ( std::size_t i = 0; i < O_host.extent( 0 ); i++ )
        O[i] = O_host( i );

    // Convert multipoles to locals using function
    Canopy::Operator::Scalar::m2l<p>( O, L, center );

    // Perform local to potential and local to force conversion to calculate 
    // values at target. Equation 3.59 in Greengard and A.11 in Rankin
    Kokkos::Array<double, 3> dPhi1 = {0.0, 0.0, 0.0};
    Kokkos::Array<double, 3> dPhi2 = {0.0, 0.0, 0.0};
    for ( int n = 0; n <= p; n++ )
    {
        for ( int m = -n; m <= n; m++ )
        {
            const int idx = Canopy::Operator::Scalar::index( n, m );
            const cdouble L_nm = L[idx];

            /* Target point 1 calculations */
            const cdouble Y_nm1 = Canopy::Operator::Scalar::Ynm(n, m, theta1, phi1);

            // d/dr
            auto dr = (L_nm * Canopy::Operator::Scalar::d_dr(r1, n, Kokkos::pow( r1, n ) * Y_nm1)).real();
            dPhi1[0] += dr;
            printf("idx %d: dr: adding %.2lf\n", idx, dr);

            // d/dtheta
            auto dtheta = (L_nm * Kokkos::pow( r1, n ) * Canopy::Operator::Scalar::d_dtheta(r1, theta1, phi1, n, m)).real();
            dPhi1[1] += dtheta;
            printf("idx %d: dtheta: adding %.2lf\n", idx, dtheta);

            // d/dphi
            auto dphi = (L_nm * Canopy::Operator::Scalar::d_dphi(m, Kokkos::pow(r1, n) * Y_nm1)).real();
            dPhi1[2] += dphi;
            printf("idx %d: dphi: adding %.2lf\n", idx, dphi);

            /* Target point 2 calculations */
            const cdouble Y_nm2 = Canopy::Operator::Scalar::Ynm(n, m, theta2, phi2);

            // d/dr
            dPhi1[0] += (L_nm * Canopy::Operator::Scalar::d_dr(r2, n, Kokkos::pow( r2, n ) * Y_nm2)).real();

            // d/dtheta
            dPhi1[1] += (L_nm * Kokkos::pow( r2, n ) * Canopy::Operator::Scalar::d_dtheta(r2, theta2, phi2, n, m)).real();

            // d/dphi
            dPhi1[2] += (L_nm * Canopy::Operator::Scalar::d_dphi(m, Kokkos::pow(r2, n) * Y_nm2)).real();
        }
    }

    // Convert to cartesian partials
    auto gradPhi1 = Canopy::Operator::partials_to_cartesian_gradient(dPhi1, r1, theta1, phi1);
    auto gradPhi2 = Canopy::Operator::partials_to_cartesian_gradient(dPhi2, r2, theta2, phi2);

    // Convert to force
    Kokkos::Array<double, 3> F_local1 = {
        -charge1 * gradPhi1[0],
        -charge1 * gradPhi1[1],
        -charge1 * gradPhi1[2]
    };
    Kokkos::Array<double, 3> F_local2 = {
        -charge2 * gradPhi2[0],
        -charge2 * gradPhi2[1],
        -charge2 * gradPhi2[2]
    };

    // Check the error
    Kokkos::Array<double, 3> error1;
    Kokkos::Array<double, 3> error2;
    for (int i = 0; i < 3; i++)
    {
        error1[i] = Kokkos::abs(F_local1[i] - force_direct1[i]);
        error2[i] = Kokkos::abs(F_local2[i] - force_direct2[i]);
    }
    
    printf("p=%d: 1: FE: (%0.5lf, %0.5lf, %0.5lf), FL: (%0.5lf, %0.5lf, %0.5lf)\n", p,
        force_direct1[0], force_direct1[1], force_direct1[2],
        F_local1[0], F_local1[1], F_local1[2]);
    // printf("p=%d: 2: FE: (%0.5lf, %0.5lf, %0.5lf), FL: (%0.5lf, %0.5lf, %0.5lf)\n", p,
    //     force_direct2[0], force_direct2[1], force_direct2[2],
    //     F_local2[0], F_local2[1], F_local2[2]);
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
// Struct tests
TEST( Struct, testP2MStruct0 ) { testP2MStruct0(); }
TEST( Struct, testM2MStruct0 ) { testM2MStruct0(); }
TEST( Struct, testM2MStruct1 ) { testM2MStruct1(); }
TEST( Struct, testM2LStruct0 ) { testM2LStruct0(); }
TEST( Struct, testM2LStruct1 ) { testM2LStruct1(); }
TEST( Struct, testL2LStruct0 ) { testL2LStruct0(); }
TEST( Struct, testL2LStruct1 ) { testL2LStruct1(); }

/*********************************
 * Device callable function tests
 ********************************/

 // P2M
TEST( Func, testP2MFunc1 ) { testP2MFunc<1>(); }
TEST( Func, testP2MFunc3 ) { testP2MFunc<3>(); }
TEST( Func, testP2MFunc5 ) { testP2MFunc<5>(); }
TEST( Func, testP2MFunc9 ) { testP2MFunc<9>(); }

// M2M
TEST( Func, testM2MFunc1 ) { testM2MFunc<1>(); }
TEST( Func, testM2MFunc3 ) { testM2MFunc<3>(); }
TEST( Func, testM2MFunc5 ) { testM2MFunc<5>(); }
TEST( Func, testM2MFunc9 ) { testM2MFunc<9>(); }

// M2L
TEST( Func, testM2LFunc1 ) { testM2LFunc<1>(); }
TEST( Func, testM2LFunc3 ) { testM2LFunc<3>(); }
TEST( Func, testM2LFunc5 ) { testM2LFunc<5>(); }
TEST( Func, testM2LFunc9 ) { testM2LFunc<9>(); }

// L2L
TEST( Func, testL2LFunc1 ) { testL2LFunc<1>(); }
TEST( Func, testL2LFunc3 ) { testL2LFunc<3>(); }
TEST( Func, testL2LFunc5 ) { testL2LFunc<5>(); }
TEST( Func, testL2LFunc9 ) { testL2LFunc<9>(); }

/******************************************
 * Test gradient and force computations
 *****************************************/
TEST(Gradient, testGrad) { testGrad(); }
// TEST(Force, testForce1 ) { testForce<1>(); }

//---------------------------------------------------------------------------//

} // end namespace Test
