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

#include "Canopy_LaplaceKernel.hpp"
#include "Canopy_SphericalCoefficients.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

namespace Test
{
//---------------------------------------------------------------------------//

using namespace Canopy;

namespace LaplaceTest
{

static constexpr int P_ORDER = 6;
using Kernel    = LaplaceKernel<double, P_ORDER>;
using complex   = Kokkos::complex<double>;
using CoeffView = Kokkos::View<complex*, TEST_MEMSPACE>;
using CoeffView2D = Kokkos::View<complex**, TEST_MEMSPACE>;

} // namespace LaplaceTest

//---------------------------------------------------------------------------//
/**
 * Verify coeff_index(n,m) produces a dense, injective mapping into
 * [0, num_coeffs_per_cell) and that num_coeffs_symmetric matches the kernel
 * constant.
 */
TEST( LaplaceKernel, testCoeffIndexing )
{
    using namespace LaplaceTest;

    EXPECT_EQ( num_coeffs_symmetric( 0 ), 1 );
    EXPECT_EQ( num_coeffs_symmetric( 1 ), 3 );
    EXPECT_EQ( num_coeffs_symmetric( 2 ), 6 );
    EXPECT_EQ( num_coeffs_symmetric( P_ORDER ), Kernel::num_coeffs_per_cell );

    const int N = Kernel::num_coeffs_per_cell;
    std::vector<bool> seen( N, false );

    for ( int n = 0; n <= P_ORDER; n++ )
    {
        for ( int m = 0; m <= n; m++ )
        {
            const int idx = coeff_index( n, m );
            EXPECT_GE( idx, 0 ) << "negative index at (" << n << "," << m << ")";
            EXPECT_LT( idx, N ) << "out-of-range index at (" << n << "," << m << ")";
            EXPECT_FALSE( seen[idx] ) << "duplicate index at (" << n << "," << m << ")";
            seen[idx] = true;
        }
    }

    for ( int i = 0; i < N; i++ )
        EXPECT_TRUE( seen[i] ) << "unused coefficient slot " << i;
}

//---------------------------------------------------------------------------//
/**
 * Verify A_{n,m} values at analytically-known points.
 *
 *   A_{0,0}  =  1 / sqrt(0! * 0!) = 1
 *   A_{1,0}  = -1 / sqrt(1! * 1!) = -1
 *   A_{1,1}  = -1 / sqrt(0! * 2!) = -1/sqrt(2)   (|m|=1 used)
 *   A_{1,-1} = same (formula uses |m|)
 *   Out-of-range (|m| > n) -> 0
 */
TEST( LaplaceKernel, testACoefficients )
{
    EXPECT_NEAR( A_coeff<double>( 0,  0 ),  1.0,                       1e-14 );
    EXPECT_NEAR( A_coeff<double>( 1,  0 ), -1.0,                       1e-14 );
    EXPECT_NEAR( A_coeff<double>( 1,  1 ), -1.0 / std::sqrt( 2.0 ),    1e-14 );
    EXPECT_NEAR( A_coeff<double>( 1, -1 ), -1.0 / std::sqrt( 2.0 ),    1e-14 );
    EXPECT_EQ(   A_coeff<double>( 1,  2 ),  0.0 );
}

//---------------------------------------------------------------------------//
/**
 * Verify Y_{n,m} on the z-axis (theta = 0).
 *
 * At theta = 0, cos(theta) = 1. The associated Legendre polynomial
 * P_n^m(1) = 0 for m > 0 and P_n^0(1) = 1. The normalization factor
 * sqrt((n-0)!/(n+0)!) = 1, so Y_{n,0}(0, phi) = 1 and Y_{n,m}(0, phi) = 0
 * for m != 0.
 */
TEST( LaplaceKernel, testYnmOnZAxis )
{
    using namespace LaplaceTest;

    const double theta = 0.0;
    const double phi   = 0.0;

    for ( int n = 0; n <= P_ORDER; n++ )
    {
        complex y0 = Ynm<double>( n, 0, theta, phi );
        EXPECT_NEAR( y0.real(), 1.0, 1e-12 ) << "Y_{" << n << ",0} real";
        EXPECT_NEAR( y0.imag(), 0.0, 1e-12 ) << "Y_{" << n << ",0} imag";

        for ( int m = 1; m <= n; m++ )
        {
            complex yp = Ynm<double>( n,  m, theta, phi );
            EXPECT_NEAR( yp.real(), 0.0, 1e-12 ) << "Y_{" << n << "," <<  m << "} real";
            EXPECT_NEAR( yp.imag(), 0.0, 1e-12 ) << "Y_{" << n << "," <<  m << "} imag";

            complex yn = Ynm<double>( n, -m, theta, phi );
            EXPECT_NEAR( yn.real(), 0.0, 1e-12 ) << "Y_{" << n << "," << -m << "} real";
            EXPECT_NEAR( yn.imag(), 0.0, 1e-12 ) << "Y_{" << n << "," << -m << "} imag";
        }
    }
}

//---------------------------------------------------------------------------//
/**
 * P2M for a single particle on the z-axis has a closed-form result.
 *
 * Particle at (0, 0, dz) relative to the cell center:
 *   rho = dz, theta = 0, phi = 0
 *   Y_{n,-m}(0, phi) = 1 if m=0, else 0
 *
 * So:  M_{n,0} = q * dz^n   and   M_{n,m} = 0  for m > 0.
 */
void testP2MSingleParticleOnAxis()
{
    using namespace LaplaceTest;

    const double q  = 2.5;
    const double dz = 0.7;

    CoeffView M_dev( "M", Kernel::num_coeffs_per_cell );
    Kokkos::deep_copy( M_dev, complex( 0.0, 0.0 ) );

    Kokkos::parallel_for(
        "P2M_axis",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 1 ),
        KOKKOS_LAMBDA( int ) {
            Kernel::p2m_contribution( q, 0.0, 0.0, dz, M_dev );
        } );
    Kokkos::fence();

    auto h_M = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M_dev );

    double rho_n = 1.0;
    for ( int n = 0; n <= P_ORDER; n++ )
    {
        EXPECT_NEAR( h_M( coeff_index( n, 0 ) ).real(), q * rho_n, 1e-12 )
            << "M_{" << n << ",0} real, n=" << n;
        EXPECT_NEAR( h_M( coeff_index( n, 0 ) ).imag(), 0.0, 1e-12 )
            << "M_{" << n << ",0} imag, n=" << n;

        for ( int m = 1; m <= n; m++ )
        {
            EXPECT_NEAR( h_M( coeff_index( n, m ) ).real(), 0.0, 1e-12 )
                << "M_{" << n << "," << m << "} real";
            EXPECT_NEAR( h_M( coeff_index( n, m ) ).imag(), 0.0, 1e-12 )
                << "M_{" << n << "," << m << "} imag";
        }

        rho_n *= dz;
    }
}
TEST( LaplaceKernel, testP2MSingleParticleOnAxis )
{
    testP2MSingleParticleOnAxis();
}

//---------------------------------------------------------------------------//
/**
 * P2M reference comparison for a random particle cloud.
 *
 * Compute P2M with the device kernel (atomics over RangePolicy) and compare
 * to a CPU reference that applies the same formula serially. This checks
 * the indexing, normalization, and atomic accumulation without any tree.
 */
void testP2MRefComparison()
{
    using namespace LaplaceTest;

    const int    N_PARTICLES     = 200;
    const double CELL_HALF_WIDTH = 0.5;

    std::mt19937 gen( 1234 );
    std::uniform_real_distribution<double> pos_dist( -CELL_HALF_WIDTH,
                                                      CELL_HALF_WIDTH );
    std::uniform_real_distribution<double> q_dist( -2.0, 2.0 );

    std::vector<double> px( N_PARTICLES ), py( N_PARTICLES ),
                        pz( N_PARTICLES ), q( N_PARTICLES );
    for ( int i = 0; i < N_PARTICLES; i++ )
    {
        px[i] = pos_dist( gen );
        py[i] = pos_dist( gen );
        pz[i] = pos_dist( gen );
        q[i]  = q_dist( gen );
    }

    // Upload to device
    Kokkos::View<double*, TEST_MEMSPACE> d_px( "px", N_PARTICLES );
    Kokkos::View<double*, TEST_MEMSPACE> d_py( "py", N_PARTICLES );
    Kokkos::View<double*, TEST_MEMSPACE> d_pz( "pz", N_PARTICLES );
    Kokkos::View<double*, TEST_MEMSPACE> d_q ( "q",  N_PARTICLES );
    {
        auto h_px = Kokkos::create_mirror_view( d_px );
        auto h_py = Kokkos::create_mirror_view( d_py );
        auto h_pz = Kokkos::create_mirror_view( d_pz );
        auto h_q  = Kokkos::create_mirror_view( d_q  );
        for ( int i = 0; i < N_PARTICLES; i++ )
        {
            h_px( i ) = px[i]; h_py( i ) = py[i];
            h_pz( i ) = pz[i]; h_q ( i ) = q[i];
        }
        Kokkos::deep_copy( d_px, h_px ); Kokkos::deep_copy( d_py, h_py );
        Kokkos::deep_copy( d_pz, h_pz ); Kokkos::deep_copy( d_q,  h_q  );
    }

    CoeffView M_dev( "M", Kernel::num_coeffs_per_cell );
    Kokkos::deep_copy( M_dev, complex( 0.0, 0.0 ) );

    Kokkos::parallel_for(
        "P2M_cloud",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, N_PARTICLES ),
        KOKKOS_LAMBDA( int p ) {
            Kernel::p2m_contribution( d_q( p ), d_px( p ), d_py( p ),
                                      d_pz( p ), M_dev );
        } );
    Kokkos::fence();

    auto h_M = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M_dev );

    // CPU reference
    std::vector<complex> M_ref( Kernel::num_coeffs_per_cell, complex( 0.0, 0.0 ) );
    for ( int p = 0; p < N_PARTICLES; p++ )
    {
        const double rho   = std::sqrt( px[p]*px[p] + py[p]*py[p] + pz[p]*pz[p] );
        const double theta = ( rho > 0.0 ) ? std::acos( pz[p] / rho ) : 0.0;
        const double phi   = std::atan2( py[p], px[p] );

        double rho_n = 1.0;
        for ( int n = 0; n <= P_ORDER; n++ )
        {
            for ( int m = 0; m <= n; m++ )
            {
                M_ref[coeff_index( n, m )] +=
                    q[p] * rho_n * Ynm<double>( n, -m, theta, phi );
            }
            rho_n *= rho;
        }
    }

    double max_rel_err = 0.0;
    for ( int idx = 0; idx < Kernel::num_coeffs_per_cell; idx++ )
    {
        const complex diff    = h_M( idx ) - M_ref[idx];
        const double abs_err  = std::sqrt( diff.real()*diff.real() +
                                           diff.imag()*diff.imag() );
        const double ref_mag  = std::sqrt( M_ref[idx].real()*M_ref[idx].real() +
                                           M_ref[idx].imag()*M_ref[idx].imag() );
        const double rel_err  = ( ref_mag > 1e-14 ) ? abs_err / ref_mag : abs_err;
        if ( rel_err > max_rel_err )
            max_rel_err = rel_err;
    }

    EXPECT_LT( max_rel_err, 1e-10 )
        << "P2M device result deviates from CPU reference; "
           "max relative error = " << max_rel_err;
}
TEST( LaplaceKernel, testP2MRefComparison )
{
    testP2MRefComparison();
}

//---------------------------------------------------------------------------//
/**
 * M2M translation exactness: P2M-to-child + M2M == direct P2M-to-parent.
 *
 * Per the Greengard M2M theorem, translating a multipole computed at a child
 * center to a parent center gives the same coefficients as a direct P2M at
 * the parent center. This must hold exactly for any P, any translation
 * vector, and any particle positions inside the child cell.
 */
void testM2MTranslationVsDirectP2M()
{
    using namespace LaplaceTest;

    // Child center offset from parent center
    const double tx = 0.25, ty = 0.15, tz = -0.10;
    const double child_hw = 0.25;
    const int N_PARTICLES = 50;

    std::mt19937 gen( 5678 );
    std::uniform_real_distribution<double> pos_dist( -child_hw, child_hw );
    std::uniform_real_distribution<double> q_dist( -1.0, 1.0 );

    // Particle positions relative to child center
    std::vector<double> lx( N_PARTICLES ), ly( N_PARTICLES ),
                        lz( N_PARTICLES ), q( N_PARTICLES );
    for ( int i = 0; i < N_PARTICLES; i++ )
    {
        lx[i] = pos_dist( gen ); ly[i] = pos_dist( gen );
        lz[i] = pos_dist( gen ); q[i]  = q_dist( gen );
    }

    // CPU reference: direct P2M to parent center (offset = child_c + local)
    std::vector<complex> M_direct( Kernel::num_coeffs_per_cell,
                                    complex( 0.0, 0.0 ) );
    for ( int p = 0; p < N_PARTICLES; p++ )
    {
        const double dx    = lx[p] + tx;
        const double dy    = ly[p] + ty;
        const double dz    = lz[p] + tz;
        const double rho   = std::sqrt( dx*dx + dy*dy + dz*dz );
        const double theta = ( rho > 0.0 ) ? std::acos( dz / rho ) : 0.0;
        const double phi   = std::atan2( dy, dx );

        double rho_n = 1.0;
        for ( int n = 0; n <= P_ORDER; n++ )
        {
            for ( int m = 0; m <= n; m++ )
            {
                M_direct[coeff_index( n, m )] +=
                    q[p] * rho_n * Ynm<double>( n, -m, theta, phi );
            }
            rho_n *= rho;
        }
    }

    // Device path: P2M to child, then M2M translate to parent
    Kokkos::View<double*, TEST_MEMSPACE> d_lx( "lx", N_PARTICLES );
    Kokkos::View<double*, TEST_MEMSPACE> d_ly( "ly", N_PARTICLES );
    Kokkos::View<double*, TEST_MEMSPACE> d_lz( "lz", N_PARTICLES );
    Kokkos::View<double*, TEST_MEMSPACE> d_q ( "q",  N_PARTICLES );
    {
        auto h_lx = Kokkos::create_mirror_view( d_lx );
        auto h_ly = Kokkos::create_mirror_view( d_ly );
        auto h_lz = Kokkos::create_mirror_view( d_lz );
        auto h_q  = Kokkos::create_mirror_view( d_q );
        for ( int i = 0; i < N_PARTICLES; i++ )
        {
            h_lx(i) = lx[i]; h_ly(i) = ly[i];
            h_lz(i) = lz[i]; h_q(i)  = q[i];
        }
        Kokkos::deep_copy( d_lx, h_lx ); Kokkos::deep_copy( d_ly, h_ly );
        Kokkos::deep_copy( d_lz, h_lz ); Kokkos::deep_copy( d_q,  h_q  );
    }

    // Step 1: P2M into child (stored as 2D view, 1 cell)
    CoeffView2D M_child_2d( "M_child", 1, Kernel::num_coeffs_per_cell );
    Kokkos::deep_copy( M_child_2d, complex( 0.0, 0.0 ) );

    Kokkos::parallel_for(
        "P2M_child",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, N_PARTICLES ),
        KOKKOS_LAMBDA( int p ) {
            auto M_out = Kokkos::subview( M_child_2d, 0, Kokkos::ALL );
            Kernel::p2m_contribution( d_q(p), d_lx(p), d_ly(p), d_lz(p),
                                      M_out );
        } );
    Kokkos::fence();

    // Step 2: M2M translate into parent (1D output)
    CoeffView M_parent_dev( "M_parent", Kernel::num_coeffs_per_cell );
    Kokkos::deep_copy( M_parent_dev, complex( 0.0, 0.0 ) );

    using team_policy = Kokkos::TeamPolicy<TEST_EXECSPACE>;
    Kokkos::parallel_for(
        "M2M_translate",
        team_policy( 1, Kokkos::AUTO ),
        KOKKOS_LAMBDA( const typename team_policy::member_type& team ) {
            Kernel::m2m_translate( team, M_child_2d, tx, ty, tz,
                                   M_parent_dev );
        } );
    Kokkos::fence();

    auto h_M_parent = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), M_parent_dev );

    double max_rel_err = 0.0;
    for ( int idx = 0; idx < Kernel::num_coeffs_per_cell; idx++ )
    {
        const complex diff   = h_M_parent( idx ) - M_direct[idx];
        const double abs_err = std::sqrt( diff.real()*diff.real() +
                                          diff.imag()*diff.imag() );
        const double ref_mag = std::sqrt( M_direct[idx].real()*M_direct[idx].real() +
                                          M_direct[idx].imag()*M_direct[idx].imag() );
        const double rel_err = ( ref_mag > 1e-14 ) ? abs_err / ref_mag : abs_err;
        if ( rel_err > max_rel_err )
            max_rel_err = rel_err;
    }

    EXPECT_LT( max_rel_err, 1e-10 )
        << "P2M+M2M deviates from direct P2M to parent; "
           "max relative error = " << max_rel_err;
}
TEST( LaplaceKernel, testM2MTranslationVsDirectP2M )
{
    testM2MTranslationVsDirectP2M();
}

//---------------------------------------------------------------------------//
/**
 * Verify get_coeff negative-m symmetry:
 *   get_coeff(n, -m) == conj(get_coeff(n, m))   for m > 0
 */
TEST( LaplaceKernel, testGetCoeffSymmetry )
{
    using namespace LaplaceTest;

    const int N = Kernel::num_coeffs_per_cell;
    CoeffView2D coeffs( "coeffs", 1, N );
    {
        auto h = Kokkos::create_mirror_view( coeffs );
        std::mt19937 gen( 42 );
        std::uniform_real_distribution<double> d( -1.0, 1.0 );
        for ( int i = 0; i < N; i++ )
            h( 0, i ) = complex( d( gen ), d( gen ) );
        Kokkos::deep_copy( coeffs, h );
    }

    auto h_coeffs = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), coeffs );

    for ( int n = 0; n <= P_ORDER; n++ )
    {
        for ( int m = 1; m <= n; m++ )
        {
            const complex pos_m = get_coeff<double>( h_coeffs, 0,  n,  m, P_ORDER );
            const complex neg_m = get_coeff<double>( h_coeffs, 0,  n, -m, P_ORDER );
            EXPECT_NEAR( neg_m.real(),  pos_m.real(), 1e-15 )
                << "symmetry real  n=" << n << " m=" << m;
            EXPECT_NEAR( neg_m.imag(), -pos_m.imag(), 1e-15 )
                << "symmetry imag  n=" << n << " m=" << m;
        }
    }
}

//---------------------------------------------------------------------------//

} // end namespace Test
