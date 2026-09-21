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

#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

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
using Kernel = LaplaceKernel<double, P_ORDER>;
using complex = Kokkos::complex<double>;
using CoeffView2D = Kokkos::View<complex**, TEST_MEMSPACE>;
using CoeffView3D = Kokkos::View<complex***, TEST_MEMSPACE>;
using ScalarView1D = Kokkos::View<double*, TEST_MEMSPACE>;
using ScalarView2D = Kokkos::View<double**, TEST_MEMSPACE>;

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
            EXPECT_GE( idx, 0 )
                << "negative index at (" << n << "," << m << ")";
            EXPECT_LT( idx, N )
                << "out-of-range index at (" << n << "," << m << ")";
            EXPECT_FALSE( seen[idx] )
                << "duplicate index at (" << n << "," << m << ")";
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
    EXPECT_NEAR( A_coeff<double>( 0, 0 ), 1.0, 1e-14 );
    EXPECT_NEAR( A_coeff<double>( 1, 0 ), -1.0, 1e-14 );
    EXPECT_NEAR( A_coeff<double>( 1, 1 ), -1.0 / std::sqrt( 2.0 ), 1e-14 );
    EXPECT_NEAR( A_coeff<double>( 1, -1 ), -1.0 / std::sqrt( 2.0 ), 1e-14 );
    EXPECT_EQ( A_coeff<double>( 1, 2 ), 0.0 );
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
    const double phi = 0.0;

    for ( int n = 0; n <= P_ORDER; n++ )
    {
        complex y0 = Ynm<double>( n, 0, theta, phi );
        EXPECT_NEAR( y0.real(), 1.0, 1e-12 ) << "Y_{" << n << ",0} real";
        EXPECT_NEAR( y0.imag(), 0.0, 1e-12 ) << "Y_{" << n << ",0} imag";

        for ( int m = 1; m <= n; m++ )
        {
            complex yp = Ynm<double>( n, m, theta, phi );
            EXPECT_NEAR( yp.real(), 0.0, 1e-12 )
                << "Y_{" << n << "," << m << "} real";
            EXPECT_NEAR( yp.imag(), 0.0, 1e-12 )
                << "Y_{" << n << "," << m << "} imag";

            complex yn = Ynm<double>( n, -m, theta, phi );
            EXPECT_NEAR( yn.real(), 0.0, 1e-12 )
                << "Y_{" << n << "," << -m << "} real";
            EXPECT_NEAR( yn.imag(), 0.0, 1e-12 )
                << "Y_{" << n << "," << -m << "} imag";
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

    const double q = 2.5;
    const double dz = 0.7;

    CoeffView3D M_dev( "M", 1, Kernel::num_coeffs_per_cell, 1 );
    Kokkos::deep_copy( M_dev, complex( 0.0, 0.0 ) );

    Kokkos::parallel_for(
        "P2M_axis", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 1 ),
        KOKKOS_LAMBDA( int ) {
            double charges[1] = { q };
            auto M_out = Kokkos::subview( M_dev, 0, Kokkos::ALL, Kokkos::ALL );
            Kernel::p2m_contribution( charges, 0.0, 0.0, dz, M_out );
        } );
    Kokkos::fence();

    auto h_M =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M_dev );

    double rho_n = 1.0;
    for ( int n = 0; n <= P_ORDER; n++ )
    {
        EXPECT_NEAR( h_M( 0, coeff_index( n, 0 ), 0 ).real(), q * rho_n, 1e-12 )
            << "M_{" << n << ",0} real, n=" << n;
        EXPECT_NEAR( h_M( 0, coeff_index( n, 0 ), 0 ).imag(), 0.0, 1e-12 )
            << "M_{" << n << ",0} imag, n=" << n;

        for ( int m = 1; m <= n; m++ )
        {
            EXPECT_NEAR( h_M( 0, coeff_index( n, m ), 0 ).real(), 0.0, 1e-12 )
                << "M_{" << n << "," << m << "} real";
            EXPECT_NEAR( h_M( 0, coeff_index( n, m ), 0 ).imag(), 0.0, 1e-12 )
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
 * to a CPU reference that applies the same formula serially.
 */
void testP2MRefComparison()
{
    using namespace LaplaceTest;

    const int N_PARTICLES = 200;
    const double CELL_HALF_WIDTH = 0.5;

    std::mt19937 gen( 1234 );
    std::uniform_real_distribution<double> pos_dist( -CELL_HALF_WIDTH,
                                                     CELL_HALF_WIDTH );
    std::uniform_real_distribution<double> q_dist( -2.0, 2.0 );

    std::vector<double> px( N_PARTICLES ), py( N_PARTICLES ), pz( N_PARTICLES ),
        q( N_PARTICLES );
    for ( int i = 0; i < N_PARTICLES; i++ )
    {
        px[i] = pos_dist( gen );
        py[i] = pos_dist( gen );
        pz[i] = pos_dist( gen );
        q[i] = q_dist( gen );
    }

    Kokkos::View<double*, TEST_MEMSPACE> d_px( "px", N_PARTICLES );
    Kokkos::View<double*, TEST_MEMSPACE> d_py( "py", N_PARTICLES );
    Kokkos::View<double*, TEST_MEMSPACE> d_pz( "pz", N_PARTICLES );
    Kokkos::View<double*, TEST_MEMSPACE> d_q( "q", N_PARTICLES );
    {
        auto h_px = Kokkos::create_mirror_view( d_px );
        auto h_py = Kokkos::create_mirror_view( d_py );
        auto h_pz = Kokkos::create_mirror_view( d_pz );
        auto h_q = Kokkos::create_mirror_view( d_q );
        for ( int i = 0; i < N_PARTICLES; i++ )
        {
            h_px( i ) = px[i];
            h_py( i ) = py[i];
            h_pz( i ) = pz[i];
            h_q( i ) = q[i];
        }
        Kokkos::deep_copy( d_px, h_px );
        Kokkos::deep_copy( d_py, h_py );
        Kokkos::deep_copy( d_pz, h_pz );
        Kokkos::deep_copy( d_q, h_q );
    }

    CoeffView3D M_dev( "M", 1, Kernel::num_coeffs_per_cell, 1 );
    Kokkos::deep_copy( M_dev, complex( 0.0, 0.0 ) );

    Kokkos::parallel_for(
        "P2M_cloud", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, N_PARTICLES ),
        KOKKOS_LAMBDA( int p ) {
            double charges[1] = { d_q( p ) };
            auto M_out = Kokkos::subview( M_dev, 0, Kokkos::ALL, Kokkos::ALL );
            Kernel::p2m_contribution( charges, d_px( p ), d_py( p ), d_pz( p ),
                                      M_out );
        } );
    Kokkos::fence();

    auto h_M =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M_dev );

    // CPU reference
    std::vector<complex> M_ref( Kernel::num_coeffs_per_cell,
                                complex( 0.0, 0.0 ) );
    for ( int p = 0; p < N_PARTICLES; p++ )
    {
        const double rho =
            std::sqrt( px[p] * px[p] + py[p] * py[p] + pz[p] * pz[p] );
        const double theta = ( rho > 0.0 ) ? std::acos( pz[p] / rho ) : 0.0;
        const double phi = std::atan2( py[p], px[p] );

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
        const complex diff = h_M( 0, idx, 0 ) - M_ref[idx];
        const double abs_err =
            std::sqrt( diff.real() * diff.real() + diff.imag() * diff.imag() );
        const double ref_mag =
            std::sqrt( M_ref[idx].real() * M_ref[idx].real() +
                       M_ref[idx].imag() * M_ref[idx].imag() );
        const double rel_err =
            ( ref_mag > 1e-14 ) ? abs_err / ref_mag : abs_err;
        if ( rel_err > max_rel_err )
            max_rel_err = rel_err;
    }

    EXPECT_LT( max_rel_err, 1e-10 )
        << "P2M device result deviates from CPU reference; "
           "max relative error = "
        << max_rel_err;
}
TEST( LaplaceKernel, testP2MRefComparison ) { testP2MRefComparison(); }

//---------------------------------------------------------------------------//
/**
 * M2M translation exactness: P2M-to-child + M2M == direct P2M-to-parent.
 *
 * Per the Greengard M2M theorem, translating a multipole computed at a child
 * center to a parent center gives the same coefficients as a direct P2M at
 * the parent center.
 */
void testM2MTranslationVsDirectP2M()
{
    using namespace LaplaceTest;

    const double tx = 0.25, ty = 0.15, tz = -0.10;
    const double child_hw = 0.25;
    const int N_PARTICLES = 50;

    std::mt19937 gen( 5678 );
    std::uniform_real_distribution<double> pos_dist( -child_hw, child_hw );
    std::uniform_real_distribution<double> q_dist( -1.0, 1.0 );

    std::vector<double> lx( N_PARTICLES ), ly( N_PARTICLES ), lz( N_PARTICLES ),
        q( N_PARTICLES );
    for ( int i = 0; i < N_PARTICLES; i++ )
    {
        lx[i] = pos_dist( gen );
        ly[i] = pos_dist( gen );
        lz[i] = pos_dist( gen );
        q[i] = q_dist( gen );
    }

    // CPU reference: direct P2M to parent center
    std::vector<complex> M_direct( Kernel::num_coeffs_per_cell,
                                   complex( 0.0, 0.0 ) );
    for ( int p = 0; p < N_PARTICLES; p++ )
    {
        const double dx = lx[p] + tx;
        const double dy = ly[p] + ty;
        const double dz = lz[p] + tz;
        const double rho = std::sqrt( dx * dx + dy * dy + dz * dz );
        const double theta = ( rho > 0.0 ) ? std::acos( dz / rho ) : 0.0;
        const double phi = std::atan2( dy, dx );

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

    Kokkos::View<double*, TEST_MEMSPACE> d_lx( "lx", N_PARTICLES );
    Kokkos::View<double*, TEST_MEMSPACE> d_ly( "ly", N_PARTICLES );
    Kokkos::View<double*, TEST_MEMSPACE> d_lz( "lz", N_PARTICLES );
    Kokkos::View<double*, TEST_MEMSPACE> d_q( "q", N_PARTICLES );
    {
        auto h_lx = Kokkos::create_mirror_view( d_lx );
        auto h_ly = Kokkos::create_mirror_view( d_ly );
        auto h_lz = Kokkos::create_mirror_view( d_lz );
        auto h_q = Kokkos::create_mirror_view( d_q );
        for ( int i = 0; i < N_PARTICLES; i++ )
        {
            h_lx( i ) = lx[i];
            h_ly( i ) = ly[i];
            h_lz( i ) = lz[i];
            h_q( i ) = q[i];
        }
        Kokkos::deep_copy( d_lx, h_lx );
        Kokkos::deep_copy( d_ly, h_ly );
        Kokkos::deep_copy( d_lz, h_lz );
        Kokkos::deep_copy( d_q, h_q );
    }

    // P2M into child (3D view: 1 cell x num_coeffs x 1 comp)
    CoeffView3D M_child( "M_child", 1, Kernel::num_coeffs_per_cell, 1 );
    Kokkos::deep_copy( M_child, complex( 0.0, 0.0 ) );

    Kokkos::parallel_for(
        "P2M_child", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, N_PARTICLES ),
        KOKKOS_LAMBDA( int p ) {
            double charges[1] = { d_q( p ) };
            auto M_out =
                Kokkos::subview( M_child, 0, Kokkos::ALL, Kokkos::ALL );
            Kernel::p2m_contribution( charges, d_lx( p ), d_ly( p ), d_lz( p ),
                                      M_out );
        } );
    Kokkos::fence();

    // A-coefficient table
    auto A = build_A_coefficients<double, TEST_MEMSPACE>( P_ORDER );

    // M2M translate child (cell 0) into parent (2D output: num_coeffs x 1)
    CoeffView2D M_parent( "M_parent", Kernel::num_coeffs_per_cell, 1 );
    Kokkos::deep_copy( M_parent, complex( 0.0, 0.0 ) );

    using team_policy = Kokkos::TeamPolicy<TEST_EXECSPACE>;
    Kokkos::parallel_for(
        "M2M_translate", team_policy( 1, Kokkos::AUTO ),
        KOKKOS_LAMBDA( const typename team_policy::member_type& team ) {
            Kernel::m2m_translate( team, M_child, 0, tx, ty, tz, A, M_parent );
        } );
    Kokkos::fence();

    auto h_M_parent =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M_parent );

    double max_rel_err = 0.0;
    for ( int idx = 0; idx < Kernel::num_coeffs_per_cell; idx++ )
    {
        const complex diff = h_M_parent( idx, 0 ) - M_direct[idx];
        const double abs_err =
            std::sqrt( diff.real() * diff.real() + diff.imag() * diff.imag() );
        const double ref_mag =
            std::sqrt( M_direct[idx].real() * M_direct[idx].real() +
                       M_direct[idx].imag() * M_direct[idx].imag() );
        const double rel_err =
            ( ref_mag > 1e-14 ) ? abs_err / ref_mag : abs_err;
        if ( rel_err > max_rel_err )
            max_rel_err = rel_err;
    }

    EXPECT_LT( max_rel_err, 1e-10 )
        << "P2M+M2M deviates from direct P2M to parent; "
           "max relative error = "
        << max_rel_err;
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

    auto h_coeffs =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), coeffs );

    for ( int n = 0; n <= P_ORDER; n++ )
    {
        for ( int m = 1; m <= n; m++ )
        {
            const complex pos_m =
                get_coeff<double>( h_coeffs, 0, n, m, P_ORDER );
            const complex neg_m =
                get_coeff<double>( h_coeffs, 0, n, -m, P_ORDER );
            EXPECT_NEAR( neg_m.real(), pos_m.real(), 1e-15 )
                << "symmetry real  n=" << n << " m=" << m;
            EXPECT_NEAR( neg_m.imag(), -pos_m.imag(), 1e-15 )
                << "symmetry imag  n=" << n << " m=" << m;
        }
    }
}

//---------------------------------------------------------------------------//
/**
 * M2L correctness: P2M at source + M2L + L2P matches direct 1/r sum.
 *
 * Source particles are randomly distributed within a box at the origin.
 * The target cell is 5 units away along x. The observation point is
 * slightly off-center inside the target cell. At P=6 with a 5:0.4 separation
 * ratio the FMM truncation error is O((0.4/5)^7) ~ 2e-8.
 */
void testM2LThenL2P()
{
    using namespace LaplaceTest;

    const int N_PARTICLES = 50;
    const double src_hw = 0.4;
    const double sep = 5.0; // target cell center along x

    std::mt19937 gen( 2345 );
    std::uniform_real_distribution<double> pos_dist( -src_hw, src_hw );
    std::uniform_real_distribution<double> q_dist( -1.0, 1.0 );

    std::vector<double> sx( N_PARTICLES ), sy( N_PARTICLES ), sz( N_PARTICLES ),
        sq( N_PARTICLES );
    for ( int i = 0; i < N_PARTICLES; i++ )
    {
        sx[i] = pos_dist( gen );
        sy[i] = pos_dist( gen );
        sz[i] = pos_dist( gen );
        sq[i] = q_dist( gen );
    }

    // Observation point relative to target cell center
    const double test_dx = 0.12, test_dy = -0.08, test_dz = 0.05;

    // Direct potential
    double phi_direct = 0.0;
    for ( int i = 0; i < N_PARTICLES; i++ )
    {
        double rx = ( sep + test_dx ) - sx[i];
        double ry = test_dy - sy[i];
        double rz = test_dz - sz[i];
        phi_direct += sq[i] / std::sqrt( rx * rx + ry * ry + rz * rz );
    }

    // Upload source particles
    Kokkos::View<double*, TEST_MEMSPACE> d_sx( "sx", N_PARTICLES );
    Kokkos::View<double*, TEST_MEMSPACE> d_sy( "sy", N_PARTICLES );
    Kokkos::View<double*, TEST_MEMSPACE> d_sz( "sz", N_PARTICLES );
    Kokkos::View<double*, TEST_MEMSPACE> d_sq( "sq", N_PARTICLES );
    {
        auto h_sx = Kokkos::create_mirror_view( d_sx );
        auto h_sy = Kokkos::create_mirror_view( d_sy );
        auto h_sz = Kokkos::create_mirror_view( d_sz );
        auto h_sq = Kokkos::create_mirror_view( d_sq );
        for ( int i = 0; i < N_PARTICLES; i++ )
        {
            h_sx( i ) = sx[i];
            h_sy( i ) = sy[i];
            h_sz( i ) = sz[i];
            h_sq( i ) = sq[i];
        }
        Kokkos::deep_copy( d_sx, h_sx );
        Kokkos::deep_copy( d_sy, h_sy );
        Kokkos::deep_copy( d_sz, h_sz );
        Kokkos::deep_copy( d_sq, h_sq );
    }

    // P2M at source cell (cell 0)
    CoeffView3D M( "M", 1, Kernel::num_coeffs_per_cell, 1 );
    Kokkos::deep_copy( M, complex( 0.0, 0.0 ) );
    Kokkos::parallel_for(
        "P2M", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, N_PARTICLES ),
        KOKKOS_LAMBDA( int p ) {
            double charges[1] = { d_sq( p ) };
            auto M_out = Kokkos::subview( M, 0, Kokkos::ALL, Kokkos::ALL );
            Kernel::p2m_contribution( charges, d_sx( p ), d_sy( p ), d_sz( p ),
                                      M_out );
        } );
    Kokkos::fence();

    // A table must cover orders up to 2*P for M2L (accesses A_{n+j, m-k})
    auto A = build_A_coefficients<double, TEST_MEMSPACE>( 2 * P_ORDER );

    // M2L: translation vector = source_center - target_center =
    // (0,0,0)-(sep,0,0)
    CoeffView3D L( "L", 1, Kernel::num_coeffs_per_cell, 1 );
    Kokkos::deep_copy( L, complex( 0.0, 0.0 ) );

    using team_policy = Kokkos::TeamPolicy<TEST_EXECSPACE>;
    Kokkos::parallel_for(
        "M2L", team_policy( 1, Kokkos::AUTO ),
        KOKKOS_LAMBDA( const typename team_policy::member_type& team ) {
            auto L_out = Kokkos::subview( L, 0, Kokkos::ALL, Kokkos::ALL );
            Kernel::m2l_translate( team, M, 0, -sep, 0.0, 0.0, A, L_out );
        } );
    Kokkos::fence();

    // L2P: evaluate at observation point (no gradient needed)
    ScalarView1D phi_dev( "phi", 1 );
    ScalarView2D grad_dev( "grad", 1, 3 ); // (comp, dim); unused here
    Kokkos::parallel_for(
        "L2P", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 1 ),
        KOKKOS_LAMBDA( int ) {
            double phi_out[1];
            Kernel::l2p_evaluate( L, 0, test_dx, test_dy, test_dz, phi_out,
                                  grad_dev, false );
            phi_dev( 0 ) = phi_out[0];
        } );
    Kokkos::fence();

    auto h_phi =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), phi_dev );

    const double abs_err = std::abs( h_phi( 0 ) - phi_direct );
    const double ref_mag = std::abs( phi_direct );
    const double rel_err = ( ref_mag > 1e-14 ) ? abs_err / ref_mag : abs_err;

    EXPECT_LT( rel_err, 1e-6 ) << "M2L+L2P deviates from direct sum; "
                                  "relative error = "
                               << rel_err << "  phi_fmm=" << h_phi( 0 )
                               << "  phi_direct=" << phi_direct;
}
TEST( LaplaceKernel, testM2LThenL2P ) { testM2LThenL2P(); }

//---------------------------------------------------------------------------//
/**
 * L2L translation correctness: P2M + M2L to parent + L2L to child + L2P
 * matches the direct 1/r sum at the same observation point.
 *
 * Source at origin, parent target at (5,0,0), child center at (5.25,0,0).
 * L2L is exact for finite expansions; any error is purely from P truncation.
 */
void testL2LTranslation()
{
    using namespace LaplaceTest;

    const int N_PARTICLES = 50;
    const double src_hw = 0.4;
    const double sep = 5.0;          // parent cell center along x
    const double child_off_x = 0.25; // child center relative to parent

    std::mt19937 gen( 3456 );
    std::uniform_real_distribution<double> pos_dist( -src_hw, src_hw );
    std::uniform_real_distribution<double> q_dist( -1.0, 1.0 );

    std::vector<double> sx( N_PARTICLES ), sy( N_PARTICLES ), sz( N_PARTICLES ),
        sq( N_PARTICLES );
    for ( int i = 0; i < N_PARTICLES; i++ )
    {
        sx[i] = pos_dist( gen );
        sy[i] = pos_dist( gen );
        sz[i] = pos_dist( gen );
        sq[i] = q_dist( gen );
    }

    // Observation point relative to child cell center
    const double test_dx = 0.05, test_dy = 0.03, test_dz = -0.04;

    // Direct potential at global position (sep + child_off_x + test_dx, ...)
    double phi_direct = 0.0;
    for ( int i = 0; i < N_PARTICLES; i++ )
    {
        double rx = ( sep + child_off_x + test_dx ) - sx[i];
        double ry = test_dy - sy[i];
        double rz = test_dz - sz[i];
        phi_direct += sq[i] / std::sqrt( rx * rx + ry * ry + rz * rz );
    }

    // Upload source particles
    Kokkos::View<double*, TEST_MEMSPACE> d_sx( "sx", N_PARTICLES );
    Kokkos::View<double*, TEST_MEMSPACE> d_sy( "sy", N_PARTICLES );
    Kokkos::View<double*, TEST_MEMSPACE> d_sz( "sz", N_PARTICLES );
    Kokkos::View<double*, TEST_MEMSPACE> d_sq( "sq", N_PARTICLES );
    {
        auto h_sx = Kokkos::create_mirror_view( d_sx );
        auto h_sy = Kokkos::create_mirror_view( d_sy );
        auto h_sz = Kokkos::create_mirror_view( d_sz );
        auto h_sq = Kokkos::create_mirror_view( d_sq );
        for ( int i = 0; i < N_PARTICLES; i++ )
        {
            h_sx( i ) = sx[i];
            h_sy( i ) = sy[i];
            h_sz( i ) = sz[i];
            h_sq( i ) = sq[i];
        }
        Kokkos::deep_copy( d_sx, h_sx );
        Kokkos::deep_copy( d_sy, h_sy );
        Kokkos::deep_copy( d_sz, h_sz );
        Kokkos::deep_copy( d_sq, h_sq );
    }

    // P2M at source cell
    CoeffView3D M( "M", 1, Kernel::num_coeffs_per_cell, 1 );
    Kokkos::deep_copy( M, complex( 0.0, 0.0 ) );
    Kokkos::parallel_for(
        "P2M", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, N_PARTICLES ),
        KOKKOS_LAMBDA( int p ) {
            double charges[1] = { d_sq( p ) };
            auto M_out = Kokkos::subview( M, 0, Kokkos::ALL, Kokkos::ALL );
            Kernel::p2m_contribution( charges, d_sx( p ), d_sy( p ), d_sz( p ),
                                      M_out );
        } );
    Kokkos::fence();

    auto A = build_A_coefficients<double, TEST_MEMSPACE>( 2 * P_ORDER );

    // M2L into parent local (cell 0)
    CoeffView3D L_parent( "L_parent", 1, Kernel::num_coeffs_per_cell, 1 );
    Kokkos::deep_copy( L_parent, complex( 0.0, 0.0 ) );

    using team_policy = Kokkos::TeamPolicy<TEST_EXECSPACE>;
    Kokkos::parallel_for(
        "M2L", team_policy( 1, Kokkos::AUTO ),
        KOKKOS_LAMBDA( const typename team_policy::member_type& team ) {
            auto L_out =
                Kokkos::subview( L_parent, 0, Kokkos::ALL, Kokkos::ALL );
            Kernel::m2l_translate( team, M, 0, -sep, 0.0, 0.0, A, L_out );
        } );
    Kokkos::fence();

    // L2L: translate parent local (at sep,0,0) to child local (at
    // sep+child_off_x,0,0) translation = child_center - parent_center =
    // (child_off_x, 0, 0)
    CoeffView3D L_child( "L_child", 1, Kernel::num_coeffs_per_cell, 1 );
    Kokkos::deep_copy( L_child, complex( 0.0, 0.0 ) );

    Kokkos::parallel_for(
        "L2L", team_policy( 1, Kokkos::AUTO ),
        KOKKOS_LAMBDA( const typename team_policy::member_type& team ) {
            auto L_out =
                Kokkos::subview( L_child, 0, Kokkos::ALL, Kokkos::ALL );
            Kernel::l2l_translate( team, L_parent, 0, child_off_x, 0.0, 0.0, A,
                                   L_out );
        } );
    Kokkos::fence();

    // L2P at child observation point
    ScalarView1D phi_dev( "phi", 1 );
    ScalarView2D grad_dev( "grad", 1, 3 );
    Kokkos::parallel_for(
        "L2P", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 1 ),
        KOKKOS_LAMBDA( int ) {
            double phi_out[1];
            Kernel::l2p_evaluate( L_child, 0, test_dx, test_dy, test_dz,
                                  phi_out, grad_dev, false );
            phi_dev( 0 ) = phi_out[0];
        } );
    Kokkos::fence();

    auto h_phi =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), phi_dev );

    const double abs_err = std::abs( h_phi( 0 ) - phi_direct );
    const double ref_mag = std::abs( phi_direct );
    const double rel_err = ( ref_mag > 1e-14 ) ? abs_err / ref_mag : abs_err;

    EXPECT_LT( rel_err, 1e-6 ) << "P2M+M2L+L2L+L2P deviates from direct sum; "
                                  "relative error = "
                               << rel_err << "  phi_fmm=" << h_phi( 0 )
                               << "  phi_direct=" << phi_direct;
}
TEST( LaplaceKernel, testL2LTranslation ) { testL2LTranslation(); }

//---------------------------------------------------------------------------//
/**
 * L2P gradient check against analytical 1/r gradient.
 *
 * A single particle at the source cell center contributes only M_{0,0}=q
 * (all higher multipoles vanish because rho=0). After M2L the local expansion
 * at the target cell represents the exact Coulomb field, so the gradient
 * computed by L2P via central finite differences must match -q*r/r^3 to
 * the accuracy of the FD step (h=1e-5 gives ~1e-10 FD error here).
 */
void testL2PGradient()
{
    using namespace LaplaceTest;

    const double q_charge = 1.5;
    const double sep = 6.0; // target cell center at (sep, 0, 0)

    // Observation point relative to target cell center
    const double test_dx = 0.10, test_dy = 0.06, test_dz = -0.04;

    // P2M: single particle at source cell center (rho=0, only M_{0,0} nonzero)
    CoeffView3D M( "M", 1, Kernel::num_coeffs_per_cell, 1 );
    Kokkos::deep_copy( M, complex( 0.0, 0.0 ) );
    Kokkos::parallel_for(
        "P2M_single", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 1 ),
        KOKKOS_LAMBDA( int ) {
            double charges[1] = { q_charge };
            auto M_out = Kokkos::subview( M, 0, Kokkos::ALL, Kokkos::ALL );
            Kernel::p2m_contribution( charges, 0.0, 0.0, 0.0, M_out );
        } );
    Kokkos::fence();

    auto A = build_A_coefficients<double, TEST_MEMSPACE>( 2 * P_ORDER );

    // M2L
    CoeffView3D L( "L", 1, Kernel::num_coeffs_per_cell, 1 );
    Kokkos::deep_copy( L, complex( 0.0, 0.0 ) );

    using team_policy = Kokkos::TeamPolicy<TEST_EXECSPACE>;
    Kokkos::parallel_for(
        "M2L", team_policy( 1, Kokkos::AUTO ),
        KOKKOS_LAMBDA( const typename team_policy::member_type& team ) {
            auto L_out = Kokkos::subview( L, 0, Kokkos::ALL, Kokkos::ALL );
            Kernel::m2l_translate( team, M, 0, -sep, 0.0, 0.0, A, L_out );
        } );
    Kokkos::fence();

    // L2P with gradient
    ScalarView1D phi_dev( "phi", 1 );
    ScalarView2D grad_dev( "grad", 1, 3 ); // (comp=0, dim=0,1,2)
    Kokkos::parallel_for(
        "L2P_grad", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 1 ),
        KOKKOS_LAMBDA( int ) {
            double phi_out[1];
            Kernel::l2p_evaluate( L, 0, test_dx, test_dy, test_dz, phi_out,
                                  grad_dev, true );
            phi_dev( 0 ) = phi_out[0];
        } );
    Kokkos::fence();

    auto h_phi =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), phi_dev );
    auto h_grad =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), grad_dev );

    // Analytical: source at (0,0,0), obs at (sep+test_dx, test_dy, test_dz)
    const double rx = sep + test_dx, ry = test_dy, rz = test_dz;
    const double r = std::sqrt( rx * rx + ry * ry + rz * rz );
    const double phi_analytic = q_charge / r;
    const double grad_analytic[3] = { -q_charge * rx / ( r * r * r ),
                                      -q_charge * ry / ( r * r * r ),
                                      -q_charge * rz / ( r * r * r ) };

    // Potential tolerance: FMM truncation ~ O((0/sep)^7) = 0 for point at
    // origin, but residual from P-truncation of the expansion is ~1e-12; use
    // 1e-8.
    EXPECT_NEAR( h_phi( 0 ), phi_analytic, 1e-8 ) << "L2P potential mismatch";

    // Gradient tolerance: FD step h=1e-5 contributes O(h^2 phi''') ~ 1e-12;
    // FMM/FD combined error budget is 1e-7.
    const char* dim_name[3] = { "x", "y", "z" };
    for ( int d = 0; d < 3; d++ )
    {
        EXPECT_NEAR( h_grad( 0, d ), grad_analytic[d], 1e-7 )
            << "L2P gradient mismatch in " << dim_name[d];
    }
}
TEST( LaplaceKernel, testL2PGradient ) { testL2PGradient(); }

//---------------------------------------------------------------------------//

} // end namespace Test
