/****************************************************************************
 * Copyright (c) 20125 by the Canopy authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Canopy library. Canopy is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Canopy_Kernels.hpp
  \brief Computational kernels for data manipluation. Assumes positions
  are the first tuple element in any data AoSoA.
*/

#ifndef CANOPY_KERNELS_HPP
#define CANOPY_KERNELS_HPP

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <cmath>
#include <memory>

#include <limits>

namespace Canopy
{

namespace Kernel
{

constexpr auto pi = Kokkos::numbers::pi_v<double>;

// Cartesian to spherical coorindates: (x, y, z) -> (r,theta,phi)
KOKKOS_INLINE_FUNCTION
void cart2sph( double x, double y, double z, double& r, double& theta,
               double& phi )
{
    r = Kokkos::sqrt( x * x + y * y + z * z );
    theta = ( r == 0.0 ? 0.0 : Kokkos::acos( z / r ) ); // polar angle
    phi = Kokkos::atan2( y, x );                        // azimuth
}

// Factorial double: (2m-1)!!
KOKKOS_INLINE_FUNCTION
double factorial( int k )
{
    // Γ(k+1) = k!
    return Kokkos::tgamma( static_cast<double>( k ) + 1.0 );
}

KOKKOS_INLINE_FUNCTION
double double_factorial( int m )
{
    // (2m − 1)!! = (2m)! / (2^m m!)
    // use gamma functions to avoid integer overflow
    const double two_m = static_cast<double>( 2 * m );
    const double m_d = static_cast<double>( m );
    return Kokkos::tgamma( two_m + 1.0 ) /
           ( Kokkos::pow( 2.0, m_d ) * Kokkos::tgamma( m_d + 1.0 ) );
}

namespace Scalar
{
//---------------------------------------------------------------------------//

/**
 * Implementation of std::assoc_legendre that is callable on the device.
 * Per equations 3.33 and 3.34 in source 4.
 */
KOKKOS_INLINE_FUNCTION
double Pnm_impl( int n, int m, double x )
{
    if ( m < 0 || m > n )
        return 0.0; // undefined outside this range

    // P_m^m(x)
    double pmm = double_factorial( m ) * std::pow( 1.0 - x * x, 0.5 * m );
    if ( m % 2 == 1 )
        pmm = -pmm; // (-1)^m factor

    if ( n == m )
        return pmm;

    // P_{m+1}^m(x)
    double pmmp1 = x * ( 2 * m + 1 ) * pmm;
    if ( n == m + 1 )
        return pmmp1;

    // Upward recurrence
    double pnm2 = pmm;
    double pnm1 = pmmp1;
    double pn = 0.0;
    for ( int l = m + 2; l <= n; ++l )
    {
        pn = ( ( 2 * l - 1 ) * x * pnm1 - ( l + m - 1 ) * pnm2 ) / ( l - m );
        pnm2 = pnm1;
        pnm1 = pn;
    }
    return pn;
}

/**
 * Compute the complex spherical harmonic (complex, condon-shortley phase)
 * Y_(n, m) (theta, phi)
 * Where:
 *  theta is the polar angle
 *  phi is the azimuthal angle
 * in spherical coorindates
 * Per equation 3.32 in source 4.
 */
KOKKOS_INLINE_FUNCTION
Kokkos::complex<double> Ynm( int n, int m, double theta, double phi )
{
    using cdouble = Kokkos::complex<double>;

    int mp = Kokkos::abs( m );
    double x = Kokkos::cos( theta );

    double Pnm = Pnm_impl( n, mp, x );

    // double Pnm_new = Pnm_impl(n, mp, x);
    // printf("n%d, mp%d, x: %0.4lf: assoc: %0.9lf, impl: %0.9lf\n", n, mp, x,
    // Pnm, Pnm_new);

    // See equation 3.27, source 4 for including sqrt((2n+1 / 4pi))
    double norm = Kokkos::sqrt( Kokkos::tgamma( n - mp + 1 ) /
                                Kokkos::tgamma( n + mp + 1 ) );

    // Equation 3.32, source 4
    cdouble y = norm * Pnm * Kokkos::polar( 1.0, double( m ) * phi );

    double phase = ( m >= 0 ? ( ( m % 2 ) ? -1.0 : 1.0 ) // (-1)^m
                            : ( ( ( -m ) % 2 ) ? -1.0 : 1.0 ) );

    return y * phase;
}

/**
 * Compute offset into flattened multipole array
 * (n,m) -> index
 */
KOKKOS_INLINE_FUNCTION
int index( int n, int m ) { return n * n + ( m + n ); }

/**
 * Operator calculates the kernel for scalar-based multipoles
 * and return the multipole coefficient matrix flattened into a
 * 1D vector.
 */
template <class MemorySpace, class ExecutionSpace>
struct P2M
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using cdouble = Kokkos::complex<double>;

    P2M( int p )
        : _p( p )
    {
        _M = Kokkos::View<cdouble*, memory_space>( "M", ( p + 1 ) * ( p + 1 ) );
        clear();
    }

  private:
    int _p;
    Kokkos::View<cdouble*, memory_space> _M;

  public:
    auto coefficients() { return _M; }

    /**
     * Clear coefficents
     */
    void clear() { Kokkos::deep_copy( _M, cdouble( 0.0, 0.0 ) ); }

    /**
     * Compute multipole coefficients M[n][m]
     * up to order p around expansion_center.
     *
     * Definition:
     *   M_n^m = Σ_i q_i * ρ_i^n * conj( Y_n^m(α_i,β_i) )
     *
     * where (ρ_i,α_i,β_i) are spherical coords of point i
     * relative to expansion_center.
     */
    template <class PositionArray, class ScalarArray>
    void operator()( const PositionArray& pos, const ScalarArray& scalar,
                     std::size_t k,
                     const Kokkos::Array<double, 3>& expansion_center ) const
    {
        int p = _p;
        auto M = _M;

        // Further optimize this code for running on the device
        Kokkos::parallel_for(
            "compute multipole coefficients",
            Kokkos::RangePolicy<execution_space>( 0, k ),
            KOKKOS_LAMBDA( const int i ) {
                double dx = pos( i, 0 ) - expansion_center[0];
                double dy = pos( i, 1 ) - expansion_center[1];
                double dz = pos( i, 2 ) - expansion_center[2];

                double rho, alpha, beta;
                cart2sph( dx, dy, dz, rho, alpha, beta );

                // Equation 3.36, source 4
                for ( int n = 0; n <= p; ++n )
                {
                    for ( int m = -n; m <= n; ++m )
                    {
                        int idx = index( n, m );
                        // Equation 3.37, source 4
                        // auto norm = Kokkos::sqrt(( ( 2.0 * n + 1 ) / ( 4.0 *
                        // pi ) ));
                        auto val = scalar( i ) * Kokkos::pow( rho, n ) *
                                   Ynm( n, -m, alpha, beta ); // / norm;
                        Kokkos::atomic_add( &M( idx ), val );
                        // printf("k%d, n%d, m%d setting index: %d\n",
                        //     i, n, m, idx);
                    }
                }
            } );
    }
};

/**
 * Equation 3.26, source 4
 */
KOKKOS_INLINE_FUNCTION
double compute_A( int n, int m )
{
    double denom = Kokkos::sqrt( factorial( n - m ) * factorial( n + m ) );
    double sign = ( n % 2 == 0 ) ? 1.0 : -1.0; // (-1)^n
    return sign / denom;
}

/**
 * Equation 3.43, source 4
 */
KOKKOS_INLINE_FUNCTION
double compute_J( int n, int m )
{
    if ( n * m < 0 )
    {
        int min_abs = ( Kokkos::abs( n ) < Kokkos::abs( m ) )
                          ? Kokkos::abs( n )
                          : Kokkos::abs( m );
        return double( ( min_abs % 2 == 0 ) ? 1 : -1 ); // (-1)^min(|n|,|m|)
    }
    else
    {
        return 1.0;
    }
}

/**
 * Equation 3.54, source 4
 */
KOKKOS_INLINE_FUNCTION
double compute_J( int n, int m, int m_p )
{
    auto minus_one_pow = []( int k ) -> double {
        return ( k % 2 == 0 ) ? 1.0 : -1.0;
    };

    if ( m * m_p < 0 )
    {
        return minus_one_pow( n ) * minus_one_pow( m );
    }
    else if ( m * m_p > 0 && std::abs( m_p ) < std::abs( m ) )
    {
        return minus_one_pow( n ) * minus_one_pow( m_p - m );
    }
    else
    {
        return minus_one_pow( n );
    }
}


/**
 * Operator calculates the multipole expansions about the centers of
 * cells not in the leaf layer using the potential field from its child cells
 * Per source 4, page 68, step 2.
 */
template <class MemorySpace, class ExecutionSpace>
struct M2M
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using cdouble = Kokkos::complex<double>;

    M2M( int p )
        : _p( p )
    {
        _M = Kokkos::View<cdouble*, memory_space>( "M", ( p + 1 ) * ( p + 1 ) );
        clear();
    }

  private:
    int _p;
    Kokkos::View<cdouble*, memory_space> _M;

  public:
    auto coefficients() { return _M; }

    /**
     * Clear coefficents
     */
    void clear() { Kokkos::deep_copy( _M, cdouble( 0.0, 0.0 ) ); }

    template <class MultipoleVector>
    void operator()( const MultipoleVector& M_orig,
                     const Kokkos::Array<double, 3>& center_orig ) const
    {
        const int p = _p;
        auto M = _M;

        // spherical coords for the displacement
        double rho, alpha, beta;
        cart2sph( center_orig[0], center_orig[1], center_orig[2], rho, alpha,
                  beta );

        for ( int j = 0; j <= p; ++j )
        {
            for ( int k = -j; k <= j; ++k )
            {
                cdouble Mjk( 0.0, 0.0 );

                for ( int n = 0; n <= j; ++n )
                {
                    int j_n = j - n;

                    for ( int m = -n; m <= n; ++m )
                    {
                        int k_m = k - m;

                        // Must satisfy |k_m| <= j_n
                        // to avoid negative values passed to compute_A.
                        if ( std::abs( k_m ) > j_n )
                            continue;

                        int orig_index = index( j_n, k_m );
                        cdouble O = M_orig( orig_index );

                        // Values for eq 3.57
                        const double J = compute_J( m, k_m );
                        const double A_nm = compute_A( n, m );
                        const double A_jn_km = compute_A( j_n, k_m );
                        const double A_jk = compute_A( j, k );
                        double rho_n = Kokkos::pow( rho, n );

                        // M_child already has its Y_nm normalized. The Ynm
                        // function performs normalization internally, so we
                        // need to un-normalize it after calling Ynm to avoid
                        // double normalization.
                        auto Y_nm = Ynm( n, -m, alpha, beta );
                        Mjk += ( O * J * A_nm * A_jn_km * rho_n * Y_nm ) / A_jk;
                    }
                }
                int parent_idx = index( j, k );
                M( parent_idx ) += Mjk;
            }
        }
    }
};

/**
 * Convert multipole expansions into local expansions using
 * Theorem 3.5.5 in source 4.
 */
template <class MemorySpace, class ExecutionSpace>
struct M2L
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using cdouble = Kokkos::complex<double>;

    M2L( int p )
        : _p( p )
    {
        _L = Kokkos::View<cdouble*, memory_space>( "L", ( p + 1 ) * ( p + 1 ) );
        clear();
    }

  private:
    int _p;
    Kokkos::View<cdouble*, memory_space> _L;

  public:
    auto coefficients() { return _L; }

    /**
     * Clear coefficents
     */
    void clear() { Kokkos::deep_copy( _L, cdouble( 0.0, 0.0 ) ); }

    /**
     * Compute local coefficients L[n][m] up to order p
     * 
     * @param O multipole coefficients centered around O_center.
     * @param O_center the ceneter of multipole coefficients O.
     */
    template <class MultipoleVector>
    void operator()( const MultipoleVector& O,
                     const Kokkos::Array<double, 3>& O_center ) const
    {
        int p = _p;
        auto L = _L;

        // Spherical coords of O_center
        double rho, alpha, beta;
        cart2sph( O_center[0], O_center[1], O_center[2], rho, alpha, beta );

        // Optimize this code for running on the device
        for ( int j = 0; j <= p; ++j )
        {
            for ( int k = -j; k <= j; ++k )
            {
                cdouble Ljk( 0.0, 0.0 );

                for ( int n = 0; n <= p; ++n )
                {
                    for ( int m = -n; m <= n; ++m )
                    {
                        // Numerator of eq 3.60
                        cdouble O_nm = O( index( n, m ) );
                        auto J_km = compute_J( k, m );
                        auto A_nm = compute_A( n, m );
                        auto A_jk = compute_A( j, k );
                        auto Y_jn_mk = Ynm( j + n, m - k, alpha, beta );

                        // Demoninator of eq 3.60
                        auto A_jn_mk = compute_A( j + n, m - k );
                        auto rho_jn = Kokkos::pow( rho, j + n + 1 );

                        Ljk += ( O_nm * J_km * A_nm * A_jk * Y_jn_mk ) /
                               ( A_jn_mk * rho_jn );
                    }
                }
                L( index( j, k ) ) = Ljk;
            }
        }
    }
};

/**
 * Translate local expansions.
 * Theorem 3.5.6 in source 4.
 */
template <class MemorySpace, class ExecutionSpace>
struct L2L
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using cdouble = Kokkos::complex<double>;

    L2L( int p )
        : _p( p )
    {
        _L = Kokkos::View<cdouble*, memory_space>( "L", ( p + 1 ) * ( p + 1 ) );
        clear();
    }

  private:
    int _p;
    Kokkos::View<cdouble*, memory_space> _L;

  public:
    auto coefficients() { return _L; }

    /**
     * Clear coefficents
     */
    void clear() { Kokkos::deep_copy( _L, cdouble( 0.0, 0.0 ) ); }

    /**
     * Compute local coefficients L[n][m] up to order p
     * 
     * @param O multipole coefficients centered around O_center.
     * @param O_center the ceneter of multipole coefficients O.
     */
    template <class LocalVector>
    void operator()( const LocalVector& O,
                     const Kokkos::Array<double, 3>& O_center ) const
    {
        int p = _p;
        auto L = _L;

        // Spherical coords of O_center
        double rho, alpha, beta;
        cart2sph( O_center[0], O_center[1], O_center[2], rho, alpha, beta );

        // Optimize this code for running on the device
        for ( int j = 0; j <= p; ++j )
        {
            for ( int k = -j; k <= j; ++k )
            {
                cdouble Ljk( 0.0, 0.0 );

                for ( int n = j; n <= p; ++n )
                {
                    for ( int m = -n; m <= n; ++m )
                    {
                        // Numerator of eq 3.60
                        cdouble O_nm = O( index( n, m ) );
                        auto J = compute_J( n-j, m-k, m );
                        auto A_nj_mk = compute_A( n-j, m-k );
                        auto A_jk = compute_A( j, k );
                        auto Y_nj_mk = Ynm( n - j, m - k, alpha, beta );
                        auto rho_nj = Kokkos::pow(rho, n-j);

                        Ljk += ( O_nm * J * A_nj_mk * A_jk * Y_nj_mk * rho_nj ) /
                               A_jk;
                    }
                }
                L( index( j, k ) ) = Ljk;
            }
        }
    }
};

} // end namespace Scalar

} // end namespace Kernel

} // end namespace Canopy

#endif // CANOPY_KERNELS_HPP
