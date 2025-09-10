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

//---------------------------------------------------------------------------//
// Factorial double: (2m-1)!!
KOKKOS_INLINE_FUNCTION
double double_factorial( int m )
{
    double res = 1.0;
    for ( int k = 1; k <= m; ++k )
        res *= ( 2 * k - 1 );
    return res;
}

/**
 * Implementation of std::assoc_legendre that is callable on the device.
 * Per equations 3.33 and 3.34 in source 4.
 */
KOKKOS_INLINE_FUNCTION
double assoc_legendre( int n, int m, double x )
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
    constexpr auto pi = Kokkos::numbers::pi_v<double>;

    int mp = Kokkos::abs( m );
    double x = Kokkos::cos( theta );

    double Pnm = assoc_legendre( n, mp, x );

    // See equation 3.27, source 4 for including sqrt((2n+1 / 4pi))
    double norm = Kokkos::sqrt( ( ( 2.0 * n + 1 ) / ( 4.0 * pi ) ) *
                                Kokkos::tgamma( n - mp + 1 ) /
                                Kokkos::tgamma( n + mp + 1 ) );

    // Equation 3.32, source 4
    cdouble y = norm * Pnm * Kokkos::polar( 1.0, double( mp ) * phi );

    if ( m < 0 )
    {
        // Y_n^{-m} = (-1)^{|m|} conj(Y_n^{|m|})
        return Kokkos::pow( -1.0, mp ) * Kokkos::conj( y );
    }
    return y;
}

// Cartesian to spherical coorindates: (x, y, z) -> (r,theta,phi)
KOKKOS_INLINE_FUNCTION
void cart2sph( double x, double y, double z, double& r, double& theta,
               double& phi )
{
    r = Kokkos::sqrt( x * x + y * y + z * z );
    theta = ( r == 0.0 ? 0.0 : Kokkos::acos( z / r ) ); // polar angle
    phi = Kokkos::atan2( y, x );                        // azimuth
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
struct ScalarP2M
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using cdouble = Kokkos::complex<double>;

    ScalarP2M( int p )
        : _p( p )
    {
    }

    int _p;

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
    Kokkos::View<cdouble*, memory_space>
    operator()( const PositionArray& pos, const ScalarArray& scalar,
                std::size_t k,
                const Kokkos::Array<double, 3>& expansion_center ) const
    {
        int p = _p;
        Kokkos::View<cdouble*, memory_space> M( "M", ( p + 1 ) * ( p + 1 ) );
        Kokkos::deep_copy( M, cdouble( 0.0 ) );

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
                        auto val = scalar( i ) * Kokkos::pow( rho, n ) *
                                   Kokkos::conj( Ynm( n, m, alpha, beta ) );
                        Kokkos::atomic_add( &M( idx ), val );
                    }
                }
            } );
        return M;
    }
};

} // end namespace Kernel

} // end namespace Canopy

#endif // CANOPY_KERNELS_HPP
