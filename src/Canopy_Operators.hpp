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

/*!
  \file Canopy_Operators.hpp
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

namespace Operator
{

// using complex = Kokkos::complex<Scalar>;
// constexpr auto pi = Kokkos::numbers::pi_v<Scalar>;

// Cartesian to spherical coorindates: (x, y, z) -> (r,theta,phi)
template <class Scalar>
KOKKOS_INLINE_FUNCTION
void cart2sph( Scalar x, Scalar y, Scalar z, Scalar& r, Scalar& theta,
               Scalar& phi )
{
    r = Kokkos::sqrt( x * x + y * y + z * z );
    theta = ( r == 0.0 ? 0.0 : Kokkos::acos( z / r ) ); // polar angle
    phi = Kokkos::atan2( y, x );                        // azimuth
}

// Convert spherical gradients to cartesian gradients
template <class Scalar>
KOKKOS_INLINE_FUNCTION
Kokkos::Array<Scalar,3>
partials_to_cartesian_gradient(const Kokkos::Array<Scalar,3>& dPartial,
    Scalar r, Scalar theta, Scalar phi )
{
    const Scalar dPartial_dr = dPartial[0];
    const Scalar dPartial_dtheta = dPartial[1];
    const Scalar dPartial_dphi = dPartial[2];

    const Scalar st = Kokkos::sin(theta);
    const Scalar ct = Kokkos::cos(theta);
    const Scalar sp = Kokkos::sin(phi);
    const Scalar cp = Kokkos::cos(phi);

    // spherical vector components of ∇Φ
    const Scalar g_r = dPartial_dr;
    const Scalar g_theta = (r > 0.0) ? dPartial_dtheta / r : 0.0;
    const Scalar g_phi   = (r > 0.0 && Kokkos::abs(st) > 1e-14)
                           ? dPartial_dphi / (r * st)
                           : 0.0;

    Kokkos::Array<Scalar,3> grad;
    grad[0] = g_r * st * cp + g_theta * ct * cp - g_phi * sp;
    grad[1] = g_r * st * sp + g_theta * ct * sp + g_phi * cp;
    grad[2] = g_r * ct      - g_theta * st;
    return grad;
}

// Factorial Scalar: (2m-1)!!
template <class Scalar>
KOKKOS_INLINE_FUNCTION
Scalar factorial( int k )
{
    // Γ(k+1) = k!
    return static_cast<Scalar>(Kokkos::tgamma( static_cast<Scalar>( k ) + 1.0 ));
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION
Scalar double_factorial( int m )
{
    // (2m − 1)!! = (2m)! / (2^m m!)
    // use gamma functions to avoid integer overflow
    const Scalar two_m = static_cast<Scalar>( 2 * m );
    const Scalar m_d = static_cast<Scalar>( m );
    return static_cast<Scalar>(Kokkos::tgamma( two_m + 1.0 ) /
           ( Kokkos::pow( 2.0, m_d ) * Kokkos::tgamma( m_d + 1.0 ) ));
}

namespace Scalar
{
//---------------------------------------------------------------------------//

/**
 * Implementation of std::assoc_legendre that is callable on the device.
 * Per equations 3.33 and 3.34 in Greengard.
 */
template <class Scalar>
KOKKOS_INLINE_FUNCTION
Scalar Pnm_impl( int n, int m, Scalar x )
{
    if ( m < 0 || m > n )
        return 0.0; // undefined outside this range

    // P_m^m(x)
    Scalar pmm = double_factorial<Scalar>( m ) * std::pow( 1.0 - x * x, 0.5 * m );
    if ( m % 2 == 1 )
        pmm = -pmm; // (-1)^m factor

    if ( n == m )
        return pmm;

    // P_{m+1}^m(x)
    Scalar pmmp1 = x * ( 2 * m + 1 ) * pmm;
    if ( n == m + 1 )
        return pmmp1;

    // Upward recurrence
    Scalar pnm2 = pmm;
    Scalar pnm1 = pmmp1;
    Scalar pn = 0.0;
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
 * Per equation 3.32 in Greengard.
 */
template <class Scalar>
KOKKOS_INLINE_FUNCTION
Kokkos::complex<Scalar> Ynm( int n, int m, Scalar theta, Scalar phi )
{
    using complex = Kokkos::complex<Scalar>;
    
    const int mp = Kokkos::abs( m );
    const Scalar x = Kokkos::cos( theta );

    const Scalar Pnm = Pnm_impl( n, mp, x );

    // See equation 3.27, Greengard for including sqrt((2n+1 / 4pi))
    const Scalar norm = Kokkos::sqrt( Kokkos::tgamma( n - mp + 1 ) /
                                      Kokkos::tgamma( n + mp + 1 ) );

    // Equation 3.32, Greengard
    const complex y = norm * Pnm * Kokkos::polar( 1.0, Scalar( m ) * phi );

    const Scalar phase = ( m >= 0 ? ( ( m % 2 ) ? -1.0 : 1.0 ) // (-1)^m
                                  : ( ( ( -m ) % 2 ) ? -1.0 : 1.0 ) );

    return y * phase;
}

/**
 * Compute offset into flattened multipole array
 * (n,m) -> index
 */
KOKKOS_INLINE_FUNCTION
int index( int n, int m ) { return n * n + ( m + n ); }

template <std::size_t p, class Scalar>
KOKKOS_INLINE_FUNCTION void
p2m( const Kokkos::Array<Scalar, 3>& pos,
     const Scalar sc,
     const Kokkos::Array<Scalar, 3>& expansion_center,
     Kokkos::Array<Kokkos::complex<Scalar>, ( p + 1 ) * ( p + 1 )>& M )
{
    Scalar dx = pos[0] - expansion_center[0];
    Scalar dy = pos[1] - expansion_center[1];
    Scalar dz = pos[2] - expansion_center[2];

    Scalar rho, alpha, beta;
    cart2sph( dx, dy, dz, rho, alpha, beta );

    // Equation 3.36, Greengard
    for ( int n = 0; n <= p; ++n )
    {
        for ( int m = -n; m <= n; ++m )
        {
            int idx = index( n, m );
            // Equation 3.37, Greengard
            // auto norm = Kokkos::sqrt(( ( 2.0 * n + 1 ) / ( 4.0 *
            // pi ) ));
            auto val = sc * Kokkos::pow( rho, n ) *
                        Ynm( n, -m, alpha, beta ); // / norm;
            M[idx] += val;
            // printf("k%d, n%d, m%d setting index: %d\n",
            //     i, n, m, idx);
        }
    }
}

/**
 * Operator calculates the kernel for scalar-based multipoles
 * and return the multipole coefficient matrix flattened into a
 * 1D vector.
 */
template <class MemorySpace, class ExecutionSpace, class Scalar>
struct P2M
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using complex = Kokkos::complex<Scalar>;

    P2M( int p )
        : _p( p )
    {
        _M = Kokkos::View<complex*, memory_space>( "M", ( p + 1 ) * ( p + 1 ) );
        clear();
    }

  private:
    int _p;
    Kokkos::View<complex*, memory_space> _M;

  public:
    auto coefficients() { return _M; }

    /**
     * Clear coefficients
     */
    void clear() { Kokkos::deep_copy( _M, complex( 0.0, 0.0 ) ); }

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
    template <class PositionArray, class ScalarArray, class CenterArray>
    void operator()( const PositionArray& pos, const ScalarArray& sc,
                     std::size_t k,
                     const CenterArray& expansion_center ) const
    {
        int p = _p;
        auto M = _M;

        // Further optimize this code for running on the device
        Kokkos::parallel_for(
            "compute multipole coefficients",
            Kokkos::RangePolicy<execution_space>( 0, k ),
            KOKKOS_LAMBDA( const int i ) {
                // Construct coordinates relative to the expansion center.
                Scalar dx = pos( i, 0 ) - expansion_center[0];
                Scalar dy = pos( i, 1 ) - expansion_center[1];
                Scalar dz = pos( i, 2 ) - expansion_center[2];

                Scalar rho, alpha, beta;
                cart2sph( dx, dy, dz, rho, alpha, beta );

                // Equation 3.36, Greengard
                for ( int n = 0; n <= p; ++n )
                {
                    for ( int m = -n; m <= n; ++m )
                    {
                        int idx = index( n, m );
                        // Equation 3.37, Greengard
                        // auto norm = Kokkos::sqrt(( ( 2.0 * n + 1 ) / ( 4.0 *
                        // pi ) ));
                        auto val = sc( i ) * Kokkos::pow( rho, n ) *
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
 * Equation 3.26, Greengard
 */
template <class Scalar>
KOKKOS_INLINE_FUNCTION
Scalar compute_A( int n, int m )
{
    Scalar denom = Kokkos::sqrt( factorial<Scalar>( n - m ) * factorial<Scalar>( n + m ) );
    Scalar sign = ( n % 2 == 0 ) ? 1.0 : -1.0; // (-1)^n
    return sign / denom;
}

/**
 * Equation 3.43, Greengard
 */
template <class Scalar>
KOKKOS_INLINE_FUNCTION
Scalar compute_J_3_43( int n, int m )
{
    if ( n * m < 0 )
    {
        int min_abs = ( Kokkos::abs( n ) < Kokkos::abs( m ) )
                          ? Kokkos::abs( n )
                          : Kokkos::abs( m );
        return Scalar( ( min_abs % 2 == 0 ) ? 1 : -1 ); // (-1)^min(|n|,|m|)
    }
    else
    {
        return 1.0;
    }
}

/**
 * Equation 3.43, Greengard
 */
template <class Scalar>
KOKKOS_INLINE_FUNCTION
Scalar compute_J_3_49( int n, int m )
{
    if ( n * m > 0 )
    {
        int min_abs = ( Kokkos::abs( n ) < Kokkos::abs( m ) )
                          ? Kokkos::abs( n )
                          : Kokkos::abs( m );
        return Kokkos::pow( -1, m ) *
               Scalar( ( min_abs % 2 == 0 )
                           ? 1
                           : -1 ); // (-1)^(m)*(-1)^min(|n|,|m|)
    }
    else
    {
        return Kokkos::pow( -1, m );
    }
}

/**
 * Equation 3.54, Greengard
 */
template <class Scalar>
KOKKOS_INLINE_FUNCTION
Scalar compute_J_3_54( int n, int m, int m_p )
{
    auto minus_one_pow = []( int k ) -> Scalar
    { return ( k % 2 == 0 ) ? 1.0 : -1.0; };

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

template <std::size_t p, class Scalar>
KOKKOS_INLINE_FUNCTION void
m2m( const Kokkos::Array<Kokkos::complex<Scalar>, ( p + 1 ) * ( p + 1 )>& M_orig,
     const Kokkos::Array<Scalar, 3>& center_orig,
     Kokkos::Array<Kokkos::complex<Scalar>, ( p + 1 ) * ( p + 1 )>& M_new )
{
    using complex = Kokkos::complex<Scalar>;

    // Spherical coords for the displacement
    Scalar rho, alpha, beta;
    cart2sph( center_orig[0], center_orig[1], center_orig[2], rho, alpha,
                beta );

    for ( int j = 0; j <= p; ++j )
    {
        for ( int k = -j; k <= j; ++k )
        {
            complex Mjk( 0.0, 0.0 );

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
                    complex O = M_orig[orig_index];

                    // Values for eq 3.57
                    const Scalar J = compute_J_3_43<Scalar>( m, k_m );
                    const Scalar A_nm = compute_A<Scalar>( n, m );
                    const Scalar A_jn_km = compute_A<Scalar>( j_n, k_m );
                    const Scalar A_jk = compute_A<Scalar>( j, k );
                    Scalar rho_n = Kokkos::pow( rho, n );

                    // M_child already has its Y_nm normalized. The Ynm
                    // function performs normalization internally, so we
                    // need to un-normalize it after calling Ynm to avoid
                    // Scalar normalization.
                    auto Y_nm = Ynm( n, -m, alpha, beta );
                    Mjk += ( O * J * A_nm * A_jn_km * rho_n * Y_nm ) / A_jk;
                }
            }
            int parent_idx = index( j, k );
            M_new[parent_idx] += Mjk;
        }
    }
}

/**
 * Operator calculates the multipole expansions about the centers of
 * cells not in the leaf layer using the potential field from its child cells
 * Per Greengard, page 68, step 2.
 */
template <class MemorySpace, class ExecutionSpace, class Scalar>
struct M2M
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using complex = Kokkos::complex<Scalar>;

    M2M( int p )
        : _p( p )
    {
        _M = Kokkos::View<complex*, memory_space>( "M", ( p + 1 ) * ( p + 1 ) );
        clear();
    }

  private:
    int _p;
    Kokkos::View<complex*, memory_space> _M;

  public:
    auto coefficients() { return _M; }

    /**
     * Clear coefficients.
     */
    void clear() { Kokkos::deep_copy( _M, complex( 0.0, 0.0 ) ); }

    template <class MultipoleVector>
    void operator()( const MultipoleVector& M_orig,
                     const Kokkos::Array<Scalar, 3>& center_orig ) const
    {
        const int p = _p;
        auto M = _M;

        // Spherical coords for the displacement
        Scalar rho, alpha, beta;
        cart2sph( center_orig[0], center_orig[1], center_orig[2], rho, alpha,
                  beta );

        for ( int j = 0; j <= p; ++j )
        {
            for ( int k = -j; k <= j; ++k )
            {
                complex Mjk( 0.0, 0.0 );

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
                        complex O = M_orig( orig_index );

                        // Values for eq 3.57
                        const Scalar J = compute_J_3_43<Scalar>( m, k_m );
                        const Scalar A_nm = compute_A<Scalar>( n, m );
                        const Scalar A_jn_km = compute_A<Scalar>( j_n, k_m );
                        const Scalar A_jk = compute_A<Scalar>( j, k );
                        Scalar rho_n = Kokkos::pow( rho, n );

                        // M_child already has its Y_nm normalized. The Ynm
                        // function performs normalization internally, so we
                        // need to un-normalize it after calling Ynm to avoid
                        // Scalar normalization.
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
 * Theorem 2.4 in Cheng.
 * Callable on the device
 */
template <std::size_t p, class Scalar>
KOKKOS_INLINE_FUNCTION void
m2l( const Kokkos::Array<Kokkos::complex<Scalar>, ( p + 1 ) * ( p + 1 )>& O,
     Kokkos::Array<Kokkos::complex<Scalar>, ( p + 1 ) * ( p + 1 )>& L,
     const Kokkos::Array<Scalar, 3>& O_center )
{
    using complex = Kokkos::complex<Scalar>;

    // Spherical coords of O_center
    Scalar rho, alpha, beta;
    cart2sph( O_center[0], O_center[1], O_center[2], rho, alpha, beta );

    for ( int j = 0; j <= p; ++j )
    {
        for ( int k = -j; k <= j; ++k )
        {
            complex Ljk( 0.0, 0.0 );

            for ( int n = 0; n <= p; ++n )
            {
                for ( int m = -n; m <= n; ++m )
                {
                    // Originally, Greengard eq. 3.60 was used, but there
                    // was a bug getting the potential calculated from the
                    // local expansion to converge to the potential
                    // calculated directly. Instead, Cheng eq. 17 is used to
                    // compute local expansions.

                    // Numerator
                    complex O_nm = O[index( n, m )];
                    complex i_unit( 0.0, 1.0 );
                    auto power = Kokkos::abs( k - m ) - Kokkos::abs( k ) -
                                 Kokkos::abs( m );
                    auto i_term = Kokkos::pow( i_unit, power );
                    auto A_nm = compute_A<Scalar>( n, m );
                    auto A_jk = compute_A<Scalar>( j, k );
                    auto Y_jn_mk = Ynm( j + n, m - k, alpha, beta );

                    // Denominator
                    auto sign = ( n % 2 == 0 ) ? 1.0 : -1.0;
                    auto A_jn_mk = compute_A<Scalar>( j + n, m - k );
                    auto rho_jn = Kokkos::pow( rho, j + n + 1 );

                    // Compute L_jk partial term
                    Ljk += ( O_nm * i_term * A_nm * A_jk * Y_jn_mk ) /
                           ( sign * A_jn_mk * rho_jn );
                }
            }
            L[index( j, k )] += Ljk;
        }
    }
}

/**
 * Convert multipole expansions into local expansions using
 * Theorem 2.4 in Cheng.
 */
template <class MemorySpace, class ExecutionSpace, class Scalar>
struct M2L
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using complex = Kokkos::complex<Scalar>;

    M2L( int p )
        : _p( p )
    {
        _L = Kokkos::View<complex*, memory_space>( "L", ( p + 1 ) * ( p + 1 ) );
        clear();
    }

  private:
    int _p;
    Kokkos::View<complex*, memory_space> _L;

  public:
    auto coefficients() { return _L; }

    /**
     * Clear coefficients
     */
    void clear() { Kokkos::deep_copy( _L, complex( 0.0, 0.0 ) ); }

    /**
     * Compute local coefficients L[n][m] up to order p
     *
     * @param O multipole coefficients centered around O_center.
     * @param O_center the center of multipole coefficients O.
     */
    template <class MultipoleVector>
    void operator()( const MultipoleVector& O,
                     const Kokkos::Array<Scalar, 3>& O_center ) const
    {
        int p = _p;
        auto L = _L;

        // Spherical coords of O_center
        Scalar rho, alpha, beta;
        cart2sph( O_center[0], O_center[1], O_center[2], rho, alpha, beta );

        // Optimize this code for running on the device
        for ( int j = 0; j <= p; ++j )
        {
            for ( int k = -j; k <= j; ++k )
            {
                complex Ljk( 0.0, 0.0 );

                for ( int n = 0; n <= p; ++n )
                {
                    for ( int m = -n; m <= n; ++m )
                    {
                        // Originally, Greengard eq. 3.60 was used, but there
                        // was a bug getting the potential calculated from the
                        // local expansion to converge to the potential
                        // calculated directly. Instead, Cheng eq. 17 is used to
                        // compute local expansions.

                        // Numerator
                        complex O_nm = O( index( n, m ) );
                        complex i_unit( 0.0, 1.0 );
                        auto power = Kokkos::abs( k - m ) - Kokkos::abs( k ) -
                                     Kokkos::abs( m );
                        auto i_term = Kokkos::pow( i_unit, power );
                        auto A_nm = compute_A<Scalar>( n, m );
                        auto A_jk = compute_A<Scalar>( j, k );
                        auto Y_jn_mk = Ynm( j + n, m - k, alpha, beta );

                        // Denominator
                        auto sign = ( n % 2 == 0 ) ? 1.0 : -1.0;
                        auto A_jn_mk = compute_A<Scalar>( j + n, m - k );
                        auto rho_jn = Kokkos::pow( rho, j + n + 1 );

                        // Compute L_jk partial term
                        Ljk += ( O_nm * i_term * A_nm * A_jk * Y_jn_mk ) /
                               ( sign * A_jn_mk * rho_jn );
                    }
                }
                L( index( j, k ) ) += Ljk;
            }
        }
    }
};

/**
 * Translate local expansions.
 * Theorem 5 in Cheng.
 * Callable on the device
 */
template <std::size_t p, class Scalar>
KOKKOS_INLINE_FUNCTION void
l2l( const Kokkos::Array<Kokkos::complex<Scalar>, ( p + 1 ) * ( p + 1 )>& L_orig,
     Kokkos::Array<Kokkos::complex<Scalar>, ( p + 1 ) * ( p + 1 )>& L,
     const Kokkos::Array<Scalar, 3>& L_center )
{
    using complex = Kokkos::complex<Scalar>;

    // Spherical coords of L_center
    Scalar rho, alpha, beta;
    cart2sph( L_center[0], L_center[1], L_center[2], rho, alpha, beta );

    for ( int j = 0; j <= p; ++j )
    {
        for ( int k = -j; k <= j; ++k )
        {
            complex Ljk( 0.0, 0.0 );

            for ( int n = j; n <= p; ++n )
            {
                for ( int m = -n; m <= n; ++m )
                {
                    // Skip regions where Y_nm is invalid.
                    if ( std::abs( m - k ) > ( n - j ) )
                        continue;

                    // Numerator
                    complex O_nm = L_orig[ index( n, m ) ];
                    complex i_unit( 0.0, 1.0 );
                    auto power = Kokkos::abs( m ) - Kokkos::abs( m - k ) -
                                    Kokkos::abs( k );
                    auto i_term = Kokkos::pow( i_unit, power );
                    auto A_nj_mk = compute_A<Scalar>( n - j, m - k );
                    auto A_jk = compute_A<Scalar>( j, k );
                    auto Y_nj_mk = Ynm( n - j, m - k, alpha, beta );
                    auto rho_nj = Kokkos::pow( rho, n - j );

                    // Denominator
                    auto sign = ( ( n + j ) % 2 == 0 ) ? 1.0 : -1.0;
                    auto A_nm = compute_A<Scalar>( n, m );

                    // Compute L_jk partial term
                    Ljk += ( O_nm * i_term * A_nj_mk * A_jk * Y_nj_mk *
                                rho_nj ) /
                            ( sign * A_nm );
                    // printf("j: %d, k: %d, n: %d, m: %d, Y: (%.3lf,
                    // %.3lf), O_nm: (%.3lf, %.3lf), Ljk_piece: (%.3lf,
                    // %.3lf)\n",
                    //     j, k, n, m,
                    //     Y_nj_mk.real(), Y_nj_mk.imag(),
                    //     O_nm.real(), O_nm.imag(), Ljk.real(),
                    //     Ljk.imag());
                }
            }
            L[ index( j, k ) ] = Ljk;
        }
    }
}

/**
 * Translate local expansions.
 * Theorem 5 in Cheng.
 */
template <class MemorySpace, class ExecutionSpace, class Scalar>
struct L2L
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using complex = Kokkos::complex<Scalar>;

    L2L( int p )
        : _p( p )
    {
        _L = Kokkos::View<complex*, memory_space>( "L", ( p + 1 ) * ( p + 1 ) );
        clear();
    }

  private:
    int _p;
    Kokkos::View<complex*, memory_space> _L;

  public:
    auto coefficients() { return _L; }

    /**
     * Clear coefficients.
     */
    void clear() { Kokkos::deep_copy( _L, complex( 0.0, 0.0 ) ); }

    /**
     * Translate local coefficients L[n][m] up to order p
     *
     * @param L_orig local coefficients centered around O_center.
     * @param L_center the center to translate L_orig to.
     */
    template <class LocalVector>
    void operator()( const LocalVector& O,
                     const Kokkos::Array<Scalar, 3>& L_center ) const
    {
        int p = _p;
        auto L = _L;

        // Spherical coords of L_center
        Scalar rho, alpha, beta;
        cart2sph( L_center[0], L_center[1], L_center[2], rho, alpha, beta );

        // Optimize this code for running on the device
        for ( int j = 0; j <= p; ++j )
        {
            for ( int k = -j; k <= j; ++k )
            {
                complex Ljk( 0.0, 0.0 );

                for ( int n = j; n <= p; ++n )
                {
                    for ( int m = -n; m <= n; ++m )
                    {
                        // Skip regions where Y_nm is invalid.
                        if ( std::abs( m - k ) > ( n - j ) )
                            continue;

                        // Numerator
                        complex O_nm = O( index( n, m ) );
                        complex i_unit( 0.0, 1.0 );
                        auto power = Kokkos::abs( m ) - Kokkos::abs( m - k ) -
                                     Kokkos::abs( k );
                        auto i_term = Kokkos::pow( i_unit, power );
                        auto A_nj_mk = compute_A<Scalar>( n - j, m - k );
                        auto A_jk = compute_A<Scalar>( j, k );
                        auto Y_nj_mk = Ynm( n - j, m - k, alpha, beta );
                        auto rho_nj = Kokkos::pow( rho, n - j );

                        // Denominator
                        auto sign = ( ( n + j ) % 2 == 0 ) ? 1.0 : -1.0;
                        auto A_nm = compute_A<Scalar>( n, m );

                        // Compute L_jk partial term
                        Ljk += ( O_nm * i_term * A_nj_mk * A_jk * Y_nj_mk *
                                 rho_nj ) /
                               ( sign * A_nm );
                        // printf("j: %d, k: %d, n: %d, m: %d, Y: (%.3lf,
                        // %.3lf), O_nm: (%.3lf, %.3lf), Ljk_piece: (%.3lf,
                        // %.3lf)\n",
                        //     j, k, n, m,
                        //     Y_nj_mk.real(), Y_nj_mk.imag(),
                        //     O_nm.real(), O_nm.imag(), Ljk.real(),
                        //     Ljk.imag());
                    }
                }
                L( index( j, k ) ) = Ljk;
            }
        }
    }
};

/**
 * Functions for partial derivatives, from Rankin, for implementation of eq. A.11.
 * In Rankin, the F*_nm((r, theta, phi)) term in the equivalent of:
 * Kokkos::pow(r, n) * Canopy::Operator::Scalar::Ynm( n, m, theta, phi ); in Canopy.
 * This term is passed as "val" in the following functions.
 * Essentially, we are taking partial derivatives of the following function,
 * where 'a' is a constant factor dependant on 'n' and 'm' and 'i'
 * is the imaginary number i:
 * Y_nm(alpha, beta) = a * P_nm(cos(alpha)) * exp(i * m * beta)
 * Incorporated into the summation for the local-to-potential calculation:
 * ∑_nm (L_nm * rho^n * Y_nm(alpha, beta)) 
 */

/**
 * Given val = r^n * Y_nm(θ, phi)
 *  d/d_r (r^n * Y_nm(θ, phi)) = n * r^(n-1) Y_nm(θ, phi)
 *                                      = n/r * (r^n * Y_nm(θ, phi))
 *                                      = n/r * val
 */ 
template <class Scalar>
KOKKOS_INLINE_FUNCTION
Kokkos::complex<Scalar> d_dr(const Scalar r, const int n, const Kokkos::complex<Scalar> val)
{
    if (Kokkos::abs(r) < 1e-10)
        return 0;

    return (Scalar(n) / r) * val;
}

/**
 * Equation A.18
 * d/d_theta (r^n * Y_nm(θ, phi))
 *  = r^n * d/d_theta (Y_nm(θ, phi))
 *  = r^n * d_theta(a * P_nm(cos(θ)) * exp(i * m * phi))
 *  = r^n * (n * cos(θ) * P_nm(cos(θ)) - (n + m) * P_(n-1)_m(cos(θ)))) / sin(θ) for m >= 0
 * XXX - optimize to pass val so we don't need to recompute
 */
template <class Scalar>
KOKKOS_INLINE_FUNCTION
Kokkos::complex<Scalar> d_dtheta(const Scalar r, const Scalar theta, const Scalar phi, const int n, const int m)
{
    using complex = Kokkos::complex<Scalar>;
    // Compute derivative of associated legendre polynomial in terms of theta
    // Defined as P_nm(x) in this code.

    // Start with setup for Y_nm
    const int mp = Kokkos::abs(m);
    const Scalar cx = Kokkos::cos(theta);
    const Scalar sx = Kokkos::sin(theta);

    const Scalar Pnm = Pnm_impl(n, mp, cx);
    const Scalar Pnm1 = (n > 0) ? Pnm_impl(n-1, mp, cx) : 0.0;

    // See equation 3.27, Greengard for including sqrt((2n+1 / 4pi))
    Scalar norm = Kokkos::sqrt( Kokkos::tgamma( n - mp + 1 ) /
                                Kokkos::tgamma( n + mp + 1 ) );
    
    // Const exp(i * m* phi) from Y_nm
    const complex e_imp = Kokkos::polar(1.0, Scalar(m) * phi);

    // Const phase from Y_nm
    const Scalar phase = ( m >= 0 ? ( ( m % 2 ) ? -1.0 : 1.0 ) // (-1)^m
                            : ( ( ( -m ) % 2 ) ? -1.0 : 1.0 ) );

    // If theta is near zero, set 1/sin(θ) to zero to avoid errors.
    const Scalar inv_sin_theta = (Kokkos::abs(sx) < 1e-10) ? 0.0 : (1.0 / sx);

    // Compute derivative: d/dtheta P_n^mp(cos(θ)) = ( n cos(θ) P_n_mp - (n+mp) P_(n-1)_mp ) / sin(θ)
    const Scalar dP_dtheta = ( Scalar(n) * cx * Pnm - Scalar(n + mp) * Pnm1) * inv_sin_theta;

    // Multiply by constant terms and return
    return complex(phase * norm * dP_dtheta, 0.0) * e_imp;
}

// Equation A.19, with a sign flip because Rankin uses opposite signs in Y_nm.
template <class Scalar>
KOKKOS_INLINE_FUNCTION
Kokkos::complex<Scalar> d_dphi(const int m, const Kokkos::complex<Scalar> val)
{
    using complex = Kokkos::complex<Scalar>;
    return complex(0.0, Scalar(m)) * val;
}


} // end namespace Scalar

} // end namespace Operator

} // end namespace Canopy

#endif // CANOPY_KERNELS_HPP
