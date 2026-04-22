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

#ifndef CANOPY_LAPLACE_KERNEL_HPP
#define CANOPY_LAPLACE_KERNEL_HPP

#include "Canopy_SphericalCoefficients.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <cstdint>

namespace Canopy
{

// ============================================================================
// Helper: device-callable double factorial n!!
// ============================================================================
template <class Scalar>
KOKKOS_INLINE_FUNCTION
Scalar double_factorial( int n )
{
    if ( n <= 1 )
        return 1.0;
    Scalar result = 1.0;
    for ( int i = n; i > 1; i -= 2 )
        result *= static_cast<Scalar>( i );
    return result;
}

// ============================================================================
// Device-callable associated Legendre polynomial P_n^m(x).
// Based on Greengard eqs. 3.33, 3.34.
// Stable upward recurrence in n.
// ============================================================================
template <class Scalar>
KOKKOS_INLINE_FUNCTION
Scalar Pnm_impl( int n, int m, Scalar x )
{
    if ( m < 0 || m > n )
        return 0.0;

    // P_m^m(x) = (-1)^m (2m-1)!! (1-x^2)^(m/2)
    Scalar pmm = double_factorial<Scalar>( 2 * m - 1 ) *
                 Kokkos::pow( 1.0 - x * x, 0.5 * m );
    if ( m % 2 == 1 )
        pmm = -pmm;

    if ( n == m )
        return pmm;

    Scalar pmmp1 = x * ( 2 * m + 1 ) * pmm;
    if ( n == m + 1 )
        return pmmp1;

    Scalar pnm2 = pmm;
    Scalar pnm1 = pmmp1;
    Scalar pn = 0.0;
    for ( int l = m + 2; l <= n; l++ )
    {
        pn = ( ( 2 * l - 1 ) * x * pnm1 -
               ( l + m - 1 ) * pnm2 ) /
             ( l - m );
        pnm2 = pnm1;
        pnm1 = pn;
    }
    return pn;
}

// ============================================================================
// Device-callable complex spherical harmonic Y_{n,m}(theta, phi).
// Uses Condon-Shortley phase convention. Per Greengard eq. 3.32.
// ============================================================================
template <class Scalar>
KOKKOS_INLINE_FUNCTION
Kokkos::complex<Scalar> Ynm( int n, int m, Scalar theta, Scalar phi )
{
    using complex = Kokkos::complex<Scalar>;

    const int mp = ( m < 0 ) ? -m : m;
    const Scalar x = Kokkos::cos( theta );

    const Scalar Pnm = Pnm_impl<Scalar>( n, mp, x );

    // Normalization sqrt((n-|m|)! / (n+|m|)!)
    const Scalar norm =
        Kokkos::sqrt( Kokkos::tgamma( static_cast<Scalar>( n - mp + 1 ) ) /
                      Kokkos::tgamma( static_cast<Scalar>( n + mp + 1 ) ) );

    const Scalar cos_mphi = Kokkos::cos( static_cast<Scalar>( m ) * phi );
    const Scalar sin_mphi = Kokkos::sin( static_cast<Scalar>( m ) * phi );

    const complex y =
        complex( norm * Pnm * cos_mphi, norm * Pnm * sin_mphi );

    // (-1)^m phase from Condon-Shortley, already applied in Pnm for m>=0.
    // For negative m, Ynm is defined via symmetry — but we have returned
    // (Pnm for |m|) * e^{i m phi}, with the Condon-Shortley phase built
    // into Pnm. The user's original code applies an extra (-1)^|m| that
    // we omit here because it's already in the P_m^m recurrence.
    return y;
}

// ============================================================================
// Convert a Cartesian offset (x, y, z) to spherical (rho, theta, phi).
//
// rho    = |r|                 >= 0
// theta  = polar angle         in [0, pi]
// phi    = azimuthal angle     in [-pi, pi]
// ============================================================================
template <class Scalar>
KOKKOS_INLINE_FUNCTION
void cartesian_to_spherical( Scalar x, Scalar y, Scalar z,
                             Scalar& rho, Scalar& theta, Scalar& phi )
{
    rho = Kokkos::sqrt( x * x + y * y + z * z );
    if ( rho > 0.0 )
    {
        theta = Kokkos::acos( z / rho );
    }
    else
    {
        theta = 0.0;
    }
    phi = Kokkos::atan2( y, x );
}

// ============================================================================
// LaplaceKernel
//
// Concrete FMM kernel for the 1/r Green's function (gravity, electrostatics).
//
// Template parameters:
//   Scalar - floating-point type (double or float)
//   P      - expansion order (max polynomial degree n)
//
// Exposed static constants:
//   max_order             = P
//   num_coeffs_per_cell   = (P+1)(P+2)/2 with symmetry
//   has_mplus_symmetry    = true
//
// Exposed static functions:
//   p2m_contribution      - add one particle's contribution to its leaf's M
//   m2m_translate         - translate a child's M to the parent's reference
// ============================================================================

template <class Scalar, int P>
struct LaplaceKernel
{
    using scalar_type = Scalar;
    using complex_type = Kokkos::complex<Scalar>;

    static constexpr int max_order = P;
    static constexpr int num_coeffs_per_cell =
        ( P + 1 ) * ( P + 2 ) / 2;
    static constexpr bool has_mplus_symmetry = true;

    // -----------------------------------------------------------------------
    // P2M: add a single particle's contribution to its leaf cell's
    // multipole coefficients.
    //
    //   For each (n, m) with 0 <= m <= n <= P:
    //     M_{n,m} += q * rho^n * Y_{n,-m}(alpha, beta)
    //
    //   (rho, alpha, beta) is the offset from the cell center to the
    //   particle in spherical coordinates.
    //
    // Parameters:
    //   charge       - source strength q
    //   dx, dy, dz   - particle position MINUS cell center (Cartesian)
    //   M_out        - view slice for this cell, size = num_coeffs_per_cell
    //                  (can be a subview or a strided view)
    //
    // This is the per-particle work. The caller parallelizes over
    // particles within a leaf and uses atomics to accumulate into M_out,
    // OR processes particles serially within a team thread.
    // -----------------------------------------------------------------------
    template <class MSliceType>
    KOKKOS_INLINE_FUNCTION
    static void p2m_contribution(
        Scalar charge,
        Scalar dx, Scalar dy, Scalar dz,
        const MSliceType& M_out )
    {
        Scalar rho, theta, phi;
        cartesian_to_spherical( dx, dy, dz, rho, theta, phi );

        // Precompute rho^n incrementally
        Scalar rho_pow_n = 1.0; // rho^0

        for ( int n = 0; n <= P; n++ )
        {
            for ( int m = 0; m <= n; m++ )
            {
                // Y_{n,-m} = (-1)^m conj(Y_{n,m}), but we call directly
                // through Ynm() with m' = -m for clarity.
                const complex_type Ynm_neg_m =
                    Ynm<Scalar>( n, -m, theta, phi );

                const complex_type contrib =
                    charge * rho_pow_n * Ynm_neg_m;

                const int idx = coeff_index( n, m );
                // Atomic add of complex by splitting into real/imag
                Kokkos::atomic_add( &M_out( idx ).real(), contrib.real() );
                Kokkos::atomic_add( &M_out( idx ).imag(), contrib.imag() );
            }
            rho_pow_n *= rho;
        }
    }

    // -----------------------------------------------------------------------
    // m2m_translate: contribute a single child's multipole to the parent's
    // multipole, translating from child center to parent center.
    //
    // For the Greengard un-normalized harmonics, the M2M formula follows
    // directly from the addition theorem for regular solid harmonics:
    //
    //   M_{j,k}^parent += sum_{n=0}^{j} sum_{m=-n}^{n}
    //       rho^n * Y_{n,-m}(alpha, beta) * M_{j-n, k-m}^child
    //
    // where (rho, alpha, beta) is child_center - parent_center in
    // spherical coordinates.
    //
    // Parameters:
    //   M_child      - 2D view for the child cell (shape 1 x num_coeffs)
    //   dx, dy, dz   - child_center - parent_center (Cartesian)
    //   M_parent_out - 1D view for the parent cell; contributions added to
    //                  entries indexed by coeff_index(j, k) with k >= 0.
    //   team_member  - Kokkos team member for hierarchical parallelism
    //
    // Parallelism model: team threads cooperate over output (j, k) pairs.
    // The inner (n, m) sum is serial within each thread.
    // -----------------------------------------------------------------------
    template <class TeamMember, class MChildType, class MParentType>
    KOKKOS_INLINE_FUNCTION
    static void m2m_translate(
        const TeamMember& team_member,
        const MChildType& M_child,
        Scalar dx, Scalar dy, Scalar dz,
        const MParentType& M_parent_out )
    {
        Scalar rho, theta, phi;
        cartesian_to_spherical( dx, dy, dz, rho, theta, phi );

        // Parallelize over the triangular set of output (j, k) with k >= 0.
        const int num_outputs = num_coeffs_per_cell;

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team_member, num_outputs ),
            [&]( const int out_idx ) {
                // Recover (j, k) from flat index.
                // out_idx = j*(j+1)/2 + k, 0 <= k <= j <= P.
                int j = ( static_cast<int>(
                             Kokkos::floor( ( -1.0 +
                                              Kokkos::sqrt(
                                                  1.0 +
                                                  8.0 *
                                                      static_cast<double>(
                                                          out_idx ) ) ) *
                                            0.5 ) ) );
                // Correct for floating-point rounding
                while ( j * ( j + 1 ) / 2 > out_idx )
                    j--;
                while ( ( j + 1 ) * ( j + 2 ) / 2 <= out_idx )
                    j++;
                const int k = out_idx - j * ( j + 1 ) / 2;

                complex_type accum( 0.0, 0.0 );

                // Inner sum: n from 0 to j, m from -n to n.
                Scalar rho_pow_n = 1.0;
                for ( int n = 0; n <= j; n++ )
                {
                    for ( int m = -n; m <= n; m++ )
                    {
                        const int j_minus_n = j - n;
                        const int k_minus_m = k - m;
                        const int abs_km = ( k_minus_m < 0 )
                                               ? -k_minus_m
                                               : k_minus_m;

                        // M_{j-n, k-m}^child is zero if |k-m| > j-n
                        if ( abs_km > j_minus_n )
                            continue;

                        const complex_type M_child_val =
                            get_coeff<Scalar>( M_child, 0,
                                               j_minus_n, k_minus_m, P );

                        const complex_type Y =
                            Ynm<Scalar>( n, -m, theta, phi );

                        accum += M_child_val * rho_pow_n * Y;
                    }
                    rho_pow_n *= rho;
                }

                M_parent_out( out_idx ) += accum;
            } );
    }
};

} // namespace Canopy

#endif // CANOPY_LAPLACE_KERNEL_HPP
