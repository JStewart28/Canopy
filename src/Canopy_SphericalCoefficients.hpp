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

#ifndef CANOPY_SPHERICAL_COEFFICIENTS_HPP
#define CANOPY_SPHERICAL_COEFFICIENTS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <cstdint>

namespace Canopy
{

// ============================================================================
// SphericalCoefficients — storage and indexing helpers for FMM expansion
// coefficients M_{n,m} and L_{n,m}.
//
// For real-sourced kernels (Laplace, gravity) in this code's Ynm
// convention (Y_{n,-m} = conj(Y_{n,m})), the coefficients satisfy
//   M_{n,-m} = conj(M_{n,m})
// so we only store m >= 0, reducing storage and work by roughly 2x.
//
// Storage layout (triangular, per cell):
//   idx(n, m) = n*(n+1)/2 + m     for m = 0, 1, ..., n
//   Total coefficients per cell = (p+1)(p+2)/2 where p = max_order
//
// Coefficients for all cells are stored in a single 2D Kokkos::View:
//   coeffs(cell_index, coeff_index)
// This gives coalesced access when cell_index is the slow dimension
// and team threads cooperate over coeff_index.
// ============================================================================

// --------------------------------------------------------------------------
// Compute the number of stored coefficients per cell for a given order p,
// assuming m >= 0 symmetry is used.
// --------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
constexpr int num_coeffs_symmetric( int p )
{
    return ( p + 1 ) * ( p + 2 ) / 2;
}

// --------------------------------------------------------------------------
// Triangular index for (n, m) with m >= 0:
//   idx(n, m) = n*(n+1)/2 + m
//
// Valid range: 0 <= m <= n <= p
// --------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
constexpr int coeff_index( int n, int m )
{
    return n * ( n + 1 ) / 2 + m;
}

// --------------------------------------------------------------------------
// Retrieve M_{n,m} from a coefficient array, handling negative m via
// symmetry for real-sourced kernels.
//
//   For m >= 0: direct lookup
//   For m < 0:  M_{n,m} = (-1)^|m| * conj(M_{n,|m|})
//
// Out-of-range queries (|m| > n or n > p) return zero.
// --------------------------------------------------------------------------
template <class Scalar, class View>
KOKKOS_INLINE_FUNCTION
Kokkos::complex<Scalar> get_coeff( const View& coeffs, int cell_idx,
                                   int n, int m, int max_order )
{
    using complex = Kokkos::complex<Scalar>;

    // Out of range
    if ( n < 0 || n > max_order )
        return complex( 0.0, 0.0 );

    const int abs_m = ( m < 0 ) ? -m : m;
    if ( abs_m > n )
        return complex( 0.0, 0.0 );

    const complex val = coeffs( cell_idx, coeff_index( n, abs_m ) );

    if ( m >= 0 )
        return val;

    // m < 0: apply symmetry  M_{n,-|m|} = conj(M_{n,|m|})
    return complex( val.real(), -val.imag() );
}

// --------------------------------------------------------------------------
// A_{n,m} normalization constant used in Greengard/Rokhlin translations.
//   A_{n,m} = (-1)^n / sqrt((n-m)! * (n+m)!)
//
// This is used both in M2M (upward translation) and M2L/L2L operators.
// Defined for |m| <= n, zero otherwise.
//
// To avoid computing factorials repeatedly at run time, we can tabulate
// these in a Kokkos::View for a given max_order. See
// build_A_coefficients() below.
// --------------------------------------------------------------------------
template <class Scalar>
KOKKOS_INLINE_FUNCTION
Scalar A_coeff( int n, int m )
{
    const int abs_m = ( m < 0 ) ? -m : m;
    if ( abs_m > n )
        return 0.0;

    // (n-m)! and (n+m)!
    // Use tgamma for device-callable factorials
    const Scalar num_fact = Kokkos::tgamma( static_cast<Scalar>( n - abs_m + 1 ) );
    const Scalar den_fact = Kokkos::tgamma( static_cast<Scalar>( n + abs_m + 1 ) );

    const Scalar sign = ( n % 2 == 0 ) ? 1.0 : -1.0;
    return sign / Kokkos::sqrt( num_fact * den_fact );
}

// --------------------------------------------------------------------------
// Pre-compute A_{n,m} for all (n, m) with -n <= m <= n, n <= max_order.
//
// Stored in a flat View with index:
//   a_index(n, m) = n*n + n + m   (valid for -n <= m <= n)
//
// This layout wastes a little space (uses (p+1)^2 slots instead of
// (p+1)(2p+1)/something triangular) but gives trivially cheap indexing.
// --------------------------------------------------------------------------
template <class Scalar, class MemorySpace>
Kokkos::View<Scalar*, MemorySpace> build_A_coefficients( int max_order )
{
    const int num_entries = ( max_order + 1 ) * ( max_order + 1 );
    Kokkos::View<Scalar*, MemorySpace> A( "A_coeffs", num_entries );
    auto h_A = Kokkos::create_mirror_view( A );

    for ( int n = 0; n <= max_order; n++ )
    {
        for ( int m = -n; m <= n; m++ )
        {
            h_A( n * n + n + m ) = A_coeff<Scalar>( n, m );
        }
    }

    Kokkos::deep_copy( A, h_A );
    return A;
}

// Index into the A-coefficient table.
KOKKOS_INLINE_FUNCTION
constexpr int a_index( int n, int m )
{
    return n * n + n + m;
}

} // namespace Canopy

#endif // CANOPY_SPHERICAL_COEFFICIENTS_HPP
