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

#ifndef CANOPY_CARTESIAN_TAYLOR_BASIS_HPP
#define CANOPY_CARTESIAN_TAYLOR_BASIS_HPP

#include <Kokkos_Core.hpp>

namespace Canopy
{
namespace CartesianTaylor
{

// ============================================================================
// Cartesian-Taylor far field, part one: the multi-index <-> flat slot map and
// the derivative ladder b_k = d^k phi for the softened kernel
//
//     phi(r) = ( |r|^2 + b )^{-1/2},     w = |r|^2 + b,     b > 0.
//
// Provenance: canopy-questions.md, the reference author's statement of the
// recurrences the reference treecode implements -- §1 for the radial ladder,
// §2 for the closed-form tensors through |k| = 3, §3 for the arbitrary-order
// multi-index recurrence. Each routine below names the section it transcribes.
//
// UNITS AND CONVENTIONS, for everything in this file:
//
//   r     The offset at which the derivatives are taken, in the same length
//         units as the tree. For the M2L this is R = c_A - c_B, TARGET center
//         minus SOURCE center (canopy-questions.md §4).
//   b     SOFTENING SQUARED -- the quantity added to r^2 -- units length^2.
//         It is NOT the softening length eps, and it is NOT eps itself; the
//         reference's `blob` is this quantity. b > 0 is a PRECONDITION: at
//         r = 0, b = 0 divides by zero. Passing b <= 0 aborts.
//   b_k   RAW derivatives d^k phi, with NO 1/k! factor, units
//         length^{-1-|k|}. The factorials live in the moment (1/q!) and in
//         the L2P evaluation (1/p!) and never here.
//
// This file is deliberately NOT a FarField contract member: no trait, no
// typedef, no static_assert on the basis, no operator. It holds the slot map
// and the b_k evaluator and nothing else (T1 of tasks/cartesian-taylor-basis.md).
// ============================================================================

//---------------------------------------------------------------------------//
// THE TOTAL ORDER ON MULTI-INDICES -- a design decision, stated here because
// it is not recoverable from the arithmetic below.
//
//   DEGREE-GRADED, then ASCENDING LEXICOGRAPHIC IN (kx, ky):
//
//     k < k'   iff   |k| < |k'|,
//              or    |k| == |k'| and (kx, ky) < (kx', ky') lexicographically
//                    ( kz = |k| - kx - ky is then determined ).
//
// Degree-graded is the load-bearing half of the choice and downstream code
// depends on it: the M2L needs b_{p+q} out to |p+q| = 2p while the moments
// only run to |q| <= p, so under a graded order the order-p slot table is a
// PREFIX of the order-2p slot table and one flat index is valid in both.
// A non-graded order would need two maps and a translation between them.
// It is also why slot() takes no order argument: the degree is read off the
// multi-index itself, and the answer does not move when p changes.
//
// The within-degree half is arbitrary; it is pinned here only so that it is
// written down somewhere. Ascending in kx, then ascending in ky.
//
// Degrees 0 through 2 come out as
//
//   slot 0                  : (0,0,0)
//   slots 1,  2,  3         : (0,0,1) (0,1,0) (1,0,0)
//   slots 4..9              : (0,0,2) (0,1,1) (0,2,0) (1,0,1) (1,1,0) (2,0,0)
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// Number of multi-indices of degree STRICTLY LESS THAN n, i.e. C(n+2,3).
// This is the flat index of the first slot of degree n.
//---------------------------------------------------------------------------//
KOKKOS_INLINE_FUNCTION
constexpr int slot_degree_base( int n ) { return n * ( n + 1 ) * ( n + 2 ) / 6; }

//---------------------------------------------------------------------------//
// Number of multi-indices of degree EXACTLY n, i.e. C(n+2,2).
//---------------------------------------------------------------------------//
KOKKOS_INLINE_FUNCTION
constexpr int num_slots_at_degree( int n ) { return ( n + 1 ) * ( n + 2 ) / 2; }

//---------------------------------------------------------------------------//
// Total number of slots covering every |k| <= p, i.e. C(p+3,3). This is the
// length the caller must allocate for derivative_ladder()'s `out`.
//---------------------------------------------------------------------------//
KOKKOS_INLINE_FUNCTION
constexpr int num_slots( int p ) { return slot_degree_base( p + 1 ); }

//---------------------------------------------------------------------------//
// slot: multi-index -> flat index, under the total order stated above.
// Returns a value in [ slot_degree_base(|k|), slot_degree_base(|k|+1) ), and
// hence in [0, num_slots(p)) for every p >= |k|.
//
// Within degree n the kx-block starts at sum_{a<kx} (n - a + 1), which is
// kx*(n+1) - kx*(kx-1)/2, and ky indexes inside that block.
//---------------------------------------------------------------------------//
KOKKOS_INLINE_FUNCTION
constexpr int slot( int kx, int ky, int kz )
{
    const int n = kx + ky + kz;
    return slot_degree_base( n ) + kx * ( n + 1 ) - kx * ( kx - 1 ) / 2 + ky;
}

//---------------------------------------------------------------------------//
// inverse_slot: flat index -> multi-index. The exact inverse of slot(): for
// every k, inverse_slot( slot(k) ) == k, and for every s >= 0,
// slot( inverse_slot(s) ) == s.
//
// In:  s     flat index, s >= 0
// Out: k[3]  the multi-index (kx, ky, kz)
//
// Both walks are over the degree and the kx-block and run in O(|k|) steps;
// |k| <= 2p is small, so no table is built.
//---------------------------------------------------------------------------//
KOKKOS_INLINE_FUNCTION
void inverse_slot( int s, int k[3] )
{
    int n = 0;
    while ( slot_degree_base( n + 1 ) <= s )
        ++n;

    const int d = s - slot_degree_base( n );

    // Largest kx whose block start is still <= d.
    int kx = 0;
    while ( kx < n &&
            ( kx + 1 ) * ( n + 1 ) - ( kx + 1 ) * kx / 2 <= d )
        ++kx;

    k[0] = kx;
    k[1] = d - ( kx * ( n + 1 ) - kx * ( kx - 1 ) / 2 );
    k[2] = n - k[0] - k[1];
}

//---------------------------------------------------------------------------//
// derivative_ladder: fill b_k(r; b) = d^k phi for every |k| <= max_order.
//
// Transcribed from canopy-questions.md §3, the multi-index recurrence obtained
// by solving w d_i phi = -r_i phi order by order:
//
//   w * b_{k+e_i} = - r_i b_k
//                   - k_i b_{k-e_i}
//                   - 2 sum_j k_j r_j     b_{k+e_i-e_j}
//                   -   sum_j k_j (k_j-1) b_{k+e_i-2e_j}
//
// with e_i the unit multi-index in direction i, and any b carrying a negative
// component identically zero (here: skipped, since its coefficient k_j or
// k_j(k_j-1) vanishes exactly on the same condition). Every term on the right
// sits at degree |k| or |k|-1, so a single forward sweep in ascending degree
// fills the table in place with no scratch. b enters only through the leading
// w -- canopy-questions.md §1: b is inert under the ladder, which is the whole
// reason this basis is blob-aware.
//
// The base case is canopy-questions.md §1: b_empty = P_0 = w^{-1/2} = phi.
//
// In:  r[3]       offset, length units; see the conventions block above
//      b          softening SQUARED (the quantity added to r^2), length^2,
//                 b > 0 required
//      max_order  fill every |k| <= max_order. For an M2L at expansion order
//                 p this is 2p, because b_{p+q} runs to |p+q| = 2p.
// Out: out[ 0 .. num_slots(max_order)-1 ], indexed by slot() above, holding
//      the RAW derivative d^k phi -- no 1/k!.
//
// Host- and device-callable. Allocates nothing; `out` is caller-provided and
// must be at least num_slots(max_order) long.
//
// Conditioning (risk R8 of tasks/cartesian-taylor-basis.md): every step
// divides by w and accumulates terms weighted by k_j(k_j-1), so error can grow
// with |k|. b > 0 bounds w >= b away from zero, so this is a conditioning
// question and never a division by zero. At p = 2 the ladder runs four steps
// and this is not a concern; tests/tstCartesianTaylor.hpp's finite-difference
// check at |k| = 2p is the instrument, and it must be re-measured, never
// assumed, if p is ever raised.
//---------------------------------------------------------------------------//
KOKKOS_INLINE_FUNCTION
void derivative_ladder( const double r[3], double b, int max_order,
                        double* out )
{
    // b > 0 is a precondition, not a defaultable argument: b = 0 at r = 0
    // divides by zero, and a defaulted return would hide it downstream.
    if ( !( b > 0.0 ) )
        Kokkos::abort( "Canopy::CartesianTaylor::derivative_ladder: b "
                       "(softening SQUARED, the quantity added to r^2) must "
                       "be strictly positive." );

    const double w = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + b;
    const double inv_w = 1.0 / w;

    // canopy-questions.md §1: P_0 = w^{-1/2} = phi.
    out[0] = 1.0 / Kokkos::sqrt( w );

    for ( int n = 0; n < max_order; ++n )
    {
        const int first = slot_degree_base( n + 1 );
        const int last = slot_degree_base( n + 2 );

        for ( int s = first; s < last; ++s )
        {
            int m[3];
            inverse_slot( s, m );

            // The direction the recurrence steps in. Any i with m_i > 0 gives
            // the same b_m; the lowest is taken so the sweep is deterministic.
            const int i = ( m[0] > 0 ) ? 0 : ( ( m[1] > 0 ) ? 1 : 2 );

            int k[3] = { m[0], m[1], m[2] };
            k[i] -= 1;

            // - r_i b_k
            double acc = -r[i] * out[slot( k[0], k[1], k[2] )];

            // - k_i b_{k-e_i}
            if ( k[i] > 0 )
            {
                int t[3] = { k[0], k[1], k[2] };
                t[i] -= 1;
                acc -= static_cast<double>( k[i] ) *
                       out[slot( t[0], t[1], t[2] )];
            }

            for ( int j = 0; j < 3; ++j )
            {
                // - 2 k_j r_j b_{k+e_i-e_j}
                if ( k[j] > 0 )
                {
                    int t[3] = { k[0], k[1], k[2] };
                    t[i] += 1;
                    t[j] -= 1;
                    acc -= 2.0 * static_cast<double>( k[j] ) * r[j] *
                           out[slot( t[0], t[1], t[2] )];
                }

                // - k_j (k_j - 1) b_{k+e_i-2e_j}
                if ( k[j] > 1 )
                {
                    int t[3] = { k[0], k[1], k[2] };
                    t[i] += 1;
                    t[j] -= 2;
                    acc -= static_cast<double>( k[j] * ( k[j] - 1 ) ) *
                           out[slot( t[0], t[1], t[2] )];
                }
            }

            out[s] = acc * inv_w;
        }
    }
}

} // namespace CartesianTaylor
} // namespace Canopy

#endif // CANOPY_CARTESIAN_TAYLOR_BASIS_HPP
