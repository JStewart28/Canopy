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

#include "Canopy_FarFieldContract.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <type_traits>

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
// Nothing in the Canopy::CartesianTaylor namespace below is a FarField
// contract member: it is the slot map and the b_k evaluator, and it is usable
// on its own (T1 of tasks/cartesian-taylor-basis.md). The contract surface --
// the traits, the typedefs, the static_asserts and the operators -- lives on
// Canopy::CartesianTaylorBasis at the bottom of this file (T2), which is built
// ON TOP of these free functions and adds nothing to them.
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

// ============================================================================
// CartesianTaylorBasis -- the FarField contract surface (T2 and T3 of
// tasks/cartesian-taylor-basis.md), built on the namespace above.
//
// A real-coefficient Cartesian Taylor expansion of the SOFTENED (Plummer)
// kernel
//
//     phi(r) = ( |r|^2 + b )^{-1/2},      b = eps^2,
//
// carrying PHYSICAL, un-normalized coefficients. It exists because the
// solid-harmonic basis expands 1/|r|, which is harmonic, and the softened
// kernel is not: no solid-harmonic expansion of it converges. The softening
// enters this basis in exactly one place -- b rides inside w = |R|^2 + b in
// the M2L's derivative ladder -- which is why M2L is the only kernel-touching
// operator here.
//
// ---------------------------------------------------------------------------
// UNITS AND CONVENTIONS for this class. None of it is recoverable from the
// code, and the Conventions table of tasks/cartesian-taylor-basis.md requires
// it on the declarations; it is collected here and repeated on each operator.
//
//   * WIDTHS. Every width parameter this basis is handed (w_self, w_child,
//     w_parent, w_source, w_target) is a HALF-WIDTH -- half the side length of
//     the cell's cube, matching Canopy::CellInfo::half_width.
//
//     EVERY ONE OF THEM IS IGNORED, and that is a positive statement about
//     this basis rather than an omission. The solid-harmonic basis divides its
//     multipole by w_self^{n+1} because a scale-invariant kernel makes the
//     normalized operator depend on the offset alone. A softened kernel has NO
//     scale invariance -- b is a fixed length^2 and does not rescale with the
//     cell -- so there is no normalization to divide out and the coefficients
//     carried here are physical. Each operator restates this on its own
//     declaration.
//
//   * OFFSETS. dx, dy, dz is always (other center - own center), but "other"
//     differs per call site and the sense is what a sign error here would
//     silently flip. Read from the sweeps (the table in the contract section
//     of tasks/cartesian-taylor-basis.md):
//
//       p2m_contribution   particle position - cell center      ( d )
//       m2m_translate      child center      - parent center    ( s )
//       m2l_translate      source center     - target center    ( -R )
//       l2l_translate      child center      - parent center    ( s )
//       l2p_evaluate       particle position - cell center      ( a )
//
//     R = c_target - c_source is the offset the derivative ladder wants
//     (canopy-questions.md §4), so the M2L NEGATES what it is handed. That is
//     T3's business; it is stated here because the two senses differing by a
//     sign is the single most likely place to lose a session.
//
//   * MULTIPOLE. M(cell, slot(q), c) is the physical Taylor moment
//
//         M_q = sum_{j in cell} ( y_j - c_cell )^q / q! * s_{j,c}
//
//     with s_{j,c} particle j's component-c charge. Units
//     charge * length^{|q|}. THE 1/q! IS IN THE MOMENT -- it is not in the
//     b_k, which are raw derivatives (see the namespace header above), and it
//     is not applied again anywhere else.
//
//   * LOCAL. L(cell, slot(p), comp_set_slot(c, 0)) is the physical Taylor
//     coefficient l_p of the potential about the cell center, so that
//
//         u(x) = sum_p ( x - c_cell )^p / p! * l_p .
//
//     Units charge * length^{-1-|p|}. The 1/p! is in the EVALUATION (l2p),
//     not in l_p. This IS a physical potential, unlike the conformance
//     fixture's dimensionless local, and is directly comparable to a direct
//     sum over the softened kernel.
//
//   * SETS. sets_per_component = 1 -- a Taylor local is one set of
//     C(p+3,3) coefficients per component. See that declaration.
//
//   * SCALAR. double only, by static_assert. A softened kernel has no scale
//     invariance, so the FP32 conditioning argument that the solid-harmonic
//     width normalizations exist for does not transfer.
// ---------------------------------------------------------------------------
//
// STATUS: complete. Every operator below is implemented, the three M2L
// members included -- they aborted through T2 and were filled in T3, which is
// also where the sign of R and the placement of the (-1)^{|q|} and the 1/q!
// were pinned against canopy-questions.md §2 by
// tests/tstCartesianTaylor.hpp. m2l_post_cell is REAL and not a stub: it is
// the only stage that writes the locals view, and a no-op there compiles
// cleanly while leaving every local coefficient zero.
// ============================================================================

template <class Scalar, int P_ORDER, int NComps = 1>
struct CartesianTaylorBasis
{
    // The Conventions row on Scalar: this basis is double only. Spelled
    // exactly as CanopyTest::MonopoleBasis does.
    static_assert( std::is_same<Scalar, double>::value,
                   "CartesianTaylorBasis: Scalar must be double" );

    static_assert( P_ORDER >= 0,
                   "CartesianTaylorBasis: P_ORDER is the Taylor order p and "
                   "must be non-negative" );

    using scalar_type = Scalar;

    // -----------------------------------------------------------------------
    // The coefficient contract. ONE REAL coefficient per slot: a Cartesian
    // Taylor coefficient of a real potential is real, unlike the solid
    // harmonics' Kokkos::complex. So scalars_per_coeff is 1 and coeff_type IS
    // the component scalar -- the case Canopy::detail::coeff_traits' PRIMARY
    // template covers, which is what makes coalesced_view_exchange and both
    // shared-cell Allreduces work here with no specialization.
    //
    // The identity element is value-initialization, coeff_type() == +0.0.
    // -----------------------------------------------------------------------
    using coeff_type = Scalar;
    using component_scalar_type = Scalar;
    static constexpr int scalars_per_coeff = 1;

    static_assert( sizeof( coeff_type ) ==
                       scalars_per_coeff * sizeof( component_scalar_type ),
                   "CartesianTaylorBasis: coeff_type is not "
                   "scalars_per_coeff contiguous component_scalar_type, so "
                   "the MPI packing in coalesced_view_exchange and in the two "
                   "shared-cell reductions would transfer the wrong byte "
                   "count" );

    // The Taylor order p. The expansion carries every multi-index with
    // |q| <= p.
    static constexpr int max_order = P_ORDER;

    static constexpr int num_components = NComps;

    // =======================================================================
    // num_coeffs_per_cell -- C(p+3,3), the number of multi-indices with
    // |q| <= p, spelled THROUGH Canopy::CartesianTaylor::num_slots and never
    // as a second binomial helper.
    //
    // That is not a style preference. num_slots is the same function the slot
    // map is built from, so a flat index produced by slot() is in range by
    // construction rather than by a coincidence between two spellings of one
    // count. m2l_num_src_coeffs below is the same call for the same reason:
    // the M2L contracts a source multipole against a target local over ONE
    // flat index, and two spellings of that count is exactly how the shared
    // index stops being shared.
    //
    // CONSTEXPR, and it must stay so: it drives the M2L's unrolling and the
    // scratch size (risk R3 of tasks/cartesian-taylor-basis.md, which has
    // already fired once at roughly +18%). num_slots is already constexpr.
    // =======================================================================
    static constexpr int num_coeffs_per_cell =
        Canopy::CartesianTaylor::num_slots( P_ORDER );

    // Flat source-coefficient count for the precomputed-operator path. The
    // M2L operator is (num_coeffs_per_cell x m2l_num_src_coeffs) -- target
    // local slots by source moment slots -- and both axes run over the SAME
    // multi-index set at order p, so this is num_coeffs_per_cell again by
    // construction and not by agreement.
    static constexpr int m2l_num_src_coeffs =
        Canopy::CartesianTaylor::num_slots( P_ORDER );

    // =======================================================================
    // sets_per_component -- ONE.
    //
    // A Taylor local is one set of C(p+3,3) coefficients per component:
    // l_p^{(c)} for every |p| <= P_ORDER. There is no second quantity that
    // accumulates over the same interaction list, so there is nothing for a
    // second set to hold. At 1 the (component, set) flattening collapses to
    // c exactly and the locals view's third extent is NComps, as it was
    // before the trait existed.
    //
    // CONSTEXPR, per R3 -- it multiplies into m2l_scratch_bytes and into the
    // sweep's TeamVectorRange zero-fill bound.
    // =======================================================================
    static constexpr int sets_per_component = 1;

    // Slots one cell's coefficient occupies in the locals view and in the M2L
    // accumulator: the locals view's third extent.
    static constexpr int num_comp_slots = NComps * sets_per_component;

    // (component, set) -> the locals view's third index. Component-major,
    // set-minor -- identical to DownwardSweep::shared_slot's factor, which is
    // what makes the shared-cell round trip put a coefficient back in its own
    // slot. At sets_per_component = 1 this is the identity on c; it is spelled
    // out anyway so that every slot expression in this file goes through ONE
    // function and a later set count cannot be added in half the places.
    KOKKOS_INLINE_FUNCTION
    static constexpr int comp_set_slot( int c, int s )
    {
        return c * sets_per_component + s;
    }

    // -----------------------------------------------------------------------
    // The M2L key contract. See Canopy::LaplaceKernel for the full statement
    // of what the five key integers are.
    // -----------------------------------------------------------------------

    // The |dd| range guard. 6, but NOT for LaplaceKernel's reason: there, 6 is
    // a precision bound on a scale-normalized operator carrying a residual
    // 2^{j|dd|} factor, and a basis carrying PHYSICAL operators inherits
    // neither that factor nor that bound. Here 6 merely bounds the key space,
    // and it is chosen so the set of pairs routed to the fallback path is the
    // same one every other basis in this repository sees -- which keeps
    // total_fallback_pair_count() comparable across bases. It is not a
    // precision claim about this basis.
    static constexpr int m2l_key_dd_max = 6;

    // TRUE, and necessarily so. This basis's operator is PHYSICAL: it is
    // b_k(R) evaluated at the real translation vector R, whose length is the
    // integer offset times the half-width at the deeper of the two depths. Two
    // pairs with the same integer offset at different levels have different R
    // and therefore different operators, so a level-blind key would alias
    // them. The solid-harmonic basis can zero max_d precisely because its
    // operator is scale-normalized; this one cannot.
    //
    // Consequence, recorded rather than worked around: set_root_half_width
    // clears the ENTIRE operator cache when the root half-width changes, and
    // only for a key_needs_level basis (src/Canopy_DownwardSweep.hpp:406-416).
    // On a drifting bounding box the cache therefore empties on every rebuild
    // (risk R6). That is a cost, not a defect -- it is also what keeps a
    // level-dependent operator from being served stale (risk R5).
    static constexpr bool key_needs_level = true;

    // Identity -- the key is returned unchanged, max_d and all, because
    // key_needs_level is true and the level is part of what distinguishes one
    // operator from another here.
    //
    // A function template on the key type, deliberately: M2LKey is a nested
    // type of DownwardSweep<..., KernelType>, so a basis cannot name it
    // without a circular dependency. The sweep passes its own M2LKey and Key
    // is deduced. Host-only, not KOKKOS_INLINE_FUNCTION -- the classify pass
    // runs on host.
    template <class Key>
    static Key canonicalize_key( Key k )
    {
        return k;
    }

    // Bytes one operator column costs: the full (Nt x Ns) dense block. DERIVED
    // from sizeof(coeff_type) and the two counts above, never a literal, so it
    // tracks a change to either. At p = 2 this is 10 * 10 * 8 = 800 bytes per
    // key.
    static constexpr std::size_t bytes_per_key =
        static_cast<std::size_t>( num_coeffs_per_cell ) *
        static_cast<std::size_t>( m2l_num_src_coeffs ) * sizeof( coeff_type );

    // A pair refused an operator column takes the basis's own per-pair
    // operator at the physical geometry. This basis can evaluate one -- the
    // derivative ladder is device-callable and allocates nothing -- so
    // PerPairTranslate is available, and it is the enumerator that REQUIRES
    // m2l_translate to be a real implementation. The other enumerator,
    // EscalateToP2P, fails a class-scope static_assert in DownwardSweep.
    static constexpr Canopy::M2LOverflow m2l_overflow_policy =
        Canopy::M2LOverflow::PerPairTranslate;

    // -----------------------------------------------------------------------
    // The M2L operator set: (num_coeffs_per_cell, m2l_num_src_coeffs,
    // n_unique_ops), LayoutLeft, element type coeff_type. The sweep addresses
    // it only by the integer op_idx its CSR carries and never indexes inside.
    //
    // An ALIAS TEMPLATE on the memory space, not a typedef: DownwardSweep
    // instantiates it at TWO spaces at once -- memory_space for the device
    // table and Kokkos::HostSpace for the persistent operator cache
    // (src/Canopy_DownwardSweep.hpp:621 and :630) -- and spells it
    //     typename KernelType::template m2l_operators_type<memory_space>.
    // A plain typedef does not compile.
    // -----------------------------------------------------------------------
    template <class MemorySpace>
    using m2l_operators_type =
        Kokkos::View<coeff_type***, Kokkos::LayoutLeft, MemorySpace>;

    // -----------------------------------------------------------------------
    // Auxiliary tables: NO TABLE, but ONE SCALAR -- the kernel's b.
    //
    // WHY b LIVES HERE, which is not obvious and is not optional. m2l_translate
    // is a DEVICE operator and its signature carries no M2LKernelParams
    // (src/Canopy_DownwardSweep.hpp:2338-2347 is the whole of what the fallback
    // path hands a basis: the team, the multipoles, the source cell, the
    // physical offset, the two half-widths, THIS struct, and the target slice).
    // So for a basis whose operator depends on a kernel parameter, aux is the
    // ONLY channel from build_aux_tables' M2LKernelParams to a device operator.
    // LaplaceKernel's own declaration says that is what the argument is for
    // (src/Canopy_LaplaceKernel.hpp:307-313) -- its own table happens not to
    // need it. This one does.
    //
    // b IS SOFTENING SQUARED, the quantity added to r^2, units length^2 -- not
    // the length eps that M2LKernelParams carries. The squaring happens in
    // build_aux_tables below and nowhere else on this path.
    //
    // FRESHNESS. Solver::_push_kernel_params (src/Canopy_Solver.hpp:729-738)
    // runs before every _upward.setup(), which is where build_aux_tables is
    // called (src/Canopy_UpwardSweep.hpp:305), so a solve never runs an M2L
    // against a b from a previous configuration. A sweep driven directly by a
    // test and never handed a configuration gets the M2LKernelParams default,
    // softening = 0, hence b = 0 -- which is why m2l_translate GUARDS b > 0
    // rather than trusting it.
    //
    // STILL NO TABLE, deliberately. The multi-index arithmetic this basis runs
    // -- inverse_slot once and slot up to seven times per ladder slot -- is
    // recomputed on every call and COULD be tabulated here. As of T3 the call
    // site that would pay for it exists (m2l_operator_block, once per key in
    // build_m2l_operators and once per pair in m2l_translate), so the
    // opportunity is now live rather than hypothetical. It is still not taken,
    // because risk R3 of tasks/cartesian-taylor-basis.md records a MEASURED
    // +18% M2L regression from a comparable indirection -- with two candidate
    // micro-causes tested and excluded -- so a table here must be MEASURED
    // against the recompute it replaces and not assumed to be faster. Tracked
    // in README.md's "Future Optimizations".
    //
    // A struct TEMPLATE on the memory space, not a plain struct, because that
    // is how both sweeps spell it:
    //     typename KernelType::template aux_tables_type<memory_space>
    // (src/Canopy_UpwardSweep.hpp:109, src/Canopy_DownwardSweep.hpp:184).
    // Neither sweep names a member, so what is inside is this file's business;
    // MemorySpace is unused here because a scalar needs no allocation, and the
    // struct stays trivially copyable so the sweeps' by-value capture works.
    // -----------------------------------------------------------------------
    template <class MemorySpace>
    struct aux_tables_type
    {
        // Softening SQUARED -- the quantity added to r^2 -- units length^2.
        // NOT the length eps. Zero means "never set", which m2l_translate
        // rejects rather than expanding an unsoftened kernel it cannot
        // expand.
        double b = 0.0;
    };

    // TWO arguments, matching the real call sites at
    // src/Canopy_UpwardSweep.hpp:305 and src/Canopy_DownwardSweep.hpp:1789.
    //
    // `order` is the Taylor order p; a basis needing a table sized by it would
    // read it here rather than from its own template arguments. This basis
    // sizes nothing by it -- its one ladder is 2*P_ORDER and comes off its own
    // template argument -- so it is ignored.
    // `kernel_params` carries the softening as a LENGTH eps; THE KERNEL'S b IS
    // eps^2 (src/Canopy_FarFieldContract.hpp:115-121), and this is the one
    // place on the device path where that square is taken.
    //
    // IT DOES NOT REJECT eps <= 0, and that is deliberate. UpwardSweep::setup
    // calls this unconditionally, including for a sweep a test drives directly
    // and never runs an M2L on, where the M2LKernelParams default of 0.0 is
    // correct and harmless. The rejection belongs at the point of USE --
    // build_m2l_operators and m2l_translate both abort on a non-positive b --
    // so a configuration that cannot work fails where it is actually about to
    // produce a wrong operator.
    //
    // A plain static host function, not KOKKOS_INLINE_FUNCTION, per the
    // Conventions row on host-side construction -- even though this one
    // allocates nothing.
    template <class MemorySpace>
    static aux_tables_type<MemorySpace>
    build_aux_tables( int order, const Canopy::M2LKernelParams& kernel_params )
    {
        (void)order;
        aux_tables_type<MemorySpace> aux;
        aux.b = kernel_params.softening * kernel_params.softening;
        return aux;
    }

    // -----------------------------------------------------------------------
    // The M2L team scratch, viewed as a scalar array.
    //
    // BASIS-INTERNAL. No sweep names this type: the sweep hands the stages raw
    // char bytes sized by m2l_scratch_bytes, and this alias is only how THIS
    // basis's own stages reinterpret them. Its shape is entirely this file's
    // business, and the accumulator layout below is the whole of the contract
    // between m2l_core (T3) and m2l_post_cell (T2).
    // -----------------------------------------------------------------------
    template <class ScratchSpace>
    using m2l_accumulator_type =
        Kokkos::View<scalar_type*, ScratchSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    // =======================================================================
    // THE M2L ACCUMULATOR LAYOUT -- settled here, in T2, even though m2l_core
    // fills it only in T3.
    //
    // It has to be settled here because m2l_post_cell READS it, and
    // m2l_post_cell is written in this task. A layout deferred to T3 would
    // leave the flush indexing something whose shape nobody had decided.
    //
    //   n_acc = num_coeffs_per_cell * NComps * sets_per_component
    //           contiguous scalar_type, holding the target cell's local
    //           accumulator,
    //
    //   acc_slot(out_idx, c, s) = out_idx * num_comp_slots
    //                             + comp_set_slot(c, s)
    //
    // COEFFICIENT-MAJOR, then component, then set -- the SAME nesting as the
    // locals view (cell, coeff, comp_set_slot), which is what makes
    // m2l_post_cell a slot-for-slot walk of one cell's slice rather than a
    // transpose. It is also the nesting T3's contraction wants: the M2L reads
    // M_full(source_cell, q_slot, c), whose component axis is likewise
    // innermost, so the component loop is stride-1 on both sides of the
    // multiply-accumulate.
    //
    // At sets_per_component = 1 this collapses to
    // acc_slot(out_idx, c, 0) = out_idx * NComps + c; the set factor is
    // carried anyway so that a later set count changes one expression.
    // =======================================================================
    KOKKOS_INLINE_FUNCTION
    static constexpr int acc_slot( int out_idx, int c, int s )
    {
        return out_idx * num_comp_slots + comp_set_slot( c, s );
    }

    // =======================================================================
    // m2l_scratch_bytes -- the per-team M2L scratch this basis needs, in
    // bytes: one contiguous scalar_type array of n_acc entries in the layout
    // above. The real/imag split the solid-harmonic basis needs has no
    // analogue here; the coefficient is real.
    //
    // CONSTEXPR, and it must stay so. The sweep assigns it to a
    // `constexpr size_t` (src/Canopy_DownwardSweep.hpp:2209) and derives the
    // TeamVectorRange zero-fill bound from it, which is what keeps every
    // extent inside the M2L stages a compile-time constant. A runtime value
    // here deoptimizes the fused kernel WITH NO CORRECTNESS SIGNAL AT ALL
    // (risk R3). num_coeffs_per_cell and sets_per_component are both
    // static constexpr on this class, so the product is a constant
    // expression. n_comps is a parameter only so the sweep can size scratch
    // without reaching into this basis's template arguments; it is only ever
    // passed num_components.
    //
    // The sweep hands the stages RAW, ZERO-FILLED bytes and relies on all-zero
    // bytes being this basis's accumulator identity. That holds: the
    // accumulator is IEEE-754 binary64, whose all-zero-bytes representation is
    // +0.0, and +0.0 is the additive identity.
    // =======================================================================
    KOKKOS_INLINE_FUNCTION
    static constexpr std::size_t m2l_scratch_bytes( int n_comps )
    {
        return static_cast<std::size_t>( num_coeffs_per_cell ) *
               static_cast<std::size_t>( n_comps ) *
               static_cast<std::size_t>( sets_per_component ) *
               sizeof( scalar_type );
    }

    // =======================================================================
    // taylor_accumulate -- the single multiply-accumulate step of every
    // kernel-blind operator below, factored out so that no two of them, and no
    // host reference in tests/tstCartesianTaylor.hpp, can drift apart.
    //
    // TWO STATEMENTS ON PURPOSE. `acc += a * b` as one statement is
    // contractible to an FMA under the default -ffp-contract=on, and nothing
    // guarantees that a device compilation and a plain host loop make the same
    // contraction decision. Splitting the product into a named local puts a
    // statement boundary between the multiply and the add, which
    // -ffp-contract=on may not cross. Per the Conventions row on guarding
    // against -ffp-contract; CanopyTest::MonopoleBasis::m2l_accumulate is the
    // same device.
    // =======================================================================
    KOKKOS_INLINE_FUNCTION
    static void taylor_accumulate( Scalar& acc, Scalar a, Scalar b )
    {
        const Scalar prod = a * b;
        acc += prod;
    }

    // =======================================================================
    // taylor_monomials -- THE Taylor shift table, and the single source of
    // truth for it. Fills
    //
    //     t[ slot(k) ] = dx^kx dy^ky dz^kz / ( kx! ky! kz! )     for |k| <= p
    //
    // i.e. d^k / k! over the same slot map every coefficient array in this
    // file uses.
    //
    // ALL FOUR kernel-blind operators are this table contracted against a
    // coefficient array, which is why it has one home:
    //
    //   P2M   M_q       += t[q] * charge                 ( d = y - c_cell )
    //   M2M   M^par_q   += sum_{q' <= q}  t[q-q'] M^ch_q'   ( s = c_ch - c_par )
    //   L2L   l^ch_p    += sum_{p' >= p}  t[p'-p] l^par_p'  ( s = c_ch - c_par )
    //   L2P   u          = sum_p          t[p]    l_p      ( a = x - c_cell )
    //
    // The 1/k! is HERE and nowhere else on these paths -- not in the b_k,
    // which are raw derivatives, and not applied a second time by any caller.
    //
    // In:  dx, dy, dz  the offset, sense per the call-site table in this
    //                  class's header comment. Length units.
    // Out: t[ 0 .. num_coeffs_per_cell )   caller-provided, no allocation.
    //
    // Built as an outer product of three per-axis tables pf_i[j] = d_i^j / j!
    // so each entry costs two multiplies; the per-axis tables are built by one
    // running product each, so no pow() and no factorial division appears in
    // an inner loop.
    // =======================================================================
    KOKKOS_INLINE_FUNCTION
    static void taylor_monomials( Scalar dx, Scalar dy, Scalar dz,
                                  Scalar ( &t )[num_coeffs_per_cell] )
    {
        Scalar pf[3][P_ORDER + 1];
        const Scalar d[3] = { dx, dy, dz };

        for ( int a = 0; a < 3; a++ )
        {
            pf[a][0] = static_cast<Scalar>( 1 );
            for ( int j = 1; j <= P_ORDER; j++ )
                pf[a][j] = pf[a][j - 1] * d[a] / static_cast<Scalar>( j );
        }

        for ( int s = 0; s < num_coeffs_per_cell; s++ )
        {
            int k[3];
            Canopy::CartesianTaylor::inverse_slot( s, k );
            t[s] = pf[0][k[0]] * pf[1][k[1]] * pf[2][k[2]];
        }
    }

    // =======================================================================
    // P2M: accumulate a particle's contribution to its leaf's Taylor moments.
    //
    //   M(slot(q), c) += d^q / q! * charge_c ,     d = particle - cell center
    //
    // canopy-questions.md §4, the P2M row. Kernel-blind: the softening does
    // not appear, because a moment is a property of the source distribution
    // and not of the kernel.
    //
    //   charges     Scalar[NComps], this particle's per-component charges
    //   dx, dy, dz  PARTICLE POSITION MINUS CELL CENTER (d), length units
    //               (src/Canopy_UpwardSweep.hpp:465-485)
    //   w_self      this leaf's HALF-WIDTH. IGNORED -- this basis's moments
    //               are physical and carry no width normalization; see the
    //               class header.
    //   M_out       2D slice M_out(coeff_slot, comp_idx) for this leaf
    //
    // ATOMIC. The sweep runs one thread per particle over a RangePolicy and
    // many particles share a leaf, so two threads can accumulate into one
    // moment slot concurrently.
    // =======================================================================
    template <class MSliceType>
    KOKKOS_INLINE_FUNCTION static void
    p2m_contribution( const Scalar ( &charges )[NComps], Scalar dx, Scalar dy,
                      Scalar dz, Scalar w_self, const MSliceType& M_out )
    {
        (void)w_self;

        Scalar t[num_coeffs_per_cell];
        taylor_monomials( dx, dy, dz, t );

        for ( int s = 0; s < num_coeffs_per_cell; s++ )
            for ( int c = 0; c < NComps; c++ )
            {
                const Scalar contrib = t[s] * charges[c];
                Kokkos::atomic_add( &M_out( s, c ), contrib );
            }
    }

    // =======================================================================
    // M2M: shift a child's moments to the parent center and accumulate.
    //
    //   M^par_q += sum_{q' <= q} s^{q-q'} / (q-q')! * M^ch_q' ,
    //                                            s = c_child - c_parent
    //
    // canopy-questions.md §4, the M2M row. This is the plain binomial Taylor
    // shift and is identical to any Cartesian FMM: it follows from
    // (y - c_par)^q / q! = sum_{q' <= q} (y - c_ch)^q' / q' * s^{q-q'}/(q-q')!
    // summed over the child's particles, so it is EXACT rather than truncated
    // for every q the parent carries.
    //
    //   dx, dy, dz          CHILD CENTER MINUS PARENT CENTER (s), length units
    //                       (src/Canopy_UpwardSweep.hpp:528-542)
    //   w_child, w_parent   both HALF-WIDTHS. BOTH IGNORED -- physical,
    //                       un-normalized moments; see the class header.
    //   aux                 empty; this basis has no tables. IGNORED.
    //   M_parent_out        2D slice M_parent_out(coeff_slot, comp_idx)
    //
    // NON-ATOMIC. The sweep runs one team per parent and that team walks its
    // own children sequentially, so no two writers touch one parent.
    // =======================================================================
    template <class TeamMember, class MView, class AuxType, class MParentType>
    KOKKOS_INLINE_FUNCTION static void
    m2m_translate( const TeamMember& team_member, const MView& M_full,
                   int child_cell, Scalar dx, Scalar dy, Scalar dz,
                   Scalar w_child, Scalar w_parent, const AuxType& aux,
                   const MParentType& M_parent_out )
    {
        (void)w_child;
        (void)w_parent;
        (void)aux;

        Scalar t[num_coeffs_per_cell];
        taylor_monomials( dx, dy, dz, t );

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team_member, num_coeffs_per_cell ),
            [&]( const int out_idx )
            {
                int q[3];
                Canopy::CartesianTaylor::inverse_slot( out_idx, q );

                for ( int c = 0; c < NComps; c++ )
                {
                    Scalar acc = static_cast<Scalar>( 0 );

                    // q' <= q componentwise; the shift multi-index is q - q',
                    // whose degree is automatically <= |q| <= P_ORDER, so no
                    // term of this sum falls outside the table.
                    for ( int ax = 0; ax <= q[0]; ax++ )
                        for ( int ay = 0; ay <= q[1]; ay++ )
                            for ( int az = 0; az <= q[2]; az++ )
                            {
                                const int src =
                                    Canopy::CartesianTaylor::slot( ax, ay,
                                                                   az );
                                const int sh =
                                    Canopy::CartesianTaylor::slot(
                                        q[0] - ax, q[1] - ay, q[2] - az );
                                taylor_accumulate(
                                    acc, t[sh],
                                    M_full( child_cell, src, c ) );
                            }

                    M_parent_out( out_idx, c ) += acc;
                }
            } );
    }

    // =======================================================================
    // THE M2L OPERATOR -- sizes, flat index, and the one function that
    // produces a value. Everything from here to m2l_operator_block is the
    // single source of truth the Conventions table of
    // tasks/cartesian-taylor-basis.md requires; build_m2l_operators,
    // m2l_translate and every host reference in tests/tstCartesianTaylor.hpp
    // reach an operator value THROUGH m2l_operator_block and nowhere else.
    // That is what makes the fused-versus-fallback comparison EXACT rather
    // than merely close.
    // =======================================================================

    // The ladder order the M2L needs: the operator reads b_{p+q} and both
    // multi-indices run to degree P_ORDER, so |p+q| reaches 2*P_ORDER. A
    // ladder one degree short does not fault -- it reads a slot belonging to
    // some other multi-index -- it silently produces a wrong operator.
    static constexpr int m2l_ladder_order = 2 * P_ORDER;

    // Length of that ladder, C(2p+3,3). SPELLED THROUGH num_slots, like
    // num_coeffs_per_cell, so the flat index this table is walked with is the
    // same one slot() produces. T1's order is DEGREE-GRADED, which is what
    // makes the order-p slot table a PREFIX of this one: a q_slot valid in
    // the multipole is the same integer here, and no second map and no
    // translation between maps exists anywhere in this basis.
    static constexpr int m2l_ladder_slots =
        Canopy::CartesianTaylor::num_slots( m2l_ladder_order );

    // Entries in one dense operator block, (Nt x Ns). CONSTEXPR, per R3: it
    // is the extent of a local array in two operators.
    static constexpr int m2l_op_entries =
        num_coeffs_per_cell * m2l_num_src_coeffs;

    // Flat index into one operator block. TARGET-SLOT-MAJOR, matching the
    // (target_slot, source_slot) axis order of m2l_operators_type, so
    // build_m2l_operators copies op[m2l_op_index(p,q)] to ops(p,q,j) with no
    // transpose.
    KOKKOS_INLINE_FUNCTION
    static constexpr int m2l_op_index( int p_slot, int q_slot )
    {
        return p_slot * m2l_num_src_coeffs + q_slot;
    }

    // =======================================================================
    // m2l_operator_block -- THE M2L operator, and the single source of truth
    // for it.
    //
    //   op[ m2l_op_index( slot(p), slot(q) ) ] = (-1)^{|q|} b_{p+q}(R)
    //
    // so that l_p^A = sum_q op(p,q) M_q^B. Provenance:
    // tasks/canopy-questions.md §4, the M2L line
    //
    //     l_p^A = sum_q [ (-1)^{|q|} b_{p+q}(R) ] M_q^B ,
    //
    // with b_n(R) = d^n phi(R) the §2/§3 tensors. The two conventions that
    // decide whether this is the right field rather than a plausible one, and
    // where each lives:
    //
    //   * THE (-1)^{|q|} IS HERE. It comes from the source-side Taylor
    //     expansion: phi(R - d) = sum_q (-d)^q/q! d^q phi(R), and the moment
    //     M_q carries d^q/q! with a PLUS sign, so the alternation belongs to
    //     the operator. Dropping it leaves an expansion that still converges,
    //     to the wrong thing (risk R1). tests/tstCartesianTaylor.hpp's
    //     m2l_ell0_closed_forms is the check that sees it, at |q| = 1.
    //
    //   * THE 1/q! IS NOT HERE. b_k are RAW derivatives (see the
    //     Canopy::CartesianTaylor header) and the factorial lives in the
    //     moment and in the L2P evaluation. Applying it a second time here is
    //     the other route to the same wrong field.
    //
    // IN:
    //   R[3]  the TRANSLATION VECTOR, R = c_target - c_source, length units.
    //         NOTE THE SENSE. The sweep's M2L key and the fallback path's
    //         offset are both SOURCE MINUS TARGET
    //         (src/Canopy_DownwardSweep.hpp:1464-1481 and :2338-2340), so
    //         both callers NEGATE before calling this. The negation is not
    //         optional and is not absorbed anywhere else; phi is even, so a
    //         missing one flips the sign of every odd-|p| coefficient and
    //         leaves the even ones right, which is exactly the failure the
    //         parity check in tests/tstCartesianTaylor.hpp is sourced to
    //         catch.
    //   b     SOFTENING SQUARED, the quantity added to r^2, length^2. NOT the
    //         length eps. b > 0 is a precondition -- derivative_ladder aborts
    //         otherwise -- and both callers guard it before arriving here.
    //
    // OUT:
    //   op    caller-provided, every one of the m2l_op_entries written.
    //
    // It evaluates the ladder ONCE per call, which is once per key in
    // build_m2l_operators and once per pair in m2l_translate. Host- and
    // device-callable; allocates nothing.
    //
    // NO dd DEPENDENCE, and hence no key dependence -- the operator is a
    // function of the physical (R, b) alone. That is why this takes R and not
    // a key: m2l_translate is handed two cell centers and cannot reconstruct
    // max_d from two half-widths, so a key-parameterized operator would be
    // unreachable from the fallback path. One consequence to expect rather
    // than debug: two keys differing only in dd produce IDENTICAL columns.
    // That is duplication in the table, not an error.
    // =======================================================================
    KOKKOS_INLINE_FUNCTION
    static void m2l_operator_block( const Scalar R[3], Scalar b,
                                    Scalar ( &op )[m2l_op_entries] )
    {
        Scalar bk[m2l_ladder_slots];
        Canopy::CartesianTaylor::derivative_ladder( R, b, m2l_ladder_order,
                                                    bk );

        for ( int q_slot = 0; q_slot < m2l_num_src_coeffs; q_slot++ )
        {
            int q[3];
            Canopy::CartesianTaylor::inverse_slot( q_slot, q );

            // (-1)^{|q|}. Exact in binary64: it is +1 or -1.
            const Scalar sign = ( ( ( q[0] + q[1] + q[2] ) & 1 ) == 0 )
                                    ? static_cast<Scalar>( 1 )
                                    : static_cast<Scalar>( -1 );

            for ( int p_slot = 0; p_slot < num_coeffs_per_cell; p_slot++ )
            {
                int p[3];
                Canopy::CartesianTaylor::inverse_slot( p_slot, p );

                // |p + q| <= 2*P_ORDER by construction, so this slot is
                // inside the ladder. Same flat index, no map translation.
                const int pq = Canopy::CartesianTaylor::slot(
                    p[0] + q[0], p[1] + q[1], p[2] + q[2] );

                op[m2l_op_index( p_slot, q_slot )] = sign * bk[pq];
            }
        }
    }

    // =======================================================================
    // M2L, per-pair fallback path.
    //
    // The sweep routes a pair here when its key trips a range guard or the
    // operator-table count cap, in which case no operator column exists for
    // it. m2l_overflow_policy selects PerPairTranslate, which is the
    // enumerator that REQUIRES this to be a real implementation.
    //
    //   L^target(p, comp_set_slot(c, s))
    //       += sum_q op(p, q) * M^source(q, c)
    //
    // IT RECONSTRUCTS NO KEY AND NEEDS NONE. The operator is a function of the
    // physical (R, b) alone (see m2l_operator_block), so there is nothing a
    // key would supply that the offset does not.
    // CanopyTest::MonopoleBasis::m2l_translate does reconstruct one and is not
    // the model to copy here: its local is dimensionless and normalized, so it
    // needs F(dd) to convert between "separation in deeper-cell half-widths"
    // and the source's own scale. A physical operator has no normalization to
    // undo -- and this function could not recover max_d from two half-widths
    // in any case.
    //
    // THE SAME ARITHMETIC AS m2l_core, TERM FOR TERM, and that is load-bearing
    // rather than incidental: both walk q in ascending slot order into a named
    // local through taylor_accumulate, so for one pair the two paths produce
    // BIT-IDENTICAL coefficients. If they did not, which pairs overflow would
    // decide the answer (risk R4). tests/tstCartesianTaylor.hpp's
    // m2l_fused_vs_fallback asserts exact equality, which is what makes a
    // non-zero total_fallback_pair_count() harmless.
    //
    //   dx, dy, dz  SOURCE CENTER MINUS TARGET CENTER
    //               (src/Canopy_DownwardSweep.hpp:2338-2345), which is -R;
    //               this NEGATES to get the R the ladder wants.
    //   w_source, w_target  both HALF-WIDTHS. BOTH IGNORED -- physical,
    //               un-normalized operators; see the class header.
    //   aux         carries b, softening SQUARED. It is the ONLY channel from
    //               the kernel parameters to this function: the fallback call
    //               site passes no M2LKernelParams. See aux_tables_type.
    //   L_target_out  2D slice L_target_out(coeff_slot, comp_set_slot)
    //
    // ATOMIC. The sweep runs one team per PAIR, so two pairs sharing a target
    // write concurrently.
    // =======================================================================
    template <class TeamMember, class MView, class AuxType, class LTargetType>
    KOKKOS_INLINE_FUNCTION static void
    m2l_translate( const TeamMember& team_member, const MView& M_full,
                   int source_cell, Scalar dx, Scalar dy, Scalar dz,
                   Scalar w_source, Scalar w_target, const AuxType& aux,
                   const LTargetType& L_target_out )
    {
        (void)w_source;
        (void)w_target;

        // b reaches a device operator only through aux, and its default is
        // zero -- a sweep nobody handed an M2LKernelParams to runs at
        // softening = 0, the unsoftened kernel, which this basis cannot
        // expand and which divides by zero at R = 0. Fail here rather than
        // deeper, where the message would name the ladder and not the
        // configuration.
        if ( !( aux.b > 0.0 ) )
            Kokkos::abort(
                "CartesianTaylorBasis::m2l_translate: aux.b (softening "
                "SQUARED, the quantity added to r^2) is not positive. It is "
                "built from M2LKernelParams::softening, a LENGTH eps whose "
                "default is 0.0 -- the unsoftened kernel, which this basis "
                "has no expansion of. Set a positive softening on the sweep "
                "or the Solver before running an M2L." );

        // R = c_target - c_source = -(source - target). The negation is the
        // whole of the sign convention; see m2l_operator_block.
        const Scalar R[3] = { -dx, -dy, -dz };

        Scalar op[m2l_op_entries];
        m2l_operator_block( R, static_cast<Scalar>( aux.b ), op );

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team_member, num_coeffs_per_cell ),
            [&]( const int out_idx )
            {
                for ( int c = 0; c < NComps; c++ )
                {
                    Scalar acc = static_cast<Scalar>( 0 );
                    for ( int q_slot = 0; q_slot < m2l_num_src_coeffs;
                          q_slot++ )
                        taylor_accumulate(
                            acc, op[m2l_op_index( out_idx, q_slot )],
                            M_full( source_cell, q_slot, c ) );

                    // Every (component, set) slot takes the same operator --
                    // there is one set here -- so this is the flat slot walk
                    // m2l_core's is, and the two stay parallel if a later set
                    // count differs.
                    for ( int st = 0; st < sets_per_component; st++ )
                        Kokkos::atomic_add(
                            &L_target_out( out_idx, comp_set_slot( c, st ) ),
                            acc );
                }
            } );
    }

    // =======================================================================
    // build_m2l_operators -- fill the dense (num_coeffs_per_cell x
    // m2l_num_src_coeffs) operator of every canonical key the sweep's
    // persistent cache is missing.
    //
    // Parameters mirror Canopy::LaplaceKernel::build_m2l_operators
    // positionally:
    //
    //   keys[0..n_keys)   canonical keys; column j of `ops` is keys[j].
    //                     Already through canonicalize_key, which for THIS
    //                     basis is the identity -- key_needs_level is true --
    //                     so max_d survives and is a real tree level here.
    //   unit_w[0..n_levels)
    //                     the HALF-WIDTH at each depth, w_root / 2^d, indexed
    //                     by the key's max_d.
    //   kernel_params     softening as a LENGTH eps; the kernel's b is eps^2.
    //
    // THE KEY-TO-R PATH, which is the production spelling of the sign
    // convention and the one the parity check in
    // tests/tstCartesianTaylor.hpp sources its first argument from. The
    // sweep's key carries
    //
    //     (ii, jj, kk) * unit_w[max_d] = c_source - c_target
    //
    // -- SOURCE MINUS TARGET, built at src/Canopy_DownwardSweep.hpp:1464-1481
    // -- while canopy-questions.md §4 wants R = c_target - c_source. So
    //
    //     R = -(ii, jj, kk) * unit_w[max_d].
    //
    // kk.dd IS READ ONLY BY THE GUARDS. This operator has no dd dependence, so
    // two keys differing only in dd get identical columns; that is duplication
    // in the table, not an error, and CanopyTest::MonopoleBasis records the
    // same effect for the same reason.
    //
    // THREE GUARDS, each naming its convention, because each of these values
    // has a reachable DEFAULT that would make the operator silently wrong
    // rather than noisy:
    //
    //   * softening <= 0. M2LKernelParams::softening defaults to 0.0,
    //     documented as the unsoftened kernel a sweep driven directly by a
    //     test runs at (src/Canopy_FarFieldContract.hpp:115-121), and the
    //     derivative ladder requires b > 0 because b = 0 at r = 0 divides by
    //     zero.
    //   * max_d outside [0, n_levels). Abort rather than index past the end
    //     of unit_w.
    //   * unit_w[max_d] <= 0. unit_w is all zeros until something calls
    //     set_root_half_width (src/Canopy_DownwardSweep.hpp:402-404), and a
    //     zero entry yields R = 0 and a finite, WRONG, physical operator.
    //
    // None of the three is reachable from the tasks in
    // tasks/cartesian-taylor-basis.md -- T4 sets an explicit positive
    // softening and Solver::_push_root_half_width runs before every
    // _downward.setup() -- so they guard a later caller rather than a failure
    // to expect.
    //
    // A PLAIN STATIC MEMBER, not KOKKOS_INLINE_FUNCTION, per the Conventions
    // row on host-side operator construction: it runs once on host and a basis
    // needing LAPACK here must be allowed to call it.
    //
    // `ops` is allocated WithoutInitializing, so EVERY entry of every column
    // is written -- the loop below is unconditional and has no continue.
    //
    // `aux` is unused: the operator needs b, and on THIS path b comes from
    // kernel_params directly. aux carries it only for m2l_translate, whose
    // signature has no kernel_params at all. See aux_tables_type.
    // =======================================================================
    template <class KeyType, class AuxType, class OpsView>
    static void build_m2l_operators( const KeyType* keys, int n_keys,
                                     const double* unit_w, int n_levels,
                                     const Canopy::M2LKernelParams&
                                         kernel_params,
                                     const AuxType& aux, const OpsView& ops )
    {
        (void)aux;

        if ( !( kernel_params.softening > 0.0 ) )
            Kokkos::abort(
                "CartesianTaylorBasis::build_m2l_operators: "
                "M2LKernelParams::softening is the Plummer softening LENGTH "
                "eps and must be strictly positive. The kernel's b is eps^2, "
                "the quantity added to r^2, and the derivative ladder "
                "requires b > 0 because b = 0 at r = 0 divides by zero. The "
                "default is 0.0 -- the unsoftened kernel -- which this basis "
                "has no expansion of." );

        // b is the softening SQUARED. The square is taken here, once.
        const double b = kernel_params.softening * kernel_params.softening;

        for ( int j = 0; j < n_keys; j++ )
        {
            const int max_d = keys[j].max_d;

            if ( max_d < 0 || max_d >= n_levels )
                Kokkos::abort(
                    "CartesianTaylorBasis::build_m2l_operators: a key's "
                    "max_d is outside [0, n_levels). This basis declares "
                    "key_needs_level = true, so canonicalize_key is the "
                    "identity and max_d is a real tree level that indexes "
                    "unit_w; an out-of-range one would read past the end of "
                    "that array." );

            const double w_unit = unit_w[max_d];

            if ( !( w_unit > 0.0 ) )
                Kokkos::abort(
                    "CartesianTaylorBasis::build_m2l_operators: "
                    "unit_w[max_d] is not positive. unit_w is the HALF-WIDTH "
                    "at each depth and is all zeros until something calls "
                    "DownwardSweep::set_root_half_width; a zero entry gives "
                    "R = 0 and a finite, wrong, physical operator rather "
                    "than a failure." );

            // R = c_target - c_source = -(ii, jj, kk) * unit_w[max_d]. The
            // key's offset is SOURCE MINUS TARGET; see the block above.
            const double R[3] = {
                -static_cast<double>( keys[j].ii ) * w_unit,
                -static_cast<double>( keys[j].jj ) * w_unit,
                -static_cast<double>( keys[j].kk ) * w_unit };

            Scalar op[m2l_op_entries];
            m2l_operator_block( R, static_cast<Scalar>( b ), op );

            for ( int q_slot = 0; q_slot < m2l_num_src_coeffs; q_slot++ )
                for ( int p_slot = 0; p_slot < num_coeffs_per_cell; p_slot++ )
                    ops( p_slot, q_slot, j ) =
                        op[m2l_op_index( p_slot, q_slot )];
        }
    }

    // =======================================================================
    // m2l_pre_cell -- NO-OP, and it must stay one.
    //
    // This basis contracts the source moments directly against the operator
    // column and has no per-source-cell work to hoist.
    //
    // It MUST NOT WRITE TO SCRATCH: the accumulator living there is zeroed
    // once per team, before the team's first pair, and carried across every
    // pair of that team (src/Canopy_DownwardSweep.hpp:2226-2247). A pre_cell
    // that cleared it would discard every pair but the last.
    // =======================================================================
    template <class TeamMember, class MView, class OpsType, class ScratchView>
    KOKKOS_INLINE_FUNCTION static void
    m2l_pre_cell( const TeamMember& team_member, const MView& M_full,
                  int source_cell, const OpsType& ops,
                  const ScratchView& scratch )
    {
        (void)team_member;
        (void)M_full;
        (void)source_cell;
        (void)ops;
        (void)scratch;
    }

    // =======================================================================
    // m2l_core -- the per-pair apply.
    //
    //   acc( acc_slot(p, c, s) )
    //       += sum_q ops(p, q, op_idx) * M(source_cell, q, c)
    //
    // with the operator column already carrying the (-1)^{|q|} b_{p+q}(R) of
    // canopy-questions.md §4 -- it was written by m2l_operator_block, and
    // nothing about the sign or the factorials is re-applied here.
    //
    // THE ACCUMULATOR LAYOUT IS NOT THIS FUNCTION'S TO CHOOSE. It is fixed at
    // acc_slot above because m2l_post_cell already reads it; acc_slot is the
    // whole of the contract between the two stages.
    //
    // The operator set is reached only through op_idx, so the sweep never
    // indexes it.
    //
    // ONE NAMED LOCAL PER (p, c), accumulated over q and added into the
    // accumulator once. Two reasons, and neither is style. First, the
    // accumulator lives in team scratch and the local keeps the inner loop in
    // a register. Second, m2l_translate has the IDENTICAL shape -- same q
    // order, same taylor_accumulate, same single write -- which is what makes
    // the two paths bit-identical for one pair (risk R4). The final `+=` into
    // a scratch slot that pair-one finds at +0.0 adds nothing and rounds
    // nothing, so the equality holds exactly and not merely closely.
    //
    // Parameters mirror Canopy::LaplaceKernel::m2l_core.
    // =======================================================================
    template <class TeamMember, class MView, class OpsType, class ScratchView>
    KOKKOS_INLINE_FUNCTION static void
    m2l_core( const TeamMember& team_member, const MView& M_full,
              int source_cell, const OpsType& ops, int op_idx,
              const ScratchView& scratch )
    {
        using acc_type =
            m2l_accumulator_type<typename ScratchView::memory_space>;
        constexpr int n_acc = num_coeffs_per_cell * num_comp_slots;
        scalar_type* acc_base =
            reinterpret_cast<scalar_type*>( scratch.data() );
        acc_type team_acc( acc_base, n_acc );

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team_member, num_coeffs_per_cell ),
            [&]( const int out_idx )
            {
                for ( int c = 0; c < NComps; c++ )
                {
                    Scalar acc = static_cast<Scalar>( 0 );
                    for ( int q_slot = 0; q_slot < m2l_num_src_coeffs;
                          q_slot++ )
                        taylor_accumulate( acc, ops( out_idx, q_slot, op_idx ),
                                           M_full( source_cell, q_slot, c ) );

                    for ( int st = 0; st < sets_per_component; st++ )
                        team_acc( acc_slot( out_idx, c, st ) ) += acc;
                }
            } );
    }

    // =======================================================================
    // m2l_post_cell -- flush the team's accumulator into the locals view.
    //
    // REAL, NOT A STUB, and written in T2 rather than deferred with the three
    // stages above. This is the ONLY stage that writes the locals view: a
    // no-op here compiles cleanly and leaves every local coefficient zero,
    // which is precisely the failure the "Deliberate deviations" section of
    // tasks/cartesian-taylor-basis.md records. It is also why the accumulator
    // layout had to be settled in T2 -- this function indexes it.
    //
    //   L_out(target_cell, p, comp_set_slot(c, s))
    //       += acc( acc_slot(p, c, s) )
    //
    // += RATHER THAN = : on a shared target, L2L from a shallower depth has
    // already written there before the per-depth M2L runs. Each target cell is
    // owned by exactly one team, so no atomics are needed.
    //
    // `ops` is in the signature because the sweep passes it to every stage;
    // this basis needs nothing from it here.
    // =======================================================================
    template <class TeamMember, class ScratchView, class LView, class OpsType>
    KOKKOS_INLINE_FUNCTION static void
    m2l_post_cell( const TeamMember& team_member, const ScratchView& scratch,
                   const LView& L_out, int target_cell, const OpsType& ops )
    {
        (void)ops;

        using acc_type =
            m2l_accumulator_type<typename ScratchView::memory_space>;
        constexpr int n_acc = num_coeffs_per_cell * num_comp_slots;
        scalar_type* acc_base =
            reinterpret_cast<scalar_type*>( scratch.data() );
        acc_type team_acc( acc_base, n_acc );

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team_member, num_coeffs_per_cell ),
            [&]( const int out_idx )
            {
                for ( int c = 0; c < NComps; c++ )
                    for ( int st = 0; st < sets_per_component; st++ )
                        L_out( target_cell, out_idx,
                               comp_set_slot( c, st ) ) +=
                            team_acc( acc_slot( out_idx, c, st ) );
            } );
    }

    // =======================================================================
    // L2L: shift the parent's local expansion to each child center.
    //
    //   l^ch_p += sum_{p' >= p} s^{p'-p} / (p'-p)! * l^par_p' ,
    //                                            s = c_child - c_parent
    //
    // canopy-questions.md §4, the L2L row. The plain binomial Taylor shift
    // again, read the other way: substituting (x - c_par) = (x - c_ch) + s
    // into u(x) = sum_p' (x-c_par)^p'/p'! l^par_p' and collecting powers of
    // (x - c_ch) gives exactly this. Unlike M2M it IS truncated -- the sum
    // only runs over the p' the parent carries -- which is the usual Taylor
    // L2L and not a defect of this basis.
    //
    //   dx, dy, dz          CHILD CENTER MINUS PARENT CENTER (s), length units
    //                       (src/Canopy_DownwardSweep.hpp:2389-2397)
    //   w_child, w_parent   both HALF-WIDTHS. BOTH IGNORED -- physical,
    //                       un-normalized coefficients; see the class header.
    //   aux                 empty; this basis has no tables. IGNORED.
    //   L_child_out         2D slice L_child_out(coeff_slot, comp_set_slot)
    //
    // NON-ATOMIC. The sweep runs one team per parent and each child has
    // exactly one parent.
    // =======================================================================
    template <class TeamMember, class LView, class AuxType, class LChildType>
    KOKKOS_INLINE_FUNCTION static void
    l2l_translate( const TeamMember& team_member, const LView& L_full,
                   int parent_cell, Scalar dx, Scalar dy, Scalar dz,
                   Scalar w_child, Scalar w_parent, const AuxType& aux,
                   const LChildType& L_child_out )
    {
        (void)w_child;
        (void)w_parent;
        (void)aux;

        Scalar t[num_coeffs_per_cell];
        taylor_monomials( dx, dy, dz, t );

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team_member, num_coeffs_per_cell ),
            [&]( const int out_idx )
            {
                int pp[3];
                Canopy::CartesianTaylor::inverse_slot( out_idx, pp );

                // p' = p + a with |p'| <= P_ORDER, so |a| <= P_ORDER - |p|.
                const int rem = P_ORDER - ( pp[0] + pp[1] + pp[2] );

                // Every (component, set) slot shifts the same way -- L2L acts
                // on each independently -- so this walks the flat slot range
                // rather than the (component, set) pair. At
                // sets_per_component = 1 the range IS the component range.
                for ( int cs = 0; cs < num_comp_slots; cs++ )
                {
                    Scalar acc = static_cast<Scalar>( 0 );

                    for ( int ax = 0; ax <= rem; ax++ )
                        for ( int ay = 0; ax + ay <= rem; ay++ )
                            for ( int az = 0; ax + ay + az <= rem; az++ )
                            {
                                const int src =
                                    Canopy::CartesianTaylor::slot(
                                        pp[0] + ax, pp[1] + ay, pp[2] + az );
                                const int sh =
                                    Canopy::CartesianTaylor::slot( ax, ay,
                                                                   az );
                                taylor_accumulate(
                                    acc, t[sh],
                                    L_full( parent_cell, src, cs ) );
                            }

                    L_child_out( out_idx, cs ) += acc;
                }
            } );
    }

    // =======================================================================
    // L2P: evaluate the local expansion, and its gradient, at a particle.
    //
    //   u(x)         = sum_p a^p / p! * l_p ,      a = particle - cell center
    //   d_i u(x)     = sum_{p : p_i >= 1} a^{p-e_i} / (p-e_i)! * l_p
    //
    // canopy-questions.md §4, the L2P row, and the shifted-multi-index
    // gradient of the design document.
    //
    // THE GRADIENT IS ANALYTIC. It is the exact derivative of the polynomial
    // being evaluated -- the same shift table read at slot(p - e_i) -- so it
    // costs one extra table lookup per term and carries no step size, no
    // cancellation and no truncation of its own. The solid-harmonic basis
    // takes a central finite difference there
    // (src/Canopy_LaplaceKernel.hpp:1378-1407); this basis does not need one,
    // and nothing here touches that file. Note |p - e_i| <= P_ORDER - 1, so
    // the same order-P_ORDER shift table serves both.
    //
    //   dx, dy, dz  PARTICLE POSITION MINUS CELL CENTER (a), length units
    //               (src/Canopy_DownwardSweep.hpp:2650-2672)
    //   w_self      this leaf's HALF-WIDTH. IGNORED -- physical,
    //               un-normalized coefficients; see the class header.
    //   phi_out     Scalar[NComps], WRITTEN WITH = , not += : the sweep
    //               accumulates it into its own output view afterwards
    //               (src/Canopy_DownwardSweep.hpp:2672-2674), exactly as it
    //               does for the solid-harmonic basis.
    //   grad_out    a GradWriter-shaped accessor: grad_out(c, dim) returns a
    //               writable Scalar& (src/Canopy_DownwardSweep.hpp:190-198).
    //               Written only when compute_gradient.
    //
    // SET 0 ONLY, which at sets_per_component = 1 is every set there is.
    //
    // Not a team operator: the sweep runs one thread per particle over a
    // RangePolicy, and each particle writes only its own phi and gradient, so
    // there is no atomic and no team range here.
    // =======================================================================
    template <class LView, class GradAccess>
    KOKKOS_INLINE_FUNCTION static void
    l2p_evaluate( const LView& L_full, int leaf_cell, Scalar dx, Scalar dy,
                  Scalar dz, Scalar w_self, Scalar ( &phi_out )[NComps],
                  const GradAccess& grad_out, bool compute_gradient )
    {
        (void)w_self;

        Scalar t[num_coeffs_per_cell];
        taylor_monomials( dx, dy, dz, t );

        for ( int c = 0; c < NComps; c++ )
        {
            Scalar acc = static_cast<Scalar>( 0 );
            for ( int s = 0; s < num_coeffs_per_cell; s++ )
                taylor_accumulate(
                    acc, t[s], L_full( leaf_cell, s, comp_set_slot( c, 0 ) ) );
            phi_out[c] = acc;
        }

        if ( !compute_gradient )
            return;

        for ( int c = 0; c < NComps; c++ )
            for ( int i = 0; i < 3; i++ )
                grad_out( c, i ) = static_cast<Scalar>( 0 );

        for ( int s = 0; s < num_coeffs_per_cell; s++ )
        {
            int p[3];
            Canopy::CartesianTaylor::inverse_slot( s, p );

            for ( int i = 0; i < 3; i++ )
            {
                // p - e_i carries a negative component exactly when p_i == 0,
                // and those terms are the constants of the polynomial in x_i
                // and differentiate to zero.
                if ( p[i] == 0 )
                    continue;

                int sh_k[3] = { p[0], p[1], p[2] };
                sh_k[i] -= 1;
                const int sh = Canopy::CartesianTaylor::slot( sh_k[0], sh_k[1],
                                                              sh_k[2] );

                for ( int c = 0; c < NComps; c++ )
                {
                    Scalar g = grad_out( c, i );
                    taylor_accumulate(
                        g, t[sh],
                        L_full( leaf_cell, s, comp_set_slot( c, 0 ) ) );
                    grad_out( c, i ) = g;
                }
            }
        }
    }
};

} // namespace Canopy

#endif // CANOPY_CARTESIAN_TAYLOR_BASIS_HPP
