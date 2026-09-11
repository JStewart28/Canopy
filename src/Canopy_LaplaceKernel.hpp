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

#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace Canopy
{

// ============================================================================
// Helper: device-callable double factorial n!!
// ============================================================================
template <class Scalar>
KOKKOS_INLINE_FUNCTION Scalar double_factorial( int n )
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
// Based on Greengard eqs. 3.33, 3.34. Upward recurrence in n.
// ============================================================================
template <class Scalar>
KOKKOS_INLINE_FUNCTION Scalar Pnm_impl( int n, int m, Scalar x )
{
    if ( m < 0 || m > n )
        return 0.0;

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
        pn = ( ( 2 * l - 1 ) * x * pnm1 - ( l + m - 1 ) * pnm2 ) / ( l - m );
        pnm2 = pnm1;
        pnm1 = pn;
    }
    return pn;
}

// ============================================================================
// Device-callable complex spherical harmonic Y_{n,m}(theta, phi).
// ============================================================================
template <class Scalar>
KOKKOS_INLINE_FUNCTION Kokkos::complex<Scalar> Ynm( int n, int m, Scalar theta,
                                                    Scalar phi )
{
    using complex = Kokkos::complex<Scalar>;

    const int mp = ( m < 0 ) ? -m : m;
    const Scalar x = Kokkos::cos( theta );

    const Scalar Pnm = Pnm_impl<Scalar>( n, mp, x );

    const Scalar norm =
        Kokkos::sqrt( Kokkos::tgamma( static_cast<Scalar>( n - mp + 1 ) ) /
                      Kokkos::tgamma( static_cast<Scalar>( n + mp + 1 ) ) );

    const Scalar cos_mphi = Kokkos::cos( static_cast<Scalar>( m ) * phi );
    const Scalar sin_mphi = Kokkos::sin( static_cast<Scalar>( m ) * phi );

    return complex( norm * Pnm * cos_mphi, norm * Pnm * sin_mphi );
}

// ============================================================================
// Convert Cartesian offset to spherical coordinates.
// ============================================================================
template <class Scalar>
KOKKOS_INLINE_FUNCTION void cartesian_to_spherical( Scalar x, Scalar y,
                                                    Scalar z, Scalar& rho,
                                                    Scalar& theta, Scalar& phi )
{
    rho = Kokkos::sqrt( x * x + y * y + z * z );
    theta = ( rho > 0.0 ) ? Kokkos::acos( z / rho ) : 0.0;
    phi = Kokkos::atan2( y, x );
}

// --------------------------------------------------------------------------
// Recover (j, k) with k >= 0 from flat triangular index idx = j(j+1)/2 + k
// --------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void unflatten_triangular( int idx, int& j, int& k )
{
    j = static_cast<int>( Kokkos::floor(
        ( -1.0 + Kokkos::sqrt( 1.0 + 8.0 * static_cast<double>( idx ) ) ) *
        0.5 ) );
    while ( j * ( j + 1 ) / 2 > idx )
        j--;
    while ( ( j + 1 ) * ( j + 2 ) / 2 <= idx )
        j++;
    k = idx - j * ( j + 1 ) / 2;
}

// ============================================================================
// LaplaceKernel
//
// FMM kernel for the 1/r Green's function (gravity, electrostatics).
//
// Template parameters:
//   Scalar  - floating-point type (double or float)
//   P       - expansion order
//   NComps  - number of simultaneous solves (default 1; 3 for
//             Biot-Savart via three parallel Laplace solves)
//
// Storage convention:
//   Multipoles M(cell_idx, coeff_idx, comp_idx)
//   Locals     L(cell_idx, coeff_idx, comp_idx)
//   Triangular coeff_idx = n*(n+1)/2 + m  for m >= 0
//
// Static methods:
//   p2m_contribution - add particle to leaf multipole
//   m2m_translate    - translate child multipole to parent
//   m2l_translate    - translate source multipole to target local
//   l2l_translate    - translate parent local to child local
//   l2p_evaluate     - evaluate local at a particle, potential + gradient
// ============================================================================

template <class Scalar, int P, int NComps = 1>
struct LaplaceKernel
{
    using scalar_type = Scalar;

    // -----------------------------------------------------------------------
    // The coefficient contract. Shared code (UpwardSweep, DownwardSweep,
    // coalesced_view_exchange) describes multipole/local storage and its MPI
    // packing through these three members and never through
    // Kokkos::complex:
    //
    //   coeff_type            - the element type of multipole/local storage.
    //                           The identity element is value-initialization,
    //                           coeff_type(), which for this basis is
    //                           (+0.0, +0.0).
    //   component_scalar_type - the real scalar MPI sees. It is Scalar, not
    //                           double: this basis carries a live float path
    //                           and the MPI datatype is selected from
    //                           sizeof(component_scalar_type) at three sites.
    //   scalars_per_coeff     - how many contiguous component_scalar_type
    //                           make up one coeff_type. 2 here; 1 for a
    //                           real-coefficient basis.
    //
    // MPI is handed (reinterpret_cast<component_scalar_type*>(coeff_ptr),
    // scalars_per_coeff * n_coeffs), so the contract a basis signs by
    // supplying these is that coeff_type is exactly scalars_per_coeff
    // contiguous component_scalar_type with no padding. The static_assert
    // below is that contract, checked at instantiation.
    // -----------------------------------------------------------------------
    using coeff_type = Kokkos::complex<Scalar>;
    using component_scalar_type = Scalar;
    static constexpr int scalars_per_coeff = 2;

    static_assert( sizeof( coeff_type ) ==
                       scalars_per_coeff * sizeof( component_scalar_type ),
                   "LaplaceKernel: coeff_type is not scalars_per_coeff "
                   "contiguous component_scalar_type, so the MPI packing in "
                   "coalesced_view_exchange and in the two shared-cell "
                   "reductions would transfer the wrong byte count" );

    // The basis-private spelling of coeff_type, used throughout the
    // solid-harmonic arithmetic below, where the quantities really are
    // complex numbers and several of them (Ynm tables, i^k tables,
    // conjugation under m -> -m) are not coefficients at all. Shared code
    // must use coeff_type; this name is retained only so the solid-harmonic
    // operators keep reading as complex arithmetic.
    using complex_type = coeff_type;

    static constexpr int max_order = P;
    static constexpr int num_coeffs_per_cell = ( P + 1 ) * ( P + 2 ) / 2;
    static constexpr int num_components = NComps;

    // -----------------------------------------------------------------------
    // The M2L operator set. Opaque to the sweep, which stores one of these,
    // hands it back to m2l_pre_cell / m2l_core / m2l_post_cell and never
    // indexes it — the only thing the sweep says about an operator is the
    // integer op_idx its CSR carries.
    //
    // For this basis it is the hashed operator table, shape
    // (num_coeffs_per_cell, m2l_num_src_coeffs, n_unique_ops): column op_idx
    // is the dense (Nt, Ns) operator for one canonicalized translation key.
    // LayoutLeft so subview(ops, ALL, ALL, op_idx) is a contiguous
    // column-major (Nt, Ns) matrix consumable by cuBLAS / hipBLAS /
    // KokkosBlas::gemm without copy or transpose.
    //
    // Parameterized on the memory space because the kernel itself is not;
    // the sweep supplies its own. Spell it
    //     typename KernelType::template m2l_operators_type<memory_space>
    // -----------------------------------------------------------------------
    // Element type is coeff_type, not complex_type: the operator-table
    // element type and the coefficient element type cannot be chosen
    // independently and silently disagree, since m2l_core contracts one
    // against the other.
    template <class MemorySpace>
    using m2l_operators_type =
        Kokkos::View<coeff_type***, Kokkos::LayoutLeft, MemorySpace>;

    // -----------------------------------------------------------------------
    // Auxiliary tables. Precomputed, order-dependent data that the basis's
    // own operators need and that shared code neither builds, indexes nor
    // knows the shape of: the sweeps store one of these, hand it back to
    // m2m_translate / m2l_translate / l2l_translate / m2l_build_operator,
    // and never look inside. A basis needing no such data returns an empty
    // struct.
    //
    // For this basis the one member is the A_{n,m} normalization table of
    // Greengard & Rokhlin's translation theorems, flat, indexed by
    // a_index(n, m) = n*n + n + m.
    //
    // Element type is scalar_type, not component_scalar_type. The two are
    // the same Scalar for this basis, so the choice moves no bits and is
    // made here only to settle what it means. component_scalar_type is a
    // *storage* trait: it names the real scalar that coeff_type decomposes
    // into for MPI packing, and it exists so the sweeps can reinterpret_cast
    // a coefficient buffer. A_{n,m} is not a coefficient, is never packed,
    // and never crosses a rank boundary; it is a real constant multiplied
    // into the translation arithmetic. Its natural type is therefore the
    // basis's *arithmetic* scalar, scalar_type — which is what
    // m2m_translate, m2l_translate, l2l_translate and m2l_build_operator
    // all read it into (`const Scalar A_jk = ...`). A basis whose packed
    // component is narrower than its arithmetic type (a blocked or
    // mixed-precision coeff_type) would otherwise silently demote this table
    // along with its storage, which is a decision about bandwidth leaking
    // into a decision about accuracy.
    //
    // Parameterized on the memory space because the kernel is not; the
    // caller supplies its own. Two spaces are genuinely in use: the three
    // device operators consume the sweep's memory_space table, while
    // m2l_build_operator runs on host over a Kokkos::HostSpace one. Spell it
    //     typename KernelType::template aux_tables_type<memory_space>
    // -----------------------------------------------------------------------
    template <class MemorySpace>
    struct aux_tables_type
    {
        Kokkos::View<scalar_type*, MemorySpace> A_table;
    };

    // Build the auxiliary tables for expansion order `order` (= max_order;
    // the argument is passed rather than read off the basis so a caller can
    // build a table for a different order in a test).
    //
    // The A_{n,m} table is built to degree 2*order, not order: M2L accesses
    // A at degree n+j where both n and j run up to P. A table one degree
    // short does not fault — m2l_build_operator skips a zero A entry with
    // `continue` — it silently produces a wrong operator. The factor of two
    // is the reason this table cannot live in shared code: it is a fact
    // about the solid-harmonic translation theorems and nothing else.
    //
    // Host function, not device-callable: it allocates and fills a View.
    template <class MemorySpace>
    static aux_tables_type<MemorySpace> build_aux_tables( int order )
    {
        aux_tables_type<MemorySpace> aux;
        aux.A_table =
            build_A_coefficients<scalar_type, MemorySpace>( 2 * order );
        return aux;
    }

    // One half of the M2L team scratch, viewed as a scalar array. See
    // m2l_scratch_bytes for the layout and for why the real/imag split is
    // arithmetic-visible rather than cosmetic.
    template <class ScratchSpace>
    using m2l_accumulator_type =
        Kokkos::View<scalar_type*, ScratchSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    // -----------------------------------------------------------------------
    // Retrieve a coefficient from 3D storage with symmetry for m < 0.
    //
    // This kernel's Ynm uses the Greengard convention
    //     Y_{n,m} = sqrt((n-|m|)!/(n+|m|)!) * P_n^{|m|}(cos theta) * exp(i m
    //     phi)
    // so Y_{n,-m} = conj(Y_{n,m}) with no extra (-1)^m phase between +m and
    // -m. For real-sourced kernels the corresponding multipole/local symmetry
    // is therefore
    //     M_{n,-m} = conj(M_{n,m})    (no (-1)^m factor).
    //
    // Note: this is NOT the Condon-Shortley-style symmetry
    //     M_{n,-m} = (-1)^m * conj(M_{n,m})
    // used by codes whose Ynm applies Condon-Shortley between +m and -m
    // (e.g. ExaFMM). Those two sign conventions cannot be mixed.
    // -----------------------------------------------------------------------
    template <class CView>
    KOKKOS_INLINE_FUNCTION static complex_type
    get_coeff_3d( const CView& C, int cell, int n, int m, int comp )
    {
        if ( n < 0 || n > P )
            return complex_type( 0.0, 0.0 );

        const int abs_m = ( m < 0 ) ? -m : m;
        if ( abs_m > n )
            return complex_type( 0.0, 0.0 );

        const complex_type val = C( cell, coeff_index( n, abs_m ), comp );
        if ( m >= 0 )
            return val;

        return complex_type( val.real(), -val.imag() );
    }

    // -----------------------------------------------------------------------
    // i^pow as a complex value (pow is reduced mod 4)
    // -----------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    static complex_type i_power( int pow )
    {
        int p = ( ( pow % 4 ) + 4 ) % 4;
        switch ( p )
        {
        case 0:
            return complex_type( 1.0, 0.0 );
        case 1:
            return complex_type( 0.0, 1.0 );
        case 2:
            return complex_type( -1.0, 0.0 );
        case 3:
        default:
            return complex_type( 0.0, -1.0 );
        }
    }

    // =======================================================================
    // P2M: add a particle's contribution to its leaf cell's multipole.
    //
    //   M_{n,m,c} += q_c * rho^n * Y_{n,-m}(alpha, beta)
    //
    // Parameters:
    //   charges     - Scalar[NComps] per-component charges for this particle
    //   dx, dy, dz  - particle_position - cell_center (Cartesian)
    //   M_out       - 2D slice: M_out(coeff_idx, comp_idx)
    // =======================================================================
    // Scale-normalized P2M: produces M̄_{n,m} = M_{n,m} / w_self^{n+1}.
    // w_self is the leaf cell's half-width. The {n+1} convention keeps
    // intermediates O(q · 2^max_d) (linear in depth) rather than
    // geometric, so FP32 stays well-conditioned at deep trees.
    template <class MSliceType>
    KOKKOS_INLINE_FUNCTION static void
    p2m_contribution( const Scalar ( &charges )[NComps], Scalar dx, Scalar dy,
                      Scalar dz, Scalar w_self, const MSliceType& M_out )
    {
        Scalar rho, theta, phi;
        cartesian_to_spherical( dx, dy, dz, rho, theta, phi );

        const Scalar inv_w = static_cast<Scalar>( 1 ) / w_self;
        // term_n = rho^n / w_self^{n+1};  term_0 = 1/w_self.
        Scalar term = inv_w;
        const Scalar rho_inv_w = rho * inv_w;
        for ( int n = 0; n <= P; n++ )
        {
            for ( int m = 0; m <= n; m++ )
            {
                const complex_type Ynm_neg_m = Ynm<Scalar>( n, -m, theta, phi );

                const int idx = coeff_index( n, m );
                for ( int c = 0; c < NComps; c++ )
                {
                    const complex_type contrib =
                        charges[c] * term * Ynm_neg_m;
                    Kokkos::atomic_add( &M_out( idx, c ).real(),
                                        contrib.real() );
                    Kokkos::atomic_add( &M_out( idx, c ).imag(),
                                        contrib.imag() );
                }
            }
            term *= rho_inv_w;
        }
    }

    // =======================================================================
    // M2M: translate a child's multipole into the parent's frame.
    // Greengard Theorem 5.22.
    // =======================================================================
    // Scale-normalized M2M: consumes M̄^c = M^c / w_c^{n+1} and produces
    // M̄^p = M^p / w_p^{j+1}. The rho^n factor becomes (rho/w_c)^n and a
    // per-output-row constant (w_c/w_p)^{j+1} is applied at the end.
    // For a standard octree (w_c = w_p/2) this is 2^{-(j+1)}, but we
    // compute it from the actual widths so the kernel doesn't assume
    // a fixed refinement ratio.
    template <class TeamMember, class MView, class AuxType, class MParentType>
    KOKKOS_INLINE_FUNCTION static void
    m2m_translate( const TeamMember& team_member, const MView& M_full,
                   int child_cell, Scalar dx, Scalar dy, Scalar dz,
                   Scalar w_child, Scalar w_parent, const AuxType& aux,
                   const MParentType& M_parent_out )
    {
        // The A_{n,m} table this basis puts in its aux tables. Named
        // locally so the translation arithmetic below reads as the
        // theorem it implements.
        const auto& A_table = aux.A_table;

        Scalar rho, theta, phi;
        cartesian_to_spherical( dx, dy, dz, rho, theta, phi );

        const Scalar inv_w_c = static_cast<Scalar>( 1 ) / w_child;
        const Scalar rho_norm = rho * inv_w_c;       // rho / w_c, O(1)
        const Scalar w_ratio = w_child / w_parent;   // w_c / w_p, O(1)

        // Precompute (w_c/w_p)^{j+1} for j = 0..P. Shared by every output
        // row in this team. Avoids a per-thread O(P) inner loop.
        Scalar w_factor_tbl[P + 1];
        w_factor_tbl[0] = w_ratio;
        for ( int e = 1; e <= P; e++ )
            w_factor_tbl[e] = w_factor_tbl[e - 1] * w_ratio;

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team_member, num_coeffs_per_cell ),
            [&]( const int out_idx )
            {
                int j, k;
                unflatten_triangular( out_idx, j, k );

                const Scalar A_jk = A_table( a_index( j, k ) );
                if ( A_jk == 0.0 )
                    return;

                complex_type accum[NComps];
                for ( int c = 0; c < NComps; c++ )
                    accum[c] = complex_type( 0.0, 0.0 );

                Scalar rho_pow_n = 1.0;
                for ( int n = 0; n <= j; n++ )
                {
                    for ( int m = -n; m <= n; m++ )
                    {
                        const int jmn = j - n;
                        const int kmm = k - m;
                        const int abs_km = ( kmm < 0 ) ? -kmm : kmm;

                        if ( abs_km > jmn )
                            continue;

                        const Scalar A_nm = A_table( a_index( n, m ) );
                        const Scalar A_jmn_kmm = A_table( a_index( jmn, kmm ) );

                        const int abs_k = k;
                        const int abs_m = ( m < 0 ) ? -m : m;
                        const complex_type ip =
                            i_power( abs_k - abs_m - abs_km );

                        const complex_type Y = Ynm<Scalar>( n, -m, theta, phi );

                        const Scalar coef_scalar =
                            A_nm * A_jmn_kmm / A_jk * rho_pow_n;
                        const complex_type pre_factor = ip * coef_scalar * Y;

                        for ( int c = 0; c < NComps; c++ )
                        {
                            const complex_type M_child_val =
                                get_coeff_3d( M_full, child_cell, jmn, kmm, c );
                            accum[c] += M_child_val * pre_factor;
                        }
                    }
                    rho_pow_n *= rho_norm;
                }

                // Apply per-row scale (w_c / w_p)^{j+1}.
                const Scalar w_factor = w_factor_tbl[j];

                for ( int c = 0; c < NComps; c++ )
                    M_parent_out( out_idx, c ) += accum[c] * w_factor;
            } );
    }

    // =======================================================================
    // M2L: translate a source cell's multipole into a target cell's local.
    // Greengard Theorem 5.23.
    //
    //   L_{j,k}^target += sum_{n=0}^{P} sum_{m=-n}^{n}
    //       M_{n,m}^source * (-1)^n * i^(|k-m|-|k|-|m|)
    //       * A_{n,m} * A_{j,k} / A_{n+j, m-k}
    //       * Y_{n+j, m-k}(alpha, beta) / rho^(n+j+1)
    //
    // The (-1)^n factor is a real sign introduced by the irregular-to-regular
    // solid-harmonic reflection and does NOT cancel inside the A ratio — with
    // A_{n,m} = (-1)^n / sqrt((n-|m|)!(n+|m|)!) the A ratio already equals a
    // positive factorial ratio.
    //
    // where (rho, alpha, beta) = source_center - target_center in spherical.
    // =======================================================================
    // Scale-normalized fallback M2L. Consumes M̄^s = M^s / w_s^{n+1} and
    // produces L̄^t = L^t · w_t^j. The original 1/rho^{n+j+1} factor
    // expands as (w_s/rho)^{n+1} · (w_t/rho)^j so each per-pair table is
    // O(1) magnitude regardless of cell depths — FP32-safe.
    template <class TeamMember, class MView, class AuxType, class LTargetType>
    KOKKOS_INLINE_FUNCTION static void
    m2l_translate( const TeamMember& team_member, const MView& M_full,
                   int source_cell, Scalar dx, Scalar dy, Scalar dz,
                   Scalar w_source, Scalar w_target, const AuxType& aux,
                   const LTargetType& L_target_out )
    {
        // The A_{n,m} table this basis puts in its aux tables. Named
        // locally so the translation arithmetic below reads as the
        // theorem it implements.
        const auto& A_table = aux.A_table;

        Scalar rho, theta, phi;
        cartesian_to_spherical( dx, dy, dz, rho, theta, phi );

        const Scalar inv_rho = ( rho > 0.0 ) ? ( 1.0 / rho ) : 0.0;
        const Scalar ws_inv_rho = w_source * inv_rho;   // (w_s/rho), O(1)
        const Scalar wt_inv_rho = w_target * inv_rho;   // (w_t/rho), O(1)

        // (w_s/rho)^{n+1} for n = 0..P  — index by (n+1).
        Scalar ws_pow_tbl[P + 2];
        ws_pow_tbl[0] = static_cast<Scalar>( 1 );
        for ( int e = 1; e <= P + 1; e++ )
            ws_pow_tbl[e] = ws_pow_tbl[e - 1] * ws_inv_rho;
        // (w_t/rho)^j for j = 0..P
        Scalar wt_pow_tbl[P + 1];
        wt_pow_tbl[0] = static_cast<Scalar>( 1 );
        for ( int e = 1; e <= P; e++ )
            wt_pow_tbl[e] = wt_pow_tbl[e - 1] * wt_inv_rho;

        constexpr int max_L = 2 * P;
        constexpr int Y_size = ( max_L + 1 ) * ( max_L + 1 );
        complex_type Y_tbl[Y_size];
        for ( int L = 0; L <= max_L; L++ )
            for ( int M = -L; M <= L; M++ )
                Y_tbl[L * L + L + M] = Ynm<Scalar>( L, M, theta, phi );

        constexpr int ip_stride = 2 * P + 1;
        constexpr int ip_size = ( P + 1 ) * ip_stride;
        complex_type ip_tbl[ip_size];
        for ( int kk = 0; kk <= P; kk++ )
            for ( int mm = -P; mm <= P; mm++ )
            {
                const int abs_kk = kk;
                const int abs_mm = ( mm < 0 ) ? -mm : mm;
                const int kmm = kk - mm;
                const int abs_kmm = ( kmm < 0 ) ? -kmm : kmm;
                ip_tbl[kk * ip_stride + ( mm + P )] =
                    i_power( abs_kmm - abs_kk - abs_mm );
            }

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team_member, num_coeffs_per_cell ),
            [&]( const int out_idx )
            {
                int j, k;
                unflatten_triangular( out_idx, j, k );

                const Scalar A_jk = A_table( a_index( j, k ) );
                const Scalar wt_pow_j = wt_pow_tbl[j];

                complex_type accum[NComps];
                for ( int c = 0; c < NComps; c++ )
                    accum[c] = complex_type( 0.0, 0.0 );

                for ( int n = 0; n <= P; n++ )
                {
                    const Scalar scale = ws_pow_tbl[n + 1] * wt_pow_j;

                    for ( int m = -n; m <= n; m++ )
                    {
                        const int npj = n + j;
                        const int mmk = m - k;
                        const int abs_mmk = ( mmk < 0 ) ? -mmk : mmk;

                        if ( abs_mmk > npj )
                            continue;

                        const Scalar A_nm = A_table( a_index( n, m ) );
                        const Scalar A_npj_mmk = A_table( a_index( npj, mmk ) );
                        if ( A_npj_mmk == 0.0 )
                            continue;

                        const complex_type ip =
                            ip_tbl[k * ip_stride + ( m + P )];
                        const complex_type Y =
                            Y_tbl[npj * npj + npj + mmk];

                        const Scalar sign_n = ( n % 2 == 0 ) ? 1.0 : -1.0;
                        const Scalar coef_scalar =
                            sign_n * A_nm * A_jk / A_npj_mmk * scale;
                        const complex_type pre_factor = ip * coef_scalar * Y;

                        for ( int c = 0; c < NComps; c++ )
                        {
                            const complex_type M_src_val =
                                get_coeff_3d( M_full, source_cell, n, m, c );
                            accum[c] += M_src_val * pre_factor;
                        }
                    }
                }

                // Atomic because the bin-major M2L driver may launch
                // multiple teams that share a target cell in the fallback
                // path (out-of-bin pairs). The non-atomic version raced
                // on CUDA when fallback pairs colliding on a target ran
                // concurrently — manifest only at nproc > 1 because the
                // multi-rank tree partition produces colliding fallback
                // pairs more often than single-rank.
                for ( int c = 0; c < NComps; c++ )
                    Kokkos::atomic_add( &L_target_out( out_idx, c ),
                                        accum[c] );
            } );
    }

    // =======================================================================
    // m2l_num_src_coeffs: flat (n,m) source-coefficient slot count for the
    // precomputed-operator path. Indexed by src_idx = n*n + n + m for
    // n = 0..P, m = -n..n. Some slots (|m| > n) are unused / left zero.
    // =======================================================================
    static constexpr int m2l_num_src_coeffs = ( P + 1 ) * ( P + 1 );

    // =======================================================================
    // The M2L key contract: m2l_key_dd_max, key_needs_level,
    // canonicalize_key.
    //
    // The sweep builds one integer key per (target, source) pair,
    //
    //   key = (max_d, dd, ii, jj, kk)
    //
    // with max_d the deeper of the two depths, dd = d_source - d_target, and
    // (ii, jj, kk) the center offset measured in half-widths at max_d (see
    // the key comment in src/Canopy_DownwardSweep.hpp). Pairs sharing a key
    // share one operator column. The three members below are how a basis says
    // WHICH of those five integers its operator actually depends on, and how
    // far dd may range before a pair is refused.
    // =======================================================================

    // The |dd| range guard. Pairs with |dd| > m2l_key_dd_max are refused by
    // the sweep's classify pass and routed to the per-pair m2l_translate
    // fallback.
    //
    // FP32 safety valve, and the reason this is a basis trait rather than a
    // sweep constant: with the scale-normalized T̃ this basis builds, the
    // |dd|-dependent residual factor reaches 2^{j·|dd|} (worst j = P). For
    // Scalar = double that is comfortable through |dd| = 6; for
    // Scalar = float a hard cut at |dd| = 4 keeps the precision loss to
    // ~8 bits, matching Greengard truncation error at P = 6. Both numbers
    // are consequences of THIS basis's width normalization, and a basis
    // carrying physical (un-normalized) operators inherits neither.
    static constexpr int m2l_key_dd_max =
        std::is_same<Scalar, float>::value ? 4 : 6;

    // Does this basis's operator depend on max_d, i.e. on the absolute level
    // the pair sits at? For the solid-harmonic basis, NO: the five width
    // normalizations make the operator a function of (dd, ii, jj, kk) alone,
    // which is exactly what canonicalize_key below encodes by zeroing max_d.
    //
    // NOTHING IN src/ CONSUMES THIS YET. `grep -rn key_needs_level src/`
    // finds no reader: T8 is the task that uses it for the operator table's
    // byte accounting (a level-carrying basis realizes more keys, so its
    // budget must account for occupied depth). It is declared now so that the
    // fact canonicalize_key encodes is also stated where a reader looks for
    // it, and so that a conformance test can assert the two agree. Do not go
    // looking for the consumer.
    static constexpr bool key_needs_level = false;

    // Reduce a key to the form this basis's operator actually depends on.
    // Called once, at key construction in the classify pass, BEFORE the key
    // is hashed — so every downstream structure (the per-thread key maps, the
    // global key_to_op, the realized key list, the operator table's column
    // order) sees only canonical keys.
    //
    // This basis ZEROES max_d. Its operators are scale-normalized and
    // therefore depth-independent given (dd, ii, jj, kk), so collapsing every
    // level onto one key is what keeps the table at the realized-offset count
    // rather than multiplying it by occupied tree depth. A basis with
    // physical operators returns the key unchanged instead and must set
    // key_needs_level = true to match.
    //
    // A FUNCTION TEMPLATE ON THE KEY TYPE, deliberately. The key struct is a
    // nested type of DownwardSweep<..., KernelType>, so a basis cannot name
    // it without a circular dependency; the sweep passes its own M2LKey and
    // deduces Key. Host-only, not KOKKOS_INLINE_FUNCTION: the classify pass
    // that calls it runs on host.
    template <class Key>
    static Key canonicalize_key( Key k )
    {
        k.max_d = 0;
        return k;
    }

    // =======================================================================
    // m2l_build_operator
    //
    // Build the per-pair M2L operator entries for a single (dx, dy, dz)
    // offset into a 2D table T_out(out_idx, src_idx). At runtime the M2L
    // contraction is then
    //
    //   L_{out_idx} += sum_{n,m} T_out(out_idx, n*n+n+m) * M_{n,m}(source)
    //
    // i.e. all the per-pair scalar work (Ynm, A factors, i_power, sign,
    // rho^-(n+j+1)) is absorbed into T_out and reused across every source-
    // target pair that shares this offset.
    // =======================================================================
    // Scale-normalized M2L operator builder.
    //
    // Key (dd, ii, jj, kk):
    //   dd            = d_source - d_target  ∈ [-DD_MAX, DD_MAX]
    //   (ii, jj, kk)  = round((c_source - c_target) / w_unit), with
    //                   w_unit = half-width at the deeper of the two depths
    //
    // The operator is depth-independent given (dd, ii, jj, kk); a per-pair
    // F(dd, n, j) factor absorbs the residual (w_s/w_t)^{n+1} or
    // (w_t/w_s)^j scaling that would otherwise live in the multipoles:
    //   dd ≥ 0  →  F = 2^{ j · dd}           (per output row j)
    //   dd <  0 →  F = 2^{-(n+1) · dd}       (per source column n)
    // F(dd=0, ·, ·) = 1, so same-depth operators have no extra scaling.
    template <class AuxType, class TView>
    KOKKOS_INLINE_FUNCTION static void
    m2l_build_operator( int dd, int ix, int iy, int iz, const AuxType& aux,
                        const TView& T_out )
    {
        // The A_{n,m} table this basis puts in its aux tables. Named
        // locally so the translation arithmetic below reads as the
        // theorem it implements.
        const auto& A_table = aux.A_table;

        // (ix, iy, iz) is the integer offset in deeper-cell half-widths.
        // T̃ depends only on this dimensionless geometry plus dd.
        const Scalar dx = static_cast<Scalar>( ix );
        const Scalar dy = static_cast<Scalar>( iy );
        const Scalar dz = static_cast<Scalar>( iz );
        Scalar rho, theta, phi;
        cartesian_to_spherical( dx, dy, dz, rho, theta, phi );

        const Scalar inv_rho = ( rho > 0.0 ) ? ( 1.0 / rho ) : 0.0;

        constexpr int max_rho_pow = 2 * P + 2;
        Scalar inv_rho_pow_tbl[max_rho_pow];
        inv_rho_pow_tbl[0] = 1.0;
        for ( int e = 1; e < max_rho_pow; e++ )
            inv_rho_pow_tbl[e] = inv_rho_pow_tbl[e - 1] * inv_rho;

        // F factor tables. Only one of these is populated; the other
        // stays all-ones. dd in [-6, 6] so the exponents are small
        // (max 7·6 = 42), exact in both double and float.
        Scalar F_row[P + 1];  // F_row[j] for dd ≥ 0
        Scalar F_col[P + 1];  // F_col[n] for dd < 0
        for ( int e = 0; e <= P; e++ )
        {
            F_row[e] = static_cast<Scalar>( 1 );
            F_col[e] = static_cast<Scalar>( 1 );
        }
        if ( dd > 0 )
        {
            // F_row[j] = 2^{j · dd}
            const Scalar step = static_cast<Scalar>( 1 << dd );
            for ( int j = 1; j <= P; j++ )
                F_row[j] = F_row[j - 1] * step;
        }
        else if ( dd < 0 )
        {
            // F_col[n] = 2^{(n+1) · |dd|}
            const Scalar step = static_cast<Scalar>( 1 << ( -dd ) );
            F_col[0] = step; // n=0 ⇒ 2^{|dd|}
            for ( int n = 1; n <= P; n++ )
                F_col[n] = F_col[n - 1] * step;
        }

        constexpr int max_L = 2 * P;
        constexpr int Y_size = ( max_L + 1 ) * ( max_L + 1 );
        complex_type Y_tbl[Y_size];
        for ( int L = 0; L <= max_L; L++ )
            for ( int M = -L; M <= L; M++ )
                Y_tbl[L * L + L + M] = Ynm<Scalar>( L, M, theta, phi );

        constexpr int ip_stride = 2 * P + 1;
        constexpr int ip_size = ( P + 1 ) * ip_stride;
        complex_type ip_tbl[ip_size];
        for ( int kk = 0; kk <= P; kk++ )
            for ( int mm = -P; mm <= P; mm++ )
            {
                const int abs_kk = kk;
                const int abs_mm = ( mm < 0 ) ? -mm : mm;
                const int kmm = kk - mm;
                const int abs_kmm = ( kmm < 0 ) ? -kmm : kmm;
                ip_tbl[kk * ip_stride + ( mm + P )] =
                    i_power( abs_kmm - abs_kk - abs_mm );
            }

        for ( int out_idx = 0; out_idx < num_coeffs_per_cell; out_idx++ )
        {
            for ( int src_idx = 0; src_idx < m2l_num_src_coeffs; src_idx++ )
                T_out( out_idx, src_idx ) = complex_type( 0.0, 0.0 );

            int j, k;
            unflatten_triangular( out_idx, j, k );
            const Scalar A_jk = A_table( a_index( j, k ) );

            // F_row[j] is non-trivial only for dd ≥ 0; F_col[n] only for
            // dd < 0. The other stays 1, so the product is the correct
            // F(dd, n, j) in either case.
            const Scalar F_j = F_row[j];

            for ( int n = 0; n <= P; n++ )
            {
                const Scalar inv_rho_pow = inv_rho_pow_tbl[n + j + 1];
                const Scalar F_nj = F_j * F_col[n];

                for ( int m = -n; m <= n; m++ )
                {
                    const int npj = n + j;
                    const int mmk = m - k;
                    const int abs_mmk = ( mmk < 0 ) ? -mmk : mmk;
                    if ( abs_mmk > npj )
                        continue;

                    const Scalar A_nm = A_table( a_index( n, m ) );
                    const Scalar A_npj_mmk = A_table( a_index( npj, mmk ) );
                    if ( A_npj_mmk == 0.0 )
                        continue;

                    const complex_type ip =
                        ip_tbl[k * ip_stride + ( m + P )];
                    const complex_type Y =
                        Y_tbl[npj * npj + npj + mmk];

                    const Scalar sign_n = ( n % 2 == 0 ) ? 1.0 : -1.0;
                    const Scalar coef_scalar =
                        sign_n * A_nm * A_jk / A_npj_mmk * inv_rho_pow * F_nj;

                    T_out( out_idx, n * n + n + m ) =
                        ip * coef_scalar * Y;
                }
            }
        }
    }

    // =======================================================================
    // M2L as three kernel-owned stages: m2l_pre_cell (per source cell),
    // m2l_core (per pair), m2l_post_cell (per target cell). The sweep owns
    // the traversal, the CSR walk, the team-per-target launch and the
    // scratch allocation; everything about *what* an M2L operator is and
    // how it is applied lives here.
    //
    // Contract with the sweep:
    //   * scratch is m2l_scratch_bytes(num_components) bytes of team
    //     scratch, zero-filled by the sweep once per team before the pair
    //     loop and shared by all three stages. Its layout is this basis's.
    //   * the operator set arrives as m2l_operators_type and is addressed
    //     only by the integer op_idx the CSR carries.
    //   * m2l_post_cell is the only stage that writes to the locals view.
    // =======================================================================

    // =======================================================================
    // m2l_scratch_bytes
    //
    // Per-team M2L scratch this basis needs, in bytes. The sweep allocates
    // exactly this much and passes it to every stage as raw bytes.
    //
    // Layout: two contiguous scalar_type arrays of
    // num_coeffs_per_cell * n_comps entries — the real parts of the target
    // local accumulator, then the imaginary parts. That split is deliberate
    // and load-bearing: each thread touches 8 bytes per accumulator update
    // rather than the 16 a single complex_type scratch would, which halves
    // shared-memory bank conflicts. Collapsing the two arrays into one
    // complex_type array is mathematically identical and *bitwise
    // different*, so it is not a simplification available here.
    //
    // n_comps is the sweep's component count; it must equal num_components,
    // and is a parameter only so the sweep can size scratch without
    // reaching into this basis's template arguments.
    // =======================================================================
    KOKKOS_INLINE_FUNCTION
    static constexpr std::size_t m2l_scratch_bytes( int n_comps )
    {
        return 2 * static_cast<std::size_t>( num_coeffs_per_cell ) *
               static_cast<std::size_t>( n_comps ) * sizeof( scalar_type );
    }

    // =======================================================================
    // m2l_pre_cell
    //
    // Optional per-source-cell pass, run once for each source cell in a
    // target team's CSR slice, immediately before m2l_core for that pair.
    // A compressed shared-basis M2L forms V_l^T M^B here so the pair loop
    // carries only the small r x r core; an FFT-accelerated M2L takes the
    // forward transform here.
    //
    // The solid-harmonic basis contracts the packed multipole directly and
    // has no per-source work, so this is a no-op. It must not write to
    // scratch: the accumulator living there is zeroed once per team and
    // carried across every pair of that team.
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
    // m2l_core
    //
    // The per-pair apply: translate `source_cell`'s multipole through the
    // operator at column `op_idx` of `ops` and accumulate into the team's
    // scratch target-local accumulator.
    //
    // Grown from the former m2l_apply_operator, which took a preselected
    // T_in slice and wrote into a caller-supplied local view. Two things
    // changed, both deliberately:
    //
    //   * The operator set is reached only through op_idx, so the sweep
    //     never indexes it and m2l_operators_type stays opaque there.
    //     ops(out_idx, n*n+n+m, op_idx) is the operator entry.
    //   * The source multipole is read by expanding the m < 0 conjugate
    //     symmetry inline rather than through get_coeff_3d. The two are
    //     arithmetically the same here — get_coeff_3d's out-of-range guards
    //     cannot fire for 0 <= n <= P and |m| <= n, and its m < 0 branch is
    //     this same conjugation — but this is the expression the fused sweep
    //     kernel evaluated before the move, and the solid-harmonic path is
    //     required to come through it bit-identical.
    //
    // Accumulation order is (out_idx) x (component) x (n) x (m) with one
    // complex accumulator per (out_idx, component), which is likewise the
    // order the fused sweep kernel used.
    // =======================================================================
    template <class TeamMember, class MView, class OpsType, class ScratchView>
    KOKKOS_INLINE_FUNCTION static void
    m2l_core( const TeamMember& team_member, const MView& M_full,
              int source_cell, const OpsType& ops, int op_idx,
              const ScratchView& scratch )
    {
        constexpr int n_acc = num_coeffs_per_cell * NComps;
        using acc_type =
            m2l_accumulator_type<typename ScratchView::memory_space>;
        scalar_type* acc_base =
            reinterpret_cast<scalar_type*>( scratch.data() );
        acc_type team_acc_re( acc_base, n_acc );
        acc_type team_acc_im( acc_base + n_acc, n_acc );

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team_member, num_coeffs_per_cell ),
            [&]( const int out_idx )
            {
                for ( int c = 0; c < NComps; c++ )
                {
                    complex_type acc( 0, 0 );
                    for ( int n = 0; n <= P; n++ )
                    {
                        for ( int m = -n; m <= n; m++ )
                        {
                            const int j = n * n + n + m;
                            const int abs_m = ( m < 0 ) ? -m : m;
                            const int storage_idx = n * ( n + 1 ) / 2 + abs_m;
                            const complex_type stored =
                                M_full( source_cell, storage_idx, c );
                            const complex_type m_val =
                                ( m >= 0 ) ? stored
                                           : complex_type( stored.real(),
                                                           -stored.imag() );
                            acc += ops( out_idx, j, op_idx ) * m_val;
                        }
                    }
                    const int slot = out_idx * NComps + c;
                    team_acc_re( slot ) += acc.real();
                    team_acc_im( slot ) += acc.imag();
                }
            } );
    }

    // =======================================================================
    // m2l_post_cell
    //
    // Optional per-target-cell pass, run once after the team's whole source
    // slice has been applied: it flushes the scratch accumulator into
    // L_out(target_cell, :, :). A compressed shared-basis M2L applies U_l
    // here; an FFT-accelerated M2L takes the inverse transform here.
    //
    // += rather than = so the write composes with state already in L_out:
    // on shared targets, L2L from shallower depths has written there before
    // the per-depth M2L runs. Each target is owned by exactly one team, so
    // no atomics are needed.
    // =======================================================================
    template <class TeamMember, class ScratchView, class LView, class OpsType>
    KOKKOS_INLINE_FUNCTION static void
    m2l_post_cell( const TeamMember& team_member, const ScratchView& scratch,
                   const LView& L_out, int target_cell, const OpsType& ops )
    {
        (void)ops;

        constexpr int n_acc = num_coeffs_per_cell * NComps;
        using acc_type =
            m2l_accumulator_type<typename ScratchView::memory_space>;
        scalar_type* acc_base =
            reinterpret_cast<scalar_type*>( scratch.data() );
        acc_type team_acc_re( acc_base, n_acc );
        acc_type team_acc_im( acc_base + n_acc, n_acc );

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team_member, num_coeffs_per_cell ),
            [&]( const int out_idx )
            {
                for ( int c = 0; c < NComps; c++ )
                {
                    const int slot = out_idx * NComps + c;
                    L_out( target_cell, out_idx, c ) += complex_type(
                        team_acc_re( slot ), team_acc_im( slot ) );
                }
            } );
    }

    // =======================================================================
    // L2L: translate parent's local expansion to child's local.
    // Greengard Theorem 5.26.
    //
    //   L_{j,k}^child += sum_{n=j}^{P} sum_{m=-n}^{n}
    //       L_{n,m}^parent * i^(|m|-|m-k|-|k|)
    //       * A_{n-j, m-k} * A_{j,k} / A_{n,m}
    //       * rho^(n-j) * Y_{n-j, m-k}(alpha, beta)
    //
    // where (rho, alpha, beta) = child_center - parent_center in spherical.
    // =======================================================================
    // Scale-normalized L2L: consumes L̄^p = L^p · w_p^n and produces
    // L̄^c = L^c · w_c^j. The rho^{n-j} factor becomes (rho/w_p)^{n-j}
    // and a per-output-row constant (w_c/w_p)^j is applied at the end.
    template <class TeamMember, class LView, class AuxType, class LChildType>
    KOKKOS_INLINE_FUNCTION static void
    l2l_translate( const TeamMember& team_member, const LView& L_full,
                   int parent_cell, Scalar dx, Scalar dy, Scalar dz,
                   Scalar w_child, Scalar w_parent, const AuxType& aux,
                   const LChildType& L_child_out )
    {
        // The A_{n,m} table this basis puts in its aux tables. Named
        // locally so the translation arithmetic below reads as the
        // theorem it implements.
        const auto& A_table = aux.A_table;

        Scalar rho, theta, phi;
        cartesian_to_spherical( dx, dy, dz, rho, theta, phi );

        const Scalar inv_w_p = static_cast<Scalar>( 1 ) / w_parent;
        const Scalar rho_norm = rho * inv_w_p;       // rho / w_p, O(1)
        const Scalar w_ratio = w_child * inv_w_p;    // w_c / w_p, O(1)

        // Precompute (w_c/w_p)^j for j = 0..P. Shared by every output
        // row in this team and across every NComps inner accumulation.
        Scalar w_factor_tbl[P + 1];
        w_factor_tbl[0] = static_cast<Scalar>( 1 );
        for ( int e = 1; e <= P; e++ )
            w_factor_tbl[e] = w_factor_tbl[e - 1] * w_ratio;

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team_member, num_coeffs_per_cell ),
            [&]( const int out_idx )
            {
                int j, k;
                unflatten_triangular( out_idx, j, k );

                const Scalar A_jk = A_table( a_index( j, k ) );
                if ( A_jk == 0.0 )
                    return;

                complex_type accum[NComps];
                for ( int c = 0; c < NComps; c++ )
                    accum[c] = complex_type( 0.0, 0.0 );

                for ( int n = j; n <= P; n++ )
                {
                    Scalar rho_pow_nmj = 1.0;
                    for ( int e = 0; e < n - j; e++ )
                        rho_pow_nmj *= rho_norm;

                    for ( int m = -n; m <= n; m++ )
                    {
                        const int nmj = n - j;
                        const int mmk = m - k;
                        const int abs_mmk = ( mmk < 0 ) ? -mmk : mmk;

                        if ( abs_mmk > nmj )
                            continue;

                        const Scalar A_nm = A_table( a_index( n, m ) );
                        if ( A_nm == 0.0 )
                            continue;

                        const Scalar A_nmj_mmk = A_table( a_index( nmj, mmk ) );

                        const int abs_k = k;
                        const int abs_m = ( m < 0 ) ? -m : m;
                        const int km = k - m;
                        const int abs_km = ( km < 0 ) ? -km : km;
                        const complex_type ip =
                            i_power( abs_m - abs_km - abs_k );

                        const complex_type Y =
                            Ynm<Scalar>( nmj, mmk, theta, phi );

                        const Scalar coef_scalar =
                            A_nmj_mmk * A_jk / A_nm * rho_pow_nmj;
                        const complex_type pre_factor = ip * coef_scalar * Y;

                        for ( int c = 0; c < NComps; c++ )
                        {
                            const complex_type L_par_val =
                                get_coeff_3d( L_full, parent_cell, n, m, c );
                            accum[c] += L_par_val * pre_factor;
                        }
                    }
                }

                // Apply per-row scale (w_c / w_p)^j (= 1 at j=0).
                const Scalar w_factor = w_factor_tbl[j];

                for ( int c = 0; c < NComps; c++ )
                    L_child_out( out_idx, c ) += accum[c] * w_factor;
            } );
    }

    // =======================================================================
    // L2P: evaluate local expansion at a particle position.
    //
    // Potential (using m >= 0 symmetry; m=0 direct, m>0 doubled real part):
    //   phi_c(r) = sum_{n=0}^{P} Re{ L_{n,0,c} rho^n Y_{n,0} }
    //              + 2 sum_{m=1}^{n} Re{ L_{n,m,c} rho^n Y_{n,m} }
    //
    // Gradient: currently via central finite differences (correctness-first;
    //           analytical derivatives can be substituted later).
    //
    // Parameters:
    //   L_full         - 3D local view
    //   leaf_cell      - cell index of this particle's leaf
    //   dx, dy, dz     - particle_position - leaf_center
    //   phi_out        - Scalar[NComps] output potentials
    //   grad_out       - 2D accessor: grad_out(c, d) for component c, dim d.
    //                    Must be valid if compute_gradient is true.
    //   compute_gradient - true to populate grad_out; false to skip.
    // =======================================================================
    // Scale-normalized L2P: consumes L̄_{n,m} = L_{n,m} · w_self^{n} and
    // evaluates phi += L̄_{n,m} · (rho_p / w_self)^n · Y_{n,m}. Fused
    // form so we never materialize the physical (rho_p)^n separately —
    // (rho_p / w_self) is O(1) for any particle inside its leaf, which
    // is the FP32-safe form.
    template <class LView, class GradAccess>
    KOKKOS_INLINE_FUNCTION static void
    l2p_evaluate( const LView& L_full, int leaf_cell, Scalar dx, Scalar dy,
                  Scalar dz, Scalar w_self, Scalar ( &phi_out )[NComps],
                  const GradAccess& grad_out, bool compute_gradient )
    {
        const Scalar inv_w = static_cast<Scalar>( 1 ) / w_self;

        // Inline evaluator for potential at an arbitrary offset
        auto eval_phi =
            [&]( Scalar ex, Scalar ey, Scalar ez, Scalar( &phi )[NComps] )
        {
            for ( int c = 0; c < NComps; c++ )
                phi[c] = 0.0;

            Scalar rho, theta, phi_ang;
            cartesian_to_spherical( ex, ey, ez, rho, theta, phi_ang );

            const Scalar rho_norm = rho * inv_w;
            Scalar rho_pow_n = 1.0;
            for ( int n = 0; n <= P; n++ )
            {
                // m = 0: count once
                {
                    const complex_type Y0 = Ynm<Scalar>( n, 0, theta, phi_ang );
                    for ( int c = 0; c < NComps; c++ )
                    {
                        const complex_type L_n0 =
                            get_coeff_3d( L_full, leaf_cell, n, 0, c );
                        const complex_type term = L_n0 * rho_pow_n * Y0;
                        phi[c] += term.real();
                    }
                }
                // m = 1..n: count twice via symmetry
                for ( int m = 1; m <= n; m++ )
                {
                    const complex_type Y = Ynm<Scalar>( n, m, theta, phi_ang );
                    for ( int c = 0; c < NComps; c++ )
                    {
                        const complex_type L_nm =
                            get_coeff_3d( L_full, leaf_cell, n, m, c );
                        const complex_type term = L_nm * rho_pow_n * Y;
                        phi[c] += 2.0 * term.real();
                    }
                }
                rho_pow_n *= rho_norm;
            }
        };

        eval_phi( dx, dy, dz, phi_out );

        if ( compute_gradient )
        {
            // Central finite difference. The step MUST scale with the cell
            // size: phi varies on the scale of w_self, so its third derivative
            // ~ phi/w_self^3 and the FD truncation error ~ h^2/w_self^3 blows up
            // for deep (tiny) cells if h is fixed. Use h ~ eps^(1/3) * w_self
            // (the roundoff/truncation optimum) so the relative error is
            // depth-independent. (Was a fixed 1e-5 — the premature full-rollup
            // NaN root cause: at w_self~3e-3 the fixed step gave O(1e5) spurious
            // gradients. TODO: replace with analytical derivatives.)
            const Scalar h = static_cast<Scalar>( 1.0e-5 ) * w_self;
            Scalar phi_px[NComps], phi_mx[NComps];
            Scalar phi_py[NComps], phi_my[NComps];
            Scalar phi_pz[NComps], phi_mz[NComps];

            eval_phi( dx + h, dy, dz, phi_px );
            eval_phi( dx - h, dy, dz, phi_mx );
            eval_phi( dx, dy + h, dz, phi_py );
            eval_phi( dx, dy - h, dz, phi_my );
            eval_phi( dx, dy, dz + h, phi_pz );
            eval_phi( dx, dy, dz - h, phi_mz );

            const Scalar inv_2h = 1.0 / ( 2.0 * h );
            for ( int c = 0; c < NComps; c++ )
            {
                grad_out( c, 0 ) = ( phi_px[c] - phi_mx[c] ) * inv_2h;
                grad_out( c, 1 ) = ( phi_py[c] - phi_my[c] ) * inv_2h;
                grad_out( c, 2 ) = ( phi_pz[c] - phi_mz[c] ) * inv_2h;
            }
        }
    }
};

} // namespace Canopy

#endif // CANOPY_LAPLACE_KERNEL_HPP
