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

#include <cstdint>

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
    using complex_type = Kokkos::complex<Scalar>;

    static constexpr int max_order = P;
    static constexpr int num_coeffs_per_cell = ( P + 1 ) * ( P + 2 ) / 2;
    static constexpr int num_components = NComps;
    static constexpr bool has_mplus_symmetry = true;

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
    template <class TeamMember, class MView, class AType, class MParentType>
    KOKKOS_INLINE_FUNCTION static void
    m2m_translate( const TeamMember& team_member, const MView& M_full,
                   int child_cell, Scalar dx, Scalar dy, Scalar dz,
                   Scalar w_child, Scalar w_parent,
                   const AType& A_table, const MParentType& M_parent_out )
    {
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
    template <class TeamMember, class MView, class AType, class LTargetType>
    KOKKOS_INLINE_FUNCTION static void
    m2l_translate( const TeamMember& team_member, const MView& M_full,
                   int source_cell, Scalar dx, Scalar dy, Scalar dz,
                   Scalar w_source, Scalar w_target,
                   const AType& A_table, const LTargetType& L_target_out )
    {
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
    template <class AType, class TView>
    KOKKOS_INLINE_FUNCTION static void
    m2l_build_operator( int dd, int ix, int iy, int iz, const AType& A_table,
                        const TView& T_out )
    {
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
    // m2l_apply_operator
    //
    // Apply a precomputed M2L operator T_in (for one source-target offset)
    // to translate `source_cell`'s multipole into `L_target_out`. T_in
    // is indexed as T_in(out_idx, n*n+n+m).
    // =======================================================================
    template <class TeamMember, class MView, class TView, class LTargetType>
    KOKKOS_INLINE_FUNCTION static void
    m2l_apply_operator( const TeamMember& team_member, const MView& M_full,
                        int source_cell, const TView& T_in,
                        const LTargetType& L_target_out )
    {
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team_member, num_coeffs_per_cell ),
            [&]( const int out_idx )
            {
                complex_type accum[NComps];
                for ( int c = 0; c < NComps; c++ )
                    accum[c] = complex_type( 0.0, 0.0 );

                for ( int n = 0; n <= P; n++ )
                {
                    for ( int m = -n; m <= n; m++ )
                    {
                        const int src_idx = n * n + n + m;
                        const complex_type T_val =
                            T_in( out_idx, src_idx );
                        for ( int c = 0; c < NComps; c++ )
                        {
                            const complex_type M_val = get_coeff_3d(
                                M_full, source_cell, n, m, c );
                            accum[c] += T_val * M_val;
                        }
                    }
                }

                for ( int c = 0; c < NComps; c++ )
                    L_target_out( out_idx, c ) += accum[c];
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
    template <class TeamMember, class LView, class AType, class LChildType>
    KOKKOS_INLINE_FUNCTION static void
    l2l_translate( const TeamMember& team_member, const LView& L_full,
                   int parent_cell, Scalar dx, Scalar dy, Scalar dz,
                   Scalar w_child, Scalar w_parent,
                   const AType& A_table, const LChildType& L_child_out )
    {
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
            // Central finite difference
            const Scalar h = 1.0e-5;
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
