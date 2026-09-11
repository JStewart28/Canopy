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

#ifndef CANOPY_TEST_MONOPOLE_BASIS_HPP
#define CANOPY_TEST_MONOPOLE_BASIS_HPP

#include "Canopy_FarFieldContract.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <type_traits>

namespace CanopyTest
{

// ============================================================================
// MonopoleBasis — a non-harmonic conformance basis for UpwardSweep /
// DownwardSweep.
//
// This is a FIXTURE, not a method. It carries one real coefficient per cell
// per component — the cell's total charge on the way up, and a dimensionless
// accumulated "potential" on the way down — so every operator in the far-field
// contract is a sum or a copy and is therefore *exactly* reproducible by a
// host recomputation. Its purpose is to prove that the trait-and-operator
// contract the two sweeps describe is a real interface rather than a rename of
// the solid-harmonic one: nothing here is a spherical harmonic, there is no
// A_{n,m} table, the coefficient is real rather than Kokkos::complex, and the
// operator table has one entry per key instead of (28, 49).
//
// It lives in tests/ deliberately. Its truncation error is O(1) at every
// separation — it is not an FMM anyone should solve with, and shipping it in
// src/ would invite exactly that.
//
// ---------------------------------------------------------------------------
// Units and conventions (required by the design document's Conventions row on
// units; none of this is recoverable from the code).
//
//   * Every width parameter this basis is handed (w_self, w_child, w_parent,
//     w_source, w_target) is a HALF-WIDTH: half the side length of the cell's
//     cube, matching Canopy::CellInfo::half_width.
//
//   * Every offset this basis is handed (dx, dy, dz) is
//     (other center - own center) in Cartesian coordinates, i.e. the
//     source-minus-target sense for M2L and the child-minus-parent sense for
//     M2M and L2L, matching LaplaceKernel.
//
//   * MULTIPOLE. M(cell, 0, c) is the total component-c charge inside the
//     cell, divided by nothing. It is dimensionless in the sense that no
//     width enters it: P2M ignores w_self and M2M ignores both widths. The
//     solid-harmonic basis divides by w_self^{n+1}; this one does not, because
//     a single monopole coefficient has no n-dependent conditioning problem to
//     normalize away.
//
//   * LOCAL. L(cell, 0, c) is the scale-normalized monopole potential, and is
//     likewise dimensionless: it is the sum of source charges each divided by
//     the separation *measured in deeper-cell half-widths*, not by a physical
//     length. It carries no 1/length factor, so it is not a physical
//     potential and must not be compared against one.
//
//   * L2P returns L unchanged as the potential and zero as the gradient. The
//     particle's offset inside its leaf is ignored: a monopole local is
//     constant over the cell.
//
// The reason the local is dimensionless is the M2L operator builder's
// signature. m2l_build_operator receives integers only — no width and no
// kernel parameters — so the only length scale available to it is the implied
// unit of its integer offset. Supplying w_unit and kernel_params to the
// builder is T9's; a physical local is not reachable here, and this basis's
// gate is locals() against a host recomputation of the same expression, not
// accuracy against a physical potential.
// ============================================================================

template <class Scalar, int Order, int NComps = 1>
struct MonopoleBasis
{
    // The design document's Conventions row: new bases are double only. A
    // monopole basis has no scale invariance to exploit and no FP32
    // conditioning argument to inherit, and the exactness this fixture is
    // built for is a double-precision claim.
    static_assert( std::is_same<Scalar, double>::value,
                   "MonopoleBasis: Scalar must be double" );

    using scalar_type = Scalar;

    // -----------------------------------------------------------------------
    // The coefficient contract (see LaplaceKernel for the full statement).
    // One real coefficient, so scalars_per_coeff is 1 and coeff_type IS the
    // component scalar. This is the case Canopy::detail::coeff_traits'
    // *primary* template covers, so coalesced_view_exchange and both
    // shared-cell Allreduces work for this basis with no specialization.
    //
    // The identity element is value-initialization, coeff_type() == +0.0.
    // -----------------------------------------------------------------------
    using coeff_type = Scalar;
    using component_scalar_type = Scalar;
    static constexpr int scalars_per_coeff = 1;

    static_assert( sizeof( coeff_type ) ==
                       scalars_per_coeff * sizeof( component_scalar_type ),
                   "MonopoleBasis: coeff_type is not scalars_per_coeff "
                   "contiguous component_scalar_type, so the MPI packing in "
                   "coalesced_view_exchange and in the two shared-cell "
                   "reductions would transfer the wrong byte count" );

    static constexpr int max_order = Order;
    static constexpr int num_coeffs_per_cell = 1;
    static constexpr int num_components = NComps;

    // Flat source-coefficient slot count for the precomputed-operator path.
    // One monopole in, so the operator is 1x1.
    static constexpr int m2l_num_src_coeffs = 1;

    // -----------------------------------------------------------------------
    // The M2L key contract: m2l_key_dd_max, key_needs_level,
    // canonicalize_key. See LaplaceKernel for the full statement of what the
    // five key integers are.
    // -----------------------------------------------------------------------

    // The |dd| range guard. 6 is the value the sweep's old hardcoded
    // non-float branch gave every basis, and it is repeated here deliberately
    // so that turning the constant into a trait moved no pair onto the
    // fallback path for this basis: total_fallback_pair_count() was 0 before
    // and must stay 0. F(dd) = 2^{max(0,-dd)} below is exact for any |dd|
    // this admits, so nothing about this basis argues for a tighter cut.
    static constexpr int m2l_key_dd_max = 6;

    // THIS BASIS NEEDS THE LEVEL, and that is the whole point of it being
    // true here: with LaplaceKernel at false and MonopoleBasis at true, both
    // branches of the key contract are exercised by the test suite rather
    // than one of them being a declaration nobody runs.
    //
    // NOTHING IN src/ CONSUMES THIS YET — T8's byte accounting is the first
    // reader; see the note on LaplaceKernel::key_needs_level. It is declared
    // now so the trait list a basis author sees is complete, and so
    // tests/tstFarFieldContract.hpp can assert it agrees with
    // canonicalize_key. Do not go looking for the consumer.
    //
    // Keeping max_d makes a per-level key, which for THIS basis produces
    // duplicate operator columns by construction: m2l_operator_entry ignores
    // max_d, so two keys differing only in it build identical 1x1 operators.
    // That is deliberate and is what the strictly-more-keys assertion in
    // tests/tstFarFieldContract.hpp measures — the level reaching the key is
    // observable in m2l_n_unique_ops() without changing a single operator
    // value, so the conformance gate stays bit-exact while proving the level
    // is not silently dropped.
    static constexpr bool key_needs_level = true;

    // Identity: the key is returned unchanged, max_d and all. The mirror
    // image of LaplaceKernel::canonicalize_key, which zeroes max_d.
    //
    // A function template on the key type for the reason given there: the key
    // struct is a nested type of DownwardSweep<..., KernelType> and a basis
    // cannot name it without a circular dependency. Host-only, not
    // KOKKOS_INLINE_FUNCTION — the classify pass runs on host.
    template <class Key>
    static Key canonicalize_key( Key k )
    {
        return k;
    }

    // -----------------------------------------------------------------------
    // The operator-table budget contract: bytes_per_key, m2l_overflow_policy.
    // See LaplaceKernel for the full statement of what the sweep does with
    // these two.
    // -----------------------------------------------------------------------

    // Bytes one operator column costs. DERIVED from sizeof(coeff_type), never
    // a literal, for the reason given on LaplaceKernel::bytes_per_key: the
    // operator table's element type follows coeff_type, and it is not the same
    // width for every basis. Here coeff_type is a bare double and the operator
    // is 1x1, so this is 8 — the cheapest key in the repository, which is why
    // an overflow policy is cheap to exercise against this basis: the count
    // cap binds long before any plausible byte budget does.
    static constexpr std::size_t bytes_per_key =
        static_cast<std::size_t>( num_coeffs_per_cell ) *
        static_cast<std::size_t>( m2l_num_src_coeffs ) * sizeof( coeff_type );

    // This basis has a per-pair m2l_translate that reconstructs the same
    // integer key from the physical geometry and calls the same
    // m2l_operator_entry the table build calls, so an overflowing pair takes
    // the per-pair path and lands on the same operator value. That is what
    // makes total_fallback_pair_count() a printed diagnostic in
    // tests/tstFarFieldContract.hpp rather than something the host reference
    // has to know about.
    //
    // The other enumerator, M2LOverflow::EscalateToP2P, does not compile: the
    // sweep's class-scope static_assert rejects it, and the permanent
    // #ifdef CANOPY_TEST_EXPECT_COMPILE_FAILURE block in
    // tests/tstFarFieldContract.hpp carries a basis that proves the assert is
    // still there.
    static constexpr Canopy::M2LOverflow m2l_overflow_policy =
        Canopy::M2LOverflow::PerPairTranslate;

    // NOTHING CONSUMES THIS YET. `grep -rn sets_per_component src/ tests/`
    // finds no reader: T10 is the task that raises the locals view to
    // multiple sets per component and teaches the sweeps to read this trait.
    // It is declared now so the trait list here is the complete one a basis
    // author sees, and so T10's diff is a change of value rather than an
    // addition. Do not go looking for the consumer.
    static constexpr int sets_per_component = 1;

    // -----------------------------------------------------------------------
    // The M2L operator set. Same shape and layout contract as the
    // solid-harmonic one — (num_coeffs_per_cell, m2l_num_src_coeffs,
    // n_unique_ops), LayoutLeft, element type coeff_type — which here is a
    // (1, 1, n_unique_ops) table of real numbers. The sweep addresses it only
    // by the integer op_idx its CSR carries and never indexes inside it.
    //
    // Parameterized on the memory space because the basis is not. Spell it
    //     typename KernelType::template m2l_operators_type<memory_space>
    // -----------------------------------------------------------------------
    template <class MemorySpace>
    using m2l_operators_type =
        Kokkos::View<coeff_type***, Kokkos::LayoutLeft, MemorySpace>;

    // -----------------------------------------------------------------------
    // Auxiliary tables: none. This basis has no precomputed order-dependent
    // data — no normalization table, no Ynm cache — so the struct is empty
    // and build_aux_tables ignores its argument.
    //
    // It is a struct TEMPLATE on the memory space, not a plain empty struct,
    // because that is how both sweeps spell it:
    //     typename KernelType::template aux_tables_type<memory_space>
    // (src/Canopy_UpwardSweep.hpp:109, src/Canopy_DownwardSweep.hpp:160) and
    //     KernelType::template build_aux_tables<memory_space>( P )
    // (src/Canopy_UpwardSweep.hpp:266, src/Canopy_DownwardSweep.hpp:1177).
    // Neither sweep names a member, which is exactly why an empty struct
    // satisfies them.
    // -----------------------------------------------------------------------
    template <class MemorySpace>
    struct aux_tables_type
    {
    };

    // Host function, not device-callable, matching the Conventions row on
    // host-side construction — even though this one allocates nothing.
    template <class MemorySpace>
    static aux_tables_type<MemorySpace> build_aux_tables( int order )
    {
        (void)order;
        return {};
    }

    // The M2L team scratch, viewed as a scalar array. One accumulator per
    // (coefficient, component); see m2l_scratch_bytes.
    template <class ScratchSpace>
    using m2l_accumulator_type =
        Kokkos::View<scalar_type*, ScratchSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    // =======================================================================
    // m2l_scratch_bytes
    //
    // Per-team M2L scratch this basis needs, in bytes: one contiguous
    // scalar_type array of num_coeffs_per_cell * n_comps entries, holding the
    // target local accumulator. The real/imag split the solid-harmonic basis
    // needs has no analogue here — the coefficient is real.
    //
    // MUST STAY constexpr. The sweep assigns it to a `constexpr size_t`
    // (src/Canopy_DownwardSweep.hpp:1562) and every extent inside the three
    // stages is derived from it; making it a runtime value is R3 (a trait
    // indirection deoptimizing the fused kernel) with no correctness signal
    // to catch it. n_comps is a parameter only so the sweep can size scratch
    // without reaching into this basis's template arguments; it is only ever
    // passed num_components.
    //
    // The sweep hands the stages raw, zero-filled bytes and relies on
    // all-zero bytes being this basis's accumulator identity. That holds
    // here: the accumulator is IEEE-754 binary64, whose all-zero-bytes
    // representation is +0.0, and +0.0 is the additive identity.
    // =======================================================================
    KOKKOS_INLINE_FUNCTION
    static constexpr std::size_t m2l_scratch_bytes( int n_comps )
    {
        return static_cast<std::size_t>( num_coeffs_per_cell ) *
               static_cast<std::size_t>( n_comps ) * sizeof( scalar_type );
    }

    // Round half away from zero, matching the std::lround the sweep's
    // pre-integer M2L key path used. Only m2l_translate needs it: the fused
    // path is handed the integers already.
    KOKKOS_INLINE_FUNCTION
    static int round_to_int( Scalar v )
    {
        const Scalar half = ( v >= static_cast<Scalar>( 0 ) )
                                ? static_cast<Scalar>( 0.5 )
                                : static_cast<Scalar>( -0.5 );
        return static_cast<int>( v + half );
    }

    // =======================================================================
    // m2l_operator_entry — THE one M2L operator value, and the single source
    // of truth for it.
    //
    //   T(dd, ix, iy, iz) = F(dd) / || (ix, iy, iz) ||
    //
    // Key convention, the same one m2l_build_operator is handed
    // (src/Canopy_DownwardSweep.hpp:312-322):
    //   dd           = d_source - d_target, the signed depth difference,
    //                  range-guarded to |dd| <= M2L_KEY_DD_MAX by the sweep.
    //   (ix, iy, iz) = round( (c_source - c_target) / w_unit ), with w_unit
    //                  the HALF-WIDTH at the deeper of the two depths. So the
    //                  offset is measured in deeper-cell half-widths and
    //                  carries no physical length; 1/||(ix,iy,iz)|| is
    //                  dimensionless.
    //
    // HOW THIS BASIS HANDLES dd — the second half of the convention, which
    // the design document requires a basis to state and does not pick.
    //
    //   F(dd) = 2^{ max(0, -dd) }
    //
    // This is LaplaceKernel's own residual factor F(dd, n, j)
    // (src/Canopy_LaplaceKernel.hpp:656-661) evaluated at the monopole term
    // n = j = 0, and it is chosen for that reason rather than invented:
    //   dd >= 0  ->  F = 2^{ j*dd}      = 2^0     = 1
    //   dd <  0  ->  F = 2^{-(n+1)*dd}  = 2^{-dd}
    // and F(0, ., .) = 1, so same-depth operators carry no extra scaling.
    //
    // Read directly rather than by analogy: the physical monopole M2L is
    // L = M / r. With this basis's multipole (a bare charge) and local (a
    // charge divided by a separation in deeper-cell half-widths), the
    // conversion factor between "separation in units of w_unit" and
    // "separation in units of w_source" is w_source / w_unit, which is 1 when
    // the source is the deeper cell (dd >= 0) and 2^{-dd} when the target is
    // (dd < 0). So F(dd) is what makes the operator depth-independent given
    // the key, which is the whole premise of hashing pairs onto keys.
    //
    // Every exponent is a small non-negative integer (|dd| <= 6, so F <= 64)
    // and every F is an exact power of two, so the repeated doubling below is
    // exact in binary64 rather than merely accurate.
    //
    // Both m2l_build_operator and the host reference in
    // tests/tstFarFieldContract.hpp call THIS function. That is what makes
    // the test's EXPECT_DOUBLE_EQ exact rather than merely close, and it is
    // why the expression must not be duplicated anywhere.
    // =======================================================================
    KOKKOS_INLINE_FUNCTION
    static scalar_type m2l_operator_entry( int dd, int ix, int iy, int iz )
    {
        const scalar_type fx = static_cast<scalar_type>( ix );
        const scalar_type fy = static_cast<scalar_type>( iy );
        const scalar_type fz = static_cast<scalar_type>( iz );
        const scalar_type r2 = fx * fx + fy * fy + fz * fz;
        // A zero offset is not reachable through a MAC-admissible pair, but
        // the range guards do not exclude it, so do not divide by zero.
        const scalar_type inv_r =
            ( r2 > static_cast<scalar_type>( 0 ) )
                ? ( static_cast<scalar_type>( 1 ) / Kokkos::sqrt( r2 ) )
                : static_cast<scalar_type>( 0 );

        scalar_type F = static_cast<scalar_type>( 1 );
        for ( int e = 0; e < -dd; e++ )
            F *= static_cast<scalar_type>( 2 );

        return F * inv_r;
    }

    // =======================================================================
    // m2l_accumulate — the single multiply-accumulate step of this basis's
    // M2L, factored out so that m2l_core and the host reference in
    // tests/tstFarFieldContract.hpp cannot drift apart.
    //
    // Written as TWO statements on purpose. `acc += op_entry * m_val` as one
    // statement is contractible to an FMA under the compiler's default
    // -ffp-contract=on, and a build that contracts it in one caller and not
    // the other would differ in the last bit — which the test's
    // EXPECT_DOUBLE_EQ (4 ULP) might absorb today and would not absorb after
    // a few hundred terms. Splitting the product into a named local puts a
    // statement boundary between the multiply and the add, which
    // -ffp-contract=on may not cross.
    // =======================================================================
    KOKKOS_INLINE_FUNCTION
    static void m2l_accumulate( scalar_type& acc, scalar_type op_entry,
                                scalar_type m_val )
    {
        const scalar_type prod = op_entry * m_val;
        acc += prod;
    }

    // =======================================================================
    // P2M: add a particle's charge to its leaf cell's monopole.
    //
    //   M(0, c) += q_c
    //
    // Parameters mirror LaplaceKernel::p2m_contribution positionally
    // (src/Canopy_LaplaceKernel.hpp:368) because the sweep calls them so.
    //   charges     - Scalar[NComps] per-component charges for this particle
    //   dx, dy, dz  - particle_position - cell_center; IGNORED, a monopole
    //                 has no positional dependence inside its cell
    //   w_self      - this leaf's half-width; IGNORED, this basis's multipole
    //                 carries no width normalization
    //   M_out       - 2D slice M_out(coeff_idx, comp_idx)
    //
    // Atomic, like the solid-harmonic P2M: the sweep runs one thread per
    // particle over a RangePolicy and many particles share a leaf.
    // =======================================================================
    template <class MSliceType>
    KOKKOS_INLINE_FUNCTION static void
    p2m_contribution( const Scalar ( &charges )[NComps], Scalar dx, Scalar dy,
                      Scalar dz, Scalar w_self, const MSliceType& M_out )
    {
        (void)dx;
        (void)dy;
        (void)dz;
        (void)w_self;

        for ( int c = 0; c < NComps; c++ )
            Kokkos::atomic_add( &M_out( 0, c ), charges[c] );
    }

    // =======================================================================
    // M2M: add a child's monopole into the parent's.
    //
    //   M^parent(0, c) += M^child(0, c)
    //
    // Charge is conserved under aggregation, so the translation is the
    // identity and both widths and the offset are ignored. Parameters mirror
    // LaplaceKernel::m2m_translate (src/Canopy_LaplaceKernel.hpp:411).
    //
    // Non-atomic, like the solid-harmonic M2M: the sweep runs one team per
    // parent and that team walks its own children sequentially, so no two
    // writers touch one parent.
    // =======================================================================
    template <class TeamMember, class MView, class AuxType, class MParentType>
    KOKKOS_INLINE_FUNCTION static void
    m2m_translate( const TeamMember& team_member, const MView& M_full,
                   int child_cell, Scalar dx, Scalar dy, Scalar dz,
                   Scalar w_child, Scalar w_parent, const AuxType& aux,
                   const MParentType& M_parent_out )
    {
        (void)dx;
        (void)dy;
        (void)dz;
        (void)w_child;
        (void)w_parent;
        (void)aux;

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team_member, num_coeffs_per_cell ),
            [&]( const int out_idx )
            {
                for ( int c = 0; c < NComps; c++ )
                    M_parent_out( out_idx, c ) +=
                        M_full( child_cell, out_idx, c );
            } );
    }

    // =======================================================================
    // M2L, per-pair fallback path. The sweep routes a pair here when its key
    // trips a range guard or the operator-table count cap
    // (src/Canopy_DownwardSweep.hpp:1662), in which case no operator column
    // exists for it.
    //
    //   L^target(0, c) += T(dd, ix, iy, iz) * M^source(0, c)
    //
    // This reconstructs the SAME integer key from the physical geometry it is
    // handed and evaluates the SAME m2l_operator_entry, so the fused and
    // fallback paths are bit-identical for this basis and the host reference
    // in tests/tstFarFieldContract.hpp does not have to know which path a
    // pair took. The reconstruction is exact, not approximate:
    //   * w_unit = min(w_source, w_target) is the half-width at the deeper of
    //     the two depths, which is the sweep's own definition;
    //   * every center difference is an exact integer multiple of that
    //     half-width, so dx/w_unit lands on an integer up to rounding and the
    //     round recovers it;
    //   * w_target / w_source is exactly 2^{dd}, since every half-width is
    //     the root half-width scaled by a power of two, so the doubling loop
    //     below recovers dd exactly.
    //
    // Parameters mirror LaplaceKernel::m2l_translate
    // (src/Canopy_LaplaceKernel.hpp:516). Atomic, like the solid-harmonic
    // fallback: the sweep runs one team per PAIR, so two pairs sharing a
    // target can write concurrently.
    // =======================================================================
    template <class TeamMember, class MView, class AuxType, class LTargetType>
    KOKKOS_INLINE_FUNCTION static void
    m2l_translate( const TeamMember& team_member, const MView& M_full,
                   int source_cell, Scalar dx, Scalar dy, Scalar dz,
                   Scalar w_source, Scalar w_target, const AuxType& aux,
                   const LTargetType& L_target_out )
    {
        (void)aux;

        const Scalar w_unit = ( w_source < w_target ) ? w_source : w_target;
        const Scalar inv_unit = ( w_unit > static_cast<Scalar>( 0 ) )
                                    ? ( static_cast<Scalar>( 1 ) / w_unit )
                                    : static_cast<Scalar>( 0 );
        const int ix = round_to_int( dx * inv_unit );
        const int iy = round_to_int( dy * inv_unit );
        const int iz = round_to_int( dz * inv_unit );

        // dd = d_source - d_target = log2( w_target / w_source ).
        int dd = 0;
        if ( w_source > static_cast<Scalar>( 0 ) )
        {
            Scalar ratio = w_target / w_source;
            while ( ratio > static_cast<Scalar>( 1.5 ) )
            {
                ratio *= static_cast<Scalar>( 0.5 );
                ++dd;
            }
            while ( ratio < static_cast<Scalar>( 0.75 ) )
            {
                ratio *= static_cast<Scalar>( 2 );
                --dd;
            }
        }

        const Scalar T = m2l_operator_entry( dd, ix, iy, iz );

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team_member, num_coeffs_per_cell ),
            [&]( const int out_idx )
            {
                for ( int c = 0; c < NComps; c++ )
                {
                    Scalar contrib = static_cast<Scalar>( 0 );
                    m2l_accumulate( contrib, T,
                                    M_full( source_cell, out_idx, c ) );
                    Kokkos::atomic_add( &L_target_out( out_idx, c ), contrib );
                }
            } );
    }

    // =======================================================================
    // m2l_build_operator — fill the 1x1 operator for one canonicalized key.
    //
    // Parameters mirror LaplaceKernel::m2l_build_operator
    // (src/Canopy_LaplaceKernel.hpp:664) positionally. The whole key/dd
    // convention is on m2l_operator_entry above; this function is only the
    // placement of that one value into the table the sweep allocated
    // WithoutInitializing, so it must write every entry, which at (1, 1) it
    // trivially does.
    //
    // KOKKOS_INLINE_FUNCTION static, mirroring LaplaceKernel, even though the
    // sweep only ever calls it on host over a Kokkos::HostSpace table inside
    // its stage-4 build. (The Conventions row that requires a plain static
    // member is about T9's host-side build_m2l_operators, which allocates and
    // may call LAPACK; this is the per-key fill.)
    // =======================================================================
    template <class AuxType, class TView>
    KOKKOS_INLINE_FUNCTION static void
    m2l_build_operator( int dd, int ix, int iy, int iz, const AuxType& aux,
                        const TView& T_out )
    {
        (void)aux;
        T_out( 0, 0 ) = m2l_operator_entry( dd, ix, iy, iz );
    }

    // =======================================================================
    // m2l_pre_cell — no-op.
    //
    // This basis contracts the source monopole directly and has no
    // per-source-cell work to hoist, so T3's finding that the hook is only
    // half-placed — team scratch is per-target, so there is nowhere to put
    // once-per-source-cell state that outlives the team — does not bite here.
    //
    // It must not write to scratch: the accumulator living there is zeroed
    // once per team and carried across every pair of that team.
    //
    // Parameters mirror LaplaceKernel::m2l_pre_cell
    // (src/Canopy_LaplaceKernel.hpp:841).
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
    // m2l_core — the per-pair apply.
    //
    //   acc(out_idx, c) += ops(out_idx, 0, op_idx) * M(source_cell, out_idx, c)
    //
    // The operator set is reached only through op_idx, so the sweep never
    // indexes it. The accumulation runs in the order the sweep walks the
    // target's CSR slice, which is the push order of
    // comm_plan.m2l_plan().interaction_lists[target_key]; the host reference
    // in tests/tstFarFieldContract.hpp iterates that same vector as-is, and
    // goes through m2l_accumulate, which is why the comparison is exact.
    //
    // Parameters mirror LaplaceKernel::m2l_core
    // (src/Canopy_LaplaceKernel.hpp:880).
    // =======================================================================
    template <class TeamMember, class MView, class OpsType, class ScratchView>
    KOKKOS_INLINE_FUNCTION static void
    m2l_core( const TeamMember& team_member, const MView& M_full,
              int source_cell, const OpsType& ops, int op_idx,
              const ScratchView& scratch )
    {
        using acc_type =
            m2l_accumulator_type<typename ScratchView::memory_space>;
        constexpr int n_acc = num_coeffs_per_cell * NComps;
        scalar_type* acc_base =
            reinterpret_cast<scalar_type*>( scratch.data() );
        acc_type team_acc( acc_base, n_acc );

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team_member, num_coeffs_per_cell ),
            [&]( const int out_idx )
            {
                const scalar_type T = ops( out_idx, 0, op_idx );
                for ( int c = 0; c < NComps; c++ )
                {
                    const int slot = out_idx * NComps + c;
                    m2l_accumulate( team_acc( slot ), T,
                                    M_full( source_cell, out_idx, c ) );
                }
            } );
    }

    // =======================================================================
    // m2l_post_cell — flush the team's accumulator into the locals view.
    //
    // += rather than =, for the same reason as the solid-harmonic basis: on a
    // shared target, L2L from a shallower depth has already written there
    // before the per-depth M2L runs. Each target is owned by exactly one
    // team, so no atomics are needed.
    //
    // Parameters mirror LaplaceKernel::m2l_post_cell
    // (src/Canopy_LaplaceKernel.hpp:937).
    // =======================================================================
    template <class TeamMember, class ScratchView, class LView, class OpsType>
    KOKKOS_INLINE_FUNCTION static void
    m2l_post_cell( const TeamMember& team_member, const ScratchView& scratch,
                   const LView& L_out, int target_cell, const OpsType& ops )
    {
        (void)ops;

        using acc_type =
            m2l_accumulator_type<typename ScratchView::memory_space>;
        constexpr int n_acc = num_coeffs_per_cell * NComps;
        scalar_type* acc_base =
            reinterpret_cast<scalar_type*>( scratch.data() );
        acc_type team_acc( acc_base, n_acc );

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team_member, num_coeffs_per_cell ),
            [&]( const int out_idx )
            {
                for ( int c = 0; c < NComps; c++ )
                {
                    const int slot = out_idx * NComps + c;
                    L_out( target_cell, out_idx, c ) += team_acc( slot );
                }
            } );
    }

    // =======================================================================
    // L2L: copy the parent's local into each child.
    //
    //   L^child(0, c) += L^parent(0, c)
    //
    // A monopole local is constant over its cell, so the restriction to a
    // child is the identity and both widths and the offset are ignored.
    // Parameters mirror LaplaceKernel::l2l_translate
    // (src/Canopy_LaplaceKernel.hpp:979).
    //
    // Non-atomic, like the solid-harmonic L2L: the sweep runs one team per
    // parent and each child has exactly one parent.
    // =======================================================================
    template <class TeamMember, class LView, class AuxType, class LChildType>
    KOKKOS_INLINE_FUNCTION static void
    l2l_translate( const TeamMember& team_member, const LView& L_full,
                   int parent_cell, Scalar dx, Scalar dy, Scalar dz,
                   Scalar w_child, Scalar w_parent, const AuxType& aux,
                   const LChildType& L_child_out )
    {
        (void)dx;
        (void)dy;
        (void)dz;
        (void)w_child;
        (void)w_parent;
        (void)aux;

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team_member, num_coeffs_per_cell ),
            [&]( const int out_idx )
            {
                for ( int c = 0; c < NComps; c++ )
                    L_child_out( out_idx, c ) +=
                        L_full( parent_cell, out_idx, c );
            } );
    }

    // =======================================================================
    // L2P: the local IS the potential; the gradient is zero.
    //
    //   phi_out[c]       = L(leaf_cell, 0, c)
    //   grad_out(c, dim) = 0
    //
    // = rather than += : the sweep accumulates phi_out into its own output
    // view afterwards (src/Canopy_DownwardSweep.hpp:2009), exactly as it does
    // for the solid-harmonic basis, whose eval_phi likewise initializes
    // rather than accumulates.
    //
    // The gradient is identically zero because a monopole local is constant
    // over the cell. That is not an approximation of a true gradient — this
    // basis has none — and it is a checkable fact rather than an omission,
    // which is why tstFarFieldContract.hpp asserts it.
    //
    // Parameters mirror LaplaceKernel::l2p_evaluate
    // (src/Canopy_LaplaceKernel.hpp:1096).
    // =======================================================================
    template <class LView, class GradAccess>
    KOKKOS_INLINE_FUNCTION static void
    l2p_evaluate( const LView& L_full, int leaf_cell, Scalar dx, Scalar dy,
                  Scalar dz, Scalar w_self, Scalar ( &phi_out )[NComps],
                  const GradAccess& grad_out, bool compute_gradient )
    {
        (void)dx;
        (void)dy;
        (void)dz;
        (void)w_self;

        for ( int c = 0; c < NComps; c++ )
            phi_out[c] = L_full( leaf_cell, 0, c );

        if ( compute_gradient )
        {
            for ( int c = 0; c < NComps; c++ )
                for ( int d = 0; d < 3; d++ )
                    grad_out( c, d ) = static_cast<Scalar>( 0 );
        }
    }
};

} // namespace CanopyTest

#endif // CANOPY_TEST_MONOPOLE_BASIS_HPP
