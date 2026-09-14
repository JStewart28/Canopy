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

#ifndef CANOPY_FAR_FIELD_CONTRACT_HPP
#define CANOPY_FAR_FIELD_CONTRACT_HPP

namespace Canopy
{

// ============================================================================
// M2LOverflow — what a basis asks the downward sweep to do with an M2L pair
// whose key did not receive an operator column.
//
// The sweep hashes every M2L pair onto a canonical key and builds one operator
// per distinct key (see the key contract in Canopy_DownwardSweep.hpp and on
// LaplaceKernel). The number of columns it will build is capped, by an
// operator count and by a memory budget in bytes; a pair whose key arrives
// after the cap binds gets op_idx = -1 and is refused a column. So is a pair
// that trips the |dd| or |offset| range guards. This enum is how a basis says
// what must then happen to that pair. It is a per-basis decision and not a
// sweep policy, because the alternatives differ in what the basis must be able
// to compute, not in what the sweep would prefer.
//
// THIS LIVES IN ITS OWN HEADER because it is contract vocabulary shared by the
// sweeps and by every basis, including bases that have nothing to do with the
// solid-harmonic one. Putting it in Canopy_LaplaceKernel.hpp would make a
// conformance basis include the solid-harmonic basis to name an enumerator.
// ============================================================================

enum class M2LOverflow
{
    // Evaluate the pair with the basis's own per-pair operator,
    // m2l_translate, at the physical geometry. The sweep's fallback tables
    // (built alongside the CSR, run by run_m2l_fallback_at_depth) already
    // implement this, and total_fallback_pair_count() counts the pairs that
    // take it. This is DIFFERENT ARITHMETIC from the operator-table path —
    // the same mathematics reassociated, evaluated per pair rather than out
    // of a precomputed table — so a configuration that moves pairs onto it
    // moves the answer at the reassociation level.
    //
    // Requires the basis to define m2l_translate. Today's behavior, and what
    // the solid-harmonic and Cartesian-Taylor bases select.
    PerPairTranslate,

    // Hand the pair to the direct sum instead: refuse it in the far field and
    // let P2P carry it. For a basis whose per-pair operator is as expensive as
    // building a column (a black-box basis doing an SVD per key, say) this is
    // the only sane response to overflow.
    //
    // NO PATH EXISTS IN THE DOWNWARD SWEEP TO TRIGGER THIS. The pair set P2P
    // evaluates is fixed by the MAC in CommunicationPlan, long before the
    // operator table is sized, and nothing downstream can add to it. A basis
    // selecting this enumerator therefore fails a static_assert in
    // DownwardSweep rather than silently producing a partial far field: a
    // basis that can neither evaluate its own operator per pair nor escalate
    // has no correct answer available, and a lenient fallback would return a
    // wrong one. Implementing escalation means adding pairs to P2P's plan,
    // which is a change to CommunicationPlan and not to this enum.
    EscalateToP2P
};

// ============================================================================
// M2LKernelParams — the kernel's own physical parameters, handed to a basis
// when it builds its M2L operators and its auxiliary tables.
//
// A basis whose operator is a function of geometry alone (the solid-harmonic
// one, whose operators are scale-normalized) ignores this entirely. A basis
// carrying PHYSICAL operators cannot: a softened kernel's operator depends on
// the softening as well as on the translation vector, and there is no other
// route from FmmConfig to the operator builder.
//
// THIS LIVES HERE, beside M2LOverflow, for the same reason: it is contract
// vocabulary that the sweeps and every basis must be able to name, including
// bases that have nothing to do with the solid-harmonic one.
//
// UNITS, which the design document requires on a declaration and which are not
// recoverable from the code:
//
//   softening  is a LENGTH, epsilon, in the same units as the particle
//              coordinates and the cell half-widths. It is NOT epsilon^2.
//              The Plummer kernel this repository's near field evaluates is
//
//                  phi(r) = 1 / sqrt( r^2 + b ),      b = epsilon^2
//
//              so a basis that wants the kernel's b must square this field.
//              The length is carried rather than b because a length is what
//              FmmConfig::softening is (Canopy_Solver.hpp) and what
//              P2P::set_softening is handed before it squares it into
//              _softening2 (Canopy_P2P.hpp) — so the one number Solver
//              decides reaches P2P, CommunicationPlan and the downward sweep
//              unchanged, and the three cannot disagree about a convention.
//
//              The value is the EFFECTIVE softening, not the configured one:
//              FmmConfig::softening < 0 selects the distribution-based
//              auto-softening, and what arrives here is the length actually
//              in force after that choice has been made.
//
// EQUALITY IS LOAD-BEARING. The downward sweep's persistent operator cache
// rests on the premise that a canonicalized key plus these parameters
// determines the operator (risk R5 in tasks/abstract-solver-backend.md), so
// the sweep compares two M2LKernelParams to decide whether the cache it holds
// is still valid. A field added here must join operator==, or a cache will
// survive a change that invalidates it.
// ============================================================================

struct M2LKernelParams
{
    // Plummer softening LENGTH epsilon; the kernel's b is epsilon^2. Zero is
    // an unsoftened kernel and is the default, which is what a sweep driven
    // directly by a test (and never handed a Solver's configuration) runs at.
    double softening = 0.0;

    bool operator==( const M2LKernelParams& o ) const noexcept
    {
        return softening == o.softening;
    }
    bool operator!=( const M2LKernelParams& o ) const noexcept
    {
        return !( *this == o );
    }
};

} // namespace Canopy

#endif // CANOPY_FAR_FIELD_CONTRACT_HPP
