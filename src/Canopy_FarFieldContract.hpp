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

} // namespace Canopy

#endif // CANOPY_FAR_FIELD_CONTRACT_HPP
