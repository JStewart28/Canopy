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

// ===========================================================================
// Phase-2 (B1) Stage-A accuracy harness — far-field softening correction.
//
// Background: tasks/near-field-softening.md. The FMM far field expands the
// UNSOFTENED 1/r kernel, so near_softening_factor = k forces any cell pair
// within k*eps onto the softened P2P path. Phase 1 showed this is the cost
// blowup under clustering. B1 adds a softening *correction* to the far field
// so k can drop. This test MEASURES how much each correction order lowers the
// far-field softening error, i.e. the achievable k per correction order, so we
// can pick the production correction order from data (validate-first).
//
// What it isolates: the SOFTENING error only. The exact unsoftened direct sum
// stands in for the multipole far field (they agree to FMM truncation
// ~ (h_s/R)^(P+1), negligible vs the softening error here), so this measures
// the softening-after-correction residual without conflating FMM truncation.
//
// The correction. For a source cell (center c_s) of charges {q_j} at offsets
// d_j = r_j - c_s, evaluated at observation point o with R = o - c_s, R=|R|:
//   phi_soft(o) - phi_unsoft(o) = sum_j q_j g(|R - d_j|),  g(s)=1/sqrt(s^2+eps^2) - 1/s
// Taylor-expanding g(|R - d_j|) about d_j = 0 and resumming over the source's
// Cartesian moments Q (monopole), P (dipole), T (second moment) gives
//   Delta ≈ Q g(R)              (monopole)
//          - grad g(R) . P       (+ dipole)
//          + 1/2 sum_ab H_ab(R) T_ab   (+ quadrupole)
// with grad g = h1(R) R,  H_ab = h2(R) R_a R_b + h1(R) delta_ab, and
//   h1(s) = s^-3 - (s^2+eps^2)^-3/2,   h2(s) = 3[(s^2+eps^2)^-5/2 - s^-5].
// These moment/derivative forms are exactly what graduate into the production
// kernel in Stage B (Q,P,T come from the source multipole M_{0..2,m}).
//
// Geometry is MAC-realistic: the source half-width is set to the largest cell
// that just passes the spherical MAC against a same-size partner at distance R,
// h_s = R*theta/(2*sqrt(3)), so the dipole/quadrupole residuals reflect the real
// worst-case cell size that the floor governs.
// ===========================================================================

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace Test
{
namespace SofteningCorrectionTest
{

// g(s) and the radial factors h1(s), h2(s) of its gradient/Hessian (see header).
inline double g_of_s( double s, double eps )
{
    return 1.0 / std::sqrt( s * s + eps * eps ) - 1.0 / s;
}
inline double h1_of_s( double s, double eps )
{
    const double a = s * s + eps * eps;
    return std::pow( s, -3.0 ) - std::pow( a, -1.5 );
}
inline double h2_of_s( double s, double eps )
{
    const double a = s * s + eps * eps;
    return 3.0 * ( std::pow( a, -2.5 ) - std::pow( s, -5.0 ) );
}

// One source cell: N charges uniformly in [-h_s, h_s]^3 about the origin.
struct SourceCell
{
    std::vector<std::array<double, 3>> d; // particle offset from center
    std::vector<double> q;                // charge
    double Q = 0.0;                       // monopole  sum q
    std::array<double, 3> P{ { 0, 0, 0 } };          // dipole    sum q d
    std::array<std::array<double, 3>, 3> T{};        // 2nd moment sum q d d
};

// Source distribution shape. ISOTROPIC fills the cell ~uniformly (small
// dipole). SHEET is a thin, OFF-CENTER slab — a vortex sheet passing through a
// cell — which has a large dipole moment, the case where the dipole correction
// term should matter.
enum class Shape
{
    Isotropic,
    Sheet
};

inline SourceCell
make_source( int N, double h_s, Shape shape, std::mt19937& gen )
{
    std::uniform_real_distribution<double> pos( -h_s, h_s );
    // Same-sign (physical: masses / co-signed vortex strength) so the monopole
    // is representative and phi does not suffer accidental cancellation.
    std::uniform_real_distribution<double> qd( 0.5, 1.5 );

    SourceCell sc;
    sc.d.resize( N );
    sc.q.resize( N );
    for ( int j = 0; j < N; j++ )
    {
        if ( shape == Shape::Sheet )
        {
            // Thin in z (10% of the cell), offset by +0.5 h_s in z so the cell
            // is half-filled => strong net dipole; full extent in x, y.
            sc.d[j] = { pos( gen ), pos( gen ),
                        0.5 * h_s + 0.1 * pos( gen ) };
        }
        else
        {
            sc.d[j] = { pos( gen ), pos( gen ), pos( gen ) };
        }
        sc.q[j] = qd( gen );
        sc.Q += sc.q[j];
        for ( int a = 0; a < 3; a++ )
        {
            sc.P[a] += sc.q[j] * sc.d[j][a];
            for ( int b = 0; b < 3; b++ )
                sc.T[a][b] += sc.q[j] * sc.d[j][a] * sc.d[j][b];
        }
    }
    return sc;
}

// Relative softening error of the far field at correction levels 0..3
// (uncorrected, +monopole, +dipole, +quadrupole), for one (source, obs).
struct LevelErrors
{
    double uncorrected, mono, dip, quad;
};

inline LevelErrors
errors_for( const SourceCell& sc, const std::array<double, 3>& obs, double eps )
{
    // Exact softened reference and unsoftened far-field stand-in.
    double phi_soft = 0.0, phi_unsoft = 0.0;
    for ( size_t j = 0; j < sc.q.size(); j++ )
    {
        double s2 = 0.0;
        for ( int a = 0; a < 3; a++ )
        {
            const double e = obs[a] - sc.d[j][a];
            s2 += e * e;
        }
        const double s = std::sqrt( s2 );
        phi_unsoft += sc.q[j] / s;
        phi_soft += sc.q[j] / std::sqrt( s2 + eps * eps );
    }

    const double R = std::sqrt( obs[0] * obs[0] + obs[1] * obs[1] +
                                obs[2] * obs[2] );
    const double g = g_of_s( R, eps );
    const double h1 = h1_of_s( R, eps );
    const double h2 = h2_of_s( R, eps );

    // Monopole: Q g(R)
    const double d_mono = sc.Q * g;
    // Dipole: -grad g . P = -h1 (R . P)
    double RdotP = 0.0;
    for ( int a = 0; a < 3; a++ )
        RdotP += obs[a] * sc.P[a];
    const double d_dip = -h1 * RdotP;
    // Quadrupole: 1/2 sum_ab H_ab T_ab = 1/2 ( h2 R^T T R + h1 tr T )
    double RTR = 0.0, trT = 0.0;
    for ( int a = 0; a < 3; a++ )
    {
        trT += sc.T[a][a];
        for ( int b = 0; b < 3; b++ )
            RTR += obs[a] * sc.T[a][b] * obs[b];
    }
    const double d_quad = 0.5 * ( h2 * RTR + h1 * trT );

    const double ref = std::abs( phi_soft );
    auto rel = [&]( double corrected ) {
        return std::abs( ( phi_unsoft + corrected ) - phi_soft ) /
               ( ref > 1e-300 ? ref : 1.0 );
    };
    LevelErrors e;
    e.uncorrected = rel( 0.0 );
    e.mono = rel( d_mono );
    e.dip = rel( d_mono + d_dip );
    e.quad = rel( d_mono + d_dip + d_quad );
    return e;
}

// Sweep k = R/eps and print a table of mean relative error per correction
// level. theta is the MAC opening angle that sets the source cell size.
inline void run_sweep( double theta, Shape shape,
                       std::vector<LevelErrors>& out_mean,
                       std::vector<double>& out_k )
{
    const double eps = 1.0;
    const int N = 200;
    const int trials = 24;
    const std::array<double, 8> ks = { 1.0, 1.25, 1.5, 2.0,
                                       2.5, 3.0, 4.0, 6.0 };
    // h_s / R for a same-size pair that just passes the spherical MAC.
    const double hs_over_R = theta / ( 2.0 * std::sqrt( 3.0 ) );

    std::mt19937 gen( 13579 );
    std::uniform_real_distribution<double> dir( -1.0, 1.0 );

    std::printf( "\n=== softening-correction accuracy sweep (theta=%.2f, "
                 "h_s/R=%.4f, source=%s, P-free direct baseline) ===\n",
                 theta, hs_over_R,
                 shape == Shape::Sheet ? "SHEET (large dipole)" : "isotropic" );
    std::printf( "%6s  %12s  %12s  %12s  %12s   %10s\n", "k=R/eps",
                 "uncorrected", "+monopole", "+dipole", "+quadrupole",
                 "1/(2k^2)" );

    out_mean.clear();
    out_k.clear();
    for ( double k : ks )
    {
        const double R = k * eps;
        const double h_s = hs_over_R * R;
        LevelErrors sum{ 0, 0, 0, 0 };
        for ( int t = 0; t < trials; t++ )
        {
            SourceCell sc = make_source( N, h_s, shape, gen );
            // Observation point at distance R in a random off-axis direction.
            std::array<double, 3> u = { dir( gen ), dir( gen ), dir( gen ) };
            double un = std::sqrt( u[0] * u[0] + u[1] * u[1] + u[2] * u[2] );
            if ( un < 1e-6 )
                un = 1.0;
            std::array<double, 3> obs = { R * u[0] / un, R * u[1] / un,
                                          R * u[2] / un };
            LevelErrors e = errors_for( sc, obs, eps );
            sum.uncorrected += e.uncorrected;
            sum.mono += e.mono;
            sum.dip += e.dip;
            sum.quad += e.quad;
        }
        LevelErrors m = { sum.uncorrected / trials, sum.mono / trials,
                          sum.dip / trials, sum.quad / trials };
        std::printf( "%6.2f  %12.3e  %12.3e  %12.3e  %12.3e   %10.3e\n", k,
                     m.uncorrected, m.mono, m.dip, m.quad,
                     1.0 / ( 2.0 * k * k ) );
        out_mean.push_back( m );
        out_k.push_back( k );
    }
}

} // namespace SofteningCorrectionTest

//---------------------------------------------------------------------------//
// The uncorrected far-field softening error tracks the analytic ~1/(2k^2),
// and every added correction order strictly lowers it. The printed table is
// the Stage-A deliverable used to pick the production correction order.
//---------------------------------------------------------------------------//
TEST( SofteningCorrection, achievableKSweep )
{
    using namespace SofteningCorrectionTest;

    for ( Shape shape : { Shape::Isotropic, Shape::Sheet } )
    {
        for ( double theta : { 0.5, 0.4 } )
        {
            std::vector<LevelErrors> mean;
            std::vector<double> ks;
            run_sweep( theta, shape, mean, ks );

            for ( size_t i = 0; i < ks.size(); i++ )
            {
                const double k = ks[i];
                const LevelErrors& e = mean[i];

                // Uncorrected error is the softening error ~ 1/(2k^2): same
                // order of magnitude (within 4x) as the analytic estimate.
                const double analytic = 1.0 / ( 2.0 * k * k );
                EXPECT_LT( e.uncorrected, 4.0 * analytic )
                    << "theta=" << theta << " k=" << k;
                EXPECT_GT( e.uncorrected, analytic / 4.0 )
                    << "theta=" << theta << " k=" << k;

                // Each cumulative correction order improves on (or matches, in
                // the roundoff floor) the previous. Monopole must strictly help.
                EXPECT_LT( e.mono, e.uncorrected )
                    << "monopole did not help at theta=" << theta
                    << " k=" << k;
                if ( e.mono > 1e-9 )
                    EXPECT_LE( e.dip, e.mono * 1.01 )
                        << "dipole regressed at theta=" << theta << " k=" << k;
                if ( e.dip > 1e-9 )
                    EXPECT_LE( e.quad, e.dip * 1.01 )
                        << "quadrupole regressed at theta=" << theta
                        << " k=" << k;
            }
        }
    }
}

//---------------------------------------------------------------------------//

} // namespace Test
