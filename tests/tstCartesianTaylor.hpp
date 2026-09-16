#include "Canopy_CartesianTaylorBasis.hpp"
#include "Canopy_Solver.hpp"

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <type_traits>
#include <vector>

namespace CartesianTaylorTest
{

//---------------------------------------------------------------------------//
// T1 of tasks/cartesian-taylor-basis.md. Three bodies:
//
//   index_map_bijection  slot()/inverse_slot() are mutually inverse and slot()
//                        is a bijection onto [0, C(p+3,3)) at orders 0..6.
//   closed_forms         the §3 recurrence reproduces the four §2 closed-form
//                        tensors, |k| <= 3, at every sampled (r, b).
//   finite_difference    above |k| = 3 the only oracle is a central finite
//                        difference of phi itself, Richardson-extrapolated.
//                        Checked at |k| = 4 .. 2p with p = 2.
//
// Pure host math over no MPI and no Kokkos parallel dispatch; TEST_MEMSPACE
// and TEST_EXECSPACE are unused here.
//---------------------------------------------------------------------------//

namespace CT = Canopy::CartesianTaylor;

// The expansion order every later task in tasks/cartesian-taylor-basis.md
// uses. The M2L reaches b_{p+q} with |p+q| = 2p, so the ladder runs to 2p.
constexpr int p_order = 2;
constexpr int max_k = 2 * p_order;

//---------------------------------------------------------------------------//
// Sampled (r, b).
//
// b is SOFTENING SQUARED, the quantity added to r^2 -- not eps. The
// downstream solver's eps = 0.025 gives b = 6.25e-4, which is in the set, and
// the set spans four decades either side of it. For each b, |r| is sampled at
// 0.01, 1 and 100 times sqrt(b), i.e. |r| << sqrt(b), |r| ~ sqrt(b) and
// |r| >> sqrt(b), in three directions: axis-aligned (so the delta_ab terms of
// §2 stand alone), a generic unit vector, and the diagonal. r = 0 exactly is
// included, which is legal because b > 0. The b -> 0 limit is NOT sampled:
// b > 0 is a precondition.
//---------------------------------------------------------------------------//
struct Sample
{
    double r[3];
    double b;
};

std::vector<Sample> buildSamples()
{
    const double bs[] = { 1.0e-6, 6.25e-4, 1.0e-2, 1.0 };
    const double scales[] = { 0.01, 1.0, 100.0 };
    const double inv_sqrt3 = 1.0 / std::sqrt( 3.0 );
    const double dirs[][3] = { { 1.0, 0.0, 0.0 },
                               { 0.36, -0.48, 0.80 },
                               { inv_sqrt3, inv_sqrt3, inv_sqrt3 } };

    std::vector<Sample> out;
    for ( double b : bs )
    {
        out.push_back( Sample{ { 0.0, 0.0, 0.0 }, b } );
        for ( double s : scales )
        {
            const double mag = s * std::sqrt( b );
            for ( const auto& d : dirs )
                out.push_back(
                    Sample{ { mag * d[0], mag * d[1], mag * d[2] }, b } );
        }
    }
    return out;
}

//---------------------------------------------------------------------------//
// The §2 oracle, hand-coded in-repo.
//
// canopy-questions.md §1: the radial ladder P_m = w^{-(2m+1)/2}, so that
// P_0 = phi. The design doc records that this file's line 56 writes P_1 as
// `1/w15`, damaged plain text for w^{-3/2}.
//---------------------------------------------------------------------------//
double P( int m, double w )
{
    return std::pow( w, -0.5 * static_cast<double>( 2 * m + 1 ) );
}

double wOf( const double r[3], double b )
{
    return r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + b;
}

double phi( const double r[3], double b ) { return 1.0 / std::sqrt( wOf( r, b ) ); }

int delta( int a, int b ) { return ( a == b ) ? 1 : 0; }

// canopy-questions.md §2:  b_empty = P_0
double ref0( const double r[3], double b ) { return P( 0, wOf( r, b ) ); }

// canopy-questions.md §2:  (d_a) = -r_a P_1
double ref1( const double r[3], double bb, int a )
{
    const double w = wOf( r, bb );
    return -r[a] * P( 1, w );
}

// canopy-questions.md §2:  (d_a d_b) = -delta_ab P_1 + 3 r_a r_b P_2
double ref2( const double r[3], double bb, int a, int b )
{
    const double w = wOf( r, bb );
    return -static_cast<double>( delta( a, b ) ) * P( 1, w ) +
           3.0 * r[a] * r[b] * P( 2, w );
}

// canopy-questions.md §2:
//   (d_a d_b d_c) = 3( delta_ab r_c + delta_ac r_b + delta_bc r_a ) P_2
//                   - 15 r_a r_b r_c P_3
double ref3( const double r[3], double bb, int a, int b, int c )
{
    const double w = wOf( r, bb );
    const double t = static_cast<double>( delta( a, b ) ) * r[c] +
                     static_cast<double>( delta( a, c ) ) * r[b] +
                     static_cast<double>( delta( b, c ) ) * r[a];
    return 3.0 * t * P( 2, w ) - 15.0 * r[a] * r[b] * r[c] * P( 3, w );
}

// Expand a multi-index into its list of directions, e.g. (2,0,1) -> {0,0,2}.
void expandDirections( const int k[3], int dirs_out[3] )
{
    int n = 0;
    for ( int d = 0; d < 3; ++d )
        for ( int c = 0; c < k[d]; ++c )
            dirs_out[n++] = d;
}

// Single entry point to the §2 oracle for any |k| <= 3.
double referenceClosedForm( const int k[3], const double r[3], double b )
{
    const int n = k[0] + k[1] + k[2];
    int d[3] = { 0, 0, 0 };
    expandDirections( k, d );
    switch ( n )
    {
    case 0:
        return ref0( r, b );
    case 1:
        return ref1( r, b, d[0] );
    case 2:
        return ref2( r, b, d[0], d[1] );
    default:
        return ref3( r, b, d[0], d[1], d[2] );
    }
}

//---------------------------------------------------------------------------//
// The finite-difference oracle for |k| > 3.
//
// Central stencils, one per direction, tensor-producted. Each 1-D stencil is
// O(h^2) accurate and its error expansion carries even powers of h only, so
// the product's does too and Richardson extrapolation applies twice:
//
//     R1(h) = ( 4 D(h/2)  - D(h)  ) /  3     kills h^2, leaves O(h^4)
//     R2(h) = ( 16 R1(h/2) - R1(h) ) / 15    kills h^4, leaves O(h^6)
//
// Two steps rather than one because this is the ONLY oracle above |k| = 3 and
// its sharpness is what bounds how small an index-map or recurrence error has
// to be to slip through (risks R2 and R8). One step bottoms out around 7e-6
// of scale, truncation-limited at h = L/64 and roundoff-limited below it, and
// leaves no usable margin. Two steps reach the roundoff floor at h = L/32:
// see the measured figures on fd_tol below.
//
// Returns the point count; offs[] are integer multiples of h and wts[] are the
// weights BEFORE the 1/h^order scaling, which fdDerivative applies once.
//---------------------------------------------------------------------------//
int stencil1D( int order, int offs[5], double wts[5] )
{
    switch ( order )
    {
    case 0:
        offs[0] = 0; wts[0] = 1.0;
        return 1;
    case 1:
        offs[0] = -1; wts[0] = -0.5;
        offs[1] = 1;  wts[1] = 0.5;
        return 2;
    case 2:
        offs[0] = -1; wts[0] = 1.0;
        offs[1] = 0;  wts[1] = -2.0;
        offs[2] = 1;  wts[2] = 1.0;
        return 3;
    case 3:
        offs[0] = -2; wts[0] = -0.5;
        offs[1] = -1; wts[1] = 1.0;
        offs[2] = 1;  wts[2] = -1.0;
        offs[3] = 2;  wts[3] = 0.5;
        return 4;
    default:
        offs[0] = -2; wts[0] = 1.0;
        offs[1] = -1; wts[1] = -4.0;
        offs[2] = 0;  wts[2] = 6.0;
        offs[3] = 1;  wts[3] = -4.0;
        offs[4] = 2;  wts[4] = 1.0;
        return 5;
    }
}

double fdDerivative( const double r0[3], double b, const int k[3], double h )
{
    int ox[5], oy[5], oz[5];
    double wx[5], wy[5], wz[5];
    const int nx = stencil1D( k[0], ox, wx );
    const int ny = stencil1D( k[1], oy, wy );
    const int nz = stencil1D( k[2], oz, wz );

    double acc = 0.0;
    for ( int ix = 0; ix < nx; ++ix )
        for ( int iy = 0; iy < ny; ++iy )
            for ( int iz = 0; iz < nz; ++iz )
            {
                const double pt[3] = { r0[0] + ox[ix] * h,
                                       r0[1] + oy[iy] * h,
                                       r0[2] + oz[iz] * h };
                acc += wx[ix] * wy[iy] * wz[iz] * phi( pt, b );
            }

    return acc / std::pow( h, static_cast<double>( k[0] + k[1] + k[2] ) );
}

double fdRichardson( const double r0[3], double b, const int k[3], double h )
{
    const double d0 = fdDerivative( r0, b, k, h );
    const double d1 = fdDerivative( r0, b, k, 0.5 * h );
    const double d2 = fdDerivative( r0, b, k, 0.25 * h );

    const double r1_coarse = ( 4.0 * d1 - d0 ) / 3.0;
    const double r1_fine = ( 4.0 * d2 - d1 ) / 3.0;

    return ( 16.0 * r1_fine - r1_coarse ) / 15.0;
}

//---------------------------------------------------------------------------//
// Test 1 -- the index map is a bijection.
//
// Run at orders 0 through 6 even though p = 2 only needs 2p = 4. It is cheap,
// and the index map is exactly where hand-derived Cartesian FMMs break (risk
// R2 of tasks/cartesian-taylor-basis.md): an index-map error is invisible
// below |k| = 4, where the map first has non-trivial structure.
//---------------------------------------------------------------------------//
void testIndexMapBijection()
{
    for ( int p = 0; p <= 6; ++p )
    {
        const int n_slots = CT::num_slots( p );
        ASSERT_EQ( n_slots, ( p + 1 ) * ( p + 2 ) * ( p + 3 ) / 6 )
            << "num_slots(" << p << ") must be C(p+3,3)";

        std::vector<int> hits( n_slots, 0 );

        for ( int kx = 0; kx <= p; ++kx )
            for ( int ky = 0; ky + kx <= p; ++ky )
                for ( int kz = 0; kz + ky + kx <= p; ++kz )
                {
                    const int s = CT::slot( kx, ky, kz );

                    ASSERT_GE( s, 0 )
                        << "slot(" << kx << "," << ky << "," << kz
                        << ") = " << s << " is out of range at order p = " << p;
                    ASSERT_LT( s, n_slots )
                        << "slot(" << kx << "," << ky << "," << kz
                        << ") = " << s << " is out of range [0," << n_slots
                        << ") at order p = " << p;

                    ASSERT_EQ( hits[s], 0 )
                        << "slot(" << kx << "," << ky << "," << kz
                        << ") = " << s
                        << " collides: that slot is already taken at order p = "
                        << p;
                    hits[s] = 1;

                    // Graded: the slot of a multi-index of degree n lies in
                    // that degree's block, so an order-p table is a prefix of
                    // any larger one.
                    const int n = kx + ky + kz;
                    ASSERT_GE( s, CT::slot_degree_base( n ) )
                        << "slot(" << kx << "," << ky << "," << kz
                        << ") = " << s << " is below the degree-" << n
                        << " block; the order is not degree-graded";
                    ASSERT_LT( s, CT::slot_degree_base( n + 1 ) )
                        << "slot(" << kx << "," << ky << "," << kz
                        << ") = " << s << " is above the degree-" << n
                        << " block; the order is not degree-graded";

                    int back[3] = { -1, -1, -1 };
                    CT::inverse_slot( s, back );
                    ASSERT_TRUE( back[0] == kx && back[1] == ky &&
                                 back[2] == kz )
                        << "inverse_slot( slot(" << kx << "," << ky << ","
                        << kz << ") = " << s << " ) gave (" << back[0] << ","
                        << back[1] << "," << back[2] << "), expected (" << kx
                        << "," << ky << "," << kz << ")";
                }

        for ( int s = 0; s < n_slots; ++s )
        {
            ASSERT_EQ( hits[s], 1 )
                << "slot " << s << " of " << n_slots
                << " is unreached at order p = " << p
                << "; slot() is not onto";

            // The other direction of the bijection: every flat index in range
            // maps back to a multi-index whose slot is itself.
            int k[3];
            CT::inverse_slot( s, k );
            ASSERT_EQ( CT::slot( k[0], k[1], k[2] ), s )
                << "slot( inverse_slot(" << s << ") = (" << k[0] << ","
                << k[1] << "," << k[2] << ") ) != " << s;
            ASSERT_LE( k[0] + k[1] + k[2], p )
                << "inverse_slot(" << s << ") = (" << k[0] << "," << k[1]
                << "," << k[2] << ") has degree above p = " << p;
        }

        std::printf( "[cartesian-taylor] index map bijective at p = %d over "
                     "%d slots\n",
                     p, n_slots );
    }
}

//---------------------------------------------------------------------------//
// Test 2 -- the §3 recurrence reproduces the §2 closed forms, |k| <= 3.
//
// Tolerance: the recurrence and the closed forms are different algebraic
// evaluations of the same quantity, so they agree to rounding, not bitwise.
// The comparison is against the natural scale of a |k|-th derivative,
//
//     scale = phi / L^|k|,     L = sqrt(w),
//
// rather than against the value itself, because individual components vanish
// identically at the sampled r (any r_a = 0 kills the odd terms) and a
// relative test would divide by zero there. closed_form_tol is a few hundred
// ulp of that scale; the achieved maximum is printed below for the record.
//---------------------------------------------------------------------------//
constexpr double closed_form_tol = 1.0e-12;

void testClosedForms()
{
    const auto samples = buildSamples();
    std::vector<double> bk( CT::num_slots( max_k ) );

    double worst = 0.0;
    int worst_k[3] = { 0, 0, 0 };

    for ( const auto& smp : samples )
    {
        CT::derivative_ladder( smp.r, smp.b, max_k, bk.data() );

        const double w = wOf( smp.r, smp.b );
        const double L = std::sqrt( w );
        const double phi0 = 1.0 / L;

        for ( int n = 0; n <= 3; ++n )
        {
            const double scale = phi0 / std::pow( L, static_cast<double>( n ) );

            for ( int kx = 0; kx <= n; ++kx )
                for ( int ky = 0; ky + kx <= n; ++ky )
                {
                    const int kz = n - kx - ky;
                    const int k[3] = { kx, ky, kz };
                    const double got = bk[CT::slot( kx, ky, kz )];
                    const double want = referenceClosedForm( k, smp.r, smp.b );
                    const double err = std::abs( got - want ) / scale;

                    if ( err > worst )
                    {
                        worst = err;
                        worst_k[0] = kx;
                        worst_k[1] = ky;
                        worst_k[2] = kz;
                    }

                    ASSERT_LE( err, closed_form_tol )
                        << "canopy-questions.md §3 recurrence disagrees with "
                        << "the §2 closed form at multi-index (" << kx << ","
                        << ky << "," << kz << "), |k| = " << n
                        << ": recurrence " << got << ", closed form " << want
                        << ", |diff| / (phi/L^|k|) = " << err
                        << " > " << closed_form_tol << "; at r = (" << smp.r[0]
                        << "," << smp.r[1] << "," << smp.r[2]
                        << "), b = " << smp.b;
                }
        }
    }

    std::printf( "[cartesian-taylor] closed forms |k| <= 3: %d samples, worst "
                 "|diff|/(phi/L^|k|) = %.3e at k = (%d,%d,%d), tol = %.1e\n",
                 static_cast<int>( samples.size() ), worst, worst_k[0],
                 worst_k[1], worst_k[2], closed_form_tol );
}

//---------------------------------------------------------------------------//
// Test 3 -- the finite-difference check, |k| = 4 .. 2p.
//
// At p = 2 that is |k| = 4 exactly. The lower bound is 4 because §2 covers
// everything below it exactly and the FD oracle is strictly worse there.
//
// Step: h = L / fd_h_divisor with L = sqrt(w), the length scale phi actually
// varies on at this (r, b). Nondimensionalizing the step this way is what
// makes one tolerance hold across four decades of b and four of |r|/sqrt(b).
//
// Tolerance, measured against the same scale = phi / L^|k| as test 2. The
// divisor was chosen by scanning it over this exact sample set at |k| = 4,
// worst case over all of it:
//
//   h = L/8    1.3e-4      truncation-limited, falling as (h/L)^6
//   h = L/16   1.9e-6
//   h = L/32   4.3e-7      the floor -- truncation and roundoff balanced here
//   h = L/64   7.2e-6      roundoff-limited, rising as (L/h)^4
//
// so fd_tol sits ~23x above the achieved 4.3e-7, which is as much margin as
// a double-precision 4th-derivative difference has to give.
//
// Risk R8 of tasks/cartesian-taylor-basis.md: this tolerance is
// SCALE-DEPENDENT and is the instrument for the forward recurrence's
// conditioning. If p is ever raised above 2, re-measure it at the (r, b)
// scales actually in use -- do not assume it, and do not widen it to
// accommodate a failure. The achieved maximum is printed below.
//---------------------------------------------------------------------------//
constexpr double fd_h_divisor = 32.0;
constexpr double fd_tol = 1.0e-5;

void testFiniteDifference()
{
    const auto samples = buildSamples();
    std::vector<double> bk( CT::num_slots( max_k ) );

    double worst = 0.0;
    int worst_k[3] = { 0, 0, 0 };
    double worst_b = 0.0;

    for ( const auto& smp : samples )
    {
        CT::derivative_ladder( smp.r, smp.b, max_k, bk.data() );

        const double w = wOf( smp.r, smp.b );
        const double L = std::sqrt( w );
        const double phi0 = 1.0 / L;
        const double h = L / fd_h_divisor;

        for ( int n = 4; n <= max_k; ++n )
        {
            const double scale = phi0 / std::pow( L, static_cast<double>( n ) );

            for ( int kx = 0; kx <= n; ++kx )
                for ( int ky = 0; ky + kx <= n; ++ky )
                {
                    const int kz = n - kx - ky;
                    const int k[3] = { kx, ky, kz };
                    const double got = bk[CT::slot( kx, ky, kz )];
                    const double want = fdRichardson( smp.r, smp.b, k, h );
                    const double err = std::abs( got - want ) / scale;

                    if ( err > worst )
                    {
                        worst = err;
                        worst_k[0] = kx;
                        worst_k[1] = ky;
                        worst_k[2] = kz;
                        worst_b = smp.b;
                    }

                    ASSERT_LE( err, fd_tol )
                        << "canopy-questions.md §3 recurrence disagrees with "
                        << "the Richardson-extrapolated central difference of "
                        << "phi at multi-index (" << kx << "," << ky << ","
                        << kz << "), |k| = " << n << ": recurrence " << got
                        << ", finite difference " << want
                        << ", |diff| / (phi/L^|k|) = " << err << " > "
                        << fd_tol << "; at r = (" << smp.r[0] << ","
                        << smp.r[1] << "," << smp.r[2] << "), b = " << smp.b
                        << ", h = L/" << fd_h_divisor
                        << ", two Richardson steps";
                }
        }
    }

    std::printf( "[cartesian-taylor] finite difference |k| = 4..%d: worst "
                 "|diff|/(phi/L^|k|) = %.3e at k = (%d,%d,%d), b = %.3e, "
                 "h = L/%.0f, tol = %.1e\n",
                 max_k, worst, worst_k[0], worst_k[1], worst_k[2], worst_b,
                 fd_h_divisor, fd_tol );
}

//===========================================================================//
// T2 of tasks/cartesian-taylor-basis.md. Four more bodies, all host-only math
// over no MPI and no tree:
//
//   m2m_shift           P2M then M2M against a direct P2M about the parent
//                       center, and the shift round trip by -s.
//   l2l_shift           L2L round trip by -s, and the shifted child
//                       polynomial against the parent polynomial at the same
//                       PHYSICAL point.
//   l2p_evaluation      L2P against a brute-force sum_p a^p/p! l_p, and its
//                       ANALYTIC gradient against a Richardson-extrapolated
//                       central difference of that same polynomial.
//   solver_instantiates COMPILE-ONLY: Solver< ..., CartesianTaylorBasis > is
//                       a complete type, which runs the six class-scope sweep
//                       guards against this basis.
//
// Each numerical body runs at TWO orders, p = 2 and p = 4. p = 2 is what T4
// and the downstream solver use; p = 4 is there because several of the checks
// below degenerate at low order -- the shift sums have few terms and the
// finite difference of a quadratic is exact for a reason that stops holding
// as the degree grows -- and a body that only ever ran at p = 2 would not
// notice a degree-dependent indexing error.
//
// The oracles here are deliberately INDEPENDENT of the basis: the reference
// monomial d^k/k! is computed by brute-force repeated multiplication and an
// explicit factorial (monoOverFact below), not by calling the basis's own
// taylor_monomials. Checking taylor_monomials against itself would pass with
// any consistent indexing error, which is precisely the failure mode
// (risk R2).
//===========================================================================//

namespace CT2
{

// Brute-force k! -- the reference factorial, never the basis's.
inline double factorial( int n )
{
    double f = 1.0;
    for ( int i = 2; i <= n; ++i )
        f *= static_cast<double>( i );
    return f;
}

// Brute-force d^k / k!, the reference Taylor monomial. Repeated
// multiplication and one division per axis; no pow(), no table, and nothing
// from Canopy::CartesianTaylorBasis.
inline double monoOverFact( const double d[3], const int k[3] )
{
    double v = 1.0;
    for ( int a = 0; a < 3; ++a )
    {
        for ( int j = 0; j < k[a]; ++j )
            v *= d[a];
        v /= factorial( k[a] );
    }
    return v;
}

// The GradWriter shape DownwardSweep hands l2p_evaluate
// (src/Canopy_DownwardSweep.hpp:190-198): operator()(c, dim) returning a
// writable reference. Reproduced here rather than reached into, because the
// sweep's is a nested type of a class this test does not instantiate.
template <class View2D>
struct GradWriter
{
    View2D g;
    KOKKOS_INLINE_FUNCTION double& operator()( int c, int d ) const
    {
        return g( c, d );
    }
};

// Number of components every body below runs at. 3 is what T4 and the
// downstream solver use; the operators are componentwise, so this is a
// multiplicity check and not a physics one.
constexpr int NC = 3;

// The synthetic source distribution. Positions are RELATIVE TO THE CHILD
// CENTER and are the same at every order, so a failure at p = 4 that is
// absent at p = 2 is a degree effect and not a geometry one. Nothing here is
// symmetric: a symmetric set would let a sign error in an odd-degree moment
// cancel.
struct Particle
{
    double d[3];
    double q[NC];
};

inline std::vector<Particle> buildParticles()
{
    return {
        Particle{ { 0.17, -0.31, 0.08 }, { 1.0, -2.0, 0.5 } },
        Particle{ { -0.42, 0.05, -0.23 }, { -0.25, 3.0, 1.5 } },
        Particle{ { 0.36, 0.44, 0.19 }, { 2.0, 0.75, -1.0 } },
        Particle{ { -0.09, -0.48, 0.41 }, { 0.5, -0.5, 2.25 } },
        Particle{ { 0.28, 0.12, -0.45 }, { -1.75, 1.25, 0.125 } },
    };
}

// The center offset s = c_child - c_parent used by both shift bodies. Not
// axis-aligned and not a power of two, so no term of a shift sum is
// accidentally exact.
constexpr double s_shift[3] = { 0.37, -0.62, 0.21 };

//---------------------------------------------------------------------------//
// Views, allocated per body. LayoutRight (cell, coeff, comp_set_slot) is the
// sweeps' own coefficient layout.
//---------------------------------------------------------------------------//
using CoeffView = Kokkos::View<double***, Kokkos::LayoutRight, TEST_MEMSPACE>;

//---------------------------------------------------------------------------//
// Run P2M for every particle into cell `cell`, with offsets measured from
// `center_offset` -- i.e. d_j = particle.d - center_offset, which lets one
// particle list be accumulated about the child center (offset 0) or about the
// parent center (offset -s) without moving the particles.
//---------------------------------------------------------------------------//
template <class Basis>
void runP2M( const CoeffView& M, int cell,
             const std::vector<Particle>& particles,
             const double center_offset[3], double w_self )
{
    const int np = static_cast<int>( particles.size() );

    Kokkos::View<double**, Kokkos::LayoutRight, TEST_MEMSPACE> pos(
        "pos", np, 3 );
    Kokkos::View<double**, Kokkos::LayoutRight, TEST_MEMSPACE> chg(
        "chg", np, NC );
    auto h_pos = Kokkos::create_mirror_view( pos );
    auto h_chg = Kokkos::create_mirror_view( chg );
    for ( int j = 0; j < np; ++j )
    {
        for ( int a = 0; a < 3; ++a )
            h_pos( j, a ) = particles[j].d[a] - center_offset[a];
        for ( int c = 0; c < NC; ++c )
            h_chg( j, c ) = particles[j].q[c];
    }
    Kokkos::deep_copy( pos, h_pos );
    Kokkos::deep_copy( chg, h_chg );

    Kokkos::parallel_for(
        "p2m", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, np ),
        KOKKOS_LAMBDA( const int j ) {
            double charges[NC];
            for ( int c = 0; c < NC; ++c )
                charges[c] = chg( j, c );
            auto M_out = Kokkos::subview( M, cell, Kokkos::ALL, Kokkos::ALL );
            Basis::p2m_contribution( charges, pos( j, 0 ), pos( j, 1 ),
                                     pos( j, 2 ), w_self, M_out );
        } );
    Kokkos::fence();
}

//---------------------------------------------------------------------------//
// One team, one M2M: shift cell `src`'s moments by (dx,dy,dz) into cell
// `dst`. The sweep runs one team per parent; a league of one reproduces that
// without a tree.
//---------------------------------------------------------------------------//
template <class Basis>
void runM2M( const CoeffView& M, int src, int dst, const double sh[3] )
{
    using policy = Kokkos::TeamPolicy<TEST_EXECSPACE>;
    typename Basis::template aux_tables_type<TEST_MEMSPACE> aux;
    const double dx = sh[0], dy = sh[1], dz = sh[2];

    Kokkos::parallel_for(
        "m2m", policy( 1, 1 ),
        KOKKOS_LAMBDA( const typename policy::member_type& team ) {
            auto M_par = Kokkos::subview( M, dst, Kokkos::ALL, Kokkos::ALL );
            // Widths are ignored by this basis; 1.0 is passed so a future
            // read of one would be visible rather than silently zero.
            Basis::m2m_translate( team, M, src, dx, dy, dz, 1.0, 2.0, aux,
                                  M_par );
        } );
    Kokkos::fence();
}

//---------------------------------------------------------------------------//
// One team, one L2L: shift cell `src`'s locals by (dx,dy,dz) into cell `dst`.
//---------------------------------------------------------------------------//
template <class Basis>
void runL2L( const CoeffView& L, int src, int dst, const double sh[3] )
{
    using policy = Kokkos::TeamPolicy<TEST_EXECSPACE>;
    typename Basis::template aux_tables_type<TEST_MEMSPACE> aux;
    const double dx = sh[0], dy = sh[1], dz = sh[2];

    Kokkos::parallel_for(
        "l2l", policy( 1, 1 ),
        KOKKOS_LAMBDA( const typename policy::member_type& team ) {
            auto L_ch = Kokkos::subview( L, dst, Kokkos::ALL, Kokkos::ALL );
            Basis::l2l_translate( team, L, src, dx, dy, dz, 1.0, 2.0, aux,
                                  L_ch );
        } );
    Kokkos::fence();
}

//---------------------------------------------------------------------------//
// L2P at one offset. Returns the potentials; fills `grad` when asked.
//---------------------------------------------------------------------------//
template <class Basis>
void runL2P( const CoeffView& L, int cell, const double a[3], double phi[NC],
             double grad[NC][3], bool compute_gradient )
{
    Kokkos::View<double*, TEST_MEMSPACE> phi_v( "phi", NC );
    Kokkos::View<double**, Kokkos::LayoutRight, TEST_MEMSPACE> grad_v(
        "grad", NC, 3 );
    const double ax = a[0], ay = a[1], az = a[2];

    Kokkos::parallel_for(
        "l2p", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 1 ),
        KOKKOS_LAMBDA( const int ) {
            double out[NC];
            GradWriter<decltype( grad_v )> w{ grad_v };
            Basis::l2p_evaluate( L, cell, ax, ay, az, 1.0, out, w,
                                 compute_gradient );
            for ( int c = 0; c < NC; ++c )
                phi_v( c ) = out[c];
        } );
    Kokkos::fence();

    auto h_phi = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                      phi_v );
    auto h_grad = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                       grad_v );
    for ( int c = 0; c < NC; ++c )
    {
        phi[c] = h_phi( c );
        for ( int d = 0; d < 3; ++d )
            grad[c][d] = h_grad( c, d );
    }
}

// Deterministic, non-symmetric synthetic local coefficients. Magnitudes vary
// across slots so that a slot permutation cannot be absorbed.
inline double syntheticLocal( int s, int c )
{
    return 1.0 + 0.37 * static_cast<double>( s ) -
           0.11 * static_cast<double>( s * s ) +
           0.53 * static_cast<double>( c + 1 ) *
               ( ( s % 3 == 0 ) ? -1.0 : 1.0 );
}

//---------------------------------------------------------------------------//
// T3 -- the M2L. Oracles, geometries and path runners.
//
// Everything here is either an ORACLE, which shares no code with the basis,
// or a RUNNER, which drives one of the basis's two M2L paths exactly as the
// sweep does. The one thing no body below re-implements is the operator
// itself: Basis::m2l_operator_block is the single source of truth the
// Conventions table requires, and every check reaches an operator value
// through it or through build_m2l_operators, never through a transcription.
//---------------------------------------------------------------------------//

// The five integers DownwardSweep::M2LKey carries
// (src/Canopy_DownwardSweep.hpp:583-595), reproduced rather than reached
// into: M2LKey is a nested type of DownwardSweep<..., KernelType>, which this
// file never instantiates, and build_m2l_operators is a template on the key
// type precisely so a caller can hand it its own.
// tests/tstFarFieldContract.hpp:912 does the same.
//
// (ii, jj, kk) * unit_w[max_d] is SOURCE CENTER MINUS TARGET CENTER
// (src/Canopy_DownwardSweep.hpp:1464-1481). It is NOT R.
struct M2LProbeKey
{
    int max_d;
    int dd;
    int ii;
    int jj;
    int kk;
};

//---------------------------------------------------------------------------//
// The multi-indexed moments of buildParticles(),
//
//     M[ q_slot * NC + c ] = sum_j d_j^q / q! * s_jc ,
//
// with d_j measured from the source cell center. Brute force through
// monoOverFact -- never p2m_contribution -- so the moments the M2L is checked
// against share no code with the basis (risk R2). THE 1/q! IS HERE, which is
// the convention half of R1: the operator carries raw derivatives and must
// not apply a factorial a second time.
//---------------------------------------------------------------------------//
inline std::vector<double> buildMoments( const std::vector<Particle>& ps,
                                         int n_slots )
{
    std::vector<double> M( static_cast<std::size_t>( n_slots ) * NC, 0.0 );
    for ( const auto& p : ps )
        for ( int s = 0; s < n_slots; ++s )
        {
            int k[3];
            CT::inverse_slot( s, k );
            const double m = monoOverFact( p.d, k );
            for ( int c = 0; c < NC; ++c )
                M[static_cast<std::size_t>( s ) * NC + c] += m * p.q[c];
        }
    return M;
}

//---------------------------------------------------------------------------//
// The reference treecode's moments: FULL SYMMETRIC TENSORS, plain sums with
// no factorial anywhere --
//
//     G_c = sum_j s_jc ,  D_{a,c} = sum_j d_ja s_jc ,
//     Q_{ab,c} = sum_j d_ja d_jb s_jc
//
// -- which is canopy-questions.md §4's "sum gamma, sum d(x)gamma,
// sum d(x)d(x)gamma". This is a DIFFERENT REPRESENTATION of the same
// distribution as buildMoments above, and converting between the two costs
// the multinomial |q|!/q!:
//
//     sum_{a,b} T_ab d_a d_b = sum_{|q|=2} (|q|!/q!) T_q d^q .
//
// Getting that factor wrong is a route to the R1 failure, so the contraction
// below applies it explicitly rather than letting the two representations
// look interchangeable.
//---------------------------------------------------------------------------//
struct TensorMoments
{
    double G[NC];
    double D[3][NC];
    double Q[3][3][NC];
};

inline TensorMoments buildTensorMoments( const std::vector<Particle>& ps )
{
    TensorMoments t;
    for ( int c = 0; c < NC; ++c )
    {
        t.G[c] = 0.0;
        for ( int a = 0; a < 3; ++a )
        {
            t.D[a][c] = 0.0;
            for ( int b = 0; b < 3; ++b )
                t.Q[a][b][c] = 0.0;
        }
    }
    for ( const auto& p : ps )
        for ( int c = 0; c < NC; ++c )
        {
            t.G[c] += p.q[c];
            for ( int a = 0; a < 3; ++a )
            {
                t.D[a][c] += p.d[a] * p.q[c];
                for ( int b = 0; b < 3; ++b )
                    t.Q[a][b][c] += p.d[a] * p.d[b] * p.q[c];
            }
        }
    return t;
}

//---------------------------------------------------------------------------//
// The reference's three arrays, TRANSCRIBED FROM canopy-questions.md §2 --
// the in-repo oracle -- and not from treecode.py, which is not in this
// repository and which no test here depends on having. Each is MINUS the
// corresponding §2 tensor shifted by one index, because the reference returns
// a VELOCITY and K = -grad phi:
//
//     K_a     =  r_a P_1                              = -( d_a phi )
//     dK_ab   =  delta_ab P_1 - 3 r_a r_b P_2         = -( d_a d_b phi )
//     ddK_abc = -3( delta_ab r_c + delta_ac r_b + delta_bc r_a ) P_2
//               + 15 r_a r_b r_c P_3                  = -( d_a d_b d_c phi )
//
// ddK carries derivatives of degree THREE, which is why contracting these
// three against the degree-0/1/2 moments gives the |p| = 1 local coefficients
// and not l_0 -- and it is the only check here that exercises b_k at |k| = 3.
//---------------------------------------------------------------------------//
inline double refKa( const double r[3], double bb, int a )
{
    return r[a] * P( 1, wOf( r, bb ) );
}

inline double refdKab( const double r[3], double bb, int a, int b )
{
    const double w = wOf( r, bb );
    return static_cast<double>( delta( a, b ) ) * P( 1, w ) -
           3.0 * r[a] * r[b] * P( 2, w );
}

inline double refddKabc( const double r[3], double bb, int a, int b, int c )
{
    const double w = wOf( r, bb );
    const double t = static_cast<double>( delta( a, b ) ) * r[c] +
                     static_cast<double>( delta( a, c ) ) * r[b] +
                     static_cast<double>( delta( b, c ) ) * r[a];
    return -3.0 * t * P( 2, w ) + 15.0 * r[a] * r[b] * r[c] * P( 3, w );
}

//---------------------------------------------------------------------------//
// The (R, b) the two convention checks run over. R = c_target - c_source, and
// |R| sits well outside the source particles' own extent (|d| <= 0.48) so the
// expansion is in its convergent regime; neither convention check depends on
// that, but a divergent configuration would make the printed scales
// uninterpretable. b is SOFTENING SQUARED: 6.25e-4 is the downstream solver's
// eps = 0.025 squared.
//---------------------------------------------------------------------------//
struct M2LGeom
{
    double R[3];
    double b;
};

inline std::vector<M2LGeom> buildM2LGeometries()
{
    // Unit directions: two axis-aligned (so the delta_ab terms of §2 stand
    // alone and no r_a r_b term masks a sign), and two generic with mixed
    // signs (so no component of any odd-degree tensor vanishes).
    const double dirs[][3] = { { 1.0, 0.0, 0.0 },
                               { 0.0, -1.0, 0.0 },
                               { 0.36, -0.48, 0.80 },
                               { -0.48, -0.60, 0.64 } };
    const double mags[] = { 2.0, 8.0 };
    const double bs[] = { 6.25e-4, 1.0e-2 };

    std::vector<M2LGeom> out;
    for ( double bb : bs )
        for ( double m : mags )
            for ( const auto& d : dirs )
                out.push_back(
                    M2LGeom{ { m * d[0], m * d[1], m * d[2] }, bb } );
    return out;
}

//---------------------------------------------------------------------------//
// Direct softened sum -- the end-to-end oracle. No expansion of any kind:
//
//     u_c(x) = sum_j s_jc / sqrt( |x - y_j|^2 + b ) ,
//              y_j = c_source + d_j
//
// through the same phi() the §2 oracle is built on.
//---------------------------------------------------------------------------//
inline void directSum( const std::vector<Particle>& ps, const double c_src[3],
                       const double x[3], double bb, double out[NC] )
{
    for ( int c = 0; c < NC; ++c )
        out[c] = 0.0;

    for ( const auto& p : ps )
    {
        const double r[3] = { x[0] - ( c_src[0] + p.d[0] ),
                              x[1] - ( c_src[1] + p.d[1] ),
                              x[2] - ( c_src[2] + p.d[2] ) };
        const double g = phi( r, bb );
        for ( int c = 0; c < NC; ++c )
            out[c] += p.q[c] * g;
    }
}

//---------------------------------------------------------------------------//
// The FUSED path, driven exactly as DownwardSweep::run_m2l_fused drives it
// (src/Canopy_DownwardSweep.hpp:2195-2251): one team per TARGET, raw
// zero-filled scratch bytes sized by m2l_scratch_bytes, m2l_pre_cell and
// m2l_core once per source, m2l_post_cell once at the end. A league of one
// reproduces that without a tree.
//
// Reproducing the loop rather than calling the sweep is what makes this a
// test of the basis: the sweep is not instantiated here and no tree, no MPI
// and no operator cache is involved.
//---------------------------------------------------------------------------//
template <class Basis>
void runM2LFused(
    const CoeffView& M, const std::vector<int>& src_cells,
    const std::vector<int>& op_idx,
    const typename Basis::template m2l_operators_type<TEST_MEMSPACE>& ops,
    const CoeffView& L, int target_cell )
{
    using policy = Kokkos::TeamPolicy<TEST_EXECSPACE>;
    using member_t = typename policy::member_type;
    using scratch_space = typename TEST_EXECSPACE::scratch_memory_space;
    using ScratchBytes =
        Kokkos::View<char*, scratch_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    const int n_pairs = static_cast<int>( src_cells.size() );
    Kokkos::View<int*, TEST_MEMSPACE> d_src( "m2l_src", n_pairs );
    Kokkos::View<int*, TEST_MEMSPACE> d_op( "m2l_op_idx", n_pairs );
    auto h_src = Kokkos::create_mirror_view( d_src );
    auto h_op = Kokkos::create_mirror_view( d_op );
    for ( int s = 0; s < n_pairs; ++s )
    {
        h_src( s ) = src_cells[s];
        h_op( s ) = op_idx[s];
    }
    Kokkos::deep_copy( d_src, h_src );
    Kokkos::deep_copy( d_op, h_op );

    // constexpr for the same reason the sweep spells it that way (risk R3):
    // the stages' internal extents stay compile-time constants.
    constexpr std::size_t scratch_bytes = Basis::m2l_scratch_bytes( NC );
    constexpr int scratch_bytes_int = static_cast<int>( scratch_bytes );

    policy pol( 1, 1 );
    pol.set_scratch_size(
        0, Kokkos::PerTeam( ScratchBytes::shmem_size( scratch_bytes ) ) );

    Kokkos::parallel_for(
        "m2l_fused", pol,
        KOKKOS_LAMBDA( const member_t& team ) {
            ScratchBytes scratch( team.team_scratch( 0 ), scratch_bytes );

            // Byte fill, as the sweep does: the layout inside is the basis's
            // and all-zero bytes are +0.0.
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange( team, scratch_bytes_int ),
                [&]( int i ) { scratch( i ) = char( 0 ); } );
            team.team_barrier();

            for ( int s = 0; s < n_pairs; ++s )
            {
                Basis::m2l_pre_cell( team, M, d_src( s ), ops, scratch );
                Basis::m2l_core( team, M, d_src( s ), ops, d_op( s ),
                                 scratch );
            }

            Basis::m2l_post_cell( team, scratch, L, target_cell, ops );
        } );
    Kokkos::fence();
}

//---------------------------------------------------------------------------//
// The PER-PAIR FALLBACK path, driven exactly as
// DownwardSweep::run_m2l_fallback drives it
// (src/Canopy_DownwardSweep.hpp:2331-2347): one team per PAIR, the physical
// offset SOURCE CENTER MINUS TARGET CENTER, the two half-widths, and the aux
// table -- which is the only channel the softening has to this path.
//
// `aux` is built through Basis::build_aux_tables from an M2LKernelParams
// carrying the softening as a LENGTH, so the eps -> b squaring under test is
// the basis's own and not this file's.
//---------------------------------------------------------------------------//
template <class Basis>
void runM2LTranslate( const CoeffView& M, int source_cell,
                      const double d_src_minus_tgt[3], double softening_eps,
                      const CoeffView& L, int target_cell )
{
    using policy = Kokkos::TeamPolicy<TEST_EXECSPACE>;
    using member_t = typename policy::member_type;

    Canopy::M2LKernelParams kp;
    kp.softening = softening_eps;
    auto aux = Basis::template build_aux_tables<TEST_MEMSPACE>(
        Basis::max_order, kp );

    const double dx = d_src_minus_tgt[0];
    const double dy = d_src_minus_tgt[1];
    const double dz = d_src_minus_tgt[2];

    Kokkos::parallel_for(
        "m2l_translate", policy( 1, 1 ),
        KOKKOS_LAMBDA( const member_t& team ) {
            auto L_t =
                Kokkos::subview( L, target_cell, Kokkos::ALL, Kokkos::ALL );
            // Widths are ignored by this basis; distinct non-unit values are
            // passed so a future read of one would be visible rather than
            // silently zero.
            Basis::m2l_translate( team, M, source_cell, dx, dy, dz, 1.0, 2.0,
                                  aux, L_t );
        } );
    Kokkos::fence();
}

//---------------------------------------------------------------------------//
// Load a moment array built by buildMoments into cell `cell` of a coefficient
// view, in the (cell, coeff, comp_set_slot) layout the sweeps use.
//---------------------------------------------------------------------------//
template <class Basis>
void loadMoments( const CoeffView& M, int cell, const std::vector<double>& m,
                  int n_slots )
{
    auto h_M = Kokkos::create_mirror_view( M );
    Kokkos::deep_copy( h_M, M );
    for ( int s = 0; s < n_slots; ++s )
        for ( int c = 0; c < NC; ++c )
            h_M( cell, s, Basis::comp_set_slot( c, 0 ) ) =
                m[static_cast<std::size_t>( s ) * NC + c];
    Kokkos::deep_copy( M, h_M );
}

} // namespace CT2

//---------------------------------------------------------------------------//
// P2M into a child, M2M to the parent, checked two ways.
//
//   (a) AGAINST A DIRECT P2M ABOUT THE PARENT CENTER. The M2M is claimed to
//       be exact, not truncated: M^par_q = sum_{q'<=q} s^{q-q'}/(q-q')! M^ch_q'
//       follows from expanding (y - c_par)^q and every term of that expansion
//       sits at degree <= |q| <= p, so nothing is cut. That makes the moments
//       of the same particles taken directly about the parent center the
//       EXACT reference -- a check the shift formula cannot satisfy by being
//       merely self-consistent.
//
//   (b) THE ROUND TRIP BY -s. The shift operator is unitriangular in the
//       multi-index partial order (q' <= q), so shift(s) . shift(-s) is the
//       identity exactly, truncation included. This is the check the task
//       statement names; (a) is the stronger one and is why P2M is covered
//       here rather than left untested.
//---------------------------------------------------------------------------//
template <int P>
void testM2MShift()
{
    using Basis = Canopy::CartesianTaylorBasis<double, P, CT2::NC>;
    constexpr int Nco = Basis::num_coeffs_per_cell;
    ASSERT_EQ( Nco, CT::num_slots( P ) );

    const auto particles = CT2::buildParticles();
    const double zero[3] = { 0.0, 0.0, 0.0 };
    // A particle at d relative to the child center sits at d + s relative to
    // the parent center, so the direct-about-parent P2M uses offset -s.
    const double minus_s[3] = { -CT2::s_shift[0], -CT2::s_shift[1],
                                -CT2::s_shift[2] };

    // cells: 0 = child, 1 = M2M'd parent, 2 = direct parent, 3 = round trip
    CT2::CoeffView M( "M", 4, Nco, CT2::NC );

    CT2::runP2M<Basis>( M, 0, particles, zero, 1.0 );
    CT2::runP2M<Basis>( M, 2, particles, minus_s, 2.0 );
    CT2::runM2M<Basis>( M, 0, 1, CT2::s_shift );
    CT2::runM2M<Basis>( M, 1, 3, minus_s );

    auto h_M = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), M );

    // The scale the two comparisons are measured against: the largest moment
    // magnitude anywhere in the problem. Relative-to-entry is unusable, since
    // individual moments can vanish for this particle set.
    double scale = 0.0;
    for ( int s = 0; s < Nco; ++s )
        for ( int c = 0; c < CT2::NC; ++c )
            for ( int cell = 0; cell < 3; ++cell )
                scale = std::max( scale, std::abs( h_M( cell, s, c ) ) );
    ASSERT_GT( scale, 0.0 );

    // The P2M itself, against the brute-force monomial. This is what makes
    // (a) a test of M2M rather than of P2M and M2M jointly.
    double worst_p2m = 0.0;
    for ( int s = 0; s < Nco; ++s )
    {
        int k[3];
        CT::inverse_slot( s, k );
        for ( int c = 0; c < CT2::NC; ++c )
        {
            double want = 0.0;
            for ( const auto& pt : particles )
                want += CT2::monoOverFact( pt.d, k ) * pt.q[c];
            worst_p2m =
                std::max( worst_p2m, std::abs( h_M( 0, s, c ) - want ) );
            ASSERT_NEAR( h_M( 0, s, c ), want, 1.0e-13 * scale )
                << "p = " << P << ": p2m_contribution disagrees with the "
                << "brute-force sum_j d_j^q/q! s_jc at q = (" << k[0] << ","
                << k[1] << "," << k[2] << "), component " << c;
        }
    }

    double worst_direct = 0.0;
    double worst_trip = 0.0;
    for ( int s = 0; s < Nco; ++s )
    {
        int k[3];
        CT::inverse_slot( s, k );
        for ( int c = 0; c < CT2::NC; ++c )
        {
            const double d1 = std::abs( h_M( 1, s, c ) - h_M( 2, s, c ) );
            worst_direct = std::max( worst_direct, d1 );
            ASSERT_NEAR( h_M( 1, s, c ), h_M( 2, s, c ), 1.0e-12 * scale )
                << "p = " << P << ": M2M by s = c_child - c_parent disagrees "
                << "with a direct P2M about the parent center at q = (" << k[0]
                << "," << k[1] << "," << k[2] << "), component " << c
                << "; M2M " << h_M( 1, s, c ) << ", direct "
                << h_M( 2, s, c );

            const double d2 = std::abs( h_M( 3, s, c ) - h_M( 0, s, c ) );
            worst_trip = std::max( worst_trip, d2 );
            ASSERT_NEAR( h_M( 3, s, c ), h_M( 0, s, c ), 1.0e-12 * scale )
                << "p = " << P << ": M2M by s then by -s did not return the "
                << "child moments at q = (" << k[0] << "," << k[1] << ","
                << k[2] << "), component " << c;
        }
    }

    std::printf( "[cartesian-taylor] m2m p = %d: worst |diff| / max|M| -- "
                 "p2m vs brute force %.3e, M2M vs direct-about-parent %.3e, "
                 "round trip by -s %.3e\n",
                 P, worst_p2m / scale, worst_direct / scale,
                 worst_trip / scale );
}

//---------------------------------------------------------------------------//
// L2L, checked two ways.
//
//   (a) THE POLYNOMIAL IS UNCHANGED AT A PHYSICAL POINT. Substituting
//       (x - c_par) = (x - c_ch) + s into u(x) = sum_p (x-c_par)^p/p! l^par_p
//       and collecting gives the L2L, and every term of that collection has
//       p <= p' <= p_order, so nothing is cut: the child expansion and the
//       parent expansion are the SAME polynomial, evaluated about different
//       centers. This checks the shift against the thing it is supposed to
//       preserve rather than against itself.
//
//   (b) THE ROUND TRIP BY -s, which the task statement names. The L2L
//       operator is unitriangular the other way (p' >= p) and every
//       intermediate index stays inside the table, so the composition is the
//       identity exactly.
//---------------------------------------------------------------------------//
template <int P>
void testL2LShift()
{
    using Basis = Canopy::CartesianTaylorBasis<double, P, CT2::NC>;
    constexpr int Nco = Basis::num_coeffs_per_cell;
    constexpr int NCS = Basis::num_comp_slots;

    // sets_per_component = 1 is a documented deliberate deviation; if it ever
    // moves, the (component, set) flattening below stops collapsing to c and
    // this body is the first thing that must be revisited.
    ASSERT_EQ( Basis::sets_per_component, 1 );
    ASSERT_EQ( NCS, CT2::NC );

    const double minus_s[3] = { -CT2::s_shift[0], -CT2::s_shift[1],
                                -CT2::s_shift[2] };

    // cells: 0 = parent, 1 = child, 2 = round trip back to the parent center
    CT2::CoeffView L( "L", 3, Nco, NCS );
    auto h_L = Kokkos::create_mirror_view( L );
    double scale = 0.0;
    for ( int s = 0; s < Nco; ++s )
        for ( int c = 0; c < CT2::NC; ++c )
        {
            h_L( 0, s, Basis::comp_set_slot( c, 0 ) ) =
                CT2::syntheticLocal( s, c );
            scale = std::max( scale, std::abs( CT2::syntheticLocal( s, c ) ) );
        }
    Kokkos::deep_copy( L, h_L );

    CT2::runL2L<Basis>( L, 0, 1, CT2::s_shift );
    CT2::runL2L<Basis>( L, 1, 2, minus_s );

    Kokkos::deep_copy( h_L, L );

    double worst_trip = 0.0;
    for ( int s = 0; s < Nco; ++s )
    {
        int k[3];
        CT::inverse_slot( s, k );
        for ( int c = 0; c < CT2::NC; ++c )
        {
            const int cs = Basis::comp_set_slot( c, 0 );
            worst_trip = std::max(
                worst_trip, std::abs( h_L( 2, s, cs ) - h_L( 0, s, cs ) ) );
            ASSERT_NEAR( h_L( 2, s, cs ), h_L( 0, s, cs ), 1.0e-12 * scale )
                << "p = " << P << ": L2L by s then by -s did not return the "
                << "parent locals at p = (" << k[0] << "," << k[1] << ","
                << k[2] << "), component " << c;
        }
    }

    // (a): three probe points, given as offsets a from the CHILD center. The
    // same physical point sits at a + s from the parent center.
    const double probes[3][3] = { { 0.13, -0.27, 0.06 },
                                  { -0.44, 0.19, 0.33 },
                                  { 0.05, 0.05, -0.41 } };
    double worst_poly = 0.0;
    for ( const auto& a : probes )
    {
        const double a_par[3] = { a[0] + CT2::s_shift[0],
                                  a[1] + CT2::s_shift[1],
                                  a[2] + CT2::s_shift[2] };
        double phi_ch[CT2::NC], phi_par[CT2::NC];
        double g[CT2::NC][3];
        CT2::runL2P<Basis>( L, 1, a, phi_ch, g, false );
        CT2::runL2P<Basis>( L, 0, a_par, phi_par, g, false );

        for ( int c = 0; c < CT2::NC; ++c )
        {
            const double u_scale =
                std::max( std::abs( phi_par[c] ), std::abs( phi_ch[c] ) );
            worst_poly =
                std::max( worst_poly, std::abs( phi_ch[c] - phi_par[c] ) );
            ASSERT_NEAR( phi_ch[c], phi_par[c], 1.0e-12 * ( u_scale + scale ) )
                << "p = " << P << ": the L2L-shifted child expansion and the "
                << "parent expansion disagree at the same physical point, "
                << "component " << c << "; a_child = (" << a[0] << "," << a[1]
                << "," << a[2] << ")";
        }
    }

    std::printf( "[cartesian-taylor] l2l p = %d: worst |diff| -- round trip "
                 "by -s %.3e, same-point polynomial %.3e (max|l| = %.3e)\n",
                 P, worst_trip, worst_poly, scale );
}

//---------------------------------------------------------------------------//
// L2P, checked two ways.
//
//   (a) THE POTENTIAL against a brute-force sum_p a^p/p! l_p built from
//       monoOverFact, which shares no code with the basis.
//
//   (b) THE ANALYTIC GRADIENT against a Richardson-extrapolated central
//       difference of the SAME polynomial, evaluated through l2p_evaluate
//       itself with compute_gradient = false.
//
//       The oracle is sharper here than it is for the derivative ladder, and
//       for a reason worth writing down: u is a POLYNOMIAL of degree p, so a
//       central difference carries error h^2/6 u''' + h^4/120 u^(5) + ... in
//       which every term above the degree vanishes identically. One Richardson
//       step kills the h^2 term, so at p <= 5 the extrapolated difference
//       equals the analytic derivative in exact arithmetic and the only
//       residual is cancellation roundoff, which is O(eps |u| / h). The
//       tolerance below is written against that scale rather than guessed.
//       At p > 5 this stops being true and the tolerance would have to be
//       re-measured -- which is why the scale is spelled out instead of a
//       bare constant being pinned.
//---------------------------------------------------------------------------//
template <int P>
void testL2PEvaluation()
{
    using Basis = Canopy::CartesianTaylorBasis<double, P, CT2::NC>;
    constexpr int Nco = Basis::num_coeffs_per_cell;
    constexpr int NCS = Basis::num_comp_slots;

    CT2::CoeffView L( "L", 1, Nco, NCS );
    auto h_L = Kokkos::create_mirror_view( L );
    for ( int s = 0; s < Nco; ++s )
        for ( int c = 0; c < CT2::NC; ++c )
            h_L( 0, s, Basis::comp_set_slot( c, 0 ) ) =
                CT2::syntheticLocal( s, c );
    Kokkos::deep_copy( L, h_L );

    const double probes[3][3] = { { 0.21, -0.34, 0.11 },
                                  { -0.46, 0.08, 0.29 },
                                  { 0.37, 0.42, -0.18 } };

    // The differencing step. Nondimensionalized against the offset scale the
    // probes sit at (order 0.5, a leaf half-width), not fixed absolutely.
    const double h = 0.5 / 8.0;

    double worst_phi = 0.0;
    double worst_grad = 0.0;

    for ( const auto& a : probes )
    {
        double phi[CT2::NC];
        double grad[CT2::NC][3];
        CT2::runL2P<Basis>( L, 0, a, phi, grad, true );

        // (a) the potential.
        for ( int c = 0; c < CT2::NC; ++c )
        {
            double want = 0.0;
            for ( int s = 0; s < Nco; ++s )
            {
                int k[3];
                CT::inverse_slot( s, k );
                want += CT2::monoOverFact( a, k ) * CT2::syntheticLocal( s, c );
            }
            worst_phi = std::max( worst_phi, std::abs( phi[c] - want ) );
            ASSERT_NEAR( phi[c], want, 1.0e-12 * ( std::abs( want ) + 1.0 ) )
                << "p = " << P << ": l2p_evaluate disagrees with the "
                << "brute-force sum_p a^p/p! l_p at component " << c
                << ", a = (" << a[0] << "," << a[1] << "," << a[2] << ")";
        }

        // (b) the gradient.
        for ( int dim = 0; dim < 3; ++dim )
        {
            double d1[CT2::NC], d2[CT2::NC];
            double u_mag = 0.0;
            for ( int step = 0; step < 2; ++step )
            {
                const double hh = ( step == 0 ) ? h : ( 0.5 * h );
                double ap[3] = { a[0], a[1], a[2] };
                double am[3] = { a[0], a[1], a[2] };
                ap[dim] += hh;
                am[dim] -= hh;

                double phip[CT2::NC], phim[CT2::NC];
                double gdummy[CT2::NC][3];
                CT2::runL2P<Basis>( L, 0, ap, phip, gdummy, false );
                CT2::runL2P<Basis>( L, 0, am, phim, gdummy, false );

                for ( int c = 0; c < CT2::NC; ++c )
                {
                    const double dd = ( phip[c] - phim[c] ) / ( 2.0 * hh );
                    if ( step == 0 )
                        d1[c] = dd;
                    else
                        d2[c] = dd;
                    u_mag = std::max(
                        u_mag, std::max( std::abs( phip[c] ),
                                         std::abs( phim[c] ) ) );
                }
            }

            for ( int c = 0; c < CT2::NC; ++c )
            {
                // R(h) = ( 4 D(h/2) - D(h) ) / 3 -- one step, which is all the
                // h^2 term needs; see the comment block above.
                const double want = ( 4.0 * d2[c] - d1[c] ) / 3.0;
                // Cancellation roundoff of a central difference:
                // eps |u| / h, with a 1e3 safety factor over machine eps.
                const double tol = 1.0e3 * 2.22e-16 * u_mag / ( 0.5 * h ) +
                                   1.0e-13 * std::abs( want );
                worst_grad =
                    std::max( worst_grad, std::abs( grad[c][dim] - want ) );
                ASSERT_NEAR( grad[c][dim], want, tol )
                    << "p = " << P << ": l2p_evaluate's ANALYTIC gradient "
                    << "disagrees with a Richardson-extrapolated central "
                    << "difference of the same polynomial, component " << c
                    << ", direction " << dim << ", a = (" << a[0] << ","
                    << a[1] << "," << a[2] << "), h = " << h;
            }
        }
    }

    std::printf( "[cartesian-taylor] l2p p = %d: worst |diff| -- potential "
                 "vs brute force %.3e, analytic gradient vs Richardson FD "
                 "%.3e\n",
                 P, worst_phi, worst_grad );
}

//===========================================================================//
// T3 of tasks/cartesian-taylor-basis.md -- the M2L, with the sign and
// normalization convention pinned. Five bodies, all host math over hand-built
// inputs and no sweep, no tree and no MPI:
//
//   m2l_ell0_closed_forms   l_0 against §2's degree-0/1/2 closed forms. The
//                           discriminator for the (-1)^{|q|} multiplier and
//                           the 1/q! placement (risk R1), both of which first
//                           bite at |q| = 1.
//   m2l_p1_contraction      the |p| = 1 coefficients against the reference's
//                           K/dK/ddK contraction, which is the only check
//                           here that exercises b_k at degree 3.
//   m2l_parity_identity     sum_q (-1)^{|q|} b_{p+q}(R) M_q against
//                           (-1)^{|p|} sum_q b_{p+q}(S) M_q, with R and S
//                           INDEPENDENTLY SOURCED.
//   m2l_fused_vs_fallback   the table path and the per-pair path, EXACTLY
//                           equal (risk R4).
//   m2l_end_to_end          P2M -> M2L -> L2P by hand against a direct
//                           softened sum, against the truncation bound.
//===========================================================================//

//---------------------------------------------------------------------------//
// l_0 = sum_q (-1)^{|q|} b_q(R) M_q against the canopy-questions.md §2 closed
// forms.
//
// THE ORACLE IS §2, hand-coded in this file at referenceClosedForm above, and
// NOT treecode.py, which is not in this repository. §2 stops at |k| = 3, so
// this body runs at p = 2 and p = 3 and no higher.
//
// WHAT IT SEES. Both halves of risk R1 land on the degree-1 term and nowhere
// lower:
//   * dropping the (-1)^{|q|} multiplier flips every odd-|q| term, so the
//     first disagreement is at |q| = 1;
//   * applying the 1/q! a second time in the operator scales every |q| >= 2
//     term, so the first disagreement is at |q| = 2.
// The degree-0 term carries neither, which is why an l_0 check that stopped
// at |q| = 0 would be vacuous.
//
// TOLERANCE. The comparison is against the sum of the term magnitudes,
// sum_q |b_q(R) M_q|, not against |l_0|: terms of opposite sign cancel and a
// relative test against the result would be a test of the cancellation. Both
// sides are the same sum in a different order, so the achieved deviation
// should sit at the roundoff floor of that scale, and it is printed.
//---------------------------------------------------------------------------//
template <int P>
void testM2LEll0ClosedForms()
{
    static_assert( P <= 3,
                   "the §2 closed forms stop at |k| = 3, so l_0 at order p "
                   "needs p <= 3" );

    using Basis = Canopy::CartesianTaylorBasis<double, P, CT2::NC>;
    constexpr int Nco = Basis::num_coeffs_per_cell;

    const auto particles = CT2::buildParticles();
    const auto M = CT2::buildMoments( particles, Nco );
    const auto geoms = CT2::buildM2LGeometries();

    // slot of the zero multi-index -- l_0's row of the operator.
    const int p0 = CT::slot( 0, 0, 0 );

    double worst_rel = 0.0;

    for ( const auto& g : geoms )
    {
        double op[Basis::m2l_op_entries];
        Basis::m2l_operator_block( g.R, g.b, op );

        for ( int c = 0; c < CT2::NC; ++c )
        {
            double got = 0.0;
            double want = 0.0;
            double scale = 0.0;

            for ( int q_slot = 0; q_slot < Nco; ++q_slot )
            {
                const double m = M[static_cast<std::size_t>( q_slot ) *
                                       CT2::NC + c];

                got += op[Basis::m2l_op_index( p0, q_slot )] * m;

                int q[3];
                CT::inverse_slot( q_slot, q );
                const int nq = q[0] + q[1] + q[2];
                const double sign = ( ( nq & 1 ) == 0 ) ? 1.0 : -1.0;
                const double bq = referenceClosedForm( q, g.R, g.b );

                want += sign * bq * m;
                scale += std::abs( bq * m );
            }

            const double tol = 1.0e-13 * scale;
            worst_rel = std::max( worst_rel, std::abs( got - want ) / scale );

            ASSERT_NEAR( got, want, tol )
                << "p = " << P << ": l_0 = sum_q (-1)^{|q|} b_q(R) M_q "
                << "disagrees with the canopy-questions.md §2 closed forms "
                << "(hand-coded in THIS file, not read from any file outside "
                << "this repository). Check the (-1)^{|q|} multiplier, which "
                << "belongs to the operator and first bites at |q| = 1, and "
                << "the 1/q! placement, which belongs to the MOMENT and to "
                << "the L2P and never to b_k. R = (" << g.R[0] << ","
                << g.R[1] << "," << g.R[2] << "), b = " << g.b
                << ", component " << c;
        }
    }

    std::printf( "[cartesian-taylor] m2l l_0 vs §2 closed forms, p = %d: "
                 "worst |got-want| / sum|b_q M_q| = %.3e over %d geometries\n",
                 P, worst_rel, static_cast<int>( geoms.size() ) );
}

//---------------------------------------------------------------------------//
// The |p| = 1 coefficients against the reference's K/dK/ddK contraction.
//
// WHAT THIS IS AND IS NOT. _expansion_batch is the whole far-field
// contribution of one source box AT THE TARGET BOX CENTER with no target-side
// expansion, and it returns a VELOCITY -- so its three arrays are
// -d phi, -dd phi and -ddd phi, and contracted against the degree-0/1/2
// moments they give the SCALAR PASS'S GRADIENT at that center, which is
// l_{e_a}. It is NOT l_0, which needs b_k at degrees 0, 1 and 2 instead.
// Comparing it against l_0 builds an oracle that cannot match and then
// invites "fixing" a correct operator against it -- the plausible-but-wrong
// outcome R1 describes.
//
// THE IDENTITY BEING ASSERTED, derived from canopy-questions.md §4's
// l_p = sum_q (-1)^{|q|} b_{p+q}(R) M_q with p = e_a, term by term:
//
//   |q| = 0 :  + b_{e_a} G            = -K_a G
//   |q| = 1 :  - sum_b b_{e_a+e_b} D_b = + sum_b dK_ab D_b
//   |q| = 2 :  + sum_{|q|=2} b_{e_a+q} M_q
//              = (1/2) sum_{b,c} (d_a d_b d_c phi) Q_bc
//              = -(1/2) sum_{b,c} ddK_abc Q_bc
//
// so    l_{e_a} = -K_a G + sum_b dK_ab D_b - (1/2) sum_{b,c} ddK_abc Q_bc ,
// i.e.  the reference's velocity-shaped contraction
//
//       V_a = K_a G - sum_b dK_ab D_b + (1/2) sum_{b,c} ddK_abc Q_bc
//
// equals MINUS l_{e_a}. THE OVERALL SIGN IS K = -grad phi AND IS STATED HERE
// AND ON THE ASSERTION rather than absorbed silently.
//
// THE 1/2 IS THE MULTINOMIAL FACTOR |q|!/q!, not a convention: Q_bc is a full
// symmetric tensor and sum_{b,c} T_bc d_b d_c = sum_{|q|=2} (2!/q!) T_q d^q.
// Getting it wrong is the other route to the same R1 failure.
//
// p = 2 ONLY. The three arrays carry exactly the degree-0/1/2 moments, so at
// p = 3 l_{e_a} would additionally carry the |q| = 3 term that the reference
// has no array for and the two would legitimately differ.
//---------------------------------------------------------------------------//
void testM2LP1Contraction()
{
    constexpr int P = 2;
    using Basis = Canopy::CartesianTaylorBasis<double, P, CT2::NC>;
    constexpr int Nco = Basis::num_coeffs_per_cell;

    const auto particles = CT2::buildParticles();
    const auto M = CT2::buildMoments( particles, Nco );
    const auto T = CT2::buildTensorMoments( particles );
    const auto geoms = CT2::buildM2LGeometries();

    double worst_rel = 0.0;

    for ( const auto& g : geoms )
    {
        double op[Basis::m2l_op_entries];
        Basis::m2l_operator_block( g.R, g.b, op );

        for ( int a = 0; a < 3; ++a )
        {
            int e_a[3] = { 0, 0, 0 };
            e_a[a] = 1;
            const int p_slot = CT::slot( e_a[0], e_a[1], e_a[2] );

            for ( int c = 0; c < CT2::NC; ++c )
            {
                // l_{e_a}, through the operator -- the quantity under test.
                double ell = 0.0;
                for ( int q_slot = 0; q_slot < Nco; ++q_slot )
                    ell += op[Basis::m2l_op_index( p_slot, q_slot )] *
                           M[static_cast<std::size_t>( q_slot ) * CT2::NC + c];

                // The reference contraction, from the §2-transcribed arrays
                // and the FULL SYMMETRIC TENSOR moments.
                double v = CT2::refKa( g.R, g.b, a ) * T.G[c];
                double scale = std::abs( CT2::refKa( g.R, g.b, a ) * T.G[c] );

                for ( int b = 0; b < 3; ++b )
                {
                    const double t =
                        CT2::refdKab( g.R, g.b, a, b ) * T.D[b][c];
                    v -= t;
                    scale += std::abs( t );
                }

                for ( int b = 0; b < 3; ++b )
                    for ( int d = 0; d < 3; ++d )
                    {
                        const double t =
                            0.5 * CT2::refddKabc( g.R, g.b, a, b, d ) *
                            T.Q[b][d][c];
                        v += t;
                        scale += std::abs( t );
                    }

                const double tol = 1.0e-13 * scale;
                worst_rel =
                    std::max( worst_rel, std::abs( -ell - v ) / scale );

                ASSERT_NEAR( -ell, v, tol )
                    << "p = 2: the |p| = 1 local coefficient disagrees with "
                    << "the reference's K/dK/ddK contraction. THE OVERALL "
                    << "SIGN IS K = -grad phi, so the reference's velocity "
                    << "equals MINUS l_{e_a} and that is what is asserted "
                    << "here. The arrays are transcribed from "
                    << "canopy-questions.md §2 (minus the §2 tensor, shifted "
                    << "by one index) and the 1/2 on the Q term is the "
                    << "multinomial |q|!/q! between full symmetric tensors "
                    << "and multi-indexed moments, not a convention. "
                    << "direction a = " << a << ", R = (" << g.R[0] << ","
                    << g.R[1] << "," << g.R[2] << "), b = " << g.b
                    << ", component " << c;
            }
        }
    }

    std::printf( "[cartesian-taylor] m2l |p|=1 vs reference K/dK/ddK "
                 "contraction, p = 2: worst |got-want| / sum|terms| = %.3e "
                 "over %d geometries\n",
                 worst_rel, static_cast<int>( geoms.size() ) );
}

//---------------------------------------------------------------------------//
// The parity identity, with its two arguments INDEPENDENTLY SOURCED.
//
// phi is even, so b_n(-R) = (-1)^{|n|} b_n(R) and the two spellings
//
//     l_p = sum_q (-1)^{|q|} b_{p+q}(R) M_q          R = c_target - c_source
//     l_p = (-1)^{|p|} sum_q b_{p+q}(S) M_q          S = -R
//
// must agree. THE EQUALITY IS AN IDENTITY IN THE VECTOR FED TO IT: handing
// both spellings the same vector makes them agree whatever its sign, so a
// check that obtained S by negating the very R the first spelling used would
// be VACUOUS and would pass over a wrong-signed operator. The two arguments
// are therefore sourced independently:
//
//   * the first from the PRODUCTION key-to-R path -- build_m2l_operators,
//     handed real keys and a real unit_w table, which is where
//     R = -(ii,jj,kk) * unit_w[max_d] is spelled;
//   * the second from the key's RAW offset S = (ii,jj,kk) * unit_w[max_d],
//     source minus target exactly as src/Canopy_DownwardSweep.hpp:1464-1481
//     builds it, fed to T1's derivative_ladder directly, with the (-1)^{|p|}
//     applied by this body.
//
// So sourced, a missing negation in build_m2l_operators makes the two
// disagree, and disagreement otherwise means the sign or the parity has been
// applied twice.
//---------------------------------------------------------------------------//
void testM2LParityIdentity()
{
    constexpr int P = 2;
    using Basis = Canopy::CartesianTaylorBasis<double, P, CT2::NC>;
    constexpr int Nco = Basis::num_coeffs_per_cell;
    constexpr int Ns = Basis::m2l_num_src_coeffs;

    const auto particles = CT2::buildParticles();
    const auto M = CT2::buildMoments( particles, Nco );

    // The half-width at each depth, w_root / 2^d, with a root half-width of
    // 1.0 -- a power of two, so every unit_w and every product below is exact
    // in binary64 and the identity is tested on the numbers, not on the
    // rounding.
    const int n_levels = 4;
    std::vector<double> unit_w( n_levels );
    for ( int d = 0; d < n_levels; ++d )
        unit_w[d] = 1.0 / static_cast<double>( 1 << d );

    // The softening as a LENGTH; the kernel's b is its square.
    const double eps = 0.025;
    const double bb = eps * eps;
    Canopy::M2LKernelParams kp;
    kp.softening = eps;

    // Keys spanning both signs on every axis, several depths, and one pair
    // differing only in dd.
    const std::vector<CT2::M2LProbeKey> keys = {
        { 2, 0, 3, -4, 2 },  { 2, 3, 3, -4, 2 },   { 3, 0, -5, 1, 4 },
        { 1, -1, 4, 0, -3 }, { 0, 0, 2, -2, 1 },   { 3, 2, -6, 5, -2 },
        { 2, 1, -2, -3, -4 } };
    const int nk = static_cast<int>( keys.size() );

    auto aux =
        Basis::template build_aux_tables<Kokkos::HostSpace>( P, kp );

    // WithoutInitializing, as the sweep allocates it
    // (src/Canopy_DownwardSweep.hpp) -- so an unwritten entry is garbage and
    // not a zero that could pass by accident.
    typename Basis::template m2l_operators_type<Kokkos::HostSpace> ops(
        Kokkos::view_alloc( Kokkos::WithoutInitializing, "m2l_ops" ), Nco, Ns,
        nk );

    Basis::build_m2l_operators( keys.data(), nk, unit_w.data(), n_levels, kp,
                                aux, ops );

    std::vector<double> bk( CT::num_slots( 2 * P ) );
    double worst_rel = 0.0;

    for ( int j = 0; j < nk; ++j )
    {
        // S -- the key's RAW offset, source minus target. NOT obtained by
        // negating anything the operator used.
        const double S[3] = {
            static_cast<double>( keys[j].ii ) * unit_w[keys[j].max_d],
            static_cast<double>( keys[j].jj ) * unit_w[keys[j].max_d],
            static_cast<double>( keys[j].kk ) * unit_w[keys[j].max_d] };

        CT::derivative_ladder( S, bb, 2 * P, bk.data() );

        for ( int p_slot = 0; p_slot < Nco; ++p_slot )
        {
            int p[3];
            CT::inverse_slot( p_slot, p );
            const int np = p[0] + p[1] + p[2];
            const double par = ( ( np & 1 ) == 0 ) ? 1.0 : -1.0;

            for ( int c = 0; c < CT2::NC; ++c )
            {
                double lhs = 0.0;
                double rhs = 0.0;
                double scale = 0.0;

                for ( int q_slot = 0; q_slot < Ns; ++q_slot )
                {
                    const double m =
                        M[static_cast<std::size_t>( q_slot ) * CT2::NC + c];

                    lhs += ops( p_slot, q_slot, j ) * m;

                    int q[3];
                    CT::inverse_slot( q_slot, q );
                    const double bpq =
                        bk[CT::slot( p[0] + q[0], p[1] + q[1],
                                     p[2] + q[2] )];
                    rhs += par * bpq * m;
                    scale += std::abs( bpq * m );
                }

                const double tol = 1.0e-13 * scale;
                worst_rel = std::max( worst_rel,
                                      std::abs( lhs - rhs ) / scale );

                ASSERT_NEAR( lhs, rhs, tol )
                    << "the parity identity fails at |p| = " << np
                    << ". The left side comes from the PRODUCTION key-to-R "
                    << "path, build_m2l_operators, where "
                    << "R = -(ii,jj,kk) * unit_w[max_d]; the right side from "
                    << "the key's RAW source-minus-target offset S fed to "
                    << "derivative_ladder with the (-1)^{|p|} applied here. "
                    << "Disagreement means the NEGATION OF R is missing, or "
                    << "that the sign or the parity has been applied twice. "
                    << "key " << j << " = (max_d " << keys[j].max_d << ", dd "
                    << keys[j].dd << ", " << keys[j].ii << "," << keys[j].jj
                    << "," << keys[j].kk << "), component " << c;
            }
        }
    }

    // The documented consequence of a key-independent operator: keys 0 and 1
    // differ only in dd, and this basis's operator has NO dd dependence, so
    // their columns are identical. Duplication in the table, not an error.
    for ( int p_slot = 0; p_slot < Nco; ++p_slot )
        for ( int q_slot = 0; q_slot < Ns; ++q_slot )
            ASSERT_EQ( ops( p_slot, q_slot, 0 ), ops( p_slot, q_slot, 1 ) )
                << "keys 0 and 1 differ only in dd, and this basis's operator "
                << "is a function of the physical (R, b) alone, so their "
                << "columns must be bit-identical. A difference means "
                << "something key-dependent leaked into the operator.";

    std::printf( "[cartesian-taylor] m2l parity identity, p = 2: worst "
                 "|lhs-rhs| / sum|b_{p+q} M_q| = %.3e over %d keys\n",
                 worst_rel, nk );
}

//---------------------------------------------------------------------------//
// The fused table path and the per-pair fallback path, EXACTLY equal.
//
// WHY EXACT AND NOT MERELY CLOSE. Both paths reach one operator function,
// walk q in ascending slot order through one taylor_accumulate, and add a
// single named local into a zero-initialized destination -- so for the same
// pair they produce bit-identical coefficients. That is what makes a non-zero
// total_fallback_pair_count() harmless; if the two disagreed, WHICH pairs
// overflow the operator-table cap would decide the answer (risk R4).
//
// THE GEOMETRY IS BUILT SO THE TWO R ARE BIT-IDENTICAL, which is what makes
// "agree exactly" achievable rather than aspirational. The table path builds
// R = -(ii,jj,kk) * unit_w[max_d]; m2l_translate builds R = -(c_s - c_t) from
// two cell centers. Those are the same value mathematically but not
// necessarily the same double -- real cell centers come from repeated halving
// off the root center (src/Canopy_TreeBuilder.hpp:289-297), so c_s - c_t need
// not round to ii * unit_w[max_d] exactly for an arbitrary root center. Here
// the root half-width is 1.0, a power of two, and both centers are exact
// dyadic multiples of unit_w[3] = 0.125, so every product and the subtraction
// are exact and ASSERT_EQ is the right assertion. A case with a non-dyadic
// center would have to assert to round-off and say so.
//---------------------------------------------------------------------------//
void testM2LFusedVsFallback()
{
    constexpr int P = 2;
    using Basis = Canopy::CartesianTaylorBasis<double, P, CT2::NC>;
    constexpr int Nco = Basis::num_coeffs_per_cell;
    constexpr int Ns = Basis::m2l_num_src_coeffs;
    constexpr int NCS = Basis::num_comp_slots;

    const int n_levels = 4;
    std::vector<double> unit_w( n_levels );
    for ( int d = 0; d < n_levels; ++d )
        unit_w[d] = 1.0 / static_cast<double>( 1 << d );

    const double eps = 0.025;
    Canopy::M2LKernelParams kp;
    kp.softening = eps;

    // Every component an exact multiple of unit_w[n_levels-1] = 0.125.
    const double c_t[3] = { 0.25, -0.5, 0.75 };

    const std::vector<CT2::M2LProbeKey> keys = {
        { 2, 0, 3, -4, 2 },  { 3, 0, -5, 1, 4 }, { 1, -1, 4, 0, -3 },
        { 0, 0, 2, -2, 1 },  { 3, 2, -6, 5, -2 } };
    const int nk = static_cast<int>( keys.size() );

    auto aux = Basis::template build_aux_tables<Kokkos::HostSpace>( P, kp );

    typename Basis::template m2l_operators_type<Kokkos::HostSpace> ops_h(
        Kokkos::view_alloc( Kokkos::WithoutInitializing, "m2l_ops_h" ), Nco,
        Ns, nk );
    Basis::build_m2l_operators( keys.data(), nk, unit_w.data(), n_levels, kp,
                                aux, ops_h );

    // Host cache -> device table, the shape DownwardSweep uses
    // (src/Canopy_DownwardSweep.hpp:621 and :630).
    typename Basis::template m2l_operators_type<TEST_MEMSPACE> ops(
        Kokkos::view_alloc( Kokkos::WithoutInitializing, "m2l_ops" ), Nco, Ns,
        nk );
    Kokkos::deep_copy( ops, ops_h );

    const auto particles = CT2::buildParticles();
    const auto M_ref = CT2::buildMoments( particles, Nco );

    // Source moments live in cells 0..nk-1 so a multi-pair run has distinct
    // sources; every one carries the same moments, which is what lets the
    // per-pair comparison below isolate the operator.
    CT2::CoeffView M( "M", nk, Nco, NCS );
    for ( int j = 0; j < nk; ++j )
        CT2::loadMoments<Basis>( M, j, M_ref, Nco );

    double worst_pair = 0.0;

    for ( int j = 0; j < nk; ++j )
    {
        const double w = unit_w[keys[j].max_d];
        // c_source = c_target + (ii,jj,kk) * unit_w[max_d]. Exact: every
        // term is a dyadic multiple of 0.125.
        const double d_src_minus_tgt[3] = {
            static_cast<double>( keys[j].ii ) * w,
            static_cast<double>( keys[j].jj ) * w,
            static_cast<double>( keys[j].kk ) * w };
        const double c_s[3] = { c_t[0] + d_src_minus_tgt[0],
                                c_t[1] + d_src_minus_tgt[1],
                                c_t[2] + d_src_minus_tgt[2] };
        const double dd[3] = { c_s[0] - c_t[0], c_s[1] - c_t[1],
                               c_s[2] - c_t[2] };

        // The premise of ASSERT_EQ below: the offset the fallback path is
        // handed is the SAME DOUBLE the table path's key arithmetic produces.
        for ( int a = 0; a < 3; ++a )
            ASSERT_EQ( dd[a], d_src_minus_tgt[a] )
                << "the test geometry is not exactly dyadic: c_s - c_t does "
                << "not reproduce (ii,jj,kk) * unit_w[max_d] bit for bit, so "
                << "the two paths' R differ and an exact comparison is not "
                << "the right assertion. axis " << a << ", key " << j;

        CT2::CoeffView L_fused( "L_fused", 1, Nco, NCS );
        CT2::runM2LFused<Basis>( M, { j }, { j }, ops, L_fused, 0 );

        CT2::CoeffView L_fb( "L_fb", 1, Nco, NCS );
        CT2::runM2LTranslate<Basis>( M, j, dd, eps, L_fb, 0 );

        auto h_fused = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), L_fused );
        auto h_fb =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), L_fb );

        for ( int s = 0; s < Nco; ++s )
            for ( int cs = 0; cs < NCS; ++cs )
            {
                worst_pair = std::max(
                    worst_pair, std::abs( h_fused( 0, s, cs ) -
                                          h_fb( 0, s, cs ) ) );
                ASSERT_EQ( h_fused( 0, s, cs ), h_fb( 0, s, cs ) )
                    << "the fused (table) M2L path and the per-pair "
                    << "m2l_translate fallback disagree for ONE pair, and "
                    << "the geometry is exactly dyadic so they must agree "
                    << "BIT FOR BIT. If they do not, which pairs overflow the "
                    << "operator-table cap decides the answer (risk R4). "
                    << "key " << j << " = (max_d " << keys[j].max_d << ", "
                    << keys[j].ii << "," << keys[j].jj << "," << keys[j].kk
                    << "), slot " << s << ", comp_set_slot " << cs;
            }
    }

    // Multi-pair: three sources into one target. The fused path accumulates
    // in team scratch across pairs and flushes once; the fallback path
    // atomically adds per pair. Same pair order, same arithmetic, so still
    // exact -- and this is the part of the accumulator contract a
    // single-pair comparison cannot reach.
    {
        const int n_pairs = 3;
        CT2::CoeffView L_fused( "L_fused_multi", 1, Nco, NCS );
        CT2::runM2LFused<Basis>( M, { 0, 1, 2 }, { 0, 1, 2 }, ops, L_fused,
                                 0 );

        CT2::CoeffView L_fb( "L_fb_multi", 1, Nco, NCS );
        for ( int j = 0; j < n_pairs; ++j )
        {
            const double w = unit_w[keys[j].max_d];
            const double dd[3] = { static_cast<double>( keys[j].ii ) * w,
                                   static_cast<double>( keys[j].jj ) * w,
                                   static_cast<double>( keys[j].kk ) * w };
            CT2::runM2LTranslate<Basis>( M, j, dd, eps, L_fb, 0 );
        }

        auto h_fused = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), L_fused );
        auto h_fb =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), L_fb );

        for ( int s = 0; s < Nco; ++s )
            for ( int cs = 0; cs < NCS; ++cs )
                ASSERT_EQ( h_fused( 0, s, cs ), h_fb( 0, s, cs ) )
                    << "the two M2L paths disagree once THREE pairs "
                    << "accumulate into one target. Single pairs agreeing "
                    << "and three not is an accumulator-order difference, "
                    << "not an operator difference: m2l_core adds one named "
                    << "local per (p, c) into the scratch and m2l_post_cell "
                    << "flushes once, which must match the fallback's "
                    << "per-pair atomic add in the same order. slot " << s
                    << ", comp_set_slot " << cs;
    }

    std::printf( "[cartesian-taylor] m2l fused vs fallback, p = 2: worst "
                 "|fused - fallback| = %.3e over %d keys (exact equality "
                 "asserted) + a 3-pair accumulation\n",
                 worst_pair, nk );
}

//---------------------------------------------------------------------------//
// P2M -> M2L -> L2P by hand, against a direct softened sum. No FMM, no tree,
// no sweep -- just the basis's own operators run in sequence over one source
// box and one target box.
//
// THIS IS THE FIRST NUMBER THAT SAYS THE BASIS COMPUTES THE RIGHT FIELD
// rather than a self-consistent one: every check above compares two
// expressions built from the same b_k, while this one compares against
// sum_j s_j / sqrt(|x - y_j|^2 + b) with no expansion in it at all.
//
// THE BOUND. Taylor truncation at order p goes as (c W / R)^{p+1}. W IS THE
// BOX HALF-WIDTH here -- stated because the achieved c is only interpretable
// against that choice, and the alternative (full width) would halve it. R is
// the center-to-center separation. The sources sit inside |d| <= 0.48 and the
// probes inside |a| <= 0.48 of a half-width 0.5 box, so the configuration is
// a nearly worst-case one for its W rather than a comfortable interior
// sample.
//
// The achieved c is PRINTED at every separation, which is the number T4 needs
// to know how much margin the 1e-3 accuracy bar has at p = 2.
//---------------------------------------------------------------------------//
void testM2LEndToEnd()
{
    constexpr int P = 2;
    using Basis = Canopy::CartesianTaylorBasis<double, P, CT2::NC>;
    constexpr int Nco = Basis::num_coeffs_per_cell;
    constexpr int NCS = Basis::num_comp_slots;

    // Both boxes: HALF-WIDTH 0.5, i.e. full width 1.0.
    const double Wh = 0.5;
    const double eps = 0.025;   // softening LENGTH
    const double bb = eps * eps;

    const double c_s[3] = { 0.0, 0.0, 0.0 };
    const double zero[3] = { 0.0, 0.0, 0.0 };

    // A generic unit direction: no component of R vanishes, so no term of any
    // odd-degree tensor drops out of the comparison.
    const double u[3] = { 0.36, -0.48, 0.80 };

    // R / W = 8, 16, 32. Canopy's MAC at theta = 0.5 admits R^2 theta^2 >
    // 3 (w_a + w_b)^2, i.e. R > 2 sqrt(3) (w_a + w_b) = 6.93 W here, so all
    // three are admissible and the first is close to the admissibility edge.
    const double seps[] = { 4.0, 8.0, 16.0 };

    // The pinned constant of the bound. MEASURED, not guessed: the achieved
    // c printed below is 1.062, 0.987 and 0.939 at R/W = 8, 16 and 32
    // (flux f3YeTspuFk3q), so 1.25 sits about 18% above the worst of them and
    // leaves a factor 1.6 to 2.4 of margin on the error itself. Tight enough
    // that a field wrong by a factor of two fails it; loose enough that it is
    // a bound and not a pinned digit. Raising it to accommodate a failure
    // would silently change what this body means -- re-measure instead, and
    // record the new figures in the T3 entry of
    // tasks/cartesian-taylor-basis-progress-log.md.
    const double c_bound = 1.25;

    const auto particles = CT2::buildParticles();

    const double probes[4][3] = { { 0.21, -0.34, 0.11 },
                                  { -0.46, 0.08, 0.29 },
                                  { 0.37, 0.42, -0.18 },
                                  { -0.12, -0.44, -0.40 } };

    for ( double sep : seps )
    {
        const double c_t[3] = { c_s[0] + sep * u[0], c_s[1] + sep * u[1],
                                c_s[2] + sep * u[2] };

        // P2M about the source center.
        CT2::CoeffView M( "M_e2e", 1, Nco, NCS );
        CT2::runP2M<Basis>( M, 0, particles, zero, Wh );

        // M2L: the per-pair path, handed SOURCE CENTER MINUS TARGET CENTER
        // exactly as src/Canopy_DownwardSweep.hpp:2338-2340 computes it.
        CT2::CoeffView L( "L_e2e", 1, Nco, NCS );
        const double d_src_minus_tgt[3] = { c_s[0] - c_t[0], c_s[1] - c_t[1],
                                            c_s[2] - c_t[2] };
        CT2::runM2LTranslate<Basis>( M, 0, d_src_minus_tgt, eps, L, 0 );

        double worst_abs = 0.0;
        double scale = 0.0;

        for ( const auto& a : probes )
        {
            double got[CT2::NC];
            double grad[CT2::NC][3];
            CT2::runL2P<Basis>( L, 0, a, got, grad, false );

            const double x[3] = { c_t[0] + a[0], c_t[1] + a[1],
                                  c_t[2] + a[2] };
            double want[CT2::NC];
            CT2::directSum( particles, c_s, x, bb, want );

            for ( int c = 0; c < CT2::NC; ++c )
            {
                worst_abs = std::max( worst_abs, std::abs( got[c] - want[c] ) );
                scale = std::max( scale, std::abs( want[c] ) );
            }
        }

        const double rel = worst_abs / scale;
        const double ratio = Wh / sep;
        const double bound = std::pow( c_bound * ratio, P + 1 );
        const double achieved_c =
            std::pow( rel, 1.0 / static_cast<double>( P + 1 ) ) / ratio;

        std::printf( "[cartesian-taylor] m2l end-to-end p = 2: R/W = %4.1f "
                     "(W = HALF-WIDTH %.3f, R = %.3f, eps = %.3f) rel err "
                     "%.4e, bound (%.2f W/R)^3 = %.4e, achieved c = %.3f\n",
                     sep / Wh, Wh, sep, eps, rel, c_bound, bound, achieved_c );

        ASSERT_LT( rel, bound )
            << "P2M -> M2L -> L2P misses the direct softened sum by more "
            << "than the truncation bound (c W / R)^{p+1} at p = 2. W IS THE "
            << "BOX HALF-WIDTH (" << Wh << "), not the full width -- the "
            << "achieved c is only interpretable against that choice. "
            << "R = " << sep << ", R/W = " << ( sep / Wh ) << ", achieved "
            << "c = " << achieved_c << " against the pinned c = " << c_bound
            << ". A miss here with every convention check above passing "
            << "points at the index map above |k| = 3 (risk R2) rather than "
            << "at the sign or the factorials.";
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//

TEST( cartesian_taylor, index_map_bijection ) { testIndexMapBijection(); }

TEST( cartesian_taylor, closed_forms ) { testClosedForms(); }

TEST( cartesian_taylor, finite_difference ) { testFiniteDifference(); }

TEST( cartesian_taylor, m2m_shift )
{
    testM2MShift<2>();
    testM2MShift<4>();
}

TEST( cartesian_taylor, l2l_shift )
{
    testL2LShift<2>();
    testL2LShift<4>();
}

TEST( cartesian_taylor, l2p_evaluation )
{
    testL2PEvaluation<2>();
    testL2PEvaluation<4>();
}

TEST( cartesian_taylor, m2l_ell0_closed_forms )
{
    testM2LEll0ClosedForms<2>();
    testM2LEll0ClosedForms<3>();
}

TEST( cartesian_taylor, m2l_p1_contraction ) { testM2LP1Contraction(); }

TEST( cartesian_taylor, m2l_parity_identity ) { testM2LParityIdentity(); }

TEST( cartesian_taylor, m2l_fused_vs_fallback ) { testM2LFusedVsFallback(); }

TEST( cartesian_taylor, m2l_end_to_end ) { testM2LEndToEnd(); }

//---------------------------------------------------------------------------//
// COMPILE-ONLY -- Solver instantiates on this basis, T2 step 6.
//
// The sizeof() is what makes this a test rather than a spelling exercise.
// Naming the type instantiates nothing; requiring it to be COMPLETE
// instantiates the class body, hence its data members, hence UpwardSweep,
// DownwardSweep and P2P on CartesianTaylorBasis -- so all six class-scope
// sweep guards actually run against this basis:
//
//   sizeof(coeff_type) == scalars_per_coeff * sizeof(component_scalar_type)
//                                      src/Canopy_UpwardSweep.hpp:74-79
//   agreement with detail::coeff_traits src/Canopy_UpwardSweep.hpp:85-93
//   the sizeof relation again           src/Canopy_DownwardSweep.hpp:124-129
//   the coeff_traits agreement again    src/Canopy_DownwardSweep.hpp:135-143
//   sets_per_component >= 1             src/Canopy_DownwardSweep.hpp:154-158
//   m2l_overflow_policy == PerPairTranslate
//                                       src/Canopy_DownwardSweep.hpp:572-581
//
// WHAT THIS DOES NOT COVER, and the reason it is not a defect in the test:
// member function bodies are compiled only for instantiations something
// calls, so Solver::solve() is NOT instantiated here (risk R7). A clean pass
// is evidence about the DECLARATIONS only. Expect T4's first build failures
// inside src/Canopy_Solver.hpp rather than in the basis, and read them as
// expected.
//
// P_ORDER = 2 and NComps = 3 is the shape T4 and the downstream solver use.
// NComps = 1 is added beside it because it costs nothing and is the shape the
// existing Solver call sites use.
//
// TEST_MEMSPACE / TEST_EXECSPACE are the macros a SERIAL unit test has
// (cmake/test_harness/TestSERIAL_Category.hpp:16-17).
//---------------------------------------------------------------------------//
TEST( cartesian_taylor, solver_instantiates )
{
    using Solver3 =
        Canopy::Solver<TEST_MEMSPACE, TEST_EXECSPACE, double, /*P_ORDER=*/2,
                       /*NComps=*/3, Canopy::CartesianTaylorBasis>;

    static_assert( sizeof( Solver3 ) > 0,
                   "Solver did not instantiate on CartesianTaylorBasis at "
                   "P_ORDER = 2, NComps = 3." );
    static_assert(
        std::is_same_v<typename Solver3::kernel_type,
                       Canopy::CartesianTaylorBasis<double, 2, 3>>,
        "Solver::kernel_type is not FarField<Scalar, P_ORDER, NComps>." );

    using Solver1 =
        Canopy::Solver<TEST_MEMSPACE, TEST_EXECSPACE, double, /*P_ORDER=*/2,
                       /*NComps=*/1, Canopy::CartesianTaylorBasis>;

    static_assert( sizeof( Solver1 ) > 0,
                   "Solver did not instantiate on CartesianTaylorBasis at "
                   "P_ORDER = 2, NComps = 1." );
    static_assert(
        std::is_same_v<typename Solver1::kernel_type,
                       Canopy::CartesianTaylorBasis<double, 2, 1>>,
        "Solver::kernel_type is not FarField<Scalar, P_ORDER, NComps>." );

    SUCCEED();
}

} // namespace CartesianTaylorTest
