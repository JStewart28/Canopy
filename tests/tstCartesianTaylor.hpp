#include "Canopy_CartesianTaylorBasis.hpp"

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
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

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//

TEST( cartesian_taylor, index_map_bijection ) { testIndexMapBijection(); }

TEST( cartesian_taylor, closed_forms ) { testClosedForms(); }

TEST( cartesian_taylor, finite_difference ) { testFiniteDifference(); }

} // namespace CartesianTaylorTest
