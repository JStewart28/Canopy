/****************************************************************************
 * Copyright (c) 20125 by the Canopy authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Canopy library. Canopy is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Canopy_Kernels.hpp
  \brief Computational kernels for data manipluation. Assumes positions
  are the first tuple element in any data AoSoA.
*/

#ifndef CANOPY_KERNELS_HPP
#define CANOPY_KERNELS_HPP

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <memory>
#include <cmath>

#include <limits>

namespace Canopy
{

namespace Kernel
{

//---------------------------------------------------------------------------//
/**
 * Compute the complex spherical harmonic (complex, condon-shortley phase)
 * Y_(n, m) (theta, phi)
 * Where:
 *  theta is the polar angle 
 *  phi is the azimuthal angle
 * in spherical coorindates
 * Per equation 3.32 in source 4.
 */
KOKKOS_INLINE_FUNCTION
Kokkos::complex<double> Ynm(int n, int m, double theta, double phi)
{
    using cdouble = Kokkos::complex<double>;
    constexpr auto pi = Kokkos::numbers::pi_v<double>;

    // Work with |m|
    int mp = Kokkos::abs(m);
    double x = Kokkos::cos(theta);

    // Associated Legendre polynomial P_n^m(cosθ)
    double Pnm = std::assoc_legendre(n, mp, x);

    // Normalization factor
    double norm = Kokkos::sqrt( ((2.0*n+1)/(4.0*pi)) *
                                Kokkos::tgamma(n-mp+1) / Kokkos::tgamma(n+mp+1) );

    // Base Y_n^m for m >= 0
    cdouble y = norm * Pnm * Kokkos::polar(1.0, double(mp) * phi);

    if (m < 0)
    {
        // Relation: Y_n^{-m} = (-1)^m * conj(Y_n^m)
        return Kokkos::pow(-1.0, mp) * Kokkos::conj(y);
    }
    return y;
}

// Cartesian to spherical coorindates: (x, y, z) -> (r,theta,phi)
KOKKOS_INLINE_FUNCTION
void cart2sph(double x, double y, double z, double &r, double &theta, double &phi)
{
    r = Kokkos::sqrt(x*x + y*y + z*z);
    theta = (r==0.0 ? 0.0 : Kokkos::acos(z/r)); // polar angle
    phi = Kokkos::atan2(y, x); // azimuth
}

/**
 * Assumes slice 0 of AoSoAType are the cartesian cooridnates of the point.
 */
template <class MemorySpace, class ExecutionSpace>
struct Scalar {
public:
    using memory_space   = MemorySpace;
    using execution_space = ExecutionSpace;
    using cdouble        = Kokkos::complex<double>;

    Scalar(int p)
        : _p(p)
    {}

    int _p;

    /**
     * Compute offset into flattened multipole array
     * (n,m) ↦ index
     */
    KOKKOS_INLINE_FUNCTION
    int index(int n, int m) const {
        return n*n + (m + n);
    }

    /**
     * Compute multipole coefficients M[n][m]
     * up to order p around expansion_center.
     *
     * Definition:
     *   M_n^m = Σ_i q_i * ρ_i^n * conj( Y_n^m(α_i,β_i) )
     *
     * where (ρ_i,α_i,β_i) are spherical coords of point i
     * relative to expansion_center.
     */
    template <class PositionArray, class ScalarArray>
    Kokkos::View<cdouble*, memory_space>
    operator()(const PositionArray& pos,
               const ScalarArray& scalar,
               std::size_t k,
               const Kokkos::Array<double, 3>& expansion_center) const
    {
        int p = _p;
        Kokkos::View<cdouble*, memory_space> M("M", (p+1)*(p+1));
        Kokkos::deep_copy(M, cdouble(0.0));

        // Compute coefficients on host (serial) for now
        // (can parallelize if needed)
        for (std::size_t i = 0; i < k; ++i)
        {
            double dx = pos(i,0) - expansion_center[0];
            double dy = pos(i,1) - expansion_center[1];
            double dz = pos(i,2) - expansion_center[2];

            double rho, alpha, beta;
            cart2sph(dx, dy, dz, rho, alpha, beta);

            for (int n = 0; n <= p; ++n)
            {
                for (int m = -n; m <= n; ++m)
                {
                    int idx = index(n,m);
                    M(idx) += scalar(i) *
                              Kokkos::pow(rho, n) *
                              Kokkos::conj(Ynm(n, m, alpha, beta));
                }
            }
        }
        return M;
    }
};




} // end namespace Kernel

} // end namespace Canopy

#endif // CANOPY_KERNELS_HPP
