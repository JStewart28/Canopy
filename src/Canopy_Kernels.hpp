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

#include <limits>

namespace Canopy
{

namespace Kernel
{

//---------------------------------------------------------------------------//
using cdouble = std::complex<double>;
const double PI = 3.141592653589793;

// Spherical harmonics (complex, condon-shortley phase)
/**
 * Compute the complex spherical harmonic Y_(n, m) (theta, phi)
 * Where:
 *  theta is the polar angle 
 *  phi is the azimuthal angle
 * in spherical coorindates
 * Per equation 3.32 in source 4.
 */
cdouble Ynm(int n, int m, double theta, double phi)
{
    // Associated Legendre P_n^m using std::assoc_legendre (C++17)
    // NOTE: std::assoc_legendre(n, m, x) with x=cos(theta)
    double x = std::cos(theta);
    double Pnm = std::assoc_legendre(n, std::abs(m), x);

    // Normalization term
    double norm = std::sqrt( ((2.0*n+1)/(4.0*PI)) *
                             std::tgamma(n-std::abs(m)+1)/std::tgamma(n+std::abs(m)+1) );
    
                             // e^(i m phi)
    cdouble eimphi = std::polar(1.0, m*phi);

    if (m < 0) {
        // Condon–Shortley phase: Y_n^{-m} = (-1)^m conj(Y_n^m)
        return std::pow(-1.0, m) * std::conj(norm * Pnm * eimphi);
    }
    return norm * Pnm * eimphi;
}



} // end namespace Kernel

} // end namespace Canopy

#endif // CANOPY_KERNELS_HPP
