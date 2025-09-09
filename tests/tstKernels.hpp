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

#include <Canopy_Kernels.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//

/**
 * Test that scalar kernels are correctly calculated
 */
void scalarKernel()
{
    // Create points and q (scalar value)
    const int num_points = 20;
    Kokkos::View<double*[3], TEST_MEMSPACE> cart_coords("cart_coords", num_points);
    Kokkos::View<double*, TEST_MEMSPACE> q("q", num_points);

    using RandomPool = Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE>;
    RandomPool rand_pool(12345); // Seed the random number generator
    Kokkos::parallel_for("populate_cart_coords", Kokkos::RangePolicy<TEST_EXECSPACE>(0, num_points),
        KOKKOS_LAMBDA(const int i) {
            auto rand_gen = rand_pool.get_state();

            // Generate random numbers between -1.0 and 1.0
            cart_coords(i, 0) = rand_gen.drand() * 2.0 - 1.0;
            cart_coords(i, 1) = rand_gen.drand() * 2.0 - 1.0;
            cart_coords(i, 2) = rand_gen.drand() * 2.0 - 1.0;

            // Set charge value to 1.5*(x_pos)
            q(i) = cart_coords(i, 0) * 1.5;

            rand_pool.free_state(rand_gen);
        });

    Kokkos::fence();

    // Expand around (0.1, -0.1, 0.1)
    Kokkos::Array<double, 3> expansion_center = {0.1, -0.1, 0.1};

    // Set truncation degree for moments of expansion
    int p = 2;

    // Create scalar kernel
    Canopy::Kernel::Scalar<TEST_MEMSPACE, TEST_EXECSPACE> kernel(p);

    // Compute moments of expansion
    auto M = kernel(cart_coords, q, num_points, expansion_center);
    auto M_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), M);

    // Evaluate at far target point P
    double Px = 15.5, Py = 13.9, Pz = -14.2;
    double r, theta, phi;
    Canopy::Kernel::cart2sph(Px-expansion_center[0], Py-expansion_center[1], Pz-expansion_center[2], r, theta, phi);
    
    using cdouble = Kokkos::complex<double>;
    cdouble phi_multipole = 0.0;
    for (int n=0; n<=p; ++n) {
        for (int m=-n; m<=n; ++m) {
            int idx = kernel.index(n,m);
            phi_multipole += M_host(idx) / Kokkos::pow(r,n+1) *
                        Canopy::Kernel::Ynm(n,m,theta,phi);
        }
    }

    // Direct potential for comparison
    auto cart_coords_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cart_coords);
    auto q_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q);
    double phi_direct = 0.0;
    for (int i=0; i<num_points; ++i) {
        double dx = Px-cart_coords_host(i, 0);
        double dy = Py-cart_coords_host(i, 1);
        double dz = Pz-cart_coords_host(i, 2);
        double dist = std::sqrt(dx*dx+dy*dy+dz*dz);
        phi_direct += q_host(i) / dist;
    }

    std::cout << "Multipole approx (p=2): " << phi_multipole.real() << "\n";
    std::cout << "Direct sum:             " << phi_direct << "\n";
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( Kernel, scalarKernel ) { scalarKernel(); }

//---------------------------------------------------------------------------//

} // end namespace Test