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
            cart_coords(i, 0) = rand_gen.drand() * 2.0 - 1.0;
            cart_coords(i, 1) = rand_gen.drand() * 2.0 - 1.0;
            cart_coords(i, 2) = rand_gen.drand() * 2.0 - 1.0;
            q(i) = cart_coords(i, 0) * 1.5;
            rand_pool.free_state(rand_gen);
        });
    Kokkos::fence();

    // Expansion center
    Kokkos::Array<double, 3> expansion_center = {0.1, -0.6, 0.3};

    // Target point
    double Px = 13.6, Py = -5.1, Pz = 2.0;
    double r, theta, phi;
    Canopy::Kernel::cart2sph(Px-expansion_center[0], Py-expansion_center[1], Pz-expansion_center[2], r, theta, phi);

    // Direct potential
    auto cart_coords_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cart_coords);
    auto q_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q);
    double phi_direct = 0.0;
    double max_rho = 0.0; // for error bound
    for (int i=0; i<num_points; ++i) {
        double dx = Px - cart_coords_host(i, 0);
        double dy = Py - cart_coords_host(i, 1);
        double dz = Pz - cart_coords_host(i, 2);
        double dist = std::sqrt(dx*dx+dy*dy+dz*dz);
        phi_direct += q_host(i) / dist;

        // distance from expansion center for error estimate
        double ddx = cart_coords_host(i,0) - expansion_center[0];
        double ddy = cart_coords_host(i,1) - expansion_center[1];
        double ddz = cart_coords_host(i,2) - expansion_center[2];
        double rho = std::sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
        max_rho = std::max(max_rho, rho);
    }

    constexpr auto pi = Kokkos::numbers::pi_v<double>;

    // Loop over truncation degree
    for (int p = 1; p <= 5; ++p)
    {
        Canopy::Kernel::Scalar<TEST_MEMSPACE, TEST_EXECSPACE> kernel(p);
        auto M = kernel(cart_coords, q, num_points, expansion_center);
        auto M_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), M);

        using cdouble = Kokkos::complex<double>;
        cdouble phi_multipole = 0.0;
        for (int n=0; n<=p; ++n) {
            double pref = 4*pi / double(2*n + 1);
            for (int m=-n; m<=n; ++m) {
                int idx = kernel.index(n,m);
                phi_multipole += pref * M_host(idx) / Kokkos::pow(r,n+1) *
                                 Canopy::Kernel::Ynm(n,m,theta,phi);
            }
        }

        double error = std::abs(phi_multipole.real() - phi_direct);
        double bound = std::pow(max_rho/r, p+1) * std::abs(phi_direct);

        // Check that the error is within 10*bound, which accounts
        // for imprecision due to imtermediate rounding.
        EXPECT_NEAR(phi_multipole.real(), phi_direct, 10*bound)
            << "p=" << p
            << " multipole=" << phi_multipole.real()
            << " direct=" << phi_direct
            << " error=" << error
            << " bound~" << bound
            << std::endl;
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( Kernel, scalarKernel ) { scalarKernel(); }

//---------------------------------------------------------------------------//

} // end namespace Test