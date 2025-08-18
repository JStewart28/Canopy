/****************************************************************************
 * Copyright (c) 2018-2023 by the Canopy authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Canopy library. Canopy is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CANOPY_TEST_CUDA_CATEGORY_HPP
#define CANOPY_TEST_CUDA_CATEGORY_HPP

#define TEST_CATEGORY cuda
#define TEST_EXECSPACE Kokkos::Cuda
#define TEST_MEMSPACE Kokkos::CudaSpace
#define TEST_DEVICE Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>

#endif // end CANOPY_TEST_CUDA_CATEGORY_HPP
