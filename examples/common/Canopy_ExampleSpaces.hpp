#ifndef CANOPY_EXAMPLE_SPACES_HPP
#define CANOPY_EXAMPLE_SPACES_HPP

#include <Kokkos_Core.hpp>

namespace CanopyExample
{

#if defined( KOKKOS_ENABLE_CUDA )
using ExecutionSpace = Kokkos::Cuda;
using MemorySpace = Kokkos::CudaSpace;
#elif defined( KOKKOS_ENABLE_HIP )
using ExecutionSpace = Kokkos::HIP;
using MemorySpace = Kokkos::HIPSpace;
#elif defined( KOKKOS_ENABLE_SYCL )
using ExecutionSpace = Kokkos::SYCL;
using MemorySpace = Kokkos::SYCLDeviceUSMSpace;
#elif defined( KOKKOS_ENABLE_OPENMP )
using ExecutionSpace = Kokkos::OpenMP;
using MemorySpace = Kokkos::HostSpace;
#elif defined( KOKKOS_ENABLE_SERIAL )
using ExecutionSpace = Kokkos::Serial;
using MemorySpace = Kokkos::HostSpace;
#else
#error "No supported Kokkos backend enabled (CUDA/HIP/SYCL/OpenMP/Serial)."
#endif

} // namespace CanopyExample

#endif
