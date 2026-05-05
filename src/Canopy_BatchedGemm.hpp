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

#ifndef CANOPY_BATCHED_GEMM_HPP
#define CANOPY_BATCHED_GEMM_HPP

#include <Kokkos_Core.hpp>

#include <stdexcept>
#include <string>
#include <type_traits>

#if defined( CANOPY_ENABLE_CUBLAS )
#include <cublas_v2.h>
#endif

#if defined( CANOPY_ENABLE_HIPBLAS )
#include <hipblas/hipblas.h>
#endif

namespace Canopy
{
namespace detail
{

// ============================================================================
// M2LBatchedGemm
//
// Thin RAII wrapper around a vendor BLAS handle that exposes a single
// non-transposed dense complex GEMM:
//
//     C(Nt, n) = T(Nt, Ns) * M(Ns, n)            alpha = 1, beta = 0
//
// All matrices are column-major (LayoutLeft) and complex of element type
// Kokkos::complex<Scalar> for Scalar in {float, double}. The handle is bound
// to the supplied execution space's native stream so the GEMM serializes
// with surrounding Kokkos kernels without a host sync.
//
// The primary template is a no-op stub that reports `available == false`.
// Backend specializations for Kokkos::Cuda (cuBLAS) and Kokkos::HIP
// (hipBLAS, dispatches to rocBLAS on AMD) override gemm_NN with a real call.
// Callers must check `available` and route to the per-pair fallback kernel
// when batched GEMM is not present.
// ============================================================================

template <class ExecSpace, class Scalar>
class M2LBatchedGemm
{
  public:
    using execution_space = ExecSpace;
    using scalar_type = Scalar;
    using complex_type = Kokkos::complex<Scalar>;

    static constexpr bool available = false;

    explicit M2LBatchedGemm( ExecSpace = ExecSpace{} ) {}

    void gemm_NN( int /*Nt*/, int /*n*/, int /*Ns*/,
                  const complex_type* /*T*/, int /*ldT*/,
                  const complex_type* /*M*/, int /*ldM*/,
                  complex_type* /*C*/, int /*ldC*/ )
    {
        throw std::runtime_error(
            "Canopy::detail::M2LBatchedGemm: no vendor BLAS backend is "
            "available for this execution space. Build Canopy with CUDA or "
            "HIP, or route M2L through the per-pair fallback kernel." );
    }
};

// ----------------------------------------------------------------------------
// CUDA / cuBLAS specialization
// ----------------------------------------------------------------------------
#if defined( CANOPY_ENABLE_CUBLAS ) && defined( KOKKOS_ENABLE_CUDA )

namespace cublas_impl
{
inline void check( cublasStatus_t s, const char* what )
{
    if ( s != CUBLAS_STATUS_SUCCESS )
        throw std::runtime_error(
            std::string( "Canopy cuBLAS error in " ) + what + ": status=" +
            std::to_string( static_cast<int>( s ) ) );
}
} // namespace cublas_impl

template <class Scalar>
class M2LBatchedGemm<Kokkos::Cuda, Scalar>
{
    static_assert( std::is_same_v<Scalar, float> ||
                       std::is_same_v<Scalar, double>,
                   "M2LBatchedGemm: Scalar must be float or double" );

  public:
    using execution_space = Kokkos::Cuda;
    using scalar_type = Scalar;
    using complex_type = Kokkos::complex<Scalar>;

    static constexpr bool available = true;

    explicit M2LBatchedGemm( Kokkos::Cuda exec = Kokkos::Cuda{} )
    {
        cublas_impl::check( cublasCreate( &_handle ), "cublasCreate" );
        cublas_impl::check( cublasSetStream( _handle, exec.cuda_stream() ),
                            "cublasSetStream" );
        cublas_impl::check(
            cublasSetPointerMode( _handle, CUBLAS_POINTER_MODE_HOST ),
            "cublasSetPointerMode" );
    }

    M2LBatchedGemm( const M2LBatchedGemm& ) = delete;
    M2LBatchedGemm& operator=( const M2LBatchedGemm& ) = delete;

    M2LBatchedGemm( M2LBatchedGemm&& other ) noexcept
        : _handle( other._handle )
    {
        other._handle = nullptr;
    }
    M2LBatchedGemm& operator=( M2LBatchedGemm&& other ) noexcept
    {
        if ( this != &other )
        {
            if ( _handle )
                cublasDestroy( _handle );
            _handle = other._handle;
            other._handle = nullptr;
        }
        return *this;
    }

    ~M2LBatchedGemm()
    {
        if ( _handle )
            cublasDestroy( _handle );
    }

    void gemm_NN( int Nt, int n, int Ns, const complex_type* T, int ldT,
                  const complex_type* M, int ldM, complex_type* C, int ldC )
    {
        if constexpr ( std::is_same_v<Scalar, float> )
        {
            const cuComplex one = make_cuComplex( 1.0f, 0.0f );
            const cuComplex zero = make_cuComplex( 0.0f, 0.0f );
            cublas_impl::check(
                cublasCgemm( _handle, CUBLAS_OP_N, CUBLAS_OP_N, Nt, n, Ns,
                             &one, reinterpret_cast<const cuComplex*>( T ),
                             ldT, reinterpret_cast<const cuComplex*>( M ),
                             ldM, &zero, reinterpret_cast<cuComplex*>( C ),
                             ldC ),
                "cublasCgemm" );
        }
        else
        {
            const cuDoubleComplex one = make_cuDoubleComplex( 1.0, 0.0 );
            const cuDoubleComplex zero = make_cuDoubleComplex( 0.0, 0.0 );
            cublas_impl::check(
                cublasZgemm(
                    _handle, CUBLAS_OP_N, CUBLAS_OP_N, Nt, n, Ns, &one,
                    reinterpret_cast<const cuDoubleComplex*>( T ), ldT,
                    reinterpret_cast<const cuDoubleComplex*>( M ), ldM,
                    &zero, reinterpret_cast<cuDoubleComplex*>( C ), ldC ),
                "cublasZgemm" );
        }
    }

  private:
    cublasHandle_t _handle = nullptr;
};

#endif // CANOPY_ENABLE_CUBLAS && KOKKOS_ENABLE_CUDA

// ----------------------------------------------------------------------------
// HIP / hipBLAS specialization (dispatches to rocBLAS on AMD)
// ----------------------------------------------------------------------------
#if defined( CANOPY_ENABLE_HIPBLAS ) && defined( KOKKOS_ENABLE_HIP )

namespace hipblas_impl
{
inline void check( hipblasStatus_t s, const char* what )
{
    if ( s != HIPBLAS_STATUS_SUCCESS )
        throw std::runtime_error(
            std::string( "Canopy hipBLAS error in " ) + what + ": status=" +
            std::to_string( static_cast<int>( s ) ) );
}
} // namespace hipblas_impl

template <class Scalar>
class M2LBatchedGemm<Kokkos::HIP, Scalar>
{
    static_assert( std::is_same_v<Scalar, float> ||
                       std::is_same_v<Scalar, double>,
                   "M2LBatchedGemm: Scalar must be float or double" );

  public:
    using execution_space = Kokkos::HIP;
    using scalar_type = Scalar;
    using complex_type = Kokkos::complex<Scalar>;

    static constexpr bool available = true;

    explicit M2LBatchedGemm( Kokkos::HIP exec = Kokkos::HIP{} )
    {
        hipblas_impl::check( hipblasCreate( &_handle ), "hipblasCreate" );
        hipblas_impl::check(
            hipblasSetStream( _handle, exec.hip_stream() ),
            "hipblasSetStream" );
        hipblas_impl::check(
            hipblasSetPointerMode( _handle, HIPBLAS_POINTER_MODE_HOST ),
            "hipblasSetPointerMode" );
    }

    M2LBatchedGemm( const M2LBatchedGemm& ) = delete;
    M2LBatchedGemm& operator=( const M2LBatchedGemm& ) = delete;

    M2LBatchedGemm( M2LBatchedGemm&& other ) noexcept
        : _handle( other._handle )
    {
        other._handle = nullptr;
    }
    M2LBatchedGemm& operator=( M2LBatchedGemm&& other ) noexcept
    {
        if ( this != &other )
        {
            if ( _handle )
                hipblasDestroy( _handle );
            _handle = other._handle;
            other._handle = nullptr;
        }
        return *this;
    }

    ~M2LBatchedGemm()
    {
        if ( _handle )
            hipblasDestroy( _handle );
    }

    void gemm_NN( int Nt, int n, int Ns, const complex_type* T, int ldT,
                  const complex_type* M, int ldM, complex_type* C, int ldC )
    {
        if constexpr ( std::is_same_v<Scalar, float> )
        {
            const hipblasComplex one = { 1.0f, 0.0f };
            const hipblasComplex zero = { 0.0f, 0.0f };
            hipblas_impl::check(
                hipblasCgemm(
                    _handle, HIPBLAS_OP_N, HIPBLAS_OP_N, Nt, n, Ns, &one,
                    reinterpret_cast<const hipblasComplex*>( T ), ldT,
                    reinterpret_cast<const hipblasComplex*>( M ), ldM, &zero,
                    reinterpret_cast<hipblasComplex*>( C ), ldC ),
                "hipblasCgemm" );
        }
        else
        {
            const hipblasDoubleComplex one = { 1.0, 0.0 };
            const hipblasDoubleComplex zero = { 0.0, 0.0 };
            hipblas_impl::check(
                hipblasZgemm(
                    _handle, HIPBLAS_OP_N, HIPBLAS_OP_N, Nt, n, Ns, &one,
                    reinterpret_cast<const hipblasDoubleComplex*>( T ), ldT,
                    reinterpret_cast<const hipblasDoubleComplex*>( M ), ldM,
                    &zero, reinterpret_cast<hipblasDoubleComplex*>( C ),
                    ldC ),
                "hipblasZgemm" );
        }
    }

  private:
    hipblasHandle_t _handle = nullptr;
};

#endif // CANOPY_ENABLE_HIPBLAS && KOKKOS_ENABLE_HIP

} // namespace detail
} // namespace Canopy

#endif // CANOPY_BATCHED_GEMM_HPP
