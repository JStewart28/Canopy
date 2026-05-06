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

#ifndef CANOPY_PROFILING_HPP
#define CANOPY_PROFILING_HPP

#include "Canopy_Config.hpp"

// ---------------------------------------------------------------------------
// Profiling level hierarchy
//
//   0 — off (no instrumentation)
//   1 — basic: top-level phases (P2M, M2M, M2L kernel, L2L, L2P, P2P, ...)
//   2 — detailed: sub-phases inside DownwardSweep::execute() — allocations,
//       interaction-list build, per-depth pre-M2L / M2L call / post-M2L
//   3 — verbose (reserved for future even finer-grained timers)
//
// Set via CMake: -DCanopy_PROFILING_LEVEL=2 (or the legacy
// -DCanopy_ENABLE_PROFILING=ON, which defaults to level 1).
// ---------------------------------------------------------------------------
#ifndef CANOPY_PROFILING_LEVEL
#  ifdef CANOPY_ENABLE_PROFILING
#    define CANOPY_PROFILING_LEVEL 1
#  else
#    define CANOPY_PROFILING_LEVEL 0
#  endif
#endif

#ifdef CANOPY_ENABLE_PROFILING

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace Canopy
{
namespace Profiling
{

// ---------------------------------------------------------------------------
// Phase key constants — used as keys in the timer registry and as labels in
// the PhaseEntry descriptors. Defined here so all instrumented headers share
// the same strings without risk of typos.
// ---------------------------------------------------------------------------

// Setup phases
static constexpr const char* TIMER_SETUP_TOTAL    = "setup_total";
static constexpr const char* TIMER_BUILDER_BUILD  = "builder_build";
static constexpr const char* TIMER_PARTITION      = "partition";
static constexpr const char* TIMER_SORT_BY_LEAF   = "sort_by_leaf";
static constexpr const char* TIMER_COMM_PLAN_BUILD = "comm_plan_build";

// Upward sweep
static constexpr const char* TIMER_UPWARD_TOTAL   = "upward_total";
static constexpr const char* TIMER_P2M            = "p2m";
static constexpr const char* TIMER_M2M            = "m2m";
static constexpr const char* TIMER_M2M_ALLREDUCE  = "m2m_allreduce";

// solve() — downward sweep
static constexpr const char* TIMER_DOWNWARD_TOTAL = "downward_total";
static constexpr const char* TIMER_M2L_COMM       = "m2l_comm";
static constexpr const char* TIMER_M2L_KERNEL     = "m2l_kernel";
static constexpr const char* TIMER_M2L_ALLREDUCE  = "m2l_allreduce";
static constexpr const char* TIMER_L2L_KERNEL     = "l2l_kernel";
static constexpr const char* TIMER_L2L_COMM       = "l2l_comm";
static constexpr const char* TIMER_L2P            = "l2p";

// Downward sweep — detailed (level 2) sub-phases.
// These dissolve unaccounted time inside DownwardSweep::execute() that the
// basic timers do not name explicitly.
static constexpr const char* TIMER_DN_ZERO_LOCALS = "dn_zero_locals";
static constexpr const char* TIMER_DN_BUILD_ILIST = "dn_build_ilist";
static constexpr const char* TIMER_DN_PRE_M2L     = "dn_pre_m2l";
static constexpr const char* TIMER_DN_M2L_CALL    = "dn_m2l_call";
static constexpr const char* TIMER_DN_POST_M2L    = "dn_post_m2l";

// solve() — P2P
static constexpr const char* TIMER_P2P_TOTAL        = "p2p_total";
static constexpr const char* TIMER_P2P_GHOST_COMM   = "p2p_ghost_comm";
static constexpr const char* TIMER_P2P_INTRA_KERNEL = "p2p_intra_kernel";
static constexpr const char* TIMER_P2P_INTER_KERNEL = "p2p_inter_kernel";

// Maintenance phases — migrate() and rebalance()
static constexpr const char* TIMER_MIGRATE_TOTAL    = "migrate_total";
static constexpr const char* TIMER_REDISTRIBUTE     = "redistribute";
static constexpr const char* TIMER_REBALANCE_TOTAL  = "rebalance_total";
static constexpr const char* TIMER_REPARTITION      = "repartition";

// ---------------------------------------------------------------------------
// Timer registry — process-local accumulator map, key -> elapsed seconds.
// Using a function-local static so this is safe in a header-only library:
// exactly one instance per process, initialized on first use.
// ---------------------------------------------------------------------------
inline std::unordered_map<std::string, double>& timer_registry()
{
    static std::unordered_map<std::string, double> s_reg;
    return s_reg;
}

inline void reset_timers()
{
    timer_registry().clear();
}

inline void accumulate( const char* key, double elapsed )
{
    timer_registry()[key] += elapsed;
}

// ---------------------------------------------------------------------------
// ScopedTimer — RAII guard. Records wall time at construction via MPI_Wtime
// and accumulates the elapsed time at destruction. Non-copyable.
//
// Usage:
//   { ScopedTimer t( TIMER_P2M ); /* work */ }  // accumulates on scope exit
//
// Because each instrumented method already has a Kokkos::fence() at the end,
// the destructor fires after device work completes, giving accurate wall time.
// ---------------------------------------------------------------------------
struct ScopedTimer
{
    const char* key;
    double      t0;

    explicit ScopedTimer( const char* phase_key )
        : key( phase_key )
        , t0( MPI_Wtime() )
    {
        Kokkos::Profiling::pushRegion( key );
    }

    ~ScopedTimer()
    {
        Kokkos::Profiling::popRegion();
        accumulate( key, MPI_Wtime() - t0 );
    }

    ScopedTimer( const ScopedTimer& ) = delete;
    ScopedTimer& operator=( const ScopedTimer& ) = delete;
};

// ---------------------------------------------------------------------------
// PhaseEntry — describes one row in the printed timing table.
// ---------------------------------------------------------------------------
struct PhaseEntry
{
    const char* label;
    const char* key;
    int         indent; // 0 = top-level, 1 = sub-phase (2 leading spaces each)
};

// ---------------------------------------------------------------------------
// Ordered phase entry lists for each print site.
// ---------------------------------------------------------------------------
inline std::vector<PhaseEntry> setup_phase_entries()
{
    return {
        { "Setup total",             TIMER_SETUP_TOTAL,    0 },
        { "Tree build (all steps)",  TIMER_BUILDER_BUILD,  1 },
        { "Partition",               TIMER_PARTITION,      1 },
        { "Sort particles by leaf",  TIMER_SORT_BY_LEAF,   1 },
        { "Comm plan build",         TIMER_COMM_PLAN_BUILD,1 },
    };
}

inline std::vector<PhaseEntry> upward_phase_entries()
{
    return {
        { "Upward sweep total",  TIMER_UPWARD_TOTAL,  0 },
        { "P2M",                 TIMER_P2M,           1 },
        { "M2M (all depths)",    TIMER_M2M,           1 },
        { "M2M Allreduce",       TIMER_M2M_ALLREDUCE, 1 },
    };
}

inline std::vector<PhaseEntry> downward_phase_entries()
{
    std::vector<PhaseEntry> entries = {
        { "Downward sweep total",       TIMER_DOWNWARD_TOTAL, 0 },
        { "M2L comm (all depths)",      TIMER_M2L_COMM,       1 },
        { "M2L kernel (all depths)",    TIMER_M2L_KERNEL,     1 },
        { "M2L Allreduce (all depths)", TIMER_M2L_ALLREDUCE,  1 },
        { "L2L kernel (all depths)",    TIMER_L2L_KERNEL,     1 },
        { "L2L comm (all depths)",      TIMER_L2L_COMM,       1 },
        { "L2P",                        TIMER_L2P,            1 },
    };
#if CANOPY_PROFILING_LEVEL >= 2
    // Detailed sub-phases — sum to the in-execute() share of "Downward
    // sweep total" not already attributed above. Indented one further
    // level in the printed table for readability.
    entries.push_back( { "[detail] Zero locals",      TIMER_DN_ZERO_LOCALS, 1 } );
    entries.push_back( { "[detail] Build ilist",     TIMER_DN_BUILD_ILIST, 1 } );
    entries.push_back( { "[detail] Pre-M2L setup",   TIMER_DN_PRE_M2L,     1 } );
    entries.push_back( { "[detail] M2L call (loop)", TIMER_DN_M2L_CALL,    1 } );
    entries.push_back( { "[detail] Post-M2L (loop)", TIMER_DN_POST_M2L,    1 } );
#endif
    return entries;
}

inline std::vector<PhaseEntry> p2p_phase_entries()
{
    return {
        { "P2P total",             TIMER_P2P_TOTAL,        0 },
        { "P2P ghost comm",        TIMER_P2P_GHOST_COMM,   1 },
        { "P2P intra-leaf kernel", TIMER_P2P_INTRA_KERNEL, 1 },
        { "P2P inter-leaf kernel", TIMER_P2P_INTER_KERNEL, 1 },
    };
}

inline std::vector<PhaseEntry> migrate_phase_entries()
{
    return {
        { "Migrate total",       TIMER_MIGRATE_TOTAL,   0 },
        { "Redistribute",        TIMER_REDISTRIBUTE,    1 },
        { "Builder build (all)", TIMER_BUILDER_BUILD,   1 },
        { "Sort by leaf",        TIMER_SORT_BY_LEAF,    1 },
        { "Comm plan build",     TIMER_COMM_PLAN_BUILD, 1 },
    };
}

inline std::vector<PhaseEntry> rebalance_phase_entries()
{
    return {
        { "Rebalance total",     TIMER_REBALANCE_TOTAL, 0 },
        { "Repartition",         TIMER_REPARTITION,     1 },
        { "Builder build (all)", TIMER_BUILDER_BUILD,   1 },
        { "Sort by leaf",        TIMER_SORT_BY_LEAF,    1 },
        { "Comm plan build",     TIMER_COMM_PLAN_BUILD, 1 },
    };
}

// ---------------------------------------------------------------------------
// print_solve_breakdown
//
// Prints a compact percentage-of-total table for Solver::solve(). Each
// execute() call resets the shared registry, so solve() captures per-phase
// wall time with raw MPI_Wtime() and passes the three values here.
// ---------------------------------------------------------------------------
inline void print_solve_breakdown( MPI_Comm comm,
                                   double t_up, double t_dn, double t_p2p )
{
    int rank, nprocs;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &nprocs );

    double local[3] = { t_up, t_dn, t_p2p };
    double sum_v[3];
    MPI_Reduce( local, sum_v, 3, MPI_DOUBLE, MPI_SUM, 0, comm );
    if ( rank != 0 )
        return;

    const double inv      = 1.0 / static_cast<double>( nprocs );
    const double mean_up  = sum_v[0] * inv;
    const double mean_dn  = sum_v[1] * inv;
    const double mean_p2p = sum_v[2] * inv;
    const double total    = mean_up + mean_dn + mean_p2p;
    const double safe_tot = ( total > 0.0 ) ? total : 1.0;

    static constexpr int COL_LABEL = 20;
    static constexpr int COL_NUM   = 9;
    std::printf( "\n[Canopy Diagnostics] solve() phase breakdown (%d MPI rank%s)\n",
                 nprocs, nprocs > 1 ? "s" : "" );
    std::printf( "  %-*s  %*s  %s\n",
                 COL_LABEL, "Phase", COL_NUM, "Mean (s)", "% of total" );
    const int sep_len = COL_LABEL + COL_NUM + 16;
    for ( int i = 0; i < sep_len; i++ )
        std::putchar( '-' );
    std::putchar( '\n' );
    std::printf( "  %-*s  %*.3f  %5.1f%%\n",
                 COL_LABEL, "Total solve", COL_NUM, total, 100.0 );
    std::printf( "  %-*s  %*.3f  %5.1f%%\n",
                 COL_LABEL, "Upward sweep", COL_NUM, mean_up,
                 mean_up / safe_tot * 100.0 );
    std::printf( "  %-*s  %*.3f  %5.1f%%\n",
                 COL_LABEL, "Downward sweep", COL_NUM, mean_dn,
                 mean_dn / safe_tot * 100.0 );
    std::printf( "  %-*s  %*.3f  %5.1f%%\n",
                 COL_LABEL, "P2P", COL_NUM, mean_p2p,
                 mean_p2p / safe_tot * 100.0 );
    std::putchar( '\n' );
    std::fflush( stdout );
}

// ---------------------------------------------------------------------------
// print_timing_table
//
// Gathers per-rank timing data via three MPI_Reduce calls (MIN, MAX, SUM)
// to rank 0. Only rank 0 prints the formatted table. Other ranks return
// immediately after the reduce, so this is a collective call.
//
// Parameters:
//   comm         - MPI communicator (same one used for the solve)
//   section_name - printed in the header line (e.g. "solve()")
//   phases       - ordered list of PhaseEntry rows to print
// ---------------------------------------------------------------------------
inline void print_timing_table( MPI_Comm comm, const char* section_name,
                                 const std::vector<PhaseEntry>& phases )
{
    int rank, nprocs;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &nprocs );

    const auto& reg = timer_registry();
    const int   N   = static_cast<int>( phases.size() );

    std::vector<double> local_vals( N, 0.0 );
    for ( int i = 0; i < N; i++ )
    {
        auto it = reg.find( phases[i].key );
        if ( it != reg.end() )
            local_vals[i] = it->second;
    }

    std::vector<double> min_vals( N ), max_vals( N ), sum_vals( N );
    MPI_Reduce( local_vals.data(), min_vals.data(), N,
                MPI_DOUBLE, MPI_MIN, 0, comm );
    MPI_Reduce( local_vals.data(), max_vals.data(), N,
                MPI_DOUBLE, MPI_MAX, 0, comm );
    MPI_Reduce( local_vals.data(), sum_vals.data(), N,
                MPI_DOUBLE, MPI_SUM, 0, comm );

    if ( rank != 0 )
        return;

    // Column widths
    static constexpr int COL_LABEL = 36;
    static constexpr int COL_NUM   = 9;

    std::printf( "\n[Canopy Diagnostics] %s timing (%d MPI rank%s)\n",
                 section_name, nprocs, nprocs > 1 ? "s" : "" );
    std::printf( "  %-*s  %*s  %*s  %*s  %s\n",
                 COL_LABEL, "Phase",
                 COL_NUM,   "Min (s)",
                 COL_NUM,   "Max (s)",
                 COL_NUM,   "Mean (s)",
                 "Imbalance" );

    const int sep_len = COL_LABEL + 3 * ( COL_NUM + 2 ) + 12;
    for ( int i = 0; i < sep_len; i++ )
        std::putchar( '-' );
    std::putchar( '\n' );

    const double inv_nprocs = 1.0 / static_cast<double>( nprocs );
    for ( int i = 0; i < N; i++ )
    {
        const double mn   = min_vals[i];
        const double mx   = max_vals[i];
        const double mean = sum_vals[i] * inv_nprocs;
        const double imb  = ( mean > 0.0 )
                            ? ( mx - mean ) / mean * 100.0
                            : 0.0;

        // Build indented label
        char buf[64];
        const int indent_spaces = 2 * phases[i].indent;
        std::snprintf( buf, sizeof( buf ), "%*s%s",
                       indent_spaces, "", phases[i].label );

        std::printf( "  %-*s  %*.3f  %*.3f  %*.3f  %.1f%%\n",
                     COL_LABEL, buf,
                     COL_NUM,   mn,
                     COL_NUM,   mx,
                     COL_NUM,   mean,
                     imb );
    }
    std::putchar( '\n' );
    std::fflush( stdout );
}

} // namespace Profiling
} // namespace Canopy

#endif // CANOPY_ENABLE_PROFILING

// ---------------------------------------------------------------------------
// Convenience macros — defined whether or not diagnostics are enabled so
// instrumentation in other headers compiles in both modes.
// ---------------------------------------------------------------------------
#ifdef CANOPY_ENABLE_PROFILING
#  define CANOPY_SCOPED_TIMER( key ) \
       ::Canopy::Profiling::ScopedTimer _canopy_timer_##__LINE__( (key) )
#  define CANOPY_RESET_TIMERS() \
       ::Canopy::Profiling::reset_timers()
#  define CANOPY_WTIME() MPI_Wtime()
#  define CANOPY_PRINT_SETUP_TIMERS( comm ) \
       ::Canopy::Profiling::print_timing_table( \
           (comm), "setup()", ::Canopy::Profiling::setup_phase_entries() )
#  define CANOPY_PRINT_UPWARD_TIMERS( comm ) \
       ::Canopy::Profiling::print_timing_table( \
           (comm), "UpwardSweep::execute()", \
           ::Canopy::Profiling::upward_phase_entries() )
#  define CANOPY_PRINT_DOWNWARD_TIMERS( comm ) \
       ::Canopy::Profiling::print_timing_table( \
           (comm), "DownwardSweep::execute()", \
           ::Canopy::Profiling::downward_phase_entries() )
#  define CANOPY_PRINT_P2P_TIMERS( comm ) \
       ::Canopy::Profiling::print_timing_table( \
           (comm), "P2P::execute()", \
           ::Canopy::Profiling::p2p_phase_entries() )
#  define CANOPY_PRINT_SOLVE_BREAKDOWN( comm, t_up, t_dn, t_p2p ) \
       ::Canopy::Profiling::print_solve_breakdown( (comm), (t_up), (t_dn), (t_p2p) )
#  define CANOPY_PRINT_MIGRATE_TIMERS( comm ) \
       ::Canopy::Profiling::print_timing_table( \
           (comm), "migrate()", ::Canopy::Profiling::migrate_phase_entries() )
#  define CANOPY_PRINT_REBALANCE_TIMERS( comm ) \
       ::Canopy::Profiling::print_timing_table( \
           (comm), "rebalance()", ::Canopy::Profiling::rebalance_phase_entries() )
// Detailed (level 2) and verbose (level 3) timer macros. They compile away
// to no-ops below the requested level so call sites can be left in place.
#  if CANOPY_PROFILING_LEVEL >= 2
#    define CANOPY_SCOPED_TIMER_DETAILED( key ) \
         ::Canopy::Profiling::ScopedTimer _canopy_timer_d_##__LINE__( (key) )
#  else
#    define CANOPY_SCOPED_TIMER_DETAILED( key ) do {} while ( 0 )
#  endif
#  if CANOPY_PROFILING_LEVEL >= 3
#    define CANOPY_SCOPED_TIMER_VERBOSE( key ) \
         ::Canopy::Profiling::ScopedTimer _canopy_timer_v_##__LINE__( (key) )
#  else
#    define CANOPY_SCOPED_TIMER_VERBOSE( key ) do {} while ( 0 )
#  endif
#else
#  define CANOPY_SCOPED_TIMER( key )            do {} while ( 0 )
#  define CANOPY_RESET_TIMERS()                 do {} while ( 0 )
#  define CANOPY_WTIME()                        0.0
#  define CANOPY_PRINT_SETUP_TIMERS( comm )     do {} while ( 0 )
#  define CANOPY_PRINT_UPWARD_TIMERS( comm )    do {} while ( 0 )
#  define CANOPY_PRINT_DOWNWARD_TIMERS( comm )  do {} while ( 0 )
#  define CANOPY_PRINT_P2P_TIMERS( comm )       do {} while ( 0 )
#  define CANOPY_PRINT_SOLVE_BREAKDOWN( comm, t_up, t_dn, t_p2p ) \
       do { (void)(comm); (void)(t_up); (void)(t_dn); (void)(t_p2p); } while(0)
#  define CANOPY_PRINT_MIGRATE_TIMERS( comm )   do {} while ( 0 )
#  define CANOPY_PRINT_REBALANCE_TIMERS( comm ) do {} while ( 0 )
#  define CANOPY_SCOPED_TIMER_DETAILED( key )    do {} while ( 0 )
#  define CANOPY_SCOPED_TIMER_VERBOSE( key )     do {} while ( 0 )
#endif

#endif // CANOPY_PROFILING_HPP
