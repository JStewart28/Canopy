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

#ifndef CANOPY_SOLVER_HPP
#define CANOPY_SOLVER_HPP

#include "Canopy_CommunicationPlan.hpp"
#include "Canopy_Profiling.hpp"
#include "Canopy_DownwardSweep.hpp"
#include "Canopy_LaplaceKernel.hpp"
#include "Canopy_P2P.hpp"
#include "Canopy_TreeBuilder.hpp"
#include "Canopy_TreePartitioner.hpp"
#include "Canopy_UpwardSweep.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <unordered_set>

namespace Canopy
{

// ============================================================================
// Solver
//
// Facade that owns the full FMM pipeline (TreeBuilder, TreePartitioner,
// CommunicationPlan, UpwardSweep, DownwardSweep, P2P) and exposes the
// minimal API a time-stepping application needs:
//
//   setup()          — one-time initialization
//   solve()          — one timestep evaluation (P2M → M2M → M2L → L2L
//                      → L2P → P2P), zeroing internal output views first
//   migrate()        — cheapest inter-step maintenance; particles moved
//                      but tree topology unchanged
//   rebalance()      — moderate maintenance; topology changed but
//                      bounding box still valid
//   rebuild()        — heavy maintenance; full do-over
//   auto_maintain()  — picks the cheapest valid path automatically
//
// Template parameters:
//   MemorySpace, ExecutionSpace - Kokkos spaces
//   Scalar  - field scalar type (default double)
//   P_ORDER - multipole expansion order
//   NComps  - number of simultaneous solves (charge components)
// ============================================================================

template <class MemorySpace, class ExecutionSpace, class Scalar = double,
          int P_ORDER = 8, int NComps = 1>
class Solver
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;

    using kernel_type = LaplaceKernel<Scalar, P_ORDER, NComps>;
    using builder_type = TreeBuilder<MemorySpace, ExecutionSpace>;
    using partitioner_type = TreePartitioner<MemorySpace, ExecutionSpace>;
    using comm_plan_type = CommunicationPlan<MemorySpace, ExecutionSpace>;
    using upward_type = UpwardSweep<MemorySpace, ExecutionSpace, kernel_type>;
    using downward_type =
        DownwardSweep<MemorySpace, ExecutionSpace, kernel_type>;
    using p2p_type = P2P<MemorySpace, ExecutionSpace, kernel_type>;

    using potential_view_type = typename downward_type::potential_view_type;
    using gradient_view_type = typename downward_type::gradient_view_type;

    enum class MaintenanceAction
    {
        Migrate,
        Rebalance,
        Rebuild
    };

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------
    // softening: near-field Plummer softening length. Bounds the pairwise
    //   force for close encounters so a divergent acceleration cannot fling a
    //   particle out of the domain and degenerate the next tree build (an
    //   unsoftened run on a clustering system escapes the bounding box, forces
    //   a rebuild into a pathological tree, and corrupts the M2L exchange).
    //
    //   Pass softening >= 0 to set it explicitly (0 = unsoftened, opt-in).
    //   Pass softening < 0 (the DEFAULT) to auto-derive it from the particle
    //   distribution at setup() time as
    //       eps = SOFTENING_FACTOR * (V / N)^(1/3)
    //   i.e. a fraction of the mean inter-particle spacing, where V is the
    //   global bounding-box volume and N the total particle count. Computed
    //   once at the first setup() and held fixed thereafter (see README for
    //   the rationale and the future option to recompute per rebuild).
    Solver( MPI_Comm comm, int ncrit, int max_depth, std::array<double, 3> bounding_box_tol, double ncrit_tol,
            int replication_depth, double imbalance_tolerance = 0.05,
            double mac_theta = 0.5, double softening = -1.0 )
        : _comm( comm )
        , _replication_depth( replication_depth )
        , _builder( comm, ncrit, max_depth, bounding_box_tol, ncrit_tol )
        , _partitioner( comm, replication_depth, imbalance_tolerance )
        , _comm_plan( comm, mac_theta )
        , _upward( comm )
        , _downward( comm )
        , _p2p( comm )
        , _num_local( 0 )
        , _softening_input( softening )
        , _softening_initialized( false )
    {
        // Explicit softening (including an explicit 0 for an unsoftened run):
        // apply it now. A negative value defers to the distribution-based
        // auto-softening computed in _full_setup().
        if ( softening >= 0.0 )
        {
            _p2p.set_softening( static_cast<Scalar>( softening ) );
            _softening_initialized = true;
        }
    }

    // -----------------------------------------------------------------------
    // setup(): one-time pipeline initialization.
    //
    //   build → partition → build → sort_by_leaf → build → comm_plan.build
    //   → upward.setup → downward.setup → p2p.setup
    //
    // After setup() the AoSoA has been permuted (particles grouped by leaf)
    // and migrated to its owning rank. The caller's num_local_particles is
    // the count BEFORE migration.
    // -----------------------------------------------------------------------
    template <int PositionIdx, int ChargeIdx, class AoSoA>
    void setup( AoSoA& particles, int num_local_particles_before )
    {
        _full_setup<PositionIdx, ChargeIdx>( particles,
                                             num_local_particles_before );
    }

    // -----------------------------------------------------------------------
    // solve(): one FMM + P2P evaluation against the current particle state.
    // Outputs are accumulated into internally-owned views (zeroed first).
    // -----------------------------------------------------------------------
    template <int PositionIdx, int ChargeIdx, class AoSoA>
    void solve( AoSoA& particles, bool compute_gradient )
    {
        auto positions = Cabana::slice<PositionIdx>( particles );
        auto charges = Cabana::slice<ChargeIdx>( particles );

        // Resize/zero output views to current local count
        if ( static_cast<int>( _potential.extent( 0 ) ) != _num_local )
            _potential = potential_view_type( "fmm_potential", _num_local );
        Kokkos::deep_copy( _potential, Scalar( 0 ) );

        if ( compute_gradient )
        {
            if ( static_cast<int>( _gradient.extent( 0 ) ) != _num_local )
                _gradient = gradient_view_type( "fmm_gradient", _num_local );
            Kokkos::deep_copy( _gradient, Scalar( 0 ) );
        }
        else if ( _gradient.extent( 0 ) != 0 )
        {
            _gradient = gradient_view_type( "fmm_gradient", 0 );
        }

        double _t0 = CANOPY_WTIME();
        _upward.execute( charges, positions, _comm_plan );
        double _t_up = CANOPY_WTIME() - _t0;

        _t0 = CANOPY_WTIME();
        _downward.execute( _upward.multipoles(), positions, _potential,
                           _gradient, compute_gradient, _comm_plan );
        double _t_dn = CANOPY_WTIME() - _t0;

        _t0 = CANOPY_WTIME();
        _p2p.execute( positions, charges, _potential, _gradient,
                      compute_gradient );
        double _t_p2p = CANOPY_WTIME() - _t0;

        CANOPY_PRINT_SOLVE_BREAKDOWN( _comm, _t_up, _t_dn, _t_p2p );
    }

    // -----------------------------------------------------------------------
    // migrate(): cheapest maintenance. Tree topology assumed unchanged;
    // only particle positions shifted (possibly across ranks).
    //
    //   redistribute → build → sort_by_leaf → build
    //   → upward.setup → downward.setup → p2p.setup
    //
    // comm_plan is reused when topology is confirmed stable. If build()
    // detects a topology change (bounding-box drift can shift cell
    // boundaries past nearby particles), falls back to
    // _finish_topology_change so the comm_plan is rebuilt before the
    // next solve.
    // -----------------------------------------------------------------------
    template <int PositionIdx, class AoSoA>
    RedistributeResult migrate( AoSoA& particles )
    {
        CANOPY_RESET_TIMERS();
        RedistributeResult result{};
        {
            CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_MIGRATE_TOTAL );

            // Snapshot the current cell-key set before rebuilding the tree.
            std::unordered_set<MortonKey> old_keys;
            old_keys.reserve( _builder.cells().size() );
            for ( const auto& c : _builder.cells() )
                old_keys.insert( c.key );

            // Rebuild tree from current positions (also recomputes bounding box).
            { CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_BUILDER_BUILD );
              auto positions = Cabana::slice<PositionIdx>( particles );
              _builder.build( positions, _num_local ); }

            // If the topology changed the existing comm_plan is invalid.
            // Fall back to the full topology-change path so it is rebuilt.
            bool topology_changed = ( _builder.cells().size() != old_keys.size() );
            if ( !topology_changed )
            {
                for ( const auto& c : _builder.cells() )
                {
                    if ( old_keys.find( c.key ) == old_keys.end() )
                    {
                        topology_changed = true;
                        break;
                    }
                }
            }

            if ( topology_changed )
            {
                _finish_topology_change<PositionIdx>( particles );
                result = RedistributeResult{ 0, 0, _num_local };
            }
            else
            {
                { CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_REDISTRIBUTE );
                  result = _partitioner.redistribute( _builder, particles,
                                                      _num_local ); }
                _num_local = _partitioner.num_local_particles();
                _finish_topology_stable<PositionIdx>( particles );
            }
        } // TIMER_MIGRATE_TOTAL destructs here
        CANOPY_PRINT_MIGRATE_TIMERS( _comm );
        CANOPY_PRINT_COMMPLAN_TIMERS( _comm );
        return result;
    }

    // -----------------------------------------------------------------------
    // rebalance(): moderate maintenance. Tree topology may change
    // (refine / coarsen) but bounding box still valid.
    //
    //   builder.update → repartition → build → sort_by_leaf → build
    //   → comm_plan.build → upward.setup → downward.setup → p2p.setup
    // -----------------------------------------------------------------------
    template <int PositionIdx, class AoSoA>
    void rebalance( AoSoA& particles )
    {
        // Full rebuild of the global tree from current positions.
        // Cheaper paths (TreeBuilder::update) can be substituted later
        // once stable — rebalance still saves work over rebuild() because
        // the partition step is a re-partition rather than the initial
        // partition. Today both are equivalent in TreePartitioner.
        CANOPY_RESET_TIMERS();
        {
            CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_REBALANCE_TOTAL );
            { CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_BUILDER_BUILD );
              auto positions = Cabana::slice<PositionIdx>( particles );
              _builder.build( positions, _num_local ); }
            _finish_topology_change<PositionIdx>( particles );
        } // TIMER_REBALANCE_TOTAL destructs here
        CANOPY_PRINT_REBALANCE_TIMERS( _comm );
        CANOPY_PRINT_COMMPLAN_TIMERS( _comm );
    }

    // -----------------------------------------------------------------------
    // rebuild(): heavy maintenance. Full do-over (e.g. particles escaped
    // the bounding box).
    //
    //   build → partition → build → sort_by_leaf → build → comm_plan.build
    //   → upward.setup → downward.setup → p2p.setup
    // -----------------------------------------------------------------------
    template <int PositionIdx, int ChargeIdx, class AoSoA>
    void rebuild( AoSoA& particles )
    {
        _full_setup<PositionIdx, ChargeIdx>( particles, _num_local );
    }

    // -----------------------------------------------------------------------
    // auto_maintain(): pick the cheapest valid maintenance flow based on
    // the current particle state, and run it.
    //
    // Decision logic (each step's check is globally consistent across ranks):
    //   1. needs_rebuild (any particle escaped the bounding box)  → rebuild()
    //   2. tree topology changed (set of cell keys differs after a fresh
    //      build of the global tree)                              → rebalance()
    //   3. otherwise                                              → migrate()
    //
    // The topology check works because TreeBuilder owns a globally-replicated
    // cell list — every rank computes the same set of Morton keys after
    // build(), so every rank reaches the same MaintenanceAction without an
    // additional collective.
    // -----------------------------------------------------------------------
    template <int PositionIdx, int ChargeIdx, class AoSoA>
    MaintenanceAction auto_maintain( AoSoA& particles )
    {
        // Ensure any pending device writes to particle positions
        // (e.g. from integrate_particles) are visible before we read
        // them on the host or feed them back into the builder. On
        // unified-memory APUs (MI300A) this fence is required for
        // correctness, not just timing.
        Kokkos::fence( "auto_maintain: pre-positions" );

        // 1) Bounding-box escape ⇒ full rebuild.
        {
            auto positions = Cabana::slice<PositionIdx>( particles );
            if ( _builder.needs_rebuild( positions, _num_local ) )
            {
                _full_setup<PositionIdx, ChargeIdx>( particles, _num_local );
                return MaintenanceAction::Rebuild;
            }
        }

        // Snapshot the current global cell-key set, then rebuild the tree
        // from current positions and compare. The cell list is identical on
        // every rank, so the comparison is deterministic.
        std::unordered_set<MortonKey> old_keys;
        old_keys.reserve( _builder.cells().size() );
        for ( const auto& c : _builder.cells() )
            old_keys.insert( c.key );

        {
            auto positions = Cabana::slice<PositionIdx>( particles );
            _builder.build( positions, _num_local );
        }

        bool topology_changed = ( _builder.cells().size() != old_keys.size() );
        if ( !topology_changed )
        {
            for ( const auto& c : _builder.cells() )
            {
                if ( old_keys.find( c.key ) == old_keys.end() )
                {
                    topology_changed = true;
                    break;
                }
            }
        }

        if ( topology_changed )
        {
            // 2) Topology changed ⇒ full rebalance (repartition + comm_plan
            //    rebuild + setups). _builder.build() already happened above.
            _finish_topology_change<PositionIdx>( particles );
            return MaintenanceAction::Rebalance;
        }

        // 3) Topology stable ⇒ cheap migrate path (reuse comm_plan).
        //    _builder is already freshly built; redistribute uses its keys.
        RedistributeResult rr =
            _partitioner.redistribute( _builder, particles, _num_local );
        (void)rr;
        _num_local = _partitioner.num_local_particles();
        _finish_topology_stable<PositionIdx>( particles );
        return MaintenanceAction::Migrate;
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    int num_local_particles() const { return _num_local; }

    const potential_view_type& potential() const { return _potential; }
    const gradient_view_type& gradient() const { return _gradient; }

    // Read-only access to internal sweep stages, primarily for tests and
    // diagnostics (e.g. asserting that the M2L bin-edge fallback path was
    // exercised). Not part of the supported runtime API.
    const downward_type& downward() const { return _downward; }

    const builder_type& builder() const { return _builder; }
    const partitioner_type& partitioner() const { return _partitioner; }
    const comm_plan_type& comm_plan() const { return _comm_plan; }

  private:
    // -----------------------------------------------------------------------
    // _full_setup: shared body for setup() and rebuild()
    // -----------------------------------------------------------------------
    template <int PositionIdx, int ChargeIdx, class AoSoA>
    void _full_setup( AoSoA& particles, int num_local_before )
    {
        CANOPY_RESET_TIMERS();
        {
            CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_SETUP_TOTAL );

            // Step 1: initial tree build on caller-provided distribution
            {
                CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_BUILDER_BUILD );
                auto positions = Cabana::slice<PositionIdx>( particles );
                _builder.build( positions, num_local_before );
            }

            // Step 2: partition (migrates particles across ranks)
            {
                CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_PARTITION );
                _partitioner.partition( _builder, particles, num_local_before );
            }
            _num_local = _partitioner.num_local_particles();

            // Step 3: rebuild for migrated particles (accumulated into
            // TIMER_BUILDER_BUILD via += in the registry)
            {
                CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_BUILDER_BUILD );
                auto positions = Cabana::slice<PositionIdx>( particles );
                _builder.build( positions, _num_local );
            }

            // Step 4: sort AoSoA by leaf (invalidates particle_keys)
            {
                CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_SORT_BY_LEAF );
                _partitioner.sort_particles_by_leaf( _builder, particles );
            }

            // Step 5: rebuild so particle_keys match sorted AoSoA order
            {
                CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_BUILDER_BUILD );
                auto positions = Cabana::slice<PositionIdx>( particles );
                _builder.build( positions, _num_local );
            }

            // Step 5b: re-partition against the FINAL tree, then rebuild +
            // re-sort. At scale (~4e8 particles) the post-migration build can
            // produce a tree with substantially fewer cells than the
            // pre-partition build (see MI300A investigation: 4.2M -> 867k).
            // Without this step, cell_owner_map (populated against the larger
            // pre-partition tree) is stale relative to the final cells passed
            // to comm_plan.build, which silently produces a phantom-send M2M
            // plan and an MPI_ERR_TRUNCATE later in the upward sweep.
            // The repeat partition is expensive (extra Zoltan2 + migrate);
            // expected to be optimized once the underlying tree-divergence
            // bug is fixed.
            {
                CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_PARTITION );
                _partitioner.partition( _builder, particles, _num_local );
            }
            _num_local = _partitioner.num_local_particles();
            {
                CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_BUILDER_BUILD );
                auto positions = Cabana::slice<PositionIdx>( particles );
                _builder.build( positions, _num_local );
            }
            {
                CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_SORT_BY_LEAF );
                _partitioner.sort_particles_by_leaf( _builder, particles );
            }
            {
                CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_BUILDER_BUILD );
                auto positions = Cabana::slice<PositionIdx>( particles );
                _builder.build( positions, _num_local );
            }

            // Distribution-based default softening (no-op if the caller passed
            // an explicit softening, or after the first auto computation). The
            // global bounding box and total particle count are available now
            // that the tree is built.
            _init_auto_softening();

            // Step 6: communication plan
            {
                CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_COMM_PLAN_BUILD );
                _comm_plan.build( _builder.cells(), _partitioner.ownership(),
                                  _partitioner.cell_owner_map(),
                                  _replication_depth );
            }
            // Tree topology and comm plan just changed; the cached M2L
            // interaction list must be rebuilt on the next solve.
            _downward.invalidate_interaction_list();

            // Step 7: setup sweeps and P2P (not individually timed)
            _upward.setup( _builder.cells(), _partitioner.cell_owner_map(),
                           _builder.particle_keys(), _num_local );
            _downward.setup( _upward, _num_local );
            _p2p.setup( _builder, _partitioner, _comm_plan );

        } // TIMER_SETUP_TOTAL destructs here
        CANOPY_PRINT_SETUP_TIMERS( _comm );
        CANOPY_PRINT_COMMPLAN_TIMERS( _comm );
    }

    // -----------------------------------------------------------------------
    // _finish_topology_change: complete a maintenance flow that has
    // already changed (or rebuilt) the tree topology. Picks up at the
    // repartition step and runs through all setups including a fresh
    // comm_plan build.
    // -----------------------------------------------------------------------
    template <int PositionIdx, class AoSoA>
    void _finish_topology_change( AoSoA& particles )
    {
        { CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_REPARTITION );
          _partitioner.repartition( _builder, particles, _num_local ); }
        _num_local = _partitioner.num_local_particles();

        { CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_BUILDER_BUILD );
          auto positions = Cabana::slice<PositionIdx>( particles );
          _builder.build( positions, _num_local ); }

        { CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_SORT_BY_LEAF );
          _partitioner.sort_particles_by_leaf( _builder, particles ); }

        { CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_BUILDER_BUILD );
          auto positions = Cabana::slice<PositionIdx>( particles );
          _builder.build( positions, _num_local ); }

        // Re-partition against the FINAL tree so cell_owner_map matches the
        // cells comm_plan will see. See bug 1 note in _full_setup.
        { CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_REPARTITION );
          _partitioner.repartition( _builder, particles, _num_local ); }
        _num_local = _partitioner.num_local_particles();
        { CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_BUILDER_BUILD );
          auto positions = Cabana::slice<PositionIdx>( particles );
          _builder.build( positions, _num_local ); }
        { CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_SORT_BY_LEAF );
          _partitioner.sort_particles_by_leaf( _builder, particles ); }
        { CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_BUILDER_BUILD );
          auto positions = Cabana::slice<PositionIdx>( particles );
          _builder.build( positions, _num_local ); }

        { CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_COMM_PLAN_BUILD );
          _comm_plan.build( _builder.cells(), _partitioner.ownership(),
                            _partitioner.cell_owner_map(), _replication_depth ); }
        // Tree topology and comm plan just changed; invalidate the cache.
        _downward.invalidate_interaction_list();

        Kokkos::fence( "_finish_topology_change: pre-setup" );
        _upward.setup( _builder.cells(), _partitioner.cell_owner_map(),
                       _builder.particle_keys(), _num_local );
        _downward.setup( _upward, _num_local );
        _p2p.setup( _builder, _partitioner, _comm_plan );
        CANOPY_PRINT_COMMPLAN_TIMERS( _comm );
    }

    // -----------------------------------------------------------------------
    // _finish_topology_stable: complete a migrate flow. Tree topology is
    // assumed unchanged so comm_plan is reused. _num_local must already
    // reflect the post-migration particle count.
    // -----------------------------------------------------------------------
    template <int PositionIdx, class AoSoA>
    void _finish_topology_stable( AoSoA& particles )
    {
        // Rebuild particle_keys for the (now migrated) particles
        { CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_BUILDER_BUILD );
          auto positions = Cabana::slice<PositionIdx>( particles );
          _builder.build( positions, _num_local ); }

        { CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_SORT_BY_LEAF );
          _partitioner.sort_particles_by_leaf( _builder, particles ); }

        { CANOPY_SCOPED_TIMER( Canopy::Profiling::TIMER_BUILDER_BUILD );
          auto positions = Cabana::slice<PositionIdx>( particles );
          _builder.build( positions, _num_local ); }

        // Reuse existing comm_plan (topology unchanged).
        Kokkos::fence( "_finish_topology_stable: pre-setup" );
        _upward.setup( _builder.cells(), _partitioner.cell_owner_map(),
                       _builder.particle_keys(), _num_local );
        _downward.setup( _upward, _num_local );
        _p2p.setup( _builder, _partitioner, _comm_plan );
        CANOPY_PRINT_COMMPLAN_TIMERS( _comm );
    }

    // -----------------------------------------------------------------------
    // Members
    // -----------------------------------------------------------------------
    MPI_Comm _comm;
    int _replication_depth;

    builder_type _builder;
    partitioner_type _partitioner;
    comm_plan_type _comm_plan;
    upward_type _upward;
    downward_type _downward;
    p2p_type _p2p;

    int _num_local;

    // Softening configuration. _softening_input < 0 selects distribution-based
    // auto-softening (computed once in _full_setup); >= 0 is an explicit value.
    // _softening_initialized guards the one-shot auto computation so it is
    // fixed at the first setup() and not recomputed on later rebuilds.
    double _softening_input;
    bool _softening_initialized;
    // Fraction of the mean inter-particle spacing used as the auto-softening
    // length. 0.1 is a standard collisionless-N-body choice and matches the
    // smallest softening empirically observed to keep the clustering test
    // stable on MI300A.
    static constexpr double SOFTENING_FACTOR = 0.1;

    // Derive and apply the auto-softening length from the current global
    // bounding box and total particle count. No-op once softening has been
    // set (explicit value, or a prior auto computation).
    void _init_auto_softening()
    {
        if ( _softening_initialized )
            return;

        const auto& box = _builder.root_box();
        double volume = 1.0;
        for ( int d = 0; d < 3; d++ )
            volume *= ( box.max[d] - box.min[d] );

        long long n_total = 0;
        for ( const auto& c : _builder.cells() )
            if ( c.key == ROOT_KEY )
            {
                n_total = static_cast<long long>( c.global_count );
                break;
            }

        double eps = 0.0;
        if ( n_total > 0 && volume > 0.0 )
            eps = SOFTENING_FACTOR *
                  std::cbrt( volume / static_cast<double>( n_total ) );

        _p2p.set_softening( static_cast<Scalar>( eps ) );
        _softening_initialized = true;

        int rank = 0;
        MPI_Comm_rank( _comm, &rank );
        if ( rank == 0 )
            std::printf( "[Canopy] auto softening eps=%.6g "
                         "(factor=%.3g, N=%lld, bbox_volume=%.6g)\n",
                         eps, SOFTENING_FACTOR, n_total, volume );
    }

    potential_view_type _potential;
    gradient_view_type _gradient;
};

template <class MemorySpace, class ExecutionSpace, class Scalar = double,
          int P_ORDER = 8, int NComps = 1>
std::shared_ptr<Solver<MemorySpace, ExecutionSpace, Scalar, P_ORDER, NComps>>
createSolver( MPI_Comm comm, int ncrit, int max_depth,
              std::array<double, 3> bounding_box_tol, double ncrit_tol,
              int replication_depth, double imbalance_tolerance = 0.05 )
{
    return std::make_shared<Solver<MemorySpace, ExecutionSpace, Scalar, P_ORDER, NComps>>(
        comm, ncrit, max_depth, bounding_box_tol, ncrit_tol,
        replication_depth, imbalance_tolerance );
}

} // namespace Canopy

#endif // CANOPY_SOLVER_HPP
