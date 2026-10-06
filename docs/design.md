# Canopy Design Notes

## Overview

This document is the authoritative record of the algorithmic design decisions
in the Canopy library. It describes *why* the core algorithms are structured
the way they are and how Canopy interfaces with the external libraries it
depends on. Implementation details that live in the code (exact function
signatures, buffer management, etc.) are documented at their definition; this
document captures the design-level decisions that are not obvious from any
single source file.

## Load Balancing

Canopy distributes work across MPI ranks by partitioning the cells of the
adaptive octree across ranks and then migrating particles to the rank that owns
their containing leaf. Every **non-shared** cell — every leaf, and every
internal cell deeper than `replication_depth` — is a vertex of a weighted graph
that [ParMETIS](https://github.com/KarypisLab/ParMETIS) partitions in parallel.
Internal cells at depth $\le$ `replication_depth` stay `OWNER_SHARED`
(replicated on every rank). The implementation lives in
[`src/Canopy_TreePartitioner.hpp`](../src/Canopy_TreePartitioner.hpp)
(`partition_cells`, `derive_internal_ownership`, `vote_internal_owners`,
`refresh_ownership_for_current_tree` and `migrate_particles`).

ParMETIS is called directly (`ParMETIS_V3_PartKway`,
`ParMETIS_V3_AdaptiveRepart`), not through Zoltan2. ParMETIS is a host-only C
library, so the partition runs on the host for every `ExecutionSpace`, and no
device fence sits on the partition path. Zoltan2's `PartitioningProblem`
instantiates its multi-jagged algorithm for any adapter type, so routing
through it would keep that code in every binary.

### The graph

The octree built by `TreeBuilder` is replicated globally: every rank holds the
identical cell list, built from all-reduced counts. Every rank therefore builds
the identical vertex list without gathering any tree data.

- **Vertices and their order.** One vertex per non-shared cell, in Morton
  pre-order: each key is shifted to the deepest vertex depth, and an ancestor
  sorts before its first descendant. A contiguous block of this list is a
  spatially compact region.
- **Weights: one balance constraint per band.** Constraint 0 is a leaf's
  `global_count` and 0 for an internal cell: the particle work of P2P, P2M and
  L2P. The non-shared depths `replication_depth + 1 .. deepest` are split into
  $B = \min(3, \text{number of those depths})$ contiguous bands of near-equal
  depth count, and constraint $1 + b$ weighs 1 for every cell, leaf or
  internal, whose depth falls in band $b$: the M2M, M2L and L2L work of that
  band. Balancing the bands spreads coarse cells across ranks instead of
  leaving them wherever their particles happen to sit. `B` is capped because
  each constraint costs ParMETIS partition quality. A band holding fewer than
  4 cells per rank is dropped: ParMETIS cannot balance it, and trying costs
  every other constraint, the particle one included (a 1200-particle two-scale
  tree with a 3-cell band left two of five ranks with no particles).
  `TreePartitioner::bands()` reports the bands the last partition kept.
- **Edges.** Parent-child edges between non-shared cells (M2M and L2L
  traffic) weigh 27; same-depth face, edge and corner neighbours (near-field
  and M2L proximity, computed from Morton keys because the interaction list is
  built after the partition) weigh 1. One parent-child edge outweighs all 26
  neighbours of a cell: with unit weights the neighbour edges dominate the cut,
  and ParMETIS cuts more parent-child pairs than the majority-vote rule would
  on the same leaves.

### Distribution and solve

- **`partition()`**: rank `r` supplies the `r`-th contiguous block of the
  Morton list, and `ParMETIS_V3_PartKway` partitions from there.
- **`repartition()`**: each rank supplies the cells it owned before; a cell new
  to the tree goes to the block rule. `ParMETIS_V3_AdaptiveRepart` starts from
  that distribution as the current partition, with unit vertex sizes and an
  inter-processor-communication-to-redistribution ratio of 100, which limits
  migration.
- ParMETIS needs each rank's global IDs contiguous, so vertices are numbered by
  (supplying rank, Morton position). It rejects a rank with no vertices, so
  those ranks are split off the solve's communicator; they still receive the
  result.
- Every constraint's tolerance is `1 + imbalance_tolerance` (constructor
  argument, default 5%); the seed is fixed. `np == 1` assigns every cell to rank
  0 without calling ParMETIS.
- Each rank's part list is all-gathered (`MPI_Allgatherv`), so every rank holds
  the full key $\to$ rank map that `CommunicationPlan` and the sweeps read. No
  rank solves ahead of the others.

These conditions throw `std::runtime_error` on every rank, naming the
condition: ranks disagreeing on the vertex count (one `MPI_Allreduce` of min
and max before the solve), a non-`METIS_OK` return on any rank, and a part
outside `[0, comm_size)`. A rank left with no leaf is legal; on a small tree it
can be the balanced answer.


### What Canopy does with the output

1. **Ownership** (`derive_internal_ownership`). Every partitioned cell takes its
   part. Internal cells at depth $\le$ `replication_depth` are `OWNER_SHARED`.
   A deeper internal cell the partition did not assign falls back to the vote
   rule (`vote_internal_owners`): the rank owning the most descendant particles,
   ties to the lowest rank so every rank derives the same owner.
2. **Migrate particles** (`migrate_particles`). Each rank looks up the owner of
   every local particle's leaf and performs a coalesced point-to-point exchange.
   Particles whose key resolves to `OWNER_SHARED` or is unresolved stay put.
3. **Refresh against the final tree** (`refresh_ownership_for_current_tree`).
   The solver rebuilds the tree after migration and refreshes ownership
   against it without re-partitioning or migrating again. Every non-shared cell
   present in the cached assignment keeps its partitioned owner; a leaf absent
   from it goes to the rank holding the most of its local particles, and an
   absent internal cell to the vote rule.
