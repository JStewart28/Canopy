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

Canopy distributes work across MPI ranks by partitioning the leaf cells of the
adaptive octree across ranks and then migrating particles to the rank that owns
their containing leaf. The partitioning itself is delegated to the
[Zoltan2](https://trilinos.github.io/zoltan2.html) geometric load balancer. The
implementation lives in
[`src/Canopy_TreePartitioner.hpp`](../src/Canopy_TreePartitioner.hpp)
(`partition_leaves`, `derive_internal_ownership`, and `migrate_particles`).

This section documents the **interface boundary** between Canopy and Zoltan 2 —
what Canopy hands to Zoltan 2, what Zoltan 2 hands back, and what Canopy does
with the result. It deliberately does not describe Zoltan 2's internal
partitioning mechanics.

### What Canopy feeds into Zoltan 2

The octree built by `TreeBuilder` is replicated globally: every rank holds the
identical set of cells. Canopy walks that cell list and gathers **one Zoltan 2
object per leaf cell**. For each leaf it prepares:

- **A global ID** — the leaf's index in the collected leaf list
  (`0 .. num_leaves-1`). Because the cell list is identical on every rank, this
  index is the same on every rank, so a part assignment expressed in terms of
  these IDs is meaningful everywhere.
- **Geometric coordinates** — the leaf cell's center `(x, y, z)`. Zoltan 2's
  geometric partitioner treats each leaf as a point at its cell center, so
  spatially-neighboring leaves tend to be assigned to the same rank (locality
  that later reduces ghost-exchange communication).
- **A weight** — the leaf's *global* particle count (`c.global_count`). The
  weight is what the partitioner balances: the goal is an even distribution of
  *particles* (hence work), not an even distribution of *cells*. Using the
  global count means every rank supplies identical weights, consistent with the
  replicated tree.

These four arrays (IDs, x, y, z, weights) are wrapped in a
`Zoltan2::BasicVectorAdapter` — the geometric-coordinates-plus-weights adapter.
The partitioning problem is configured with:

- `algorithm = "multijagged"` — the multi-jagged (MJ) geometric partitioner.
  (The deterministic `rcb` algorithm is not used because it fails on the
  Tuolumne platform.)
- `num_global_parts = comm size` — one part per MPI rank.
- `imbalance_tolerance` — the allowed load imbalance (constructor argument,
  default 5%).

Because the multi-jagged algorithm is **non-deterministic**, letting every rank
solve independently would produce *different* assignments per rank, which would
in turn trigger a spurious, potentially multi-GB re-migration. To avoid this,
Canopy solves the partitioning problem on **rank 0 only**, running Zoltan 2 over
a Teuchos `SerialComm` (so Zoltan 2's internal messaging never enters MPI at
all). The single-rank solve is valid precisely because rank 0 already holds the
complete, globally-replicated leaf set.

### What Zoltan 2 returns

Zoltan 2's solution is a **part assignment**: a list with one entry per leaf,
where entry `i` is the part number (i.e. the target MPI rank, `0 .. comm_size-1`)
assigned to the leaf with global ID `i`. Canopy reads this back via
`getSolution().getPartListView()` on rank 0.

### What Canopy does with the output

1. **Broadcast.** Rank 0 broadcasts the per-leaf part array to all ranks with a
   single `MPI_Bcast`. After this, every rank holds the same leaf→rank
   assignment despite only rank 0 having run the solver.
2. **Build the leaf ownership map.** Each rank turns the broadcast array into a
   `MortonKey → owning-rank` map for the leaves (`partition_leaves`' return
   value).
3. **Derive internal-cell ownership** (`derive_internal_ownership`). Leaf
   ownership comes straight from Zoltan 2. Internal (non-leaf) cells are
   assigned by a separate, non-Zoltan rule:
   - Cells at or above the configured `replication_depth` are marked
     `OWNER_SHARED` (replicated on every rank) — the coarse layers are cheap
     enough to duplicate.
   - Deeper internal cells are owned by the rank holding the most *descendant*
     particles, tallied by walking each leaf's particle count up to the root.
     Ties are broken by lowest rank so every rank derives the identical owner.
4. **Migrate particles** (`migrate_particles`). Using the assembled
   `MortonKey → rank` ownership map, each rank looks up the destination rank of
   every local particle (via its leaf key), then performs a coalesced
   point-to-point exchange to move each particle to its owning rank. Particles
   whose key resolves to `OWNER_SHARED` or is unresolved stay on the current
   rank.

The cached leaf assignment is also retained so that ownership can be refreshed
against a changed tree (`refresh_ownership_for_current_tree`) *without* re-running
the non-deterministic Zoltan 2 solve — again avoiding a spurious large-scale
re-migration.
