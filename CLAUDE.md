# Canopy

## System detection

Build and run commands differ by system. Before building or running anything,
run `hostname` and match the result against the table below. Then Read the
matching system-specific instructions file and follow it for the rest of the
session.

| Hostname pattern | Instructions file          |
| ---------------- | -------------------------- |
| `tuolumne*`      | `docs/tuolumne/claude.md`  |
| `dane*`          | `docs/dane/claude.md`      |

The pattern is the alphabetic prefix of the host (e.g. `dane1234` matches
`dane*`, `lassen708` matches `lassen*`). To add support for a new system,
create `docs/<system>/claude.md` and add a row above.

Each `docs/<system>/` directory also holds a `spack.yaml` snapshot — a copy of
the project's spack environment file for that system, kept as a record of the
exact spec set the environment was concretized from. When the live environment
(`~/spack_envs/<system>_trilinos/spack.yaml`) changes, update the matching
snapshot in the same change.

If the hostname does not match any row, or the matching file is missing one of
the required sections below, ask the user to fill in the gap and update (or
create) the doc before proceeding.

### Required sections in every `docs/<system>/claude.md`

1. **Spack environment** — the `spack env activate ...` command that must be
   run before compiling or running any binary from this library.
2. **CMake args** — system-specific args that must be passed to `cmake` (or to
   any helper bash script that wraps `cmake`).
3. **Build command** — how to build a target on this system. Default:
   `make [EXECUTABLE]` (the user specifies the target when appropriate). If
   the system installs via spack, the build command is `spack install`
   instead. Every `docs/<system>/claude.md` must state which of the two
   applies.
4. **Run command for binaries** — the command template for running a built
   binary. Default starting point:
   `mpirun --oversubscribe -n [num_procs] [EXECUTABLE] [EXTRA_ARGS]`. Replace
   `mpirun` with `flux run`, `srun`, or whatever the system uses.
5. **Job-scheduler batch template** — if the system has a scheduler (flux,
   slurm, …), include a template batch script that can be filled in and
   submitted (e.g. `flux batch <script>`) to run binaries when the user is
   not inside an interactive allocation. Save concrete scripts to
   `scripts/<hostname>/` (create the directory if it does not exist).
6. **Running non-test binaries** — when asked to run something other than a
   test (e.g. an `examples/` problem), ask the user for the example name and
   args, then plug them into sections 4 and 5.

The required tests themselves (names + MPI rank counts) are project-wide and
live in [Minimum test set](#minimum-test-set) below, not in the per-system
doc. The per-system doc only describes *how* to run any given test on that
machine.

## Minimum test set

These tests must pass before any code change ships. Each entry lists the
test name and the MPI rank counts it must be run at. Use the run command and
batch template from the active system's `docs/<system>/claude.md` to execute
them.

The minimum test set:

- `Canopy_Test_MultiSolve_MPI_SERIAL` — at 1, 2, 3, 4, 5, 6 ranks.

Other tests in `tests/` may be built and run at the user's discretion. If
you believe an additional test should be built to verify the correctness of
a new feature, confirm with the user before adding it to the required test
set for that session's feature work.

## Plans

When creating plans via plan mode, save plan files to `./plans/` in this
repository, not the default plan location.

## General guidelines

- **Checkpoint commits in plans.** When planning a large code change, include
  explicit checkpoints in the plan file where progress should be committed.
  If a later step fails (test failure, performance regression), we can roll
  back to the nearest checkpoint and retry.
- **Follow `.clang-format`.** If `.clang-format` exists at the repo root,
  follow its formatting rules for any C/C++ code you write or edit. If it
  does not exist, ignore this rule. For fast formatting, run the
  `clangformat.sh` script at the repo root — it formats every `.cpp`/`.hpp`
  in the tree (skipping `*build*` directories) in place with `clang-format -i`.
- **Keep `README.md` in sync.** When a public-facing API changes, or when the
  arguments accepted by an example problem change, update `README.md` in the
  same change so its documentation stays accurate.
- **Track optimization opportunities.** If, after completing a new
  implementation, you notice an optimization opportunity (a performance or
  scalability refinement that is not a correctness issue), ask the user whether
  they want it recorded in the "Future Optimizations" section of `README.md`
  for tracking. Only add it if they say yes.
