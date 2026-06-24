# Canopy

## Background / task logs

Ongoing multi-phase problems are documented in [`tasks/`](tasks/). Each file
records *why* a problem is being worked and *how* it is being attacked, with a
dated progress log, so a later session can pick up the thread. At the start of a
session that touches one of these areas, read the relevant `tasks/<topic>.md`
first for context, and append progress there as work lands. Current logs:

- [`tasks/near-field-softening.md`](tasks/near-field-softening.md) — the
  near-field/P2P cost blowup under roll-up caused by the unsoftened FMM far field
  (the `near_softening_factor` floor); multi-phase diagnose → fix effort.

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

## Build & run profile

Orthogonal to *which system* you are on is *how the work is being done* this
session, which changes the spack env to activate, where binaries land, and how
the test gate runs:

- **manual** — spack provides only Canopy's *dependencies*; binaries are
  hand-compiled out-of-tree into `build-<system>/` and live there. The fast dev
  loop (`run_cmake_<system>.sh`, ccache, incremental `make -j <target>`). The
  gate is `ctest` in the build dir.
- **spack** — `spack develop canopy` + `spack install +testing`; binaries are
  installed onto `PATH` as `Canopy_Test_<name>_<DEVICE>`. Prod/integration runs
  always build this way; there is a **dev** env and, optionally, a **prod** env.
  No build tree survives, so the gate runs the installed binaries via the
  scheduler at ranks 1–6 (not `ctest`).

The choice is recorded **per checkout** in `scripts/<system>/profile.local.sh`
(gitignored), which overrides the committed defaults in
`scripts/<system>/profile.defaults.sh`. Both are sourced by
[`scripts/lib/canopy_env.sh`](scripts/lib/canopy_env.sh) — the resolver every
batch script sources first to activate spack and locate binaries. A missing
`profile.local.sh` ⇒ the committed defaults (manual mode, the historical env and
`build-<system>/`), so existing checkouts keep working zero-config.

### Determining the profile at session start

Before the first build or run action in a session:

1. If `scripts/<system>/profile.local.sh` exists, read it and use it — **do not
   re-ask**.
2. Otherwise (or if the user says the mode changed) ask with **AskUserQuestion**:
   - **Build mode** — spack (spack develop + spack install) or manual
     (cmake + make)?
   - **spack** ⇒ the **dev** spack env (required) and **prod** spack env
     (optional).
   - **manual** ⇒ the dependency spack env to activate (default
     `${HOME}/spack_envs/<system>_trilinos`) and the build dir (default
     `build-<system>`).
3. Write/update `scripts/<system>/profile.local.sh` from
   `scripts/<system>/profile.defaults.sh` as the template, setting
   `CANOPY_BUILD_MODE`, `CANOPY_BIN_MODE` (`build-dir` for manual, `path` for
   spack), `CANOPY_SPACK_ENV` (dev env), `CANOPY_SPACK_PROD_ENV` (prod, spack
   only), and `CANOPY_BUILD_DIR` (manual). Leave the file absent to stay on the
   defaults.

The resolver exposes `canopy_exe <relpath|name>` (a build-dir path in manual
mode, a bare on-PATH name in spack mode) and honors `CANOPY_USE_PROD=1` to
select the prod env. Set `CANOPY_NO_SPACK_ACTIVATE=1` to preview what a script
would activate/run without touching spack (useful for dry-run validation).

### Required sections in every `docs/<system>/claude.md`

1. **Spack environment** — the concrete spack env path(s) for this system: the
   dependency/dev env baked into `scripts/<system>/profile.defaults.sh` as
   `CANOPY_SPACK_ENV`, and (if used) the prod env. Both build modes activate
   through the resolver (see [Build & run profile](#build--run-profile)); the
   doc records the concrete paths so `profile.local.sh` is fillable.
2. **CMake args** — system-specific args that must be passed to `cmake` (or to
   any helper bash script that wraps `cmake`).
3. **Build command** — document **both** modes: manual
   (`run_cmake_<system>.sh` + `make -j [TARGET]`, binaries in
   `$CANOPY_BUILD_DIR`) and spack (`spack install +testing +examples …`,
   binaries on `PATH`). The user specifies the target when appropriate.
4. **Run command for binaries** — the scheduler launch template (`flux run`,
   `srun`, or whatever the system uses). Binary location follows
   `CANOPY_BIN_MODE`: `$CANOPY_BUILD_DIR/…` in manual mode, the bare on-PATH
   name in spack mode — resolve either with `canopy_exe` from the resolver.
5. **Job-scheduler batch template** — if the system has a scheduler (flux,
   slurm, …), include a template batch script that can be filled in and
   submitted (e.g. `flux batch <script>`) to run binaries when the user is
   not inside an interactive allocation. Concrete scripts live in
   `scripts/<system>/` and `source scripts/lib/canopy_env.sh` first, then
   branch on `CANOPY_BIN_MODE` for the gate (ctest in the build dir vs the
   scheduler rank-loop on the installed binary).
6. **Running non-test binaries** — when asked to run something other than a
   test (e.g. an `examples/` problem), ask the user for the example name and
   args, then plug them into sections 4 and 5.

The required tests themselves (names + MPI rank counts) are project-wide and
live in [Minimum test set](#minimum-test-set) below, not in the per-system
doc. The per-system doc only describes *how* to run any given test on that
machine.

## Minimum test set

The required gate before any code change ships is **every test carrying the
`regression` CTest label**, run on the SERIAL backend at MPI ranks 1–6:

```bash
ctest --output-on-failure -L regression -R MPI_SERIAL
```

The `regression` label covers the full-pipeline FMM solve (`MultiSolve`) — it
composes the entire pipeline end-to-end, so if it passes the pipeline is
correct. Tests are tagged in [tests/CMakeLists.txt](tests/CMakeLists.txt); use
the run command and batch template from the active system's
`docs/<system>/claude.md` to execute them (the
`scripts/<system>/run_ctest_minset.*` wrappers run exactly this gate).

The complementary `unit` label covers utilities, math kernels, and individual
FMM-phase/component tests (`ctest -L unit`), including the single-tree
`SingleSolve` solve. These are not part of the ship gate but are the diagnostic
layer — run them to localize *which* phase a regression failure comes from, and
when changing a specific component. (`SingleSolve` is currently a Known Issue —
see README — so do not add it to the gate yet.) If you believe a new test
should be promoted into the `regression` gate, confirm with the user before
relabeling it.

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
- **Record known issues.** Known defects deferred to a later session are
  tracked in the "Known Issues" section of `README.md`. When a test failure or
  bug is confirmed but not fixed this session, note it there (what fails, how it
  reproduces, and whether it predates the current work).
