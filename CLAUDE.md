# Canopy

## System detection

Build and run commands differ by system. Before building or running anything,
run `hostname` and match the result against the table below. Then Read the
matching system-specific instructions file and follow it for the rest of the
session.

| Hostname pattern | Instructions file          |
| ---------------- | -------------------------- |
| `tuolumne*`      | `systems/tuolumne/claude.md`  |
| `dane*`          | `systems/dane/claude.md`      |

The pattern is the alphabetic prefix of the host (e.g. `dane1234` matches
`dane*`, `lassen708` matches `lassen*`). To add support for a new system,
create `systems/<system>/claude.md` and add a row above.

Each `systems/<system>/` directory also holds a `spack.yaml` snapshot — a copy of
the project's spack environment file for that system, kept as a record of the
exact spec set the environment was concretized from. When the live environment
(`~/spack_envs/<system>_trilinos/spack.yaml`) changes, update the matching
snapshot in the same change.

If the hostname does not match any row, or the matching file is missing one of
the required sections below, ask the user to fill in the gap and update (or
create) the doc before proceeding.

### Required sections in every `systems/<system>/claude.md`

1. **Spack environment** — the `spack env activate ...` command that must be
   run before compiling or running any binary from this library.
2. **CMake args** — system-specific args that must be passed to `cmake` (or to
   any helper bash script that wraps `cmake`).
3. **Build command** — how to build a target on this system. Default:
   `make [EXECUTABLE]` (the user specifies the target when appropriate). If
   the system installs via spack, the build command is `spack install`
   instead. Every `systems/<system>/claude.md` must state which of the two
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

Which tests a change must pass is specified by the task being worked (its exit
criterion), not by this file and not by the per-system doc. The per-system doc
only describes *how* to run any given test on that machine.

## Builds

**Only build the targets a task explicitly needs. Never a full build** — a
whole-project `make -j` here takes long enough that it is never the right
default, and a task that names two test stems needs two targets.

- Build exactly the targets named by the task's exit criterion, and nothing
  else. `cmake/test_harness/test_harness.cmake` generates one target per
  (stem, backend): `Canopy_Test_<Stem>_MPI_SERIAL` for an MPI stem,
  `Canopy_Test_<Stem>_SERIAL` for a non-MPI one.
- `ctest -N` lists what is registered without building or running anything.
- If something outside the task's list looks like it needs building, say why
  and ask before building it.

## Tests

The tests that must pass are **whatever the task specifies**. A task's exit
criterion names the stems and the MPI rank counts, and that list is the gate
for that change — do not substitute a broader or a narrower run for it.

Match ctest entries with an **anchored** regex, or an unanchored stem name
pulls in its neighbours (`-R MultiSolve` also matches `SolveFusedM2L`'s host
stem, `-R CartesianTaylor` also matches `CartesianTaylorSolve`):

```bash
ctest --output-on-failure -R '^Canopy_Test_(StemA|StemB)_MPI_SERIAL_np_[1-6]$'
```

Tests carry CTest labels in [tests/CMakeLists.txt](tests/CMakeLists.txt), and
the labels stay useful for selecting a diagnostic sweep — `ctest -L unit`
covers utilities, math kernels and individual FMM-phase/component tests, which
is how you localize *which* phase a failure comes from. Relabeling a test
changes what other work is held to, so confirm with the user before moving one
between labels.

## Plans

When creating plans via plan mode, save plan files to `./plans/` in this
repository, not the default plan location.

## General guidelines

- **Checkpoint commits in plans.** When planning a large code change, include
  explicit checkpoints in the plan file where progress should be committed.
  If a later step fails (test failure, performance regression), we can roll
  back to the nearest checkpoint and retry.
- Do not clang format.
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
- **Comments in code.** Keep comments succinct. Do not comment code whose
  behavior is obvious. 

## Math in markdown
Write math with KaTeX delimiters, not Doxygen ones:
- Inline: `$ ... $`  — NOT `\f$ ... \f$`
- Display: `$$ ... $$` on their own lines, blank line above and below — NOT `\f[ ... \f]`

`\f[`/`\f$` are Doxygen-only. In a plain markdown reader (VSCode preview,
GitHub, mdBook) they don't open a math region, so the body is parsed as prose
and CommonMark strips the backslash from every escaped punctuation character —
`\;` becomes `;`, `\,` becomes `,`, `\_` disappears — leaving unreadable output.

KaTeX delimiter rules worth respecting:
- No space just inside the delimiters: `$x + y$`, not `$ x + y $`.
- Don't put a digit immediately after a closing `$`.
- Keep inline math on one line.
- For a literal dollar sign in prose, escape it: `\$`.
