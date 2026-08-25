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
`systems/<system>/claude.md` to execute them (the
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
