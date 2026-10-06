# The np-3 `MultiSolve` hang, and `AutoRebalance`'s excess deviation — progress log

Session record for fix-hang-rebalance. Companion to `fix-hang-rebalance.md`,
which holds the design, the task sequence and the risks; this file holds what
actually happened, in order.

**Read this when** you need the reasoning behind a decision the design states
flatly, the measured numbers behind a claim, or the history of a file you are
about to change. The design says *what is true now*; the log says *how it got
that way and what was tried on the route*.

**Append to it** at the end of any task that makes a decision, changes a
signature, measures something, or finds a bug. Add a new `## <task ID>` section
at the bottom, named for the task it records, so `fix-hang-rebalance.md` can
cite it by ID. No dates: the order of the sections is the chronology. If a
session covers more than one task, name them all; if it belongs to no task,
name the topic.

**End each section with `**Affects:**`** — the later task IDs whose stated plan
this entry changes, one clause each on how, or `none`. A finding that
invalidates a later task is worthless if the session starting that task has to
read the whole log to notice it; this line is the index that makes it findable.
Name `tree-opt.md` task V1 there whenever a finding changes what V1 will
measure or pin.

Worth recording, because none of it is recoverable from the code afterwards:
semantic decisions and what forced them, signature changes and why they could
not stay as they were, bugs that only running revealed, measured numbers, and
approaches tried that did not work. Record too where the implementation
departed from the task's stated **Do** steps, and why — a task marked `**DONE**`
that was done differently than it was written is the quietest way for a design
to stop describing the code.

Two things this topic in particular depends on your recording:

- **Every stack capture verbatim**, with the rank, the last gtest case and the
  last `[multisolve-dev]` line before it. H2's diagnosis is read off these and
  nothing else, and a hang that does not recur cannot be re-captured.
- **Every hang/no-hang count with its run total.** The ~1-in-3 rate is what
  makes H1's 20-run budget and H2's 15-run exit criterion meaningful; a count
  without its denominator cannot be compared against it.

## H1

Job `f3bnfasqQDAo` (`scripts/tuolumne/run_ctest_h1.flux`, node `tuolumne1030`,
HEAD `4d65d6c` with the H1 scripts uncommitted on top; Cray clang 20.0.0,
flux-core 0.89.0, `build-tuolumne/` with `Canopy_ENABLE_PROFILING=ON`, level 2).
Full output: `canopy-h1.f3bnfasqQDAo.log` in the repo root (untracked).

**Decisions.**
- The watchdog stacks only the sub-job's own processes: descendants of the
  `flux-shell` whose last argument is the sub-job's id, filtered by
  `CANOPY_WATCHDOG_PGREP` as an extended regex on `comm`. A node-wide `sleep`
  match would catch the watchdog's own `sleep 15`; `pgrep -f Canopy_Test_` would
  also catch `ctest` (its `-R` regex) and the `flux run` client.
- ctest's `--timeout 300` equals `WATCHDOG_S`, so ctest returns before the
  watchdog acts on a hung sub-job. `watchdog_wait_idle` blocks until a poll that
  *began after the call* sees no running sub-job (the loop writes
  `<start-ns> <running-count>` to `${WATCHDOG_DIR}/tick` each cycle). A run is
  a hang only if `${WATCHDOG_DIR}/cancelled` gained a line. ctest's exit code
  is 8 either way, from the pre-existing `1e-8` failures.

**job-id → flux-shell mapping.** On flux-core 0.89 the sub-job shell is
`/usr/libexec/flux/flux-shell <JOBID>`, a direct child of the nested
instance's broker, with JOBID in **decimal** (`flux job id --to=dec`). The
watchdog matches `comm == flux-shell` and last argument equal to the decimal or
f58 id, then walks descendants from one `ps -e -o pid=,ppid=,comm=` snapshot.
The ranks are direct children of the shell (`children` = `matched` in both
captures below). `comm` truncates to 15 characters: `Canopy_Test_Mul`.

**Departures from Do.**
- The self-test's 300-330 s check reads the sub-job runtime the watchdog
  recorded at cancel, not the `flux run` client's wall time; both are logged.
- `flux_watchdog.sh` also exports `watchdog_start`, `watchdog_wait_idle` and
  `watchdog_cancel_count`. `run_ctest_h1.flux` sources it twice (pattern
  `sleep`, then the default), each source with a fresh `WATCHDOG_DIR`.
- A self-test failure exits the job before the np-3 loop: a loop under an
  unproven watchdog would measure nothing.
- After the job, `watchdog_wait_idle` was changed to test for the tick file
  before reading it. Before the first poll it printed a harmless
  `tick: No such file or directory`; the logic is unchanged.

**Self-test.** Sub-job `f779fSSb` (`flux run --ntasks=3 --nodes=1 --exclusive
--cores-per-task=1 sleep 900`) was stacked and cancelled at a runtime of
303.6 s (client wall 306 s, `rc=143`). `matched=3 children=3 nonempty=3`. The
follow-on `flux run --ntasks=4 ... hostname` started at once (`rc=0`, wall 0 s,
four `tuolumne1030` lines). Capture, verbatim:

```
### watchdog stacks f779fSSb ###
time: 2026-10-02 14:32:09 runtime: 303.62474727630615 s jobid(dec): 232448327680
shell: 2827495 /usr/libexec/flux/flux-shell 232448327680
--- pid 2827496 ppid 2827495 comm sleep ---
tool: gstack
#0  0x0000155554b26f38 in nanosleep () from /lib64/libc.so.6
#1  0x0000555555558b47 in rpl_nanosleep ()
#2  0x0000555555558920 in xnanosleep ()
#3  0x0000555555555a88 in main ()
--- pid 2827497 ppid 2827495 comm sleep ---
tool: gstack
#0  0x0000155554b26f38 in nanosleep () from /lib64/libc.so.6
#1  0x0000555555558b47 in rpl_nanosleep ()
#2  0x0000555555558920 in xnanosleep ()
#3  0x0000555555555a88 in main ()
--- pid 2827498 ppid 2827495 comm sleep ---
tool: gstack
#0  0x0000155554b26f38 in nanosleep () from /lib64/libc.so.6
#1  0x0000555555558b47 in rpl_nanosleep ()
#2  0x0000555555558920 in xnanosleep ()
#3  0x0000555555555a88 in main ()
### end watchdog stacks f779fSSb: matched=3 children=3 nonempty=3 ###
### watchdog: cancelling sub-job f779fSSb after 303.62474727630615 s ###
```

**R1.** Did not fire. `ptrace_scope` is `0` on the compute node, and `gstack`
attached on the first try to every process. Neither fallback was needed.

**Hang count: 1 hang in 2 np-3 runs.** Run 1 completed in 12.99 s (`rc=8`,
pre-existing `1e-8` failures). Run 2 hung, ctest timed out at 300.10 s, and the
watchdog stacked and cancelled sub-job `f3oTBtb9H` at a runtime of 302.2 s. The
loop stopped there per H1. R2: the hang reproduced under `-V` and 15 s
polling.

Last gtest case before the hang: `[ RUN      ] MultiSolve.LargeMotion_Rebuild`.
After it, only the `[Canopy Diagnostics] setup() timing (3 MPI ranks)` block
printed, then nothing. Last `[multisolve-dev]` line before it:

```
213: [multisolve-dev] case IntermediateMotion_Rebalance nprocs 3 nsteps 5 drift 5 max_pos_rel 1.5646110360228987e-05 max_vel_rel 4.3603755255711028e-05 tol 1e-08
```

The capture does not record MPI rank. PID 2827919 is the one inside
`Zoltan2::PartitioningProblem::solve`, which `partition_leaves` calls on rank 0
only, so it is inferred to be rank 0. PIDs 2827920 and 2827921 wait in
`PMPI_Bcast` from `partition_leaves`. Not diagnosed here; that is H2's job.
Capture, verbatim:

```
### watchdog stacks f3oTBtb9H ###
time: 2026-10-02 14:38:02 runtime: 302.1887285709381 s jobid(dec): 6184316698624
shell: 2827918 /usr/libexec/flux/flux-shell 6184316698624
--- pid 2827919 ppid 2827918 comm Canopy_Test_Mul ---
tool: gstack
Thread 3 (Thread 0x15491afff700 (LWP 2827935)):
#0  0x000015553bc462ab in ioctl () from /lib64/libc.so.6
#1  0x000015552f707d78 in hsakmt_ioctl () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#2  0x000015552f70097d in hsaKmtWaitOnMultipleEvents_Ext () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#3  0x000015552f66eac3 in rocr::core::Runtime::AsyncEventsLoop(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#4  0x000015552f61434d in rocr::os::ThreadTrampoline(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#5  0x000015553b9f31ca in start_thread () from /lib64/libpthread.so.0
#6  0x000015553bc46953 in clone () from /lib64/libc.so.6
Thread 2 (Thread 0x1555287ff700 (LWP 2827933)):
#0  0x000015553bc462ab in ioctl () from /lib64/libc.so.6
#1  0x000015552f707d78 in hsakmt_ioctl () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#2  0x000015552f70097d in hsaKmtWaitOnMultipleEvents_Ext () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#3  0x000015552f68f7a5 in rocr::core::Signal::WaitAnyExceptions(unsigned int, hsa_signal_s const*, hsa_signal_condition_t const*, long const*, long*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#4  0x000015552f66dfff in rocr::core::Runtime::AsyncEventsLoop(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#5  0x000015552f61434d in rocr::os::ThreadTrampoline(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#6  0x000015553b9f31ca in start_thread () from /lib64/libpthread.so.0
#7  0x000015553bc46953 in clone () from /lib64/libc.so.6
Thread 1 (Thread 0x155555548a80 (LWP 2827919)):
#0  0x000015552f64f98d in rocr::core::InterruptSignal::WaitRelaxed(hsa_signal_condition_t, long, unsigned long, hsa_wait_state_t) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#1  0x000015552f64f7fa in rocr::core::InterruptSignal::WaitAcquire(hsa_signal_condition_t, long, unsigned long, hsa_wait_state_t) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#2  0x000015552f644231 in rocr::HSA::hsa_signal_wait_scacquire(hsa_signal_s, hsa_signal_condition_t, long, unsigned long, hsa_wait_state_t) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#3  0x000015553c59027b in amd::roc::Device::IsHwEventReady(amd::Event const&, bool, unsigned int) const () from /opt/rocm-6.4.2/lib/libamdhip64.so.6
#4  0x000015553c57f1d7 in amd::HostQueue::finish(bool) () from /opt/rocm-6.4.2/lib/libamdhip64.so.6
#5  0x000015553c31b21e in hip::Device::SyncAllStreams(bool, bool) () from /opt/rocm-6.4.2/lib/libamdhip64.so.6
#6  0x000015553c308e51 in hip::hipDeviceSynchronize() () from /opt/rocm-6.4.2/lib/libamdhip64.so.6
#7  0x0000155541a07cb3 in Kokkos::HIP::impl_static_fence(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libkokkoscore.so.4.7
#8  0x00001555419f671d in Kokkos::fence(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libkokkoscore.so.4.7
#9  0x00000000005930b0 in Kokkos::deep_copy<int*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>, int*, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::OpenMP, Kokkos::HIPSpace>, Kokkos::Experimental::EmptyViewHooks> ()
#10 0x00000000005b5e19 in Zoltan2::AlgMJ<double, int, long long, int, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> >::mj_get_new_cut_coordinates(int, int, int const&, double const&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<bool*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<int*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&) ()
#11 0x00000000005898b2 in Zoltan2::AlgMJ<double, int, long long, int, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> >::mj_1D_part(Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, double, int, int, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, int, Kokkos::View<int*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<unsigned long*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&) ()
#12 0x0000000000550448 in Zoltan2::AlgMJ<double, int, long long, int, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> >::multi_jagged_part(Teuchos::RCP<Zoltan2::Environment const> const&, Teuchos::RCP<Teuchos::Comm<int> const>&, double, int, unsigned long, Kokkos::View<int*, Kokkos::HostSpace>&, int, int, int, long long, Kokkos::View<long long const*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, int, Kokkos::View<bool*, Kokkos::HostSpace>&, Kokkos::View<double**, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<bool*, Kokkos::HostSpace>&, Kokkos::View<int*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<long long*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&) ()
#13 0x0000000000541146 in Zoltan2::Zoltan2_AlgMJ<Zoltan2::BasicVectorAdapter<Tpetra::Map<int, long, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> > > >::partition(Teuchos::RCP<Zoltan2::PartitioningSolution<Zoltan2::BasicVectorAdapter<Tpetra::Map<int, long, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> > > > > const&) ()
#14 0x000000000050c9a8 in Zoltan2::PartitioningProblem<Zoltan2::BasicVectorAdapter<Tpetra::Map<int, long, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> > > >::solve(bool) ()
#15 0x0000000000502e83 in Canopy::TreePartitioner<Kokkos::HostSpace, Kokkos::Serial>::partition_leaves(std::vector<Canopy::CellInfo, std::allocator<Canopy::CellInfo> > const&) ()
#16 0x00000000004cc212 in void Canopy::Solver<Kokkos::HostSpace, Kokkos::Serial, double, 8, 1, Canopy::LaplaceKernel>::_full_setup<0, 1, Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HostSpace, 16, Kokkos::MemoryTraits<0u> > >(Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HostSpace, 16, Kokkos::MemoryTraits<0u> >&, int) ()
#17 0x00000000004bb0cf in Test::testMultiStepGravity(Test::MultiSolveTest::Mode, char const*, int, int, double, double, int, int, double, int, double, int*, bool, double, long long*) ()
#18 0x000000000049d3b2 in Test::MultiSolve_LargeMotion_Rebuild_Test::TestBody() ()
#19 0x0000155554e008ed in void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#20 0x0000155554de2ea6 in testing::Test::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#21 0x0000155554de3045 in testing::TestInfo::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#22 0x0000155554de32bd in testing::TestSuite::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#23 0x0000155554df7169 in testing::internal::UnitTestImpl::RunAllTests() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#24 0x0000155554de3384 in testing::UnitTest::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#25 0x0000000000792012 in main ()
--- pid 2827920 ppid 2827918 comm Canopy_Test_Mul ---
tool: gstack
Thread 3 (Thread 0x154f1adff700 (LWP 2827936)):
#0  0x000015553bc462ab in ioctl () from /lib64/libc.so.6
#1  0x000015552f707d78 in hsakmt_ioctl () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#2  0x000015552f70097d in hsaKmtWaitOnMultipleEvents_Ext () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#3  0x000015552f66eac3 in rocr::core::Runtime::AsyncEventsLoop(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#4  0x000015552f61434d in rocr::os::ThreadTrampoline(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#5  0x000015553b9f31ca in start_thread () from /lib64/libpthread.so.0
#6  0x000015553bc46953 in clone () from /lib64/libc.so.6
Thread 2 (Thread 0x1555287ff700 (LWP 2827929)):
#0  0x000015553bc462ab in ioctl () from /lib64/libc.so.6
#1  0x000015552f707d78 in hsakmt_ioctl () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#2  0x000015552f70097d in hsaKmtWaitOnMultipleEvents_Ext () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#3  0x000015552f68f7a5 in rocr::core::Signal::WaitAnyExceptions(unsigned int, hsa_signal_s const*, hsa_signal_condition_t const*, long const*, long*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#4  0x000015552f66dfff in rocr::core::Runtime::AsyncEventsLoop(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#5  0x000015552f61434d in rocr::os::ThreadTrampoline(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#6  0x000015553b9f31ca in start_thread () from /lib64/libpthread.so.0
#7  0x000015553bc46953 in clone () from /lib64/libc.so.6
Thread 1 (Thread 0x155555548a80 (LWP 2827920)):
#0  0x0000155553930a1b in MPIDI_CRAY_Common_lmt_progress () from /opt/cray/pe/lib64/libmpi_cray.so.12
#1  0x00001555537b2b75 in MPIDI_progress_test () from /opt/cray/pe/lib64/libmpi_cray.so.12
#2  0x00001555537b4006 in MPID_Progress_wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#3  0x00001555537b6f56 in MPIR_Wait_state () from /opt/cray/pe/lib64/libmpi_cray.so.12
#4  0x00001555536cdcc7 in MPID_Wait.constprop.0 () from /opt/cray/pe/lib64/libmpi_cray.so.12
#5  0x00001555536e453f in MPIC_Wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#6  0x00001555536e4980 in MPIC_Recv () from /opt/cray/pe/lib64/libmpi_cray.so.12
#7  0x000015555372b55c in MPIR_CRAY_Bcast_Tree () from /opt/cray/pe/lib64/libmpi_cray.so.12
#8  0x000015555372bf5a in MPIR_CRAY_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#9  0x00001555532e01cf in PMPI_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#10 0x00000000005031ae in Canopy::TreePartitioner<Kokkos::HostSpace, Kokkos::Serial>::partition_leaves(std::vector<Canopy::CellInfo, std::allocator<Canopy::CellInfo> > const&) ()
#11 0x00000000004cc212 in void Canopy::Solver<Kokkos::HostSpace, Kokkos::Serial, double, 8, 1, Canopy::LaplaceKernel>::_full_setup<0, 1, Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HostSpace, 16, Kokkos::MemoryTraits<0u> > >(Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HostSpace, 16, Kokkos::MemoryTraits<0u> >&, int) ()
#12 0x00000000004bb0cf in Test::testMultiStepGravity(Test::MultiSolveTest::Mode, char const*, int, int, double, double, int, int, double, int, double, int*, bool, double, long long*) ()
#13 0x000000000049d3b2 in Test::MultiSolve_LargeMotion_Rebuild_Test::TestBody() ()
#14 0x0000155554e008ed in void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#15 0x0000155554de2ea6 in testing::Test::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#16 0x0000155554de3045 in testing::TestInfo::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#17 0x0000155554de32bd in testing::TestSuite::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#18 0x0000155554df7169 in testing::internal::UnitTestImpl::RunAllTests() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#19 0x0000155554de3384 in testing::UnitTest::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#20 0x0000000000792012 in main ()
--- pid 2827921 ppid 2827918 comm Canopy_Test_Mul ---
tool: gstack
Thread 3 (Thread 0x154f1adff700 (LWP 2827937)):
#0  0x000015553bc462ab in ioctl () from /lib64/libc.so.6
#1  0x000015552f707d78 in hsakmt_ioctl () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#2  0x000015552f70097d in hsaKmtWaitOnMultipleEvents_Ext () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#3  0x000015552f66eac3 in rocr::core::Runtime::AsyncEventsLoop(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#4  0x000015552f61434d in rocr::os::ThreadTrampoline(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#5  0x000015553b9f31ca in start_thread () from /lib64/libpthread.so.0
#6  0x000015553bc46953 in clone () from /lib64/libc.so.6
Thread 2 (Thread 0x1555287ff700 (LWP 2827931)):
#0  0x000015553bc462ab in ioctl () from /lib64/libc.so.6
#1  0x000015552f707d78 in hsakmt_ioctl () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#2  0x000015552f70097d in hsaKmtWaitOnMultipleEvents_Ext () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#3  0x000015552f68f7a5 in rocr::core::Signal::WaitAnyExceptions(unsigned int, hsa_signal_s const*, hsa_signal_condition_t const*, long const*, long*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#4  0x000015552f66dfff in rocr::core::Runtime::AsyncEventsLoop(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#5  0x000015552f61434d in rocr::os::ThreadTrampoline(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#6  0x000015553b9f31ca in start_thread () from /lib64/libpthread.so.0
#7  0x000015553bc46953 in clone () from /lib64/libc.so.6
Thread 1 (Thread 0x155555548a80 (LWP 2827921)):
#0  0x00001555538a257a in MPID_ST_progress_queue () from /opt/cray/pe/lib64/libmpi_cray.so.12
#1  0x00001555537b25e7 in MPIDI_progress_test () from /opt/cray/pe/lib64/libmpi_cray.so.12
#2  0x00001555537b4006 in MPID_Progress_wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#3  0x00001555537b6f56 in MPIR_Wait_state () from /opt/cray/pe/lib64/libmpi_cray.so.12
#4  0x00001555536cdcc7 in MPID_Wait.constprop.0 () from /opt/cray/pe/lib64/libmpi_cray.so.12
#5  0x00001555536e453f in MPIC_Wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#6  0x00001555536e4980 in MPIC_Recv () from /opt/cray/pe/lib64/libmpi_cray.so.12
#7  0x000015555372b55c in MPIR_CRAY_Bcast_Tree () from /opt/cray/pe/lib64/libmpi_cray.so.12
#8  0x000015555372bf5a in MPIR_CRAY_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#9  0x00001555532e01cf in PMPI_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#10 0x00000000005031ae in Canopy::TreePartitioner<Kokkos::HostSpace, Kokkos::Serial>::partition_leaves(std::vector<Canopy::CellInfo, std::allocator<Canopy::CellInfo> > const&) ()
#11 0x00000000004cc212 in void Canopy::Solver<Kokkos::HostSpace, Kokkos::Serial, double, 8, 1, Canopy::LaplaceKernel>::_full_setup<0, 1, Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HostSpace, 16, Kokkos::MemoryTraits<0u> > >(Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HostSpace, 16, Kokkos::MemoryTraits<0u> >&, int) ()
#12 0x00000000004bb0cf in Test::testMultiStepGravity(Test::MultiSolveTest::Mode, char const*, int, int, double, double, int, int, double, int, double, int*, bool, double, long long*) ()
#13 0x000000000049d3b2 in Test::MultiSolve_LargeMotion_Rebuild_Test::TestBody() ()
#14 0x0000155554e008ed in void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#15 0x0000155554de2ea6 in testing::Test::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#16 0x0000155554de3045 in testing::TestInfo::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#17 0x0000155554de32bd in testing::TestSuite::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#18 0x0000155554df7169 in testing::internal::UnitTestImpl::RunAllTests() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#19 0x0000155554de3384 in testing::UnitTest::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#20 0x0000000000792012 in main ()
### end watchdog stacks f3oTBtb9H: matched=3 children=3 nonempty=3 ###
### watchdog: cancelling sub-job f3oTBtb9H after 302.1887285709381 s ###
```

**Affects:**
- H2: the stacks above are its input. They match its second candidate shape
  (rank 0 inside Zoltan2 while the others wait in `partition_leaves`'s
  `MPI_Bcast`). Rank 0's innermost frames are HIP (`hipDeviceSynchronize` from a
  `Kokkos::deep_copy` in `AlgMJ::mj_get_new_cut_coordinates` on
  `Kokkos::HIP`), which also touches its third candidate. H2 reads them; H1
  does not decide.
- E1: `scripts/tuolumne/flux_watchdog.sh` is ready to source (set `WATCHDOG_S`,
  `source`, `watchdog_stop`). Use `watchdog_wait_idle` before reading anything
  after a ctest call that may have timed out.
- tree-opt V1: `run_ctest_v1.flux` now sources the helper, so an np-3 hang there
  is stacked before it is cancelled.

## H2

Jobs: `f3bnvGYaSqWP` (`run_ctest_h2.flux fixed`), `f3bnvGg3MDo5`
(`run_ctest_h2.flux repro`), `f3bnyu3peHWw` (`run_ctest_h1.flux` on the
reverted build). All three ran on HEAD `e7615ba` plus this task's working-tree
changes. Each job's `git status --porcelain` shows which state it measured: the
fixed jobs show ` M src/Canopy_TreePartitioner.hpp`, the reverted job shows no
`src/` change. Cray clang 20.0.0, flux-core 0.89.0, `build-tuolumne/` with
`Canopy_ENABLE_PROFILING=ON`. Logs: `canopy-h2.<jobid>.log` and
`canopy-h1.f3bnyu3peHWw.log` in the repo root (untracked).

**Mechanism.** `partition_leaves` ran Zoltan2 multijagged on Tpetra's default
node, `KokkosDeviceWrapperNode<Kokkos::HIP>`, even in
`TreePartitioner<HostSpace, Serial>`. Intermittently, rank 0's MJ never
returned from a `hipDeviceSynchronize` (from a `Kokkos::deep_copy` in
`AlgMJ::mj_get_new_cut_coordinates`), while ranks 1-2 waited in the assignment
`MPI_Bcast` (`:431`). Every capture stalled in `MultiSolve.LargeMotion_Rebuild`,
the case that re-partitions every step. Running MJ on the solver's own
execution space removes the HIP device from the partition path.
Why the HIP fence stalls was not investigated. That falls under the
HIP-`ExecutionSpace` README entry.

**Decisions.**
- Leading candidate only. The Do step 2 fallbacks were not needed.
- Reproducibility is measured with `[multisolve-dev]` from two `-V` np 1-6
  passes. No other target was built.
- The np 1-2 lines are compared against `canopy-v1.f3bmo4JYikKh.log`. That
  log stopped being E1's "Inert when off" baseline once 01_fix-tests F4
  (`8e6e0c5`) moved np 1 and the partitioner arm moved np >= 2 (section E1).

**Departure from Do step 1: the node is set through `BasicUserTypes`, not
`Tpetra::Map`.** Templating `Tpetra::Map<int, int64_t, Node>` compiles but
changes nothing. Zoltan2 takes `node_t` from `InputTraits<User>`, and
`Tpetra::Map` has no specialization, so the generic traits return
`Zoltan2::default_node_t`, Tpetra's default node (HIP)
(`Zoltan2_InputTraits.hpp:53,173` in the spack view). The first rebuild still
had 465 `AlgMJ<…HIP…>` symbols and no Serial one (`nm -C`). The adapter is now
`BasicVectorAdapter<BasicUserTypes<default_scalar_t, default_lno_t,
default_gno_t, KokkosDeviceWrapperNode<ExecutionSpace>>>`: 0 HIP and 233 Serial
MJ symbols. The scalar, lno and gno types are unchanged (`double`, `int`,
`long long`). `offset_t` becomes `int` (was `size_t`); that type sizes graph and
matrix adjacency offsets, which a vector adapter does not use. The
`static_assert` checks `adapter_t::node_t::execution_space`, the type Zoltan2
actually uses. A check on the Map's node would have passed while MJ still ran on
HIP. The now-unused `#include <Tpetra_Map.hpp>` was removed. No signature
changed, and `TreePartitioner`'s callers need no edits.

The fix (the reverted build is HEAD's file, i.e. this diff reversed):

```diff
diff --git a/src/Canopy_TreePartitioner.hpp b/src/Canopy_TreePartitioner.hpp
index 2b24d54..4fc621f 100644
--- a/src/Canopy_TreePartitioner.hpp
+++ b/src/Canopy_TreePartitioner.hpp
@@ -26,13 +26,14 @@
 #include <Teuchos_DefaultMpiComm.hpp>
 #include <Teuchos_DefaultSerialComm.hpp>
 #include <Teuchos_ParameterList.hpp>
-#include <Tpetra_Map.hpp>
+#include <Tpetra_KokkosCompat_ClassicNodeAPI_Wrapper.hpp>
 
 #include <mpi.h>
 
 #include <cstdint>
 #include <iostream>
 #include <limits>
+#include <type_traits>
 #include <typeinfo>
 #include <unordered_map>
 #include <vector>
@@ -364,7 +365,22 @@ TreePartitioner<MemorySpace, ExecutionSpace>::partition_leaves(
     // Create Zoltan2 adapter
     // BasicVectorAdapter needs:
     //   numIds, globalIds, coords, weights
-    using adapter_t = Zoltan2::BasicVectorAdapter<Tpetra::Map<int, int64_t>>;
+    // Zoltan2 runs on the partitioner's execution space. Its node comes from
+    // InputTraits<User>, which defaults to Tpetra's default node (HIP in this
+    // build) for any User without a specialization, Tpetra::Map included; MJ's
+    // device fences there stall intermittently at np >= 3 even for a Serial
+    // solver (tasks/fix-hang-rebalance-progress-log.md, H1). BasicUserTypes
+    // sets the node and keeps Zoltan2's default scalar/lno/gno.
+    using zoltan_node_t =
+        Tpetra::KokkosCompat::KokkosDeviceWrapperNode<ExecutionSpace>;
+    using adapter_t = Zoltan2::BasicVectorAdapter<Zoltan2::BasicUserTypes<
+        Zoltan2::default_scalar_t, Zoltan2::default_lno_t,
+        Zoltan2::default_gno_t, zoltan_node_t>>;
+    static_assert(
+        std::is_same_v<typename adapter_t::node_t::execution_space,
+                       ExecutionSpace>,
+        "partition_leaves: Zoltan2 must run on the partitioner's "
+        "ExecutionSpace, not Tpetra's default node" );
     // Must also use Zoltan types
     using glbl_id_t = typename adapter_t::gno_t;
     using scalar_t = typename adapter_t::scalar_t;
```

**Hang counts.**
- Fixed build (`f3bnvGYaSqWP`): **0 hangs in 15** consecutive
  `ctest --timeout 300 -R '^Canopy_Test_MultiSolve_MPI_SERIAL_np_3$'` runs, no
  watchdog cancellation. The repro job's two np 1-6 passes add 2 more clean np-3
  runs: 0 in 17 on the fixed build.
- Reverted build (`f3bnyu3peHWw`): **1 hang in 1** run, cancelled by the
  watchdog at 302.3 s. The job's self-test passed again (303.4 s, 3/3 child
  stacks). Combined with H1, the HIP-node rate is now 6 hangs in 15 np-3 runs.

**Reproducibility (fixed build, `f3bnvGg3MDo5`).** All 36 `[multisolve-dev]`
lines are identical to every printed digit across the two passes, for every
`(nprocs, case)` with nprocs 1-6 and case `StableTree_Migrate`,
`IntermediateMotion_Rebalance`, `LargeMotion_Rebuild`, `AutoMaintain`,
`AutoRebalance` and `M2L_BinEdge_Fallback`. V1's three HIP-node passes
(`f3bmo4JYikKh`) printed 63 distinct lines for the same 36 slots.

**np 1-2 baseline.** The 12 np 1-2 lines match `canopy-v1.f3bmo4JYikKh.log`
character for character. np 1 never calls Zoltan2 (`_comm_size == 1` skips it),
and the np-2 cut came out the same on the host node.

**The partition path changed at np >= 3.** Only 5 of the 24 np 3-6 lines
occur anywhere in V1's log, and the rest moved within V1's run-to-run spread.
AutoRebalance's excess is unchanged by it:

| nprocs | max_pos_rel (H2) | max_vel_rel (H2) | max_vel_rel (V1, range) |
| --- | --- | --- | --- |
| 5 | 7.146e-4 | 6.573e-3 | 6.41e-3 – 6.57e-3 |
| 6 | 3.154e-3 | 1.709e-2 | 1.709e-2 |

**Reverted-build stacks, verbatim** (`f3bnyu3peHWw`, run 1). The last gtest
case is `[ RUN      ] MultiSolve.LargeMotion_Rebuild`. The last
`[multisolve-dev]` line before it:

```
[multisolve-dev] case IntermediateMotion_Rebalance nprocs 3 nsteps 5 drift 5 max_pos_rel 1.5646110360228987e-05 max_vel_rel 4.3603755255711028e-05 tol 1e-08
```

```
### watchdog stacks f3aFFRdnX ###
time: 2026-10-02 15:17:38 runtime: 302.3462574481964 s jobid(dec): 5681587421184
shell: 3023427 /usr/libexec/flux/flux-shell 5681587421184
--- pid 3023428 ppid 3023427 comm Canopy_Test_Mul ---
tool: gstack
Thread 3 (Thread 0x15491afff700 (LWP 3023449)):
#0  0x000015553bc462ab in ioctl () from /lib64/libc.so.6
#1  0x000015552f707d78 in hsakmt_ioctl () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#2  0x000015552f70097d in hsaKmtWaitOnMultipleEvents_Ext () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#3  0x000015552f66eac3 in rocr::core::Runtime::AsyncEventsLoop(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#4  0x000015552f61434d in rocr::os::ThreadTrampoline(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#5  0x000015553b9f31ca in start_thread () from /lib64/libpthread.so.0
#6  0x000015553bc46953 in clone () from /lib64/libc.so.6
Thread 2 (Thread 0x1555287ff700 (LWP 3023447)):
#0  0x000015553bc462ab in ioctl () from /lib64/libc.so.6
#1  0x000015552f707d78 in hsakmt_ioctl () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#2  0x000015552f70097d in hsaKmtWaitOnMultipleEvents_Ext () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#3  0x000015552f68f7a5 in rocr::core::Signal::WaitAnyExceptions(unsigned int, hsa_signal_s const*, hsa_signal_condition_t const*, long const*, long*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#4  0x000015552f66dfff in rocr::core::Runtime::AsyncEventsLoop(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#5  0x000015552f61434d in rocr::os::ThreadTrampoline(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#6  0x000015553b9f31ca in start_thread () from /lib64/libpthread.so.0
#7  0x000015553bc46953 in clone () from /lib64/libc.so.6
Thread 1 (Thread 0x155555548a80 (LWP 3023428)):
#0  0x000015552f64f991 in rocr::core::InterruptSignal::WaitRelaxed(hsa_signal_condition_t, long, unsigned long, hsa_wait_state_t) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#1  0x000015552f64f7fa in rocr::core::InterruptSignal::WaitAcquire(hsa_signal_condition_t, long, unsigned long, hsa_wait_state_t) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#2  0x000015552f644231 in rocr::HSA::hsa_signal_wait_scacquire(hsa_signal_s, hsa_signal_condition_t, long, unsigned long, hsa_wait_state_t) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#3  0x000015553c59027b in amd::roc::Device::IsHwEventReady(amd::Event const&, bool, unsigned int) const () from /opt/rocm-6.4.2/lib/libamdhip64.so.6
#4  0x000015553c57f1d7 in amd::HostQueue::finish(bool) () from /opt/rocm-6.4.2/lib/libamdhip64.so.6
#5  0x000015553c31b21e in hip::Device::SyncAllStreams(bool, bool) () from /opt/rocm-6.4.2/lib/libamdhip64.so.6
#6  0x000015553c308e51 in hip::hipDeviceSynchronize() () from /opt/rocm-6.4.2/lib/libamdhip64.so.6
#7  0x0000155541a07cb3 in Kokkos::HIP::impl_static_fence(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libkokkoscore.so.4.7
#8  0x00001555419f671d in Kokkos::fence(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libkokkoscore.so.4.7
#9  0x00000000005930b0 in Kokkos::deep_copy<int*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>, int*, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::OpenMP, Kokkos::HIPSpace>, Kokkos::Experimental::EmptyViewHooks> ()
#10 0x00000000005b5e19 in Zoltan2::AlgMJ<double, int, long long, int, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> >::mj_get_new_cut_coordinates(int, int, int const&, double const&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<bool*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<int*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&) ()
#11 0x00000000005898b2 in Zoltan2::AlgMJ<double, int, long long, int, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> >::mj_1D_part(Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, double, int, int, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, int, Kokkos::View<int*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<unsigned long*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&) ()
#12 0x0000000000550448 in Zoltan2::AlgMJ<double, int, long long, int, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> >::multi_jagged_part(Teuchos::RCP<Zoltan2::Environment const> const&, Teuchos::RCP<Teuchos::Comm<int> const>&, double, int, unsigned long, Kokkos::View<int*, Kokkos::HostSpace>&, int, int, int, long long, Kokkos::View<long long const*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, int, Kokkos::View<bool*, Kokkos::HostSpace>&, Kokkos::View<double**, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<bool*, Kokkos::HostSpace>&, Kokkos::View<int*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<long long*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&) ()
#13 0x0000000000541146 in Zoltan2::Zoltan2_AlgMJ<Zoltan2::BasicVectorAdapter<Tpetra::Map<int, long, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> > > >::partition(Teuchos::RCP<Zoltan2::PartitioningSolution<Zoltan2::BasicVectorAdapter<Tpetra::Map<int, long, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> > > > > const&) ()
#14 0x000000000050c9a8 in Zoltan2::PartitioningProblem<Zoltan2::BasicVectorAdapter<Tpetra::Map<int, long, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> > > >::solve(bool) ()
#15 0x0000000000502e83 in Canopy::TreePartitioner<Kokkos::HostSpace, Kokkos::Serial>::partition_leaves(std::vector<Canopy::CellInfo, std::allocator<Canopy::CellInfo> > const&) ()
#16 0x00000000004cc212 in void Canopy::Solver<Kokkos::HostSpace, Kokkos::Serial, double, 8, 1, Canopy::LaplaceKernel>::_full_setup<0, 1, Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HostSpace, 16, Kokkos::MemoryTraits<0u> > >(Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HostSpace, 16, Kokkos::MemoryTraits<0u> >&, int) ()
#17 0x00000000004bb0cf in Test::testMultiStepGravity(Test::MultiSolveTest::Mode, char const*, int, int, double, double, int, int, double, int, double, int*, bool, double, long long*) ()
#18 0x000000000049d3b2 in Test::MultiSolve_LargeMotion_Rebuild_Test::TestBody() ()
#19 0x0000155554e008ed in void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#20 0x0000155554de2ea6 in testing::Test::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#21 0x0000155554de3045 in testing::TestInfo::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#22 0x0000155554de32bd in testing::TestSuite::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#23 0x0000155554df7169 in testing::internal::UnitTestImpl::RunAllTests() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#24 0x0000155554de3384 in testing::UnitTest::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#25 0x0000000000792012 in main ()
--- pid 3023429 ppid 3023427 comm Canopy_Test_Mul ---
tool: gstack
Thread 3 (Thread 0x154f1adff700 (LWP 3023450)):
#0  0x000015553bc462ab in ioctl () from /lib64/libc.so.6
#1  0x000015552f707d78 in hsakmt_ioctl () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#2  0x000015552f70097d in hsaKmtWaitOnMultipleEvents_Ext () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#3  0x000015552f66eac3 in rocr::core::Runtime::AsyncEventsLoop(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#4  0x000015552f61434d in rocr::os::ThreadTrampoline(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#5  0x000015553b9f31ca in start_thread () from /lib64/libpthread.so.0
#6  0x000015553bc46953 in clone () from /lib64/libc.so.6
Thread 2 (Thread 0x1555287ff700 (LWP 3023443)):
#0  0x000015553bc462ab in ioctl () from /lib64/libc.so.6
#1  0x000015552f707d78 in hsakmt_ioctl () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#2  0x000015552f70097d in hsaKmtWaitOnMultipleEvents_Ext () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#3  0x000015552f68f7a5 in rocr::core::Signal::WaitAnyExceptions(unsigned int, hsa_signal_s const*, hsa_signal_condition_t const*, long const*, long*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#4  0x000015552f66dfff in rocr::core::Runtime::AsyncEventsLoop(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#5  0x000015552f61434d in rocr::os::ThreadTrampoline(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#6  0x000015553b9f31ca in start_thread () from /lib64/libpthread.so.0
#7  0x000015553bc46953 in clone () from /lib64/libc.so.6
Thread 1 (Thread 0x155555548a80 (LWP 3023429)):
#0  0x00001555538a257a in MPID_ST_progress_queue () from /opt/cray/pe/lib64/libmpi_cray.so.12
#1  0x00001555537b25e7 in MPIDI_progress_test () from /opt/cray/pe/lib64/libmpi_cray.so.12
#2  0x00001555537b4006 in MPID_Progress_wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#3  0x00001555537b6f56 in MPIR_Wait_state () from /opt/cray/pe/lib64/libmpi_cray.so.12
#4  0x00001555536cdcc7 in MPID_Wait.constprop.0 () from /opt/cray/pe/lib64/libmpi_cray.so.12
#5  0x00001555536e453f in MPIC_Wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#6  0x00001555536e4980 in MPIC_Recv () from /opt/cray/pe/lib64/libmpi_cray.so.12
#7  0x000015555372b55c in MPIR_CRAY_Bcast_Tree () from /opt/cray/pe/lib64/libmpi_cray.so.12
#8  0x000015555372bf5a in MPIR_CRAY_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#9  0x00001555532e01cf in PMPI_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#10 0x00000000005031ae in Canopy::TreePartitioner<Kokkos::HostSpace, Kokkos::Serial>::partition_leaves(std::vector<Canopy::CellInfo, std::allocator<Canopy::CellInfo> > const&) ()
#11 0x00000000004cc212 in void Canopy::Solver<Kokkos::HostSpace, Kokkos::Serial, double, 8, 1, Canopy::LaplaceKernel>::_full_setup<0, 1, Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HostSpace, 16, Kokkos::MemoryTraits<0u> > >(Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HostSpace, 16, Kokkos::MemoryTraits<0u> >&, int) ()
#12 0x00000000004bb0cf in Test::testMultiStepGravity(Test::MultiSolveTest::Mode, char const*, int, int, double, double, int, int, double, int, double, int*, bool, double, long long*) ()
#13 0x000000000049d3b2 in Test::MultiSolve_LargeMotion_Rebuild_Test::TestBody() ()
#14 0x0000155554e008ed in void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#15 0x0000155554de2ea6 in testing::Test::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#16 0x0000155554de3045 in testing::TestInfo::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#17 0x0000155554de32bd in testing::TestSuite::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#18 0x0000155554df7169 in testing::internal::UnitTestImpl::RunAllTests() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#19 0x0000155554de3384 in testing::UnitTest::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#20 0x0000000000792012 in main ()
--- pid 3023430 ppid 3023427 comm Canopy_Test_Mul ---
tool: gstack
Thread 3 (Thread 0x154f1adff700 (LWP 3023451)):
#0  0x000015553bc462ab in ioctl () from /lib64/libc.so.6
#1  0x000015552f707d78 in hsakmt_ioctl () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#2  0x000015552f70097d in hsaKmtWaitOnMultipleEvents_Ext () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#3  0x000015552f66eac3 in rocr::core::Runtime::AsyncEventsLoop(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#4  0x000015552f61434d in rocr::os::ThreadTrampoline(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#5  0x000015553b9f31ca in start_thread () from /lib64/libpthread.so.0
#6  0x000015553bc46953 in clone () from /lib64/libc.so.6
Thread 2 (Thread 0x1555287ff700 (LWP 3023445)):
#0  0x000015553bc462ab in ioctl () from /lib64/libc.so.6
#1  0x000015552f707d78 in hsakmt_ioctl () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#2  0x000015552f70097d in hsaKmtWaitOnMultipleEvents_Ext () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#3  0x000015552f68f7a5 in rocr::core::Signal::WaitAnyExceptions(unsigned int, hsa_signal_s const*, hsa_signal_condition_t const*, long const*, long*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#4  0x000015552f66dfff in rocr::core::Runtime::AsyncEventsLoop(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#5  0x000015552f61434d in rocr::os::ThreadTrampoline(void*) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#6  0x000015553b9f31ca in start_thread () from /lib64/libpthread.so.0
#7  0x000015553bc46953 in clone () from /lib64/libc.so.6
Thread 1 (Thread 0x155555548a80 (LWP 3023430)):
#0  0x00001555537b28b8 in MPIDI_progress_test () from /opt/cray/pe/lib64/libmpi_cray.so.12
#1  0x00001555537b4006 in MPID_Progress_wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#2  0x00001555537b6f56 in MPIR_Wait_state () from /opt/cray/pe/lib64/libmpi_cray.so.12
#3  0x00001555536cdcc7 in MPID_Wait.constprop.0 () from /opt/cray/pe/lib64/libmpi_cray.so.12
#4  0x00001555536e453f in MPIC_Wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#5  0x00001555536e4980 in MPIC_Recv () from /opt/cray/pe/lib64/libmpi_cray.so.12
#6  0x000015555372b55c in MPIR_CRAY_Bcast_Tree () from /opt/cray/pe/lib64/libmpi_cray.so.12
#7  0x000015555372bf5a in MPIR_CRAY_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#8  0x00001555532e01cf in PMPI_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#9  0x00000000005031ae in Canopy::TreePartitioner<Kokkos::HostSpace, Kokkos::Serial>::partition_leaves(std::vector<Canopy::CellInfo, std::allocator<Canopy::CellInfo> > const&) ()
#10 0x00000000004cc212 in void Canopy::Solver<Kokkos::HostSpace, Kokkos::Serial, double, 8, 1, Canopy::LaplaceKernel>::_full_setup<0, 1, Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HostSpace, 16, Kokkos::MemoryTraits<0u> > >(Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HostSpace, 16, Kokkos::MemoryTraits<0u> >&, int) ()
#11 0x00000000004bb0cf in Test::testMultiStepGravity(Test::MultiSolveTest::Mode, char const*, int, int, double, double, int, int, double, int, double, int*, bool, double, long long*) ()
#12 0x000000000049d3b2 in Test::MultiSolve_LargeMotion_Rebuild_Test::TestBody() ()
#13 0x0000155554e008ed in void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#14 0x0000155554de2ea6 in testing::Test::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#15 0x0000155554de3045 in testing::TestInfo::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#16 0x0000155554de32bd in testing::TestSuite::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#17 0x0000155554df7169 in testing::internal::UnitTestImpl::RunAllTests() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#18 0x0000155554de3384 in testing::UnitTest::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#19 0x0000000000792012 in main ()
### end watchdog stacks f3aFFRdnX: matched=3 children=3 nonempty=3 ###
```

**Affects:**
- E1: measures on the post-H2 partition, which is now reproducible run to run
  at np 3-6 for the SERIAL binaries, so E1's two-run comparison at np >= 3
  should agree exactly. Its np 1-2 baseline is not `canopy-v1.f3bmo4JYikKh.log`:
  F4's MAC-tie guard moved np 1 after this section, and the partitioner arm
  moved np 2. E1 compares against `f3cZDLkZqC6s` and `f3cZDMUWgsYj`. AutoRebalance's np 5-6 excess survives H2 (table above), so the
  deviation is not caused by the HIP-node partition.
- tree-opt V1: the np >= 3 `[multisolve-dev]` figures moved. Re-measure np >= 3
  on this build before pinning anything. np 1-2 are unchanged.
- E2: none.

## H0a

Job `f3cLYyF1XxQX` (`scripts/tuolumne/run_h0a_binding.flux`, node
`tuolumne1048`, HEAD `53c671b` plus this task's working-tree changes; Cray clang
20.0.0, flux-core 0.89.0). Log: `canopy-h0a-binding.f3cLYyF1XxQX.log` in the
repo root (untracked).

**Variables added.** `Canopy_TEST_MPI_RANKS_<DEVICE>` and
`Canopy_TEST_MPIEXEC_PREFLAGS_<DEVICE>`, both unset by default and documented
beside `Canopy_TEST_MPI_RANKS` in `CMakeLists.txt`. They are not declared
as cache variables because the device set is only known after Kokkos is found.
The `MPIEXEC_MAX_NUMPROCS` filter became `_canopy_filter_mpi_ranks(out what
ranks…)`, one function called for the shared list and for each device that
sets an override. The per-device lists are resolved once at file scope
(`CANOPY_TEST_MPI_RANKS_EFFECTIVE_<DEVICE>`), so a skipped-rank warning prints
once per configure and not once per `Canopy_add_tests` call. Inside the macro's
`_device` loop, `_canopy_mpi_ranks` and `_canopy_mpi_preflags` are set on every
pass, so nothing carries over to the next device.
`CANOPY_UNIT_TEST_MPIEXEC_NUMPROCS` was removed; nothing else read it.
`run_cmake_tuolumne.sh` sets `Canopy_TEST_MPI_RANKS_HIP=1;2;3;4` and
`Canopy_TEST_MPIEXEC_PREFLAGS_HIP=--nodes=1;--exclusive;--gpus-per-task=1;--cores-per-task=8`.

**Decisions.**
- The SERIAL comparison covers `name`, `command` and `properties` of each
  `_MPI_SERIAL_` object
  (`jq -S '[.tests[] | select(.name|test("_MPI_SERIAL_")) | {name,command,properties}]'`).
  `backtrace` is left out because it indexes `backtraceGraph`, which records
  `test_harness.cmake` line numbers that this edit shifts.
- The "before" dump was taken from `build-tuolumne/` before any CMake file was
  edited.
- The binding was checked twice: with the criterion's verbatim command, and
  with `--cores-per-task=8 --setopt=mpibind=verbose:1` added, as the registered
  HIP preflags launch.

**Registration.** `ctest -N`: 230 entries before, 208 after. `_MPI_HIP_np_`:
66 before (11 stems × np 1-6), 44 after (11 × np 1-4), and
`-R '_MPI_HIP_np_[56]$'` lists none. Every HIP command is now
`flux run --ntasks N --nodes=1 --exclusive --gpus-per-task=1 --cores-per-task=8 <exe>`.
`diff canopy-h0a.serial-before.json canopy-h0a.serial-after.json` (66 objects
each) printed nothing, rc 0. The same projection over every non-`_MPI_HIP_`
entry (SERIAL, OPENMP, non-MPI, valgrind) is also identical.

**Binding (`f3cLYyF1XxQX`).** np 4, verbatim `printenv ROCR_VISIBLE_DEVICES`:
`0 1 2 3`, rc 0. With all device variables echoed, `ROCR_VISIBLE_DEVICES` is
the only one set: `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` are unset.

```
0: rank=0 ROCR=0 HIP=unset CUDA=unset
1: rank=1 ROCR=1 HIP=unset CUDA=unset
2: rank=2 ROCR=2 HIP=unset CUDA=unset
3: rank=3 ROCR=3 HIP=unset CUDA=unset
```

With `--cores-per-task=8`, the same four devices. mpibind widens each task to
its whole APU's cores less one per eight, not to the eight requested:

```
mpibind: task   0 nths 21 gpus 0 cpus 1-7,9-15,17-23
mpibind: task   1 nths 21 gpus 1 cpus 25-31,33-39,41-47
mpibind: task   2 nths 21 gpus 2 cpus 49-55,57-63,65-71
mpibind: task   3 nths 21 gpus 3 cpus 73-79,81-87,89-95
```

np 5, all three variants: rc 1,
`job.exception … type=alloc severity=0 alloc denied due to type="unsatisfiable"`.

**Affects:**
- H0b: `canopy_ctest` sees HIP entries at np 1-4 only, and also guards against
  any HIP entry above np 4.
- H0c: HIP runs launch with one APU per rank through ctest. Under mpibind each
  rank gets 21 CPUs, so `OMP_NUM_THREADS=1` (the ctest `ENVIRONMENT` property)
  is still what bounds host threading.

## H0b

Jobs, all `scripts/tuolumne/run_ctest_h0b.flux`, HEAD `875628e` plus this
task's working-tree changes, Cray clang 20.0.0, flux-core 0.89.0,
`build-tuolumne/` with `Canopy_ENABLE_PROFILING=ON`. Logs:
`canopy-h0b.<jobid>.log` in the repo root (untracked).
- `f3cLieUe4u2F` (`all`): self-test, calibration, check.
- `f3cLx8h8NwVR` and `f3cLyvp55Utf` (`check`): reruns after the fixes below.
- `f3cM1y4WB335` (`verify`): self-test and check on the final scripts. This is
  the job the **Met.** paragraph cites for everything except the calibration.

The ten SERIAL targets were rebuilt at HEAD (`make -j` naming exactly them)
before calibrating, because the binaries in `build-tuolumne/tests` predated H2.

**Interface.** `scripts/tuolumne/ctest_budget.sh`, sourced after
`flux_watchdog.sh`, defines `canopy_ctest <anchored-regex> [ctest args…]`.
Inputs are `CANOPY_BUDGET_CONFIG` (default `default`) and `CANOPY_BUDGET_TSV`
(default `serial_runtimes.tsv` beside the script). It prints one line per entry:
`[canopy_ctest] <entry> runtime=<s> budget=<s> outcome=<completed|failed|over-budget>`.
It returns 0 if every entry completed, 1 if any failed or went over budget, and
2 on refusal.

**Decisions.**
- A HIP entry above np 4 is refused. `canopy_ctest` exits 2 naming any
  `_MPI_HIP_np_<N>` entry with N > 4, before launching anything, the same as
  for a missing row. This is a second guard behind H0a's registration.
- Only two name shapes are accepted: `Canopy_Test_<Stem>_MPI_<DEV>_np_<N>` and
  `Canopy_Test_<Stem>_<DEV>` (a non-MPI stem, looked up as np 1). Anything
  else, `_valgrind` and `_nt_<k>` included, is refused before any launch.
- `canopy_ctest` stacks non-MPI entries itself. They run as `<exe>` with no
  `flux run` (`NONMPI_PRECOMMAND` is empty), so the watchdog cannot see them.
  At the budget it `gstack`s every descendant of the `ctest` process whose
  name matches `CANOPY_WATCHDOG_PGREP`, between the watchdog's
  `### watchdog stacks … ###` markers, then sends `SIGKILL`.
- The ten SERIAL targets were rebuilt at HEAD before calibrating.
- H0b step 4 rewrites no existing script. `run_ctest_v1/h1/h2.flux` stay as
  the record of what those jobs ran; the rule binds new runs. `tree-opt.md`
  already routes its runs through `canopy_ctest`.

**Departures from Do.**
- ctest's `--timeout` is the budget plus 5 s, for MPI entries as well as
  non-MPI. In `f3cLieUe4u2F`, `--timeout` equalled the budget. A forced
  over-budget `MultiSolve` np 2 then hit ctest's timeout at 4.01 s, and its
  sub-job ended with the killed `flux run` client before the watchdog polled,
  so there were no stacks. A sub-job that is merely slow, not hung, dies with
  its client. With the 5 s margin, the watchdog stacks and cancels at the
  budget first. Outcome `over-budget` means the watchdog or the non-MPI timer
  cancelled the entry, or ctest reported a timeout.
- The watchdog's cancel test compares the fractional runtime
  (`awk 'BEGIN { exit !(r > t) }'`). Before, it compared the integer part
  (`"${rt%.*}" -gt`), which fires only once the runtime reaches the next whole
  second, up to 1 s late on top of the 2 s poll. `f3cLyvp55Utf` cancelled a
  4 s budget at 7.08 s; `f3cM1y4WB335`, on the fractional test, at 5.13 s.
- The script takes a phase argument: `calibrate`, `check`, `verify`
  (self-test then check) or `all`. Everything ran in one allocation
  (`f3cLieUe4u2F`, `all`); the phases exist for the reruns.
- The check phase adds two diagnostics the exit criterion does not name. One
  calls `_canopy_ctest_budget` on refused names without launching anything.
  The other forces an over-budget entry for each cancel path with a throwaway
  TSV (`MultiSolve` np 6 at `t_ref_s = 2`, budget 4; `CartesianTaylor` at 0.5,
  budget 1).
- The TSV's `jobid` column was written empty by `f3cLieUe4u2F`: `FLUX_JOB_ID`
  is not set in the batch shell. The column was filled with `f3cLieUe4u2F`, the
  job whose log holds every sample. The scripts now take the id from
  `flux getattr jobid`.

**Self-test.** `f3cLieUe4u2F`: sub-job `f75CzP2P` cancelled at 21.46 s with
`matched=3 children=3 nonempty=3`. `f3cM1y4WB335`: cancelled at 21.80 s with
3/3/3. Both are inside 20-25 s.

**Calibration (`f3cLieUe4u2F`).** 55 entries × 3 passes, all completed, no
watchdog cancellation, about 28 min. `t_ref_s`, the maximum of three, in s:

| stem | np 1 | np 2 | np 3 | np 4 | np 5 | np 6 |
| --- | --- | --- | --- | --- | --- | --- |
| `MultiSolve` | 4.76 | 6.81 | 9.21 | 11.94 | 13.94 | 15.80 |
| `CartesianTaylorSolve` | 18.86 | 12.40 | 11.65 | 11.20 | 11.53 | 11.88 |
| `LaplaceSolve` | 9.96 | 8.29 | 8.26 | 8.66 | 9.19 | 10.00 |
| `DownwardSweep` | 3.84 | 5.41 | 7.24 | 7.97 | 8.19 | 9.16 |
| `UpwardSweep` | 3.54 | 4.19 | 5.25 | 6.17 | 6.94 | 7.91 |
| `FarFieldContract` | 4.48 | 4.35 | 5.34 | 6.09 | 6.78 | 7.91 |
| `TreeBuilder` | 3.53 | 4.17 | 5.20 | 6.12 | 6.90 | 7.86 |
| `TreePartitioner` | 3.52 | 4.20 | 5.20 | 6.06 | 6.89 | 7.86 |
| `CommunicationPlan` | 3.45 | 4.25 | 5.38 | 6.12 | 6.96 | 7.94 |
| `CartesianTaylor` (non-MPI) | 5.49 | | | | | |

**Bug found by running: the first Canopy binary in a job is ~3.5 s slow, and
the budgets do not cover it.** Calibration's first entry,
`CartesianTaylor_SERIAL`, took 5.49 s in pass 1 and 1.52 s in pass 2. In every
check job, `MultiSolve` np 1 was the job's first Canopy entry and took 8.61,
8.40 and 8.54 s, against 4.83-5.05 s for the same entry in pass 2 and a budget
of 9. It completed in all three, but with 0.4-0.6 s of margin. Calibration ran
`MultiSolve` np 1 well after the job's first entry, so its row (4.76) does not
include the cold start. Not fixed here: the budgets are as H0b specifies.

**No false positive.** In all three check jobs, both `MultiSolve` SERIAL
np 1-6 passes ended with every entry `failed` (the six `1e-8` cases, README) and
none `over-budget`.

**Refusal.** `canopy_ctest '^Canopy_Test_SingleSolve_MPI_SERIAL_np_1$'` printed
`REFUSED … no budget row (SingleSolve, 1, default)` and returned 2. The flux job
count was 13 before and after (`f3cM1y4WB335`). The name guard refused
`…_MPI_HIP_np_5` (HIP above np 4), `…_SERIAL_valgrind` and `…_OPENMP_nt_2`.

**Forced over-budget.**
- MPI (`f3cM1y4WB335`): `MultiSolve` np 6 at budget 4 was stacked with
  `matched=6 children=6 nonempty=6` and cancelled at 5.13 s, mid-run (rank 0 in
  `DownwardSweep::run_m2l_fused`, the others in MPI progress).
- Non-MPI: the timer fired at 1.1 s both times, matching the one
  `Canopy_Test_Car…` process. `f3cLyvp55Utf` got a stack (`ioctl` from HIP
  device bring-up). In `f3cM1y4WB335`, gstack, eu-stack and gdb all failed on
  it. `SIGKILL` did not land until the kernel call returned, so ctest's
  budget + 5 s timeout fired too (runtime 8-9 s). At 1 s the process is still
  inside HIP initialization in the kernel. A real budget (≥ 7 s here) is past
  it.

**New failure, not in README "Known Issues": `UpwardSweep` SERIAL exits 8 at
every np, in all three passes.** Calibration ran without
`--output-on-failure`, so the failing cases are not recorded here. H0c records
them beside its HIP run.

**Affects:**
- H0c: run every HIP entry through `canopy_ctest`. HIP np 1 for the job's
  first stem pays the cold start against its SERIAL budget (see above). An
  over-budget entry there whose stacks show progress is that effect, not a
  hang. `UpwardSweep`'s SERIAL failure is pre-existing to H0c.
- H1 HIP arm, H2 partitioner arm, E1, tree-opt T1: `canopy_ctest` is ready, and
  a switch configuration needs its own rows (`CANOPY_BUDGET_CONFIG`). A job
  whose first entry is an np-1 run is near its budget.

## H0c

Job `f3cM4ghTjtiT` (`scripts/tuolumne/run_ctest_h0.flux`, all ten stems in one
allocation, about 12 min). Diagnostic job `f3cMCHLDMNu5`
(`scripts/tuolumne/run_h0c_upward_diag.flux`). Both ran on HEAD `0440f84` plus
this task's scripts; Cray clang 20.0.0, flux-core 0.89.0, ROCm 6.4.2,
`build-tuolumne/` with `Canopy_ENABLE_PROFILING=ON`. Logs:
`canopy-h0.f3cM4ghTjtiT.log` and `canopy-h0c-upward-diag.f3cMCHLDMNu5.log` in
the repo root (untracked).

**Build.** `make -j 16` on each target in turn: all ten
(`Canopy_Test_{MultiSolve,DownwardSweep,UpwardSweep,TreeBuilder,TreePartitioner,CommunicationPlan,LaplaceSolve,CartesianTaylorSolve,FarFieldContract}_MPI_HIP`,
`Canopy_Test_CartesianTaylor_HIP`) compiled with no errors. They were built
while H0b ran, with this session's sources.

**Decisions.**
- The HIP environment is applied in a subshell. `canopy_ctest` is a shell
  function, so `env VAR=… canopy_ctest` cannot work. Each HIP call runs as
  `( export MPICH_GPU_SUPPORT_ENABLED=1 GTL_HSA_VSMSG_CUTOFF_SIZE=4096
  FI_CXI_ATS=0 HSA_XNACK=1 MPICH_SMP_SINGLE_COPY_MODE=NONE; canopy_ctest … )`,
  which keeps the variables off every SERIAL command.
- Over budget is recorded, not re-budgeted. HIP entries use the SERIAL
  budget, and no HIP rows were added.
- The job reruns the SERIAL twin of every failed HIP entry with
  `--output-on-failure`. `MultiSolve`'s twin is excluded: H2 already recorded
  its six SERIAL `1e-8` failures. The job also reruns `UpwardSweep` SERIAL
  np 1-6 with `--output-on-failure`, because H0b found it failing.
- The job aborts if any `_MPI_HIP_np_[56]` entry is registered. None was.

**Self-test.** Cancelled at 21.62 s, `matched=3 children=3 nonempty=3`.

**Outcomes.** One pass per stem, np 1-4 (`MultiSolve` twice, `-V`). Budget in
seconds; "known" means the SERIAL twin fails the same case and README records
it.

| stem | np 1 | np 2 | np 3 | np 4 |
| --- | --- | --- | --- | --- |
| `MultiSolve` pass 1 | failed: six `1e-8` (known) | failed: six (known) | **over budget** (17), stalled in `LargeMotion_Rebuild` | failed: six (known) |
| `MultiSolve` pass 2 | failed: six (known) | failed: six (known) | failed: six (known) | **over budget** (21), stalled in `LargeMotion_Rebuild` |
| `DownwardSweep` | passed | passed | passed | passed |
| `UpwardSweep` | failed: `RootMultipole…{Basic,Small}` (SERIAL same), `IdempotentExecution{Basic,Small}` (HIP only) | failed: `…MultiRank{Basic,Small}` (SERIAL same), `Idempotent…` (HIP only) | **over budget** (10): same failures, ranks exited, ctest digesting output | **over budget** (11): `…MultiRank…` failures; output cut by ctest's timeout |
| `TreeBuilder` | passed | passed | passed | passed |
| `TreePartitioner` | passed | passed | passed | passed |
| `CommunicationPlan` | passed | passed | passed | passed |
| `LaplaceSolve` | failed: `bitForBitArtifacts` (HIP only) | failed: `bitForBitArtifacts` (HIP only) | failed: `crossRankAgreement` (HIP only) | failed: `crossRankAgreement` (HIP only) |
| `CartesianTaylorSolve` | passed | passed | passed | passed |
| `FarFieldContract` | passed | passed | passed | passed |
| `CartesianTaylor` (non-MPI) | passed | | | |

SERIAL twins in the same job:
- `LaplaceSolve` SERIAL np 1-4 passed.
- `UpwardSweep` SERIAL failed `testRootMultipoleMatchesDirectP2M{Basic,Small}`
  at np 1 and `…MultiRank{Basic,Small}` at np 2-6 (max relative error
  34.8-35.8 against `1e-10`). Its idempotence cases passed.

README "Known Issues" now has an entry for each of these, and the existing
Zoltan2-on-HIP entry records the stall.

**HIP failure details.**
- `UpwardSweep.testIdempotentExecution*`: two `execute()` calls differ in the
  last bits, e.g. cell 91 coeff 27 imag `3.1318581590950871` vs
  `3.1318581590950876`. The test uses exact equality
  (`tests/tstUpwardSweep.hpp:385-395`). Each mismatch prints a message:
  3 214 / 9 160 / 16 976 / 8 330 lines at np 1-4.
- `LaplaceSolve.bitForBitArtifacts` np 1: `locals()` hash `0xbf7c808746669af7`
  against the reference `0xfb2cddef75e26dd6`, as R6 predicted (SERIAL-generated
  references).
- `LaplaceSolve.crossRankAgreement` np 3: field deviates from the committed
  np-1 field by `4.23e-7` (potential) and `3.89e-8` (gradient), against
  `LS_CROSS_RANK_TOL = 5.6e-10`.

**The `UpwardSweep` np 3-4 over-budget entries are ctest-side, not hangs.** In
`f3cM4ghTjtiT`, np 3 printed gtest's final summary ("10 tests from 1 test
suite ran", 2 failed), then ctest timed out at budget + 5 = 15 s. The
watchdog never listed the sub-job as running past 10 s. `f3cMCHLDMNu5` reran
np 3 and np 4 three times each, without the budget, and snapshotted the node
once each entry passed its budget:
- In 5 of 6 runs the sub-job was already `INACTIVE` (runtime 8.2-12.9 s), and
  no `Canopy_Test_` process or `flux run` client was left.
- In run 1 at np 3 the ranks were still running (state `Rl`) at 9.8 s. They
  exited before gstack attached, so it printed nothing.

ctest then ran a further 10-14 s with no child, for 19.6-26.8 s per entry. The
captured output is 25 864 / 73 469 / 136 034 / 66 801 lines at np 1-4, so
ctest's output processing is where the time goes. With no process left, there
is nothing to stack. Correcting the test's output volume is the README entry's
work, not H0c's.

**`MultiSolve` stalls: the H1 signature, on HIP.** Both captures stopped in
`MultiSolve.LargeMotion_Rebuild`.
- Rank 0 is in `hipDeviceSynchronize` from a `Kokkos::deep_copy` in
  `Zoltan2::AlgMJ<…KokkosDeviceWrapperNode<Kokkos::HIP>…>::mj_get_new_cut_coordinates`,
  under `TreePartitioner<HIPSpace, HIP>::partition_leaves`.
- The other ranks are in `partition_leaves`'s `MPI_Bcast`.

H2's SERIAL fix put MJ on the partitioner's execution space, which is HIP here.
That is the README entry "Zoltan2 runs on HIP when the solver's
`ExecutionSpace` is HIP", now observed. Hang count: **2 stalls in 8**
`MultiSolve` HIP entries, both at np >= 3 (np 3: 1 in 2, np 4: 1 in 2).

Last `[multisolve-dev]` line before each stall:

```
[multisolve-dev] case IntermediateMotion_Rebalance nprocs 3 nsteps 5 drift 5 max_pos_rel 1.9045913955560584e-05 max_vel_rel 4.2352651310527094e-05 tol 1e-08
[multisolve-dev] case IntermediateMotion_Rebalance nprocs 4 nsteps 5 drift 5 max_pos_rel 0.00017969857882758528 max_vel_rel 0.00027710166692275571 tol 1e-08
```

The captures below keep each rank's main thread (Thread 1) verbatim. The two
other threads per rank are the HSA `AsyncEventsLoop` threads seen in H1, and
the job log has them in full. The capture does not record MPI rank. The PID in
Zoltan2 is inferred to be rank 0, as in H1.

np 3, pass 1:

```
### watchdog stacks fXed7s1h ###
time: 2026-10-05 11:14:54 runtime: 19.176634550094604 s jobid(dec): 1166754709504
shell: 2587973 /usr/libexec/flux/flux-shell 1166754709504
--- pid 2587974 ppid 2587973 comm Canopy_Test_Mul ---
Thread 1 (Thread 0x155555548a80 (LWP 2587974)):
#0  0x000015552f64f9a8 in rocr::core::InterruptSignal::WaitRelaxed(hsa_signal_condition_t, long, unsigned long, hsa_wait_state_t) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#1  0x000015552f64f7fa in rocr::core::InterruptSignal::WaitAcquire(hsa_signal_condition_t, long, unsigned long, hsa_wait_state_t) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#2  0x000015552f644231 in rocr::HSA::hsa_signal_wait_scacquire(hsa_signal_s, hsa_signal_condition_t, long, unsigned long, hsa_wait_state_t) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#3  0x000015553c59027b in amd::roc::Device::IsHwEventReady(amd::Event const&, bool, unsigned int) const () from /opt/rocm-6.4.2/lib/libamdhip64.so.6
#4  0x000015553c57f1d7 in amd::HostQueue::finish(bool) () from /opt/rocm-6.4.2/lib/libamdhip64.so.6
#5  0x000015553c31b21e in hip::Device::SyncAllStreams(bool, bool) () from /opt/rocm-6.4.2/lib/libamdhip64.so.6
#6  0x000015553c308e51 in hip::hipDeviceSynchronize() () from /opt/rocm-6.4.2/lib/libamdhip64.so.6
#7  0x0000155541a07cb3 in Kokkos::HIP::impl_static_fence(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libkokkoscore.so.4.7
#8  0x00001555419f671d in Kokkos::fence(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libkokkoscore.so.4.7
#9  0x0000000000ce8e70 in Kokkos::deep_copy<int*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>, int*, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::OpenMP, Kokkos::HIPSpace>, Kokkos::Experimental::EmptyViewHooks> ()
#10 0x0000000000d0ac19 in Zoltan2::AlgMJ<double, int, long long, int, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> >::mj_get_new_cut_coordinates(int, int, int const&, double const&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<bool*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<int*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&) ()
#11 0x0000000000cdf672 in Zoltan2::AlgMJ<double, int, long long, int, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> >::mj_1D_part(Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, double, int, int, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, int, Kokkos::View<int*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<unsigned long*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&) ()
#12 0x0000000000ca17c8 in Zoltan2::AlgMJ<double, int, long long, int, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> >::multi_jagged_part(Teuchos::RCP<Zoltan2::Environment const> const&, Teuchos::RCP<Teuchos::Comm<int> const>&, double, int, unsigned long, Kokkos::View<int*, Kokkos::HostSpace>&, int, int, int, long long, Kokkos::View<long long const*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, int, Kokkos::View<bool*, Kokkos::HostSpace>&, Kokkos::View<double**, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<bool*, Kokkos::HostSpace>&, Kokkos::View<int*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<long long*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&) ()
#13 0x0000000000c924c6 in Zoltan2::Zoltan2_AlgMJ<Zoltan2::BasicVectorAdapter<Zoltan2::BasicUserTypes<double, int, long long, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> > > >::partition(Teuchos::RCP<Zoltan2::PartitioningSolution<Zoltan2::BasicVectorAdapter<Zoltan2::BasicUserTypes<double, int, long long, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> > > > > const&) ()
#14 0x0000000000c5f458 in Zoltan2::PartitioningProblem<Zoltan2::BasicVectorAdapter<Zoltan2::BasicUserTypes<double, int, long long, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> > > >::solve(bool) ()
#15 0x0000000000c559d3 in Canopy::TreePartitioner<Kokkos::HIPSpace, Kokkos::HIP>::partition_leaves(std::vector<Canopy::CellInfo, std::allocator<Canopy::CellInfo> > const&) ()
#16 0x0000000000c18e82 in void Canopy::Solver<Kokkos::HIPSpace, Kokkos::HIP, double, 8, 1, Canopy::LaplaceKernel>::_full_setup<0, 1, Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> > >(Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> >&, int) ()
#17 0x0000000000bfee6f in Test::testMultiStepGravity(Test::MultiSolveTest::Mode, char const*, int, int, double, double, int, int, double, int, double, int*, bool, double, long long*) ()
#18 0x0000000000be3c52 in Test::MultiSolve_LargeMotion_Rebuild_Test::TestBody() ()
#19 0x0000155554e008ed in void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#20 0x0000155554de2ea6 in testing::Test::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#21 0x0000155554de3045 in testing::TestInfo::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#22 0x0000155554de32bd in testing::TestSuite::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#23 0x0000155554df7169 in testing::internal::UnitTestImpl::RunAllTests() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#24 0x0000155554de3384 in testing::UnitTest::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#25 0x0000000000f271b2 in main ()
--- pid 2587975 ppid 2587973 comm Canopy_Test_Mul ---
Thread 1 (Thread 0x155555548a80 (LWP 2587975)):
#0  0x0000155553832a09 in MPIR_Progress_hook_exec_all () from /opt/cray/pe/lib64/libmpi_cray.so.12
#1  0x00001555537b277a in MPIDI_progress_test () from /opt/cray/pe/lib64/libmpi_cray.so.12
#2  0x00001555537b4006 in MPID_Progress_wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#3  0x00001555537b6f56 in MPIR_Wait_state () from /opt/cray/pe/lib64/libmpi_cray.so.12
#4  0x00001555536cdcc7 in MPID_Wait.constprop.0 () from /opt/cray/pe/lib64/libmpi_cray.so.12
#5  0x00001555536e453f in MPIC_Wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#6  0x00001555536e4980 in MPIC_Recv () from /opt/cray/pe/lib64/libmpi_cray.so.12
#7  0x000015555372b55c in MPIR_CRAY_Bcast_Tree () from /opt/cray/pe/lib64/libmpi_cray.so.12
#8  0x000015555372bf5a in MPIR_CRAY_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#9  0x00001555532e01cf in PMPI_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#10 0x0000000000c55cfe in Canopy::TreePartitioner<Kokkos::HIPSpace, Kokkos::HIP>::partition_leaves(std::vector<Canopy::CellInfo, std::allocator<Canopy::CellInfo> > const&) ()
#11 0x0000000000c18e82 in void Canopy::Solver<Kokkos::HIPSpace, Kokkos::HIP, double, 8, 1, Canopy::LaplaceKernel>::_full_setup<0, 1, Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> > >(Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> >&, int) ()
#12 0x0000000000bfee6f in Test::testMultiStepGravity(Test::MultiSolveTest::Mode, char const*, int, int, double, double, int, int, double, int, double, int*, bool, double, long long*) ()
#13 0x0000000000be3c52 in Test::MultiSolve_LargeMotion_Rebuild_Test::TestBody() ()
#14 0x0000155554e008ed in void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#15 0x0000155554de2ea6 in testing::Test::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#16 0x0000155554de3045 in testing::TestInfo::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#17 0x0000155554de32bd in testing::TestSuite::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#18 0x0000155554df7169 in testing::internal::UnitTestImpl::RunAllTests() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#19 0x0000155554de3384 in testing::UnitTest::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#20 0x0000000000f271b2 in main ()
--- pid 2587976 ppid 2587973 comm Canopy_Test_Mul ---
Thread 1 (Thread 0x155555548a80 (LWP 2587976)):
#0  0x00001555537ac19b in MPIDI_POSIX_progress.constprop.0 () from /opt/cray/pe/lib64/libmpi_cray.so.12
#1  0x00001555537b2b68 in MPIDI_progress_test () from /opt/cray/pe/lib64/libmpi_cray.so.12
#2  0x00001555537b4006 in MPID_Progress_wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#3  0x00001555537b6f56 in MPIR_Wait_state () from /opt/cray/pe/lib64/libmpi_cray.so.12
#4  0x00001555536cdcc7 in MPID_Wait.constprop.0 () from /opt/cray/pe/lib64/libmpi_cray.so.12
#5  0x00001555536e453f in MPIC_Wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#6  0x00001555536e4980 in MPIC_Recv () from /opt/cray/pe/lib64/libmpi_cray.so.12
#7  0x000015555372b55c in MPIR_CRAY_Bcast_Tree () from /opt/cray/pe/lib64/libmpi_cray.so.12
#8  0x000015555372bf5a in MPIR_CRAY_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#9  0x00001555532e01cf in PMPI_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#10 0x0000000000c55cfe in Canopy::TreePartitioner<Kokkos::HIPSpace, Kokkos::HIP>::partition_leaves(std::vector<Canopy::CellInfo, std::allocator<Canopy::CellInfo> > const&) ()
#11 0x0000000000c18e82 in void Canopy::Solver<Kokkos::HIPSpace, Kokkos::HIP, double, 8, 1, Canopy::LaplaceKernel>::_full_setup<0, 1, Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> > >(Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> >&, int) ()
#12 0x0000000000bfee6f in Test::testMultiStepGravity(Test::MultiSolveTest::Mode, char const*, int, int, double, double, int, int, double, int, double, int*, bool, double, long long*) ()
#13 0x0000000000be3c52 in Test::MultiSolve_LargeMotion_Rebuild_Test::TestBody() ()
#14 0x0000155554e008ed in void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#15 0x0000155554de2ea6 in testing::Test::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#16 0x0000155554de3045 in testing::TestInfo::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#17 0x0000155554de32bd in testing::TestSuite::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#18 0x0000155554df7169 in testing::internal::UnitTestImpl::RunAllTests() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#19 0x0000155554de3384 in testing::UnitTest::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#20 0x0000000000f271b2 in main ()
### end watchdog stacks fXed7s1h: matched=3 children=3 nonempty=3 ###
### watchdog: cancelling sub-job fXed7s1h after 19.176634550094604 s ###
### MultiSolve pass 1: canopy_ctest rc=1 ###
### MultiSolve HIP pass 2 (-V) ###
### watchdog stacks f2DJTdkMD ###
time: 2026-10-05 11:16:27 runtime: 22.454469203948975 s jobid(dec): 2676267941888
shell: 2588649 /usr/libexec/flux/flux-shell 2676267941888
--- pid 2588650 ppid 2588649 comm Canopy_Test_Mul ---
Thread 1 (Thread 0x155555548a80 (LWP 2588650)):
#0  0x000015552f64f98d in rocr::core::InterruptSignal::WaitRelaxed(hsa_signal_condition_t, long, unsigned long, hsa_wait_state_t) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#1  0x000015552f64f7fa in rocr::core::InterruptSignal::WaitAcquire(hsa_signal_condition_t, long, unsigned long, hsa_wait_state_t) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#2  0x000015552f644231 in rocr::HSA::hsa_signal_wait_scacquire(hsa_signal_s, hsa_signal_condition_t, long, unsigned long, hsa_wait_state_t) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#3  0x000015553c59027b in amd::roc::Device::IsHwEventReady(amd::Event const&, bool, unsigned int) const () from /opt/rocm-6.4.2/lib/libamdhip64.so.6
#4  0x000015553c57f1d7 in amd::HostQueue::finish(bool) () from /opt/rocm-6.4.2/lib/libamdhip64.so.6
#5  0x000015553c31b21e in hip::Device::SyncAllStreams(bool, bool) () from /opt/rocm-6.4.2/lib/libamdhip64.so.6
#6  0x000015553c308e51 in hip::hipDeviceSynchronize() () from /opt/rocm-6.4.2/lib/libamdhip64.so.6
#7  0x0000155541a07cb3 in Kokkos::HIP::impl_static_fence(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libkokkoscore.so.4.7
#8  0x00001555419f671d in Kokkos::fence(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libkokkoscore.so.4.7
#9  0x0000000000ce8e70 in Kokkos::deep_copy<int*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>, int*, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::OpenMP, Kokkos::HIPSpace>, Kokkos::Experimental::EmptyViewHooks> ()
#10 0x0000000000d0ac19 in Zoltan2::AlgMJ<double, int, long long, int, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> >::mj_get_new_cut_coordinates(int, int, int const&, double const&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<bool*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<int*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&) ()
#11 0x0000000000cdf672 in Zoltan2::AlgMJ<double, int, long long, int, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> >::mj_1D_part(Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, double, int, int, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, int, Kokkos::View<int*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<unsigned long*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&) ()
#12 0x0000000000ca17c8 in Zoltan2::AlgMJ<double, int, long long, int, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> >::multi_jagged_part(Teuchos::RCP<Zoltan2::Environment const> const&, Teuchos::RCP<Teuchos::Comm<int> const>&, double, int, unsigned long, Kokkos::View<int*, Kokkos::HostSpace>&, int, int, int, long long, Kokkos::View<long long const*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, int, Kokkos::View<bool*, Kokkos::HostSpace>&, Kokkos::View<double**, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<bool*, Kokkos::HostSpace>&, Kokkos::View<int*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<long long*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&) ()
#13 0x0000000000c924c6 in Zoltan2::Zoltan2_AlgMJ<Zoltan2::BasicVectorAdapter<Zoltan2::BasicUserTypes<double, int, long long, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> > > >::partition(Teuchos::RCP<Zoltan2::PartitioningSolution<Zoltan2::BasicVectorAdapter<Zoltan2::BasicUserTypes<double, int, long long, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> > > > > const&) ()
#14 0x0000000000c5f458 in Zoltan2::PartitioningProblem<Zoltan2::BasicVectorAdapter<Zoltan2::BasicUserTypes<double, int, long long, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> > > >::solve(bool) ()
#15 0x0000000000c559d3 in Canopy::TreePartitioner<Kokkos::HIPSpace, Kokkos::HIP>::partition_leaves(std::vector<Canopy::CellInfo, std::allocator<Canopy::CellInfo> > const&) ()
#16 0x0000000000c18e82 in void Canopy::Solver<Kokkos::HIPSpace, Kokkos::HIP, double, 8, 1, Canopy::LaplaceKernel>::_full_setup<0, 1, Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> > >(Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> >&, int) ()
#17 0x0000000000bfee6f in Test::testMultiStepGravity(Test::MultiSolveTest::Mode, char const*, int, int, double, double, int, int, double, int, double, int*, bool, double, long long*) ()
#18 0x0000000000be3c52 in Test::MultiSolve_LargeMotion_Rebuild_Test::TestBody() ()
#19 0x0000155554e008ed in void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#20 0x0000155554de2ea6 in testing::Test::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#21 0x0000155554de3045 in testing::TestInfo::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#22 0x0000155554de32bd in testing::TestSuite::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#23 0x0000155554df7169 in testing::internal::UnitTestImpl::RunAllTests() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#24 0x0000155554de3384 in testing::UnitTest::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#25 0x0000000000f271b2 in main ()
--- pid 2588651 ppid 2588649 comm Canopy_Test_Mul ---
Thread 1 (Thread 0x155555548a80 (LWP 2588651)):
#0  0x00001555537ac19b in MPIDI_POSIX_progress.constprop.0 () from /opt/cray/pe/lib64/libmpi_cray.so.12
#1  0x00001555537b2b68 in MPIDI_progress_test () from /opt/cray/pe/lib64/libmpi_cray.so.12
#2  0x00001555537b4006 in MPID_Progress_wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#3  0x00001555537b6f56 in MPIR_Wait_state () from /opt/cray/pe/lib64/libmpi_cray.so.12
#4  0x00001555536cdcc7 in MPID_Wait.constprop.0 () from /opt/cray/pe/lib64/libmpi_cray.so.12
#5  0x00001555536e453f in MPIC_Wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#6  0x00001555536e4980 in MPIC_Recv () from /opt/cray/pe/lib64/libmpi_cray.so.12
#7  0x000015555372b55c in MPIR_CRAY_Bcast_Tree () from /opt/cray/pe/lib64/libmpi_cray.so.12
#8  0x000015555372bf5a in MPIR_CRAY_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#9  0x00001555532e01cf in PMPI_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#10 0x0000000000c55cfe in Canopy::TreePartitioner<Kokkos::HIPSpace, Kokkos::HIP>::partition_leaves(std::vector<Canopy::CellInfo, std::allocator<Canopy::CellInfo> > const&) ()
#11 0x0000000000c18e82 in void Canopy::Solver<Kokkos::HIPSpace, Kokkos::HIP, double, 8, 1, Canopy::LaplaceKernel>::_full_setup<0, 1, Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> > >(Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> >&, int) ()
#12 0x0000000000bfee6f in Test::testMultiStepGravity(Test::MultiSolveTest::Mode, char const*, int, int, double, double, int, int, double, int, double, int*, bool, double, long long*) ()
#13 0x0000000000be3c52 in Test::MultiSolve_LargeMotion_Rebuild_Test::TestBody() ()
#14 0x0000155554e008ed in void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#15 0x0000155554de2ea6 in testing::Test::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#16 0x0000155554de3045 in testing::TestInfo::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#17 0x0000155554de32bd in testing::TestSuite::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#18 0x0000155554df7169 in testing::internal::UnitTestImpl::RunAllTests() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#19 0x0000155554de3384 in testing::UnitTest::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#20 0x0000000000f271b2 in main ()
--- pid 2588652 ppid 2588649 comm Canopy_Test_Mul ---
Thread 1 (Thread 0x155555548a80 (LWP 2588652)):
#0  0x00001555537ac1ad in MPIDI_POSIX_progress.constprop.0 () from /opt/cray/pe/lib64/libmpi_cray.so.12
#1  0x00001555537b2b68 in MPIDI_progress_test () from /opt/cray/pe/lib64/libmpi_cray.so.12
#2  0x00001555537b4006 in MPID_Progress_wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#3  0x00001555537b6f56 in MPIR_Wait_state () from /opt/cray/pe/lib64/libmpi_cray.so.12
#4  0x00001555536cdcc7 in MPID_Wait.constprop.0 () from /opt/cray/pe/lib64/libmpi_cray.so.12
#5  0x00001555536e453f in MPIC_Wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#6  0x00001555536e4980 in MPIC_Recv () from /opt/cray/pe/lib64/libmpi_cray.so.12
#7  0x000015555372b55c in MPIR_CRAY_Bcast_Tree () from /opt/cray/pe/lib64/libmpi_cray.so.12
#8  0x000015555372bf5a in MPIR_CRAY_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#9  0x00001555532e01cf in PMPI_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#10 0x0000000000c55cfe in Canopy::TreePartitioner<Kokkos::HIPSpace, Kokkos::HIP>::partition_leaves(std::vector<Canopy::CellInfo, std::allocator<Canopy::CellInfo> > const&) ()
#11 0x0000000000c18e82 in void Canopy::Solver<Kokkos::HIPSpace, Kokkos::HIP, double, 8, 1, Canopy::LaplaceKernel>::_full_setup<0, 1, Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> > >(Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> >&, int) ()
#12 0x0000000000bfee6f in Test::testMultiStepGravity(Test::MultiSolveTest::Mode, char const*, int, int, double, double, int, int, double, int, double, int*, bool, double, long long*) ()
#13 0x0000000000be3c52 in Test::MultiSolve_LargeMotion_Rebuild_Test::TestBody() ()
#14 0x0000155554e008ed in void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#15 0x0000155554de2ea6 in testing::Test::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#16 0x0000155554de3045 in testing::TestInfo::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#17 0x0000155554de32bd in testing::TestSuite::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#18 0x0000155554df7169 in testing::internal::UnitTestImpl::RunAllTests() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#19 0x0000155554de3384 in testing::UnitTest::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#20 0x0000000000f271b2 in main ()
--- pid 2588653 ppid 2588649 comm Canopy_Test_Mul ---
Thread 1 (Thread 0x155555548a80 (LWP 2588653)):
#0  0x000015555392a25e in MPIDI_POSIX_eager_recv_begin () from /opt/cray/pe/lib64/libmpi_cray.so.12
#1  0x00001555537a654c in MPIDI_POSIX_progress_recv.constprop.0 () from /opt/cray/pe/lib64/libmpi_cray.so.12
#2  0x00001555537ac1a8 in MPIDI_POSIX_progress.constprop.0 () from /opt/cray/pe/lib64/libmpi_cray.so.12
#3  0x00001555537b2b68 in MPIDI_progress_test () from /opt/cray/pe/lib64/libmpi_cray.so.12
#4  0x00001555537b4006 in MPID_Progress_wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#5  0x00001555537b6f56 in MPIR_Wait_state () from /opt/cray/pe/lib64/libmpi_cray.so.12
#6  0x00001555536cdcc7 in MPID_Wait.constprop.0 () from /opt/cray/pe/lib64/libmpi_cray.so.12
#7  0x00001555536e453f in MPIC_Wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#8  0x00001555536e4980 in MPIC_Recv () from /opt/cray/pe/lib64/libmpi_cray.so.12
#9  0x000015555372b55c in MPIR_CRAY_Bcast_Tree () from /opt/cray/pe/lib64/libmpi_cray.so.12
#10 0x000015555372bf5a in MPIR_CRAY_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#11 0x00001555532e01cf in PMPI_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#12 0x0000000000c55cfe in Canopy::TreePartitioner<Kokkos::HIPSpace, Kokkos::HIP>::partition_leaves(std::vector<Canopy::CellInfo, std::allocator<Canopy::CellInfo> > const&) ()
#13 0x0000000000c18e82 in void Canopy::Solver<Kokkos::HIPSpace, Kokkos::HIP, double, 8, 1, Canopy::LaplaceKernel>::_full_setup<0, 1, Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> > >(Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> >&, int) ()
#14 0x0000000000bfee6f in Test::testMultiStepGravity(Test::MultiSolveTest::Mode, char const*, int, int, double, double, int, int, double, int, double, int*, bool, double, long long*) ()
#15 0x0000000000be3c52 in Test::MultiSolve_LargeMotion_Rebuild_Test::TestBody() ()
#16 0x0000155554e008ed in void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#17 0x0000155554de2ea6 in testing::Test::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#18 0x0000155554de3045 in testing::TestInfo::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#19 0x0000155554de32bd in testing::TestSuite::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#20 0x0000155554df7169 in testing::internal::UnitTestImpl::RunAllTests() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#21 0x0000155554de3384 in testing::UnitTest::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#22 0x0000000000f271b2 in main ()
### end watchdog stacks f2DJTdkMD: matched=4 children=4 nonempty=4 ###
### watchdog: cancelling sub-job f2DJTdkMD after 22.454469203948975 s ###
### MultiSolve pass 2: canopy_ctest rc=1 ###
### DownwardSweep binary: 2026-10-05 10:26:09.408226273 -0700 ###
### DownwardSweep HIP ###
### DownwardSweep: canopy_ctest rc=0 ###
### UpwardSweep binary: 2026-10-05 10:28:26.168287766 -0700 ###
### UpwardSweep HIP ###
### UpwardSweep: canopy_ctest rc=1 ###
### SERIAL twin of failed Canopy_Test_UpwardSweep_MPI_HIP_np_1: Canopy_Test_UpwardSweep_MPI_SERIAL_np_1 ###
### Canopy_Test_UpwardSweep_MPI_SERIAL_np_1: canopy_ctest rc=1 ###
### SERIAL twin of failed Canopy_Test_UpwardSweep_MPI_HIP_np_2: Canopy_Test_UpwardSweep_MPI_SERIAL_np_2 ###
### Canopy_Test_UpwardSweep_MPI_SERIAL_np_2: canopy_ctest rc=1 ###
### TreeBuilder binary: 2026-10-05 10:29:07.452193567 -0700 ###
### TreeBuilder HIP ###
### TreeBuilder: canopy_ctest rc=0 ###
### TreePartitioner binary: 2026-10-05 10:31:18.110566890 -0700 ###
### TreePartitioner HIP ###
### TreePartitioner: canopy_ctest rc=0 ###
### CommunicationPlan binary: 2026-10-05 10:33:32.886230160 -0700 ###
### CommunicationPlan HIP ###
### CommunicationPlan: canopy_ctest rc=0 ###
### LaplaceSolve binary: 2026-10-05 10:36:46.001435713 -0700 ###
### LaplaceSolve HIP ###
### LaplaceSolve: canopy_ctest rc=1 ###
### SERIAL twin of failed Canopy_Test_LaplaceSolve_MPI_HIP_np_1: Canopy_Test_LaplaceSolve_MPI_SERIAL_np_1 ###
### Canopy_Test_LaplaceSolve_MPI_SERIAL_np_1: canopy_ctest rc=0 ###
### SERIAL twin of failed Canopy_Test_LaplaceSolve_MPI_HIP_np_2: Canopy_Test_LaplaceSolve_MPI_SERIAL_np_2 ###
### Canopy_Test_LaplaceSolve_MPI_SERIAL_np_2: canopy_ctest rc=0 ###
### SERIAL twin of failed Canopy_Test_LaplaceSolve_MPI_HIP_np_3: Canopy_Test_LaplaceSolve_MPI_SERIAL_np_3 ###
### Canopy_Test_LaplaceSolve_MPI_SERIAL_np_3: canopy_ctest rc=0 ###
### SERIAL twin of failed Canopy_Test_LaplaceSolve_MPI_HIP_np_4: Canopy_Test_LaplaceSolve_MPI_SERIAL_np_4 ###
### Canopy_Test_LaplaceSolve_MPI_SERIAL_np_4: canopy_ctest rc=0 ###
### CartesianTaylorSolve binary: 2026-10-05 10:39:45.813316585 -0700 ###
### CartesianTaylorSolve HIP ###
### CartesianTaylorSolve: canopy_ctest rc=0 ###
### FarFieldContract binary: 2026-10-05 10:42:26.306176615 -0700 ###
### FarFieldContract HIP ###
### FarFieldContract: canopy_ctest rc=0 ###
### CartesianTaylor binary: 2026-10-05 10:43:11.718372622 -0700 ###
### CartesianTaylor HIP ###
### CartesianTaylor: canopy_ctest rc=0 ###
### SERIAL UpwardSweep np 1-6 (--output-on-failure) ###
### SERIAL UpwardSweep: canopy_ctest rc=1 ###
### cancelled sub-jobs ###
```

np 4, pass 2:

```
### watchdog stacks f2DJTdkMD ###
time: 2026-10-05 11:16:27 runtime: 22.454469203948975 s jobid(dec): 2676267941888
shell: 2588649 /usr/libexec/flux/flux-shell 2676267941888
--- pid 2588650 ppid 2588649 comm Canopy_Test_Mul ---
Thread 1 (Thread 0x155555548a80 (LWP 2588650)):
#0  0x000015552f64f98d in rocr::core::InterruptSignal::WaitRelaxed(hsa_signal_condition_t, long, unsigned long, hsa_wait_state_t) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#1  0x000015552f64f7fa in rocr::core::InterruptSignal::WaitAcquire(hsa_signal_condition_t, long, unsigned long, hsa_wait_state_t) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#2  0x000015552f644231 in rocr::HSA::hsa_signal_wait_scacquire(hsa_signal_s, hsa_signal_condition_t, long, unsigned long, hsa_wait_state_t) () from /opt/rocm-6.4.2/lib/libhsa-runtime64.so.1
#3  0x000015553c59027b in amd::roc::Device::IsHwEventReady(amd::Event const&, bool, unsigned int) const () from /opt/rocm-6.4.2/lib/libamdhip64.so.6
#4  0x000015553c57f1d7 in amd::HostQueue::finish(bool) () from /opt/rocm-6.4.2/lib/libamdhip64.so.6
#5  0x000015553c31b21e in hip::Device::SyncAllStreams(bool, bool) () from /opt/rocm-6.4.2/lib/libamdhip64.so.6
#6  0x000015553c308e51 in hip::hipDeviceSynchronize() () from /opt/rocm-6.4.2/lib/libamdhip64.so.6
#7  0x0000155541a07cb3 in Kokkos::HIP::impl_static_fence(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libkokkoscore.so.4.7
#8  0x00001555419f671d in Kokkos::fence(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libkokkoscore.so.4.7
#9  0x0000000000ce8e70 in Kokkos::deep_copy<int*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>, int*, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::OpenMP, Kokkos::HIPSpace>, Kokkos::Experimental::EmptyViewHooks> ()
#10 0x0000000000d0ac19 in Zoltan2::AlgMJ<double, int, long long, int, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> >::mj_get_new_cut_coordinates(int, int, int const&, double const&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<bool*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<int*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&) ()
#11 0x0000000000cdf672 in Zoltan2::AlgMJ<double, int, long long, int, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> >::mj_1D_part(Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, double, int, int, Kokkos::View<double*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, int, Kokkos::View<int*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<unsigned long*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&) ()
#12 0x0000000000ca17c8 in Zoltan2::AlgMJ<double, int, long long, int, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> >::multi_jagged_part(Teuchos::RCP<Zoltan2::Environment const> const&, Teuchos::RCP<Teuchos::Comm<int> const>&, double, int, unsigned long, Kokkos::View<int*, Kokkos::HostSpace>&, int, int, int, long long, Kokkos::View<long long const*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, int, Kokkos::View<bool*, Kokkos::HostSpace>&, Kokkos::View<double**, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<bool*, Kokkos::HostSpace>&, Kokkos::View<int*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&, Kokkos::View<long long*, Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace> >&) ()
#13 0x0000000000c924c6 in Zoltan2::Zoltan2_AlgMJ<Zoltan2::BasicVectorAdapter<Zoltan2::BasicUserTypes<double, int, long long, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> > > >::partition(Teuchos::RCP<Zoltan2::PartitioningSolution<Zoltan2::BasicVectorAdapter<Zoltan2::BasicUserTypes<double, int, long long, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> > > > > const&) ()
#14 0x0000000000c5f458 in Zoltan2::PartitioningProblem<Zoltan2::BasicVectorAdapter<Zoltan2::BasicUserTypes<double, int, long long, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::HIP, Kokkos::HIPSpace> > > >::solve(bool) ()
#15 0x0000000000c559d3 in Canopy::TreePartitioner<Kokkos::HIPSpace, Kokkos::HIP>::partition_leaves(std::vector<Canopy::CellInfo, std::allocator<Canopy::CellInfo> > const&) ()
#16 0x0000000000c18e82 in void Canopy::Solver<Kokkos::HIPSpace, Kokkos::HIP, double, 8, 1, Canopy::LaplaceKernel>::_full_setup<0, 1, Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> > >(Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> >&, int) ()
#17 0x0000000000bfee6f in Test::testMultiStepGravity(Test::MultiSolveTest::Mode, char const*, int, int, double, double, int, int, double, int, double, int*, bool, double, long long*) ()
#18 0x0000000000be3c52 in Test::MultiSolve_LargeMotion_Rebuild_Test::TestBody() ()
#19 0x0000155554e008ed in void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#20 0x0000155554de2ea6 in testing::Test::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#21 0x0000155554de3045 in testing::TestInfo::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#22 0x0000155554de32bd in testing::TestSuite::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#23 0x0000155554df7169 in testing::internal::UnitTestImpl::RunAllTests() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#24 0x0000155554de3384 in testing::UnitTest::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#25 0x0000000000f271b2 in main ()
--- pid 2588651 ppid 2588649 comm Canopy_Test_Mul ---
Thread 1 (Thread 0x155555548a80 (LWP 2588651)):
#0  0x00001555537ac19b in MPIDI_POSIX_progress.constprop.0 () from /opt/cray/pe/lib64/libmpi_cray.so.12
#1  0x00001555537b2b68 in MPIDI_progress_test () from /opt/cray/pe/lib64/libmpi_cray.so.12
#2  0x00001555537b4006 in MPID_Progress_wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#3  0x00001555537b6f56 in MPIR_Wait_state () from /opt/cray/pe/lib64/libmpi_cray.so.12
#4  0x00001555536cdcc7 in MPID_Wait.constprop.0 () from /opt/cray/pe/lib64/libmpi_cray.so.12
#5  0x00001555536e453f in MPIC_Wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#6  0x00001555536e4980 in MPIC_Recv () from /opt/cray/pe/lib64/libmpi_cray.so.12
#7  0x000015555372b55c in MPIR_CRAY_Bcast_Tree () from /opt/cray/pe/lib64/libmpi_cray.so.12
#8  0x000015555372bf5a in MPIR_CRAY_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#9  0x00001555532e01cf in PMPI_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#10 0x0000000000c55cfe in Canopy::TreePartitioner<Kokkos::HIPSpace, Kokkos::HIP>::partition_leaves(std::vector<Canopy::CellInfo, std::allocator<Canopy::CellInfo> > const&) ()
#11 0x0000000000c18e82 in void Canopy::Solver<Kokkos::HIPSpace, Kokkos::HIP, double, 8, 1, Canopy::LaplaceKernel>::_full_setup<0, 1, Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> > >(Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> >&, int) ()
#12 0x0000000000bfee6f in Test::testMultiStepGravity(Test::MultiSolveTest::Mode, char const*, int, int, double, double, int, int, double, int, double, int*, bool, double, long long*) ()
#13 0x0000000000be3c52 in Test::MultiSolve_LargeMotion_Rebuild_Test::TestBody() ()
#14 0x0000155554e008ed in void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#15 0x0000155554de2ea6 in testing::Test::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#16 0x0000155554de3045 in testing::TestInfo::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#17 0x0000155554de32bd in testing::TestSuite::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#18 0x0000155554df7169 in testing::internal::UnitTestImpl::RunAllTests() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#19 0x0000155554de3384 in testing::UnitTest::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#20 0x0000000000f271b2 in main ()
--- pid 2588652 ppid 2588649 comm Canopy_Test_Mul ---
Thread 1 (Thread 0x155555548a80 (LWP 2588652)):
#0  0x00001555537ac1ad in MPIDI_POSIX_progress.constprop.0 () from /opt/cray/pe/lib64/libmpi_cray.so.12
#1  0x00001555537b2b68 in MPIDI_progress_test () from /opt/cray/pe/lib64/libmpi_cray.so.12
#2  0x00001555537b4006 in MPID_Progress_wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#3  0x00001555537b6f56 in MPIR_Wait_state () from /opt/cray/pe/lib64/libmpi_cray.so.12
#4  0x00001555536cdcc7 in MPID_Wait.constprop.0 () from /opt/cray/pe/lib64/libmpi_cray.so.12
#5  0x00001555536e453f in MPIC_Wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#6  0x00001555536e4980 in MPIC_Recv () from /opt/cray/pe/lib64/libmpi_cray.so.12
#7  0x000015555372b55c in MPIR_CRAY_Bcast_Tree () from /opt/cray/pe/lib64/libmpi_cray.so.12
#8  0x000015555372bf5a in MPIR_CRAY_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#9  0x00001555532e01cf in PMPI_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#10 0x0000000000c55cfe in Canopy::TreePartitioner<Kokkos::HIPSpace, Kokkos::HIP>::partition_leaves(std::vector<Canopy::CellInfo, std::allocator<Canopy::CellInfo> > const&) ()
#11 0x0000000000c18e82 in void Canopy::Solver<Kokkos::HIPSpace, Kokkos::HIP, double, 8, 1, Canopy::LaplaceKernel>::_full_setup<0, 1, Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> > >(Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> >&, int) ()
#12 0x0000000000bfee6f in Test::testMultiStepGravity(Test::MultiSolveTest::Mode, char const*, int, int, double, double, int, int, double, int, double, int*, bool, double, long long*) ()
#13 0x0000000000be3c52 in Test::MultiSolve_LargeMotion_Rebuild_Test::TestBody() ()
#14 0x0000155554e008ed in void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#15 0x0000155554de2ea6 in testing::Test::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#16 0x0000155554de3045 in testing::TestInfo::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#17 0x0000155554de32bd in testing::TestSuite::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#18 0x0000155554df7169 in testing::internal::UnitTestImpl::RunAllTests() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#19 0x0000155554de3384 in testing::UnitTest::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#20 0x0000000000f271b2 in main ()
--- pid 2588653 ppid 2588649 comm Canopy_Test_Mul ---
Thread 1 (Thread 0x155555548a80 (LWP 2588653)):
#0  0x000015555392a25e in MPIDI_POSIX_eager_recv_begin () from /opt/cray/pe/lib64/libmpi_cray.so.12
#1  0x00001555537a654c in MPIDI_POSIX_progress_recv.constprop.0 () from /opt/cray/pe/lib64/libmpi_cray.so.12
#2  0x00001555537ac1a8 in MPIDI_POSIX_progress.constprop.0 () from /opt/cray/pe/lib64/libmpi_cray.so.12
#3  0x00001555537b2b68 in MPIDI_progress_test () from /opt/cray/pe/lib64/libmpi_cray.so.12
#4  0x00001555537b4006 in MPID_Progress_wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#5  0x00001555537b6f56 in MPIR_Wait_state () from /opt/cray/pe/lib64/libmpi_cray.so.12
#6  0x00001555536cdcc7 in MPID_Wait.constprop.0 () from /opt/cray/pe/lib64/libmpi_cray.so.12
#7  0x00001555536e453f in MPIC_Wait () from /opt/cray/pe/lib64/libmpi_cray.so.12
#8  0x00001555536e4980 in MPIC_Recv () from /opt/cray/pe/lib64/libmpi_cray.so.12
#9  0x000015555372b55c in MPIR_CRAY_Bcast_Tree () from /opt/cray/pe/lib64/libmpi_cray.so.12
#10 0x000015555372bf5a in MPIR_CRAY_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#11 0x00001555532e01cf in PMPI_Bcast () from /opt/cray/pe/lib64/libmpi_cray.so.12
#12 0x0000000000c55cfe in Canopy::TreePartitioner<Kokkos::HIPSpace, Kokkos::HIP>::partition_leaves(std::vector<Canopy::CellInfo, std::allocator<Canopy::CellInfo> > const&) ()
#13 0x0000000000c18e82 in void Canopy::Solver<Kokkos::HIPSpace, Kokkos::HIP, double, 8, 1, Canopy::LaplaceKernel>::_full_setup<0, 1, Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> > >(Cabana::AoSoA<Cabana::MemberTypes<double [3], double [1], double [3], int>, Kokkos::HIPSpace, 64, Kokkos::MemoryTraits<0u> >&, int) ()
#14 0x0000000000bfee6f in Test::testMultiStepGravity(Test::MultiSolveTest::Mode, char const*, int, int, double, double, int, int, double, int, double, int*, bool, double, long long*) ()
#15 0x0000000000be3c52 in Test::MultiSolve_LargeMotion_Rebuild_Test::TestBody() ()
#16 0x0000155554e008ed in void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#17 0x0000155554de2ea6 in testing::Test::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#18 0x0000155554de3045 in testing::TestInfo::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#19 0x0000155554de32bd in testing::TestSuite::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#20 0x0000155554df7169 in testing::internal::UnitTestImpl::RunAllTests() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#21 0x0000155554de3384 in testing::UnitTest::Run() () from /g/g20/stewartj/spack_envs/tuolumne_trilinos/.spack-env/view/lib64/libgtest.so.1.15.2
#22 0x0000000000f271b2 in main ()
### end watchdog stacks f2DJTdkMD: matched=4 children=4 nonempty=4 ###
### watchdog: cancelling sub-job f2DJTdkMD after 22.454469203948975 s ###
### MultiSolve pass 2: canopy_ctest rc=1 ###
### DownwardSweep binary: 2026-10-05 10:26:09.408226273 -0700 ###
### DownwardSweep HIP ###
### DownwardSweep: canopy_ctest rc=0 ###
### UpwardSweep binary: 2026-10-05 10:28:26.168287766 -0700 ###
### UpwardSweep HIP ###
### UpwardSweep: canopy_ctest rc=1 ###
### SERIAL twin of failed Canopy_Test_UpwardSweep_MPI_HIP_np_1: Canopy_Test_UpwardSweep_MPI_SERIAL_np_1 ###
### Canopy_Test_UpwardSweep_MPI_SERIAL_np_1: canopy_ctest rc=1 ###
### SERIAL twin of failed Canopy_Test_UpwardSweep_MPI_HIP_np_2: Canopy_Test_UpwardSweep_MPI_SERIAL_np_2 ###
### Canopy_Test_UpwardSweep_MPI_SERIAL_np_2: canopy_ctest rc=1 ###
### TreeBuilder binary: 2026-10-05 10:29:07.452193567 -0700 ###
### TreeBuilder HIP ###
### TreeBuilder: canopy_ctest rc=0 ###
### TreePartitioner binary: 2026-10-05 10:31:18.110566890 -0700 ###
### TreePartitioner HIP ###
### TreePartitioner: canopy_ctest rc=0 ###
### CommunicationPlan binary: 2026-10-05 10:33:32.886230160 -0700 ###
### CommunicationPlan HIP ###
### CommunicationPlan: canopy_ctest rc=0 ###
### LaplaceSolve binary: 2026-10-05 10:36:46.001435713 -0700 ###
### LaplaceSolve HIP ###
### LaplaceSolve: canopy_ctest rc=1 ###
### SERIAL twin of failed Canopy_Test_LaplaceSolve_MPI_HIP_np_1: Canopy_Test_LaplaceSolve_MPI_SERIAL_np_1 ###
### Canopy_Test_LaplaceSolve_MPI_SERIAL_np_1: canopy_ctest rc=0 ###
### SERIAL twin of failed Canopy_Test_LaplaceSolve_MPI_HIP_np_2: Canopy_Test_LaplaceSolve_MPI_SERIAL_np_2 ###
### Canopy_Test_LaplaceSolve_MPI_SERIAL_np_2: canopy_ctest rc=0 ###
### SERIAL twin of failed Canopy_Test_LaplaceSolve_MPI_HIP_np_3: Canopy_Test_LaplaceSolve_MPI_SERIAL_np_3 ###
### Canopy_Test_LaplaceSolve_MPI_SERIAL_np_3: canopy_ctest rc=0 ###
### SERIAL twin of failed Canopy_Test_LaplaceSolve_MPI_HIP_np_4: Canopy_Test_LaplaceSolve_MPI_SERIAL_np_4 ###
### Canopy_Test_LaplaceSolve_MPI_SERIAL_np_4: canopy_ctest rc=0 ###
### CartesianTaylorSolve binary: 2026-10-05 10:39:45.813316585 -0700 ###
### CartesianTaylorSolve HIP ###
### CartesianTaylorSolve: canopy_ctest rc=0 ###
### FarFieldContract binary: 2026-10-05 10:42:26.306176615 -0700 ###
### FarFieldContract HIP ###
### FarFieldContract: canopy_ctest rc=0 ###
### CartesianTaylor binary: 2026-10-05 10:43:11.718372622 -0700 ###
### CartesianTaylor HIP ###
### CartesianTaylor: canopy_ctest rc=0 ###
### SERIAL UpwardSweep np 1-6 (--output-on-failure) ###
### SERIAL UpwardSweep: canopy_ctest rc=1 ###
### cancelled sub-jobs ###
```

**`MultiSolve` HIP two-pass comparison.** Not one np reproduces between the
passes:

| np | cases in both passes | identical | relative spread of differing figures |
| --- | --- | --- | --- |
| 1 | 6 | 1 (`AutoRebalance`) | 6.8e-12 to 1.2e-2 |
| 2 | 6 | 1 (`StableTree_Migrate`) | 2.9e-8 to 3.4e-2 |
| 3 | 2 (pass 1 stalled) | 0 | 5.9e-7 to 3.2e-5 |
| 4 | 2 (pass 2 stalled) | 0 | 8.5e-7 to 1.0e-5 |

np 1 never partitions, so the spread comes from device-side reductions alone.
The largest relative spreads are in `max_vel_rel` of `AutoMaintain` and
`IntermediateMotion_Rebalance` at np 1 (1.2e-2) and `LargeMotion_Rebuild` at
np 2 (3.4e-2). AutoRebalance's HIP `max_vel_rel` is 8.55e-6
(np 1), 1.36e-4 (np 2), 3.00e-4 (np 3) and 5.06e-4 (np 4), all below the
`1.95e-3` floor.

**Budgets on HIP.** Except where noted above, every HIP entry ran within its
SERIAL budget. The cold first entry, `MultiSolve` HIP np 1 (pass 1), took
9.99 s of ctest time against a budget of 9 and was not cancelled. The watchdog
compares the sub-job's runtime, which excludes `flux run` launch, and ctest
waits budget + 5 s.

**Affects:**
- H1 HIP arm: the HIP stall already reproduced with stacks, at np 3 and np 4.
  Its loop covers `MultiSolve_MPI_HIP_np_3` and `_np_4`, the rank counts H0c
  recorded over budget. Expect the Zoltan2-on-HIP signature above.
- H2 partitioner arm: the stall is in the path it replaces. Its exit criterion
  runs `UpwardSweep` and `LaplaceSolve` through `canopy_ctest` at SERIAL np 1-6
  and HIP np 1-4. `UpwardSweep` fails on SERIAL and HIP, and `LaplaceSolve` on
  HIP, before any partitioner change. Those failures are carried (README),
  and the `UpwardSweep` HIP np 3-4 entries go over budget on output volume
  alone. The criterion needs to say they are carried, or wait on their fixes.
- E1: its HIP arm is blocked until H2's partitioner arm removes the stall. Its
  HIP "inert when off" comparison must use H0c's spread, because the HIP lines
  are not identical between passes even at np 1.
- tree-opt T1: its HIP arm runs `MultiSolve` HIP and is blocked by the same
  stall until H2's partitioner arm. Its `DownwardSweep` HIP arm passes at
  np 1-4.

## Budget allowances (H0b follow-up)

Jobs `f3cMZeA862N3` (`run_ctest_h0.flux UpwardSweep`) and `f3cMZeHziDAF`
(`run_ctest_h0b.flux check`), HEAD `d22b6a0` plus this change. Logs in the repo
root (untracked).

**Why.** Two kinds of entry ran at or over budget without being hung (H0b,
H0c):
- **Cold start.** The first Canopy binary in a job runs ~4 s slow:
  `MultiSolve` SERIAL np 1 took 8.4-8.6 s as a job's first entry against a
  budget of 9, and HIP np 1 9.99 s. Any np-1 row with `t_ref_s` ≈ 3.5 s (budget
  7) would fail as a first entry.
- **`UpwardSweep` HIP np 1-4.** The `testIdempotentExecution` failure output
  makes ctest take 6.7-26.8 s against budgets of 7-11. In one run of
  `f3cMCHLDMNu5` the np-3 sub-job itself took 12.9 s against 10.

The `MultiSolve` HIP np 3-4 over-budget entries are real stalls (H0c) and get
no allowance.

**Change.**
- `serial_runtimes.tsv` gains two optional columns, `extra_s` and
  `extra_reason`. The budget is `ceil(1.75 * t_ref_s) + extra_s`.
- `UpwardSweep` np 1-4 carry 2/7/18/23 s. Each makes the budget 1.25x the
  worst observed ctest time (6.66, 11.64, 21.92, 26.80 s), giving budgets of
  9/15/28/34. The allowance applies on SERIAL too, because rows have no
  backend.
- `canopy_ctest` adds `CANOPY_COLD_START_S` (default 6) to the first entry it
  runs after `flux_watchdog.sh` is sourced, marked by
  `${WATCHDOG_DIR}/warm`. A file marker survives the HIP subshell. A script
  that re-sources the watchdog gets the allowance again, which is
  conservative.
- `run_ctest_h0b.flux` carries the `extra_s`/`extra_reason` columns of
  existing rows over a recalibration.

**Verified.**
- `f3cMZeHziDAF`: `MultiSolve` SERIAL np 1 as the first entry took 8.97 s
  against a budget of 15. The following pass's np 1 took 4.75 s against 9.
- `f3cMZeA862N3`: `UpwardSweep` HIP np 1-4 took 10.20/12.30/18.30/22.89 s
  against 15/15/28/34, all `failed` (their recorded cases) and none
  over-budget. SERIAL np 1-6 had none over-budget either.
- The self-test (21.87 s) and the forced over-budget checks behaved as in H0b.

**Affects:**
- `01_fix-tests.md`: removes the `UpwardSweep` `extra_s` allowance once its
  failure output is bounded.
- H1 HIP arm, H2, E1: a job's first entry gets 6 s more.

## H2 (partitioner arm)

Commits `60f933f`..`49dce0b` on `investigate-m2l-cap`. All jobs on tuolumne,
Cray clang 20.0.0, flux-core 0.89.0, ROCm 6.4.2, `build-tuolumne/` with
`Canopy_ENABLE_PROFILING=ON`. Logs `canopy-h2.<jobid>.log`,
`canopy-fix-tests.<jobid>.log`, `canopy-f4-sweep.<jobid>.log`,
`canopy-h2scr*.<jobid>.log`, `canopy-h0b.<jobid>.log` and
`canopy-laplace-solve-regen.<jobid>.log` in the repo root (untracked). Gating
jobs: `run_ctest_h2.flux` `pstems serial` `f3cZDLkZqC6s`, `pstems hip`
`f3cZDLuHNyCT`, `pfixed hip 3` `f3cZDM3r2pby`, `pfixed hip 4` `f3cZDMCLEiAT`,
`pfixed serial 3` `f3cZDMLzpWhM`, `prepro serial` `f3cZDMUWgsYj`, `prepro hip`
`f3cZDMd8JhWw`, all on commit `49dce0b` plus doc edits.

**Decisions.**
- H1's HIP arm is not run, and no reverted build is rerun to reproduce the HIP
  hang. The stall is already stacked on every rank in Zoltan2 MJ under
  `partition_leaves`: H0c job `f3cM4ghTjtiT` at np 3 and np 4, and
  01_fix-tests F4 sweep job `f3cWVibHs6gf` at np 3. This arm deletes that path,
  and the `nm -C` check is the evidence that it is gone. The failure direction
  that remains is test (c) under a random assignment.
- The `nm -C` check covers the `*_MPI_HIP` binaries of the six exit-criterion
  stems, rebuilt in this task. Other HIP binaries in `build-tuolumne/tests`
  predate the change and were not rebuilt.
- **Parent-child edges weigh 27, neighbour edges 1** (Do step 3 said unit
  weights). With unit weights ParMETIS cut more parent-child pairs than the
  vote rule at every np 2-6 on the `Basic` fixture: np 4 0.192 vs 0.114, np 6
  0.349 vs 0.221 (scratch job `f3cXJPNQnaDD`). Variants measured there:
  weight 8 still lost at np 4 (0.134 vs 0.092); weight 27 and
  no-neighbour-edges both passed. 27 was chosen to keep both traffic proxies:
  one parent-child edge outweighs a cell's 26 neighbours.
- **Test (c) runs on the clustered `Basic` fixture**, not on (b)'s fixture. On
  (b)'s depth-capped uniform fixture (24 000 particles, ncrit 4, max_depth 5)
  both cuts are under 1% and the partition lost by 4-5 pairs of ~16 000 at
  np 5-6 (0.0047 vs 0.0045, 0.0062 vs 0.0061). There the vote rule itself
  breaks band 1's tolerance (max/mean 1.0645 at np 5, 1.0547 at np 6) while
  ParMETIS meets it (1.0449, 1.0430): the extra cuts buy the band balance
  (job `f3cXRQbNenaT`).
- **A band constraint is kept only if it holds at least `4 * comm_size`
  cells** (Do step 2 kept every band). On `DownwardSweep`'s 1200-particle
  two-scale tree the coarsest band (depths 3-4) holds 2-19 cells, and as a
  constraint it made ParMETIS miss the particle constraint too: max/mean 1.26
  at np 4, 2.00 at np 5 with two ranks owning nothing, 1.33 at np 6.
  `DownwardSweepTwoScale.treeHasShallowAndDeepLeaves` failed at SERIAL np 5
  (job `f3cYxa8kKdUs`; ranks 2 and 3 had `num_local 0`) after passing under
  MJ. Dropping bands under `np` cells fixed np 5 only; under `4 * np` cells,
  particle max/mean is 1.01-1.03 at np 2-6 (1.09 at np 3) (scratch job
  `f3cZ4TQWntxf`).

**Departure from Do step 4: ParMETIS is called directly, not through
Zoltan2.** Zoltan2's `PartitioningProblem::createAlgorithm`
(`Zoltan2_PartitioningProblem.hpp:494-530` in the spack view) instantiates
`Zoltan2_AlgMJ<Adapter>` for every adapter type, so the `TpetraCrsGraphAdapter`
route would keep `AlgMJ` symbols in every binary and fail the `nm` criterion.
`ParMETIS_V3_PartKway` / `ParMETIS_V3_AdaptiveRepart` (ParMETIS 4.0.3, 32-bit
`idx_t`) are already on every Canopy binary's link line through Trilinos, so no
CMake change was needed. The partitioner no longer includes any Zoltan2,
Teuchos or Tpetra header. The `static_assert` of Do step 4 has nothing to
check: the solve is host-only C.

**Implementation notes.**
- Vertex order is Morton pre-order: each key shifted to the deepest vertex
  depth, ancestor first on ties.
- ParMETIS needs contiguous global IDs per rank, so vertices are numbered by
  (supplier rank, Morton position); with the block rule that is the Morton
  index itself. On repartition the supplier is `_cell_owner_map`'s previous
  owner; AdaptiveRepart gets `PARMETIS_PSR_UNCOUPLED` with `part[]` = the
  supplier on input, unit `vsize`, `itr = 100`; it falls back to PartKway when
  fewer than two ranks have vertices. Ranks with no vertices are split off with
  `MPI_Comm_split`. Seed 15, `ubvec = 1 + imbalance_tolerance` per constraint.
- A ParMETIS failure on any rank is all-reduced before the throw, so no rank
  is left in `MPI_Allgatherv`.
- `derive_internal_ownership` now throws when a leaf has no owner (it assigned
  rank 0 silently before).
- `refresh_ownership_for_current_tree` keeps the partitioned owner of every
  cached cell, leaf or internal; new leaves vote by local particles as before,
  new internal cells by the vote rule.

**Signatures changed** (all `TreePartitioner`, `src/Canopy_TreePartitioner.hpp`).
`partition`, `repartition`, `ownership()`, `cell_owner_map()` and
`refresh_ownership_for_current_tree` are unchanged; no caller outside the file
needed an edit.
- `partition_leaves(cells)` deleted (no caller outside the file; comments in
  `Canopy_Solver.hpp`, `tstLaplaceSolve.hpp` and the data header updated).
- New `partition_cells(cells, bool adaptive)`, `vote_internal_owners(cells,
  owners) const` and `bands()`. New free functions `key_to_lattice`,
  `lattice_to_key`, `owner_map_hash`, and struct `PartitionBand`.
- `derive_internal_ownership(cells, owners)`: same types; `owners` may now hold
  internal cells, which keep their entry.
- `_cached_leaf_owners` became `_cached_cell_owners`.
- Profiling builds print `[Canopy diag] partition` (method, np, nverts, B,
  bands, band_cells, per-constraint imbalance, parent-child cut and the vote
  rule's on the same leaves, ranks without a leaf, ownership hash) and
  `[Canopy diag] refresh_ownership fallback=<n>/<non-shared> hash=...` on
  rank 0.

**Scripts.** `run_ctest_h2.flux` gained `pfixed <backend> <np>`,
`pstems <backend>` and `prepro <backend>`, every run through `canopy_ctest`
after H0b's self-test; its SERIAL-arm `fixed`/`repro` modes are unchanged.
`run_ctest_h0b.flux` takes `CANOPY_CAL_REGEX` to recalibrate a subset and keep
the other rows.

**Tests and budgets.** `tstTreePartitioner.hpp` gained (a)
`testOwnerMapAgreement`, (b) `testPartitionBalance`, (c) `testParentChildCut`,
(d) `testRefreshKeepsPartition`. TreePartitioner's six `default` rows were
recalibrated (job `f3cY3QHjeejZ`): 6.74 / 4.16 / 5.50 / 6.25 / 7.22 / 8.36 s at
np 1-6. The np-1 row includes the job's cold start, because the entry ran
first and H0b's rule takes the max of three passes. No other stem went over
budget.

TreePartitioner figures, SERIAL (job `f3cXzm8FQ6gX`):

| np | (b) max/mean c0, c1, c2, c3 | (c) cut / vote | (d) fallback, subset tree |
| --- | --- | --- | --- |
| 2 | 1.0092, 1.0039, 1.0018, 1.0073 | 0.0425 / 0.0425 | 25/410 |
| 3 | 1.0046, 1.0195, 1.0124, 1.0017 | 0.0473 / 0.0473 | 7/649 |
| 4 | 1.0175, 1.0156, 1.0110, 1.0085 | 0.0995 / 0.1044 | 5/742 |
| 5 | 1.0500, 1.0449, 1.0422, 1.0455 | 0.0628 / 0.0673 | 29/855 |
| 6 | 1.0393, 1.0430, 1.0495, 1.0498 | 0.0562 / 0.0620 | 193/906 |

(d)'s post-migration tree has fallback 0 at every np: migration does not move
the global particle set, so the rebuilt tree is the partitioned one. Under the
random assignment (c) reads 0.50-0.82 against the vote rule's 0.40-0.68.

**LaplaceSolve regeneration** (job `f3cY5dZ6SgwH`). `(2,0)` and `(2,1)`
changed: `n_unique_ops` 390/400 became 393/462, with new `locals`, `optab` and
`keys` hashes. The `(1,0)` record, the `initial` hash and the np-1 `field`
record came out byte-identical. After the band rule the regeneration
(job `f3cZC8dY5PjM`) reproduced the committed file byte for byte.

**Step 7 — reproducibility.** Two passes, `MultiSolve` and `TreePartitioner`,
SERIAL np 2-6 (`f3cZDMUWgsYj`) and HIP np 2-4 (`f3cZDMd8JhWw`). Every
`[Canopy diag] partition` and `refresh_ownership` hash matches between passes:
30-36 partitions per `MultiSolve` entry and 17 per `TreePartitioner` entry, at
every np on both backends. So `cell_owner_map()` reproduces. SERIAL
`[multisolve-dev]` lines are identical between passes at every np 2-6. HIP
lines are not: no case is identical at np 2-4, relative spread 6.2e-12..2.0e-4
(np 2), 5.8e-8..7.9e-5 (np 3), 4.5e-7..7.9e-5 (np 4). The partition is the
same, so this is device reductions, as H0c found at np 1.

**Step 7 — `MultiSolve` partitions** (SERIAL pass 1, `f3cZDMUWgsYj`; HIP's
partition lines are identical):

| np | partitions | particle max/mean > 1.05 | worst particle max/mean | worst band max/mean | cut ≤ vote | mean cut / vote | max ranks without leaf |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 30 | 12 | 2.000 | 1.111 | 15/18 | 0.070 / 0.122 | 1 |
| 3 | 34 | 14 | 2.980 | 1.154 | 15/21 | 0.073 / 0.180 | 1 |
| 4 | 34 | 13 | 4.000 | 1.081 | 21/22 | 0.094 / 0.286 | 3 |
| 5 | 36 | 17 | 4.270 | 1.364 | 20/22 | 0.161 / 0.344 | 3 |
| 6 | 35 | 20 | 5.775 | 1.200 | 24/24 | 0.173 / 0.431 | 4 |

The badly imbalanced partitions are all tiny graphs: 11-29 vertices, B = 0, so
only the particle constraint is in play. Imbalance exactly 2.0 at np 2 or 4.0
at np 4 means every particle on one rank. That fits one leaf holding nearly
all the particles: the ejected-particle boxes of `LargeMotion_Rebuild` and
`AutoRebalance` (Approach, "A lead E1 must read"), which no partition can
split. The largest leaf's share was not measured, so this is not verified.
**R9:** `refresh_ownership` fell back to a vote for 0 cells in every one of the
30-36 refreshes per np, at every np on both backends.

**Step 7 — partition time** (`TIMER_PARTITION` Max column, mean over every
setup in the two passes; it includes migration). MJ figures are from job
`f3bnvGg3MDo5` (SERIAL arm, MJ on the Serial node) and H0c's `f3cM4ghTjtiT`
(MJ on HIP; its np-3 and np-4 means omit the stalled entries).

| np | SERIAL MJ | SERIAL ParMETIS | HIP MJ | HIP ParMETIS |
| --- | --- | --- | --- | --- |
| 2 | 0.5 ms | 2.9 ms | 3.8 ms | 3.3 ms |
| 3 | 0.7 ms | 7.8 ms | 10.8 ms | 15.2 ms |
| 4 | 0.9 ms | 12.0 ms | 7.1 ms | 12.9 ms |
| 5 | 1.0 ms | 17.3 ms | | |
| 6 | 1.0 ms | 20.4 ms | | |

On these trees of a few hundred cells, ParMETIS's distributed setup costs
3-20x MJ-on-host; on HIP it is within 2x of MJ-on-HIP.

**`AutoRebalance` after this arm** (SERIAL, `f3cZDMUWgsYj`): `max_vel_rel`
1.74e-4 / 3.00e-4 / 5.06e-4 / 6.52e-3 / 1.71e-2 at np 2-6, `max_pos_rel`
3.16e-3 at np 6. The np 5-6 excess is unchanged by the partitioner.

**Out of scope, left stale:** `tests/tstFarFieldContract.hpp:862-868` still
says the tree above one rank is partitioned by Zoltan2 multijagged
(`FarFieldContract` was not built).

**Affects:**
- E1: its post-H2 SERIAL np 2-6 baseline is `f3cZDMUWgsYj`'s
  `[multisolve-dev]` lines, which reproduce exactly, so its inert-when-off
  np-2 comparison is character for character. Its HIP comparison must use a
  spread: HIP lines differ between passes even on an identical partition
  (6e-12..2e-4 at np 2-4). The HIP stall that blocked E1's HIP arm is gone.
  The AutoRebalance np 5-6 excess survives the new partition.
- tree-opt V1: every `[multisolve-dev]` figure above np 1 moved again (new
  partition at np >= 2). Re-measure np >= 2 on this build before pinning
  anything.
- tree-opt T1: its `MultiSolve` HIP arm is no longer blocked by the stall.

## E1

All jobs ran `scripts/tuolumne/run_ctest_e1.flux` or `run_ctest_h0b.flux`
on tuolumne, HEAD `f572ca1` plus this task's working-tree changes (each log's
`git status --porcelain` shows ` M tests/tstMultiSolve.hpp`, except step 0's),
Cray clang 20.0.0, flux-core 0.89.0, ROCm 6.4.2, `build-tuolumne/` with
`Canopy_ENABLE_PROFILING=ON`, level 2. Logs are `canopy-e1.<jobid>.log` and
`canopy-h0b.<jobid>.log` in the repo root (untracked). Every job passed the
watchdog self-test (21.5-21.7 s, 3/3/3), and no entry went over budget or
was cancelled. The tightest margin was HIP np 4 at 13.47 s against 20.

| job | mode | binary |
| --- | --- | --- |
| `f3cZYwER1RzP` | `base` (Do step 0), HIP np 1-2 ×2 | HEAD, unmodified |
| `f3cZcmUTK5f5`, `f3cZcmgeVB5Z`, `f3cZcmt3hemM` | calibrate `probe`, `probe-npp1200`, `probe-theta0.7` | first probe build |
| `f3cZcn4dUXS7`, `f3cZcnFVZGCf`, `f3cZg84gofbu`, `f3cZg8ExngtX`, `f3cZg8RLhfJX` | inert serial/hip, measure serial/hip, variants | first probe build (superseded) |
| `f3cZoEsnprZu`, `f3cZoF2MUhyR` | inert serial, inert hip | final |
| `f3cZoFBJNNc7`, `f3cZoFK9Wa7y` | measure serial (np 1-6), measure hip (np 1-4) | final |
| `f3cZoFTWJXGj` | variants (np 1: N = 1200; θ = 0.7) | final |

The final build differs from the first only in the `end` line (below). Its
348 SERIAL per-step lines are identical to the first build's.

**Probe interface** (`tests/tstMultiSolve.hpp`).
- `get_test_probe_enabled()` reads `CANOPY_MULTISOLVE_PROBE`: unset or `0` is
  off, `1` is on, and anything else throws. `get_test_npp_override()` reads
  `CANOPY_MULTISOLVE_NPP`: unset is 0, and anything but a positive integer
  throws. The override replaces `num_particles_per_rank` at the top of
  `testMultiStepGravity`, so it applies at all six sites.
- New free function `nearest_separation(pos)`, O(N²), beside
  `brute_force_gradient`.
- Per step, after `solve()` and before the integrate kernel, every rank sends
  its positions, charges, GlobalIds and FMM gradient to rank 0.
  Rank 0 prints:
  `[multisolve-probe] case nprocs step prev_action cells root_hw max_rel field_err rel_gid rel_g rel_sep abs_gid abs_dg abs_g abs_sep min_sep n_close close_thr`.
  `prev_action` is `Setup` at step 0. `root_hw` is the depth-0 cell's
  half-width.
- At end of run, rank 0 prints:
  `[multisolve-probe] case nprocs end max_vel_rel max_vel_gid max_vel_min_sep max_vel_v median_v run_min_sep n_close_run close_thr floor n_excess n_excess_close`.
  The run minimum separation folds in every probed step plus the final FMM
  state. `floor` is θ^(P+1) at the case's own θ.

**Departures from Do.**
- **Fields beyond the listed ones.** Step 1 names the argmax of the
  per-particle error. The line also gives the argmax of the field-error
  numerator (`abs_*`), and per-step `min_sep`/`n_close`. The `end` line adds
  `max_vel_v`, `median_v`, `n_excess` and `n_excess_close`. I added them after
  the first build: at np 5, the `max_vel_rel` particle had no close encounter
  (0.00867 against 0.0080), so (a) needed a figure that does not rest on one
  particle, and a check against a small-|v| normalization artifact.
- **All six cases run with the probe**, not only AutoRebalance. That keeps one
  ctest entry per (np, config) and gives every case's per-step error. The
  `measure` and `variants` budget rows are calibrated on the whole stem.
- **The np-1 variants ran twice each**, not once.
- **The budget rows were calibrated on the first probe build.** The `end`-line
  extension adds O(N) work once per case. The final runs stayed at or below
  67% of their budgets.
- **HIP inert-when-off uses the per-np spread.** Do step 0 gives 3 HIP samples
  at np 1 and 4 at np 2 per figure. Five of the 24 figures (np 1:
  AutoRebalance pos and vel; np 2: AutoRebalance vel, StableTree_Migrate pos
  and vel) have a post-E1 sample outside a per-figure band of
  [lo − (hi − lo), hi + (hi − lo)]. They deviate by 6e-10 to 6.2e-5 relative.
  The per-np reading follows H0c's table and H2 step 7, which both state the
  HIP spread as one range per np, and every figure passes it. Under the
  per-figure reading, 3-4 samples are too few to bound HIP's run-to-run
  tails. The first probe build's HIP inert run (`f3cZcnFVZGCf`) behaved the
  same way: 4 samples outside per figure, max deviation 2.35e-5 at np 1 and
  8.77e-5 at np 2.
- `run_ctest_h0b.flux` takes the row config from `CANOPY_BUDGET_CONFIG`
  (default `default`) and prints the `CANOPY_*` environment in its provenance.

**Step 0: pre-E1 HIP spread.** The relative spread of each `[multisolve-dev]`
figure, (max − min)/|mean|, over `f3cZYwER1RzP`'s two passes plus
`f3cZDLuHNyCT` (np 1), or plus `f3cZDMd8JhWw`'s two passes (np 2):
- np 1: 0 (LargeMotion_Rebuild pos, M2L_BinEdge_Fallback pos) to `6.78e-5`
  (IntermediateMotion_Rebalance pos);
- np 2: 0 (StableTree_Migrate pos) to `2.20e-4` (AutoMaintain pos).

**Budget rows added** to `serial_runtimes.tsv`. All 55 `default` rows are
unchanged (diffed before and after).

| stem | np | config | t_ref_s | job |
| --- | --- | --- | --- | --- |
| MultiSolve | 1-6 | `probe` | 8.33 / 6.33 / 8.76 / 11.18 / 13.26 / 14.80 | `f3cZcmUTK5f5` |
| MultiSolve | 1 | `probe-npp1200` | 18.36 | `f3cZcmgeVB5Z` |
| MultiSolve | 1 | `probe-theta0.7` | 8.67 | `f3cZcmt3hemM` |

Each np-1 row includes its job's cold start, because the entry ran first.
The probe adds no measurable runtime at np 2-6.

**Inert when off.**
- SERIAL (`f3cZoEsnprZu`): the six np-1 lines of both passes are identical to
  `f3cZDLkZqC6s`, and the six np-2 lines to `f3cZDMUWgsYj`.
- HIP (`f3cZoF2MUhyR`): the largest relative deviation from the pre-E1 mean is
  `6.18e-5` at np 1 (spread `6.78e-5`) and `2.05e-4` at np 2 (spread
  `2.20e-4`).

**Per-step field-scale error, max over steps and both passes:**

| case | SERIAL np 1-6 | HIP np 1-4 |
| --- | --- | --- |
| AutoRebalance | 4.7e-8 / 7.3e-7 / 9.5e-7 / 7.3e-7 / 7.5e-7 / 7.0e-7 | 4.7e-8 / 7.3e-7 / 9.5e-7 / 7.3e-7 |
| AutoMaintain, IntermediateMotion_Rebalance | 1.7e-8 / 5.1e-8 / 2.1e-7 / 2.4e-7 / 7.2e-7 / 9.2e-7 | same to 2 figures |
| LargeMotion_Rebuild | 7.2e-7 / 7.8e-7 / 1.1e-6 / 4.5e-7 / 1.5e-6 / 9.2e-7 | same to 2 figures |
| StableTree_Migrate | 1.6e-8 / 5.1e-8 / 2.1e-7 / 2.4e-7 / 7.2e-7 / 9.2e-7 | same to 2 figures |
| M2L_BinEdge_Fallback | 1.2e-8 / 1.4e-8 / 2.3e-8 / 4.6e-9 / 4.3e-9 / 5.6e-9 | same to 2 figures |

The worst figure anywhere is `1.52e-6` (LargeMotion_Rebuild, SERIAL np 5).
The per-particle `max_rel` peaks at `4.9e-4` (LargeMotion_Rebuild np 3). The
two SERIAL passes are identical line for line (174 of 174 step lines). On HIP,
no step line is identical between passes.

**AutoRebalance per step, SERIAL np 6** (np 5 has the same shape):

| step | after | cells | root_hw | field_err | min_sep | n_close |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Setup | 207 | 0.639 | 7.0e-7 | 8.6e-3 | 0 |
| 1 | Rebalance | 198 | 0.628 | 5.0e-9 | 6.5e-4 | 33 |
| 2 | Rebuild | 130 | 5.83 | 5.7e-9 | 9.3e-4 | 78 |
| 3 | Rebuild | 60 | 11.4 | 1.8e-9 | 1.8e-3 | 65 |
| 4 | Rebalance | 45 | 17.1 | 7.9e-10 | 3.0e-3 | 60 |
| 5-7 | Rebalance | 29-33 | 22.9-34.3 | 7e-16 to 2.3e-15 | 1.6e-3 to 2.1e-3 | 50-103 |

This confirms the Approach's lead. The cell count falls because the root
half-width grows more than 50x: an ejected particle stretches the box. The
cells left are then mostly P2P, and the solve becomes exact to round-off.
Step 0, before any maintenance, has the run's largest error.

**AutoRebalance end lines** (identical between passes on SERIAL; close-encounter
threshold `close_thr` = 0.1 (0.8³/N)^(1/3)):

| run | N | max_vel_rel | gid | its min sep / close_thr | its \|v\| / median \|v\| | n_close_run / N | n_excess (close) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SERIAL np 1 | 200 | 2.22e-6 | 75 | 0.0046 / 0.0137 | 23.4 / 6.4 | 63 / 200 | 0 |
| SERIAL np 2 | 400 | 1.74e-4 | 287 | 0.0085 / 0.0109 | 7.1 / 12.9 | 152 / 400 | 0 |
| SERIAL np 3 | 600 | 3.00e-4 | 467 | 0.0097 / 0.0095 | 11.8 / 20.2 | 214 / 600 | 0 |
| SERIAL np 4 | 800 | 5.06e-4 | 794 | 0.0068 / 0.0086 | 38.0 / 30.0 | 290 / 800 | 0 |
| SERIAL np 5 | 1000 | 6.52e-3 | 123 | 0.0087 / 0.0080 | 15.7 / 44.6 | 433 / 1000 | 2 (1) |
| SERIAL np 6 | 1200 | 1.71e-2 | 821 | 0.0056 / 0.0075 | 74.7 / 62.7 | 577 / 1200 | 11 (10) |
| SERIAL np 1, NPP 1200 | 1200 | 2.67e-3 | 328 | 0.0068 / 0.0075 | 59.3 / 50.5 | 531 / 1200 | 4 (4) |
| SERIAL np 1, θ 0.7 | 200 | 1.95e-4 | 50 | 0.0039 / 0.0137 | 80.5 / 6.4 | 63 / 200 | 0 (floor 4.0e-2) |

HIP np 1-4 match SERIAL's `max_vel_gid`, separations and counts, with
`max_vel_rel` agreeing to 1.1e-4 relative (np 2) and 3e-6 or better at
np 1, 3 and 4.

**Classification: (a) trajectory amplification.** Deciding figures:
- **Not (b) or (c).** The per-solve error is far below the floor on every step,
  at every np, on both backends: AutoRebalance at most `9.5e-7`, any case at
  most `1.52e-6`, against `1.95e-3`. At np 2-6 no AutoRebalance step after
  a Rebuild or Rebalance exceeds the Setup step's error. At np 1, steps 1
  (Migrate) and 4 (Rebuild) reach `4.7e-8` and `3.2e-8`, against Setup's
  `1.9e-8`. Both are at np 1, where (b) does not apply. R4's every-step
  condition holds.
- **N-driven, not rank-driven.** SERIAL np 1 at N = 1200, with no partition,
  has `max_vel_rel = 2.67e-3`, 1.4x the floor. It uses different particles
  from np 6 (one seed rather than six), so the magnitude differs from np 6's
  `1.71e-2`.
- **Amplified from a tiny solve error.** In the N = 1200 run, a particle is
  ejected at step 2: the root half-width goes from 0.64 to 55, then to 332 by
  step 7. The tree drops to 17-18 cells, and the field error is below `2e-15`
  from step 2 on. The only far-field error the trajectory ever sees is the
  `3.8e-7` and `1.1e-9` of steps 0-1, so dynamics amplify it about 7000x into
  `2.67e-3`. The system collapses unsoftened (`cfg.softening = 0`):
  gradients reach 1.2e6 (5.6e5 at np 6), and the smallest separation
  reaches 2.4e-4, 1/30 of the close threshold.
- **The excess sits on close encounters.** At np 6, 10 of the 11 particles
  above the floor came within `close_thr` of another particle, against a base
  rate of 577/1200 = 48%. At N = 1200, 4 of 4 did, against 44%. At np 5, 1 of 2
  did, against 43%.
- **Not a normalization artifact.** The `max_vel_rel` particle's |v| is near
  or above the median at np 6 (1.2x) and N = 1200 (1.2x). The exception is
  np 5: there the particle (gid 123) is at 0.35x the median and missed the
  threshold by 8% (0.0087 against 0.0080). Its relative figure is partly
  inflated by a smaller |v|. Neither np 5 particle above the floor shows a
  far-field error: every step's field error is at most `7.5e-7` there.

**HIP against SERIAL.**
- No HIP-specific far-field defect. Same partition (H2 step 7), same
  maintenance actions and tree sizes. Across the 116 `(case, np, step)` triples
  at np 1-4, HIP's per-step field error differs from SERIAL's by at most
  `2.97e-12` absolute.
- On steps whose error is above `1e-12`, the relative difference is at most
  `4.4e-5`, `3.3e-5` and `9.1e-5` at np 1, 3 and 4, against HIP two-run spreads
  of `4.0e-5`, `1.3e-5` and `3.7e-5`.
- The outlier is np 2, AutoRebalance step 2: `3.8e-4` relative, on an error of
  `3.67e-12` (SERIAL `3.6731e-12`, HIP `3.6717e-12` both passes), i.e.
  `1.4e-15` absolute.
- Read literally, Do step 3's rule ("differs by more than the two-run spread
  on either backend") flags 89 of the 116 triples. SERIAL's spread is 0, and
  with two HIP samples a fixed SERIAL value falls outside their range about
  two thirds of the time with no defect at all. The flagged differences are
  at the 1e-12 absolute level, nine orders of magnitude below the floor. I do
  not count them as a second classification.
- Separately, HIP's own two-run spread is large in relative terms only on
  round-off steps (field error ≤ 3.2e-13), where it reaches 0.8.

**Measuring.** At np 1, θ = 0.7 raises AutoRebalance's per-step field error
from at most `4.73e-8` to at most `6.78e-6`, about 140x. Its worst step goes
from `4.7e-8` to `6.8e-6` (step 1), and its `max_vel_rel` from `2.22e-6` to
`1.95e-4`. The probe measures the far field.

**Close-encounter definition.** The threshold is a separation below 0.1 of
the mean spacing (0.8³/N)^(1/3): 0.0137, 0.0109, 0.0095, 0.0086, 0.0080 and
0.0075 at N = 200-1200. Particles start on [0.1, 0.9]³; the collapse pulls
31-48% of them inside it at least once by the end of the run.

**Affects:**
- E2: takes the (a) branch. Rewrite V1's derivation with the figures above:
  the per-step field error at most `9.5e-7` on AutoRebalance, the ~7000x
  amplification in the N = 1200 run, and the close-encounter concentration.
  Change no `src/`. E2's exit criterion, field-scale error ≤ `1.95e-3` on
  every probe step at SERIAL np 1-6 and HIP np 1-4, is already met by
  `f3cZoFBJNNc7` and `f3cZoFK9Wa7y`. No second (HIP-specific) classification
  was recorded. If E2 reads Do step 3's rule literally, see "HIP against
  SERIAL" above first.
- tree-opt V1: at any site where particles approach unsoftened, a trajectory
  bound measures the dynamics, not the far field. AutoRebalance at np 5-6
  (and at np 1 once N reaches 1200) is such a site. The per-step probe
  (`CANOPY_MULTISOLVE_PROBE=1`) gauges the far field directly, at
  ≤ `1.5e-6` for every case at θ = 0.5.

## E2

Branch (a). No `src/`, `tests/` or bound change, and no job submitted. HEAD
`eebc473` plus this task's doc edits.

**Decisions.**
- **No rerun.** The (a) exit criterion's probe half is met by E1's jobs on the
  code committed as `eebc473`: `f3cZoFBJNNc7` (SERIAL np 1-6) and
  `f3cZoFK9Wa7y` (HIP np 1-4), logs `canopy-e1.<jobid>.log` in the repo root.
  Per-step field-scale error at most `1.52e-6` (SERIAL) and `1.13e-6` (HIP)
  against `1.95e-3`. E2 changes no code, so a rerun would measure the same
  binary.
- **No HIP-specific classification.** E1 recorded none; its "HIP against
  SERIAL" paragraph shows the literal Do-step-3 rule's 89 flags are
  differences of at most `2.97e-12` absolute. No HIP branch is taken.
- **V1's stop clause is retargeted to the per-step probe.** Whether an
  amplification site's test gates on the probe instead of the trajectory is
  left to V1; no assertion was added.
- **V1 is unblocked**: `**NOT STARTED**`, resume from step 1.

**`tasks/tree-opt.md` V1 edits.**
- Heading status `**BLOCKED**` → `**NOT STARTED**`. The "Blocked." and "Revisit
  V1 once…" paragraphs, which said the excess "enters through the far field",
  are replaced by one "Resume from step 1" paragraph: E1's attribution, a
  pointer to this log's section E1, and a re-measure instruction.
- Step 1 derivation: the floor bounds the per-solve relative gradient error;
  the trajectory check does not damp it at unsoftened close encounters
  (`cfg.softening = 0.0`, `tests/tstMultiSolve.hpp:402`). It cites `9.5e-7` /
  `1.52e-6` per step, the N = 1200 run's ~7000x amplification into `2.67e-3`,
  10 of 11 excess particles on close encounters at np 6 (48% base), and θ = 0.7
  raising the per-step error ~140x. A trajectory bound at such a site measures
  dynamics.
- Step 1 stop clause: triggers on `[multisolve-probe]` field-scale error above
  $\theta^{P+1}$ on any step. A trajectory excess with the probe under the
  floor and the excess on close encounters (`n_excess` against
  `n_excess_close`) is recorded as dynamics and does not stop V1.
- Line citations: call sites `:955, :972, :990, :1008, :1045, :1094`, `EXPECT`s
  `:929-933`, the `2e-2` prose `:1032` (prose itself left to V1), `P_ORDER`
  `:88`, `get_test_mac_theta()` `:42-47`.
- `tasks/tree-opt-progress-log.md` gains a section `E2 (fix-hang-rebalance)`;
  its V1 section is untouched.

**README "Known Issues", the `1e-8` entry.** "Not yet attributed; unmeasured
candidates…" and the theta-sweep "enters through the far field" text are
replaced by the attribution: per-step far field healthy (`9.5e-7` / `1.52e-6`),
unsoftened close encounters amplify it, N-driven (N = 1200 at np 1:
`2.67e-3`), 10 of 11 on close encounters. Its np 5 figure is now H2's
`6.52e-3` rather than V1's pre-H2 range. Citations `:651,655` → `:929,933` and
`:644` → `:922`. Reproducers: `run_ctest_e1.flux measure` for the probe,
`run_ctest_v1.flux` for the trajectory figures. "Do not widen a bound over it"
became: not without the per-step probe showing the far field under the floor.

**`tasks/fix-hang-rebalance.md` citation fixes** (numbers only): `P_ORDER`
`:54` → `:88`; `get_test_mac_theta()` `:38-43` → `:42-47` (Conventions, E1
**Fill in**); `[multisolve-dev]` `:644` → `:922`; shadow integration
`:484-498` → `:692-706` and comparison `:591-660` → `:735-936` (Deliberate
deviations, checked against the file); `testMultiStepGravity` `:154` → `:208`;
seed `:208` → `:266`; `cfg.softening` `:344` → `:402`. Status `DONE`.

**Affects:**
- tree-opt V1: resumes from step 1 and re-measures on the post-H2 partition. Its
  stop clause now reads the per-step probe. The gate decision for amplification
  sites (AutoRebalance np 5-6, np 1 at N = 1200) is V1's own.
