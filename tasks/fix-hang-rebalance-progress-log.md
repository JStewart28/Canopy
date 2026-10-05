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
- The np 1-2 lines are compared against `canopy-v1.f3bmo4JYikKh.log`, the
  baseline for E1's "Inert when off" criterion.

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
  should agree exactly. Its np 1-2 baseline, `canopy-v1.f3bmo4JYikKh.log`,
  still holds. AutoRebalance's np 5-6 excess survives H2 (table above), so the
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
