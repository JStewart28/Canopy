# Tuolumne runtime environment — exported into every batch script by the shared
# resolver (scripts/lib/canopy_env.sh sources this file when CANOPY_SYSTEM is
# tuolumne). Kept in ONE place so every script gets the full set and they cannot
# drift apart. These are runtime (launch-time) vars that must reach the
# flux-launched task; flux inherits the batch shell environment, so exporting
# them here suffices. See docs/tuolumne/claude.md for the rationale of each.

# Cray-MPICH GPU-aware comm + HIP/HMM (required for device pointers in MPI and
# for the MI300A APU to reach host allocations). Harmless for CPU/Serial runs.
export MPICH_GPU_SUPPORT_ENABLED=1
export GTL_HSA_VSMSG_CUTOFF_SIZE=4096
export FI_CXI_ATS=0
export HSA_XNACK=1
export MPICH_SMP_SINGLE_COPY_MODE=NONE

# OpenMP placement (also used by the OpenMP backend; harmless for Serial).
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

# Cray static-TLS workaround. Canopy binaries link multiple cray-libsci
# libraries whose static TLS blocks exhaust the default loader surplus at
# startup ("libsci_cray_mp.so.6: cannot allocate memory in static TLS block",
# exit 127). Enlarging the glibc static-TLS surplus lets the loader place them.
# Required for every Canopy binary on Tuolumne (CPU and GPU).
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000
