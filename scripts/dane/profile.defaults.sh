# Dane build/run profile — committed manual-mode defaults.
#
# Sourced by scripts/lib/canopy_env.sh, then overlaid by a gitignored
# scripts/dane/profile.local.sh (per-checkout override). These defaults
# reproduce the historical hard-coded values so scripts run zero-config in
# manual (by-hand cmake + make) mode. To work in spack-install mode, write a
# profile.local.sh setting CANOPY_BUILD_MODE=spack, CANOPY_BIN_MODE=path, and the
# dev (and optionally prod) env. See CLAUDE.md "Build & run profile".
#
# NOTE: Dane also requires `module load gcc/13.3.1 openmpi/4.1.2` BEFORE the
# spack env is activated (see docs/dane/claude.md §0). The scheduler scripts load
# the modules explicitly before sourcing canopy_env.sh; the profile only carries
# the spack/build-dir variables.
#
# Each var uses := so a value already set in the environment wins over the
# default; profile.local.sh (sourced after this file) uses plain = and overrides
# both.

: "${CANOPY_BUILD_MODE:=manual}"                                # manual | spack
: "${CANOPY_SPACK_SETUP:=/usr/workspace/stewartj/spack/share/spack/setup-env.sh}"
: "${CANOPY_SPACK_ENV:=${HOME}/spack_envs/dane_trilinos}"       # dep env (manual) / dev env (spack)
: "${CANOPY_SPACK_PROD_ENV:=}"                                  # spack mode only, optional
: "${CANOPY_BUILD_DIR:=${CANOPY_REPO}/build-dane}"              # manual mode build tree
: "${CANOPY_BIN_MODE:=build-dir}"                               # build-dir | path
