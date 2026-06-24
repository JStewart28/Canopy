# Shared resolver for Canopy build/run scripts. Source this FIRST from any
# batch script, before referencing CANOPY_BUILD_DIR / binaries:
#
#   source "$(dirname "${BASH_SOURCE[0]}")/../lib/canopy_env.sh"
#
# It resolves the repo root and active system, loads the build/run profile
# (committed scripts/<system>/profile.defaults.sh overlaid by a gitignored
# per-checkout profile.local.sh), activates the spack environment, and exposes
# canopy_exe() for locating binaries in either build mode. The profile variables
# and the two build modes are documented in docs/<system>/claude.md and CLAUDE.md
# ("Build & run profile").
#
# Knobs read from the environment (not the profile):
#   CANOPY_SYSTEM             override hostname-based system detection
#   CANOPY_REPO               override the resolved repo root
#   CANOPY_USE_PROD=1         activate the prod env instead of the dev env
#   CANOPY_NO_SPACK_ACTIVATE=1  resolve the profile but skip spack activation
#                             (used for dry-run / echo validation of spack mode)

# --- locate self / repo -----------------------------------------------------
_canopy_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CANOPY_REPO:=$(cd "${_canopy_lib_dir}/../.." && pwd)}"   # scripts/lib -> repo
export CANOPY_REPO

# --- system detection (hostname -> system) ----------------------------------
# Mirrors the table in CLAUDE.md "System detection". Add a case to extend.
if [ -z "${CANOPY_SYSTEM:-}" ]; then
  case "$(hostname)" in
    tuolumne*) CANOPY_SYSTEM=tuolumne ;;
    dane*)     CANOPY_SYSTEM=dane ;;
    *)
      echo "canopy_env: unrecognized hostname '$(hostname)'; set CANOPY_SYSTEM explicitly" >&2
      return 1 2>/dev/null || exit 1
      ;;
  esac
fi
export CANOPY_SYSTEM

_canopy_profile_dir="${CANOPY_REPO}/scripts/${CANOPY_SYSTEM}"

# --- load profile: committed defaults, then gitignored local override -------
if [ -f "${_canopy_profile_dir}/profile.defaults.sh" ]; then
  . "${_canopy_profile_dir}/profile.defaults.sh"
else
  echo "canopy_env: missing ${_canopy_profile_dir}/profile.defaults.sh" >&2
  return 1 2>/dev/null || exit 1
fi
if [ -f "${_canopy_profile_dir}/profile.local.sh" ]; then
  . "${_canopy_profile_dir}/profile.local.sh"   # plain assignments here win
fi

# --- derive bin mode from build mode if the profile left it unset -----------
if [ -z "${CANOPY_BIN_MODE:-}" ]; then
  if [ "${CANOPY_BUILD_MODE:-manual}" = spack ]; then
    CANOPY_BIN_MODE=path
  else
    CANOPY_BIN_MODE=build-dir
  fi
fi
export CANOPY_BUILD_MODE CANOPY_BIN_MODE CANOPY_BUILD_DIR

# --- activate spack environment (dev, or prod when CANOPY_USE_PROD=1) --------
_canopy_env="${CANOPY_SPACK_ENV:-}"
if [ "${CANOPY_USE_PROD:-0}" = 1 ]; then
  if [ -z "${CANOPY_SPACK_PROD_ENV:-}" ]; then
    echo "canopy_env: CANOPY_USE_PROD=1 but CANOPY_SPACK_PROD_ENV is empty (set it in profile.local.sh)" >&2
    return 1 2>/dev/null || exit 1
  fi
  _canopy_env="${CANOPY_SPACK_PROD_ENV}"
fi

if [ "${CANOPY_NO_SPACK_ACTIVATE:-0}" = 1 ]; then
  echo "canopy_env: CANOPY_NO_SPACK_ACTIVATE=1 -> would 'spack env activate ${_canopy_env}' (skipped)" >&2
else
  if [ -n "${CANOPY_SPACK_SETUP:-}" ] && [ -f "${CANOPY_SPACK_SETUP}" ]; then
    . "${CANOPY_SPACK_SETUP}"
  fi
  spack env activate "${_canopy_env}" || {
    echo "canopy_env: 'spack env activate ${_canopy_env}' failed" >&2
    return 1 2>/dev/null || exit 1
  }
fi

# --- binary locator ---------------------------------------------------------
# canopy_exe <relpath-from-build-dir | binary-name>
#   build-dir mode -> $CANOPY_BUILD_DIR/<relpath>   (manual build)
#   path mode      -> the basename, found on PATH at run time (spack install,
#                     where setup_run_environment puts Canopy_Test_* on PATH)
canopy_exe() {
  if [ "${CANOPY_BIN_MODE}" = path ]; then
    basename "$1"
  else
    printf '%s/%s\n' "${CANOPY_BUILD_DIR}" "$1"
  fi
}
