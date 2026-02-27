#!/bin/bash

# Install Intel oneAPI C++ Essentials for SYCL builds on Windows.
# Usage: Windows.sh <VERSION>
# Example: Windows.sh 2025.3.1

set -euo pipefail

VERSION=${1:?'Usage: Windows.sh <VERSION>  (e.g. Windows.sh 2025.3.1)'}

if [[ ! "${VERSION}" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
  echo "Error: VERSION '${VERSION}' does not match expected X.Y or X.Y.Z format." >&2
  exit 1
fi

# Lookup table of Intel C++ Essentials *online* installer URLs per oneAPI version.
# URLs are obtained from (select Windows / Online Installer):
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=cpp-essentials&cpp-essentials-os=windows&cpp-essentials-win=online
# To add a new version, append:  ["X.Y.Z"]="https://registrationcenter-download.intel.com/..."
declare -A INSTALLER_URLS=(
  ["2025.1.0"]="https://registrationcenter-download.intel.com/akdlm/IRC_NAS/1e635719-29c5-4775-8252-268d2f87d529/intel-cpp-essentials-2025.1.0.570.exe"
  ["2025.1"]="https://registrationcenter-download.intel.com/akdlm/IRC_NAS/1e635719-29c5-4775-8252-268d2f87d529/intel-cpp-essentials-2025.1.0.570.exe"
  ["2025.2.0"]="https://registrationcenter-download.intel.com/akdlm/IRC_NAS/5b271b39-0773-49a3-b78d-c73ec42d1621/intel-cpp-essentials-2025.2.0.533.exe"
  ["2025.2"]="https://registrationcenter-download.intel.com/akdlm/IRC_NAS/5b271b39-0773-49a3-b78d-c73ec42d1621/intel-cpp-essentials-2025.2.0.533.exe"
  ["2025.3.1"]="https://registrationcenter-download.intel.com/akdlm/IRC_NAS/c61634af-e4dd-4a14-8341-0b35a9ebc22e/intel-cpp-essentials-2025.3.1.25.exe"
  ["2025.3"]="https://registrationcenter-download.intel.com/akdlm/IRC_NAS/c61634af-e4dd-4a14-8341-0b35a9ebc22e/intel-cpp-essentials-2025.3.1.25.exe"
)

INSTALLER_URL="${INSTALLER_URLS[${VERSION}]:-}"
if [[ -z "${INSTALLER_URL}" || "${INSTALLER_URL}" == "FILL_IN" ]]; then
  echo "Error: No installer URL found for oneAPI version '${VERSION}'." >&2
  echo "Add it to the INSTALLER_URLS table in $(basename "${BASH_SOURCE[0]}")." >&2
  echo "Download page: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=cpp-essentials&cpp-essentials-os=windows&cpp-essentials-win=online" >&2
  exit 1
fi

# Install only compiler + oneDPL + oneTBB; override via ONEAPI_WINDOWS_COMPONENTS.
ONEAPI_WINDOWS_COMPONENTS="${ONEAPI_WINDOWS_COMPONENTS:-intel.oneapi.win.dpcpp-cpp-compiler;intel.oneapi.win.dpl;intel.oneapi.win.tbb.devel}"

INSTALLER_FILE="/tmp/w_cpp-essentials_p_${VERSION}.exe"
echo "Downloading Intel C++ Essentials online installer from: ${INSTALLER_URL}"
curl -fL "${INSTALLER_URL}" --output "${INSTALLER_FILE}"

echo "Installing components: ${ONEAPI_WINDOWS_COMPONENTS}"
PowerShell -NoProfile -Command "\$p = Start-Process -FilePath '${INSTALLER_FILE}' -ArgumentList '-s --action install --eula accept --components=${ONEAPI_WINDOWS_COMPONENTS}' -Wait -PassThru -NoNewWindow; exit \$p.ExitCode"
rm -f "${INSTALLER_FILE}"

ONEAPI_SETVARS="/c/Program Files (x86)/Intel/oneAPI/setvars.bat"
if [[ ! -f "${ONEAPI_SETVARS}" ]]; then
  echo "Error: oneAPI installation completed but setvars.bat was not found at '${ONEAPI_SETVARS}'." >&2
  exit 1
fi

echo "Verifying oneAPI environment..."
if ! cmd //C '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul && where icx'; then
  echo "Error: icx (Intel DPC++/C++ Compiler) not found after sourcing setvars.bat." >&2
  exit 1
fi

echo "Intel oneAPI installation complete."
