#!/bin/bash

# Install Intel oneAPI C++ Essentials for SYCL builds.
# Usage: Linux.sh <VERSION>
# Example: Linux.sh 2025.3.1

VERSION=${1:?'Usage: Linux.sh <VERSION>  (e.g. Linux.sh 2025.3.1)'}

# The apt package uses X.Y version format (e.g. 2025.3), while the pip package
# (intel-sycl-rt) may use X.Y.Z format (e.g. 2025.3.1). Truncate to X.Y.
APT_VERSION=$(echo "${VERSION}" | grep -oP '^\d+\.\d+')
if [ -z "${APT_VERSION}" ]; then
  echo "Error: VERSION '${VERSION}' does not match expected X.Y or X.Y.Z format." >&2
  exit 1
fi

wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor \
  | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt-get -qq update
sudo apt-get install -y intel-cpp-essentials-${APT_VERSION}
sudo apt clean
