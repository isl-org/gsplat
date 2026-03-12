#!/bin/bash

# Install Intel oneAPI C++ Essentials for SYCL builds.
# Usage: Linux.sh <VERSION>
# Example: Linux.sh 2025.3.1

VERSION=${1:?'Usage: Linux.sh <VERSION>  (e.g. Linux.sh 2025.3.1)'}

wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor \
  | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt-get -qq update
sudo apt-get install -y intel-cpp-essentials-${VERSION}
sudo apt clean
