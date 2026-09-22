#!/usr/bin/env bash
# One-shot dependency setup for building Twisty on Ubuntu.
set -euo pipefail

sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libopenvdb-dev \
    libtbb-dev \
    libomp-dev \
    nlohmann-json3-dev \
    libboost-dev
