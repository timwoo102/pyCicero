#!/usr/bin/env bash
# Install skggm 0.2.8 (QUIC graphical lasso) with two import fixes for current scikit-learn/joblib.
# The PyPI release imports sklearn.utils.testing and sklearn.externals.joblib, which scikit-learn
# removed in 0.24, and its setup.py needs numpy/scipy/Cython present at build time. scikit-learn
# must be < 1.8 (skggm passes force_all_finite to as_float_array, removed in 1.8).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

python -m pip install "numpy>=1.24,<2" "scipy>=1.10" "scikit-learn>=1.3,<1.8" "cython<3" setuptools wheel
python -m pip download --no-deps --no-binary :all: --no-build-isolation skggm==0.2.8 -d "$TMP"
tar xzf "$TMP"/skggm-0.2.8.tar.gz -C "$TMP"
patch -d "$TMP"/skggm-0.2.8 -p1 < "$HERE"/skggm-0.2.8-sklearn-compat.patch
python -m pip install --no-build-isolation --no-deps "$TMP"/skggm-0.2.8   # its listed deps (nilearn, black, nose, ...) are not needed
python -c "from inverse_covariance import QuicGraphicalLasso; print('skggm OK')"
