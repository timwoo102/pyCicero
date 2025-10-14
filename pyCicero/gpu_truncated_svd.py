import numpy as np
import scipy.sparse as sp

import cupy as cp
import cupyx
import cupyx.scipy.sparse as cpx
from cupyx.scipy.sparse.linalg import svds


class TruncatedSVD:
    """
    CuPy-backed TruncatedSVD (LSA) with a scikit-learn-like interface.

    - Accepts CPU (NumPy / SciPy) or GPU (CuPy / cupyx) inputs
      and converts to GPU automatically.
    - Uses cupyx.scipy.sparse.linalg.svds for truncated SVD.
    - Returns outputs on the same side as the input ("auto"):
        * CPU input  -> returns NumPy arrays
        * GPU input  -> returns CuPy arrays

    Attributes after fit():
        n_components_: int
        components_: array-like, shape (n_components, n_features)  # V^T
        singular_values_: array-like, shape (n_components,)
        explained_variance_: array-like, shape (n_components,)
        explained_variance_ratio_: array-like, shape (n_components,)
        n_features_in_: int
        n_samples_: int
        dtype_: np.dtype (float32)
        device_: 'cpu' | 'gpu'  (based on the *fit* input)
    """

    def __init__(self, 
                 n_components=50, 
                 tol=1e-4, 
                 maxiter=1000, 
                 dtype=np.float32, 
                 device = "gpu", 
                 save_attributes = False #set to false to save memory
                 ):
        if n_components <= 0:
            raise ValueError("n_components must be > 0")
        self.n_components = int(n_components)
        self.tol = tol
        self.maxiter = maxiter
        self.dtype = np.dtype(dtype)
        self.device_ = device
        self.save_attributes = save_attributes

        # Populated after fit
        self.n_components_ = None
        self.components_ = None         # V^T
        self.singular_values_ = None    # S
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_features_in_ = None
        self.n_samples_ = None
        self.dtype_ = None
        

    # -----------------------
    # Public API
    # -----------------------
    def fit(self, X):
        X_gpu, on_gpu = self._to_gpu(X)
        m, n = X_gpu.shape

        if self.n_components > min(m, n):
            raise ValueError(
                f"n_components={self.n_components} must be <= min(n_samples, n_features)={min(m, n)}"
            )

        # Compute truncated SVD on GPU
        # svds returns singular values in ascending order; we sort descending to match sklearn
        u, s, vt = svds(
            X_gpu, k=self.n_components, which="LM", tol=self.tol, maxiter=self.maxiter
        )
        order = cp.argsort(s)[::-1]
        s = s[order]
        vt = vt[order]        # (k, n)
        # (We don't need U to store components_, but it's used in fit_transform shortcut)

        if self.save_attributes:
            # Explained variance (sklearn formula for TSVD):
            # explained_variance_j = s_j^2 / (n_samples - 1)
            # explained_variance_ratio_j = explained_variance_j / total_var
            # where total_var = ||X||_F^2 / (n_samples - 1) (no centering)
            s2 = s**2
            denom = (m - 1) if m > 1 else 1.0
            explained_variance = s2 / denom

            total_ss = self._sum_of_squares(X_gpu)
            total_var = total_ss / denom
            # Avoid divide-by-zero if matrix is all zeros
            evr = explained_variance / cp.asarray(total_var) if total_var != 0 else cp.zeros_like(explained_variance)

            # Save attributes (on GPU)
            self.n_components_ = self.n_components
            self.components_ = vt.astype(self.dtype, copy=False)
            self.singular_values_ = s.astype(self.dtype, copy=False)
            self.explained_variance_ = explained_variance.astype(self.dtype, copy=False)
            self.explained_variance_ratio_ = evr.astype(self.dtype, copy=False)
            self.n_features_in_ = int(n)
            self.n_samples_ = int(m)
            self.dtype_ = np.dtype(self.dtype)
            # self.device_ = 'gpu' if on_gpu else 'cpu'
            if self.device_ == "cpu":
                self._move_attrs_to_cpu()

        return self

    def transform(self, X):
        """Project X into the TSVD space: X @ components_.T  (returns U * S)."""
        self._check_is_fitted()
        X_gpu, on_gpu = self._to_gpu(X)

        vt_gpu = self.components_
        if not isinstance(vt_gpu, cp.ndarray):
            # ensure GPU copy for multiplication
            vt_gpu = cp.asarray(vt_gpu)

        # X @ V  (since components_ = V^T, we multiply by its transpose)
        Z = self._matmul(X_gpu, vt_gpu.T)  # shape (m, k)

        # Fit/transform equivalence: X V = U S, so Z is already U*S
        # Keep type consistent with dtype_
        Z = Z.astype(self.dtype_, copy=False)

        if self.device_ == "cpu":
            return cp.asnumpy(Z)  # CuPy
        else:
            return Z  # NumPy

    def fit_transform(self, X):
        """Efficient path using U*S from the SVD computed during fit."""
        X_gpu, on_gpu = self._to_gpu(X)
        m, n = X_gpu.shape

        if self.n_components > min(m, n):
            raise ValueError(
                f"n_components={self.n_components} must be <= min(n_samples, n_features)={min(m, n)}"
            )

        # Compute SVD once here to avoid projecting again
        u, s, vt = svds(
            X_gpu, k=self.n_components, which="LM", tol=self.tol, maxiter=self.maxiter
        )
        order = cp.argsort(s)[::-1]
        s = s[order].astype(self.dtype, copy=False)
        vt = vt[order].astype(self.dtype, copy=False)
        u = u[:, order].astype(self.dtype, copy=False)

        if self.save_attributes:
            # Save fitted attributes
            s2 = s**2
            denom = (m - 1) if m > 1 else 1.0
            explained_variance = s2 / denom
            total_ss = self._sum_of_squares(X_gpu)
            total_var = total_ss / denom
            evr = explained_variance / cp.asarray(total_var) if total_var != 0 else cp.zeros_like(explained_variance)

            self.n_components_ = self.n_components
            self.components_ = vt
            self.singular_values_ = s
            self.explained_variance_ = explained_variance
            self.explained_variance_ratio_ = evr
            self.n_features_in_ = int(n)
            self.n_samples_ = int(m)
            self.dtype_ = np.dtype(self.dtype)
            if self.device_ == "cpu":
                self._move_attrs_to_cpu()
            # self.device_ = 'gpu' if on_gpu else 'cpu'

        # Return U * S without forming a dense diag
        Z = u * s[None, :]

        if self.device_ == "cpu":        
            return cp.asnumpy(Z)
        else:
            return Z

    # -----------------------
    # Helpers
    # -----------------------
    def _to_gpu(self, X):
        """Return (X_gpu, on_gpu_bool). Converts dtype->float32 if needed.
           Supports: NumPy ndarray, SciPy CSR/CSC/COO; CuPy ndarray; cupyx sparse."""
        # CuPy dense
        if isinstance(X, cp.ndarray):
            return X.astype(self.dtype, copy=False), True

        # cupyx sparse
        if cupyx.scipy.sparse.isspmatrix(X):
            return X.astype(self.dtype, copy=False), True

        # SciPy sparse (CPU) -> CuPy sparse (GPU)
        if sp.issparse(X):
            X = X.astype(self.dtype, copy=False)
            # Normalize to CSR on GPU
            if sp.isspmatrix_csr(X):
                return cpx.csr_matrix(X), False
            elif sp.isspmatrix_csc(X):
                return cpx.csc_matrix(X), False
            else:
                # coo/other -> csr
                return cpx.csr_matrix(X.tocsr()), False

        # NumPy dense (CPU) -> CuPy dense (GPU)
        if isinstance(X, np.ndarray):
            return cp.asarray(X, dtype=self.dtype), False

        raise TypeError(
            "Unsupported X type. Provide NumPy/CuPy array or SciPy/cupyx sparse matrix."
        )

    @staticmethod
    def _matmul(A, B):
        """Sparse-safe matmul on GPU (A can be dense or cupyx sparse)."""
        if cupyx.scipy.sparse.isspmatrix(A):
            return A @ B
        else:
            return A.dot(B)

    @staticmethod
    def _sum_of_squares(X_gpu):
        """Sum of squares ||X||_F^2 on GPU (works for dense or cupyx sparse)."""
        if cupyx.scipy.sparse.isspmatrix(X_gpu):
            # Only data are nonzero entries
            return (X_gpu.data.astype(cp.float64) ** 2).sum()
        else:
            return (X_gpu.astype(cp.float64) ** 2).sum()

    def _move_attrs_to_cpu(self):
        """Move learned attributes to CPU (NumPy)."""
        self.components_ = cp.asnumpy(self.components_) if isinstance(self.components_, cp.ndarray) else self.components_
        self.singular_values_ = cp.asnumpy(self.singular_values_) if isinstance(self.singular_values_, cp.ndarray) else self.singular_values_
        self.explained_variance_ = cp.asnumpy(self.explained_variance_) if isinstance(self.explained_variance_, cp.ndarray) else self.explained_variance_
        self.explained_variance_ratio_ = cp.asnumpy(self.explained_variance_ratio_) if isinstance(self.explained_variance_ratio_, cp.ndarray) else self.explained_variance_ratio_

    def _check_is_fitted(self):
        if self.components_ is None:
            raise RuntimeError("This TruncatedSVD instance is not fitted yet. Call 'fit' or 'fit_transform' first.")
