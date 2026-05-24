import numpy as np


def compute_pca(matrix, n_components=None):
    X = matrix.astype(float).copy()
    n_samples, n_features = X.shape
    steps = []

    if n_components is None:
        n_components = min(n_samples, n_features)
    n_components = min(n_components, min(n_samples, n_features))

    # finding centre
    mean_vec = X.mean(axis=0)
    X_c = X - mean_vec
    steps.append({
        "desc": "Step 1 – Centre the data (subtract column means)",
        "formula": "X_centered = X - mean(X, axis=0)",
        "matrix": X_c.copy(),
        "label": "X_centered",
        "extra": f"Mean vector: {np.round(mean_vec, 4).tolist()}",
    })

    # finding covariance
    if n_samples > 1:
        cov = (X_c.T @ X_c) / (n_samples - 1)
    else:
        cov = (X_c.T @ X_c)
    steps.append({
        "desc": "Step 2 – Covariance matrix",
        "formula": "C = (X_c^T * X_c) / (n - 1)",
        "matrix": cov.copy(),
        "label": "Covariance matrix C",
    })

    # finding eigen
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending order; flip to descending
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    steps.append({
        "desc": "Step 3 – Eigendecompose C (eigenvalues = variance along each PC)",
        "formula": "C · v_i = λ_i · v_i   (v_i are principal components)",
        "matrix": eigenvectors.copy(),
        "label": "Eigenvectors (principal components)",
        "eigenvalues": eigenvalues.copy(),
    })

    # select top-k
    components = eigenvectors[:, :n_components]
    selected_vals = eigenvalues[:n_components]
    steps.append({
        "desc": f"Step 4 – Select top {n_components} components",
        "formula": f"W = V[:, :n_components]   shape = ({n_features}, {n_components})",
        "matrix": components.copy(),
        "label": f"W (top {n_components} eigenvectors)",
        "eigenvalues": selected_vals.copy(),
    })

    # project
    X_pca = X_c @ components
    steps.append({
        "desc": "Step 5 – Project data onto principal components",
        "formula": "X_pca = X_centered * W",
        "matrix": X_pca.copy(),
        "label": "Projected data (X_pca)",
    })

    # explained variance
    total_var = float(eigenvalues.sum()) if eigenvalues.sum() > 0 else 1.0
    explained = eigenvalues[:n_components] / total_var
    steps.append({
        "desc": "Step 6 – Explained variance ratio per component",
        "formula": "explained_ratio_i = λ_i / sum(λ)",
        "matrix": np.diag(selected_vals),
        "label": "Selected eigenvalues (diagonal)",
        "eigenvalues": explained.copy(),
        "extra": (
            "  ".join(
                f"PC{i+1}: {r*100:.2f}%"
                for i, r in enumerate(explained)
            )
            + f"  |  Cumulative: {explained.sum()*100:.2f}%"
        ),
    })

    return {
        "Mean Vector": mean_vec.reshape(1, -1),
        "Covariance Matrix": np.round(cov, 4),
        "Principal Components (W)": np.round(components, 4),
        "Eigenvalues": eigenvalues[:n_components].reshape(1, -1),
        "Projected Data": np.round(X_pca, 4),
        "Explained Variance %": (explained * 100).reshape(1, -1),
        "_steps": steps,
        "_explained": explained,
        "_n_components": n_components,
    }
