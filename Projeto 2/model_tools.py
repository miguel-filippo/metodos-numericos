# main.py  ─────────  Projeto 2  (Mínimos-Quadrados + Seleção de Variáveis)
#
# Estrutura:
#   1. Utilidades gerais  .............  carregamento, normalização, LS helpers
#   2. Fitters (BFGS ∕ GD)  ...........  como ajustar θ para um Xb qualquer
#   3. Validador k-fold  ..............  estima erro fora-da-amostra
#   4. Seleção de variáveis ...........  escolhe melhor subconjunto p/ cada R
#   5. Pipeline main() ................  orquestra → treina → prevê → salva

import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ╭──────────────────────── UTILIDADES GERAIS ───────────────────────╮
def load_csv(path: str) -> np.ndarray:
    """
    Load CSV without header that uses commas both as separator and decimal.

    Parameters
    ----------
    path : str
        Relative/absolute path to the CSV file.

    Returns
    -------
    ndarray (N, D)
        Raw data matrix.
    """
    return pd.read_csv(path, header=None, decimal=',').to_numpy()

def create_nonlinear_features(X: np.ndarray) -> np.ndarray:
    """
    Create nonlinear features for each column in X.
    For each j ∈ [0, D-1], append:
    - xj²
    - xj³
    - log(xj)
    - xj * xl (for each l ≠ j and l ∈ [0, D-1])
    Parameters
    ----------
    X : ndarray (N, D)
        Input feature matrix.

    Returns
    -------
    X_new : ndarray (N, D')
        Output feature matrix with added nonlinear features.
    """
    N, D = X.shape
    X_new = np.zeros((N, (D * 4) + len(list(combinations(range(D), 2)))))  # Preallocate for new features

    for i in range(N):
        X_new[i, :D] = X[i, :]  # Copy original features
        # Add xj², xj³ and log(xj) for each j
        # Note: log(0) is undefined, so we add a small constant to avoid it
        for j in range(D):
            X_new[i][D + j] = X[i][j] ** 2
            X_new[i][2 * D + j] = X[i][j] ** 3
            X_new[i][3 * D + j] = np.log(np.abs(X[i][j]) + 1e-8)

        # Add xj * xl for each j and l ≠ j
        index_counter = 0
        for cols in combinations(range(D), 2):
            j, l = cols
            X_new[i][4 * D + index_counter] = X[i][j] * X[i][l]
            index_counter += 1

    return X_new

def normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score each column: (x-E(x))/Var(x).

    Returns
    -------
    X_norm : ndarray
        Normalised matrix (mean 0, std 1 per feature).
    mean   : ndarray
        Feature-wise means (for inverse transform / test norm).
    std    : ndarray
        Feature-wise stds.
    """
    mean, std = X.mean(0), X.std(0)
    return (X - mean) / std, mean, std

def add_bias(X: np.ndarray) -> np.ndarray:
    """Prepend a column of 1 s → handles θ₀ (intercept)."""
    return np.hstack([np.ones((X.shape[0], 1)), X])

def mse(theta: np.ndarray, Xb: np.ndarray, y: np.ndarray) -> float:
    """Mean-squared error   L(θ) = 1/N ‖y − X_b θ‖² ."""
    return float(np.mean((y - Xb @ theta) ** 2))

def gradient(theta: np.ndarray, Xb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Analytic ∇θ L (convex LS)."""
    return (2 / Xb.shape[0]) * Xb.T @ (Xb @ theta - y)

def gd(Xb: np.ndarray, y: np.ndarray, *,
    alpha=0.05, max_iter=5_000, tol=1e-6) -> np.ndarray:
    """
    Plain batch Gradient Descent.

    Stops when ‖∇L‖ < tol **or** max_iter reached.
    """
    theta = np.zeros(Xb.shape[1])
    prev_loss = mse(theta, Xb, y)

    for _ in range(max_iter):
     g = gradient(theta, Xb, y)
     if np.linalg.norm(g) < tol:
         break

     theta -= alpha * g

     # Check relative tolerance
     current_loss = mse(theta, Xb, y)
     if abs(prev_loss - current_loss) / (abs(prev_loss) + 1e-8) < tol:
         break
     prev_loss = current_loss

    return theta
# ╰──────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────── FITTERS (BFGS / GD) ─────────────────╮
def fit_bfgs(Xb: np.ndarray, y: np.ndarray, print_steps=False) -> np.ndarray:
    """
    Minimise LS using quasi-Newton BFGS (SciPy).

    Returns
    -------
    θ̂ : ndarray
        Optimal coefficients.
    """
    iteration = 0
    theta0 = np.zeros(Xb.shape[1])

    if print_steps:
        def print_iteration(xk):
            nonlocal iteration
            print(f"Iteration {iteration}: {xk}")
            iteration += 1
    else:
        print_iteration = None

    return minimize(mse, theta0, args=(Xb, y), method="BFGS", callback=print_iteration).x

def fit_gd(Xb: np.ndarray, y: np.ndarray, **gd_kw) -> np.ndarray:
    """
    Wrapper to run our own GD.

    Extra kwargs (alpha, max_iter, tol) are forwarded.
    """
    return gd(Xb, y, **gd_kw)
# ╰──────────────────────────────────────────────────────────────────╯


# ╭────────────────────────── VALIDADORES (k-fold) ──────────────────╮
def kfold_mse(cols, X_norm, y, *, k: int, fit_function, **fit_kw) -> float:
    """
    Generic k-fold CV.

    Parameters
    ----------
    cols    : tuple[int]
        Indices of selected features.
    X_norm  : ndarray (N, D)
        Normalised full matrix.
    y       : ndarray (N,)
        Target vector.
    k       : int
        Number of folds (e.g. 5).
    fit_function : callable
        Function that returns θ̂ given (Xb, y).
    fit_kw  : dict
        Extra kwargs forwarded to fit_function.

    Returns
    -------
    float
        Mean MSE across folds.
    """
    N = X_norm.shape[0]
    idx = np.random.default_rng(42).permutation(N)  # reproducible shuffle
    folds = np.array_split(idx, k)
    losses = []

    for i in range(k):
        val, train = folds[i], np.hstack(folds[:i] + folds[i+1:])
        # --- split ----
        Xtr, ytr = X_norm[train][:, cols], y[train]
        Xv,  yv  = X_norm[val ][:, cols], y[val]
        # --- train ----
        theta = fit_function(add_bias(Xtr), ytr, **fit_kw)
        # --- validate --
        losses.append(mse(theta, add_bias(Xv), yv))

    return float(np.mean(losses))
# ╰──────────────────────────────────────────────────────────────────╯

# ╭──────────────────────── SELEÇÃO DE VARIÁVEIS ────────────────────╮
def best_subset_by_R(X_norm, y, R_vals=(1, 2, 3, 4), *,
                     k=5, fit_function, **fit_kw):
    """
    For each R ∈ R_vals, test all comb(5, R) feature sets and
    return the one with minimal k-fold MSE.
    """
    result = {}
    for R in R_vals:
        best = {"cols": None, "mse": np.inf}
        for cols in combinations(range(X_norm.shape[1]), R):
            err = kfold_mse(cols, X_norm, y, k=k, fit_function=fit_function, **fit_kw)
            if err < best["mse"]:
                best = {"cols": cols, "mse": err}
        result[R] = best
        print(f"R={R} | CV-MSE={best['mse']:<18} | cols={best['cols']}")

    return result
# ╰──────────────────────────────────────────────────────────────────╯

# ╭────────────────────── HARDCODED MODEL GENERATOR ─────────────────╮
def generate_model(R: int) -> dict:
    """
    Returns pre-computed optimal theta values for each R based on best features.
    Calculates thetas using the actual fitters with the correct features.

    Parameters
    ----------
    R : int
        Number of features (1, 2, 3, or 4)

    Returns
    -------
    dict
        Dictionary containing 'cols', 'theta'
    """
    # Load and prepare data
    Xy = load_csv('dados/Concreto - treino.csv')
    y = Xy[:, -1]
    X = create_nonlinear_features(Xy[:, :-1])
    X_norm, mean, std = normalize(X)

    # Define the best columns for each R based on the file
    best_cols = {
        1: (0,),
        2: (22, 25),
        3: (3, 15, 24),
        4: (9, 14, 15, 27)
    }

    if R not in best_cols:
        raise ValueError(f"R must be one of {list(best_cols.keys())}, got {R}")

    # Get the subset of features for this R
    cols = best_cols[R]
    Xb_full = add_bias(X_norm[:, cols])

    # Calculate theta using BFGS
    theta = fit_bfgs(Xb_full, y)

    # Calculate mse for validation
    mse_value = kfold_mse(cols, X_norm, y, k=5, fit_function=fit_bfgs)

    return {
        'cols': cols,
        'theta': theta,
        'mean': mean,
        'std': std,
        'mse': mse_value
    }
# ╰──────────────────────────────────────────────────────────────────╯

class Model:
    def __init__(self, R: int):
        self.R = R
        data = generate_model(R)
        self.cols = data['cols']
        self.theta = data['theta']
        self.mean = data['mean']
        self.std = data['std']
        self.mse = data['mse']

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict using the model on new data.

        Parameters
        ----------
        X_test : ndarray (N, D)
            New data to predict.

        Returns
        -------
        ndarray (N,)
            Predicted values.
        """
        # Normalize the test data using the training mean and std
        X_test_engineered = create_nonlinear_features(X_test)
        X_test_norm = (X_test_engineered - self.mean) / self.std
        Xb_test = add_bias(X_test_norm[:, self.cols])
        return Xb_test @ self.theta

# ╭────────────────────────────── PIPELINE ──────────────────────────╮
def main():
    """Entire workflow: load → select vars → train → predict."""
    # 1. Load & z-score -------------------------------------------------
    Xy = load_csv('dados/Concreto - treino.csv')
    y = Xy[:, -1]
    X = create_nonlinear_features(Xy[:, :-1])
    X_norm, mean, std = normalize(X)

    # 2. Variable selection via CV -------------------------------------
    print("Variable selection via CV using BFGS:")
    print("-" * 50)
    best_bfgs = best_subset_by_R(X_norm, y, fit_function=fit_bfgs)
    print("-" * 50)

    """ print()
    print("Variable selection via CV using GD:")
    print("-" * 50)
    best_gd   = best_subset_by_R(X_norm, y, fit_function=fit_gd, alpha=0.05)
    print("-" * 50) """

    # 3.1. Final training on full data -----------------------------------
    models = {}
    print()
    print("Final training on full data:")
    print("-" * 100)
    for R, info in best_bfgs.items():
        cols = info["cols"]
        Xb_full = add_bias(X_norm[:, cols])
        models[R] = {
            "cols": cols,
            "theta_bfgs": fit_bfgs(Xb_full, y),
            "theta_gd":   fit_gd(Xb_full, y, alpha=0.05)
        }

        # Print truncated theta vectors for aligned output
        theta_bfgs_str = np.array2string(models[R]['theta_bfgs'], precision=3, max_line_width=80, threshold=6)
        theta_gd_str   = np.array2string(models[R]['theta_gd'],   precision=3, max_line_width=80, threshold=6)
        print(f"R={R}   |   θ_bfgs={theta_bfgs_str:<40} |   θ_gd={theta_gd_str:<40}")
    print("-" * 100)

    # 4. Predict on hidden test set and Plotting ------------------------------------
    # 4.1. Predict on hidden test set
    Xtest = load_csv('dados/Concreto - teste.csv')
    Xtest_norm = (Xtest - mean) / std
    print()
    print("Differences between BFGS and GD:")
    print("-" * 50)
    for R, m in models.items():
        diff = np.linalg.norm(m["theta_bfgs"] - m["theta_gd"])
        print(f"R={R}  ‖θ_bfgs-θ_gd‖ = {diff}")

        Xtbias = add_bias(Xtest_norm[:, m["cols"]])
        ypred_bfgs = Xtbias @ m["theta_bfgs"]
        ypred_gd   = Xtbias @ m["theta_gd"]

    # 4.2. Plotting ------------------------------------------------------
    PLOT_DIR = Path("graficos")
    PLOT_DIR.mkdir(exist_ok=True)

    # ---------------------------------------------------------------------
    # 4.2.1. Distribuição dos dados (cada xj com cor diferente)
    # ---------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    for j in range(X_norm.shape[1]):
        plt.scatter(X_norm[:, j], y,
                    s=10, alpha=0.6, label=f"x{j}")
    plt.xlabel("Valor normalizado da feature")
    plt.ylabel("y")
    plt.title("Dispersão y × cada feature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "dist_features.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------------------
    # 4.2.2. Para cada R: scatter + linha de regressão BFGS + GD
    #    (usamos a primeira feature do subconjunto só para visualização 2-D)
    # ---------------------------------------------------------------------
    """ for R, info in best_gd.items():                     # mesmo subset usado p/ fit
        cols = info["cols"]
        x1 = X_norm[:, cols[0]]                         # feature escolhida p/ eixo x
        idx = np.argsort(x1)
        Xb_full = add_bias(X_norm[:, cols])
        th_b, th_g = models[R]["theta_bfgs"], models[R]["theta_gd"]

        plt.figure(figsize=(8, 6))
        plt.scatter(x1, y, s=10, alpha=0.4, label="dados")
        plt.plot(x1[idx], add_bias(X_norm[idx][:, cols]) @ th_b,
                label="BFGS", linewidth=2)
        plt.plot(x1[idx], add_bias(X_norm[idx][:, cols]) @ th_g,
                "--", label="GD", linewidth=2)
        plt.xlabel(f"Primeira feature do subconjunto (x{cols[0]})")
        plt.ylabel("y")
        plt.title(f"R={R}  ·  Regressão BFGS × GD")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"lines_R{R}.png", dpi=300)
        plt.close() """

    # ---------------------------------------------------------------------
    # 4.2.3. Diferença entre vetores θ  (‖θ_BFGS − θ_GD‖₂  por R)
    # ---------------------------------------------------------------------
    norms = [np.linalg.norm(models[R]["theta_bfgs"] - models[R]["theta_gd"])
            for R in (1, 2, 3, 4)]
    plt.figure(figsize=(6, 4))
    plt.bar(["R1", "R2", "R3", "R4"], norms)
    plt.ylabel("‖θ_BFGS − θ_GD‖₂")
    plt.title("Diferença entre parâmetros")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "theta_diff.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------------------
    # 4.2.4. Diferença entre predições (histograma BFGS − GD) para cada R
    # ---------------------------------------------------------------------
    for R, m in models.items():
        Xtbias = add_bias(Xtest_norm[:, m["cols"]])
        diff   = (Xtbias @ m["theta_bfgs"]) - (Xtbias @ m["theta_gd"])

        plt.figure(figsize=(6, 4))
        plt.hist(diff, bins=30)
        plt.title(f"Predições: BFGS − GD  (R={R})")
        plt.xlabel("Diferença")
        plt.ylabel("Frequência")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"pred_diff_R{R}.png", dpi=300)
        plt.close()

    print(f"\n🖼  Gráficos salvos em {PLOT_DIR.absolute()}")

    # save each vector in its own CSV
    out_path_bfgs = Path(f'previsoes/Y_pred_BFGS_R{R}.csv')
    out_path_bfgs.write_text('\n'.join(f'{v:.6f}' for v in ypred_bfgs))

    out_path_gd = Path(f'previsoes/Y_pred_GD_R{R}.csv')
    out_path_gd.write_text('\n'.join(f'{v:.6f}' for v in ypred_gd))
    print("-" * 50)
    print()
    print("Predictions saved to ./previsoes/")


if __name__ == "__main__":
    main()
# ╰──────────────────────────────────────────────────────────────────╯
