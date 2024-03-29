import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF


def pd_options():
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)


def svd(X):
    n, m = X.shape
    # Compute full SVD
    U, Sigma, Vh = np.linalg.svd(X,
        full_matrices=False,
        compute_uv=True)

    X_svd = np.dot(U, Sigma)
    return X_svd


if __name__ == '__main__':
    pd_options()
    df = pd.read_csv('bars.csv', header=None)

    U, S, VT = np.linalg.svd(df.values)

    print("Left Singular Vectors:")
    print(U)
    print("Singular Values:")
    print(np.diag(S))
    print("Right Singular Vectors:")
    print(VT)

    svd_ = TruncatedSVD(n_components=20)
    A_transf = svd_.fit_transform(df.values)
    recovered_matrix = A_transf.dot(svd_.components_)
    print(recovered_matrix[0])
    print("Transformed Matrix after reducing to 20 features:")
    print(A_transf)

    nmf_ = NMF(n_components=20)
    NMF_transf = nmf_.fit_transform(df.values)
    recovered_nmf = NMF_transf.dot(nmf_.components_)
    print("Recovered NMF first line:")
    print(recovered_nmf[0])
    print(np.shape(NMF_transf))
    print("SVD")
    print(np.linalg.norm(df.values - recovered_matrix, ord="fro"))
    print("NMF")
    print(np.linalg.norm(df.values - recovered_nmf, ord="fro"))


