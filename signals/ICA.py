import math
import numpy as np
from scipy import linalg, signal

from .utils import getRGB, detrend


def ICA(frames, Fs):
    # Cut off frequency.
    LPF = 0.7
    HPF = 2.5
    rgb_data = getRGB(frames)

    NyquistF = 0.5 * Fs
    BGRNorm = np.zeros(rgb_data.shape)
    Lambda = 100
    for c in range(3):
        BGRDetrend = detrend(rgb_data[:, c], Lambda)
        BGRNorm[:, c] = (BGRDetrend - np.mean(BGRDetrend)) / np.std(BGRDetrend)
    _, S = ica(BGRNorm.T, 3)

    # select BVP Source
    MaxPx = np.zeros((1, 3))
    for c in range(3):
        FF = np.fft.fft(S[c, :])  # T x 1
        FF = FF[1:]
        N = FF.shape[0]
        Px = np.abs(FF[:math.floor(N / 2)]) ** 2
        Px = Px / np.sum(Px, axis=0)
        MaxPx[0, c] = np.max(Px)
    MaxComp = np.argmax(MaxPx)
    BVP_I = S[MaxComp, :]
    B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], 'bandpass')
    BVP = signal.filtfilt(B, A, BVP_I.astype(np.double))

    return BVP


def ica(X, Nsources, Wprev=0):
    nRows = X.shape[0]
    nCols = X.shape[1]
    if nRows > nCols:
        print("Warning - The number of rows is cannot be greater than the number of columns.")
        print("Please transpose input.")

    if Nsources > min(nRows, nCols):
        Nsources = min(nRows, nCols)
        print('Warning - The number of soures cannot exceed number of observation channels.')
        print('The number of sources will be reduced to the number of observation channels ',
              Nsources)

    Winv, Zhat = jade(X, Nsources, Wprev)
    W = np.linalg.pinv(Winv)  # 求解伪逆
    return W, Zhat


def jade(X, m, Wprev):
    n = X.shape[0]
    T = X.shape[1]
    nem = m
    seuil = 1 / math.sqrt(T) / 100
    if m < n:
        D, U = np.linalg.eig(np.matmul(X, X.T) / T)
        Diag = D
        k = np.argsort(Diag)
        pu = Diag[k]
        ibl = np.sqrt(pu[n - m:n] - np.mean(pu[0:n - m]))
        bl = np.true_divide(np.ones(m, 1), ibl)
        W = np.matmul(np.diag(bl), np.transpose(U[0:n, k[n - m:n]]))
        IW = np.matmul(U[0:n, k[n - m:n]], np.diag(ibl))
    else:
        IW = linalg.sqrtm(np.matmul(X, X.T) / T)
        W = np.linalg.inv(IW)

    Y = np.matmul(W, X)
    R = np.matmul(Y, Y.T) / T
    C = np.matmul(Y, Y.T) / T
    Q = np.zeros((m ** 4, 1))
    index = 0

    for lx in range(m):
        Y1 = Y[lx, :]
        for kx in range(m):
            Yk1 = np.multiply(Y1, np.conj(Y[kx, :]))
            for jx in range(m):
                Yjk1 = np.multiply(Yk1, np.conj(Y[jx, :]))
                for ix in range(m):
                    Q[index] = np.matmul(Yjk1 / math.sqrt(T),
                                         Y[ix, :].T / math.sqrt(T)) - R[ix, jx] * R[lx, kx] - \
                                         R[ix, kx] * R[lx, jx] - C[ix, lx] * np.conj(C[jx, kx])
                    index += 1
    # Compute and Reshape the significant Eigen
    D, U = np.linalg.eig(Q.reshape(m * m, m * m))
    Diag = abs(D)
    K = np.argsort(Diag)
    la = Diag[K]
    M = np.zeros((m, nem * m), dtype=complex)
    h = m * m - 1
    for u in range(0, nem * m, m):
        Z = U[:, K[h]].reshape((m, m))
        M[:, u:u + m] = la[h] * Z
        h = h - 1
    # Approximate the Diagonalization of the Eigen Matrices:
    B = np.array([[1, 0, 0], [0, 1, 1], [0, 0 - 1j, 0 + 1j]])
    Bt = B.T

    encore = 1
    if Wprev == 0:
        V = np.eye(m).astype(complex)
    else:
        V = np.linalg.inv(Wprev)
    # Main Loop:
    while encore:
        encore = 0
        for p in range(m - 1):
            for q in range(p + 1, m):
                Ip = np.arange(p, nem * m, m)
                Iq = np.arange(q, nem * m, m)
                g = np.array([M[p, Ip] - M[q, Iq], M[p, Iq], M[q, Ip]])
                temp1 = np.matmul(g, g.T)
                temp2 = np.matmul(B, temp1)
                temp = np.matmul(temp2, Bt)
                D, vcp = np.linalg.eig(np.real(temp))
                K = np.argsort(D)
                angles = vcp[:, K[2]]
                if angles[0] < 0:  # 3 x 1
                    angles = -angles
                c = np.sqrt(0.5 + angles[0] / 2)
                s = 0.5 * (angles[1] - 1j * angles[2]) / c

                if abs(s) > seuil:
                    encore = 1
                    pair = [p, q]
                    G = np.array([[c, -np.conj(s)], [s, c]])
                    V[:, pair] = np.matmul(V[:, pair], G)
                    M[pair, :] = np.matmul(G.T, M[pair, :])
                    temp1 = c * M[:, Ip] + s * M[:, Iq]
                    temp2 = -np.conj(s) * M[:, Ip] + c * M[:, Iq]
                    M[:, Ip] = temp1
                    M[:, Iq] = temp2

    # Whiten the Matrix
    # Estimation of the Mixing Matrix and Signal Separation
    A = np.matmul(IW, V)
    S = np.matmul(V.T, Y)
    return A, S
