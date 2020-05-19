import numpy as np
from scipy import sparse


class CalcTools:
    @staticmethod
    def matrix_to_3D_block_diag(rx):
        """Takes (m x n) 2D array, rx, and returns an (n x m x m) 3D block diagonal matrix, rxB,
        where, the jth column of rx is put in the jth (m x m) block diagonal matrix of the rxB
        i.e., rx[:, j] is in the diagonal of rxB[j, :, :]"""

        # define empty 3D array
        rxB = np.zeros((rx.T).shape + (rx.T).shape[-1:], dtype=(rx.T).dtype)

        # get the diagonals of each block
        diagonals_rxB = np.diagonal(rxB, axis1=-2, axis2=-1)

        # set flag to write on the diagonal of each block
        diagonals_rxB.setflags(write=True)

        # write the geometric factor of each element to the corresponding block
        diagonals_rxB[:] = rx.T

        return rxB

    @staticmethod
    def block_diag_to_block_vec(rxB):
        """ Takes a (m x n x n) 3D diagonal array and returns a 3D (m x n x 1) array"""
        m = rxB.shape[0]
        n = rxB.shape[1]
        rx = np.diagonal(rxB, axis1=1, axis2=2)
        rxB = rx.reshape(m, n, 1)

        return rxB

    @staticmethod
    def matrix_to_repeating_block_vec(H, n, shape1, shape2):
        """Takes an (shape1 x shape2) matrix and creates an (n x shape1 x shape2) 3D block array where each of the nth
        entry contain H"""
        HB = np.block([H] * n).T.reshape(n, shape1, shape2).transpose(0, 2, 1)

        return HB

