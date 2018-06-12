class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        def rec_rotate(matrix, start, end):
            if start >= end:
                return
            for i in range(start, end):
                
                 matrix[i][end], matrix[end][end-i+start], matrix[end-i+start][start], matrix[start][i] = matrix[start][i], matrix[i][end], matrix[end][end-i+start], matrix[end-i+start][start]
            rec_rotate(matrix, start+1, end-1)
                
        rec_rotate(matrix, 0, len(matrix) - 1)