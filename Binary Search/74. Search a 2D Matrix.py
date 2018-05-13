class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if len(matrix) == 0 or len(matrix[0]) == 0: return False
        
        for row in matrix:
            if target <= row[-1] and target >= row[0]:
                a,b = 0, len(row) - 1
                while a<=b:
                    m = (a+b) // 2
                    if row[m] == target:
                        return True
                    elif row[m] > target:
                        b -= 1
                    else:
                        a += 1
                return False
        return False