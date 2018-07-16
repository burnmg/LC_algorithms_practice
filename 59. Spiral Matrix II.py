class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        M = [[0 for _ in range(n)] for _ in range(n)]
        
        i,j = 0, 0
        state = 0 # 0 for moving right, 1 for down, 2 for left, 3 for up
        low_bound, high_bound = 0, n - 1
        for num in range(1, n ** 2 + 1):
            M[i][j] = num
            
            if state == 0:
                j += 1
                if j == high_bound:
                    state = 1
            elif state == 1:
                i += 1
                if i == high_bound:
                    state = 2
            elif state == 2:
                j -= 1
                if j == low_bound:
                    state = 3
            else:
                i -= 1
                if i == low_bound:
                    state = 0
                    i += 1
                    j += 1
                    high_bound -= 1
                    low_bound += 1
        return M