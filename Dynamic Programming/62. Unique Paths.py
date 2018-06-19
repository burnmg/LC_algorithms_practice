class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        grid = [[1] * n for _ in xrange(m)]
        
        for i in xrange(1, m):
            for j in xrange(1, n):
                grid[i][j] = grid[i-1][j] + grid[i][j-1]
            
        return grid[m-1][n-1]
    