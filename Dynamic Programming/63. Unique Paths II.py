class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        grid = [[0 for _ in range(len(obstacleGrid[0]))] for _ in range(len(obstacleGrid))]
        

        for i in range(len(grid[0])):
            if obstacleGrid[0][i] != 1:
                grid[0][i] = 1
            else:
                break
        
        for i in range(len(grid)):
            if obstacleGrid[i][0] != 1: 
                grid[i][0] = 1
            else:
                break
        
        for i in range(1, len(grid)):
            for j in range(1, len(grid[0])):
                if obstacleGrid[i][j] != 1:
                    grid[i][j] = grid[i-1][j] + grid[i][j-1]
        
        return grid[-1][-1]