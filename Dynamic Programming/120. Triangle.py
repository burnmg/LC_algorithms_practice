class Solution:
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        if len(triangle) == 0:
            return 0
        if len(triangle) == 1:
            return triangle[0][0]
        
        sum_tri = [[0] * i for i in range(1, len(triangle))]
        sum_tri.append(triangle[-1])
        
        for i in range(len(sum_tri)-2, -1, -1):
            for j in range(len(sum_tri[i])):
                sum_tri[i][j] = triangle[i][j] + min(sum_tri[i+1][j], sum_tri[i+1][j+1])
                
        return sum_tri[0][0]