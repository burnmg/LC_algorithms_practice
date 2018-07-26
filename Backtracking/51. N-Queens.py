class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        res = []
        self.backtrack([], [], [], n, res)
        print(res)
        
        
        final_res = []
        for sol in res:
            str_res = [ ['.' for _ in range(n)] for _ in range(n)]
            for i, j in enumerate(sol):
                str_res[i][j] = 'Q'
            final_res.append(map(lambda x: "".join(x), str_res))
                
        return final_res
    
    
    def backtrack(self, queens, xy_sum, xy_diff, n, results):
        '''
        queens:
            its indices are queens horizontal locations
            its values are queens vertical locations
        '''
        
        if len(queens) == n:
            results.append(queens)
            return
        
        for i in range(n):
            _sum = i+len(queens) 
            _diff = i-len(queens) 
            if i not in queens and _sum not in xy_sum and _diff not in xy_diff:
                self.backtrack(queens + [i], xy_sum + [_sum], xy_diff + [_diff], n, results)
        