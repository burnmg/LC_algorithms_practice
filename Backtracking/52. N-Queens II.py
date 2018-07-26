class Solution(object):
    def totalNQueens(self, n):
        """
        :type n: int
        :rtype: int
        """
    
        res = [0]
        
        self.backtrack(set(), n, set(), set(), res)
        return res[0]
    
    def backtrack(self, queens, n, xy_sums, xy_diffs, res):
        
        if len(queens) == n:
            res[0] += 1
        
        cur_row_index = len(queens)
        for i in range(n):
            _sum = i+cur_row_index
            _diff = i-cur_row_index
            if i not in queens and _sum not in xy_sums and _diff not in xy_diffs:
                self.backtrack(queens | set([i]), n, xy_sums | set([_sum]), xy_diffs | set([_diff]), res)
        
                
            