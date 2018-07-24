class Solution(object):
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        res = []
        self.backtracking(k, n, 1, [], res)
            
        return res
    
    def backtracking(self, k, n, start, result, results):
    
        if k == 0:
            if n == 0:
                results.append(result)
            return
        
        for x in range(start, 10):
            if n >= x:
                self.backtracking(k-1, n-x, x+1, result + [x], results)
            else:
                break
    
            
            