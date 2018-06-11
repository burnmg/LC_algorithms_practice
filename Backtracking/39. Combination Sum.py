class Solution(object):
    def combinationSum(self, candidates, target):
        candidates.sort()
        res = []
        self.backtracking(candidates, 0, target, [], res)
        return res
        
    def backtracking(self, candidates, start, target, result, results):
        
        if target < 0:
            return 
        elif target == 0:
            results.append(result)
            return
        else:
            for i in range(start, len(candidates)):
                self.backtracking(candidates, i, target-candidates[i], result+[candidates[i]], results)
    
                    
                    