class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        res = []
        candidates.sort()
        self.helper(candidates, target, 0, [], res)
        return res
    
    def helper(self, nums, target, start, result, results):
        
        if target == 0:
            results.append(result)
        
        for i in range(start, len(nums)):
            
            if i > start and nums[i-1] == nums[i]:
                continue
            
            if nums[i] > target: # cut the search space
                return
            
            self.helper(nums, target-nums[i], i + 1, result + [nums[i]], results)
    
        

            
        
        
        
        