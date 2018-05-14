class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        nums.sort()
        results = []
        self.NSum(nums, target, 4, [], results)
        
        return results
        
    def NSum(self, nums, target, N, result, results):
        if len(nums) < N or N < 2:
            return
        
        if N == 2:
            l,r = 0, len(nums) - 1
            while l < r:
                s = nums[l] + nums[r] 
                if s == target:
                    
                    results.append(result + [nums[l], nums[r]])
                    r -= 1
                    l += 1
                    while l < r and nums[l] == nums[l-1]: # Remove the replicate results
                        l += 1 
                    while l < r and nums[r] == nums[r+1]:  # Remove the replicate results
                        r -= 1
                elif s < target:
                    l += 1
                else:
                    r -= 1
            
        else:
            for i in range(len(nums)-N+1):
                if target < N*nums[i] or target > N*nums[-1]: # Avoid unnecessary search. 
                    break
                if i == 0 or i > 0 and nums[i-1] != nums[i]:
                    self.NSum(nums[i+1:], target-nums[i], N-1, result+[nums[i]], results)
                    
        return 
                    
                
            