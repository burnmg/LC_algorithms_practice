class Solution:
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums)
        return max(self.dp_single_path(nums[1:]), self.dp_single_path(nums[:len(nums)-1]))
    
    def dp_single_path(self, nums):
        
        
        prev_prev_max, prev_max = nums[0], max(nums[0], nums[1])
        cur_max = prev_max
        
        for i in range(2, len(nums)):
            cur_max = max(nums[i] + prev_prev_max, prev_max)
            prev_prev_max = prev_max
            prev_max = cur_max
        
        return prev_max
            
jr