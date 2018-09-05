class Solution:
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # special case TODO
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0], nums[1])
        
        rob_max = [0] * len(nums)
        rob_max[0], rob_max[1] = nums[0], max(nums[0], nums[1])
        _max = rob_max[1]
        
        for i in range(2, len(rob_max)):
            rob_max[i] = max(nums[i] + rob_max[i-2], rob_max[i-1])
            _max = max(_max, rob_max[i])
            
        return _max
        