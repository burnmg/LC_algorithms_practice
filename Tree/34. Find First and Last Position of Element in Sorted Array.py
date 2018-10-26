class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        left, right = self.helper(nums, target, 0, len(nums))
        return [left, right]
    def helper(self, nums, target, start, end):
        
        """
        return left, right
        """
        if start >= end:
            return -1, -1
        m = (start + end) // 2
        if nums[m] == target:
            if (m - 1 >= 0 and nums[m-1] != nums[m]) or m == 0:
                left = m
            else:
                left,_ = self.helper(nums, target, start, m)  
            if (m + 1 < end and nums[m+1] != nums[m]) or m == end - 1:
                right = m
            else:
                _,right = self.helper(nums, target, m+1, end) 
            return left, right
        
        elif nums[m] > target:
            return self.helper(nums, target, start, m)
        else:
            return self.helper(nums, target, m+1, end)
            
        
        
                