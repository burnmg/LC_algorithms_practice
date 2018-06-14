class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        a,b = 0, len(nums)
        
        min_val = float('inf')
        while a<b:
            m = (a+b) // 2
            
            if nums[m] < nums[b-1]:
                min_val = min(min_val, nums[m])
                b = m
            else:
                min_val = min(min_val, nums[a], nums[m])
                a = m + 1
        return min_val