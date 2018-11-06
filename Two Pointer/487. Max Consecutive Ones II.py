class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        slow = 0
        zero_count = 0
        max_len = 0
        for i in range(len(nums)):
            if nums[i] == 0:
                zero_count += 1
                while zero_count > 1 and slow <= i:
                    if nums[slow] == 0:
                        zero_count -= 1
                    slow += 1
            
            max_len = max(i-slow+1, max_len)
        return max_len