class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        slow = 0
        for fast in range(len(nums)):
            while nums[slow] != 0 and slow < fast:
                slow += 1
            
            if nums[fast] != 0:
                nums[fast], nums[slow] = nums[slow], nums[fast]
