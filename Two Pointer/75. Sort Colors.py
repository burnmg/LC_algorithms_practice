class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        
        j = len(nums) - 1
        for i in range(len(nums)):
            if nums[i] == 0:
                continue
            else:
                while i < j and nums[j] != 0:
                    j -= 1
                if j == i:
                    break
                else:
                    nums[i], nums[j] = nums[j], nums[i]
        
        i = 0
        for j in range(len(nums) - 1 , -1, -1):
            if nums[j] == 2:
                continue
            else:
                while i < j and nums[i] != 2:
                    i += 1
                if i == j:
                    break
                else:
                    nums[i], nums[j] = nums[j], nums[i]
               
            