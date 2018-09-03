class Solution:
    def wiggleSort(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        if len(nums) == 0 or len(nums) == 1:
            return
        
        mid = 1
        while mid < len(nums) - 1:
            max_i = self.argmax(nums, mid-1, mid+2)
            nums[mid], nums[max_i] = nums[max_i], nums[mid]
            mid += 2
        
        if mid == len(nums) -1 :
            max_i = self.argmax(nums, mid-1, mid+1)
            nums[mid], nums[max_i] = nums[max_i], nums[mid]
        
    
    def argmax(self, nums, a, b):
        
        temp = nums[a:b]
        _max = max(temp)
        return a + temp.index(_max)
    
        