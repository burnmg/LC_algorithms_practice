class Solution:
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        if len(nums) == 1:
            return 0
        
        a, b = 0, len(nums)
        
        while a < b:
            mid = (a+b) // 2
            if mid == 0:
                if nums[mid] > nums[mid+1]:
                    return mid
                else:
                    a = mid + 1
            elif mid == len(nums) - 1:
                if nums[mid] > nums[mid-1]:
                    return mid
                else:
                    b = mid
            else:
                if nums[mid-1] < nums[mid]  > nums[mid+1]:
                    return mid
                if nums[mid+1] > nums[mid]:
                    a = mid + 1
                else:
                    b = mid
        
                
        