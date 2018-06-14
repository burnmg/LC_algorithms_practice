class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        a, b = 0, len(nums)
        
        while a<b:
            m = (a+b) // 2
            if nums[m] == target:
                return True
            
            right_is_sorted = True
            
            if nums[m] == nums[b-1]:
                i = m+1
                while i < b:
                    if nums[i] > nums[m]:
                        right_is_sorted = False
                        break
                    i += 1
            elif nums[m] > nums[b-1]:
                right_is_sorted = False
            
            if right_is_sorted:
                if nums[m] < target <= nums[b-1]:
                    a = m + 1
                else:
                    b = m
            else:
                if nums[a] <= target <= nums[m-1]:
                    b = m
                else:
                    a = m + 1                    
        
        return False