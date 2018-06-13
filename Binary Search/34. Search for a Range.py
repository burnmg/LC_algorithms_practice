class Solution(object):
    def searchRange2(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
    
        def recur(nums, target):
            found = False
            a, b = 0, len(nums) - 1
            pos = -1
            while a<=b:
                m = (a + b) // 2
                if nums[m] == target:
                    pos = m
                    break
                elif nums[m] > target:
                    b = m-1
                else:
                    a = m+1
            
            if pos != -1:
                left = recur(nums[:pos], target)
                
                if left[0] != -1:
                    left_pos = left[0]
                else:
                    left_pos = pos
                
                right = recur(nums[pos+1:], target)
                if right[0] != -1:
                    right_pos = pos+right[1]+1
                else:
                    right_pos = pos
                return [left_pos, right_pos]
            
            else:
                return [-1, -1]
        
        return recur(nums, target)
    
    def searchRange(self, nums, target):
        
        if len(nums) == 0:
            return [-1, -1]
        def findLeftBound(nums, target):
            
            a,b = 0,len(nums)
            
            while a<b:
                m = (a + b) // 2
                if nums[m] >= target:
                    b = m
                else:
                    a = m + 1
            
            if a >= 0 and a < len(nums) and nums[a] == target:
                return a
            else:
                return -1

        def findRightBound(nums, target):
            
            a,b = 0, len(nums)
            
            while a<b:
                m = (a+b) // 2
                if nums[m] > target:
                    b = m
                else:
                    a = m + 1
            
            return a
        
        left = findLeftBound(nums, target)
        if left != -1:
            return [left, left + findRightBound(nums[left:], target)-1]
        else:
            return [-1, -1]
            
                    
                
                    
                
            