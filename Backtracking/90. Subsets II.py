class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        _, res = self.rec(nums)
        return res
        
    def rec(self, nums):
        
        if len(nums) == 0:
            return 0, [[]]
        
        res = []
        new_items_count_from_last_step, tail = self.rec(nums[1:])
        
        if len(nums) >= 2 and nums[0] == nums[1]:
            count = 0
            for i in range(new_items_count_from_last_step):
                count += 1
                res.append([nums[0]] + tail[i])

            return count, res + tail
        else:
            count = 0
            for x in tail:
                count += 1
                res.append([nums[0]] + x)
            return count, res + tail
            
                
            
            