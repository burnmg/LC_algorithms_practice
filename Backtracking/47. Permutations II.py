class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def recur(nums):
            if len(nums) == 1: return [nums]
            dup = set()
            res = []
            for i in range(len(nums)):
                if nums[i] in dup:
                    continue
                dup.add(nums[i])
                for x in recur(nums[:i] + nums[i+1:]):
                    res.append([nums[i]] + x)
            
            return res
            
        return recur(nums)