class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def per(nums):
            if len(nums) == 1: return [[nums[0]]]
            res = []
            for i,x in enumerate(nums):
                for s in per(nums[:i]+nums[i+1:]):
                    res += [[x] + s]
            return res
        return per(nums)
        