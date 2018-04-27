class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        d = {}
        for i,x in enumerate(nums):
            if x in d and i-d[x] <= k:
                    return True
            d[x] = i
        
        return False