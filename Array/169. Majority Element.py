class Solution:
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        d = collections.Counter(nums)
        threshold = len(nums) // 2
        for key in d:
            if d[key] > threshold:
                return key
        