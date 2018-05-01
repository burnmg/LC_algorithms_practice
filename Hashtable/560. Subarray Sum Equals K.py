class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        cur_sum = 0
        count = 0
        _hash = {0:1}
        for i in range(len(nums)):
            cur_sum += nums[i]
            diff = cur_sum - k
            if diff in _hash:
                count += _hash[diff]
                
            if cur_sum in _hash:
                _hash[cur_sum] += 1
            else:
                _hash[cur_sum] = 1
                
        return count