class Solution:
    def largestDivisibleSubset(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if len(nums) == 0:
            return []
        
        dp = []
        nums.sort()
        
        for n in nums:
            cur_set = [n]
            for _set in dp:
                if n % _set[-1] == 0 and len(_set) + 1 > len(cur_set):
                    cur_set = _set + [n]
            
            dp.append(cur_set)
        
        return max(dp, key = lambda x: len(x))
                    