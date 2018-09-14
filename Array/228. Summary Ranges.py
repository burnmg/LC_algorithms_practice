class Solution:
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        if len(nums) == 0:
            return []
        
        res = []
        _min, _max = nums[0], nums[0]
        
        for i in range(1, len(nums)):
            if nums[i] - nums[i-1] == 1:
                is_range = True
                _max = nums[i]
            else:
                if _min != _max:
                    res.append(str(_min) + '->' + str(_max))
                else:
                    res.append(str(_min))
                _min = _max = nums[i]
        if _min != _max:
            res.append(str(_min) + '->' + str(_max))
        else:
            res.append(str(_min))
        
        return res
        