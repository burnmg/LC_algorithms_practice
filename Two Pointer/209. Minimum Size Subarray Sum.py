class Solution:
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        
        if len(nums) == 0:
            return 0
        
        tail = 0
        min_len = sys.maxsize
        _sum = 0
        found = False
        
        for i in range(len(nums)):
            _sum += nums[i]
            if _sum >= s:
                found = True
                while tail <= i and _sum >= s:
                    _sum -= nums[tail]
                    tail += 1
                    
                min_len = min(i-tail + 2, min_len)
                
        return min_len if _sum >= s else 0x
                    
        
        