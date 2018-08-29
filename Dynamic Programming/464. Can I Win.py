class Solution(object):
    
    memo = {}
    def canIWin(self, maxChoosableInteger, desiredTotal):
        """
        :type maxChoosableInteger: int
        :type desiredTotal: int
        :rtype: bool
        """
        if (1 + maxChoosableInteger) * maxChoosableInteger/2 < desiredTotal:
            return False
        
        return self.helper(range(1, maxChoosableInteger + 1), desiredTotal)

        
    def helper(self, nums, desiredTotal):
        
        _hash = str(nums)
        if _hash in self.memo:
            return self.memo[_hash]
        
        if nums[-1] >= desiredTotal:
            self.memo[_hash] = True
            return True
        
        for i in range(len(nums)):
            if not self.helper(nums[:i] + nums[i+1:], desiredTotal - nums[i]):
                self.memo[_hash] = True
                return True
        
        self.memo[_hash] = False
        return False