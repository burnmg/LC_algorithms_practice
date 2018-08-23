class Solution(object):
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [1 for _ in range(n+1)]
        
        for i in range(2, n+1):
            _max_val = -1
            for j in range(1, i):
                _max_val = max(_max_val, max(dp[j], j) * max(dp[i-j], i-j))
            dp[i] = _max_val
        return dp[-1]