class Solution(object):
    
    _memo = {}
    def canWin(self, s):
        def helper(s):
            memo = self._memo
            if s in memo:
                return memo[s]
            
            memo[s] = any(s[i:i+2] == '++' and not self.canWin(s[:i] + '--' + s[i+2:]) for i in range(len(s)))
            return memo[s]
        return helper(s)