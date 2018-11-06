class Solution:
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        dp = [False] * len(s)
        
        for i in range(len(s)):
            for w in wordDict:
                if len(w) <= i + 1 and (dp[i-len(w)] or i-len(w) == -1) and s[i-len(w)+1:i+1] == w:
                    dp[i] = True
            
            
        return dp[-1]
        

    def wordBreak(self, s, wordDict): # my second solution
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        starts = [0]
        word_set = set(wordDict)
        for i in range(len(s)):
            if any(s[j:i+1] in word_set for j in starts):
                starts.append(i+1)
        return starts[-1] == len(s)