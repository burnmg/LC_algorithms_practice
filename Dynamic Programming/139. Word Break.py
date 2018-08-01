class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        if len(wordDict) == 0:
            return False
        word_dict = set(wordDict)
        truths = [-1]

        for i in range(len(s)):
            for j in truths:
                if s[j+1:i+1] in word_dict:
                    truths.append(i)
                    break
      
        if len(truths) == 1:
            return False
        return truths[-1] == len(s) - 1