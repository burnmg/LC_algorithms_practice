class Solution(object):
    def shortestWordDistance(self, words, word1, word2):
        """
        :type words: List[str]
        :type word1: str
        :type word2: str
        :rtype: int
        """
        if word1 != word2:
            return self.helper(words, word1, word2)
        else:
            index = -len(words)
            min_dist = len(words)
            for i in range(len(words)):
                if words[i] == word1:
                    min_dist = min(min_dist, i - index)
                    index = i
            return min_dist

    def helper(self, words, word1, word2):
        """
        :type words: List[str]
        :type word1: str
        :type word2: str
        :rtype: int
        """
        min_dist = len(words)
        index1, index2 = len(words), len(words)
        
        for i in range(len(words)):
            if words[i] == word1:
                index1 = i
                min_dist = min(min_dist, abs(index1-index2))
            elif words[i] == word2:
                index2 = i
                min_dist = min(min_dist, abs(index1-index2))                
                
        return min_dist
            