class WordDistance(object):

    def __init__(self, words):
        """
        :type words: List[str]
        """
        self.words = words
        self.memo = {}
        self.indices = collections.defaultdict(list)
        for i, x in enumerate(words):
            self.indices[x].append(i)
        


    def shortest(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        min_dist = len(self.words)
        indices1, indices2 = self.indices[word1], self.indices[word2]
        i, j = 0, 0
        while i < len(indices1) and j < len(indices2):
            min_dist = min(min_dist, abs(indices1[i]-indices2[j]))
            if indices1[i] < indices2[j]:
                i += 1
            else:
                j += 1

        return min_dist
        


# Your WordDistance object will be instantiated and called as such:
# obj = WordDistance(words)
# param_1 = obj.shortest(word1,word2)