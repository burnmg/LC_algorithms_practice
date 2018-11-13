import collections


class Solution(object):
    def wordSquares(self, words):
        """
        :type words: List[str]
        :rtype: List[List[str]]
        """
        _dict = collections.defaultdict(list)
        for w in words:
            for i in range(len(w)):
                _dict[w[:i]].append(w)

        res = []
        self.backtrack(_dict, [], res, len(words[0]))

        return res

    def backtrack(self, _dict, result, results, n):

        if len(result) == n:
            results.append(result[:])
            return

        match = []
        for i in range(len(result)):
            match.append(result[i][len(result)])

        for new_word in _dict["".join(match)]:
            result.append(new_word)
            self.backtrack(_dict, result, results, n)
            result.pop()
