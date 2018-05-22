class Solution(object):
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        
        letters='abcdefghijklmnopqrstuvwxyz'
        index = []
        for l in letters:
            if s.count(l) == 1:
                index += [s.index(l)]
        return min(index) if len(index) > 0 else -1