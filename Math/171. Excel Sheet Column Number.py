class Solution(object):
    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """
        factor = 1
        res = 0
        for c in reversed(s):
            res += (ord(c) - 65 + 1) * factor
            factor *= 26
        return res
        