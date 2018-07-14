class Solution(object):
    def convertToTitle(self, n):
        """
        :type n: int
        :rtype: str
        """
        
        res = ''
        while n > 0:
            res += chr((n-1) % 26+65)
            n = (n-1) // 26
        
        return res[::-1]