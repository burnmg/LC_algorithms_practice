class Solution(object):
    def grayCode(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        
        if n == 0:
            return [0]
        prev = self.grayCode(n-1)
        
        res = []
        for x in prev[::-1]:
            res.append(2**(n-1)+x)
        
        return prev + res
            