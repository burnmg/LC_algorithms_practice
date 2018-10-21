class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        a, b = 0, x+1
        
        while a<b:
            m = (a+b) // 2
            sq = m**2
            if sq == x:
                return m
            elif sq < x:
                a = m + 1
            else:
                b = m
        
        return a-1
        