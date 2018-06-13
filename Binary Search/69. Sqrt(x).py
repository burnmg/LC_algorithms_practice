class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x == 1:
            return 1
        
        a,b = 1,x
        while a<b:
            m = (a+b)//2
            square = m*m
            if square == x:
                return m
            elif square < x:
                a = m + 1
            else:
                b = m
        
        return a-1