class Solution(object):
    def isPerfectSquare(self, num):
        """
        :type num: int
        :rtype: bool
        """
        res = -1
        a,b = 0, num
        while a<=b:
            
            m = (a + b) // 2
            square = m*m
            if square == num:
                return True
            elif square < num:
                a = m + 1
            else:
                b = m - 1
        
        return False 