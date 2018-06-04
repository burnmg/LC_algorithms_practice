class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        
        abs_x = abs(x)
        res = 0
        while abs_x > 0:
            new_x = abs_x // 10
            res = res * 10 + abs_x - new_x * 10
            if res > 2147483647: return 0
            abs_x = new_x    
            
        
        return res if x>=0 else res * -1