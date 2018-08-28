class Solution(object):
    def countNumbersWithUniqueDigits(self, n):
        """
        :type n: int
        :rtype: int
        """
        
        n = 10 if n > 10 else n
        
        l = [9, 9, 8, 7, 6 ,5 ,4, 3, 2, 1]
        
        pro = 1
        _sum = 1
        for i in range(n):
            pro *= l[i]
            _sum += pro
            
        return _sum