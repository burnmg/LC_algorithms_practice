
class Solution(object):
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        if num == 0: 
            return [0]
        elif num == 1: 
            return [0,1]
        
        res= [0] * (num+1)
        res[0],res[1] = 0,1
        
        square = 1
        for i in range(2, num+1):
            if i == square * 2:
                square *= 2
            
            res[i] = res[i-square] + 1
            
        return res
                
            