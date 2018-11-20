class Solution:
    def dailyTemperatures(self, T):
        """
        :type T: List[int]
        :rtype: List[int]
        """
        
        if len(T) == 1:
            return [0]
        
        res = [[0 for _ in range(len(T))]]
        
        stack = []
        
        stack.append(len(T)-1)
        for i in range(len(T)-2, -1, -1):
            while len(stack) > 0 and T[i] >= T[stack[-1]]:
                stack.pop()
            
            if len(stack) > 0:
                res[i] = stack[-1] - i
            stack.append(i)
            
        return res
    
        