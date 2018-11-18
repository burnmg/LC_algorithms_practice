class Solution(object):
    def beautifulArray(self, N):
        """
        :type N: int
        :rtype: List[int]
        """
        return self.helper(list(range(1,N+1)))
        
    def helper(self, lst):
        if len(lst)<=2:         
            return lst
        return self.helper(lst[::2]) + self.helper(lst[1::2])