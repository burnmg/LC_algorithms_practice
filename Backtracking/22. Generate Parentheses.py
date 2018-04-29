class Solution:
    def generateParenthesis(self, n):
        def recur(n):
            if n==1: return set(["()"])
            
            res = set()
            for x in recur(n-1):
                for i in range(len(x)):
                    res.add(x[:i] + "()" + x[i:])
            
            return res
        return list(recur(n))