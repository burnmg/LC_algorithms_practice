class Solution(object):
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        if len(s) > 12:
            return []
        results = []
        self.backtrack(s, 4, '', results)
        return results

    
    def backtrack(self, s, n, result, results):
        if len(s) == 0:
            if n == 0:
                results.append(result[1:])
            return
        
        if len(s) >= 1:
            self.backtrack(s[1:], n-1, result+'.' + s[0], results)
        
        if s[0] == '0':       
            return
        
        if len(s) >= 2:
            self.backtrack(s[2:], n-1, result + '.' + s[:2], results)
        
        if len(s) >= 3 and int(s[:3]) < 256:
            self.backtrack(s[3:], n-1, result + '.' + s[:3], results)
        