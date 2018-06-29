class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        results = []
        self.backtrack(s, [], results)
        return results
    
    def backtrack(self, s, result, results):

        if not s:
            results.append(result)
            return
        for i in range(1, len(s)+1):
            if s[:i]== s[:i][::-1]:
                self.backtrack(s[i:], result + [s[:i]], results)
        
        