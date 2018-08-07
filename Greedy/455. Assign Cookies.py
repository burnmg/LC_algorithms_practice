class Solution(object):
    def findContentChildren(self, g, s):
        """
        :type g: List[int]
        :type s: List[int]
        :rtype: int
        """
        g.sort()
        s.sort()
        
        i = 0
        count = 0
        for cookie in s:
            if i >= len(g):
                break
            if cookie >= g[i]:
                count += 1
                i += 1
        return count
                