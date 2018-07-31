class Solution(object):
    def isOneEditDistance(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """

        
        if abs(len(s) - len(t)) > 1:
            return False
        if s == t:
            return False
        s, t = (s, t) if len(s) < len(t) else (t, s)
        for i in range(len(s)):
            if s[i] != t[i]:
                return s[i:] == t[i+1:] or s[i+1:] == t[i+1:]
        
        return True
            