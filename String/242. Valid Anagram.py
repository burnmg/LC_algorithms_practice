class Solution(object):
    def isAnagram2(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        
        return collections.Counter(s) == collections.Counter(t)
    
    def isAnagram(self, s, t):
            return all([s.count(c)==t.count(c) for c in string.ascii_lowercase])