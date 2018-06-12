class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        s = str.strip(str(s))
        splited = s.split(' ')
        
        if len(splited) == 0:
            return 0
        else:
            return len(splited[-1])