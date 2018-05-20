class Solution(object):
    def reverseWords2(self, s):
        """
        :type s: str
        :rtype: str
        """
        sl = s.split(' ')
        res = []
        for word in sl:
            res += list(word)[::-1] + [' ']
        
        return "".join(res[:-1])
    
    def reverseWords(self, s): # One line version, faster. 
        return ' '.join(x[::-1] for x in s.split())