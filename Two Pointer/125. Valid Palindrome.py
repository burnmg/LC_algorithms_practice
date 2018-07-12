class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        s = s.lower()
        
        a, b = 0, len(s) - 1
        alphabet = 'abcdefghijklm'
        
        while a < b:
            while not s[a].isalnum() and a < b:
                a += 1
            while not s[b].isalnum() and a < b:
                b -= 1
            
            if a < b and s[a] != s[b]:
                return False
            a += 1
            b -= 1
        
        return True
                
                
            
        