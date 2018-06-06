class Solution(object):
    def longestPalindrome2(self, s):
        """
        :type s: str
        :rtype: str
        """
        
        max_sub_str = ''
        for i in range(len(new_s)):
            a, b = i, i+1
            length = 1
            
            while a>= 0 and b < len(new_s) and new_s[a] == new_s[b]:
                length += 2
                a -= 1
                b += 1
            if length > len(max_sub_str):
                max_sub_str = length
                max_sub_str = new_s[a+1:b]
        
        return max_sub_str
    
    def longestPalindrome(self, s):
        if len(s) == 0:
            return 0
        
        max_len = 1
        max_end = 0
        for i in range(1, len(s)):
            start = (i - max_len) 
            if s[start:i+1] == s[start:i+1][::-1]:
                max_len += 1
                max_end = i
            elif start >= 1 and s[start-1:i+1] == s[start-1:i+1][::-1]:
                max_len += 2
                max_end = i
                
        return s[max_end-max_len+1:max_end+1]
                
    
    
     