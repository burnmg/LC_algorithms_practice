class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s[0] == '0':
            return 0
        if len(s) == 1 or len(s) == 0:
            return len(s)
        
        prev_res = 1
        if s[1] == '0':
            if s[0] == '1' or s[0] == '2':
                prev_res = 1
            else:
                return 0
        elif int(s[0:2]) < 27:
            prev_res = 2
        
        prev_prev_res = 1
        
        for i in range(2, len(s)):
            cur_res = 0
            if s[i] == '0':
                if s[i-1] == '1' or s[i-1] == '2':
                    cur_res = prev_prev_res
                else:
                    return 0
            elif s[i-1] == '0':
                cur_res = prev_res
            elif int(s[i-1:i+1]) <= 26:
                cur_res = prev_res + prev_prev_res
            else:
                cur_res = prev_res
            
            prev_prev_res = prev_res
            prev_res = cur_res
            
        return prev_res
            