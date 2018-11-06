class Solution(object):
    def shortestPalindrome(self, s):
        A=s+"*"+s[::-1]
        kmp = [0]
        for i in range(1, len(A)):
            previous_i = kmp[i-1]
            
            while(A[previous_i] != A[i] and previous_i > 0 ):
                previous_i = kmp[previous_i-1]
            kmp.append(previous_i+(1 if A[previous_i] == A[i] else 0))
            
        return s[kmp[-1]:][::-1] + s
        