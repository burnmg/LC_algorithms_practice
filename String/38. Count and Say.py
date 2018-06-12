class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        if n == 1:
            return '1'
        s = '1'
        for x in range(n-1):
            cur_digit = s[0]
            cur_count = 1
            new_s = []
            for i in range(1, len(s)):
                if s[i] != cur_digit:
                    new_s += [str(cur_count)] + [str(cur_digit)]
                    cur_count = 1
                    cur_digit = s[i]
                else:
                    cur_count += 1
            new_s += [str(cur_count)] + [str(cur_digit)]
            s = ''.join(new_s)
        return s