class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s) == 0:
            return len(s)

        slow,fast = 0,0
        _hash = {}
        max_len = 1
        while fast < len(s):
            if s[fast] in _hash and slow <= _hash[s[fast]]:
                slow = _hash[s[fast]] + 1
            else:
                max_len = max(max_len, fast - slow + 1)
            _hash[s[fast]] = fast
            fast += 1
  
        return max_len