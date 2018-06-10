class Solution(object):
    def longestCommonPrefix2(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) == 0:
            return ''
        
        common = min(strs, key=len)
        for i in range(len(strs)):
            j = 0
            while j < len(common) and j < len(strs[i]) and common[j] == strs[i][j]:
                j += 1
            common = common[:j]
            if len(common) == 0: return common
        
        return common
    
    def longestCommonPrefix(self, strs): # better answer
        common = ''
        for x in zip(*strs):
            _set = set(x)
            if len(_set) == 1:
                common += _set.pop()
            if len(common) == 0: return common
        return common
            
            
            
        