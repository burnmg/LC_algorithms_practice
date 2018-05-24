class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        d = {}
        for x in strs:
            sorted_x = "".join(sorted(x))
            if sorted_x in d:
                d[sorted_x].append(x)
            else:
                d[sorted_x] = [x]
        
        return d.values()