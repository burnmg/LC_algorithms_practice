class Solution(object):
    def totalHammingDistance(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return sum((_tuple.count('0') * _tuple.count('1')) for _tuple in zip(*map(lambda x: '{0:032b}'.format(x), nums)))
        