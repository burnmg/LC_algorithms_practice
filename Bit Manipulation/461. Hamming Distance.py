class Solution(object):
    def hammingDistance(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        return sum(map(lambda x:int(x), '{0:b}'.format(x ^ y)))