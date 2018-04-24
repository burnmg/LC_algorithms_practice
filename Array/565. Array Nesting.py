class Solution(object):
    def arrayNesting(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        unvisited = set(range(0, len(nums)))
        max_l = 1
        while len(unvisited) > 0:
            start = unvisited.pop()
            next = nums[start]
            l = 1
            while next != start:
                unvisited.remove(next)
                next = nums[next]
                l += 1
                max_l = max(l, max_l)

        return max_l
        