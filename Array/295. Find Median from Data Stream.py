class MedianFinder(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.nums = []
        self.median = 0
        

    def addNum(self, num):
        """
        :type num: int
        :rtype: void
        """
        
        # use BS to find the place to insert the num
        self.nums.insert(bisect.bisect_left(self.nums, num), num)
        
        if len(self.nums) == 1:
            self.median = num
        else:
            # print(len(self.nums) // 2, len(self.nums) // 2 - 1)
            self.median = float(self.nums[len(self.nums) // 2]) if len(self.nums) % 2 != 0 else float(self.nums[len(self.nums) // 2] + self.nums[len(self.nums) // 2 - 1]) / 2
        
        

    def findMedian(self):
        """
        :rtype: float
        """
        return self.median
        
        
        


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()