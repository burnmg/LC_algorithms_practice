class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        lowest_price_by_now = float('inf')
        profit = 0
        for x in prices:
            lowest_price_by_now = min(lowest_price_by_now, x)
            if x > lowest_price_by_now:
                profit += x - lowest_price_by_now
                lowest_price_by_now = x
        return profit