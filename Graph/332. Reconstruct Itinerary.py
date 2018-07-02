class Solution(object):

    def findItinerary(self, tickets):
        """
        :type tickets: List[List[str]]
        :rtype: List[str]
        """
        tickets_dict = collections.defaultdict(list)
        for x, y in tickets:
            tickets_dict[x].append([False, y])  # [is_used, destination]
        for x in tickets_dict:
            tickets_dict[x].sort(key=lambda x: x[1])

        results = []

        self.backtrack('JFK', tickets_dict, ['JFK'], results, len(tickets))

        return results[0]


    def backtrack(self, start, tickets, result, results, tickets_num):

        if len(result) == tickets_num + 1:
            results.append(result)
            return

        for ticket in tickets[start]:
            if ticket[0]:  # if the ticket has been used
                continue
            ticket[0] = True
            self.backtrack(ticket[1], tickets, result + [ticket[1]], results, tickets_num)
            if len(results) > 0:
                return

            ticket[0] = False  # backtrack, reset the ticket
        return

