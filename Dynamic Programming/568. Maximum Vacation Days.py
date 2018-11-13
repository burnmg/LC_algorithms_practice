class Solution(object):
    def maxVacationDays(self, flights, days):
        """
        :type flights: List[List[int]]
        :type days: List[List[int]]
        :rtype: int
        """
        # preprocess the flights
        for i in range(len(flights)):
            flights[i][i] = 1

        cur_vocations = [days[i][0] if flights[0][i] != 0 else float('-inf') for i in range(len(days))]
        longest_voc = max(cur_vocations)

        for i in range(1, len(days[0])):  # for each day
            new_vocations = []
            for j in range(len(flights)):  # for each arriving place. j is the new place. k is the previous place
                new_vocations.append(max(
                    [days[j][i] + cur_vocations[k] if flights[k][j] == 1 else float('-inf') for k in range(len(flights))]))
                longest_voc = max(longest_voc, new_vocations[j])
            cur_vocations = new_vocations
        return longest_voc

