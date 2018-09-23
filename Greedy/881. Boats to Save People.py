class Solution:
    def numRescueBoats(self, people, limit):
        """
        :type people: List[int]
        :type limit: int
        :rtype: int
        """
        people.sort()
        
        boat = 0
        
        i, j = 0, len(people) - 1
        
        while i <= j:
            if people[i] + people[j] <= limit:
                i += 1
                j -= 1
            else:
                j -= 1
            boat += 1
            
        return boat                