class Solution(object):
    def judgeCircle(self, moves):
        """
        :type moves: str
        :rtype: bool
        """
        pos = [0,0]
        for x in moves:
            if x == 'U':
                pos[0] += 1
            elif x == 'D':
                pos[0] -= 1
            elif x == 'L':
                pos[1] -= 1
            else:
                pos[1] += 1
        
        return pos == [0,0]