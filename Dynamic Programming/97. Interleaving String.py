class Solution(object):
    def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """
        if len(s1) + len(s2) != len(s3):
            return False
        
        stack = [(0,0)]
        visited = set([(0,0)])
        
        while stack:
            coo = stack.pop() # coordinate
            
            if coo[0] == len(s1) and coo[1] == len(s2):
                return True      
            
            down = (coo[0]+1, coo[1]) 
            if down[0]-1 < len(s1) and down not in visited and s1[down[0]-1] == s3[down[0] + down[1] -1]: # move rightward on the matrix. Use '-1' converts the matrix coordiante to string's index.
                visited.add(down)
                stack.append(down)
            
            right = (coo[0], coo[1]+1)
            if right[1]-1 < len(s2) and right not in visited and s2[right[1]-1] == s3[right[0] + right[1] -1]: # move downward on the matrix 
                visited.add(right)
                stack.append(right)
        
        return False