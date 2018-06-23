
class Solution(object):
    def findCircleNum2(self, M):
        """
        :type M: List[List[int]]
        :rtype: int
        """
        visited = set()
        circle_count = 0
        for i in range(len(M)):
            if i not in visited:
                stack = [i]
                visited.add(i)
                while stack:
                    x = stack.pop()
                    for j in range(len(M[x])):
                        if M[x][j] == 1 and j not in visited:
                            stack.append(j)
                            visited.add(j)
                circle_count += 1
                
        return circle_count
    
    def findCircleNum(self, M): # recursive DFS
        
        visited = set()
        def rec(root):
            for i in range(len(M[root])):
                if M[root][i] and i not in visited:
                    visited.add(i)
                    rec(i)
                    
        count = 0
        
        for i in range(len(M)):
            if i not in visited:
                visited.add(i)
                count +=1
                rec(i)
        return count
                
    
        
                             