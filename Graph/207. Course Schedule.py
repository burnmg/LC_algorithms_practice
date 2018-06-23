class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
    
        graph = [[] for _ in xrange(numCourses)] 
        visited = [0 for _ in xrange(numCourses)] # record visits
        
        for x,y in prerequisites:
            graph[x].append(y)
        
        def dfs(i, visited):
            if visited[i] == -1: # when this node has not been traced back
                return False
            if visited[i] == 1: # when this node has been traced back
                return True
            
            visited[i] = -1
            for out in graph[i]:
                if not dfs(out, visited):
                    return False
            visited[i] = 1
            return True
        
        for i in xrange(numCourses):
            if not dfs(i, visited):
                return False
        
        return True
            
            
            