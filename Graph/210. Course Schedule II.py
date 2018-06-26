class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        
        in_degrees = [0 for _ in xrange(numCourses)]
        neighbours = [[] for _ in xrange(numCourses)]
        
        for x,y in prerequisites:
            neighbours[y].append(x)
            in_degrees[x] += 1
        
        zero_in_degree_nodes = []
        
        for i, in_degree in enumerate(in_degrees):
            if in_degree == 0:
                zero_in_degree_nodes.append(i)
        
        res = []
        count = 0
        while zero_in_degree_nodes:
            x = zero_in_degree_nodes.pop()
            res.append(x)
            
            for neighbour in neighbours[x]:
                in_degrees[neighbour] -= 1
                if in_degrees[neighbour] == 0:
                    zero_in_degree_nodes.append(neighbour)
            count += 1
            
        if count == numCourses:
            return res
        else:
            return []
            
       