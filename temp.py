class Solution(object):
    def validTree(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: bool
        """
        
        neighbors = collections.defaultdict(list)
        
        for x, y in edges:
            neighbors[x].append(y)
            neighbors[y].append(x)
        
        visited = [0 for _ in range(n)]
        stack = [edge[0][0]]
        visited[None, edge[0][0]] = -1
        
        while stack:
            parent, x = stack.pop()

            no_unvisited_neighbor = True
            for nei in neighbors[x]:
                if visited[nei] == 0:
                    stack.append([x, nei])
                    visited[nei] == -1
                    no_unvisited_neighbor = False
                elif visited[nei] == -1 and visited[nei] != parent:
                    return False
            if no_unvisited_neighbor:
                visited[x] = 1

        return True