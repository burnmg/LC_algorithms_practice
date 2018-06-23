class Solution(object):
    def findMinHeightTrees(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        if n == 1:
            return [0]
        neighbours = collections.defaultdict(list)
        degrees = collections.defaultdict(int)
        for u,v in edges:
            neighbours[u].append(v)
            neighbours[v].append(u)
            degrees[u] += 1
            degrees[v] +=1
        
        # find leaves
        prelevel = []
        for u in degrees:
            if degrees[u] == 1:
                prelevel.append(u)
        
        visited = set(prelevel)
        i = 0
        while len(visited) < n:
            thislevel = []
            for u in prelevel:
                for nei in neighbours[u]:
                    if nei not in visited:
                        degrees[nei] -= 1
                        if degrees[nei] == 1:
                            thislevel.append(nei)
                            visited.add(nei)
            prelevel = thislevel
            i += 1
        
        return prelevel
                