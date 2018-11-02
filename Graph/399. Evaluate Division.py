class Solution(object):
    def calcEquation(self, equations, values, queries):
        """
        :type equations: List[List[str]]
        :type values: List[float]
        :type queries: List[List[str]]
        :rtype: List[float]
        """

        # build the graph

        # graph: {in: out}

        graph = collections.defaultdict(set)
        for i in range(len(equations)):
            graph[equations[i][0]].add((equations[i][1], values[i]))
            graph[equations[i][1]].add((equations[i][0], 1 / float(values[i])))

        res = []
        for i, (a, b) in enumerate(queries):
            # ["a", "b"]
            if a not in graph or b not in graph:
                res.append(-1.0)
                continue
            stack = [(a, 1.0)]
            visit = set([a])
            while stack:
                x, cumprod = stack.pop()
                if x == b:
                    res.append(cumprod)
                for out, val in graph[x]:
                    if out not in visit:
                        visit.add(out)
                        stack.append((out, cumprod * val))
            if len(res) < i + 1:
                res.append(-1.0)
        return res