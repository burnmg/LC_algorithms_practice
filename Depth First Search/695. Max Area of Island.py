class Solution(object):
    def maxAreaOfIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        visited = set([])
        max_area = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if (i, j) in visited or grid[i][j] != 1:
                    continue
                stack = [(i, j)]
                visited.add((i, j))
                area = 0
                while stack:
                    x, y = stack.pop()
                    area += 1
                    if (x - 1, y) not in visited and x - 1 >= 0 and grid[x - 1][y] == 1:
                        stack.append((x - 1, y))
                        visited.add((x - 1, y))
                    if (x + 1, y) not in visited and x + 1 < len(grid) and grid[x + 1][y] == 1:
                        stack.append((x + 1, y))
                        visited.add((x + 1, y))
                    if (x, y - 1) not in visited and y - 1 >= 0 and grid[x][y - 1] == 1:
                        stack.append((x, y - 1))
                        visited.add((x, y - 1))
                    if (x, y + 1) not in visited and y + 1 < len(grid[0]) and grid[x][y + 1] == 1:
                        stack.append((x, y + 1))
                        visited.add((x, y + 1))
                max_area = max(area, max_area)

        return max_area