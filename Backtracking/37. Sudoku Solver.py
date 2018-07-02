class Solution(object):
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        
        solution = None
        self.backtrack(board, collections.defaultdict(set),collections.defaultdict(set),collections.defaultdict(set))
        return solution
    
    def backtrack(self, board, square, hor, ver, solution):
        
        x = y = -1
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    x,y = i,j
        if x == -1: # not finding any empty space
            solution = board
            
        for num in str(range(1, 10)):
            corner_x, corner_y = x // 3, y // 3
            if num not in square[(corner_x,corner_y)] and num not in hor[x] and num not in ver[y]:
                board[x][y] = num
                square[(corner_x,corner_y)].add(num)
                hor[x].add(num)
                ver[y].add(num)
                self.backtrack(board, square, hor, ver, solution)
                if solution:
                    return
                
                board[x][y] = '.'
                square[(corner_x,corner_y)].remove(num) # backtrack
                hor[x].remove(num)
                ver[y].remove(num)
                
        return
        