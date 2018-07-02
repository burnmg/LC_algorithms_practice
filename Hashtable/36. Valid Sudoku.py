class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        for i in range(len(board)):
            _hash = set()
            for j in range(len(board[i])):
                if board[i][j] != '.':
                    if board[i][j] in _hash:
                        print(_hash)
                        return False
                    else:
                        _hash.add(board[i][j])  
        
        for j in range(len(board[0])):
            _hash = set()
            for i in range(len(board)):
                if board[i][j] != '.':
                    
                    if board[i][j] in _hash:
                        print(_hash)
                        return False
                    else:
                        _hash.add(board[i][j])     
        
        corners = [(i, j) for i in range(0, 9, 3) for j in range(0, 9, 3)]
        
        for x,y in corners:
            _hash = set()
            for i in range(x, x+3):
                for j in range(y, y+3):
                    if board[i][j] != '.':
                        if board[i][j] in _hash:
                            return False
                        else:
                            _hash.add(board[i][j])                    
        
        return True
            