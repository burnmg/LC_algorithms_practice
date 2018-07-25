# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sortedListToBST2(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        if not head:
            return None
        if not head.next:
            return TreeNode(head.val)
        
        slow = fast = head
        prev = None
        
        while fast and fast.next:
            fast = fast.next.next
            prev = slow
            slow = slow.next
        
        root = TreeNode(slow.val)
        if prev:
            prev.next = None
        
        
        # left branch
        left_root = None
        while head:
            new_root = TreeNode(head.val)
            new_root.left = left_root
            left_root = new_root
            head = head.next
        
        # right branch
        right_root = None
        slow = slow.next
        while slow:
            new_root = TreeNode(slow.val)
            new_root.left = right_root
            right_root = new_root
            slow = slow.next
            
        root.left = left_root
        root.right = right_root
        
        return root
        
    def sortedListToBST(self, head):  

        _list = []
        while head:
            _list.append(head.val)
            head = head.next
        
        return self.build_tree(_list)
            
    def build_tree(self, vals):
        
        if len(vals) == 0:
            return None
        
        index = len(vals) // 2
        
        root = TreeNode(vals[index])
        
        root.left = self.build_tree(vals[:index])
        root.right = self.build_tree(vals[index+1:])
        
        return root
