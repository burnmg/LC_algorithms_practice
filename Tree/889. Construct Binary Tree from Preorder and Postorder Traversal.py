# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def constructFromPrePost(self, pre, post):
        """
        :type pre: List[int]
        :type post: List[int]
        :rtype: TreeNode
        """
        
        return self.helper(pre, post, 0, len(pre)-1, 0, len(post)-1)
        
    def helper(self, pre, post, pre_start, pre_end, post_start, post_end):

        # pre[pre_start] is the current node
        # if pre_start != post_end, then there is a branch on this current node
        # find post_start that ends at pre_start, send these post to left node
        # use this left sub post's length, get left sub pre

        # rest goes to the right

        if pre_start < pre_end:
            root = TreeNode(pre[pre_start])
            if pre[pre_start + 1] != post[post_end - 1]:
                left_post_end = post.index(pre[pre_start + 1])
                left_post_start = post_start

                left_pre_start = pre_start + 1
                left_pre_end = pre_start + (left_post_end - left_post_start) + 1

                right_pre_start = left_pre_end + 1
                right_pre_end = pre_end

                right_post_start = left_post_end + 1
                right_post_end = post_end - 1

                root.left = self.helper(pre, post, left_pre_start, left_pre_end, left_post_start, left_post_end)
                root.right = self.helper(pre, post, right_pre_start, right_pre_end, right_post_start, right_post_end)

                return root
            else:
                root.left = self.helper(pre, post, pre_start + 1, pre_end, post_start, post_end - 1)
                return root

        else:
            return TreeNode(pre[pre_start])
        