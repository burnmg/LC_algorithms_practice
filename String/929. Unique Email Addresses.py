class Solution:
    def numUniqueEmails(self, emails):
        """
        :type emails: List[str]
        :rtype: int
        """
        _hash = set([])
        for email in emails:
            prefix, sufix = email.split("@")
            reduced_prefix = prefix[:prefix.index('+')]
            reduced_prefix_split = reduced_prefix.split(".")
            reduced = "".join(reduced_prefix_split)
            _hash.add((reduced, sufix))
        
        return len(_hash)