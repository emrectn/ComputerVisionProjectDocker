class FolderStructureError(Exception):

    # Raised when an operation attempts a state
    # transition that's not allowed.
    def __init__(self, msg):
        # Error message thrown is saved in msg
        self.msg = msg
    def getMessage(self):
        return self.msg

