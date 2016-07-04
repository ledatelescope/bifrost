"""@package block
This file defines a generic block class.

Right now the only possible block type is one
of a simple transform which works on a span by span basis.
"""

class Block(object):
    """Defines things which are needed by all types of blocks"""
    def __init__(self):
        self.inputs = []


class TransformBlock(Block):
    """Defines the structure for a transform block"""
    def __init__(self):
        super(TransformBlock, self).__init__()
