import numpy as np

FRAMES_PER_SECOND = 30 # Number of frames in a second
FRAME_HEIGHT = 270
FRAME_WIDTH = 480
FRAME_DEPTH = 3

class Buffer:
    def __init__(self, max_seconds=10):
        self.max_size = max_seconds
        self.buffer = []
        self.dequeued = 0

    def __len__(self):
        return len(self.buffer)

        
    def pop(self):
        secondOfFrames = np.copy(self.buffer.pop(0))
        return secondOfFrames

    def peek(self):
        if self.buffer:
            return self.buffer[0]
        return None

    def add(self, data):
        newData = np.copy(data)
        self.buffer.append(newData)

    def clear(self):
        self.buffer.clear()
        self.buffer = []
