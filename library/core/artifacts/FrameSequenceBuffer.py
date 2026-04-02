from queue import Queue, Full, Empty
from library.core.artifacts.FrameSequence import FrameSequence

class FrameSequenceBuffer:

    #TODO: da rendere iterable, cosi lo puoi chiamare con "for frame_sequence in buffer"

    def __init__(self, buffer_size: int):
        self._queue = Queue(buffer_size)

    def put(self, frame_sequence: FrameSequence, timeout=None):
        self._queue.put(frame_sequence, timeout=timeout)

    def get(self, timeout=None) -> FrameSequence:
        return self._queue.get(timeout=timeout)

    def size(self):
        return self._queue.qsize()



