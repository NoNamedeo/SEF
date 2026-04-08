from queue import Queue

from library.core.artifacts.Frame import Frame

class FrameBuffer:

    #TODO: non ci fidiamo di python: probabilmente get e put non gestiscono il fatto che sia pieno/vuoto

    def __init__(self, buffer_size: int):
        self._queue = Queue(buffer_size)
        self.closed = False

    def put(self, frame: Frame, timeout = None):
        self._queue.put(frame, timeout = timeout)

    def get(self, timeout = None) -> Frame:
        return self._queue.get(timeout = timeout)

    def close(self):
        self.closed = True

    def is_empty(self) -> bool:
        return self._queue.empty()

    def size(self):
        return self._queue.qsize()

    def __iter__(self):
        while not (self.closed and self.is_empty()):
            frame = self.get(timeout = 10)
            yield frame



