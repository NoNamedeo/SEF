from SEF.artifacts.Frame import Frame

class FrameSequence:
    def __init__(self, frames: list[Frame]):
        self.frames = frames

    def __len__(self):
        return len(self.frames)



