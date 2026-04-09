from library.core.pipeline.Pipeline import Pipeline

class PipelineValidator:
    def validate(self, pipeline: Pipeline) -> None:
        if not pipeline.frame_extractor:
            raise ValueError("Frame extractor not valid")
        
        if not pipeline.signal_extractor:
            raise ValueError("Signal extractor not valid")
        
        if not pipeline.analyzers:
            raise ValueError("Analyzer list not valid")