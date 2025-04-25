from ezstitcher import Pipeline, Step, IP
from ezstitcher.core.utils import stack
from ezstitcher.steps import PositionGenerationStep, ImageStitchingStep
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator
from n2v.models import N2V
from basicpy import BaSiC
from pathlib import Path
import numpy as np

# Custom processing functions
def n2v_process(images, model_path):
    """Apply Noise2Void denoising to images"""
    model = N2V(None, model_path, 'N2V')
    return [model.predict(img, 'N2V') for img in images]

def basic_process(images):
    """Apply BaSiC illumination correction"""
    basic = BaSiC()
    basic.fit(np.stack(images))
    return list(basic.transform(np.stack(images)))

def generate_position_pipeline(orchestrator, n2v_model_path):
    """Generate pipeline for position file creation"""
    return Pipeline(
        steps=[
            # Normalize and process z-stacks
            Step(func=IP.stack_percentile_normalize,
                 input_dir=orchestrator.workspace_path),
            Step(func=(IP.create_projection, {'method': 'max_projection'}),
                variable_components=['z_index']),
            # Create composite for position generation
            Step(func=IP.create_composite,
                variable_components=['channel']),
            PositionGenerationStep()
        ])

def generate_stitching_pipeline(orchestrator, n2v_model_path):
    """Generate pipeline for image stitching"""
    return Pipeline(
        steps=[
            # Advanced image processing
            Step(func=(stack(n2v_process), {'model_path': n2v_model_path}),
                 input_dir=orchestrator.workspace_path),
            Step(func=stack(basic_process)),
            # Final image corrections
            Step(func=IP.stack_percentile_normalize),
            Step(func=IP.stack_histogram_match),
            ImageStitchingStep(positions_file='positions.json')
        ])

def process_plate(plate_path, n2v_model_path, num_workers=4):
    """Process a single plate with position generation and stitching"""
    config = PipelineConfig(num_workers=num_workers)
    orchestrator = PipelineOrchestrator(config=config, plate_path=plate_path)
    
    # Generate and run both pipelines
    pos_pipeline = generate_position_pipeline(orchestrator, n2v_model_path)
    stitch_pipeline = generate_stitching_pipeline(orchestrator, n2v_model_path)
    orchestrator.run(pipelines=[pos_pipeline, stitch_pipeline])

if __name__ == "__main__":
    n2v_model_path = "path/to/trained_n2v_model.h5"
    plate_paths = [
        Path("/data/plate1"),
        Path("/data/plate2"),
        Path("/data/plate3")
    ]
    
    for plate_path in plate_paths:
        print(f"Processing plate: {plate_path}")
        process_plate(plate_path, n2v_model_path)
        print(f"Completed processing: {plate_path}")