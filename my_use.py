from ezstitcher.core.config import PipelineConfig, StitcherConfig
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.core.pipeline import Pipeline
from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
from ezstitcher.core.image_processor import ImageProcessor as IP
from pathlib import Path

def calcein_process(stack):
    """Apply tophat filter to Calcein images."""
    return [IP.tophat(img) for img in stack]

def dapi_process(stack):
    """Apply tophat filter to DAPI images."""
    stack = IP.stack_percentile_normalize(stack, low_percentile=0.1, high_percentile=99.9)
    return [IP.tophat(img) for img in stack]


config = PipelineConfig(
    stitcher=StitcherConfig(
        tile_overlap=10.0,
        max_shift=50,
        margin_ratio=0.1
    ),
    num_workers=8
)

plate_folders = []
# Add your plate paths here
def add_plate_path(base, suffix):
    """Helper function to create and add plate paths."""
    plate_folders.append(base + suffix)

# First plate
add_plate_path(
    '/home/ts/nvme_usb/Opera/20250407TS-12w_axoTest/20250407TS-12w_axoTest',
    '/20250407TS-12w_axoTest__2025-04-07T14_16_59-Measurement_2'
)

# Second plate
add_plate_path(
    '/home/ts/nvme_usb/Opera/20250407TS-12w_axoTest/20250407TS-12w_axoTest-2',
    '/20250407TS-12w_axoTest-2__2025-04-07T15_10_15-Measurement_2'
)


def gen_pos_pipeline_generator(plate_folder):
    return Pipeline(
        steps=[

            Step(
                func=(IP.create_projection, {'method': 'max_projection'}),
                variable_components=['z_index'],
                input_dir=plate_folder,
            ),

            Step(
                func=(IP.stack_percentile_normalize,
                     {'low_percentile': 0.01, 'high_percentile': 99.9}),
            ),

            Step(
                func=(IP.create_composite,
                    {'weights':[0.5, 0.5,0]}),
                variable_components=['channel']
            ),

            PositionGenerationStep()
        ],
        name="Position Generation Pipeline"
    )

def best_focus_assembly_pipeline_generator(plate_folder):
    return Pipeline(
        steps=[
            Step(
                func=(IP.stack_percentile_normalize,
                     {'low_percentile': 0.01, 'high_percentile': 99.9}),
                input_dir=plate_folder
            ),
            Step(
                func=(IP.create_projection, {'method': 'best_focus'}),
                variable_components=['z_index'],
            ),
            ImageStitchingStep()
        ],
        name="Best Focus Pipeline"
    )

# Create and run pipeline
def run_pipeline(plate_folder, pipeline_generator, config):
    print(f"Processing plate: {plate_folder}")
    print(f"Number of worker threads: {config.num_workers}")
    print(f"Initiaing orchestrator")
    orchestrator = PipelineOrchestrator(config=config, plate_path=plate_folder)
    pipeline = pipeline_generator(orchestrator.workspace_path)
    print(f"Initiated pipeline: {pipeline.name}")
    orchestrator.run(pipelines=[pipeline])
    print(f"Finished processing plate: {plate_folder}")

for plate_folder in plate_folders:
    run_pipeline(plate_folder, best_focus_assembly_pipeline_generator, config)
