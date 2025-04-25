from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.core.config import StitcherConfig, PipelineConfig
from ezstitcher.core.image_processor import ImageProcessor as IP

def calcein_process(stack):
    """Apply tophat filter to Calcein images."""
    return [IP.tophat(img) for img in stack]

def dapi_process(stack):
    """Apply tophat filter to DAPI images."""
    stack = IP.stack_percentile_normalize(stack, low_percentile=0.1, high_percentile=99.9)
    return [IP.tophat(img) for img in stack]

# Create a simplified configuration
config = PipelineConfig(
    stitcher=StitcherConfig(
        tile_overlap=10.0,
        max_shift=50,
        margin_ratio=0.1
    ),
    num_workers=4
)

plate_folders = []
# Example plate paths (commented out for reference)
# plate_folders.append('/path/to/plate1')
# plate_folders.append('/path/to/plate2')

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


# Create and run pipeline
for plate_folder in plate_folders:
    print(f"\nProcessing plate: {plate_folder}")


    print(f"Number of worker threads: {config.num_workers}")

    # Create and run the pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run(plate_folder)
