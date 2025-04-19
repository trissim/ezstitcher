from ezstitcher.core.processing_pipeline import PipelineOrchestrator
from ezstitcher.core.config import StitcherConfig, PipelineConfig
from ezstitcher.core.image_preprocessor import ImagePreprocessor as IP 

def calcein_process(stack):
    """Apply tophat filter to Calcein images."""
    return [IP.tophat(img) for img in stack]

def dapi_process(stack):
    """Apply tophat filter to DAPI images."""
    stack = IP.stack_percentile_normalize(stack,low_percentile=0.1,high_percentile=99.9)
    return [IP.tophat(img) for img in stack]

config = PipelineConfig(
        reference_processing={
            "3": calcein_process,
            "2": dapi_process
        },
        reference_channels=["3", "2"],
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50,
            margin_ratio=0.1
        ),
    )

plate_folders = []
#plate_folders.append('/home/ts/nvme_usb/IMX/mar-20-axotomy-fca-dmso/mar-20-axotomy-fca-dmso-Plate-1_Plate_13053_all/mar-20-axotomy-fca-dmso-Plate-1_Plate_13053')
#plate_folders.append('/home/ts/nvme_usb/IMX/mar-20-axotomy-fca-dmso/mar-20-axotomy-fca-dmso-Plate-2_Plate_13054_all/mar-20-axotomy-fca-dmso-Plate-2_Plate_13054')

plate_folders = []
config = PipelineConfig(
        reference_processing={
            "1": dapi_process,
            "2": calcein_process,
        },
        reference_channels=["1", "2"],
        reference_flatten="max_projection",
        stitch_flatten="max_projection",
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50,
            margin_ratio=0.1
        ),
        #well_filter=["r01c01"],

    )
plate_folders.append('/home/ts/nvme_usb/Opera/20250407TS-12w_axoTest/20250407TS-12w_axoTest/20250407TS-12w_axoTest__2025-04-07T14_16_59-Measurement_2')
plate_folders.append('/home/ts/nvme_usb/Opera/20250407TS-12w_axoTest/20250407TS-12w_axoTest-2/20250407TS-12w_axoTest-2__2025-04-07T15_10_15-Measurement_2')

 # Create and run pipeline
for plate_folder in plate_folders:
    pipeline = PipelineOrchestrator(config)
    pipeline.run(plate_folder)
