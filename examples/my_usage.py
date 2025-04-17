from ezstitcher.core.main import process_plate_auto
from pathlib import Path

folder1 = Path('/home/ts/nvme_usb/IMX/mar-20-axotomy-fca-dmso/mar-20-axotomy-fca-dmso-Plate-2_Plate_13054_all/mar-20-axotomy-fca-dmso-Plate-2_Plate_13054')
folder2 = Path('/home/ts/nvme_usb/IMX/mar-20-axotomy-fca-dmso/mar-20-axotomy-fca-dmso-Plate-2_Plate_13054_all/mar-20-axotomy-fca-dmso-Plate-2_Plate_13054')

success = process_plate_auto(
    folder1,
    **{"reference_channels": ["2"]}
)
success = process_plate_auto(
    folder2,
    **{"reference_channels": ["2"]}
)
