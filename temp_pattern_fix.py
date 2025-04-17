import re

# Define the correct pattern
pattern = re.compile(r'(?:.*?_)?([A-Z]\d+)(?:_s(\d+))?(?:_w(\d+))?(?:_z(\d+))?(\.\w+)?$')

# Test the pattern
test_filenames = [
    "A01_s15_w2.tif",
    "A01_s001_w2.tif",
    "A01_s001_z003.tif",
    "A01_w2_z003.tif",
    "A01_s001_w2.tif",
    "A01.tif"
]

for filename in test_filenames:
    match = pattern.match(filename)
    if match:
        well, site, channel, z, ext = match.groups()
        print(f"{filename}: well={well}, site={site}, channel={channel}, z={z}, ext={ext}")
    else:
        print(f"{filename}: No match")
