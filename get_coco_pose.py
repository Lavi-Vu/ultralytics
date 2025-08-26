from pathlib import Path

from ultralytics.utils.downloads import download

# Download labels
dir = Path("/media/lavi/Data/ultralytics/data_coco_pose/")  # dataset root dir
url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
urls = [f"{url}coco2017labels-pose.zip"]
download(urls, dir=dir.parent)
# Download data
urls = [
"http://images.cocodataset.org/zips/train2017.zip",  # 19G, 118k images
"http://images.cocodataset.org/zips/val2017.zip",  # 1G, 5k images
]
download(urls, dir=dir / "images", threads=3)