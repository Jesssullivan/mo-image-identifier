from load_dwca import MODwca
from build_images_json import BuildImagesJson
if __name__ == "__main__":
    dwca = MODwca()
    imagesJson = BuildImagesJson()
    imagesJson.write_images_json()
