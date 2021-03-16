import os
import csv
import json

# load common names:
from common import *


class BuildImagesJson:

    def __init__(self,
                 _static_path=STATIC_PATH,
                 _sample_assets_path=SAMPLE_ASSETS_PATH):

        # common paths
        self.static_path = _static_path
        self.sample_assets_path = _sample_assets_path

        # we'll use dictionary `json_df` to populate `images.json`:
        self.json_df = []

        if not os.path.exists(self.static_path):
            os.makedirs(self.static_path)

        with open(self.sample_assets_path, newline='') as f:

            # open up Nathan and  Joe's merged csv files from Google Sheets:
            ln = csv.reader(f, delimiter=',', quotechar='|')

            for row in ln:
                try:
                    # make sure we only adding integer keys:
                    if int(row[0]):
                        _obj = {'id': str(row[0]),
                                'category_id': str(row[1]),
                                'url': "https://images.mushroomobserver.org/640/" + str(row[0]) + ".jpg",
                                'src': row[2]}
                        self.json_df.append(_obj)

                except ValueError:
                    # not an integer, skip it
                    pass

    def write_images_json(self, _path=None):

        # let the caller save off archive somewhere else if they want with optional _path argument
        path = self.static_path + 'images.json' if _path is None else _path

        if len(self.json_df) > 0:

            with open(path, 'w') as f:
                json.dump(self.json_df, f)

            print("Wrote out a fresh " + path + " file! \n" +
                  "  exported data length: " + str(len(self.json_df)) + "\n" +
                  "  output file size: " + str(os.path.getsize(path)) + " bytes")

        else:
            print("...Hmm, didn't write images.json, no assets found!")
