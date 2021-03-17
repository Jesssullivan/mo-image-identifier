import os
import csv
import json
import subprocess
import tarfile
import uuid

# load common names:
from common import *

# import MO dwca class:
from load_dwca import MODwca


class BuildImages:

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

                        _dir_name = str(row[1]).replace(" ", "_").lower()

                        _obj = {'id': str(row[0]),
                                'category_id': str(row[1]),
                                'url': 'http://3.223.117.17/static/images/' + _dir_name + '/' + str(row[2]).split('/')[-1],
                                'src': str(row[2])
                                }

                        self.json_df.append(_obj)

                except ValueError:
                    # not an integer, skip it
                    pass

    def fetch_leaflet_tool(self):

        print('Fetching Leaflet annotator binaries...')

        if not os.path.exists(self.static_path + "js/"):
            os.makedirs(self.static_path + "js/")

        for _obj in LEAFLET_URL, LEAFLET_URL + ".LICENSE.txt":
            _cmd = "curl -L " + _obj + " --output ./static/js/" + _obj.split('/')[-1]
            subprocess.Popen(_cmd, shell=True).wait()

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

    def fetch_online_images(self, _json):

        print('Fetching online images from images.mushroomobserver.org...  \n...this may take a while :)')

        if not os.path.exists(self.static_path + "images/"):
            os.makedirs(self.static_path + "images/")

        with open(_json, 'r') as f:
            images_json = json.load(f)

        print("Found " + str(len(images_json)) + " assets in " + _json + "...")

        for asset in images_json:

            _dir_name = asset['category_id'].replace(" ", "_").lower()

            if not os.path.exists(self.static_path + "images/" + _dir_name):
                os.makedirs(self.static_path + "images/" + _dir_name)

            _cmd = str("curl -L " + asset['src'] +
                       " --output ./static/images/" + _dir_name + "/" + asset['src'].split('/')[-1])

            subprocess.Popen(_cmd, shell=True).wait()

    def export_tgz(self):

        if os.path.exists(self.static_path + "images/"):

            # _fname = self.static_path + "images_" + str(uuid.uuid4())[0:6] + ".tgz"  # uniquely named export not as useful as I thought...
            _fname = self.static_path + "images.tgz"

            print("Writing tgz image archive to " + _fname + "...\n...")

            with tarfile.open(_fname, "w:gz") as tar:
                tar.add(self.static_path + "images/", arcname=os.path.basename(self.static_path + "images/"))

            print("Finished writing tgz image archive! :)")

        else:

            print("Couldn't find static/images/ directory! :(")
