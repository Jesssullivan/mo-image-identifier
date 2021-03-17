# image-identifer

Tools for doing image-based mushroom identification


- - -


#### *Setup:*
```
# venv:
python3 -m venv mushroomobserver_venv
source mushroomobserver_venv/bin/activate
pip3 install -r requirements.txt
```


|***Artifacts:***|[*train.tgz*](https://mo.columbari.us/static/train.tgz)|[*test.tgz*](https://mo.columbari.us/static/test.tgz)|
|---|---|---|
|[*images.tgz*](https://mo.columbari.us/static/images.tgz) |[*images.json*](https://mo.columbari.us/static/images.json)|[*gbif.zip*](https://mo.columbari.us/static/gbif.zip)|


#### *Run preprocessing scripts like this:*

```
python3 preprocess
```

- Fetches & saves off gbif archive to `./static/`
  - Checks the archive, tries loading it into memory etc
- Fetches Leaflet Annotator binary & licenses from [JessSullivan/MerlinAI-Interpreters](https://github.com/Jesssullivan/MerlinAI-Interpreters);  Need to commit annotator *(as of 03/16/21)*, still fussing with a version for Mushroom Observer  
- Generates an `images.json` file from the 500 assets selected by Joe & Nathan
- Downloads, organizes the 500 selected assets from *images.mushroomoberver.org* at `./static/images/<category>/<id>.jpg`
  - writes out images archive
- More or less randomly divvies up testing & training image sets
  - writes out example testing/training archives; (while training it'll probably be easier to resample directly from images.tgz from keras)


#### *Training w/ Notebooks & Google Colab*


@gvanhorn38 pointed out Google Colabs's neat Jupter notebook service will train models for free if things are small enough- I have no idea what the limits are- fiddle with their [***intro to image classification on Google Colab here***](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/classification.ipynb), its super cool!  Added more or less verbatim MO version of this to [./train/training_v1](./train/training_v1.ipynb) as well.  One can open and run the notebook locally like this-
```
jupyter notebook
# or without authentication, something along the lines of:
jupyter notebook --ip='*' --NotebookApp.token='' --NotebookApp.password=''
```

- - -


- *Leaflet Annotator `images.json` Structure:*
  - **id**: *taxonID* The MO taxon id
  - **category_id**: The binomen defined in the `./static/sample_select_assets.csv`; for directories and URIs this is converted to snake case.
  - **url**: Temporary elastic ip address this asset will be available from, just to reduce any excessive / redundant traffic to *images.mushroomobserver.org*
  - **src**: *imageURL* The asset's source URL form  Mushroom Observer
  ```
  [{
    "id": "12326",
    "category_id": "Peltula euploca",
     "url": "https://mo.columbari.us/static/images/peltula_euploca/290214.jpg"
     "src": "https://images.mushroomobserver.org/640/290214.jpg"
  }]
  ```
- *Selected asset directory structure:*
  ```
  ├── static
    ├── gbif.zip
    ├── images
    |   ...
    │   └── peltula_euploca
    │       ├── 290214.jpg
    │       ...
    │       └── 522128.jpg
    │   ...
    ├── images.json
    ├── images.tgz
    ├── js
    │   ├── leaflet.annotation.js
    │   └── leaflet.annotation.js.LICENSE.txt
    └── sample_select_assets.csv
  ...
  ```


- - -


#### *Notes:*



*Fiddling with the archive:*
- `MODwca.gbif[1].id`: Integer:  This is the Mushroom Observer taxon id, e.g.
  - `https://mushroomobserver.org/13`
  - `https://images.mushroomobserver.org/640/13.jpg`

- `MODwca.gbif[1].data:`: Dictionary: DWCA row data, e.g.
  - `MODwca.gbif[1].data['http://rs.gbif.org/terms/1.0/gbifID']` = `13`
  - `MODwca.gbif[1].data['http://rs.tdwg.org/dwc/terms/recordedBy']` = `Nathan Wilson`
