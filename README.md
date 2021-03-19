### *Tools for doing image-based mushroom identification*
- [**@github.com**](https://github.com/Jesssullivan/image-identifer) <br>
- [**@github.io**](https://jesssullivan.github.io/image-identifer/) <br>

***Overview:***
- [**Setup**](#setup) <br>
- [**Artifacts**](#artifacts) <br>
- [**Preprocess**](#preprocess) <br>
- [**Artifacts**](#artifacts) <br>
- [**Train**](#train) <br>
- [**Structures**](#structures) <br>
- [**Notes**](#notes) <br>



- - -


<h4 id="setup"> </h4>     



### *Setup:*
```
# venv:
python3 -m venv mushroomobserver_venv
source mushroomobserver_venv/bin/activate
pip3 install -r requirements.txt
```


<h4 id="artifacts"> </h4>


|***Artifacts:***|[*train.tgz*](https://mo.columbari.us/static/train.tgz)|[*test.tgz*](https://mo.columbari.us/static/test.tgz)|
|---|---|---|
|[*images.tgz*](https://mo.columbari.us/static/images.tgz) |[*images.json*](https://mo.columbari.us/static/images.json)|[*gbif.zip*](https://mo.columbari.us/static/gbif.zip)|


- - - 


<h4 id="preprocess"> </h4>


### *Preprocess:*

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



### *Train:*

```
python3 train
```

- Fetches, divvies & shuffles train / validation sets from within Keras using archive available at [*mo.columbari.us/static/images.tgz*](https://mo.columbari.us/static/images.tgz)
- More or less running Google's demo transfer learning training script in [`train/training_v1.py`](train/training_v1.py) as of *03/17/21*, still need to bring in training operations and whatnot from merlin_ai/ repo --> experiment with Danish Mycology Society's ImageNet v4 notes


***Google Colab:***

- [@gvanhorn38](https://github.com/gvanhorn38/) pointed out Google Colabs's neat Juptyer notebook service will train models for free if things are small enough- I have no idea what the limits are- fiddle with their [***intro to image classification on Google Colab here***](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/classification.ipynb), its super cool!


***Jupyter:***

- One may also open and run notebooks locally like this:
  - [rendered pdf version available over here](train/notebook/training_v1.pdf)
  - rename ipython notebook:
  ```
  cp train/notebook/training_v1.ipynb.bak train/notebook/training_v1.ipynb
  ```
  - launch jupyter:
  ```
  jupyter notebook
  ```
  - or without authentication:
  ```
  jupyter notebook --ip='*' --NotebookApp.token='' --NotebookApp.password=''
  ```


- - -


<h4 id="structures"> </h4>


### *Structures:*


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


<h4 id="notes"> </h4>


#### *Notes:*


*...Progress:*  <br/>


| ... | *51%*, ...a ways to go... |
|---|---|
| ![](https://www.transscendsurvival.org/wp-content/uploads/2021/03/f1-281x300.png) | ![](https://www.transscendsurvival.org/wp-content/uploads/2021/03/f2-300x151.png) |


*Fiddling with the archive:*
- `MODwca.gbif[1].id`: Integer:  This is the Mushroom Observer taxon id, e.g.
  - `https://mushroomobserver.org/13`
  - `https://images.mushroomobserver.org/640/13.jpg`

- `MODwca.gbif[1].data:`: Dictionary: DWCA row data, e.g.
  - `MODwca.gbif[1].data['http://rs.gbif.org/terms/1.0/gbifID']` = `13`
  - `MODwca.gbif[1].data['http://rs.tdwg.org/dwc/terms/recordedBy']` = `Nathan Wilson`
