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



#### *Run preprocessing scripts like this:*

```
python3 preprocess
```

- Fetches & saves off gbif archive to `./static/`
  - Checks the archive, tries loading it into memory etc
- Fetches Leaflet Annotator binary & licenses from [JessSullivan/MerlinAI-Interpreters](https://github.com/Jesssullivan/MerlinAI-Interpreters);  Need to commit annotator *(as of 03/16/21)*, still fussing with a version for Mushroom Observer  
- Generates an `images.json` file from the 500 assets selected by Joe & Nathan
  - Leaflet Annotator `images.json` Structure:
    - **id**: *taxonID* The MO taxon id
    - **category_id**: The binomen defined in the `./static/sample_select_assets.csv`; for directories and URIs this is converted to snake case.
    - **url**: Temporary elastic ip address this asset will be available from, just to reduce any excessive / redundant traffic to *images.mushroomoberver.org*
    - **src**: *imageURL* The asset's source URL form  Mushroom Observer
    ```
    [{
      "id": "12326",
      "category_id": "Peltula euploca",
       "url": "http://3.223.117.17/static/images/peltula_euploca/12326.jpg"
       "src": "https://images.mushroomobserver.org/640/12326.jpg"
    }]
    ```

- Downloads the 500 selected assets from *images.mushroomoberver.org* at `./static/images/<category>/<id>.jpg`;
  - selected asset directory structure:
  ```
  ├── static
  │   ├── gbif.zip
  │   ├── images
  │   │   ├── amanita_volvata
  │   │   │   └── 1441.jpg
  │   │   ├── ductifera_pululahuana
  │   │   ├── peltula_euploca
  │   │   │   └── 12326.jpg
  │   │   ├── russula_modesta
  │   │   │   └── 16580.jpg
  │   │   └── thamnolia_subuliformis
  │   │       └── 14148.jpg
  │   │   ...
  │   ├── images.json
  │   ├── js
  │   │   ├── leaflet.annotation.js
  │   │   └── leaflet.annotation.js.LICENSE.txt
  │   └── sample_select_assets.csv
  ...
  ```


- - -

#### *Fiddling with the archive:*
- `MODwca.gbif[1].id`: Integer:  This is the Mushroom Observer id, e.g.
  - `https://mushroomobserver.org/13`
  - `https://images.mushroomobserver.org/640/13.jpg`

- `MODwca.gbif[1].data:`: Dictionary: DWCA row data, e.g.
  - `MODwca.gbif[1].data['http://rs.gbif.org/terms/1.0/gbifID']` = `13`
  - `MODwca.gbif[1].data['http://rs.tdwg.org/dwc/terms/recordedBy']` = `Nathan Wilson`
