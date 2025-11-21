## General layout
### Folders and files that you need to create yourself
- A folder containing a Python environment, for example as described in `README.md`.
- A folder `data` in the top-level directory of this repo (on the same level as `CONTRIBUTING.md`) containing the historical data from Polymarket.
- A text file `secrets.txt` in the top-level directory of this repo (on the same level as `CONTRIBUTING.md`) containing secrets like API keys.
### Existing folders
- `algorithms` contains the algorithms needed for the arbitrage pipeline. Each file should have (a) one function to train the algorithm, and (b) one function to run the algorithm on new data. The former can be an empty function if the algorithm is not learning-based.
- `

## Syncing specific files with rest of team
### `requirements.txt`
If you find that a package you need was not installed by `pip install -r requirements.txt`, there are still not sufficient, add it to the file.

### `secrets.txt`
Write the API keys in this file in the format agreed upon in the group chat and documents.

## Running "Jupyter" in VSCode
You should edit `generate_presentation_materials.py` Visual Studio Code as described here. Do not create Jupyter notebook; the `.py` files can be run interactively as if they were Jupyter notebooks if you use VSCode. The file is like below:
```python
#%%
import numpy as np  # this code is in cell 1
#%%
x = np.array([1, 2, 3])  # this code is in cell 2
#%%
x  # this code is in cell 3
```
VSCode will detect the `#%%` symbols as new Jupyter cells. The UI will allow you to click a "Run Cell" option, which will open up an interactive terminal displaying the output of running just one cell.
