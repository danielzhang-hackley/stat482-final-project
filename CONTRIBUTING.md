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