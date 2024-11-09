# AfroLID

This is a port of UBC-NLP's AfroLID from FairSeq to Pytorch. Refer to the [original repository](https://github.com/UBC-NLP/afrolid) for license and citation info.


## Setup

### Using Poetry
You'll need [Poetry](https://python-poetry.org/docs/). Then run `poetry install` from the `afrolid` working directory.
## Known Issues

* Currently, the model's probabilities are a little off the Fairseq implementation. This may be related to the position ids and embedding.
