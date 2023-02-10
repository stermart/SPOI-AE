[![DOI](https://zenodo.org/badge/598136988.svg)](https://zenodo.org/badge/latestdoi/598136988)

# SPOI-AE

Code and data for the SPOI-AE project. 

### Environment Setup

To recreate the environment used for this project, simply run the command `conda env create --file=conda.yml`. If you would like to have the environment be some name other than `SPOI-AE` simply set the `--name` flag in the `conda env create` call. 

### Directory Structure

The trained models referenced in the paper can be found in the directory [Trained Models/Paper Models](Trained%20Models/Paper%20Models/). Any models that you train will appear in the directory [Trained Models/User Create](Trained%20Models/User%20Created/). All data used is stored in the [Data](Data/) directory. The data is saved in the `.mat` format. 

### Training/Evaluation Scripts

The script used to train the models is [train_model.py](train_model.py) and the parameters to modify the training are in the file itself. The various evaluation scripts are [evaluate_model.py](evaluate_model.py), [optical_inversion_demo.py](optical_inversion_demo.py), and [processing_pipeline_demo.py](processing_pipeline_demo.py).
