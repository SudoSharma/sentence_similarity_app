# BiMPM Application
This repository hosts an application built using a very minimal version of the [full BiMPM implementation](https://github.com/SudoSharma/bimpm_implementation). It is very useful for testing pairs of queries. I've cleaned away all the extraneous code that was initially important for training the model in the `model/utils.py` and `evaluate.py` scripts, and going forward, will only update the `model/layers.py` script or the `data/args.pkl` file with the results of my experiments in the other repository. 

# Requirements
## Environment
The `setup.sh` script will create an `bimpm` conda environment for the CPU.  It requires you to specify the specific distribution of Anaconda you have on your computer or VM, so please be sure to edit this in the script. This script can replace a vaniilla `requirements.txt` file with a processed one that removes the en-core-web-sm model from spaCy requirement with a direct release download url because of a peculiarity with spaCy. It is commented out for now, but you can enable it again if you need to. Run the script with the following command:

    ./setup.sh

## System
- OS: Ubuntu 16.04 LTS (64 bit)
- No GPU required. Runs fine on a CPU. 

## Instructions
This is the directory structure you should have once the git repository is cloned. 

    $ tree -I __pycache__ -F -n
    .
    ├── app.py
    ├── data/
    │   ├── args.pkl
    │   └── TEXT.pkl
    ├── evaluate.py
    ├── model/
    │   ├── bimpm.py
    │   ├── layers.py
    │   └── utils.py
    ├── README.md
    ├── requirements.txt
    ├── saved_models/
    └── setup.sh* 

Note that I've pruned the actual file in the `saved_models/`folder in this tree because this model will be updated with the results of experiments on the full implementation.

## Testing
In order to test out the application, launch the `app.py` file:

    python app.py

This will use Flask to serve an endpoint for a RESTful API, which you can query from whatever client of your choosing. Here's an example of a cURL command:

    $ curl -X GET http://127.0.0.1:5000/ -d q1="How old are you?" -d q2='How old am I?"

    {
        "query_one": "how old are you?",
        "query_two": "how old am i?",
        "neg_layer_output": 0.7569,
        "pos_layer_output": -0.4659,
        "neg_probability": 0.7726,
        "pos_probability": 0.2274,
        "prediction": "not similar"
    }
