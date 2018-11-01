# BiMPM Application
This repository hosts an application built using a very minimal version of the [full BiMPM implementation](https://github.com/SudoSharma/bimpm_implementation). It is very useful for testing pairs of queries. I've cleaned away all the extraneous code that was initially important for training the model in the `model/utils.py` and `evaluate.py` scripts, and going forward, will only update the `model/layers.py` script or the `data/args.pkl` file with the results of my experiments on the original implementation.

# Requirements
## Environment
The `setup.sh` script will create an `bimpm_app` conda environment for the CPU.  It requires you to specify the distribution of Anaconda you have on your computer or VM, so please be sure to edit this in the script. Run the script with the following command:

    ./setup.sh

## System
- OS: Ubuntu 16.04 LTS (64 bit)
- No GPU required. Runs fine on a CPU. 

## Instructions
This is the directory structure you should have once you've cloned this repository. 

    $ tree -I __pycache__ -F -n
    .
    ├── app.py
    ├── data/
    │   ├── args.pkl
    │   └── TEXT.pkl
    ├── evaluate.py
    ├── LICENSE.md
    ├── model/
    │   ├── bimpm.py
    │   ├── layers.py
    │   └── utils.py
    ├── README.md
    ├── saved_models/
    └── setup.sh* 

Note that I've removed the pre-trained model in the `saved_models/`folder in this directory tree diagram because it will be periodically updated with new models.

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

### Test 1
#### Does the network understand word order?
    curl -X GET http://127.0.0.1:5000/ -d q1="my dog ate my homework" -d q2="my homework ate my dog"

### Test 2
#### Does the network understand pronouns? 
    curl -X GET http://127.0.0.1:5000/ -d q1="how old are you?" -d q2="how old am I?"

### Test 3
#### Does the network understand semantics?
    curl -X GET http://127.0.0.1:5000/ -d q1="where is the moon?" -d q2="where is my car?"

    curl -X GET http://127.0.0.1:5000/ -d q1="can I bring my cat to the hospital?" -d q2="can i bring my dog to the hospital?"

    curl -X GET http://127.0.0.1:5000/ -d q1="can I bring my cat to the hospital?" -d q2="will I be able to take my cat to the hospital?"

    curl -X GET http://127.0.0.1:5000/ -d q1="how far is it from earth to the moon?" -d q2="what's the distance from earth to the moon?"

    curl -X GET http://127.0.0.1:5000/ -d q1="how far is it from earth to the moon?" -d q2="what's the distance from earth to mars?"

