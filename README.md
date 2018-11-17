[![Build Status](https://travis-ci.com/SudoSharma/sentence_similarity_app.svg?branch=master)](https://travis-ci.com/SudoSharma/sentence_similarity_app)

# Sentence Similarity Application
This is an application to test how similar two sentences are, based on the BiMPM model by Wang et al., with a few enhancements. For more details, please check out my [implementation](https://github.com/SudoSharma/bimpm_implementation) of the original paper. 
I've essentially modified the architecture of my original implementation to accept a pair of sentences instead of a batch, and am serving up a RESTful API endpoint using Flask for inference. 

# Requirements
## Environment
The `setup.sh` script will create a `sentence_similarity_app` conda environment.  It will require you to specify the distribution of Anaconda you have on your computer or VM, so please be sure to edit this in the script. Run the script with the following command:

    ./setup.sh

## System
- OS: Ubuntu 16.04 LTS (64 bit)
- No GPU required or utilized.

## Instructions
This is the directory structure you should have once you've cloned this repository. You'll have to make sure you have [git lfs](https://git-lfs.github.com/) installed before cloning.

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

    curl -X GET http://127.0.0.1:5000/ -d q1="how far is it from earth to the moon?" -d q2="how far is it from earth to mars?"

# References
1. Wang, Zhiguo, Wael Hamza, and Radu Florian. "Bilateral Multi-Perspective Matching for Natural Language Sentences." Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence, July 14, 2017. Accessed October 10, 2018. doi:10.24963/ijcai.2017/579. 
