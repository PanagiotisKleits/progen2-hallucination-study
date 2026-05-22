#Instructions on how to Install Progen2

1. Clone the repo

    git clone https://github.com/salesforce/progen.git
    cd progen/progen2

2. Create and activate a virtual Environment

    python3 -m venv progen_env
    source progen_env/bin/activate

3. Install the required packages

    pip install torch --extra-index-url https://download.pytorch.org/whl/cu117
    pip install transformers tokenizers
    pip install numpy scipy

4. Download the models checkpoints

    bash scripts/download.sh progen2-small checkpoints


#Instructions on how to execute the script

1. Take a tsv file with the parameters (t,p)

2. Open the run_progen2.py

3. Define the path to the progen2 directory

4. Define the path to the checkpoints

5. Define the name of the parameters file

6. Define the name of the output directory