#### Overview

This is a working copy of makemore repository by Andrej Karpathy with a simple inference script.

  The only differences are:
  1) Added some minor comments explaining work in the train script for gpt
  2) Added an option to run a small network vs a large network
  3) Added an option to save the model and run inference in a dedicated inference script

Full credit on the basis of this repo is to Andrej Karpathy. 

#### Set up

```
python -m venv venv
source venv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

#### Running

Training

```
python gpt.py
```

Inference

```
python gpt-inference.py
```
