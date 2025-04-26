# What is this

Working copy of makemore repository by Andrej Karpathy with a simple inference script

# Set up

```
python -m venv venv
source venv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

# Running

Training

```
python gpt.py
```

Inference

```
python gpt-inference.py
```
