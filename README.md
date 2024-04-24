# OP

### Requirements

Setup package management:

```
yes | sudo apt-get install python3 python3-pip python3-dev
```

```
yes | sudo apt-get install libasound2-dev python3-augeas swig
```

Clone Python MIDI:

```
git clone --single-branch --branch feature/python3 https://github.com/vishnubob/python-midi.git
```

Install other dependencies:

```
pip install -r requirements.txt
```



### Usage

To train a new model, run the following command:

```
python train.py
```

To generate music, run the following command:

```
python generate.py
```

Use the help command to see CLI arguments:

```
python generate.py --help
```