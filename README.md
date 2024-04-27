## Prerequisite

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



## Usage

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



## Reference

* Mao, H. H., Shin, T., & Cottrell, G. (2018). DeepJ: Style-Specific Music Generation. 2018 IEEE 12th International Conference on Semantic Computing (ICSC), 377–382. https://doi.org/10.1109/ICSC.2018.00077
