### Ear Detector

We use Viola Jones algorithm to detect ears. Firstly you need to run the following command to install all the required requirements.

```buildoutcfg
pip install -r requirements.txt
```

To reproduce the reuslts of Viola Jones algortihm you should run `` python evaluation.py ``. You will get the TP, FP numbers and also precision, recall and F1-score. These numbers are for scaleFactor = 1.01 and minNeighbours = 4. If you want to change these parameters you need to change it in `config.py`, where you can also change variable `use_nose_detection` to True and in this case the algortihm will detect left and right ears according to the detected nose. 

After that you should run `python main.py` to detect ears with modified parameters and after that `python evaluate.py` to get the results. 