# MAGCL
This is our Tensorflow implementation for the paper "Multi-aspect Graph Contrastive Learning for Review-enhanced Recommendation".
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.10.0
* numpy == 1.19.5
* scipy == 1.5.2
* sentence-transformers == 0.4.1.2

Data
-----------------
`python data/data_preprocessing.py ` 
  * Dataset: [Amazon Product Review dataset](http://jmcauley.ucsd.edu/data/amazon/links.html)/Digital Music
  * Download S-BERT model [all-MiniLM-L6-v2 ](https://huggingface.co/sentence-transformers)
  * generate `data.pkl` and `Digital_Music.csv`

Run MAGCL
-----------------
`python model/MAGCL.py ` 
  * model type: MAGCL-dec-5
