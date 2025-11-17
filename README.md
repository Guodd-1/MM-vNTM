# MM-vNTM

This repository includes the source code of "Multimodal Topic Discovery in Web Media via von Mises–Fisher
Mixture Neural Topic Models"

## Core Dependency

* Python 3.10  
* PyTorch 2.9.0 + CUDA 13.0 (`torch==2.9.0+cu130`)  

* Transformers (`transformers==4.40.2`)  
* Sentence-Transformers (`sentence-transformers==2.5.0`) 
* Gensim (`gensim==4.2.0`)  
* Scikit-learn (`scikit-learn==1.5.0`)  
* NumPy (`numpy==1.26.4`)  
* Pandas (`pandas==2.2.1`)
* PyTorch Metric Learning (`pytorch-metric-learning==2.5.0`)  
* tqdm (`tqdm==4.66.4`)  
* NLTK (`nltk==3.8.1`)   
* Pillow (`pillow==10.3.0`) 


## An example to run

Demo is coming soon..

## Custom modification
* **Custom Dataset**  
  * `path_file`: A CSV file containing `image_id`, `caption`, and `img_path`.  
  * `image_emb_file`: Pre-extracted image embeddings (e.g., in `.npy` format).  
  * `image_emb_file_csv`: Image embeddings saved in CSV format (alternative format).  
  * `image_embeddings_index_path`: A file (e.g., `.pkl`) mapping embedding indices to `image_id`s.  
  * `sbert_model_path`: Path to a pretrained vision-language model (e.g., `clip-ViT-B-32`).  
  * `word2vec_path`: Pretrained Word2Vec model (e.g., `GoogleNews-vectors-negative300.bin`) for coherence evaluation.

