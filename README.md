# Fine-Tune Bert Model for Classification Task

The task is subdivided into four parts, namely, data cleaning, data preparation, model training, and evaluation. The data cleaning section loads the data into memory, extracts extra white spaces, removes special characters and non-alphabetical characters. Additionally, words that are less than or equal to two characters in length are removed. The cleaned dataset is then transformed to match the PyTorch dataset format, making it suitable to be segmented into batches using the PyTorch DataLoader utility function. 

The program fine-tunes a pre-trained DistilBERT model for a classification task. It also has an option to select between DistilBERT and RoBERTa models and to select the layer or block to fine-tune. By default, it selects the DistilBERT model and fine-tunes the last block. The process loads the original model weights and freezes them, then adds a pre-classifier layer and a classifier on top of the last block. The weights of the pre-classifier and the output classifier are initialized and fine-tuned during the training process. 

The fine-tuned model achieves an accuracy of 83 percent on the test dataset.

