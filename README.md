# tf-tb-visualisation
This repository provides a starter code for using tensorboard via tensorflow for visualising embeddings

Word Embeddings:

For visualising word embeddings run the following from the command line:
```
python visualise_text_embeddings.py -b $baseDir -f $filename_embedding -l $filename_embedding
tensorboard --logdir=$baseDir
```

