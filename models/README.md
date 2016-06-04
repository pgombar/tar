Pre-trained word2vec models
============

Models are too big for GitHub.

To obtain download the zip archive (822MB): http://nlp.stanford.edu/data/glove.6B.zip.

Unzip and add one line to the beginning of each txt file. Line will contain space-separated two integers.
First one is vocabulary size (400000), and second one is dimension (from the filename).

So to the `glove.6B.50d.txt` we prepend line:

```
400000 50
```

.
