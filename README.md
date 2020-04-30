# RLcaching
Testing caching reinforcement learning agents

## Extracting file access data


Environment gives tokenized filename, filesize. It always caches the file. 
Actor returns a probability p (0, 1.0) that it thinks the file will be found in cache.
If file was found in the cache it gets rewarded (p+0.05)*filesize if it misses it gets penalized the same ammount. 

Once cache is full actor is called for each file but with learning disabled. Files are ordered according to probabilities returned and ones with lowest probability removed.

