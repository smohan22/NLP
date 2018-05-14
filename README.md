# N-gram Kneser-Ney
An implementation of N-gram language modeling with Kneser-Ney smoothingin Python3.

# Usage
### Sample Text used for Training: Pride and Prejudice, Chapter 1 ngram for training text
```
chapter = ""
with open ('Austen_Pride.txt','r') as f:
for line in f:
chapter += line
chapter = chapter.replace('\n', ' ').replace("ï»¿", "").strip("'").strip("`")

```

### Train and fit models
```
import KneyserNey

ngram_order = 3

prideKN = kneyserNey()
prideKN.fit(chapter, ngram_order)
```
### Show model fitting
```
phrase='truth universally hated'
d = 0.75 #the discounting factor

prideKN.score(phrase, ngram_order, d)

```

