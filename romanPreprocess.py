import os
from nltk.tokenize import sent_tokenize
import re

files = os.listdir("les Romans_Preprocessed")
skipList = ["*", "-", "[", "FIN", "  "]

for i in range(len(files)):
    result = []
    print("les Romans_Preprocessed/" + files[i] + "작업중..")
    with open("les Romans_Preprocessed/" + files[i], 'r', encoding = 'utf-8') as f:
        text = f.read().replace("\n", "")
        sentences = sent_tokenize(text, language = 'french')

        for j in range(len(sentences)):
            if len(sentences[j].split()) > 10 and all(skip not in sentences[j] for skip in skipList):
                result.append(sentences[j])

    with open("romans/OUTPUT/" + files[i], 'w', encoding = 'utf-8') as f:
        for line in result:
            f.write(line)
            f.write('\n')