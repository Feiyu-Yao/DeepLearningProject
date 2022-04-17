import nltk
import spacy #python -m spacy download en_core_web_sm
# nltk.download("wordnet")
import random
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


#
# nlp = spacy.load('en_core_web_sm')
# sent = "I shot an elephant"
# doc=nlp(sent)
#
# sub_toks = [tok for tok in doc if (tok.dep_ == "nsubj") ]
#
# print(sub_toks)



# sentence = "At eight o'clock on Thursday morning Arthur didn't feel very good."
# tokens = nltk.word_tokenize(sentence)
# print(tokens)
# tagged = nltk.pos_tag(tokens)
# print("tagged:",tagged)

import json
path='a2d_sentences_single_frame_test_annotations.json'
f = open(path,'r',encoding='utf-8')
m = json.load(f)
#print(m)
sentences=[]
librarys=[]

for i in m:
    sentences.append(i[0])
    for j in nltk.word_tokenize(i[0]):
        #是否需要这个
        # j=lemmatizer.lemmatize(j)
        librarys.append(j)

#去重

sentences=list(set(sentences))
#print("sentences",sentences)

librarys=list(set(librarys))
# print("librarys",librarys)

adv_lib=[]
noun_lib=[]
dir_lib=['front', 'back' ,'left', 'right','up', 'down', 'top','under' ]

tagged_lib = nltk.pos_tag(librarys)
# print(tagged_lib)
for tagged in tagged_lib:
    if tagged[1]=='NN' or tagged[1]=='NNS' or tagged[1]=='NNP' or tagged[1]=='NNPS':
        noun_lib.append(tagged[0])
    if tagged[1]=='JJ' or tagged[1]=='JJR' or tagged[1]=='JJS' :
        adv_lib.append(tagged[0])
    # if tagged[1]=='NN' or tagged[1]=='NNS' or tagged[1]=='NNP' or tagged[1]=='NNPS':
    #     noun_lib.append(tagged[0])
# print(noun_lib)



######step2: 对每个整句，获得tag进行替换;

nlp = spacy.load('en_core_web_sm')


tagged_m= []# tagged_m 不去重获得原本数据的tagged , tagged_lib去重获得lib
adv_toks=[]
sub_toks=[]
dir_toks=['front', 'back' ,'left', 'right','up', 'down', 'top','under' ]
new_m=[]
for i in m:
    sent=i[0]
    tagged_m.append(sent)
    tokens = nltk.word_tokenize(sent)
    tagged = nltk.pos_tag(tokens)
    doc = nlp(sent)
    sub_toks = [tok for tok in doc if (tok.dep_ == "nsubj")]
    if not sub_toks==[]:
        sub_toks=str(sub_toks[0])
    for k in tagged:
        if k[1]=='JJ' or k[1]=='JJR' or k[1]=='JJS' :
            adv_toks.append(k[0])

    new_sent=[]
    for j in tokens:
        if j==sub_toks:
            sub_random=random.choice(noun_lib)
            new_sent.append(sub_random)
        elif j in adv_toks:
            sub_random=random.choice(adv_lib)
            new_sent.append(sub_random)
        elif j in dir_toks:
            #sub_random=random.choice(dir_lib)
            if j=='front':
                sub_random='back'
            if j == 'back':
                sub_random = 'front'
            if j == 'left':
                sub_random = 'right'
            if j == 'right':
                sub_random = 'left'
            if j == 'up':
                sub_random = 'down'
            if j == 'down':
                sub_random = 'up'
            if j == 'top':
                sub_random = 'under'
            if j == 'under':
                sub_random = 'top'

            new_sent.append(sub_random)
        else:
            new_sent.append(j)

    new_sent1=' '
    new_sent1=new_sent1.join(new_sent)
    #i.insert(1, sent)
    i.insert(1, new_sent1)
    new_m.append(i)


filename='with_inverse_a2d_sentences_single_frame_test_annotations.json'
with open(filename,'w') as f:
    json.dump(new_m,f)

#print(new_m[1])

# sentence = "At eight o'clock on Thursday morning Arthur didn't feel very good."
# tokens = nltk.word_tokenize(sentence)
# print(tokens)
# tagged = nltk.pos_tag(tokens)
# print("tagged:",tagged)
