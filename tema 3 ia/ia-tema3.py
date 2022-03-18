import spacy
import glob
import math
import matplotlib.pyplot as plt

class document:            
        def __init__(self, url, tokens):      
               self.url=url
               self.tokens=tokens
        def __str__(self):
            return "Document \"" + self.url + "\""

def add_dict(word, dict):
    if word in dict:
        dict[word] += 1
    else:
        dict[word] = 1

def get_dict(word, dict):
    if word in dict:
        return dict[word]
    else:
        return 0

def sub_dic(news_words, abstr_words):
    news_but_not_abstr_word_count = 0
    news_but_not_abstr = {}
    for word in news_words:
        if not word in abstr_words:
            news_but_not_abstr[word] = news_words[word]
            news_but_not_abstr_word_count += news_words[word]
        else:
            if news_words[word] - abstr_words[word] < 0:
                news_but_not_abstr[word] = 0
            else:
                news_but_not_abstr[word] = news_words[word] - abstr_words[word]
                news_but_not_abstr_word_count += news_words[word] - abstr_words[word]
    return (news_but_not_abstr, news_but_not_abstr_word_count)

def calc_log(token_text, dic, word_count):
    return math.log((get_dict(token_text, dic) + 1) / (word_count + len(dic)))

def is_sent_in_abstr_bayes(sent, procentages, word_counts, dics, stopwords, lemma): #daca e lemma, trebuie ca eplantul sa puna informatiile de dict de lema
    (p_sumarry, not_p_sumarry) = procentages
    (abstr_word_count, news_but_not_abstr_word_count) = word_counts
    (abstr_words, news_but_not_abstr) = dics
    
    log_sum_abstr = 1
    log_sum_not_abstr = 1

    if stopwords:
        for token in sent:
            if lemma:
                if not token.is_stop: 
                    log_sum_abstr += calc_log(token.lemma_, abstr_words, abstr_word_count)
                    log_sum_not_abstr += calc_log(token.lemma_, news_but_not_abstr, news_but_not_abstr_word_count)
            else:
                if not token.is_stop:
                    log_sum_abstr += calc_log(token.text, abstr_words, abstr_word_count)
                    log_sum_not_abstr += calc_log(token.text, news_but_not_abstr, news_but_not_abstr_word_count)
                    
    else:
        for token in sent:
                log_sum_abstr += calc_log(token.text, abstr_words, abstr_word_count)
                log_sum_not_abstr += calc_log(token.text, news_but_not_abstr, news_but_not_abstr_word_count)
            
    log_sum_abstr += math.log(p_sumarry)
    log_sum_not_abstr += math.log(not_p_sumarry)
       
    if log_sum_abstr < log_sum_not_abstr:
        return True
    else:
        return False
def create_abstract_bayes(doc, procentages, word_count, dics, stop_words = False, lemma = False):    
    abstr = ""

    for sent in doc.sents:
        if is_sent_in_abstr_bayes(sent, procentages, word_count, dics, stop_words, lemma):
            abstr += sent.text
    
    return abstr

def tf(word, doc): #word e txt
    docSize = len([t for t in doc])
    appereances = 0
    for t in doc:
        if t.lemma_ == word:
            appereances += 1
    
    return appereances / docSize

def idf(word, docs):
    appereances = 1
    for item in docs.items():
        if word in item[1]:
            appereances += 1

    return math.log(len(docs) / appereances)

def tf_idf(word, doc, docs):
    return tf(word, doc) * idf(word, docs)

def similarity(sent, title_dic, title_size):
    similar_words = 0
    
    for token in sent:
        if token.lemma_ in title_dic:
            similar_words += 1
    
    return similar_words/title_size

def sent_tf_idf_score(sent, title_dic, title_size, doc, docs_appereance_noun, docs, noun, similarity1, similarity_weight):
    score = 0
    
    for token in sent:
        if noun:
            if(token.pos_ == "NOUN"):
                score += tf_idf(token.lemma_, doc, docs_appereance_noun)
        else:
            score += tf_idf(token.lemma_, doc, docs)
        
    if similarity1:
        score += similarity_weight * similarity(sent, title_dic, title_size) #0.8
    return score

def weight(x):
    return -x/6 + 1 

def weightp(x):
    return x

def create_abstract_tf_idf(doc, docs, docs_appereance_noun, title_dic, title_size, k, noun, similarity, similarity_weight, weight1):    
    abstr = ""
    docScore = []
    copy = []
    docList = list(doc.sents)
    sentNr = len(docList)
    
    for i in range(sentNr):
        procent = i / (sentNr - 1)
        if weight1:
            weight2 = 1 + weight(procent)
        else :
            weight2 = 1
        docScore.append((sent_tf_idf_score(docList[i], title_dic, title_size, doc, docs, docs_appereance_noun, noun=noun,
                                           similarity1=similarity, similarity_weight=similarity_weight) * weight2, docList[i]))
    
    copy = docScore.copy()
    
    copy.sort(reverse=True)
    
    copy = copy[:k]
    
    copy2 = []
    
    for t in copy:
        copy2.append(t[1].text)
    
    for (_, sent) in docScore:
        if sent.text in copy2:
            abstr += sent.text
    
    return abstr

def create_ngrams(n, doc):
    docSize = len(list(set([t for t in doc])))
    ngrams = []
    for i in range(docSize - n + 1):
        pair = ()
        for j in range(i, i + n):
            pair += (doc[j].text, )
        ngrams.append(pair)
    return ngrams


def rougen(n, docSrc, docTest):
    ngramsSrc = create_ngrams(n, docSrc)
    ngramsTest = create_ngrams(n, docTest)
    return len(list(set(ngramsSrc)&set(ngramsTest))) / len(list(set(ngramsSrc)))
    
    
def bleun(n, docSrc, docTest):
    ngramsSrc = create_ngrams(n, docSrc)
    ngramsTest = create_ngrams(n, docTest)
    return len(list(set(ngramsSrc)&set(ngramsTest))) / len(list(set(ngramsTest)))    

nlp = spacy.load("en_core_web_sm")
nlp.remove_pipe("ner")

news_sent_count = 0 
abstr_sent_count = 0   

news_word_count = 0 
abstr_word_count = 0   

news_words = {}
abstr_words = {}
news_but_not_abstr = {}

docs_appereance = {}
docs_appereance_noun = {}

#PARAMETERS
summ_type = "tf-idf" #bayes or tf-idf

noun=True
similarity1=True
similarity_weight = 0.8 # 0.4 0.8 1.2
weight1=True
sent_count = 3

lemma = True
stop_words = True

root = "BBC News Summary"
typesText = ["News Articles", "Summaries"]
#classesText = ["business", "entertainment", "politics", "sport", "tech"]
classesText = ["entertainment"]

k = 1
print_texts = [380]

#END PARAMETERS

for type in typesText:
    for classText in classesText:
        classUrl = root + "/" + type + "/" + classText
        docs = glob.glob(classUrl + "/*.txt")
        doc_count = len(docs)
        
        for i in range(0, int(doc_count*3/4)):
            text = open(docs[i], 'r').read()
            doc = nlp(text);
            if type == "News Articles":
                t = (classText, i)
                docs_appereance[t] = {}
                docs_appereance_noun[t] = {}
                news_sent_count += len(list(doc.sents))
                for token in doc:
                    docs_appereance[t][token.lemma_] = token.pos_
                    if(token.pos_ == "NOUN" or token.pos_ == "PROPN"):
                        docs_appereance_noun[t][token.lemma_] = token.pos_
                    if lemma:
                        add_dict(token.lemma_, news_words)
                    else:
                        add_dict(token.text, news_words)
                    news_word_count += 1
            else:
                abstr_sent_count += len(list(doc.sents))
                for token in doc:
                    if lemma:
                        add_dict(token.lemma_, abstr_words)
                    else:
                        add_dict(token.text, abstr_words)
                    abstr_word_count += 1 

p_sumarry = abstr_sent_count/(news_sent_count);
not_p_sumarry = 1 - p_sumarry;
(news_but_not_abstr, news_but_not_abstr_word_count) = sub_dic(news_words, abstr_words)

sum_blue = 0
sum_red = 0

res = []

for classText in classesText:
    classUrl = "BBC News Summary/News Articles/" + classText
    docs = glob.glob(classUrl + "/*.txt")
    doc_count = len(docs)
    
    for i in range(int(doc_count*3/4), int(doc_count)):
        classUrl = root + "/" + "News Articles" + "/" + classText
        docs = glob.glob(classUrl + "/*.txt")
        textS = open(docs[i], 'r').read()
        docSrc = nlp(textS)
        
        classUrl = root + "/" + "Summaries" + "/" + classText
        docs = glob.glob(classUrl + "/*.txt")
        textA = open(docs[i], 'r').read()
        docAbstr = nlp(textA)
        
        procentages = (p_sumarry, not_p_sumarry)
        word_count = (abstr_word_count, news_but_not_abstr_word_count)
        dics = (abstr_words, news_but_not_abstr)
        
        if summ_type == "bayes":
            textAbstr = create_abstract_bayes(docSrc, procentages, word_count, dics, stop_words = stop_words, lemma = lemma)
        
        if summ_type == "tf-idf":
            title = list(docSrc.sents)[0]
            titel_dic = {}
            title_size = len(title)
            for token in title:
                titel_dic[token.text] = True
            textAbstr = create_abstract_tf_idf(docSrc, docs_appereance, docs_appereance_noun, titel_dic, title_size, sent_count,
                                               noun=noun, similarity=similarity1, similarity_weight=similarity_weight, weight1 = weight1)
        
        docTest = nlp(textAbstr)
                
        if i in print_texts:
            print(i)
            print("Articol---------------\n")
            print(textS)
            print("Sumar uman------------\n")
            print(textA)
            print("Sumar automat---------\n")
            print(textAbstr)
            print("---------------")
        
        red = rougen(k, docAbstr, docTest)
        bleu = bleun(k, docAbstr, docTest)
        
        res.append((i, (bleu, red)))
        
        sum_blue += bleu
        sum_red += red


xb = []
yb = []
for item in res:
    xb.append(item[0])
    yb.append(item[1][0])
    
xr = []
yr = []
for item in res:
    xr.append(item[0])
    yr.append(item[1][1])

plt.plot(xb, yb, label = "bleu "+ str(k) +"")
plt.plot(xr, yr, label = "rouge "+ str(k) +"")
plt.xlabel('doc number')
plt.ylim(0, 1)
plt.legend()
plt.show()

print("Average bleu " + str(sum_blue/len(res)) + " average rouge " + str(sum_red/len(res)) + " for k = " + str(k))