from gensim.models.word2vec import Word2Vec
import codecs, sys
from nltk.tokenize import sent_tokenize, word_tokenize
import shutil, os, operator

def collect_sents(src):
    with codecs.open(src, "r", "utf-8") as inF:
        for line in inF:
            line = line.strip()
            if line:
                for s in sent_tokenize(line):
                    yield [w for w in word_tokenize(s)]

src_path = "/home/likewise-open/UKP/kuznetsov/Resources/corpora/ElsevierArticles20102016/plaintext/"

src = {}
src["CS"] = src_path+"cs_11.txt"
src["MS"] = src_path+"ms_19.txt"
src["PH"] = src_path+"ph_24.txt"
#src["MAIN"] = ...

s = 0  # sentence index
w = 0  # word index
f = 0  # feature index
vocab = []

tgt_path = "/home/likewise-open/UKP/kuznetsov/Experiments/ScienceIE_Shared_Task/domain_embeddings/source/"
if os.path.exists(tgt_path):
    shutil.rmtree(tgt_path)
os.mkdir(tgt_path)
data_tgt = tgt_path+"data.txt"
vocab_tgt = tgt_path+"vocab.txt"
feat_tgt = tgt_path+"feat.txt"
vocab = {}

with codecs.open(data_tgt, "w", "utf-8") as dataOut:
    for domain in src:
        print "Preparing data for", domain
        for sent in collect_sents(src[domain]):
            for word in sent:
                vocab[word] = vocab.get(word, 0) + 1
            dataOut.write("\t".join([str(s), domain, " ".join(sent)])+"\n")
            s += 1
            if s%5000==0:
                print s
        print len(vocab), domain

with codecs.open(vocab_tgt, "w", "utf-8") as vocabOut:
    print "Preparing vocabulary"
    for x, y in sorted(vocab.items(), key=operator.itemgetter(1), reverse=True):
        vocabOut.write("\t".join([str(y), x])+"\n")
        w += 1

with codecs.open(feat_tgt, "w", "utf-8") as featOut:
    print "Preparing features"
    for domain in src:
        featOut.write(domain+"\n")
        f += 1

print "Done"




