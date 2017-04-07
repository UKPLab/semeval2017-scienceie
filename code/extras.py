# Various extras
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk.data
import re
import numpy as np
import reader
import sys
import codecs


def segment_text(text):
    sentence_id = 0
    token_id = 0
    tail = text
    accumulator = 0
    sentences = [sentence for sentence in SentenceSplitter().split(text)]
    sentence_object_array = []
    for sentence in sentences:
        escaped_sentence = re.escape(sentence)
        sentence_occurrence = re.search(escaped_sentence, tail)
        s_start, s_end = sentence_occurrence.span()
        sentence_start = accumulator + s_start
        sentence_end = accumulator + s_end

        tokens = [word for word in word_tokenize(sentence)]
        token_object_array = []
        tail_for_token_search = sentence
        token_accumulator = 0
        for token in tokens:
            escaped_token = re.escape(token)
            token_occurrence = re.search(escaped_token, tail_for_token_search)
            t_start, t_end = token_occurrence.span()
            # global offsets
            token_start = sentence_start + token_accumulator + t_start
            token_end = sentence_start + token_accumulator + t_end
            token_accumulator += t_end

            token_object = reader.Token(token_start, token_end, utf8ify(token), token_id)
            token_object_array.append(token_object)
            # keep searching in the rest
            tail_for_token_search = tail_for_token_search[t_end:]
            token_id += 1

        sentence_object = reader.Sentence(sentence_start, sentence_end, token_object_array, utf8ify(sentence), sentence_id)
        sentence_object_array.append(sentence_object)
        for tok in sentence_object.token_array:
            tok.sentence = sentence_object

        accumulator += s_end
        # keep rest of text for searching
        tail = tail[s_end:]
        sentence_id += 1

    return sentence_object_array


class SentenceSplitter:
    class Splitters:
        def __init__(self):
            pass

        Punkt, Normal = range(2)

    def __init__(self, splitter_name=Splitters.Punkt):
        self.splitterName = splitter_name
        if self.splitterName == SentenceSplitter.Splitters.Punkt:
            self.splitter = self.__punkt_sentence_splitter()
        else:
            self.splitter = self.__normal_sentence_splitter()

    def __punkt_sentence_splitter(self):
#        print("initializing punkt sentence splitter")
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        return lambda x: sent_tokenizer.tokenize(x)

    def __normal_sentence_splitter(self):
#        print("initializing default nltk sentence splitter")
        return lambda x: sent_tokenize(x)

    def split(self, text):
        return self.splitter(text)


def offset_tokenize(text):
    tail = text
    accum = 0
    tokens = [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]
    info_tokens = []
    for tok in tokens:
        scaped_tok = re.escape(tok)
        m = re.search(scaped_tok, tail)
        start, end = m.span()
        # global offsets
        gs = accum + start
        ge = accum + end
        accum += end
        # keep searching in the rest
        tail = tail[end:]
        info_tokens.append((tok, (gs, ge)))
    return info_tokens


class VSM:  # lookup provider for VSMs, e.g. http://nlp.stanford.edu/data/glove.6B.zip
    def __init__(self, src):
        self.map = {}
        self.dim = None
        self.source = src.split("/")[-1] if src is not None else "NA"
        # create dictionary for mapping from word to its embedding
        if src is not None:
            with open(src) as f:
                i = 0
                for line in f:
                    word = line.split()[0]
                    embedding = line.split()[1:]
                    self.map[word] = np.array(embedding, dtype=np.float32)
                    i += 1
                self.dim = len(embedding)
        else:
            self.dim = 1

    def get(self, word, domain=None):
        word = word.lower()  # glove is lowercase
        if word in self.map:
            return self.map[word]
        else:  # glove has no unknown token
            return np.zeros(self.dim)  # TODO: unknown words as zero vectors, not a good practice

class GeoVSM:
    def __init__(self, src):
        self.map = {}
        self.dim = None
        self.source = src.split("/")[-1] if src is not None else "NA"
        if src is not None:
            with open(src) as f:
                first_line = True
                i = 0
                for line in f:
                    line = line.strip()
                    if first_line and len(line.split())<=3:
                        first_line = False
                    else:
			first_line = False
                        domain = line.split(" ")[0].split("\t")[-1]
                        word = line.split(" ")[1]
                        embedding = line.split(" ")[2:]
                        self.map[domain] = self.map.get(domain, {})
                        self.map[domain][word] = np.array(embedding, dtype=np.float32)
                        i += 1
                self.dim = len(embedding)
        else:
            self.dim = 1
        print self.map.keys()

    def get(self, word, domain=None):
        if domain is None:
            domain = "MAIN"
        word = word.lower()  # glove is lowercase
        if word in self.map[domain] and domain != "MAIN":
            return self.map["MAIN"].get(word, np.zeros(self.dim))+self.map[domain][word]
        else:  # glove has no unknown token
            if word in self.map["MAIN"]:
                return self.map["MAIN"][word]
            else:
                return np.zeros(self.dim)


class DomainProvider():
    def __init__(self, src):
        firstLine = True
        header = []
        data = []
        with codecs.open(src, "r", "utf-8") as f:
            for line in f:
                line = line.strip().split("\t")[1:]
                if firstLine:
                    firstLine = False
                    header = line
                else:
                    data += [line]
        header = [x.split("/")[1].split(".")[0] for x in header]
        self.domain_data = {x: y for (x,y) in zip(header, data[0])}  # TODO add averaging/max vote

    def get_domain(self, filename):
        print filename
        filename = filename.split(".")[0]
        if filename not in self.domain_data:
            print filename, "NO DATA :("
            return None
        else:
            print filename, self.domain_data[filename]
            return self.domain_data[filename]


def __test_segmenter():
    print("\n-----------\nTest Segmenter")
    document = "A test sentence. Maybe, a second one as well. And fourth."
    for sentence in segment_text(document):
        print(sentence.sid, sentence)
        for token in sentence.token_array:
            print("\t", token.tid, token)
        first_sentence_token = (sentence.token_array[0]).tid
        last_sentence_token = (sentence.token_array[len(sentence.token_array)-1]).tid
        print("\t", first_sentence_token, last_sentence_token)


def __test_sentence_splitter():
    print("\n-----------\nTest Sentence Splitter")
    document = "A test sentence about Mr. Doe et. al. ends here. Maybe, a second one as well. And fourth."
    for sentence in SentenceSplitter().split(document):
        print(sentence)


def utf8ify(obj):
    if sys.version_info < (3,):
        return obj.encode("utf-8")
    else:
        return str(obj)


def read_and_map(src, mapper, y_values = None, domain_file = None):
    r = reader.ScienceIEBratReader(src)
    X = []
    y = []
    entities = []
    # r.read(domain_file) was to enable document classification. Since it doesn't help, we disabled it
    for document in r.read():
        for entity in document.entities:
            if entity.uid in document.cover_index:  # only proceed if entity has been successfully mapped to tokens
                X += [mapper.to_repr(entity, document)]
                y += [entity.etype]
                entities += [entity]

    X = np.vstack(X)

    y_values = y_values if y_values is not None else list(set(y))
    try:
      y = np.array([y_values.index(y_val) for y_val in y])
    except ValueError:
      y = np.array([0 for y_val in y])
    return X, y, y_values, entities

# str(entity.uid) + "\t" + str(entity.etype) + " " + str(entity.start) + " " + str(entity.end) + "\t" + utf8ify(entity.string))

def read_and_write(src,pred_list,outdir):
    r = reader.ScienceIEBratReader(src)
    entities = []
    i = 0
    default = "Material"
    for document in r.read():
	fout = open(outdir+"/"+document.name,"w")
        for entity in document.entities:
            if entity.uid in document.cover_index:  # only proceed if entity has been successfully mapped to tokens
                #X += [mapper.to_repr(entity, document)]
                #y += [entity.etype]
                #entities += [entity]
		pred_type = pred_list[i]
		i += 1
		fout.write(str(entity.uid) + "\t" + pred_type + " " + str(entity.start) + " " + str(entity.end) + "\t" + utf8ify(entity.string)+"\n")
	    else:
		sys.stderr.write("No valid prediction\n")
		fout.write(str(entity.uid) + "\t" + default + " " + str(entity.start) + " " + str(entity.end) + "\t" + utf8ify(entity.string)+"\n")
	fout.close()


if __name__ == "__main__":
    #__test_segmenter()
    #__test_sentence_splitter()
    vsm = GeoVSM("/home/likewise-open/UKP/kuznetsov/Experiments/ScienceIE_Shared_Task/domain_embeddings/out.embeddings")
    print vsm.get("graph", "CS")
    print vsm.get("graph", "PH")
    print vsm.get("graph", "MS")
    dp = DomainProvider("/home/likewise-open/UKP/kuznetsov/Experiments/ScienceIE_Shared_Task/scienceie2017_dev/domains_IK.tsv")
    print dp.get_domain("S0038092X14004824")
    print dp.get_domain("S0038092X14AA04824")
