# Stuff for reading the data
# -*- coding: UTF-8 -*-

import codecs, os, glob
from extras import offset_tokenize
from extras import segment_text
from extras import utf8ify

import sys

# this wreaks havoc on the application if used with python3
if sys.version_info < (3,):
    reload(sys)
    sys.setdefaultencoding("utf-8")


# Entity container. Stores annotation information.
class Entity:
    def __init__(self, start, end, etype, string, uid, docid):
        self.start = start
        self.end = end
        self.etype = etype
        self.string = string
        self.uid = uid
        self.docid = docid

    def __str__(self):
        return "["+self.docid+"_"+self.uid+"("+str(self.start)+","+str(self.end)+","+str(self.string)+"):"+self.etype+"]"


# Relation container. Links entities.
class Relation:
    def __init__(self, from_entity, to_entity, label, directed, uid=None):
        self.from_entity = from_entity
        self.to_entity = to_entity
        self.label = label
        self.directed = directed

    def __str__(self):
        return str(self.from_entity)+"-"+self.label+("->" if self.directed else "-")+str(self.to_entity)


# Token container, nothing special.
class Token:
    def __init__(self, start, end, string, tid = None, sentence = None):
        self.start = start
        self.end = end
        self.string = string
        self.tid = tid
        self.sentence = sentence

    def __str__(self):
        return "[Token("+str(self.start)+","+str(self.end)+"):"+self.string+"]"


# Sentence container
class Sentence:
    def __init__(self, start, end, token_array, string, sid = None):
        self.start = start
        self.end = end
        self.token_array = token_array
        self.string = string
        self.sid = sid

    def __str__(self):
        return "[Sentence("+str(self.start)+","+str(self.end)+"):"+self.string+"]"


# Document container. Contains the mapping from tokens to entities.
class Document:
    def __init__(self, name, entities, relations, text, domain=None):
        self.name = name
        self.entities = entities
        self.relations = relations
        self.entity_index = {}
        self.token_index = {}
        self.sentence_index = {}
        self.cover_index = {}  # for each entity which tokens it includes
        self.text = text
        self.tokens = []
        self.sentences = []

        self.errors = 0
        self.fixed = 0
        self.index()
        self.domain = domain

    def index_sentences_and_tokens(self):
        segmented_text = segment_text(self.text)
        for sentence in segmented_text:
            self.sentence_index[sentence.sid] = sentence
            self.sentences.append(sentence)
            for token in sentence.token_array:
                self.token_index[token.tid] = token
                self.tokens.append(token)

    def index(self):
        for e in self.entities:
            self.entity_index[e.uid] = e
        for r in self.relations:
            r.from_entity = self.entity_index[r.from_entity]
            r.to_entity = self.entity_index[r.to_entity]

        self.index_sentences_and_tokens()

        for e in self.entities:
            covered_tokens = []
            for sentence in self.sentences:
                if e.start >= sentence.start and e.end <= sentence.end:
                    for t in sentence.token_array:
                        if t.start >= e.start and t.end <= e.end:
                            covered_tokens += [t]
                    break
            if len(covered_tokens)==0:  # ERR: annotation does not cover a single full token
#                print(str(self.name), "ERR: Not covering any token", e, " ".join([str(t.string) for t in self.tokens]))
                self.errors += 1
                for t in self.tokens:  # FIX: try to expand it to the single covering token
                    if t.start <= e.start and t.end >= e.end:
                        covered_tokens += [t]
                        self.fixed += 1
#                        print(str(self.name), "FIXED:", e, " ".join([str(t.string) for t in covered_tokens]))
                        break
            else:
                if covered_tokens[0].start!=e.start:  # ERR: annotation start doesn't match any token's start
#                    print(str(self.name), "ERR: no matching token start", e, " ".join([str(t.string) for t in covered_tokens]))
                    self.errors += 1
                    e.start = covered_tokens[0].start  # FIX: expand the annotatio to the left
                    self.fixed += 1
#                    print(str(self.name), "FIXED:", e, " ".join([str(t.string) for t in covered_tokens]))
                # ERR: annotation end doesn't match any token's end
                if covered_tokens[-1 if len(covered_tokens)>1 else 0].end!=e.end:
#                    print(str(self.name), "ERR: no matching token end", e, " ".join([str(t.string) for t in covered_tokens]))
                    self.errors += 1
                    # FIX: move the annotation to the last token's end offset
                    e.end = covered_tokens[-1 if len(covered_tokens)>1 else 0].end
                    self.fixed += 1
#                    print(str(self.name), "FIXED:", e, " ".join([str(t.string) for t in covered_tokens]))

            # important: we IGNORE entities that don't cover a token due to a data/tokenization glitch
            if len(covered_tokens) > 0:
                self.cover_index[e.uid] = covered_tokens

# Simple reader
class ScienceIEBratReader:
    def __init__(self, src_folder, domain_provider=None):
        self.files = []
        self.domain_provider = domain_provider
        for ann_file in glob.glob(os.path.join(src_folder, '*.ann')):
            txt_file = ann_file[:-4]+".txt"
            self.files += [(ann_file, txt_file)]

    def read(self):
        for (ann_file, txt_file) in self.files:
            entities = []
            rels = []
            with codecs.open(ann_file, "r", "utf-8") as annf:
                for line in annf.readlines():
                    line = line.strip()
                    if line:
                        docid = txt_file.split("/")[-1].split(".")[0]
                        if self.is_relation(line):
                            rels += self.parse_relation(line)
                        else:
                            entities += [self.parse_entity(line, docid)]
            with codecs.open(txt_file, "r", "utf-8") as f:
                txt = f.read().strip()

            domain = None
            if self.domain_provider is not None:
                docname = os.path.split(ann_file)[1]
                domain = self.domain_provider.get_domain(docname)
            yield Document(os.path.split(ann_file)[1], entities, rels, txt, domain)

    def is_relation(self, string):
        return not string.startswith("T")

    def parse_relation(self, string):
        uid, ann = string.strip().split("\t")
        relname, rest = ann.split(" ", 1)
        if relname == "Hyponym-of":
            relfrom, relto = rest.split(" ")
            relfrom = relfrom.split(":")[1]
            relto = relto.split(":")[1]
            return [Relation(relfrom, relto, relname, True)]  # For hyponyms, one relation is created
        elif relname == "Synonym-of":
            rels = []
            for t1 in rest.split(" "):
                for t2 in rest.split(" "):
                    if t1!=t2:
                        rels += [Relation(t1, t2, relname, False)]  # For synonyms, a set of relations is created for each pair
            return rels
        else:
            raise Exception("Unknown relation type")

    def parse_entity(self, string, docid):
        uid, ann, string = string.split("\t")
        if ";" not in ann:
            etype, start, end = ann.split(" ")
            return Entity(int(start), int(end), etype, utf8ify(string), uid, docid)
        else:
            # Multiwords are covered from first token's start to the last token's end
            spans = ann.split(";")
            etype = spans[0].split(" ")[0]
            start = spans[0].split(" ")[1]
            end = spans[-1].split(" ")[1]
            return Entity(int(start), int(end), etype, utf8ify(string), uid, docid)


if __name__ == "__main__":
    r = ScienceIEBratReader("/home/likewise-open/UKP/kuznetsov/Experiments/ScienceIE_Shared_Task/scienceie2017_train/train2")
    err = 0
    fix = 0
    total_entities = 0
    for d in r.read():
        err += d.errors
        fix += d.fixed
        total_entities += len(d.entities)
    print(err, fix, total_entities)


