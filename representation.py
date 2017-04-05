#  Mappers for converting input into a representation which can be fed into a classifier
#  We might want different representations and mappers for different tasks
import numpy as np
import reader
from keras.preprocessing import sequence
import math


class VeryStupidCBOWMapper:
    def __init__(self, vsm, window=2, sentence_boundaries=True):
        self.vsm = vsm
        self.window = window
        self.sentence_boundaries = sentence_boundaries


    def to_repr(self, entity, document):
        covered_tokens = document.cover_index[entity.uid]
        domain = document.domain

        if self.sentence_boundaries:
            sentence = covered_tokens[0].sentence
            first_token = (sentence.token_array[0]).tid
            last_token = (sentence.token_array[-1]).tid
        else:
            first_token = 0
            last_token = len(document.tokens)-1

        left_min_index = max(first_token, covered_tokens[0].tid - self.window)
        left_max_index = covered_tokens[0].tid

        if left_max_index <= left_min_index:
            context_left = []
        else:
            context_left = document.tokens[left_min_index:left_max_index]

        right_min_index = covered_tokens[-1].tid + 1
        right_max_index = min(last_token, covered_tokens[-1].tid + self.window + 1)

        if right_min_index >= right_max_index:
            context_right = []
        else:
            context_right = document.tokens[right_min_index:right_max_index]

        covered_emb = np.mean([self.vsm.get(t.string, domain) for t in covered_tokens], axis=0)
        context_left_emb = np.mean([self.vsm.get(t.string, domain) for t in context_left], axis=0) \
            if len(context_left) > 0 else np.zeros(self.vsm.dim)
        context_right_emb = np.mean([self.vsm.get(t.string, domain) for t in context_right], axis=0) \
            if len(context_right) > 0 else np.zeros(self.vsm.dim)

        return np.concatenate((context_left_emb, covered_emb, context_right_emb), axis=0)


class ConcatMapper:
    def __init__(self, vsm, window=2, sentence_boundaries=True):
        self.vsm = vsm
        self.window = window
        self.sentence_boundaries = sentence_boundaries

    def to_repr(self, entity, document):
        covered_tokens = document.cover_index[entity.uid]
        domain = document.domain

        if self.sentence_boundaries:
            span = covered_tokens[0].sentence
            first_token = (span.token_array[0]).tid
            last_token = (span.token_array[-1]).tid
        else:
            span = reader.Token(document.tokens[0].start, document.tokens[-1].end, "")
            first_token = 0
            last_token = len(document.tokens)-1

        left_min_index = max(first_token, covered_tokens[0].tid - self.window)
        left_max_index = covered_tokens[0].tid
        if left_max_index <= left_min_index:
            context_left = []
        else:
            context_left = document.tokens[left_min_index:left_max_index]

        right_min_index = covered_tokens[-1].tid + 1
        right_max_index = min(last_token, covered_tokens[-1].tid + self.window + 1)
        if right_min_index >= right_max_index:
            context_right = []
        else:
            context_right = document.tokens[right_min_index:right_max_index]

        cl = len(context_left)
        cr = len(context_right)
        K=self.vsm.dim
        context_left = [reader.Token(span.start-1, span.start-1, "#BEGIN_OF_SENTENCE#")] * (self.window-cl) + context_left
        context_right = context_right + [reader.Token(span.end+1, span.end+1, "#END_OF_SENTENCE#")]*(self.window-cr)

        # take average embedding as representation
        covered_emb = np.mean([self.vsm.get(t.string, domain) for t in covered_tokens], axis=0)
        # take concatenated embedding as representation 
        # keep only the first m tokens: improve upon this
        m = 4
        if len(covered_tokens) > m:
        #  # simple heuristic: kick out short words
            for t in covered_tokens:
                if len(t.string) <= 3:
                    covered_tokens.remove(t)
                    if len(covered_tokens) <= m:
                        break
        #  covered_tokens = filter(lambda x: len(t.string)>3,covered_tokens) 
        my_center = np.concatenate([self.vsm.get(t.string, domain) for t in covered_tokens])
        covered_emb = sequence.pad_sequences([my_center],m*K,truncating="post",dtype="float32")[0] 
        context_left_emb = np.concatenate([self.vsm.get(t.string, domain) for t in context_left])
        context_right_emb = np.concatenate([self.vsm.get(t.string, domain) for t in context_right])
        # check if it is alright
        print([t.string for t in context_left],[t.string for t in covered_tokens],[t.string for t in context_right])

        return np.concatenate((context_left_emb, covered_emb, context_right_emb), axis=0)


class CharMapper:
    def __init__(self, vsm, window = 2, L=30,M=50,R=30, sentence_boundaries=True):
        self.vsm = vsm
        self.window = window
        self.globalHash = {}
        self.curVal = 0
	self.L = L
	self.M = M
	self.R = R
        self.sentence_boundaries = sentence_boundaries

    def to_repr(self, entity, document):
        covered_tokens = document.cover_index[entity.uid]
        if self.sentence_boundaries:
            span = covered_tokens[0].sentence
            first_token = (span.token_array[0]).tid
            last_token = (span.token_array[-1]).tid
        else:
            span = reader.Token(document.tokens[0].start, document.tokens[-1].end, "")
            first_token = 0
            last_token = len(document.tokens)-1

        left_min_index = max(first_token, covered_tokens[0].tid - self.window)
        left_max_index = covered_tokens[0].tid
        if left_max_index <= left_min_index:
            context_left = []
        else:
            context_left = document.tokens[left_min_index:left_max_index]

        right_min_index = covered_tokens[-1].tid + 1
        right_max_index = min(last_token, covered_tokens[-1].tid + self.window + 1)
        if right_min_index >= right_max_index:
            context_right = []
        else:
            context_right = document.tokens[right_min_index:right_max_index]

        cl = len(context_left)
        cr = len(context_right)
        K=100
        context_left = [reader.Token(span.start-1, span.start-1, "")] * (self.window-cl) + context_left
        context_right = context_right + [reader.Token(span.end+1, span.end+1, "")]*(self.window-cr)

        token_representation_covered = " ".join([t.string for t in covered_tokens])
        token_representation_left = " ".join([t.string for t in context_left])
        token_representation_right = " ".join([t.string for t in context_right])
        my_repr = []
        # this is a bit stupid
        # for submission, let's just do the character2index mapping offline
        for x in token_representation_covered:
            if x not in self.globalHash:
                self.globalHash[x] = self.curVal
                self.curVal += 1
            my_repr.append(self.globalHash[x])
        my_repr_left = []
        for x in token_representation_left:
            if x not in self.globalHash:
                self.globalHash[x] = self.curVal
                self.curVal += 1
            my_repr_left.append(self.globalHash[x])
        my_repr_right = []
        for x in token_representation_right:
            if x not in self.globalHash:
                self.globalHash[x] = self.curVal
                self.curVal += 1
            my_repr_right.append(self.globalHash[x])
        #print("%s\t%s\t%s"%(token_representation_left, token_representation_covered, token_representation_right))
        my_repr = list(sequence.pad_sequences([my_repr],self.M)[0])
        my_repr_left = list(sequence.pad_sequences([my_repr_left],self.L)[0])
        my_repr_right = list(sequence.pad_sequences([my_repr_right],self.R)[0])


        return my_repr_left+my_repr+my_repr_right

        # EOFunction


class IndexListMapper:
    def __init__(self, vsm, word_index, maxlen, context='right'):
        self.vsm = vsm
        self.word_index = word_index
        self.maxlen = maxlen
        self.context = context

    def to_repr(self, entity, document):
        if self.context == "right":
            return self.to_list_repr(entity, document)
        elif self.context == "both":
            return self.to_list_repr_both(entity, document)
        else:
            raise ValueError("IndexListMapper.context has to be 'right' or 'both'.")

    # left and right context
    def to_list_repr_both(self, entity, document):
        covered_tokens = document.cover_index[entity.uid]

        if len(covered_tokens) > self.maxlen:
            covered_tokens = covered_tokens[:self.maxlen]
            context_left = []
            context_right = []
        else:
            cl_len = int(math.floor((self.maxlen - len(covered_tokens))/2))
            cr_len = self.maxlen - len(covered_tokens) - cl_len
            context_left = document.tokens[max(0, covered_tokens[0].tid - cl_len):covered_tokens[0].tid]
            context_right = document.tokens[covered_tokens[-1].tid+1:min(len(document.tokens), covered_tokens[-1].tid + 1 + cr_len)]
            # pad
            context_left = [reader.Token(-1,-1,"#BEGIN_OF_TEXT#")]*(cl_len-len(context_left)) + context_left
            context_right = context_right + [reader.Token(-1,-1,"#END_OF_TEXT#")]*(cr_len-len(context_right))

        left = [t.string for t in context_left]
        right = [t.string for t in context_right]
        covered = [t.string for t in covered_tokens] 
        words = left + covered + right

        words_indexes = []
        for w in words:
            if w.lower() in self.word_index:
                words_indexes.append(self.word_index[w.lower()])
            else:
                words_indexes.append("-1")
        if len(words_indexes) != self.maxlen:
            raise ValueError("Only",len(words_indexes),"words, but should be",self.maxlen)
        return words_indexes


    # only right context
    def to_list_repr(self, entity, document):
        covered_tokens = document.cover_index[entity.uid]
        words = document.tokens[covered_tokens[0].tid:][:self.maxlen]
        words = words + [reader.Token(-1,-1,"#END_OF_TEXT#")]*(self.maxlen-len(words))

        words = [t.string for t in words]

        words_indexes = []
        for w in words:
            if w.lower() in self.word_index:
                words_indexes.append(self.word_index[w.lower()])
            else:
                words_indexes.append("-1")
        if len(words_indexes) != self.maxlen:
            raise ValueError("Only",len(words_indexes),"words, but should be",self.maxlen)
        return words_indexes


if __name__ == "__main__":
    x = ["1", "2", "3"]
    context_left = x[1:2]
    context_left = 2 * ["-x1-"] + context_left
    print(context_left)
