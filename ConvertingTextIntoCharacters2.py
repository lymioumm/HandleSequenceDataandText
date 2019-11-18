import torch
from torchtext import data
from torchtext.datasets import IMDB

# field 字段
def manifest():

    # set up fields
    TEXT = data.Field(lower = True,batch_first = True,fix_length = 40)
    LABEL = data.Field(sequential=False)

    # make splits for data
    train,test = IMDB.splits(TEXT,LABEL)
    # print(f'type(train):\n{type(train)}')    # <class 'torchtext.datasets.imdb.IMDB'>

    fields_ = train.fields
    len_ = len(train)
    vars_ = vars(train[0])
    print('train.fields', train.fields)        # train.fields {'text': <torchtext.data.field.Field object at 0x2ab0d2060828>, 'label': <torchtext.data.field.Field object at 0x2ab0d20608d0>}
    print('len(train)', len(train))            # len(train) 25000
    print('vars(train[0])', vars(train[0]))    # vars(train[0]) {'text': ['deliverance', 'is', 'a', 'stunning', 'thriller,', 'every', 'bit', 'as', 'exciting', 'as', 'any', 'good', 'thriller', 'should', 'aspire', 'to', 'be', 'but', 'also', 'stomach-churningly', 'frightening.', 'though', 'it', 'is', 'not', 'a', 'horror', 'movie,', 'it', 'is', 'just', 'as', 'terrifying', 'as', 'any', 'classic', 'horror', 'film.', 'the', 'very', 'thought', 'of', 'being', 'a', 'normal', 'red-blooded', 'male', 'enjoying', 'an', 'adventure', 'weekend', 'miles', 'from', 'any', 'form', 'of', 'civilisation,', 'only', 'to', 'be', 'captured', 'and', 'sodomised', 'by', 'a', 'couple', 'of', 'violent', 'hillbillies,', 'is', 'surely', 'the', 'worst', 'nightmare', 'of', '99.9%', 'of', 'the', "world's", 'population.', 'it', 'would', 'have', 'been', 'easy', 'for', 'deliverance', 'to', 'slip', 'into', 'exploitation', 'territory,', 'but', 'john', 'boorman', 'has', 'cleverly', 'avoided', 'the', 'temptation', 'to', 'go', 'down', 'such', 'a', 'route', 'and', 'has', 'made', 'a', 'film', 'that', 'explores,', 'questions', 'and', 'challenges', 'the', 'very', 'meaning', 'of', 'masculinity.', 'with', 'so', 'many', 'films,', 'you', 'come', 'away', 'wishing', 'to', 'heaven', 'that', 'you', 'could', 'step', 'into', 'the', "hero's", 'shoes,', 'performing', 'heroic', 'deeds', 'and', 'saving', 'the', 'day', 'and', 'getting', 'the', 'girl....', 'but', 'with', 'deliverance,', 'you', 'come', 'away', 'praying', 'to', 'god', 'that', "you'll", 'never', 'have', 'to', 'experience', 'what', 'these', 'four', 'protagonists', 'go', 'through.<br', '/><br', '/>four', 'city', 'guys', '-', 'ed', '(jon', 'voight),', 'lewis', '(burt', 'reynolds),', 'drew', '(ronny', 'cox)', 'and', 'bobby', '(ned', 'beatty)', '-', 'head', 'out', 'into', 'the', 'wilderness', 'to', 'spend', 'a', 'few', 'days', 'canoing', 'down', 'a', 'soon-to-be-dammed', 'river.', 'the', 'guys', 'are', 'riding', 'the', 'rapids', 'in', 'pairs,', 'and', 'ed', 'and', 'bobby', 'inadvertently', 'get', 'a', 'little', 'too', 'far', 'ahead', 'of', 'the', 'others', 'so', 'they', 'pull', 'in', 'to', 'the', 'riverside', 'and', 'await', 'their', 'pals', 'in', 'the', 'adjacent', 'woodland.', 'here,', 'they', 'fall', 'foul', 'of', 'two', 'local', 'woodlanders', '(bill', 'mckinney', 'and', 'herbert', 'coward),', 'who', 'tie', 'ed', 'to', 'a', 'tree,', 'while', 'one', 'of', 'them', 'strips', 'and', 'rapes', 'bobby', 'instructing', 'him,', 'perversely,', 'to', '"squeal', 'like', 'a', 'pig".', 'lewis', 'and', 'drew', 'arrive', 'unseen', 'and', 'lewis,', 'being', 'a', 'fair', 'archer,', 'kills', 'the', 'rapist', 'while', 'the', 'other', 'hillbilly', 'beats', 'a', 'hasty', 'retreat', 'into', 'the', 'forest.', 'under', 'great', 'emotional', 'stress,', 'the', 'four', 'canoeists', 'decide', 'to', 'conceal', 'the', 'event', 'and', 'get', 'out', 'of', 'the', 'area.', 'but', 'they', 'find', 'the', 'river', 'increasingly', 'dangerous', 'to', 'negotiate', 'as', 'they', 'journey', 'downstream,', 'and', 'the', 'risk', 'to', 'their', 'lives', 'heightens', 'when', 'the', 'surviving', 'hillbilly', 'returns', 'to', 'take', 'shots', 'at', 'them', 'with', 'his', 'rifle', 'from', 'some', 'unseen', 'vantage', 'point', 'in', 'the', 'rocky', 'cliffs', 'beside', 'the', 'river.<br', '/><br', '/>deliverance', 'is', 'very', 'powerful', 'as', 'a', 'survival', 'tale,', 'but', 'even', 'more', 'powerful', '(and', 'disturbing)', 'as', 'a', 'study', 'of', 'macho', 'attitudes', 'being', 'torn', 'apart', 'and', 'left', 'in', 'humiliated', 'tatters.', 'though', 'all', 'the', 'performances', 'are', 'remarkable,', 'one', 'must', 'take', 'particular', 'note', 'of', "beatty's", 'efforts', 'in', 'a', 'role', 'that', 'many', 'actors', "would've", 'turned', 'down.', 'the', 'film', 'is', 'very', 'similar', 'thematically', 'to', 'the', '1971', 'film', 'straw', 'dogs', '-', 'both', 'films', 'deal', 'with', 'terrifying', 'sexual', 'violence', 'in', 'isolated', 'locales,', 'and', 'in', 'both', 'the', 'eventual', 'violent', 'revenge', 'exacted', 'by', 'the', 'victim', 'does', 'not', 'result', 'in', 'any', 'sense', 'of', 'satisfaction.', 'the', 'backdrop', 'of', 'the', 'rugged', 'countryside', 'in', 'deliverance', 'is', 'beautiful', 'to', 'look', 'at,', 'but', 'it', 'also', 'adds', 'to', 'the', 'tension', 'by', 'placing', 'the', 'four', 'canoeists', 'in', 'a', 'setting', 'where', 'they', 'are', 'at', 'the', 'mercy', 'of', 'the', 'hillbillies', 'and', 'the', 'landscape,', 'with', 'nobody', 'to', 'rely', 'on', 'other', 'than', 'themselves.', 'this', 'truly', 'is', 'suspenseful', 'film-making', 'at', 'its', 'finest.'], 'label': 'pos'}














def main():
    manifest()


    pass

if __name__ == '__main__':
    main()