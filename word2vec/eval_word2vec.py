import gensim
import random
import logging

# configuration
trains = "../../temp_results/word2vec_hindi.txt"
create = 1
topn = 10

data_folder = '../data/word2vec_evaluation/'
TARGET_SYN     = data_folder+'syntactic.questions.txt'
TARGET_SEM_OP  = data_folder+'semantic_op.questions.txt'
TARGET_SEM_BM  = data_folder+'semantic_bm.questions.txt'
TARGET_SEM_DF  = data_folder+'semantic_df.questions.txt'
SRC_NOUNS      = data_folder+'nouns.txt'
SRC_BESTMATCH  = data_folder+'bestmatch.txt'
SRC_DOESNTFIT  = data_folder+'doesntfit.txt'
SRC_OPPOSITE   = data_folder+'opposite.txt'
PATTERN_SYN = [('nouns', 'SI/PL', SRC_NOUNS, 0, 1)]
#logger.write(filename=train.strip() + '.result', format='%(asctime)s : %(message)s', level=logging.INFO)
print ("TEST")
# function create_syntactic_testset
# ... creates syntactic test set and writes it into a file
# @return void
def create_syntactic_testset():
    print ("TEST")
    with open(TARGET_SYN, 'w') as t:
        for label, short, src, index1, index2 in PATTERN_SYN:
            t.write(': ' + label + ': ' + short + '\n')
            for q in create_questions(src, index1, index2):
                t.write(q + '\n')


# function create_semantic_testset
# ... creates semantic test set and writes it into a file
# @return void
def create_semantic_testset():
    # opposite
    print ("TEST")
    with open(TARGET_SEM_OP, 'w') as t:
        for q in create_questions(SRC_OPPOSITE):
            t.write(q + '\n')
        logging.info('created opposite questions')
    # best match
    with open(TARGET_SEM_BM, 'w') as t:
        groups = open(SRC_BESTMATCH).read().split(':')
        groups.pop(0) # remove first empty group
        for group in groups:
            questions = group.splitlines()
            questions.pop(0)
            while questions:
                for i in range(1,len(questions)):
                    question = questions[0].split('-') + questions[i].split('-')
                    t.write(' '.join(question) + '\n')
                questions.pop(0)
    # doesn't fit
    with open(TARGET_SEM_DF, 'w') as t:
        for line in open(SRC_DOESNTFIT):
            words = line.split()
            for wrongword in words[-1].split('-'):
                question = ' '.join(words[:3] + [wrongword])
                t.write(question + '\n')


# function create_questions
# ... creates single questions from given source
# @param string  src    source file to load words from
# @param integer index2    index of first word in a line to focus on
# @param integer index2    index of second word in a line to focus on
# @param integer combinate number of combinations with random other lines
# @return list of question words
def create_questions(src, index1=0, index2=1):
    # get source content
   
    with open(src) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        
    questions = []

    for line in content:
        for i in range(0, 10):
            # get current word pair
            question = list(line.split('-'))
            # get random word pair that is not the current
            random_line = random.choice(list(set(content) - {line}))
            random_word = list(random_line.split('-'))
            # merge both word pairs to one question
            question.extend(random_word)
            questions.append(' '.join(question))
    print (len(questions))
    return questions


# function test_mostsimilar
# ... tests given model to most similar word
# @param word2vec model to test
# @param string   src   source file to load words from
# @param string   label to print current test case
# @param integer  topn  number of top matches
def test_mostsimilar(model, src, label='most similar', topn=5):
    
    num_lines = sum(1 for line in open(src))
    num_questions = 0
    num_right = 0
    num_topn = 0
    # get questions
    import codecs
    with codecs.open(src,encoding='utf-8') as f:
        questions = f.readlines()
        questions = [x.strip() for x in questions]
    
    # test each question
    for question in questions:
        words = question.split()
        # check if all words exist in vocabulary
        if all(x in model.index2word for x in words):
            num_questions += 1
            bestmatches = model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=topn)
            # best match
            if words[3] in bestmatches[0]:
                num_right += 1
            # topn match
            for topmatches in bestmatches[:topn]:
                if words[3] in topmatches:
                    num_topn += 1
                    break
    # calculate result
    correct_matches = round(num_right/float(num_questions)*100, 1) if num_questions>0 else 0.0
    topn_matches = round(num_topn/float(num_questions)*100, 1) if num_questions>0 else 0.0
    coverage = round(num_questions/float(num_lines)*100, 1) if num_lines>0 else 0.0
    # log result
    print (correct_matches)
    print (topn_matches)
    print (coverage)

# function test_mostsimilar
# ... tests given model to most similar word
# @param word2vec model to test
# @param string   src   source file to load words from
# @param integer  topn  number of top matches
def test_mostsimilar_groups(model, src, topn=10):
    num_lines = 0
    num_questions = 0
    num_right = 0
    num_topn = 0
    # test each group
    groups = open(src).read().split('\n: ')
    for group in groups:
        questions = group.splitlines()
        label = questions.pop(0)
        label = label[2:] if label.startswith(': ') else label # handle first group
        num_group_lines = len(questions)
        num_group_questions = 0
        num_group_right = 0
        num_group_topn = 0
        # test each question of current group
        for question in questions:
            words = question.decode('utf-8').split()
            # check if all words exist in vocabulary
            if all(x in model.index2word for x in words):
                num_group_questions += 1
                bestmatches = model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=topn)
                # best match
                if words[3] in bestmatches[0]:
                    num_group_right += 1
                # topn match
                for topmatches in bestmatches[:topn]:
                    if words[3] in topmatches:
                        num_group_topn += 1
                        break
        # calculate result
        correct_group_matches = round(num_group_right/float(num_group_questions)*100, 1) if num_group_questions>0 else 0.0
        topn_group_matches = round(num_group_topn/float(num_group_questions)*100, 1) if num_group_questions>0 else 0.0
        group_coverage = round(num_group_questions/float(num_group_lines)*100, 1) if num_group_lines>0 else 0.0
        # log result
        # total numbers
        num_lines += num_group_lines
        num_questions += num_group_questions
        num_right += num_group_right
        num_topn += num_group_topn
    # calculate result
    correct_matches = round(num_right/float(num_questions)*100, 1) if num_questions>0 else 0.0
    topn_matches = round(num_topn/float(num_questions)*100, 1) if num_questions>0 else 0.0
    coverage = round(num_questions/float(num_lines)*100, 1) if num_lines>0 else 0.0

# function test_doesntfit
# ... tests given model to most not fitting word
# @param word2vec model to test
# @param string   src   source file to load words from
def test_doesntfit(model, src):
    num_lines = sum(1 for line in open(src))
    num_questions = 0
    num_right = 0
    # get questions
    with open(src) as f:
        questions = f.readlines()
        questions = [x.strip() for x in questions]
    # test each question
    for question in questions:
        words = question.decode('utf-8').split()
        # check if all words exist in vocabulary
        if all(x in model.index2word for x in words):
            num_questions += 1
            if model.doesnt_match(words) == words[3]:
                num_right += 1
    # calculate result
    correct_matches = round(num_right/float(num_questions)*100, 1) if num_questions>0 else 0.0
    coverage = round(num_questions/float(num_lines)*100, 1) if num_lines>0 else 0.0
                
if create == 1:
    create_syntactic_testset()
    create_semantic_testset()

# get trained model
model = gensim.models.KeyedVectors.load_word2vec_format(trains.strip())
print ("word 2 vec read successfully.")
# execute evaluation

test_mostsimilar_groups(model, TARGET_SYN, topn)
test_mostsimilar(model, TARGET_SEM_OP, 'opposite', topn)
test_mostsimilar(model, TARGET_SEM_BM, 'best match', topn)
test_doesntfit(model, TARGET_SEM_DF)