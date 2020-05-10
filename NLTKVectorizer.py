from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from functools import partial

class NLTKVectorizer(TfidfVectorizer):
    

    def __init__(self, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, analyzer='word', stop_words=None, token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1,1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False):
        super().__init__(input=input, encoding=encoding, decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer, stop_words=stop_words, token_pattern=token_pattern, ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features, vocabulary=vocabulary, binary=binary, dtype=dtype, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
        self.lemmatizer = WordNetLemmatizer()
    
    def tokenize_document(self, document):
        '''
        tokenize a document
        '''

        document_tokens = word_tokenize(document)

        return document_tokens

    def remove_non_alphanumeric_tokens(self, tokens_list):
        '''
        remove the tokens that are not alphanumeric (e.g. punctuations, special characters, etc ...)
        '''

        alphanumeric_tokens = [token for token in tokens_list if token.isalnum()]

        return alphanumeric_tokens

    def remove_numeric_tokens(self, tokens_list):
        '''
        remove the tokens that consists of only numeric characters
        '''

        non_numeric_tokens = [token for token in tokens_list if not token.isnumeric()]

        return non_numeric_tokens
    
    def lemmatize_document(self, tokens_list):
        '''
        use `NLTK` `WordNetLemmatizer` to lemmatize tokens
        '''

        lemmas = [self.lemmatizer.lemmatize(token) for token in tokens_list]

        return lemmas

    def convert_to_lowercase(self, document_tokens):
        '''
        convert tokens to lowercase
        '''

        lowercase_tokens = [token.lower() for token in document_tokens]
        
        return lowercase_tokens

    def remove_stop_words(self, document_tokens):
        '''
        remove stopwords from the document tokens
        '''
        pass

    # def is_clean_token(self, token):
    #     return (
    #         not token.is_stop                   # remove stop words
    #         and not token.is_digit              # remove digits
    #         and not token.like_num              # remove numbers
    #         and not token.like_email            # remove emails
    #         and not token.like_url              # remove URLs
    #         and token.is_alpha                  # keep only alphabetic tokens
    #         and token.is_ascii                  # keep only ascii tokens
    #         and len(token.lemma_.lower()) > 2   # keep only token which has length greater than two letters
    #     )
    
    # def analyze_document(self, document):
        
    #     # apply the language pipeline on the passed document
    #     # for quicker execution, disable `parser` and `ner` pipeline steps
    #     doc = nlp(document, disable=['parser', 'ner'])
        
    #     # clean document
    #     tokens = [token.lemma_.lower() for token in doc if self.is_clean_token(token)] 
        
    #     return tokens

    def analyze_document(self, document):

        # tokenize the document
        document_tokens = self.tokenize_document(document)

        # remove tokens which are not alpha numeric
        document_tokens = self.remove_non_alphanumeric_tokens(document)

        # remove only numeric tokens
        document_tokens = self.remove_numeric_tokens(document)

        # lemmatize tokens
        document_tokens = self.lemmatize_document(document)

        # convert tokens to lower case
        document_tokens = self.convert_to_lowercase(document_tokens)

        # remove wtopwords
        document_tokens = self.remove_stop_words(document_tokens)

        pass
    
    def build_analyzer(self):
        return partial(self.analyze_document(document))