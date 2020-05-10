from sklearn.fea

class SpacyVectorizer(TfidfVectorizer):
    
    def is_clean_token(self, token):
        return (
            not token.is_stop                   # remove stop words
            and not token.is_digit              # remove digits
            and not token.like_num              # remove numbers
            and not token.like_email            # remove emails
            and not token.like_url              # remove URLs
            and token.is_alpha                  # keep only alphabetic tokens
            and token.is_ascii                  # keep only ascii tokens
            and len(token.lemma_.lower()) > 2   # keep only token which has length greater than two letters
        )
    
    def analyze_document(self, document):
        
        # apply the language pipeline on the passed document
        # for quicker execution, disable `parser` and `ner` pipeline steps
        doc = nlp(document, disable=['parser', 'ner'])
        
        # clean document
        tokens = [token.lemma_.lower() for token in doc if self.is_clean_token(token)] 
        
        return tokens
    
    def build_analyzer(self):
        return partial(self.analyze_document(document))