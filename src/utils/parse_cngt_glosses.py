import re
from typing import Set, Union

def parse_cngt_gloss(gloss: str, canonical_vocab: Set) -> Union[str, None]:
    '''
    Parse Corpus NGT glosses with same criteria that Looking for The Signs paper
    '''
    
    if gloss is None:
        return None
    
    gloss = gloss.strip()
    
    # only apply a vocaulary filter when a non-empty vocabulary set is passed
    apply_vocab_filter = bool(canonical_vocab)

    # convert our existing vocabulary to upper cases to treat each word as a gloss
    upper_vocab = set([word.upper() for word in canonical_vocab])
    
    # gloss condidates separated by /
    gloss_candidates = gloss.split("/")

    for gloss in gloss_candidates:
        
        if gloss == '':
            continue
        
        # compound words are included in signbank, so the remnants of old notation are removed
        if '^' in gloss:
            continue
            
        # space delimited glosses aren't clear so remove them
        if ' ' in gloss:
            continue
            
        # remove uncertain glosses
        if '??' in gloss:
            continue
            
        # remove intended signs
        if '~' in gloss:
            continue
        
        # remove implicit negation
        if '-NOT' in gloss:
            continue
    
        # remove pointing signs
        if 'PT:' in gloss or 'PT-' in gloss:
            continue
        
        # remove palm up signs
        if gloss == 'PO':
            continue
    
        # remove classifiers 
        classifier_tokens = ['MOVE', 'PIVOT', 'AT', 'BE']
        if any([token in gloss for token in classifier_tokens]):
            continue
    
        # remove non-visible signs
        if '!' in gloss:
            continue
    
        # remove blended signs (SIGN+SIGN)
        regexp = r"\+(?:[\dA-Z]{2,})"
        match = re.search(regexp, gloss)
        if match:
            continue
            
        # remove glosses with _ since they're almost always part of a handshape
        # or similar annotation
        if '_' in gloss:
            continue
    
        # keep fingerspelling
        if gloss.startswith('#'):
            gloss = gloss[1:]
            
        # keep inferred signs
        if gloss.endswith('?'):
            gloss = gloss[:-1]
            
        # remove signs with shape constructions and number articulated in signs
        # since in the signbank these type of signs are always constructors
        regexp = r"\+(?:[\dA-Z](?![\dA-Z]))"
        match = re.search(regexp, gloss)
        if match:
            #gloss = gloss.replace(gloss[match.start():match.end()], "")
            continue
        
        # we don't unify variants, since they are in the SignBank
        # regexp = r"\w+(?:-[A-Z])"
        # match = re.match(regexp, gloss)
        # if match:
        #     gloss = gloss[:gloss.index('-')]

        # keep signs with pointing, and don't modify the gloss since it's included
        # with the pointing token in the SignBank
        # if gloss.startswith('1:'):
        #     gloss = gloss[2:]
        # if gloss.endswith(':1'):
        #     gloss = gloss[:-2]
            
        # if no member of the supplied vocabulary is present in any part of the gloss,
        # we reject the word
        if apply_vocab_filter and not gloss in upper_vocab:
            continue
        
        if gloss in upper_vocab or not apply_vocab_filter:
            return gloss
        else:
            continue
    return None