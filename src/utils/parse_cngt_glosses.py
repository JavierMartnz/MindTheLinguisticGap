import re
from typing import Set, Union


def parse_cngt_gloss(gloss: str, sb_vocab):

    if gloss is None:
        return None

    gloss = gloss.strip()

    # gloss candidates separated by /
    gloss_candidates = gloss.split("/")

    # if there's more than a candidate, don't keep any of them
    if len(gloss_candidates) > 1:
        return None

    gloss = gloss_candidates[0]

    # empty gloss
    if gloss == '':
        return None

    # compound words are included in signbank, so the remnants of old notation are removed
    if '^' in gloss:
        return None

    # space delimited glosses aren't clear so remove them
    if ' ' in gloss:
        return None

    # remove uncertain glosses
    if '??' in gloss:
        return None

    # remove intended signs
    if '~' in gloss:
        return None

    # remove implicit negation
    if '-NOT' in gloss:
        return None

    # # remove pointing signs
    # if 'PT:' in gloss or 'PT-' in gloss:
    #     continue

    # # remove palm up signs
    # if gloss == 'PO':
    #     continue

    # remove classifiers
    classifier_tokens = ['MOVE', 'PIVOT', 'AT', 'BE']
    if any([token in gloss for token in classifier_tokens]):
        return None

    # remove non-visible signs (hidden behind for example body part but inferred from context)
    if '!' in gloss:
        return None

    # remove blended signs (SIGN+SIGN)
    regexp = r"\+(?:[\dA-Z]{2,})"
    match = re.search(regexp, gloss)
    if match:
        return None

    # remove glosses with _ since they're almost always part of a handshape
    # or similar annotation
    if '_' in gloss:
        return None

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
        # gloss = gloss.replace(gloss[match.start():match.end()], "")
        return None

    # we don't unify variants, since they are in the SignBank
    # regexp = r"\w+(?:-[A-Z])"
    # match = re.match(regexp, gloss)
    # if match:
    #     gloss = gloss[:gloss.index('-')]

    # keep signs with pointing, and don't modify the gloss since it's included with the pointing token in the SignBank
    # if gloss.startswith('1:'):
    #     gloss = gloss[2:]
    # if gloss.endswith(':1'):
    #     gloss = gloss[:-2]

    # before any parsing, we must unify the gloss to Dutch
    if gloss not in sb_vocab["gloss_to_id"].keys():  # if gloss is not in Dutch
        # if the gloss isn't found and cannot be translated, this next line returns None
        gloss = sb_vocab["english_to_dutch"].get(gloss)

    return gloss
