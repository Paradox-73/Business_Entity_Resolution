"""Map detector flags to generator-level NAME operations (one generator op can raise several flags)."""
def name_groups(o):
    o = set(o)
    g = set()
    if 'n_script' in o: g.add('SCRIPT')
    if 'n_missing' in o: g.add('NMISSING')
    if 'n_domain' in o: g.add('DOMAIN'); return g
    if 'n_squash' in o: g.add('SQUASH'); o -= {'n_lower', 'n_upper'}
    if 'n_acronym' in o: g.add('ACRONYM'); o -= {'n_upper', 'n_lower'}
    if 'n_tradingas' in o:
        g.add('TRADINGAS'); o = {x for x in o if not x.startswith('n_add:') and not x.startswith('n_swap:')}
    if 'n_idtag' in o:
        g.add('IDTAG'); o = {x for x in o if not x.startswith('n_add:other')}
    if 'n_amp_swap' in o:
        g.add('AMP'); o = {x for x in o if not x.startswith('n_swap:noise') and x != 'n_add:noise' and x != 'n_drop:noise'}
    if o & {'n_lower', 'n_upper', 'n_case_other', 'n_title'}: g.add('CASE')
    if o & {'n_acc_add', 'n_acc_strip'}: g.add('ACCENT')
    if 'n_dblspace' in o: g.add('DBLSPACE')
    if 'n_bracket' in o: g.add('BRACKET')
    if 'n_hyphen' in o: g.add('HYPHEN')
    if o & {'n_word_order', 'n_comma_add'}: g.add('ORDER')
    if 'n_legal_dot' in o: g.add('LEGAL_DOT')
    if 'n_legal_abbrev' in o: g.add('LEGAL_ABBREV')
    if 'n_legal_drop' in o: g.add('LEGAL_DROP')
    if 'n_legal_add' in o: g.add('LEGAL_ADD')
    if o & {'n_legal_change', 'n_legal_partial'}: g.add('LEGAL_CHANGE')
    if any(x.startswith('n_typo') for x in o): g.add('TYPO')
    if 'n_leet' in o: g.add('LEET')
    if 'n_word_dup' in o: g.add('DUP')
    if 'n_add:title' in o or 'n_drop:title' in o: g.add('TITLE')
    if any(x.startswith('n_drop:') and x != 'n_drop:title' for x in o): g.add('DROP')
    if any(x.startswith('n_add:') and x != 'n_add:title' for x in o): g.add('ADD')
    if any(x.startswith('n_swap:') for x in o): g.add('SWAP')
    return g

CONTENT = {'TYPO', 'LEET', 'SWAP', 'ADD', 'DROP', 'LEGAL_ADD', 'LEGAL_DROP', 'LEGAL_DOT', 'LEGAL_ABBREV', 'LEGAL_CHANGE',
           'TITLE', 'ORDER', 'SQUASH', 'DOMAIN', 'ACRONYM', 'TRADINGAS', 'AMP', 'IDTAG', 'DUP', 'LOWER'}


def num_state(o):
    o = set(o)
    if 'a_missing' in o: return 'AMISS'
    if 'a_num_missing' in o: return 'NMISS'
    up = [x for x in o if x.startswith('a_num_up')]
    dn = [x for x in o if x.startswith('a_num_down')]
    if up: return 'UP' + up[0][8:]
    if dn: return 'DOWN' + dn[0][10:]
    if 'a_num_add' in o: return 'NADD'
    return 'NSAME'


def signature(o):
    g = name_groups(o)
    if 'n_lower' in o and not ({'DOMAIN', 'SQUASH'} & g): g.add('LOWER')
    c = sorted(g & CONTENT)
    return c, num_state(o)
