def ask_question(q_str, special='None', default=None):
    """
    function that ask the question, and parses the answer to prefered format
    q_str = string containing the question to be asked
    returns string, bool (spec='bool'), int (spec='int') or float (spec='float')
    """

    import sys
    
    while True:
        question = str(input(q_str)) if sys.version_info[0] == 3 \
                   else str(raw_input(q_str))
        funcs = {'None': [str, {}],
                 'bool': [bool, {'y': True, 'yes': True, 'n': False, 'no': False}],
                 'int': [int, {}], 'float': [float, {}]}
        
        if not question and default:
            question = str(default)
        if question in funcs[special][1]:
            return funcs[special][1][question]
        elif special is not 'bool':
            try:
                return funcs[special][0](question)
            except ValueError:
                pass
        print("Input not recognised. Please try again.")
