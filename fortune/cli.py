'''
Command line interface
'''

import docopt
from . import train
from . import generate

HELPSTRING = '''machine-learning-fortune

Usage:
    machine-learning-fortune
    machine-learning-fortune train <quotes>

When run with no parameters, generate a new, random saying based on either the defaults
or the model stored at ${HOME}/.machine-learning-fortune.

When run to train, a file with one quote per line will be read, and then compiled into
${HOME}/.machine-learning-fortune.
'''

def execute():
    '''
    Execute the command line interface, parsing parameters from the command line, 
    and then dispatching.
    '''
    arguments = docopt.docopt(HELPSTRING)
    if arguments['train']:
        train.execute(arguments['<quotes>'])
    else:
        generate.execute()