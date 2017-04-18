import os
import argparse
import models
from settings import SAVED_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', 'evaluate', 'load'], help="Run/Load to use default hparams, Evaluate to use random hparams")
    parser.add_argument('model', choices=['auto-encoder', 'gan'])
    parser.add_argument('--tries', help='The number of tries for evaluate', default=1)
    parser.add_argument('--filename', help='The filename to load', default='model.pkl')
    args = parser.parse_args()

    # Model selection
    if args.model == 'auto-encoder':
        model = models.AutoEncoder
    elif args.model == 'gan':
        model = models.GenerativeAdversarialNetwork
    else:
        raise NotImplementedError

    # Mode selection
    if args.mode == 'run':
        m = model()
        m.build()
        m.run()

    elif args.mode == 'load':
        m = model()
        m.load(filename=args.filename)
        m.run()

    elif args.mode == 'evaluate':
        out_file = os.path.join(SAVED_DIR, 'random_search.txt')
        tries = args.tries
        for i in range(tries):
            print('------------------------')
            print('... Parameter search #%d' % (i))
            m = model()
            m.evaluate()
            with open(out_file, 'a') as f:
                f.write("%016.8f\t%s\r\n" % (m.hparams['eval_score'], m.hparams))
            print('================================')
