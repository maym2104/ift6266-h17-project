"""
Copyright (c) 2017 - Philip Paquette

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import os
import argparse
import models
from settings import SAVED_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', 'evaluate', 'load'], help="Run/Load to use default hparams, Evaluate to use random hparams")
    parser.add_argument('model', choices=['auto-encoder', 'gan', 'pixelCNN'])
    parser.add_argument('--tries', help='The number of tries for evaluate', default=1)
    parser.add_argument('--filename', help='The filename to load', default='model.pkl')
    args = parser.parse_args()

    # Model selection
    if args.model == 'auto-encoder':
        model = models.AutoEncoder
    elif args.model == 'gan':
        model = models.GenerativeAdversarialNetwork
    elif args.model == 'pixelCNN':
        model = models.PixelCNN
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
