# -*- coding: utf-8 -*-
#!/usr/bin/env python
from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import training
from chainer.training import extensions

class CNN(chainer.Chain):
    def __init__(self, n_units, n_out):
        w = I.Normal(scale=0.05) # initialize para
        super(MLP, self).__init__(
            conv1 = L.Convolution2D( 1, 16, 3, 1, 0), # (filter num : 16)
            conv2 = L.Convolution2D(16, 32, 3, 1, 0), # (filter num : 32)
            l3    = L.Linear(None, n_out, initialW=w),
        )
    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)),  ksize=2, stride=2) # 最大値プーリングは2×2，活性化関数はReLU
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), ksize=2, stride=2)
        y  = self.l3(h2)
        return y

def main():
    parser = argparse.ArgumentParser(description='Chainer CNN')
    parser.add_argument('--batchsize', '-b', type=int, default=100, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20, help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000, help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # train, test = chainer.datasets.get_mnist(ndim=3)

    import pickle
    dataset = pickle.load(open('./pn_dataset.dat', 'r''b'), encoding = 'bytes')

    x = dataset[b'data']
    y = dataset[b'target']
    x = x.astype(np.float32)
    y = y.astype(np.int32)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    train = np.vstack((x_train, y_train))
    test  = np.vstack((x_test, y_test))

    model = L.Classifier(MLP(args.unit, 10), lossfun=F.softmax_cross_entropy)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
    optimizer = chainer.optimizers.Adam()

    optimizer.setup(model)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter  = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)
    updater    = training.StandardUpdater(train_iter, optimizer, device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()
    model.to_cpu()

    modelname = args.out + "/cnn.model"
    print('save the trained model: {}'.format(modelname))
    chainer.serializers.save_npz(modelname, model)

if __name__ == '__main__':
    main()
