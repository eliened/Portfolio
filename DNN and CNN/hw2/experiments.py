import os
import sys
import json
import torch
import random
import argparse
import itertools
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import numpy as np
from cs236781.train_results import FitResult

from .cnn import CNN, ResNet, YourCNN
from .mlp import MLP
from .training import ClassifierTrainer
from .classifier import ArgMaxClassifier, BinaryClassifier, select_roc_thresh

DATA_DIR = os.path.expanduser("~/.pytorch-datasets")

MODEL_TYPES = {
    ###
    "cnn": CNN,
    "resnet": ResNet,
    "ycn": YourCNN,
}


def mlp_experiment(
    depth: int,
    width: int,
    dl_train: DataLoader,
    dl_valid: DataLoader,
    dl_test: DataLoader,
    n_epochs: int,
):
    # TODO:
    #  - Create a BinaryClassifier model.
    #  - Train using our ClassifierTrainer for n_epochs, while validating on the
    #    validation set.
    #  - Use the validation set for threshold selection.
    #  - Set optimal threshold and evaluate one epoch on the test set.
    #  - Return the model, the optimal threshold value, the accuracy on the validation
    #    set (from the last epoch) and the accuracy on the test set (from a single
    #    epoch).
    #  Note: use print_every=0, verbose=False, plot=False where relevant to prevent
    #  output from this function.
    # ====== YOUR CODE: ======
    """X_test = []
    y_test = []

    for batch in dl_test:
        X, y= batch
        X_test.append(X)
        y_test.append(y)

    X_test = torch.concat(X_test, 0)
    y_test = torch.concat(y_test, 0)"""

    model, thresh, valid_acc, test_acc = None, None, None, None
    if width == 2 and depth == 1:
        # generation of the model
        mlp = MLP(2, [width] * depth, ['relu'] * depth)
        model = BinaryClassifier(mlp)
        loss_fn = torch.nn.CrossEntropyLoss()

        # HyperParameters
        lr = 0.01
        momentum = 0.9
        thresh = 0.5


        # Optimizer
        optim = torch.optim.SGD(model.parameters(), lr,momentum)
        trainer = ClassifierTrainer(model, loss_fn, optim)

        # Training
        fit_result = trainer.fit(dl_train, dl_valid, n_epochs, print_every=0, verbose = False)

        valid_acc = np.mean(fit_result.test_acc)

        # Testing
        test_accuracies = []
        thresh = select_roc_thresh(model, *dl_valid.dataset.tensors)
        model.threshold = thresh

        for batch in dl_test:
            X_test, y_test = batch
            results = model.classify(X_test)
            test_accuracies.append((torch.sum(results == y_test))/len(y_test))
        test_acc = np.mean(test_accuracies) * 100

    if width == 2 and depth == 2:
        # generation of the model
        mlp = MLP(2, [width]*2, ['relu','none'])
        model = BinaryClassifier(mlp)
        loss_fn = torch.nn.CrossEntropyLoss()

        # HyperParameters
        lr = 0.045
        momentum = 1
        thresh = 0.5


        # Optimizer
        optim = torch.optim.SGD(model.parameters(), lr)
        trainer = ClassifierTrainer(model, loss_fn, optim)

        # Training
        fit_result = trainer.fit(dl_train, dl_valid, n_epochs, print_every=0, verbose = False)

        valid_acc = np.mean(fit_result.test_acc)
        thresh = select_roc_thresh(model, *dl_valid.dataset.tensors)
        model.threshold = thresh
        # Testing
        test_accuracies = []

        for batch in dl_test:
            X_test, y_test = batch
            results = model.classify(X_test)
            test_accuracies.append((torch.sum(results == y_test))/len(y_test))
        test_acc = np.mean(test_accuracies) * 100

    if width == 2 and depth == 4:
        # generation of the model
        mlp = MLP(2, [width]*depth, ['relu','relu', 'relu','none'])
        model = BinaryClassifier(mlp)
        loss_fn = torch.nn.CrossEntropyLoss()

        # HyperParameters
        lr = 0.05
        momentum = 1
        thresh = 0.5


        # Optimizer
        optim = torch.optim.SGD(model.parameters(), lr)
        trainer = ClassifierTrainer(model, loss_fn, optim)

        # Training
        fit_result = trainer.fit(dl_train, dl_valid, n_epochs, print_every=0, verbose = False)

        valid_acc = np.mean(fit_result.test_acc)
        thresh = select_roc_thresh(model, *dl_valid.dataset.tensors)
        model.threshold = thresh

        # Testing
        test_accuracies = []

        for batch in dl_test:
            X_test, y_test = batch
            results = model.classify(X_test)
            test_accuracies.append((torch.sum(results == y_test))/len(y_test))
        test_acc = np.mean(test_accuracies) * 100

    if width == 8 and depth == 1:
        # generation of the model
        mlp = MLP(2, [width]*1 + [2], ['relu']*1 + ['none'])
        model = BinaryClassifier(mlp)
        loss_fn = torch.nn.CrossEntropyLoss()

        # HyperParameters
        lr = 0.53
        momentum = 0.1
        thresh = 0.5


        # Optimizer
        optim = torch.optim.SGD(model.parameters(), lr,momentum)
        trainer = ClassifierTrainer(model, loss_fn, optim)

        # Training
        fit_result = trainer.fit(dl_train, dl_valid, n_epochs, print_every=0, verbose = False)

        valid_acc = np.mean(fit_result.test_acc)


        # Testing
        test_accuracies = []

        thresh = select_roc_thresh(model, *dl_valid.dataset.tensors)
        model.threshold = thresh

        for batch in dl_test:
            X_test, y_test = batch
            results = model.classify(X_test)
            test_accuracies.append((torch.sum(results == y_test))/len(y_test))
        test_acc = np.mean(test_accuracies) * 100

    if width == 8 and depth == 2:
        # generation of the model
        mlp = MLP(2, [width]*depth + [2], ['relu']*depth + ['none'])
        model = BinaryClassifier(mlp)
        loss_fn = torch.nn.CrossEntropyLoss()

        # HyperParameters
        lr = 0.5
        momentum = 0.1
        thresh = 0.5


        # Optimizer
        optim = torch.optim.SGD(model.parameters(), lr,momentum)
        trainer = ClassifierTrainer(model, loss_fn, optim)

        # Training
        fit_result = trainer.fit(dl_train, dl_valid, n_epochs, print_every=0, verbose = False)

        valid_acc = np.mean(fit_result.test_acc)

        # Testing
        test_accuracies = []

        thresh = select_roc_thresh(model, *dl_valid.dataset.tensors)
        model.threshold = thresh

        for batch in dl_test:
            X_test, y_test = batch
            results = model.classify(X_test)
            test_accuracies.append((torch.sum(results == y_test))/len(y_test))
        test_acc = np.mean(test_accuracies) * 100

    if width == 8 and depth == 4:
        # generation of the model
        mlp = MLP(2, [width]*depth + [2], ['relu']*depth + ['none'])
        model = BinaryClassifier(mlp)
        loss_fn = torch.nn.CrossEntropyLoss()

        # HyperParameters
        lr = 0.49
        momentum = 0.05

        # Optimizer
        optim = torch.optim.SGD(model.parameters(), lr,momentum)
        trainer = ClassifierTrainer(model, loss_fn, optim)

        # Training
        fit_result = trainer.fit(dl_train, dl_valid, n_epochs, print_every=0, verbose = False)

        valid_acc = np.mean(fit_result.test_acc)

        # Testing
        test_accuracies = []

        thresh = select_roc_thresh(model, *dl_valid.dataset.tensors)
        model.threshold = thresh

        for batch in dl_test:
            X_test, y_test = batch
            results = model.classify(X_test)
            test_accuracies.append((torch.sum(results == y_test))/len(y_test))
        test_acc = np.mean(test_accuracies) * 100

    if width == 32 and depth == 1:
        # generation of the model
        mlp = MLP(2, [width]*depth + [2], ['relu']*depth + ['none'])
        model = BinaryClassifier(mlp)
        loss_fn = torch.nn.CrossEntropyLoss()

        # HyperParameters
        lr = 0.63
        momentum = 0.05

        # Optimizer
        optim = torch.optim.SGD(model.parameters(), lr,momentum)
        trainer = ClassifierTrainer(model, loss_fn, optim)

        # Training
        fit_result = trainer.fit(dl_train, dl_valid, n_epochs, print_every=0, verbose = False)

        valid_acc = np.mean(fit_result.test_acc)

        # Testing
        test_accuracies = []

        thresh = select_roc_thresh(model, *dl_valid.dataset.tensors)
        model.threshold = thresh

        for batch in dl_test:
            X_test, y_test = batch
            results = model.classify(X_test)
            test_accuracies.append((torch.sum(results == y_test))/len(y_test))
        test_acc = np.mean(test_accuracies) * 100

    if width == 32 and depth == 2:
        # generation of the model
        mlp = MLP(2, [width]*depth + [2], ['relu']*depth + ['none'])
        model = BinaryClassifier(mlp)
        loss_fn = torch.nn.CrossEntropyLoss()

        # HyperParameters
        lr = .1
        momentum = 0.05

        # Optimizer
        optim = torch.optim.SGD(model.parameters(), lr,momentum)
        trainer = ClassifierTrainer(model, loss_fn, optim)

        # Training
        fit_result = trainer.fit(dl_train, dl_valid, n_epochs, print_every=0, verbose = False)

        valid_acc = np.mean(fit_result.test_acc)

        # Testing
        test_accuracies = []

        thresh = select_roc_thresh(model, *dl_valid.dataset.tensors)
        model.threshold = thresh

        for batch in dl_test:
            X_test, y_test = batch
            results = model.classify(X_test)
            test_accuracies.append((torch.sum(results == y_test))/len(y_test))
        test_acc = np.mean(test_accuracies) * 100

    if width == 32 and depth == 4:
        # generation of the model
        mlp = MLP(2, [width]*depth + [2], ['relu']*depth + ['none'])
        model = BinaryClassifier(mlp)
        loss_fn = torch.nn.CrossEntropyLoss()

        # HyperParameters
        lr = .5
        momentum = 0.01

        # Optimizer
        optim = torch.optim.SGD(model.parameters(), lr,momentum)
        trainer = ClassifierTrainer(model, loss_fn, optim)

        # Training
        fit_result = trainer.fit(dl_train, dl_valid, n_epochs, print_every=0, verbose = False)

        valid_acc = np.mean(fit_result.test_acc)

        # Testing
        test_accuracies = []

        thresh = select_roc_thresh(model, *dl_valid.dataset.tensors)
        model.threshold = thresh

        for batch in dl_test:
            X_test, y_test = batch
            results = model.classify(X_test)
            test_accuracies.append((torch.sum(results == y_test))/len(y_test))
        test_acc = np.mean(test_accuracies) * 100

    if width == 128 and depth == 1:
        # generation of the model
        mlp = MLP(2, [width]*depth + [2], ['relu']*depth + ['none'])
        model = BinaryClassifier(mlp)
        loss_fn = torch.nn.CrossEntropyLoss()

        # HyperParameters
        lr = .45
        momentum = 0.01

        # Optimizer
        optim = torch.optim.SGD(model.parameters(), lr,momentum)
        trainer = ClassifierTrainer(model, loss_fn, optim)

        # Training
        fit_result = trainer.fit(dl_train, dl_valid, n_epochs, print_every=0, verbose = False)

        valid_acc = np.mean(fit_result.test_acc)

        # Testing
        test_accuracies = []

        thresh = select_roc_thresh(model, *dl_valid.dataset.tensors)
        model.threshold = thresh

        for batch in dl_test:
            X_test, y_test = batch
            results = model.classify(X_test)
            test_accuracies.append((torch.sum(results == y_test))/len(y_test))
        test_acc = np.mean(test_accuracies) * 100

    if width == 128 and depth == 2:
        # generation of the model
        mlp = MLP(2, [width]*depth + [2], ['relu']*depth + ['none'])
        model = BinaryClassifier(mlp)
        loss_fn = torch.nn.CrossEntropyLoss()

        # HyperParameters
        lr = .49
        momentum = 0.01

        # Optimizer
        optim = torch.optim.SGD(model.parameters(), lr,momentum)
        trainer = ClassifierTrainer(model, loss_fn, optim)

        # Training
        fit_result = trainer.fit(dl_train, dl_valid, n_epochs, print_every=0, verbose = False)

        valid_acc = np.mean(fit_result.test_acc)

        # Testing
        test_accuracies = []

        thresh = select_roc_thresh(model, *dl_valid.dataset.tensors)
        model.threshold = thresh

        for batch in dl_test:
            X_test, y_test = batch
            results = model.classify(X_test)
            test_accuracies.append((torch.sum(results == y_test))/len(y_test))
        test_acc = np.mean(test_accuracies) * 100

    if width == 128 and depth == 4:

        # generation of the model
        mlp = MLP(2, [width]*depth + [2], ['relu']*depth + ['none'])
        model = BinaryClassifier(mlp)
        loss_fn = torch.nn.CrossEntropyLoss()

        # HyperParameters
        momentum = 0.01
        lr = 0.4363636363636364

        # Optimizer
        optim = torch.optim.SGD(model.parameters(), lr,momentum)
        trainer = ClassifierTrainer(model, loss_fn, optim)

        # Training
        fit_result = trainer.fit(dl_train, dl_valid, n_epochs, print_every=0, verbose = False)

        valid_acc = np.mean(fit_result.test_acc)

        # Testing
        test_accuracies = []

        thresh = select_roc_thresh(model, *dl_valid.dataset.tensors)
        model.threshold = thresh

        for batch in dl_test:
            X_test, y_test = batch
            results = model.classify(X_test)
            test_accuracies.append((torch.sum(results == y_test))/len(y_test))
        test_acc = np.mean(test_accuracies) * 100


    # ========================


    print(f'with depth = {depth} and width = {width} valid = {valid_acc}, test = {test_acc}')
    return model, thresh, valid_acc, test_acc


def cnn_experiment(
    run_name,
    out_dir="./results",
    seed=None,
    device=None,
    # Training params
    bs_train=128,
    bs_test=None,
    batches=100,
    epochs=100,
    early_stopping=3,
    checkpoints=None,
    lr=1e-3,
    reg=1e-3,
    # Model params
    filters_per_layer=[64],
    layers_per_block=2,
    pool_every=2,
    hidden_dims=[1024],
    model_type="cnn",
    # You can add extra configuration for your experiments here
    conv_params: dict = {'kernel_size': 5, 'stride': 1, 'padding': 2},
    pooling_params: dict = {'kernel_size': 2},
    **kw,
):
    """
    Executes a single run of a Part3 experiment with a single configuration.

    These parameters are populated by the CLI parser below.
    See the help string of each parameter for it's meaning.
    """
    if not seed:
        seed = random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = max([bs_train // 4, 1])
    cfg = locals()

    tf = torchvision.transforms.ToTensor()
    ds_train = CIFAR10(root=DATA_DIR, download=True, train=True, transform=tf)
    ds_test = CIFAR10(root=DATA_DIR, download=True, train=False, transform=tf)



    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select model class
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")
    model_cls = MODEL_TYPES[model_type]

    # TODO: Train
    #  - Create model, loss, optimizer and trainer based on the parameters.
    #    Use the model you've implemented previously, cross entropy loss and
    #    any optimizer that you wish.
    #  - Run training and save the FitResults in the fit_res variable.
    #  - The fit results and all the experiment parameters will then be saved
    #   for you automatically.
    fit_res = None
    # ====== YOUR CODE: ======
    #Creation of the Dataloaders
    dl_train = DataLoader(ds_train, batch_size=bs_train,sampler=range(bs_train*batches))
    dl_test = DataLoader(ds_test, batch_size=bs_test, sampler=range(bs_test*batches))

    def make_layers(layers, L):
        new_channels = []
        for layer in layers:
            new_channels.extend([layer]*L)
        return new_channels

    cnn = None
    in_size = ds_train[0][0].shape
    n_classes = len(ds_train.classes)
    layers = make_layers(filters_per_layer, layers_per_block)

    if model_type == 'resnet':
        cnn = ResNet(in_size=in_size, out_classes=n_classes,channels=layers, pool_every=pool_every,
                     hidden_dims=hidden_dims,pooling_type='max', pooling_params=pooling_params, conv_params=conv_params,
                     activation_type='relu', dropout=0.2, batchnorm=True)
    elif model_type == 'cnn':
        cnn = CNN(in_size=in_size, out_classes=n_classes, channels=layers, pool_every=pool_every,
                  hidden_dims=hidden_dims, pooling_type='max', pooling_params=pooling_params, conv_params= conv_params,
                  activation_type='relu')
    else:
        cnn = YourCNN(in_size=in_size, out_classes=n_classes, channels=layers, pool_every=pool_every,
                      hidden_dims=hidden_dims)

    # Classifier over our model
    classifier = ArgMaxClassifier(cnn)
    classifier.to(device)

    # Optimization
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(classifier.parameters(),0.1)

    # Training
    trainer = ClassifierTrainer(classifier,loss_fn, optim,device)
    fit_res = trainer.fit(dl_train,dl_test,num_epochs=epochs)
    # ========================

    save_experiment(run_name, out_dir, cfg, fit_res)


def save_experiment(run_name, out_dir, cfg, fit_res):
    output = dict(config=cfg, results=fit_res._asdict())

    cfg_LK = (
        f'L{cfg["layers_per_block"]}_K'
        f'{"-".join(map(str, cfg["filters_per_layer"]))}'
    )
    output_filename = f"{os.path.join(out_dir, run_name)}_{cfg_LK}.json"
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"*** Output file {output_filename} written")


def load_experiment(filename):
    with open(filename, "r") as f:
        output = json.load(f)

    config = output["config"]
    fit_res = FitResult(**output["results"])

    return config, fit_res


def parse_cli():
    p = argparse.ArgumentParser(description="CS236781 HW2 Experiments")
    sp = p.add_subparsers(help="Sub-commands")

    # Experiment config
    sp_exp = sp.add_parser(
        "run-exp", help="Run experiment with a single " "configuration"
    )
    sp_exp.set_defaults(subcmd_fn=cnn_experiment)
    sp_exp.add_argument(
        "--run-name", "-n", type=str, help="Name of run and output file", required=True
    )
    sp_exp.add_argument(
        "--out-dir",
        "-o",
        type=str,
        help="Output folder",
        default="./results",
        required=False,
    )
    sp_exp.add_argument(
        "--seed", "-s", type=int, help="Random seed", default=None, required=False
    )
    sp_exp.add_argument(
        "--device",
        "-d",
        type=str,
        help="Device (default is autodetect)",
        default=None,
        required=False,
    )

    # # Training
    sp_exp.add_argument(
        "--bs-train",
        type=int,
        help="Train batch size",
        default=128,
        metavar="BATCH_SIZE",
    )
    sp_exp.add_argument(
        "--bs-test", type=int, help="Test batch size", metavar="BATCH_SIZE"
    )
    sp_exp.add_argument(
        "--batches", type=int, help="Number of batches per epoch", default=100
    )
    sp_exp.add_argument(
        "--epochs", type=int, help="Maximal number of epochs", default=100
    )
    sp_exp.add_argument(
        "--early-stopping",
        type=int,
        help="Stop after this many epochs without " "improvement",
        default=3,
    )
    sp_exp.add_argument(
        "--checkpoints",
        type=int,
        help="Save model checkpoints to this file when test " "accuracy improves",
        default=None,
    )
    sp_exp.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    sp_exp.add_argument("--reg", type=float, help="L2 regularization", default=1e-3)

    # # Model
    sp_exp.add_argument(
        "--filters-per-layer",
        "-K",
        type=int,
        nargs="+",
        help="Number of filters per conv layer in a block",
        metavar="K",
        required=True,
    )
    sp_exp.add_argument(
        "--layers-per-block",
        "-L",
        type=int,
        metavar="L",
        help="Number of layers in each block",
        required=True,
    )
    sp_exp.add_argument(
        "--pool-every",
        "-P",
        type=int,
        metavar="P",
        help="Pool after this number of conv layers",
        required=True,
    )
    sp_exp.add_argument(
        "--hidden-dims",
        "-H",
        type=int,
        nargs="+",
        help="Output size of hidden linear layers",
        metavar="H",
        required=True,
    )
    sp_exp.add_argument(
        "--model-type",
        "-M",
        choices=MODEL_TYPES.keys(),
        default="cnn",
        help="Which model instance to create",
    )

    parsed = p.parse_args()

    if "subcmd_fn" not in parsed:
        p.print_help()
        sys.exit()
    return parsed


if __name__ == "__main__":
    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    print(f"*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}")
    subcmd_fn(**vars(parsed_args))
