import warnings

import keras.backend
import keras_retinanet.bin.train
import pytest


@pytest.fixture(autouse=True)
def clear_session():
    # run before test (do nothing)
    yield
    # run after test, clear keras session
    keras.backend.clear_session()


def test_coco():
    # ignore warnings in this test
    warnings.simplefilter('ignore')

    # run training / evaluation
    keras_retinanet.bin.train.main([
        '--epochs=1',
        '--steps=1',
        '--no-weights',
        '--no-snapshots',
        'coco',
        'tests/test-data/coco',
    ])


def test_pascal():
    # ignore warnings in this test
    warnings.simplefilter('ignore')

    # run training / evaluation
    keras_retinanet.bin.train.main([
        '--epochs=1',
        '--steps=1',
        '--no-weights',
        '--no-snapshots',
        'pascal',
        'tests/test-data/pascal',
    ])


def test_csv():
    # ignore warnings in this test
    warnings.simplefilter('ignore')

    # run training / evaluation
    keras_retinanet.bin.train.main([
        '--epochs=1',
        '--steps=1',
        '--no-weights',
        '--no-snapshots',
        'csv',
        'tests/test-data/csv/annotations.csv',
        'tests/test-data/csv/classes.csv',
    ])


def test_vgg():
    # ignore warnings in this test
    warnings.simplefilter('ignore')

    # run training / evaluation
    keras_retinanet.bin.train.main([
        '--backbone=vgg16',
        '--epochs=1',
        '--steps=1',
        '--no-weights',
        '--no-snapshots',
        '--freeze-backbone',
        'coco',
        'tests/test-data/coco',
    ])
