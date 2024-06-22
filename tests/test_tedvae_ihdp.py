import numpy as np
import common
import logging
import pytest
from unittest.mock import Mock
from tedvae_ihdp import entry


@pytest.fixture
def mock_args():
    # 创建一个Mock对象来模拟命令行参数
    mock_arg = Mock()
    mock_arg.param1 = 'mock_value1'
    mock_arg.param2 = 'mock_value2'
    mock_arg.feature_dim=25
    mock_arg.latent_dim=20
    mock_arg.latent_dim_t=10
    mock_arg.latent_dim_y=10
    mock_arg.hidden_dim=500
    mock_arg.num_layers=4
    mock_arg.num_epochs=200
    mock_arg.batch_size=1000
    mock_arg.learning_rate=1e-3
    mock_arg.learning_rate_decay=0.01
    mock_arg.weight_decay=1e-4
    mock_arg.seed=1234567890
    mock_arg.jit = True
    mock_arg.cuda = False
    return mock_arg


def test_main_entry(mock_args):
    tedvae_pehe = np.zeros((100,1))
    tedvae_pehe_train = np.zeros((100,1))
    path = "./IHDP_b/"
    args = mock_args
    for i in range(1):
        logging.info("Dataset {:d}".format(i+1))
        tedvae_pehe[i,0], tedvae_pehe_train[i,0] = entry(args, i+1, path)