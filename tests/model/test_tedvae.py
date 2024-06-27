import pytest
from unittest.mock import Mock
from model.tedvae import TEDVAE


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
    mock_arg.cuda = True
    return mock_arg


def test_TEDVAE():
    # 解决参数调用 . 的问题，不能用字典
    args = mock_args
    contfeats = 6
    binfeats = 19
    tedvae = TEDVAE(
        feature_dim=args.feature_dim, 
        continuous_dim=contfeats,
        binary_dim=binfeats,
        latent_dim=args.latent_dim, 
        latent_dim_t=args.latent_dim_t, 
        latent_dim_y=args.latent_dim_y,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_samples=10)
    tedvae.fit(
        
    )