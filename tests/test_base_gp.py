import pytest
from unittest.mock import Mock, patch
import torch
import numpy as np
from discobax import BaseGPModel


@pytest.fixture
def base_gp_model_instance():
    data_dim = 1
    instance = BaseGPModel(data_dim)
    return instance


@pytest.fixture
def mock_data_source():
    mock = Mock()
    mock.get_data.return_value = np.array([[1.0], [2.0], [3.0]])
    return mock


def test_predict(base_gp_model_instance, mock_data_source):
    result = base_gp_model_instance.predict(mock_data_source)
    assert len(result) == 4  # means, stds, margins, samples
    assert isinstance(result[0], np.ndarray)


def test_load_and_save(base_gp_model_instance, tmpdir):
    file_path = tmpdir.join("model.pt")
    base_gp_model_instance.save(file_path.strpath)

    loaded_model = BaseGPModel.load(file_path.strpath)
    assert isinstance(loaded_model, BaseGPModel)


def test_get_save_file_extension(base_gp_model_instance):
    ext = BaseGPModel.get_save_file_extension()
    assert ext == '.pt'
