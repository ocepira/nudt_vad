import sys
import os
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torchattacks.wrappers.lgv import LGV


# 创建一个简单的模型用于测试
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)
        
    def forward(self, x):
        return self.fc(x)


# 创建一个简单的数据加载器用于测试
class SimpleDataLoader:
    def __init__(self):
        self.data = [(torch.randn(2, 10), torch.randint(0, 5, (2,))) for _ in range(5)]
        
    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return len(self.data)


class TestLGVWrapper:
    """
    测试LGV包装器类的功能
    """
    
    @pytest.fixture
    def setup(self):
        """设置测试环境"""
        model = SimpleModel()
        trainloader = SimpleDataLoader()
        lgv = LGV(model, trainloader, epochs=1, nb_models_epoch=1)
        return model, trainloader, lgv
    
    def test_load_models_with_valid_list(self, setup):
        """
        测试load_models方法使用有效的模型列表
        
        该测试验证当传入有效的模型列表时，load_models方法能正确加载模型
        """
        _, _, lgv = setup
        
        # 创建模型列表
        models = [SimpleModel() for _ in range(3)]
        
        # 加载模型
        lgv.load_models(models)
        
        # 验证模型已正确加载
        assert len(lgv.list_models) == 3
        assert all(isinstance(model, nn.Module) for model in lgv.list_models)
        assert lgv.list_models == models
    
    def test_load_models_with_empty_list(self, setup):
        """
        测试load_models方法使用空列表
        
        该测试验证当传入空列表时，load_models方法能正确处理
        """
        _, _, lgv = setup
        
        # 加载空模型列表
        lgv.load_models([])
        
        # 验证模型列表为空
        assert len(lgv.list_models) == 0
        assert lgv.list_models == []
    
    def test_load_models_with_non_list(self, setup):
        """
        测试load_models方法使用非列表参数
        
        该测试验证当传入非列表参数时，load_models方法会抛出ValueError异常
        """
        _, _, lgv = setup
        
        # 测试各种非列表类型
        invalid_inputs = [
            None,
            "not a list",
            123,
            SimpleModel(),  # 单个模型而不是列表
            {"models": [SimpleModel()]},  # 字典
            (SimpleModel(), SimpleModel())  # 元组
        ]
        
        for invalid_input in invalid_inputs:
            with pytest.raises(ValueError) as excinfo:
                lgv.load_models(invalid_input)
            assert "list_models should be a list of pytorch models" in str(excinfo.value)
    
    def test_load_models_with_mixed_content(self, setup):
        """
        测试load_models方法使用包含非模型对象的列表
        
        该测试验证当传入包含非模型对象的列表时，load_models方法能正确处理
        注意：原始实现只检查是否为列表，不检查列表内容
        """
        _, _, lgv = setup
        
        # 创建包含非模型对象的列表
        mixed_list = [SimpleModel(), "not a model", 123, SimpleModel()]
        
        # 加载混合列表
        lgv.load_models(mixed_list)
        
        # 验证列表已加载（原始实现不检查列表内容）
        assert len(lgv.list_models) == 4
        assert lgv.list_models == mixed_list