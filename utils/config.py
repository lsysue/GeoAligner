# utils/config.py
import yaml

class Config:
    """
    自定义配置类：支持点操作符访问，但不会强制转换赋值的字典。
    解决了 EasyDict 无法接受 integer key 字典 (如 {1: 'person'}) 的问题。
    """
    def __init__(self, dictionary=None):
        if dictionary:
            for k, v in dictionary.items():
                # 仅在初始化时，对 YAML 里的嵌套字典进行递归转换
                if isinstance(v, dict):
                    setattr(self, k, Config(v))
                else:
                    setattr(self, k, v)

    def __repr__(self):
        return str(self.__dict__)

def load_config(config_path: str = "config.yaml"):
    """
    从 YAML 文件加载配置。
    
    使用 easydict 库将嵌套的字典转换为可以通过点属性访问的对象
    (例如, cfg.paths.output_dir)
    """
    try:
        with open(config_path, 'r') as file:
            cfg_dict = yaml.safe_load(file)
        
        # 将嵌套的字典转换为 easydict 对象
        return Config(cfg_dict)
    
    except FileNotFoundError:
        raise FileNotFoundError(f"错误: 配置文件未找到于 {config_path}")
    except Exception as e:
        raise Exception(f"加载配置时出错: {e}")

# --- 独立测试块 ---
if __name__ == "__main__":
    print("--- 正在测试配置加载器 ---")
    try:
        cfg = load_config("../config.yaml") # 假设从 utils 目录运行，返回上一级
        print("配置加载成功！")
        
        # 测试 easydict 功能
        print(f"输出目录: {cfg.paths.output_dir}")
        print(f"S-Dim 维度: {cfg.model_dims.s_dim}")
        print(f"阶段一学习率: {cfg.training.lr_stage1}")
        
        assert cfg.paths.output_dir == "checkpoints"
        assert cfg.model_dims.s_dim == 768
        
        print("\n[SUCCESS] Config loader 测试通过！")
    
    except Exception as e:
        print(f"\n[FAILURE] 配置加载测试失败: {e}")
        print("请确保您已创建 config.yaml 并位于项目根目录中。")