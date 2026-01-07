# 自动驾驶场景

本项目基于VAD库实现自动驾驶场景。支持推理、攻击、防御。

## 环境变量
## 只需 python main.py --process xx ,不指定后面的变量名都会使用默认值
| 变量名 | 是否必填 | 描述 |
|--------|---------|------|
| input_path | 选填，默认输入路径 | 指定输入路径，在此路径下有权重文件和数据集文件 |
| output_path | 选填，默认输出路径 | 指定输出路径，在此路径下保存生成的对抗样本和防御训练的权重 |
| process | 必填/选填，默认test | 指定进程名称，支持枚举值（第一个为默认值）: `test`, `attack`, `defense` |
| image-path | 选填，默认第一个 | 输入图像路径，当process为`attack`或`defense`时默认第一个 |
| attack-method | 选填 | 指定攻击方法，若process为`attack`则必填，支持枚举值（第一个为默认值）: `fgsm`, `pgd`, `bim`,`badnet`, `squareattack`, `nes` |
| defense-method | 选填 | 指定防御方法，若process为`defense`则必填，支持枚举值（第一个为默认值）: `fgsm_denoise`, `pgd_purifier` |
| save-path | 选填 | 对抗样本保存路径 |
| save-original-size | 选填 | 是否保存原始尺寸的对抗样本 |
| config | 选填 | test config file path，当process为`test`时必填 |
| checkpoint | 选填 | checkpoint file，当process为`test`时必填 |
| device | 选填，默认为0 | 使用哪个gpu |
| workers | 选填，默认为0 | 加载数据集时workers的数量 |
| steps | 选填，默认为10 | 攻击迭代次数(PGD/BIM) |
| alpha | 选填，默认为2/255 | 攻击步长(PGD/BIM) |
| epsilon | 选填，默认为8/255 | 扰动强度 |
| tv-weight | 选填，默认为1.0 | 空间平滑参数 |
| l2-weight | 选填，默认为0.01| L2保真权重 |
| epsilon   |选填，默认 8.0  | 扰动强度限制|





## 下载nuscene_tiny数据集和model 都放在input

## 其中在model中有三个model vad_tiny对于测试, resnet50对应防御 , standard 对应攻击

## 快速开始
python main.py  --process xx

## 构建 Docker 镜像
docker build -t vad:latest .

## 运行 Docker 镜像：四个模式，建议先依次运行

# test
docker run --rm --gpus all -v ./input:/project/input:ro -v ./output:/project/output:rw -e INPUT_PATH=/project/input -e
 OUTPUT_PATH=/project/output -e process=test  vad:latest

# adv
docker run --rm --gpus all   -v ./input:/project/input:ro -v ./output:/project/output:rw -e INPUT_PATH=/project/input -e OUTPUT_PATH=/project/output -e process=adv vad:latest

# attack
docker run --rm --gpus all -v ./input:/project/input:ro -v ./output:/project/output:rw -e INPUT_PATH=/project/input -e 
OUTPUT_PATH=/project/output -e process=attack  vad:latest

# defense
docker run --rm --gpus all -v ./input:/project/input:ro -v ./output:/project/output:rw -e INPUT_PATH=/project/input -e
 OUTPUT_PATH=/project/output -e process=defense  vad:latest


