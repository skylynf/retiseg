# retiseg

眼底照片上四类糖尿病视网膜病变病灶（MA、HE、EX、SE）的统一评测和训练入口。文献账本在 `literature/meta/LEDGER.tsv`，进度在 `literature/synthesis/00_进度.txt`。

## 克隆

```bash
git clone git@github.com:skylynf/retiseg.git
cd retiseg
git submodule update --init --depth 1
```

`official_code/` 只保存作者仓库的提交指针。不要修改这些目录里的代码。`--depth 1` 与当初的浅克隆一致。

DeepLabv3plus 的上游是整个 `tensorflow/models`，体积很大。第一批实验的 DeepLab 使用 torchvision，不导入这个目录。磁盘紧时可以跳过它：

```bash
git submodule update --init --depth 1 -- \
  official_code/M2MRF \
  official_code/HACDR-Net \
  official_code/H2Former \
  official_code/Swin-Unet \
  official_code/FCT \
  official_code/HRNet-Semantic-Segmentation
```

## 不在 Git 里

这些目录留在本机，克隆后要另行准备：

- `dataset/`：原始发布和 `dataset/prepared/`
- `runs/`：训练输出
- `pretrained/`：ImageNet 等权重

需要的权重见 `literature/synthesis/09_第一批任务.txt` 第九节。
