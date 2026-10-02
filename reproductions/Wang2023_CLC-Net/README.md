# CLC-Net 复现

Wang 等，Neurocomputing 2023。设计说明在 `DESIGN.md`。锁定的数字在 `clcnet/assumptions.py`。

论文没有官方代码。这里复现的是文中写明的结构、损失和超参数。没写明的部分按 `DESIGN.md` 的假设固定，不用原文 AUPR 表去搜索学习率或损失权重。

额外输入：无。只有 RGB 眼底图像。种子：20230108。

## 环境

```bash
pip install -r requirements.txt
```

## 数据

`--data-root` 指向一个数据集目录，或指向含有 `manifest.csv` 的目录。清单列名为 `id,split,image,HE,MA,EX,SE`。`split` 取 `train`、`val`、`test`。没有的病灶留空。

没有清单时按常见发布目录查找：

- IDRiD：训练/测试原图，以及带 `_MA`、`_HE`、`_EX`、`_SE` 后缀的掩膜。
- DDR：`train`、`valid`、`test` 下的 `image` 与 `label/EX|HE|MA|SE`。
- e-ophtha：同时出现在 MA 与 EX 病变目录、且不在 healthy 里的 21 张图。第一次扫描会把划分写到 `splits/eophtha_seed20230108.json`。

## 训练和评估

在本目录执行。完整模型是 `--variant full`，训练 60 轮，结果检查点是 `epoch_060.pt`。

```bash
python train.py --dataset idrid --data-root /path/to/IDRiD --output runs/idrid_full
python evaluate.py --checkpoint runs/idrid_full/epoch_060.pt --dataset idrid --data-root /path/to/IDRiD --split test
```

评估默认使用局部分支。同一套 `full` 权重用 `--head context` 得到表 1 里标星号的上下文输出。

其他训练设定：`loc`、`loc_cl`、`cont`、`cont_cl`、`loc_cont`、`loc_cont_att`、`full`。

评估写出的 JSON 里 `result_source` 为「本项目重训」。原文数字在 `paper_reported.tsv`，不写进这个 JSON。

## 不依赖数据的检查

```bash
python tests/test_locked.py
python tools/complexity.py
```
