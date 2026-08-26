# 环境配置

## 硬件

| 项 | 值 |
|---|---|
| 芯片 | 昇腾 **910B4**（Atlas A2 **训练**产品，AIC/AIV 分离架构） |
| 计算核 | 20 AIC（Cube） + 40 AIV（Vector） |
| L2 | 192 MB |
| HBM | 32 GB |
| 实测带宽 | 读 828 GB/s · 写 803 GB/s · copy 722 GB/s（自研 40 核 AIV 流式 kernel 标定） |

> ⚠️ 这三个算子的 kernel 针对 **Atlas A2 训练卡** 编写。A2 推理产品（Atlas 200I/500 A2）
> 支持 Fixpipe 直写 UB，本代码没有用到那条通路，但核配比与 `KERNEL_TASK_TYPE`
> 的取值是按训练卡的 20/40 配置定的，换卡需要重新标定。

## 软件

| 组件 | 版本 | 说明 |
|---|---|---|
| CANN | **9.1.0** | 含 bisheng 编译器与 `tiling_api` |
| Python | 3.12 | |
| PyTorch | 见下方 `requirements.txt` | |
| torch_npu | 与 torch 主版本对应 | 需要 `torch_npu.contrib.transfer_to_npu` |
| g++ | 支持 `-std=c++17` | 链接 host 扩展 |
| pybind11 | 任意近期版本 | |

CANN 版本比较关键的两处：

* `matmul_intf.h` 在 `__NPU_ARCH__==2201` 上默认把 `Matmul` 展开成 `MatmulClient`
  （把计算请求投给跑在 AIV 上的 KFC server）。代码里在 include 之前 `#define ASCENDC_CUBE_ONLY`
  让它退化成 `MatmulImpl` 直算。换 CANN 大版本时要确认这个宏仍然有效。
* `tiling_api` / `ascendc_runtime` 在 9.x 只有**静态库**，且位于
  `$ASCEND_HOME_PATH/{arch}-linux/lib64` 而不是 `$ASCEND_HOME_PATH/lib64`。
  构建脚本会搜索多个目录并且 `.so` / `.a` 都认。

## 环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

| 变量 | 必需 | 默认 | 说明 |
|---|---|---|---|
| `ASCEND_HOME_PATH` | 否 | 自动探测 `/usr/local/Ascend/ascend-toolkit/latest` | CANN 根目录 |
| `KS_BISHENG` | 否 | 自动探测 | 直接指向 bisheng 可执行文件；自动探测顺序为 `PATH` → `$A/bin/bisheng` → `$A/{aarch64,x86_64}-linux/ccec_compiler/bin/bisheng` → `$A/compiler/.../bisheng` |
| `KS_NPU_ARCH` | 否 | `dav-2201` | 910B 的架构串 |

提交代码本身**不依赖任何环境变量开关**：没有性能相关的 env 分支，
上面三个只是用来定位工具链的。

## 首次运行

三个 `ModelNew.__init__` 都会用 bisheng 即时编译自己的 kernel，首次约 1-2 分钟。
产物按「源码 + 构建配方版本 + torch 版本」的哈希缓存在：

```
~/.cache/ks_kernels/<name>_<md5前12位>/
```

**编译在 `__init__` 内完成，不计入评测的计时区间。**

## 依赖安装

```bash
pip install -r requirements.txt
```

torch / torch_npu 请按昇腾官方配套表选择版本安装，不要用 `requirements.txt` 里的
版本号直接 `pip install`（torch_npu 需要与 CANN 和 torch 严格配套）。
