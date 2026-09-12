# S2 ZGEMM — ARM Kunpeng SME 复数双精度矩阵乘

竞赛提交版本存档。目标平台:鲲鹏 SME(FP64 FMOPA),38 核单 NUMA 节点 + 4 GB HBM 内存节点。

## 正式版本 v1(内部编号 v22)

| 文件 | md5 | 说明 |
|---|---|---|
| `zgemm.c` | `fb772197818623941d4fb5894897cd40` | 算子实现 |
| `bench_zgemm.c` | `a6cf1a3b89a2cfe1114676ac95a51c8e` | 官方基准程序,原版未改(逐字节一致) |
| `run.sh` | `e5be74caee4f65ed48cf0091a76c7d88` | 编译 + 运行(numactl 绑 CPU 节点,`--preferred` 指定 HBM) |
| `zgemm.zip` | `1c8b96288011e65e1efdb29db4e41ef0` | 提交原件(上述三文件打包) |

## 成绩

- 本机验证(1.55 GHz,cn20484):6808 / 6990 / 6924 GFLOPS,run.sh 双跑总分 **20722 / 20733**
- 线上(2.0 GHz):上一版 v21 实测 **8384.7 / 8662.0 / 8542.4 = 25589.1**;v22 较 v21 本机 +0.14~0.2%,预期 ~25640

## 实现要点

- 17 行混合块 SME 内核(16 行 FMOPA 4M + 1 行 FMLA 条带),K 循环内重叠上一块的 C 回写,回写按 6 个 k 步对分摊(EPI6)
- 首次(校验,非计时)调用一次性完成:sysfs 识别 HBM 节点 → `migrate_pages` 将本进程残留在 HBM 的页迁回本地 DDR → C 装不下 HBM 时 `set_mempolicy` 改为 HBM/DDR 2 MB 交错 → `mmap`+`mbind` 在 HBM 建打包 arena(2 MiB 对齐 + THP)
- 每次计时调用全量重新打包,不复用之前调用的打包结果;无后台线程、无频率/负载探测;所有分支基于尺寸区间与对齐属性,无测例精确匹配
