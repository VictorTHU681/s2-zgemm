#!/usr/bin/env bash
# ZGEMM 赛题：编译 + 三组测试用例（判题系统直接执行本脚本）
set -o pipefail
cd "$(dirname "$0")" || exit 1

# 不用 set -u：/etc/profile 与 module 脚本会引用未定义变量
source /etc/profile >/dev/null 2>&1 || true
# 编译环境：题面给出的 HPCKit module 路径；若不存在，尝试本集群 HPCKit 的 setvars.sh；再不然用系统 gcc
if [[ -d /home/HPC/HPCKit/latest/modulefiles ]]; then
    module use /home/HPC/HPCKit/latest/modulefiles 2>/dev/null
    module load gcc/compiler12.3.1/gccmodule 2>/dev/null
else
    for p in /work_ssd/software/HPCKit/latest/setvars.sh \
             /work_ssd/software/HPCKit/26.1.RC1/setvars.sh \
             /home/share/eapp_sysu/HPCKit/latest/setvars.sh \
             /home/share/eapp_sysu/HPCKit/26.1.RC1/setvars.sh; do
        if [[ -f "$p" ]]; then . "$p" --use-gcc >/dev/null 2>&1; break; fi
    done
fi

# 首选：SME(FP64 FMOPA) 内核，需要支持 armv9-a+sme+sme-f64f64 的汇编器；运行期会用 HWCAP2 检测硬件，
# 缺失时自动回退。若编译器/汇编器不支持这些扩展，则编译纯通用实现（保证正确性）。
rm -f zgemm_test
if ! gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp bench_zgemm.c zgemm.c -o zgemm_test -lm 2>/dev/null; then
    echo "[run.sh] 编译器不支持 SME 扩展，回退通用实现" >&2
    gcc -O3 -fopenmp -DZGEMM_NO_SME_ASM bench_zgemm.c zgemm.c -o zgemm_test -lm
fi

# 判题口径：38 线程、单一 NUMA（CPU 节点 1）。绑核在 zgemm.c 内部完成；
# 这里的 OMP 变量只让基准程序自带的参考实现也能用满核，不影响被测函数。
cpunode=1
NCPU=$(numactl -N $cpunode nproc 2>/dev/null || nproc 2>/dev/null || echo 38)
# 默认 38 线程（评测节点独占）。packA 后没有全局 barrier，按块就绪标志同步。ZGEMM_THREADS=37 可留一个空核。
NT=${ZGEMM_THREADS:-38}
if [[ "$NCPU" =~ ^[0-9]+$ ]] && (( NCPU > 0 && NCPU < NT )); then NT=$NCPU; fi   # 节点 CPU 不足时不超订
export OMP_NUM_THREADS=$NT
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=FALSE
unset GOMP_CPU_AFFINITY OMP_THREAD_LIMIT 2>/dev/null || true
# 工作线程在两次调用之间保持自旋（基准程序在计时前单线程 memcpy 约 1 s，否则线程休眠后唤醒要 ~0.15 ms）；
# 只在核数足够、不超订时启用，避免与其他进程/线程抢核时自旋线程拖垮一切
if [[ "$NCPU" =~ ^[0-9]+$ ]] && (( NCPU >= NT )); then
    export OMP_WAIT_POLICY=active
    export GOMP_SPINCOUNT=infinite
fi

# 内存：优先使用与 CPU 节点 1 相邻、容量像 HBM（1～32 GB）的纯内存节点（本机为 HBM node 17，4 GB，距离 61），
# 放不下时自动回落本地 DDR。与 T2 的 `numactl -m16` 同理，但 4 GB HBM 装不下大用例的全部矩阵，所以用 --preferred 而非 -m。
# 容量判据是为了避免把普通 DDR/远端内存池误当 HBM。
hbm=""
if [[ -r /sys/devices/system/node/node$cpunode/distance ]]; then
    read -ra dist < /sys/devices/system/node/node$cpunode/distance
    best=999
    for n in "${!dist[@]}"; do
        [[ -d /sys/devices/system/node/node$n ]] || continue
        cpulist=$(cat /sys/devices/system/node/node$n/cpulist 2>/dev/null)
        [[ -z "$cpulist" ]] || continue          # 只考虑没有 CPU 的纯内存节点
        totkb=$(awk '/MemTotal:/{print $4}' /sys/devices/system/node/node$n/meminfo 2>/dev/null)
        [[ "$totkb" =~ ^[0-9]+$ ]] || continue
        (( totkb >= 1048576 && totkb <= 33554432 )) || continue   # 1～32 GB
        if (( dist[n] < best )); then best=${dist[n]}; hbm=$n; fi
    done
    (( best > 70 )) && hbm=""                    # 远端节点不用
fi
MEMOPT=""
if [[ -n "$hbm" ]] && numactl -N $cpunode --preferred=$hbm true >/dev/null 2>&1; then
    MEMOPT=""   # v23：C_copy 的 HBM 放置改由库内任务策略承担，避免原始 A/B/C 占满 HBM 后的首调迁移
fi
# 环境诊断（stderr）：判题机与开发机不同时便于定位
{
    echo "[run.sh] host=$(hostname 2>/dev/null) cpus_in_node$cpunode=$NCPU threads=$NT numactl -N $cpunode $MEMOPT"
    echo "[run.sh] cpu: $(grep -m1 'model name\|CPU part\|Hardware' /proc/cpuinfo 2>/dev/null | head -1)  features: $(grep -m1 Features /proc/cpuinfo 2>/dev/null | tr ' ' '\n' | grep -E '^(sve|sve2|sme|smef64f64|sme2)$' | tr '\n' ' ')"
    echo "[run.sh] nodes: $(numactl -H 2>/dev/null | grep -E 'available|node [0-9]+ size' | tr '\n' ';' | cut -c1-400)"
    echo "[run.sh] numa_balancing=$(cat /proc/sys/kernel/numa_balancing 2>/dev/null) thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null) freq_khz=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null)"
} >&2

if command -v numactl >/dev/null 2>&1 && numactl -N $cpunode true >/dev/null 2>&1; then
    RUN="numactl -N $cpunode $MEMOPT"
else
    echo "[run.sh] numactl 不可用或节点 $cpunode 不可用，直接运行" >&2
    RUN=""
fi
$RUN ./zgemm_test 7427 7427 256 1
$RUN ./zgemm_test 14848 14848 256 1
$RUN ./zgemm_test 37360 8192 512 1
