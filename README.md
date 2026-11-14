# randomness-health-monitor

一个只监测**原始随机字节**的 Python 健康检查库。它消费连续的 32-bit 样本及其采样时间，
在滑动窗口上统计健康信号，用于发现"熵源停滞"这类故障：服务卡在等待熵、虚拟环境持续
返回重复块，而应用层毫无察觉。

## 监测信号

| 信号 | 说明 | 失败标记 |
| --- | --- | --- |
| 窗口熵 | 窗口内 32-bit 样本值的归一化 Shannon 熵 | `low_entropy` |
| 重复块 | 窗口内固定大小（默认 4 样本）块的重复比例 | `repeated_blocks` |
| 位偏差 | 全部比特中 1 的比例与 0.5 的偏离 | `bit_bias` |
| 恒定长串 | 同一值连续重复的最长长度 | `constant_run` |
| 全零 / 全一长串 | `0x00000000` / `0xFFFFFFFF` 连续出现的最长长度 | `all_zero_run` / `all_one_run` |

## 状态机

- 初始为 `healthy`；任一信号越限即转为 `degraded`，并把**触发窗口**（样本、时间戳、
  评估报告）保留在 `monitor.trigger_window` 中供事后排查，恢复后依然可读。
- 恢复需要连续 `confirm_windows` 次干净评估（可配置），期间任何一次失败都会重置
  确认计数，避免抖动造成的状态来回翻转。
- 时间戳非递增的样本（重启重放、乱序投递）会被丢弃，**不会**被重复计为新证据。

## 用法

```python
from entropy_health import EntropyHealthMonitor, MonitorConfig, HealthState

mon = EntropyHealthMonitor(MonitorConfig(window_size=256, confirm_windows=3))
state = mon.add_sample(0x9E3779B9, timestamp=1718440000.0)
if state is HealthState.DEGRADED:
    inspect = mon.trigger_window  # 触发降级的窗口与失败原因

# 持久化与恢复（JSON 安全）
payload = mon.snapshot_json()
mon2 = EntropyHealthMonitor.restore_json(payload)
# 重启后重放旧时间戳的样本会被丢弃，不会重复计数。
```

主要配置项（`MonitorConfig`）：`window_size`、`block_size`、`step`（每多少个新样本评估
一次）、`min_entropy_ratio`、`max_repeat_block_ratio`、`max_bit_bias`、
`max_constant_run`、`confirm_windows`、`history_limit`。

内存有硬上限：滑动窗口、评估历史、触发窗口分别受 `window_size`、`history_limit`、
`window_size` 约束，长时间运行不会膨胀。库本身确定性运行：不读墙钟、不调用 RNG，
所有时间信息来自调用方传入的采样时间。

## 测试

```bash
python -m unittest
```

覆盖伪随机流、常数流、缓慢重复、短期抖动（误报抑制）、恢复确认窗口、状态迁移、
快照保存/恢复、内存上限与确定性。

## 重要限制

- **本库不能替代系统 CSPRNG。** 它只是一个旁路健康监测器，输出"看起来病了/看起来
  正常"的信号；生成密钥、nonce 等安全敏感随机数必须继续使用操作系统的 CSPRNG
  （如 `os.urandom` / `secrets`）。
- **通过全部检查不能证明随机源无偏。** 这些窗口统计量只能发现停滞、重复、明显偏差
  等粗粒度故障；一个有偏但构造得足够平滑的源可以完全通过所有检查。统计检验只能
  证伪，不能证真。
- 阈值是误报/漏报的权衡：窗口越小越灵敏但越容易误报，请按部署环境的样本速率与
  可接受的告警噪声调整 `MonitorConfig`。
