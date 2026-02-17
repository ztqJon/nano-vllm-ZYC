
# Nano-vLLM 问答（基于仓库实现）

以下问答从代码实现里提取，包括AI生成及平时与AI问答的实时记录。

## 架构与流程
**Q**: 这个项目的入口类是什么？它负责什么？

**A**: `LLMEngine` 是入口类，负责创建 `Config`、`tokenizer`、`Scheduler`、`ModelRunner` 并提供 `generate()`。

**Q**: 生成流程的主循环是什么样的？

**A**: `generate()` 循环调用 `step()`；`step()` 从 `Scheduler` 获取本轮需要 prefill/decode 的序列，并调用 `ModelRunner.run()`。

**Q**: prefill 和 decode 的区别是什么？

**A**: prefill 处理完整 prompt（含 prefix cache 命中）；decode 每次只为每个序列生成一个新 token。

**Q**: Nano-vLLM 是不是“prefill 优先”？

**A**: 是的，`Scheduler.schedule()` 先尝试 对 `waiting` 中的序列做 prefill，没有可 prefill 的序列才做 decode。

**Q**: `waiting` 和 `running` 队列分别表示什么？

**A**: `waiting` 是尚未完成 prefill 的序列，`running` 是已 prefill 并进入 decode 的序列。

## 调度与 KV Cache

**Q**: 调度时受哪些约束？

**A**: `max_num_seqs`、`max_num_batched_tokens`，以及 KV Cache 是否还有可用 block。

**Q**: KV Cache 为什么要分 block 管理？

**A**: 通过 block table 把逻辑序列映射到物理缓存，支持 PagedAttention 风格的非连续缓存。

**Q**: block 大小在这里是多少？

**A**: `Config.kv_cache_block_size` 默认 256，`Sequence.block_size` 也固定为 256。

**Q**: 什么是 prefix caching，这里怎么实现？

**A**: 对满 block 计算 hash（xxhash），用 `hash_to_block_id` 复用相同 token 的 block。

**Q**: 为什么还要比对 token 列表，hash 不够吗？

**A**: 防止 hash 冲突，命中 hash 后仍会校验 token 列表。

**Q**: decode 阶段如果 block 不够怎么办？

**A**: 调度器会 preempt：从 running 队列尾部取序列移回 waiting 并释放其 block，直到可以 append；必要时也会把当前序列 preempt。

**Q**: 什么情况下会新增 block？

**A**: decode 前的 may_append 中，若当前长度满足 `len(seq) % block_size == 1`，就分配新 block（实现以这个条件判定）。

**Q**: PagedAttention 的核心数据结构是什么？

**A**: 每个序列维护 block_table，用它把逻辑 token 映射到 KV Cache 物理位置。

## 模型执行与上下文

**Q**: ModelRunner 的主要职责是什么？

**A**: 负责前向、构造 prefill/decode 输入、采样 token、管理 CUDA Graph 和 KV Cache。

**Q**: 注意力相关的上下文如何传递？

**A**: 通过全局 Context，由 prepare_prefill/prepare_decode 填充，Attention 读取。

**Q**: prefill 输入里 cu_seqlens 和 slot_mapping 的作用？

**A**: cu_seqlens 供 FlashAttention 变长序列计算。slot_mapping 记录写入 KV Cache 的物理位置。

**Q**: slot_mapping 是干嘛的？

**A**: 记录每个需要写入 KV Cache 的 token 在缓存中的物理位置索引；Triton kernel 按这个映射把 k/v 写入对应 slot。

**Q**: slot_mapping 的物理位置索引具体是什么？

**A**: 是每层 k_cache/v_cache 的线性索引：block_id * block_size + block 内偏移。prefill 时对所有待写入 token 逐个生成这个索引（遍历未缓存的 block，按 [start,end) 展开）；decode 时只为新 token 生成一个索引（last_block_id * block_size + last_block_num_tokens - 1）。

**Q**: prefix cache 命中时 prefill 的注意力如何处理？

**A**: 通过 block_tables 让 FlashAttention 访问缓存中的前缀 token。

**Q**: decode 阶段如何组织输入？

**A**: 只取每个序列最后一个 token，并用 context_lens 与 block_tables 读历史。

## 注意力与 Triton

**Q**: KV Cache 写入是怎么做的？

**A**: 用 Triton kernel 把新的 key/value 写入对应 slot（store_kvcache_kernel）。

**Q**: 什么是 Triton？

**A**: Triton 是一个用于编写高性能 GPU 内核的 Python 语言与编译器（DSL+compiler），通过 @triton.jit 把自定义 kernel JIT 到 GPU；在本仓库里用它实现 KV Cache 写入等自定义算子。

**Q**: 为什么 prefill 和 decode 用不同 FlashAttention 接口？

**A**: prefill 用变长接口 flash_attn_varlen_func，decode 用 flash_attn_with_kvcache。

**Q**: flash_attn_varlen_func / flash_attn_with_kvcache 的原理和入参？

**A**: 两者都是 FlashAttention（分块/融合的注意力 kernel）。prefill 用变长接口，输入拼接后的 q/k/v 与 cu_seqlens、max_seqlen 来描述每条序列；decode 用带 KV cache 的接口，输入当前 step 的 q，配合 k_cache/v_cache、context_lens 与 block_table 直接在缓存上做注意力。

**Q**: block_table 里只有 block id，flash_attn 怎么知道去哪里读？

**A**: kernel 同时拿到 k_cache/v_cache 的基址与 stride；block_table 提供的是物理 block 的索引，kernel用 block_id * block_size + offset 计算线性位置，再结合缓存张量的 stride 计算实际内存地址。

## 采样

**Q**: 采样策略是什么？

**A**: 温度缩放后 softmax，再用指数分布采样（Gumbel-max 形式）取 argmax。

**Q**: 为什么不允许 temperature 太小？

**A**: SamplingParams 断言 temperature > 1e-10，避免退化为 greedy。

## CUDA Graph

**Q**: CUDA Graph 在这里用在哪个阶段？

**A**: 仅用于 decode，且 enforce_eager=False 时；输入序列数 `>512` 会直接走 eager，`<=512` 才按预捕获的 batch bucket 复用图。

**Q**: 为什么 prefill 不走 CUDA Graph？

**A**: prefill 形状变化大，decode 形状更固定，更适合图捕获。

## 并行与多进程

**Q**: 这个项目的张量并行怎么做的？

**A**: 使用 PyTorch Distributed + NCCL，线性层用列并行/行并行/QKV 并行切分。

**Q**: 为什么 RowParallelLinear 需要 all-reduce？

**A**: 行切分后每卡只算部分输出，需要跨设备求和得到完整结果。

**Q**: 多进程之间如何同步调用？

**A**: rank0 写入 SharedMemory 并触发事件，其它 rank 在 loop 中读取并执行。

## KV Cache 规模估算

**Q**: KV Cache 的 block 数量如何估算？

**A**: 读取显存使用情况，按 gpu_memory_utilization 预算减去峰值，再除以单 block 字节数。

**Q**: 单 block 的字节数怎么估算？

**A**: 为 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size。

## 模型与权重加载

**Q**: 支持哪些模型结构？

**A**: llama、qwen2、qwen3、qwen3_moe（见 model_dict）。

**Q**: 权重加载有什么特殊处理？

**A**: 支持 packed module 的权重映射，并通过 weight_loader 做分片加载。

## 设计细节与局限

**Q**: Sequence 为什么要实现 __getstate__/__setstate__？

**A**: 便于序列在序列化时只传必要字段（如 last_token）。

**Q**: __getstate__/__setstate__ 在哪会被调用到？

**A**: 在多进程场景里通过 pickle 传输 Sequence 时会触发，比如 rank0 把 seqs 写入共享内存（pickle.dumps）给子进程；也适用于用户手动 pickle。

**Q**: __getstate__/__setstate__ 是在多进程中自动触发吗？

**A**: 是的，序列被 pickle 序列化/反序列化时会自动调用这两个方法；多进程只是在这个实现里触发该流程的主要场景。

**Q**: 最大 batch/序列长度限制来自哪里？

**A**: 由 max_num_batched_tokens、max_num_seqs、max_model_len 共同约束。

**Q**: 为什么 kv_cache_block_size 必须是 256 的倍数？

**A**: 配置里有断言，且 Sequence.block_size 固定为 256，需要保持对齐假设。

**Q**: 这套实现和 vLLM 官方版本相比可能少了什么？

**A**: 只覆盖核心推理路径，未包含完整服务化、复杂采样策略等能力。

## 可被追问的“为什么”

**Q**: 为什么 prefix cache 命中后仍要保留 block_table？

**A**: 注意力仍需通过 block_table 定位缓存中的历史 token。

**Q**: 为什么 decode 阶段每次只生成一个 token？

**A**: 自回归生成的标准做法，decode阶段每步只算最后一个token的预测。

**Q**: 为什么 warmup 要用 max_model_len 的伪序列？

**A**: 触发最坏情况的 kernel 初始化，并获得更稳定的显存峰值统计。

**Q**: 为什么 run_model 对 batch size 做阈值判断？

**A**: CUDA Graph 只为 batch size ≤512 预捕获了图；超过阈值就回退到普通前向。

**Q**: can_append 为什么看 len(seq) % block_size == 1？

**A**: 实现里用 `len(seq) % block_size == 1` 作为“需要新 block”的判定条件，因此 can_append 只在该条件成立时要求有空闲 block。
