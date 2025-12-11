### 遇到的问题：
1.eval freq和epoch的选择，太长的话跑的太慢，太短的话曲线不够连续
2.如何调整lr，如何调整batch_size(去适配显卡)?
3.数据解析问题(cs336官方给的原始代码测评有问题)，写代码重新进行测评
4.full_size的时候测试集准确率大幅下降 / 其它size也都有开始时候最高，越训练越下降的趋势
5.full SFT 和 LoRA/QLoRA 的对比


### prompt长度记录
sft_train.jsonl
=== Prompt token length percentiles ===
50%: 78.0
90%: 112.0
95%: 123.0
99%: 151.0
99.5%: 162.0

=== Answer token length percentiles ===
50%: 111.0
90%: 195.0
95%: 229.0
99%: 287.3
99.5%: 314.9

=== Total token length (prompt+answer) percentiles ===
50%: 191.0
90%: 298.0
95%: 335.0
99%: 412.3
99.5%: 441.9

sft_test.jsonl
=== Prompt token length percentiles ===
50%: 80.0
90%: 115.0
95%: 128.0
99%: 150.6
99.5%: 159.2

=== Answer token length percentiles ===
50%: 115.0
90%: 197.0
95%: 221.0
99%: 281.6
99.5%: 308.3

=== Total token length (prompt+answer) percentiles ===
50%: 199.0
90%: 297.2
95%: 334.0
99%: 402.8
99.5%: 450.7


### gen_max_new_tokens 调参
对不同 train_size 的 SFT 实验，引入了 trunc_rate 和 avg_gen_len 两个指标，用来分析生成是否被长度上限截断。

将 gen_max_new_tokens 从 512 调整为 1024（并将 vLLM 的 max_model_len 提到 4096）后：
full_size run 的 trunc_rate 从原先的潜在高值显著降到 < 3%，
与此同时 format_accuracy 从 ~0.5 提升到 ~0.99，accuracy 也从 ~0.52 提升到 ~0.60 左右，
证明之前 full_size 实验中的性能劣化主要源于生成被过短的长度上限截断，而非大数据本身的问题。

对小规模训练集（尤其是 128）：
trunc_rate 明显高于其他规模，说明在 20% 左右的问题上，模型会持续生成直到触碰长度上限；
但这部分样本在 1024 之前仍然往往包含 #### 答案，因此 format 几乎都正确，数值 accuracy 也还不错；
这反映出小数据 SFT 在“何时停止生成”这一点上的不稳定性

### batch_size 调参
当 batch_size 从 8 提高到 12 时，特别是在 full-size 训练下，trunc_rate 显著上升（最高接近 0.4），说明模型在相当一部分问题上会生成到长度上限才停止。

然而 format_accuracy 和 accuracy 并未明显下降，说明在被截断之前模型已经输出了 #### 和正确答案，截断主要发生在答案之后的冗余文本。

因此我们认为当前的 gen_max_new_tokens = 1024 已足够容纳完整的解题过程和答案，无需继续增大；相反，更重要的是通过 prompt 或训练配置来控制模型的冗余生成。(我试着把gen_max_len调整到2048, 发现accuracy没有显著提高，avg_len却显著提高，说明模型已经输出了正确答案，后面在说冗余文本)

![alt text](<truc_rate.png>)

同时batch_size = 12 比 batch_size = 8 tok / s提高约20%

![alt text](tok_speed.png)

### lr 调参
1e-5的accuracy曲线综合各类size来看，表现优于2e-5,2e-5又优于3e-5
![alt text](accuracy.png)

### train_size的scale
在[128, 256, 512, 1024, full(7000+)]的不同的train_size下(epochs = 3), 512的train_size测试集效果最佳，1024和full性能相比512没有显著提高，128和256accuracy弱于512

### EI(专家迭代)的OOM情况
#### 代码片段
```
with torch.no_grad():
    probs = torch.softmax(logits, dim=-1)      # <--- 创建又一个 7.46 GB
    log_probs = torch.log_softmax(logits, dim=-1) # <--- 创建第三个 7.46 GB
    token_ent = -(probs * log_probs).sum(dim=-1)
```
在这一瞬间，为了计算一个用来画图的 entropy 指标，你同时持有了：logits (7.5 GB)probs (7.5 GB)log_probs (7.5 GB)仅这三项就占用了 ~22.5 GB 显存。其他显存占用:模型权重 (bf16): ~3 GBOptimizer States (AdamW): 约 6-9 GB (取决于实现)Gradients: ~3 GBActivations: ~2-4 GB总计: $22.5 + 3 + 9 + 3 + 2 \approx \mathbf{39.5 \text{ GB}}$。考虑到显存碎片化（Fragmentation）和 PyTorch 上下文开销，这直接撑爆了你的 44GB 显存
