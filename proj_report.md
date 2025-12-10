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