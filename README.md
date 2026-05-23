# 开源齐民

开源农产品价格预测系统。基于历史价格、气温、降水预测农产品价格长期走势。希望能够抛砖引玉，作为更强模型的基础，为中国农户生产采收起到参考作用。

## 运行说明

```shell
# 爬取气象数据
uv run python -m qixiang
# 爬取商务预报价格数据
uv run python -m swyb
# 训练单天预测模型
uv run python -m kyqm --pipeline short --model all
# 训练长期预测模型（7/30/90 天）：
uv run python -m kyqm --pipeline long --model all
# 可视化界面
uv run streamlit run app.py
```
