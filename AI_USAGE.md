# AI 使用声明

**代码由人类编写。** `cs336_basics/` 下的实现与 `tests/adapters.py` 的接线均为本人完成，AI 工具没有生成或改动这些文件。

AI 工具的使用范围：

- 概念解释：einsum 的归约语义、RMSNorm 的公式、SwiGLU 的门控结构、RoPE 的配对方式
- API 用法：`nn.Module`、`nn.Parameter`、`torch.ones`、`trunc_normal_`、`state_dict` 的装载
- 代码检查：指出 `Linear` 未接入 `device`/`dtype`、`RMSNorm` 沿 batch 求均值、`Embedding` 用 einsum 导致沿词表求和等问题，修改由本人完成
- 笔记整理：`笔记/02-Model.md` 与 `笔记/CLAUDE.md` 由 AI 起草

提交记录里的 `Co-Authored-By: Claude` 对应笔记与提交信息的整理，不对应实现代码。

未使用任何第三方实现或外部代码。

2026-10-25

---

The implementation code under `cs336_basics/` and the wiring in `tests/adapters.py` were written and edited by me. AI tools were used for concept explanations, API usage questions, and code review; they did not generate or modify the implementation files. The notes under `笔记/` were drafted with AI assistance, as indicated by the `Co-Authored-By: Claude` trailers in the commit history.
