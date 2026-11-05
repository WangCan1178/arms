# ARMS AAAI-27 — Advisor-comment Revision Copy

这是从 2026-07-22 当前 Overleaf 源码建立的独立匿名投稿副本。

## 主要文件

- `arms_aaai.tex`：主论文 LaTeX 源码
- `arms_aaai_anonymous.bib`：匿名投稿参考文献
- `ReproducibilityChecklist.tex`：AAAI reproducibility checklist
- `fig_outlined/`：论文图件
- `ADVISOR_COMMENT_RESPONSE.md`：11 条导师意见的修改与验证记录

## 编译

```sh
pdflatex arms_aaai.tex
bibtex arms_aaai
pdflatex arms_aaai.tex
pdflatex arms_aaai.tex
```

Checklist 可单独编译：

```sh
pdflatex ReproducibilityChecklist.tex
```

验证环境需要提供 `algorithm` 与 `algorithmic` 宏包（TeX Live 中的 `algorithms` package）。
