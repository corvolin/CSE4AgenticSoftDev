# FASE
Artifacts for the paper FASE: Fast Adaptive Semantic Entropy for Code Quality
# Abstract
Multi-agent code generation offers a promising paradigm for autonomous software development by simulating the human software engineering lifecycle. However, system reliability remains hindered by LLM hallucinations and error propagation across interacting agents. While semantic entropy provides a principled way to quantify uncertainty without ground-truth answers, current methods often rely on costly LLM-driven equivalence checks. In this work, we introduce Fast Adaptive Semantic Entropy (FASE), a novel metric that approximates functional correctness based on the minimum spanning tree of structural and semantic dissimilarity graphs. Evaluations on HumanEval and BigCodeBench demonstrate that FASE outperforms state-of-the-art semantic entropy by LLM entailment, achieving a 25% average improvement in Spearman correlation and a 19% increase in ROCAUC score against Pass@1 from ground-truth test cases when using the Qwen3-Embedding-8B model. Furthermore, by eliminating costly LLM-driven equivalence evaluation, FASE incurs negligible computational overhead, requiring only approximately 0.3\ of the runtime cost of traditional semantic entropy approaches. These results position FASE as a practical, cost-effective solution for optimizing uncertainty quantification in real-world multi-agent workflows.
# To start
Running the code generation with 
```bash
python main.py --dataset HumanEval.jsonl --multi_agent funcAnalyst_coder_funcReviewer --model mistralai/Mistral-7B-Instruct-v0.2 --do_generation --do_evaluation
```

Get the baseline and proposed metrics with
```bash
python main.py --dataset HumanEval.jsonl --multi_agent funcAnalyst_coder_funcReviewer --model mistralai/Mistral-7B-Instruct-v0.2 --do_report
```

As mentioned in the paper, this study covers agentic code generation using Mistral, CodeLlama, Deepseek-Coder and Qwen2.5-Coder on tasks from dataset of HumanEval and bigCodeBench-hard. The semantic entropy calculated through embedding cosine similarity are derived with the help of AllMiniLM, ModernBERT, Nemotron and Qwen3-Embedding models.
