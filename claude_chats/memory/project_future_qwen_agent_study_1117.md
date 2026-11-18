---
name: project-future-qwen-agent-study-1117
description: "FUTURE (17/11): user wants to work on Qwen-Agent later; facts already proven from its source, clone deleted"
metadata:
  node_type: memory
  type: project
  originSessionId: bba1f130-f2e8-4e9a-888b-ffeed7b71e70
  modified: 2026-11-17T06:31:04.003Z
---

17/11: user asked about https://github.com/QwenLM/Qwen-Agent and said "we will work on this later". Checked from the source (commit 31a4d36, 4 Mar 2026):
- Pure Python: 0 C/C++/CUDA files, no GPU kernels, no training code (0 hits for .backward/torch.optim/zero_grad/Trainer).
- Sits above the inference server: sends requests over HTTP through the `openai` client to vLLM/Ollama/DashScope. Optional `transformers_llm.py` calls HF generate(), the only torch import.
- Agent loop = qwen_agent/agents/fncall_agent.py `_run`: call model -> parse `<tool_call>` -> run Python tool -> FUNCTION message -> repeat; max 20 model calls (QWEN_AGENT_MAX_LLM_CALL_PER_RUN).
- The "kernel" in code_interpreter.py = a Jupyter kernel (a process in Docker), not a GPU kernel.
- The model learned the format in Qwen post-training: the Qwen3 chat template has the same "# Tools / <tools> / <tool_call>" text; <tool_call>, </tool_call>, <tool_response>, </tool_response> are special tokens.

The scratchpad clone (41 MB) was deleted on the user's word the same day.

**Why:** user wants to come back to it; don't redo the audit from zero.
**How to apply:** when it comes back, re-clone and re-check the latest commit first ([[feedback_no_assumptions_decide_by_tests]]). The link to our work: SFT data for tool use must use this exact <tool_call>/<tool_response> format and its special tokens ([[project_chat_template_decision_1015]]).
