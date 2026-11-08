# 青岚合成制度问答 · 2026-11-v2

本地可运行的面试准备复现。服务入口为 `w6_hr_policy.app.main:app`。语料为青岚实验公司的虚构制度，适用于指定的合成对象。仓库根目录的 Streamlit 演示和本目录的 FastAPI 复现各有独立入口、语料和评估口径。

## 在 Windows PowerShell 启动

从仓库根目录先执行 `cd fastapi_demo`。本示例在 Python 3.13 下验证。第一次可以省略模型配置，验证能启动、无依据拒答和有候选时的降级。

```powershell
python -m venv .venv
& '.venv/Scripts/python.exe' -m pip install -r requirements.lock.txt
$env:W6_API_KEY = [Guid]::NewGuid().ToString('N')
& '.venv/Scripts/python.exe' -m uvicorn w6_hr_policy.app.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

在另一个终端设置相同的 `W6_API_KEY` 后执行请求，或打开 `http://127.0.0.1:8000/docs` 查看接口。已配置模型的实例需在启动前通过本地环境变量提供 `DEEPSEEK_API_KEY`，默认 `DEEPSEEK_BASE_URL=https://api.deepseek.com`、`DEEPSEEK_MODEL=deepseek-flash`。密钥通过已有的本地凭据方式注入。

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe -N --get 'http://127.0.0.1:8000/chat/stream' -H "X-API-Key: $env:W6_API_KEY" --data-urlencode 'q=试用期一般多久？' --data-urlencode 'session_id=demo-01'
curl.exe -N --get 'http://127.0.0.1:8000/chat/stream' -H "X-API-Key: $env:W6_API_KEY" --data-urlencode 'q=公司有免费班车吗？' --data-urlencode 'session_id=demo-02'
```

- 试用期题的合成出处为 HR-V2-02。运行正常时答复会引用完整条款。模型配置缺失时返回 `degraded` 并展示候选原文。
- 班车题为 `refused`，旧版或外包对象为 `clarify`。类别参数可限定在假勤、入离职、差旅、薪酬福利，默认搜索全部。
- `status` 事件含阶段、耗时、来源编号和模型增量到达情况。上游确实按流式协议发送。下游只实时发送进度和增量长度，最终 `message` 一次送出已校验的制度原文。运行异常使用 `error`，结束用 `done`，其中有业务状态和轨迹落盘结果。
- 通过 `GET /runs/{run_id}` 带相同 Key 查看脱敏轨迹。SQLite 默认 `data/runs.sqlite3`，可用 `W6_DB` 指定空白路径切换或回退。每次运行都有独立编号。

## 测试和评估

```powershell
& '.venv/Scripts/python.exe' -m unittest discover -s tests -v
& '.venv/Scripts/python.exe' eval_run.py --mode fixture --out evidence/recheck-fixture-20.json
& '.venv/Scripts/python.exe' eval_run.py --mode real --out evidence/recheck-real-20.json
& '.venv/Scripts/python.exe' eval_run.py --replay evidence/real-20.json --out evidence/recheck-replay.json
```

测试包含本机回环与临时 SQLite。评估默认固定 20 题，另有 `--dataset diagnostics-v2.json` 和 `--dataset session-12-v2.json`。使用 `--limit 2` 可先跑冒烟。Fixture 模式零外部调用，桩固定选择首个候选。Real 模式使用已有凭据，会计入模型用量；遇到降级或执行错误提前停止并标注未执行。逐题证据含完整合成输入、实际注入条款、业务状态、引用、时延、模型用量。

主集支持命中按 16 个标为可答的题统计，要求期望条款出现在候选集合。答案与引用合规还要求期望条款真的进入生成输入、成为实际引用，并且程序返回其完整原文。拒答按 4 个无依据题统计。一次 HTTP 200 可以承载回答、拒答、澄清或降级。诊断与 12 轮各自计分，历史口径保留独立。

## 设计边界

- BM25 分数只排序。别名门控、完整条款装包、类别筛选和固定语料版本控制候选范围。词表漏覆盖会召回失败，模型也可能从合法候选中选错条款。没有向量检索、重排和多租户权限层。
- 模型只选择一个合法条款编号，由程序输出原文。适合核对数值和否定条件；需要自然语言归纳或跨条款组合时，当前复现仍有表达范围限制。
- 服务面向单用户本机学习。一个共享 Key 管服务入口；会话 ID 只分组。历史在本机 SQLite，模型输入使用独立的有界历史和完整候选条款。第一轮引用以最近保留窗口内首个可定位的历史主题为准。同一会话进行中的并发请求会收到业务冲突。
- 日志仅保留阶段、散列会话标识、工具名、参数字段名、状态、耗时和错误类型。常见号码与邮箱在保存和模型调用前打码。输入仍限合成数据，正则只覆盖常见模式。启动参数关闭访问日志，避免在本机请求日志记录 URL 里的问题。
- 模型请求超时默认 12 秒、只读工具默认 2 秒、处理总限时默认 30 秒，分别由 `API_TIMEOUT`、`W6_TOOL_TIMEOUT`、`W6_TOTAL_TIMEOUT` 配置。SDK 重试关闭；应用层只会在尚未收到内容时对瞬态问题再试一次。超时取消已验证本地协程及连接清理，远端供应商收到的请求能否停止计算由供应商决定。

## 已验证结果

本次服务化改造与运行验证由 Codex 在隔离环境完成，使用合成制度 v2。主集 20 道固定题中，16 道可答题均返回正确条款，4 道无依据题由程序直接拒答。真实模型使用 DeepSeek Flash 非思考模式。两套入口的题集和评分口径各自独立。

| 证据 | 结果 |
| --- | --- |
| [主评估 20 题](evidence/real-20.json) | 20/20。包含逐题引用、支持片段和模型用量。 |
| [单轮诊断](evidence/real-diagnostics.json) | 9/9。包含范围澄清、数字条件、诱导和多候选选择。 |
| [12 轮联调](evidence/real-12-turns.json) | 12/12。首测的第 12 轮曾在正确召回后误澄清，修正指代输入后复测。 |
| [真实工具与服务重启](evidence/real-tool-loopback.json) | 本机 HTTP 请求触发只读检索工具，重启后按运行编号读回相同轨迹。 |

主评估、诊断与多轮报告保留检索、用量和最终输出事件。工具示例保留完整事件序列。更多关于评分的说明见上面的测试章节。
