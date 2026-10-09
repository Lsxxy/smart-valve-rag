"""可观测性适配器：可选的 LangSmith Tracing 数据上报。

入库质量报告和门禁由本地代码执行。
此包只包含 RAG 运行时用来附加业务元数据的 LangSmith 适配器。当 LANGSMITH_TRACING
启用时，pipeline 收尾阶段调用 record_query_trace() 将问答请求记录到 LangSmith。

设计决策：
- LangSmith 是可选 Trace 旁路：未启用或网络失败时只记录日志，不影响用户请求。
- 入库质量检查不依赖 LangSmith。

调用顺序：pipeline 收尾或管理 API -> observability。
"""
