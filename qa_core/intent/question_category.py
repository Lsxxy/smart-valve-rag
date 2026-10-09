"""RAG 问题类别识别 — 决定检索计划、Prompt 风险模板和输出要求。

意图回答“用户想做什么”，问题类别控制“系统应该怎样回答”。通用规则保留给已有
业务场景；智能阀销售场景使用更贴近销售边界的覆盖规则，避免把型号咨询、安装条件
等正常售前问题误判为排障。
"""

from __future__ import annotations
import re
from typing import Literal
QuestionCategory = Literal[
    "default",
    "pricing",
    "compliance",
    "troubleshooting",
    "summary",
    "after_sales_handoff",
]

PRICING_HINTS = re.compile(
    r"(费用|价格|学费|报价|优惠|折扣|退费|退款|发票|账单|支付|开票|续费|合同金额|收费|付款|"
    r"赔付|赔款|金额|回款|预算|采购|报销|信用证|保证金|付款条件|收款账户|打款|结算|税费)"
)
COMPLIANCE_HINTS = re.compile(
    r"(合规|隐私|个人信息|客户数据|外部平台|合同|条款|审计|风险|违规|保密|权限审批|数据出境|授权|责任|"
    r"制裁|名单|出口管制|强制性|规范|标准|图纸冲突|除外|免责|不赔|拒赔|HS\s*编码|海关|报关|申报|"
    r"归类|最终用途|最终用户|尽调|受限空间|作业审批|安全确认|安全技术交底|检验批|隐蔽工程|验收|"
    r"既往症|如实告知|保单复效)"
)
TROUBLESHOOTING_HINTS = re.compile(r"(故障|报错|失败|无法|异常|告警|报警|排查|维修|巡检|安装|配置|连接不上|超时|错误码)")
SUMMARY_HINTS = re.compile(r"(总结|概括|归纳|整理|对比|区别|有哪些|都包括|主要内容|学习内容|大纲)")

# 智能阀销售场景只回答产品介绍、初步选型和系统集成咨询。下面的规则刻意不复用
# 通用“采购/安装/故障”等宽泛词：例如“采购 EVG 选哪个型号”“如何安装”仍属于
# 正常售前资料检索，不应被错误送去价格口径或售后流程。
SMART_VALVE_PRICING_HINTS = re.compile(
    r"(价格|报价|优惠|折扣|交期|交货期|货期|付款|回款|开票|发票|税费|合同金额|结算|账期)"
)
SMART_VALVE_COMPLIANCE_HINTS = re.compile(
    r"(合同|条款|质保|保修|认证|防爆|合规|法规|准入|责任|承诺|节能率|能效承诺|"
    r"安全责任|第三方检测|验收承诺)"
)
SMART_VALVE_AFTER_SALES_HINTS = re.compile(
    r"(故障|报警|告警|报错|错误码|不动作|无法启动|异常|维修|拆机|拆卸|"
    r"恢复出厂|返修|漏水|卡死|现场处置|现场调试|故障排查)"
)
SMART_VALVE_SUMMARY_HINTS = re.compile(r"(总结|概括|归纳|汇总|对比|差异|优缺点|选型对比)")
TABLE_QUERY_HINTS = re.compile(
    r"(表格|清单|台账|明细|字段|列名|行号|sheet|工作表|导出|列表|矩阵|评分表|检查表|对账|"
    r"材料清单|验收项|付款节点|状态|责任人|金额|数量|单价|收款账户|银行卡|发票号码|箱单|装箱单)"
)


def infer_question_category(query: str, *, scenario_id: str | None = None) -> QuestionCategory:
    """判断 RAG 回答风险类别。

    风险类别驱动检索策略（召回阈值/上下文大小）、Prompt 模板和必要时的确定性转人工。
    ``smart_valve_sales`` 采用销售边界优先级：价格 > 合规承诺 > 售后转人工 > 总结 > 默认；
    其他场景维持原有通用优先级，保证不改变既有业务的分类结果。

    调用顺序：检索准备阶段 -> infer_question_category()。
    """
    normalized = query.strip().lower()
    if scenario_id == "smart_valve_sales":
        if SMART_VALVE_PRICING_HINTS.search(normalized):
            return "pricing"
        if SMART_VALVE_COMPLIANCE_HINTS.search(normalized):
            return "compliance"
        if SMART_VALVE_AFTER_SALES_HINTS.search(normalized):
            return "after_sales_handoff"
        if SMART_VALVE_SUMMARY_HINTS.search(normalized):
            return "summary"
        return "default"
    # 费用/价格类（高优先级）：返回金额/价格错误可能导致经济损失或客诉，需最严格的检索控制，召回阈值 ≥ 0.86
    if PRICING_HINTS.search(normalized):
        return "pricing"
    # 合规/法律类（极高优先级）：合规答案错误可能导致监管处罚，检索门槛最高（≥0.86），并启用免责声明模板
    if COMPLIANCE_HINTS.search(normalized):
        return "compliance"
    # 排障类（中优先级）：用户正在遭遇问题，需要丰富且准确的步骤化上下文，Prompt 侧重操作指引
    if TROUBLESHOOTING_HINTS.search(normalized):
        return "troubleshooting"
    # 总结类（低优先级）：信息量需求大但精度要求相对宽松，召回阈值可降至 0.5 以获取更多上下文
    if SUMMARY_HINTS.search(normalized):
        return "summary"
    # 默认类：常规检索策略和通用 Prompt 模板，不做特殊约束
    return "default"


def is_table_query(query: str) -> bool:
    """判断是否为表格/清单查询。示例："清单有哪些字段"→True；"费用多少"→False。（★★ 理解）

    表格查询的业务影响：匹配到表格类时，检索策略会扩大文档候选并禁用模糊 FAQ 直出，
    因为表格内容语义相似但不同列/行会导致严重误导（如"验收项清单"和"付款节点清单"语义相近但内容完全不同）。

    调用顺序：检索准备阶段 -> is_table_query()。
    """
    normalized = query.strip().lower()
    # 命中任意关键词即判定为表格类查询，后续链路据此调整检索策略和后处理逻辑
    return bool(TABLE_QUERY_HINTS.search(normalized))

