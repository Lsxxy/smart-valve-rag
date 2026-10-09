"""入库文档元数据标准化：为文档补充全量标准元数据字段。

写入 Milvus 前，为每个 Document 补充以下标准元数据：
  - 来源信息：source、file_path、file_name、file_type、doc_id。
  - 场景信息：scenario_id、kb_version、version_seq。
  - 数据域隔离：tenant_id、dataset_id、visibility、allowed_roles。
  - 版本管理：valid_from_seq/valid_to_seq 有效期窗口。
  - 内容类型：record_type、content_type、page_index。

设计决策：
- doc_id 基于文件路径和内容生成稳定哈希，同一文件多次入库 doc_id 不变。
- content_type 优先保留 loader 已标记的值（如 table_row），loader 未标记时
  默认为 "text"，保证表格行和普通文本的 metadata 结构统一。
- 版本管理字段统一由 version_metadata() 注入，避免各模块各自拼装。
"""

from __future__ import annotations
import re
from pathlib import Path
from langchain_core.documents import Document
from qa_core.config.logging_config import get_logger
from qa_core.config.settings import get_settings
from qa_core.document_metadata import is_reviewed_ocr_metadata, is_table_metadata
from qa_core.governance.data_scope import DataScope, resolve_data_scope
from qa_core.governance.kb_versions import version_metadata
from qa_core.scenarios.registry import resolve_scenario
from qa_core.utils import file_fingerprint

logger = get_logger(__name__)

# Markdown 结构噪声：Docling 把 PDF 版面导出成 Markdown 后，表格分隔行（|---|---|）、
# 图片占位符（<!-- image -->）和水平分隔线都会成为独立行。它们不含任何检索语义，
# 却会被切分器当成正文块，最终触发质量门禁的 too_short / low_unique_ratio /
# duplicate_content 告警，因此在这里（切分之前）统一清理。
_HTML_COMMENT_LINE_RE = re.compile(r"^\s*<!--.*?-->\s*$")
_HTML_COMMENT_INLINE_RE = re.compile(r"<!--.*?-->")
_TABLE_SEPARATOR_CHARS = set("-:| \t")
_HORIZONTAL_RULE_RE = re.compile(r"^\s*(?:[-*_]\s*){3,}$")


def _is_markdown_table_separator(line: str) -> bool:
    """判断一行是否只是 Markdown 表格分隔线/框线。（★★★ 核心）

    调用顺序：strip_markdown_artifacts() -> _is_markdown_table_separator()。
    """
    core = line.strip()
    if not core or set(core) - _TABLE_SEPARATOR_CHARS:
        return False
    # 表格分隔线必然包含竖线；纯短横线交给水平分隔线规则处理
    return "|" in core


def strip_markdown_artifacts(text: str) -> tuple[str, dict[str, int]]:
    """清理 Docling Markdown 导出中的结构噪声行，返回清理后的正文与计数。（★★★ 核心）

    清理对象（只删除纯结构行，不碰任何带文字的正文）：
      - 表格分隔行，例如 `|------|------|`、`----|----`；
      - 图片/注释占位符行，例如 `<!-- image -->`；行内注释只删注释本身，保留同行文字；
      - Markdown 水平分隔线，例如 `---`、`***`。

    参数：
        text: loader（主要是 Docling）导出的 Markdown 正文。

    返回：
        (cleaned_text, stats) 元组。stats 记录各类被删除的行数，用于入库报告审计，
        保证清理动作可追溯而不是静默丢弃。

    调用顺序：normalize_documents() -> strip_markdown_artifacts()。
    """
    stats = {
        "table_separator_lines_removed": 0,
        "image_placeholder_lines_removed": 0,
        "horizontal_rule_lines_removed": 0,
    }
    if not text:
        return text, stats
    kept: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if _HTML_COMMENT_LINE_RE.match(line):
            # 整行都是 HTML 注释（Docling 的图片占位符）→ 该行没有文字内容，直接丢弃
            stats["image_placeholder_lines_removed"] += 1
            continue
        if "<!--" in line:
            # 行内注释：只摘掉注释片段，保留同一行的正文
            line = _HTML_COMMENT_INLINE_RE.sub("", line).strip()
            if not line:
                stats["image_placeholder_lines_removed"] += 1
                continue
        if _is_markdown_table_separator(line):
            stats["table_separator_lines_removed"] += 1
            continue
        if _HORIZONTAL_RULE_RE.match(line):
            stats["horizontal_rule_lines_removed"] += 1
            continue
        kept.append(line)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return cleaned, stats



def _is_markdown_table_row(line: str) -> bool:
    """判断一行是否属于 Markdown 表格（至少两个竖线）。（★★★ 核心）

    调用顺序：segment_markdown_blocks() -> _is_markdown_table_row()。
    """
    return line.count("|") >= 2


def segment_markdown_blocks(text: str) -> list[dict[str, object]]:
    """把清理后的正文切分为“普通文本块 / Markdown 表格块”两类片段。（★★★ 核心）

    背景：Docling 把 PDF 表格导出成 Markdown 表格后，普通父子切分会把一行很宽的表
    格按字符数拦腰截断，产生大量“半行”chunk（字符重复率高、脱离表头后没有意义）。
    这里先把表格识别成独立片段，交给下游按“表格行”处理：一行一个证据单元，保留
    表头与行号元数据，既不拆散行列关系，也不会触发低唯一率误报。

    参数：
        text: 已完成结构清理的正文。

    返回：
        片段列表。普通文本片段为 {"kind": "text", "content": str}；
        表格片段为 {"kind": "table", "header": str, "rows": [str, ...]}。

    调用顺序：normalize_documents() -> segment_markdown_blocks()。
    """
    segments: list[dict[str, object]] = []
    text_buffer: list[str] = []
    table_rows: list[str] = []

    def flush_text() -> None:
        content = "\n".join(text_buffer).strip()
        if content:
            segments.append({"kind": "text", "content": content})
        text_buffer.clear()

    def flush_table() -> None:
        rows = [row.strip() for row in table_rows if row.strip()]
        table_rows.clear()
        if not rows:
            return
        if len(rows) == 1:
            # 只有表头、没有数据行：按普通文本处理，避免内容被当成空表格丢弃
            text_buffer.extend(rows)
            return
        segments.append({"kind": "table", "header": rows[0], "rows": rows[1:]})

    for line in text.splitlines():
        if _is_markdown_table_row(line):
            flush_text()
            table_rows.append(line)
            continue
        flush_table()
        text_buffer.append(line)
    flush_table()
    flush_text()
    return segments

def normalize_documents(
    documents: list[Document],
    file_path: Path,
    source: str,
    kb_version: str | None = None,
    scenario_id: str | None = None,
    version_seq: int | None = None,
    data_scope: DataScope | None = None,
    allowed_roles: list[str] | None = None,
) -> list[Document]:
    """为文档补充项目标准元数据：source、scenario_id、数据域、文件信息、doc_id、版本信息。（★★★ 核心）

    执行流程：
      1. 基于文件路径和内容生成稳定的 doc_id，用于去重和增量检测。
      2. 解析场景配置，获取 scenario_id 和版本元数据。
      3. 为每个 Document 注入 source、kb_version、version_seq、数据域隔离等元数据。
      4. 保留 loader 已标记的元数据（如 page_index、content_type 等）。

    参数：
        documents: LangChain Document 列表（loader 原始输出）。
        file_path: 源文件路径，用于生成 doc_id 和 file_name/file_path 元数据。
        source: 业务分类 source，必须属于场景 valid_sources。
        kb_version: 知识库版本号。
        scenario_id: 场景 ID。
        version_seq: 版本序列号，用于 valid_from_seq。
        data_scope: 数据域对象，含 tenant_id/dataset_id/visibility 等隔离信息。
        allowed_roles: 允许的角色列表，用于 RBAC 过滤。

    返回：
        标准化后的 Document 列表，每个 Document 的 metadata 包含完整标准字段。

    调用顺序：入库脚本或索引服务 -> normalize_documents()。
    """
    # 基于文件路径和内容生成稳定的 doc_id，同一文件多次入库 doc_id 不变，用于去重和增量检测
    doc_id = file_fingerprint(file_path)
    scenario = resolve_scenario(scenario_id)
    # 未显式传入 data_scope 时使用全局默认值（default tenant/dataset/public），
    # 确保所有文档至少有一个安全的数据域边界
    settings = get_settings()
    scope = data_scope or resolve_data_scope()
    version_meta = version_metadata(kb_version, scenario.scenario_id, version_seq=version_seq)
    normalized: list[Document] = []
    cleanup_stats = {
        "documents_cleaned": 0,
        "table_separator_lines_removed": 0,
        "image_placeholder_lines_removed": 0,
        "horizontal_rule_lines_removed": 0,
        "markdown_tables_segmented": 0,
        "markdown_table_rows_emitted": 0,
    }
    for index, doc in enumerate(documents):
        loader_metadata = dict(doc.metadata)
        page_index = loader_metadata.get("page", index)
        emitted: list[tuple[str, dict[str, object]]] = []
        # 表格行与人工复核过的 OCR 文本是治理后的完整证据单元，原样保留；
        # 其余正文先做 Markdown 结构清理，再按“文本块 / 表格块”分段。
        if is_table_metadata(loader_metadata) or is_reviewed_ocr_metadata(loader_metadata):
            content = (doc.page_content or "").strip()
            if content:
                emitted.append((content, {}))
        else:
            cleaned, file_stats = strip_markdown_artifacts(doc.page_content or "")
            if any(file_stats.values()):
                cleanup_stats["documents_cleaned"] += 1
            for key, value in file_stats.items():
                cleanup_stats[key] += value
            for segment_index, segment in enumerate(segment_markdown_blocks(cleaned)):
                if segment.get("kind") != "table":
                    emitted.append((str(segment.get("content") or ""), {}))
                    continue
                # 表格按行输出：一行就是一个完整证据单元，附上表头与行号便于引用回查
                cleanup_stats["markdown_tables_segmented"] += 1
                table_id = f"{doc_id}-t{segment_index}"
                header = str(segment.get("header") or "")
                rows = [str(row) for row in (segment.get("rows") or [])]
                for row_number, row in enumerate(rows, start=1):
                    if len(row) > max(settings.parent_chunk_size * 2, 2000):
                        logger.warning(
                            "Markdown 表格行超长，可能触发 too_long 检查：文件=%s 行号=%s 长度=%s",
                            file_path.name,
                            row_number,
                            len(row),
                        )
                    cleanup_stats["markdown_table_rows_emitted"] += 1
                    emitted.append(
                        (
                            row,
                            {
                                "content_type": "table_row",
                                "table_id": table_id,
                                "table_headers": header,
                                "row_number": row_number,
                            },
                        )
                    )
        for content, extra_metadata in emitted:
            metadata = {**loader_metadata, **extra_metadata}
            metadata.update(
                {
                    "source": source,
                    "scenario_id": scenario.scenario_id,
                    # 文档采用引用式增量；在线检索按 version_seq 有效期窗口判断可见性。
                    "source_type": "doc",
                    "record_type": "doc_chunk",
                    "versioning_mode": "reference_incremental",
                    "version_filter_mode": "validity_window",
                    **scope.metadata(allowed_roles=allowed_roles),
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_type": file_path.suffix.lower(),
                    "doc_id": doc_id,
                    # page_index 优先使用 loader 自带页码，fallback 到文档列表中的索引位置
                    "page_index": page_index,
                    # content_type 只覆盖空值：loader 或其分段已标记的值（如 table_row）保持不变
                    "content_type": metadata.get("content_type") or "text",
                    **version_meta,
                }
            )
            normalized.append(Document(page_content=content, metadata=metadata))
    if any(cleanup_stats.values()):
        # 记录清理计数，保证入库过程可审计：删了多少结构行、影响了多少文件
        logger.info(
            "入库前 Markdown 结构清理：文件=%s 计数=%s",
            file_path.name,
            cleanup_stats,
        )
    return normalized
