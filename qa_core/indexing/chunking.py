"""文档切分策略：parent-child 多粒度切片，平衡检索精度与上下文完整性。

把标准化后的 Document 按 parent-child 双层策略切分：
  - parent 块：较大的文本片段（默认 1024 字符），提供完整上下文窗口。
  - child 块：较小的子片段（默认 256 字符），用于精确语义匹配。
  检索时子块命中后携带父块上下文一同返回，实现在精确召回率和上下文
  完整性之间的平衡。

设计决策：
- 表格行和已复核 OCR 文本不经过父子切分——它们已经是治理后的完整证据单元。
- Markdown 文件先通过标题切分器结构化（按 #/##/### 分层），再进入递归切分。
- chunk_id 和 parent_id 由内容稳定哈希生成，相同内容产生相同 id，支持增量重建。

依赖分层：
- langchain_text_splitters：MarkdownHeaderTextSplitter、RecursiveCharacterTextSplitter。
- qa_core.config.settings：parent_chunk_size / child_chunk_size 等配置。
- qa_core.utils.stable_hash：基于内容生成稳定 id。
"""

from __future__ import annotations
import re
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from qa_core.config.logging_config import get_logger
from qa_core.config.settings import get_settings
from qa_core.document_metadata import is_reviewed_ocr_metadata, is_table_metadata
from qa_core.utils import stable_hash

logger = get_logger(__name__)

# 与 qa_core.quality.chunk 的 too_short 规则（长度 < 30）保持一致：低于该长度的
# 碎片不单独入库，而是合并进同文件的相邻块，避免标题、页码、表头残留变成噪声 chunk。
MIN_MEANINGFUL_CHUNK_CHARS = 30
_NON_CONTENT_RE = re.compile(r"[\W_]+", re.UNICODE)

CHINESE_SEPARATORS = [
    "\n\n",
    "\n",
    "。", "！", "？", "；",
    ";", ".", "!", "?",
    "，", ",",
    " ",
    "",
    # 原因： 中英文混排文档需要同时支持中文句号/感叹号/问号和英文句点/分号作为切分边界，递归切分器按 separator 顺序优先匹配大粒度分隔符
]

def _has_meaningful_content(content: str) -> bool:
    """判断 chunk 去掉标点与符号后是否还剩实际文字（中文、字母或数字）。（★★★ 核心）

    调用顺序：govern_chunks() -> _has_meaningful_content()。
    """
    return bool(_NON_CONTENT_RE.sub("", content or "").strip())


def _chunk_file_key(metadata: dict) -> str:
    """返回“同一文件”判定键，保证跨文件的内容不会被误合并或误去重。

    调用顺序：govern_chunks() -> _chunk_file_key()。
    """
    return str(
        metadata.get("doc_id")
        or metadata.get("file_path")
        or metadata.get("file_name")
        or metadata.get("source")
        or ""
    )


def _is_governed_unit(metadata: dict) -> bool:
    """表格行与已复核 OCR 文本是治理后的完整证据单元，不参与合并与去重。

    调用顺序：govern_chunks() -> _is_governed_unit()。
    """
    return is_table_metadata(metadata) or is_reviewed_ocr_metadata(metadata)


def govern_chunks(chunks: list[Document]) -> tuple[list[Document], dict[str, int]]:
    """对切分结果做入库前治理：丢弃无文字块、合并过短碎片、同文件内容去重。（★★★ 核心）

    背景：Docling 把 PDF 版面导出成 Markdown 后，标题、图片占位符、表格线会形成
    极短或完全重复的 chunk。这类 chunk 没有检索价值，却会造成重复召回、引用噪声，
    并触发入库质量门禁的 too_short / duplicate_content / low_unique_ratio 告警。

    处理规则（只作用于普通正文块，表格行与已复核 OCR 文本原样保留）：
      1. 丢弃去标点后没有文字的块（纯符号、纯空白）。
      2. 长度 < MIN_MEANINGFUL_CHUNK_CHARS 的碎片：优先合并进同文件中后面的同源块，
         其次合并进前面的同源块；被合并的碎片同时补进目标块的 parent_content，
         保证“子块内容 ⊆ 父块内容”的既有不变式不被破坏。
      3. 同一文件内内容完全相同的块只保留首次出现的一份，避免重复召回。

    参数：
        chunks: split_documents() 切分后的子块列表。

    返回：
        (governed_chunks, stats) 元组。stats 记录丢弃/合并/去重数量，供日志与入库
        报告审计，保证治理动作可追溯。

    调用顺序：split_documents() -> govern_chunks()。
    """
    stats = {
        "dropped_punctuation_chunks": 0,
        "merged_short_fragments": 0,
        "deduplicated_chunks": 0,
        "kept_orphan_fragments": 0,
    }
    # ── 规则 1：丢弃去标点后没有文字的块 ──
    retained: list[Document] = []
    for chunk in chunks:
        if _is_governed_unit(chunk.metadata) or _has_meaningful_content(chunk.page_content):
            retained.append(chunk)
            continue
        stats["dropped_punctuation_chunks"] += 1

    # ── 规则 2：过短碎片合并进同文件相邻块 ──
    merged: list[Document] = []
    for index, chunk in enumerate(retained):
        content = (chunk.page_content or "").strip()
        if _is_governed_unit(chunk.metadata) or len(content) >= MIN_MEANINGFUL_CHUNK_CHARS:
            merged.append(chunk)
            continue
        file_key = _chunk_file_key(chunk.metadata)
        # 先找后面的同文件块：标题类碎片与紧随其后的正文合并，语义最自然
        target = next(
            (
                candidate
                for candidate in retained[index + 1 :]
                if not _is_governed_unit(candidate.metadata)
                and _chunk_file_key(candidate.metadata) == file_key
            ),
            None,
        )
        if target is not None:
            target.page_content = f"{content}\n\n{target.page_content}".strip()
            parent_content = str(target.metadata.get("parent_content") or "").strip()
            target.metadata["parent_content"] = (
                f"{content}\n\n{parent_content}" if parent_content else content
            )
            stats["merged_short_fragments"] += 1
            continue
        # 文件末尾的孤立碎片：并入前面最近的同文件块，并同步扩展其父块内容
        previous = next(
            (
                candidate
                for candidate in reversed(merged)
                if not _is_governed_unit(candidate.metadata)
                and _chunk_file_key(candidate.metadata) == file_key
            ),
            None,
        )
        if previous is not None:
            previous.page_content = f"{previous.page_content.strip()}\n\n{content}"
            parent_content = str(previous.metadata.get("parent_content") or previous.page_content)
            if content not in parent_content:
                previous.metadata["parent_content"] = f"{parent_content}\n\n{content}".strip()
            stats["merged_short_fragments"] += 1
            continue
        # 整个文件只有一个过短块时无从合并，保留原样并计数，交由质量报告暴露
        merged.append(chunk)
        stats["kept_orphan_fragments"] += 1

    # ── 规则 3：同一文件内的重复内容去重 ──
    # 表格行同样参与去重：同一文件内内容完全相同的表格行会造成重复召回，且行内容
    # 本身不含差异信息时保留多份没有检索价值（表头与行号仍保留在首次出现的块上）。
    governed: list[Document] = []
    seen: set[tuple[str, str]] = set()
    for chunk in merged:
        if is_reviewed_ocr_metadata(chunk.metadata):
            governed.append(chunk)
            continue
        key = (_chunk_file_key(chunk.metadata), "".join((chunk.page_content or "").split()))
        if key in seen:
            stats["deduplicated_chunks"] += 1
            continue
        seen.add(key)
        governed.append(chunk)
    return governed, stats
def chunk_identity(page_content: str, metadata: dict) -> tuple[str, str]:
    """基于正文和标准元数据生成 parent_id 与 chunk_id。

    调用顺序：入库脚本或索引服务 -> chunk_identity()。
    """
    parent_content = str(metadata.get("parent_content") or page_content or "").strip()
    if is_table_metadata(metadata):
        # 表格行的 parent_id 额外包含 table_id、sheet_name、row_number，确保同一张表的同一行
        # 在多次入库时 id 稳定，且不同行的 chunk 不会混淆；行列关系由 table_id + row_number 唯一锁定
        parent_id = stable_hash(
            metadata.get("scenario_id"),
            metadata.get("kb_version"),
            metadata.get("embedding_model_version"),
            metadata.get("chunk_schema_version"),
            metadata.get("doc_id"),
            metadata.get("table_id"),
            metadata.get("sheet_name"),
            metadata.get("row_number"),
            parent_content,
        )
        # 表格行的子块 id 只基于 parent_id + parent_content 生成，因为表格行不会进一步切分子块
        chunk_id = stable_hash(parent_id, parent_content)
        return parent_id, chunk_id

    parent_id = stable_hash(
        metadata.get("scenario_id"),
        metadata.get("kb_version"),
        metadata.get("embedding_model_version"),
        metadata.get("chunk_schema_version"),
        metadata.get("doc_id"),
        parent_content,
    )
    # 普通文档的 chunk_id 使用子块自身的 page_content（而非 parent_content）参与 hash，
    # 使得同一父块下不同子块拥有不同 chunk_id
    chunk_id = stable_hash(parent_id, page_content)
    return parent_id, chunk_id


def split_documents(documents: list[Document]) -> tuple[list[Document], list[str]]:
    """将文档切成可检索的子块并保留父块上下文。（★★★ 核心）

    子块用于精确召回，父块上下文保存在 metadata.parent_content 中。
    检索时子块命中后携带父块上下文一同返回。

    执行流程：
      1. 对每个 Document，根据 file_type 和 metadata 选择切分策略。
      2. 表格行和已复核 OCR：不切分，整行/整段作为唯一块。
      3. Markdown：先按标题切分（#/##/###），再递归文本切分。
      4. 其他文本：直接按 parent_chunk_size 递归切分。
      5. 对每个父块生成 child 子块，计算稳定 parent_id 和 chunk_id。

    参数：
        documents: 标准化后的 LangChain Document 列表。

    返回：
        (chunks_list, ids_list) 元组。chunks_list 包含所有子块，每个块
        携带 parent_content（父块全文）和 parent_id/chunk_id。
        ids_list 是所有子块的 chunk_id 列表，与 chunks_list 一一对应。

    调用顺序：入库脚本或索引服务 -> split_documents()。
    """
    # 原因： parent-child 分别切分使子块保持精确命中而父块提供完整上下文窗口，比单一切片在精确召回率和上下文完整性之间取得更好平衡
    settings = get_settings()
    markdown_headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=settings.parent_overlap,
        separators=CHINESE_SEPARATORS,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_overlap,
        separators=CHINESE_SEPARATORS,
    )

    # ── 主处理循环：逐个 Document 执行切分 ──
    chunks: list[Document] = []
    ids: list[str] = []
    # Docling 导出的 Markdown 可能包含重复的占位标记、页眉或短表格片段。原始
    # 内容哈希在这种情况下会产生相同的 chunk_id，而 Milvus 一次批量写入要求
    # 主键唯一。计数器以本次文件的切分顺序为准：相同输入会得到相同序号，因而
    # 仍保留增量入库需要的稳定性。
    chunk_id_occurrences: dict[str, int] = {}

    def make_unique_chunk_id(base_chunk_id: str, metadata: dict) -> str:
        occurrence = chunk_id_occurrences.get(base_chunk_id, 0)
        chunk_id_occurrences[base_chunk_id] = occurrence + 1
        if occurrence == 0:
            return base_chunk_id
        metadata["chunk_occurrence"] = occurrence
        return stable_hash(base_chunk_id, occurrence)

    for doc in documents:
        file_type = str(doc.metadata.get("file_type", "")).lower()
        parent_docs: list[Document]
        if is_table_metadata(doc.metadata) or is_reviewed_ocr_metadata(doc.metadata):
            # 表格行和已复核 OCR 文本都是治理后的完整证据单元。
            # 表格不能被拆散行列关系；OCR 复核稿不能丢失复核状态、置信度和原始文件说明。
            parent_content = doc.page_content.strip()
            # 空行跳过：表格中全空行或 OCR 空白页没有检索价值，不生成 chunk 以节省 Milvus 存储
            if not parent_content:
                continue
            metadata = dict(doc.metadata)
            metadata["parent_content"] = parent_content
            parent_id, chunk_id = chunk_identity(parent_content, metadata)
            chunk_id = make_unique_chunk_id(chunk_id, metadata)
            metadata.update(
                {
                    "parent_id": parent_id,
                    "chunk_id": chunk_id,
                }
            )
            # 表格/OCR 不经过父子切分：直接以整行/整段作为唯一块，parent_id == chunk_id
            chunks.append(Document(page_content=parent_content, metadata=metadata))
            ids.append(chunk_id)
            continue
        elif file_type == ".md":
            header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=markdown_headers)
            # Markdown 标题会先转成结构化元数据，再进入递归切分，能提升来源标签和上下文质量。
            # 如果 Markdown 解析失败，说明资料格式需要修复；入库阶段应该暴露异常并进入
            # 异常文件报告，而不是悄悄按普通文本切分，造成章节 metadata 丢失。
            header_docs = header_splitter.split_text(doc.page_content)
            for header_doc in header_docs:
                # Markdown 标题切分器会生成新的 Document，这里把原始文件 metadata
                # 补回去，避免切分后丢失 source、file_name、doc_id 等关键字段。
                header_doc.metadata.update(doc.metadata)
            parent_docs = parent_splitter.split_documents(header_docs)
        else:
            # 非 Markdown 普通文本：直接按父块大小切分，不再经过标题解析
            parent_docs = parent_splitter.split_documents([doc])

        # ── 父块循环处理：对每个父块生成子块 ──
        for parent_doc in parent_docs:
            parent_content = parent_doc.page_content
            # parent_id 和 chunk_id 都纳入 kb_version、embedding_model_version 和 chunk_schema_version。
            # 这样同一个文件在两个知识库版本里可以同时存在，不会因为内容相同而主键冲突。
            child_docs = child_splitter.split_documents([parent_doc])
            # 子块为空意味着父块小于最小子块尺寸，此时跳过（父块本身仍可检索）
            for child_doc in child_docs:
                # chunk_id 由父块和子块内容共同决定。同一文件未变化时 id 稳定；文件变化时
                # id 会变化，配合 manifest 删除旧 chunk 后重建。
                metadata = dict(child_doc.metadata)
                metadata["parent_content"] = parent_content
                parent_id, chunk_id = chunk_identity(child_doc.page_content, metadata)
                chunk_id = make_unique_chunk_id(chunk_id, metadata)
                metadata.update(
                    {
                        "parent_id": parent_id,
                        "chunk_id": chunk_id,
                    }
                )
                chunks.append(Document(page_content=child_doc.page_content, metadata=metadata))
                ids.append(chunk_id)

    # ── 切分后治理：丢弃无文字块、合并过短碎片、同文件去重 ──
    # 治理会修改块内容，因此 chunk_id/parent_id 必须在治理之后统一重算，否则会出现
    # 内容与主键不一致的 chunk（增量重建会因此错删或重写）。
    chunks, cleanup_stats = govern_chunks(chunks)
    if any(cleanup_stats.values()):
        logger.info("入库切分治理：计数=%s", cleanup_stats)
    ids = []
    governed_occurrences: dict[str, int] = {}
    for chunk in chunks:
        metadata = dict(chunk.metadata)
        metadata.pop("chunk_occurrence", None)
        parent_content = str(metadata.get("parent_content") or chunk.page_content).strip()
        metadata["parent_content"] = parent_content
        parent_id, base_chunk_id = chunk_identity(chunk.page_content, metadata)
        occurrence = governed_occurrences.get(base_chunk_id, 0)
        governed_occurrences[base_chunk_id] = occurrence + 1
        if occurrence:
            metadata["chunk_occurrence"] = occurrence
            chunk_id = stable_hash(base_chunk_id, occurrence)
        else:
            chunk_id = base_chunk_id
        metadata["parent_id"] = parent_id
        metadata["chunk_id"] = chunk_id
        chunk.metadata = metadata
        ids.append(chunk_id)
    return chunks, ids


