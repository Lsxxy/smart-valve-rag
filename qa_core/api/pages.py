"""页面、健康检查和会话创建路由。不参与 RAG 检索和答案生成。

包含以下路由：
- GET /                — 单页聊天界面
- GET /admin           — 本地状态页
- GET /health          — 容器与本地健康检查
- POST /api/create_session — 创建会话 ID

设计原则：
- 所有页面路由返回带有 Cache-Control: no-store 头部的静态文件，禁止浏览器缓存。
- 健康检查接口返回当前活跃场景、LLM 状态和检索预热状态。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter
from fastapi.responses import FileResponse

from qa_core.llm.client import llm_runtime_status
from qa_core.retrieval.factory import retrieval_warmup_state
from qa_core.scenarios.registry import resolve_scenario


router = APIRouter()

def _static_page(path: str):
    """返回一个禁止浏览器缓存的静态文件响应。

    使用 Cache-Control: no-store 确保每次请求都从服务器获取最新文件，
    避免浏览器缓存旧版 JS 导致前端功能异常。

    参数：
        path: 静态文件路径。

    返回：
        FileResponse 对象，包含 no-store 缓存控制头部。

    调用顺序：FastAPI 路由层 -> _static_page()。
    """
    return FileResponse(path, headers={"Cache-Control": "no-store"})

@router.get("/")
def read_root():
    """提供单页聊天界面，并禁止浏览器缓存旧版 JS。

    HTTP 路由 GET /

    返回：
        禁止缓存的 index.html 页面。

    调用顺序：FastAPI 路由层 -> read_root()。
    """
    return _static_page("static/index.html")

@router.get("/admin")
def read_admin_page():
    """提供本地状态页。

    HTTP 路由 GET /admin

    返回：
        禁止缓存的 admin.html 页面。

    调用顺序：FastAPI 路由层 -> read_admin_page()。
    """
    return _static_page("static/admin.html")

@router.get("/health")
def health_check():
    """容器与本地健康检查接口。

    HTTP 路由 GET /health

    返回当前系统状态：引擎类型、活跃场景、LLM 连通状态和检索预热状态。
    供容器编排平台（K8s/Docker）和负载均衡器定期探测。

    返回：
        - status: "healthy"。
        - engine: 检索引擎类型。
        - active_scenario_id: 当前活跃场景 ID。
        - active_scenario_name: 当前活跃场景名称。
        - llm: LLM 运行状态。
        - retrieval_warmup: 检索预热状态。

    调用顺序：FastAPI 路由层 -> health_check()。
    """
    scenario = resolve_scenario()
    return {
        "status": "healthy",
        "engine": "langchain_milvus_hybrid",
        "active_scenario_id": scenario.scenario_id,
        "active_scenario_name": scenario.display_name,
        "llm": llm_runtime_status(),
        "retrieval_warmup": retrieval_warmup_state(),
    }

@router.post("/api/create_session")
def create_session():
    """创建页面端使用的会话 ID。

    HTTP 路由 POST /api/create_session

    会话 ID 格式：`{scenario_id}:{uuid4}`，其中 scenario_id 来自解析后的场景配置。
    前端在建立 WebSocket 连接时使用此会话 ID 标识当前对话。

    参数：
    返回：
        包含 session_id 和 scenario_id 的字典。

    调用顺序：FastAPI 路由层 -> create_session()。
    """
    scenario = resolve_scenario()
    return {"session_id": f"{scenario.scenario_id}:{uuid.uuid4()}", "scenario_id": scenario.scenario_id}
