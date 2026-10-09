"""演示岗位及资料授权；角色选择不替代身份认证。"""

BUSINESS_ROLES = ("sales", "customer_service")


def normalize_business_roles(roles=None):
    """只接受两个业务岗位，未指定时默认销售。"""
    if roles is None or roles == []:
        return ["sales"]
    if not isinstance(roles, (list, tuple)):
        raise ValueError("user_roles 必须是角色列表")
    result = []
    for role in roles:
        if role not in BUSINESS_ROLES:
            raise ValueError(f"不支持的业务角色：{role}")
        if role not in result:
            result.append(role)
    return result
