# -*- coding: utf-8 -*-
# w6_hr_policy/app/auth.py
"""v1.7 公网部署前置：API Key 准入控制（fail-closed）。
- 服务端未配置 W6_API_KEY → 所有受保护路由一律 401（绝不因漏配而裸奔公网，启动时有日志提示）；
- 请求头 X-API-Key 与 W6_API_KEY 用 secrets.compare_digest 常量时间比对，防时序侧信道；
- /health 保持公开：B4 部署 Gate 与监控探活依赖它，且不泄露业务数据。
Key 本体只放 .env（W6_API_KEY），代码零硬编码；.env 为孟轩手动编辑（ Administrators 只读 ACL）。"""
import os, secrets
from fastapi import Header, HTTPException

def require_api_key(x_api_key: str = Header(default=None)):
    expected = (os.getenv("W6_API_KEY") or "").strip()
    if not expected:
        # fail-closed：未配置 Key 视为「服务未就绪」，拒绝一切业务请求
        raise HTTPException(status_code=401,
                            detail="服务端未配置 W6_API_KEY，已按 fail-closed 拒绝请求")
    supplied = (x_api_key or "").strip()
    if not supplied or not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="无效 API Key")
