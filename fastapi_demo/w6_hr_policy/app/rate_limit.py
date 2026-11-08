# -*- coding: utf-8 -*-
# w6_hr_policy/app/rate_limit.py
"""v1.7 公网部署前置：每 IP 滑动窗口限流（单实例内存版）。
- 每 IP 60 秒窗口最多 60 次（W6_RL_LIMIT / W6_RL_WINDOW 可配），超限 429；
- threading.Lock 保证并发安全（同步依赖在线程池、异步依赖在事件循环中都会并发进入）；
- 「到达即记账」：被拒请求也占窗口，持续轰击持续被拒，正常用户停 60s 自动恢复；
- IP 条目数达到上限时按最后活跃时间淘汰最旧一半，防伪造海量 IP 撑爆内存；
- 局限（注释即文档）：单进程内存计数，uvicorn --workers N 或多副本部署时每个进程各算各的，
  生产环境应换 Redis 集中计数；本阶段单实例部署够用。
"""
import os, threading, time
from collections import deque
from fastapi import HTTPException, Request

class SlidingWindowLimiter:
    def __init__(self, limit=60, window=60.0, max_ips=10000):
        self.limit = limit
        self.window = window
        self.max_ips = max_ips
        self._hits = {}                      # ip -> deque[time.monotonic()]
        self._lock = threading.Lock()

    def check(self, ip):
        """到达即记账；返回 True=放行 / False=超限。"""
        now = time.monotonic()
        with self._lock:
            if len(self._hits) >= self.max_ips and ip not in self._hits:
                self._evict()
            dq = self._hits.setdefault(ip, deque(maxlen=self.limit + 1))
            while dq and dq[0] <= now - self.window:
                dq.popleft()
            dq.append(now)
            return len(dq) <= self.limit

    def _evict(self):
        """内存自保：按最后活跃时间淘汰一半最旧 IP（时间复杂度 O(n)，触发频率极低）。"""
        oldest = sorted(self._hits.items(), key=lambda kv: kv[1][-1])[: self.max_ips // 2]
        for ip, _ in oldest:
            self._hits.pop(ip, None)

    def reset(self):
        """仅测试用：清空计数，避免用例间窗口串扰。"""
        with self._lock:
            self._hits.clear()

def _int_env(name, default):
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default

limiter = SlidingWindowLimiter(
    limit=_int_env("W6_RL_LIMIT", 60),
    window=float(_int_env("W6_RL_WINDOW", 60)),
)

async def rate_limit(request: Request):
    """FastAPI 依赖：排在 require_api_key 之前执行——先限流再验 Key，
    顺带把 Key 爆破速率也压到每分钟 60 次。"""
    ip = request.client.host if request.client else "unknown"
    if not limiter.check(ip):
        raise HTTPException(status_code=429,
                            detail=f"请求过于频繁（每 {int(limiter.window)} 秒最多 {limiter.limit} 次），请稍后再试")
