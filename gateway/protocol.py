"""SPSL L2 跨语言网关协议 — stdin/stdout JSON 行协议.

======================= 协议规范 (10 行) =======================
1. 子进程从 stdin 逐行读 JSON 帧, 每行独立处理; stdout 只写响应帧, 日志走 stderr。
2. 握手: 主进程首帧发 {"type":"hello","protocol":"northstar-gateway/1","spec":{...}};
   子进程必须回 {"type":"hello_ack","impl":"<实现名>","version":"<版本>"}。
3. 计算请求: {"type":"eval","id":<int>,"input":{...}} (id 为请求序号, 回帧须带同 id)。
4. input 两形态 (对应规格 inputs.type):
   - {"type":"contingency_table","table":[[...]]}  (单参数: 计数表)
   - {"type":"two_samples","x":[...],"y":[...]}    (两参数: 两样本数组)
   非有限浮点编码为字符串 "nan"/"inf"/"-inf"; null/字符串原样传 (L4 畸形输入经此传输)。
5. 成功回帧: {"type":"result","id":<同 id>,"statistic":<float|null>,
   "pvalue":<float>,"ci":<数组|null>} —— 字段按考卷规格 outputs 声明; 至少须回 pvalue。
6. 失败回帧 (候选抛异常): {"type":"error","id":<同 id>,"error":"<原因字符串>"}, 进程不退出。
7. 主进程 EOF 关闭 stdin = 终止信号, 子进程收到后正常退出 (exit 0)。
8. 候选逻辑只负责"输入 -> 统计量/p 值/ci", 判定全在主进程 (四层考卷), 子进程无判定权。
9. 违反协议 (非 JSON 行 / 缺字段 / id 不匹配 / 超时 / 子进程崩溃) -> 主进程按
   "候选抛异常" 处理 (L1 记非有限不计分, L2/L3 该 run 全层记拒, L4 记诚实失败), 不静默。
10. 确定性: 帧内容仅由输入决定 (无时间戳/随机量), 同输入两次运行逐字节一致。
=================================================================

主进程侧实现 = ProtocolCandidate: 与 python 函数同调用约定的桥接对象
(__call__(table) / __call__(x, y), 与 spsl.l1.call_candidate 分派一致),
可直接喂给 spsl.run.run_four_layers / gateway.batch。
错误帧 / 协议违反 -> 抛 ProtocolCandidateError (语义 = python 候选抛异常)。
"""
import math
import select
import subprocess
import sys
import threading

from gateway import PROTOCOL  # noqa: E402


class ProtocolCandidateError(RuntimeError):
    """候选侧错误 (错误帧/协议违反/超时/进程死亡) —— 与 python 候选抛异常同语义."""


def _encode_value(v):
    """np 标量/数组/非有限 -> JSON 安全值 (L4 畸形输入经协议传输的关键).

    float 非有限 -> 字符串 "nan"/"inf"/"-inf" (与考卷 L4 的 JSON 编码同款);
    np 数组 -> 嵌套 list; None/字符串/其他原样 (object dtype 表原样保留)。
    """
    import numpy as np

    if isinstance(v, np.ndarray):
        return [_encode_value(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float):
        if not math.isfinite(v):          # 仅非有限转字符串 (L4 编码同款)
            return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
        return v
    if isinstance(v, (list, tuple)):
        return [_encode_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _encode_value(x) for k, x in v.items()}
    return v


class ProtocolCandidate:
    """把任意语言的候选实现包装成可调用的判定对象 (跨语言网关).

    用法:
        cand = ProtocolCandidate(["node", "my_impl.js"],
                                 impl="node-chi2", spec=exam["spec"])
        p = cand([[10, 20], [30, 40]])        # contingency_table
        p = cand([1.0, 2.0], [3.0, 4.0])      # two_samples
        cand.close()

    cmd: 子进程启动命令 (list)。spec: 考卷规格 (握手帧携带, 子进程可校验契约)。
    timeout: 单次 eval 超时 (秒), 超时抛 ProtocolCandidateError。
    """

    def __init__(self, cmd, impl="?", version="?", spec=None, timeout=30.0,
                 cwd=None):
        self.impl, self.version, self.spec = impl, version, spec
        self.timeout = timeout
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, cwd=cwd)
        self._err_lines = []
        self._err_thread = threading.Thread(target=self._drain_stderr,
                                            daemon=True)
        self._err_thread.start()
        self._id = 0
        self._closed = False
        self._handshake()

    # ---- 握手 ----

    def _handshake(self):
        self._send({"type": "hello", "protocol": PROTOCOL, "spec": self.spec})
        ack = self._read_line()
        if ack is None:
            raise ProtocolCandidateError(
                f"handshake failed: subprocess exited without hello_ack "
                f"(stderr: {self._stderr_tail()})")
        if ack.get("type") != "hello_ack":
            raise ProtocolCandidateError(
                f"handshake failed: first frame is not hello_ack: {ack!r}")
        self.impl = ack.get("impl", self.impl)
        self.version = ack.get("version", self.version)

    # ---- 调用约定 (与 spsl.l1.call_candidate 分派一致) ----

    def __call__(self, *args):
        """一参 = 计数表; 两参 = 两样本. 返回 float p 值; 失败抛异常."""
        if len(args) == 1:
            body = {"type": "contingency_table",
                    "table": _encode_value(args[0])}
        elif len(args) == 2:
            body = {"type": "two_samples",
                    "x": _encode_value(args[0]), "y": _encode_value(args[1])}
        else:
            raise ProtocolCandidateError(f"candidate accepts only 1 arg (table) or 2 args (two samples), "
                                         f"got {len(args)}")
        self._id += 1
        self._send({"type": "eval", "id": self._id, "input": body})
        resp = self._read_line()
        if resp is None:
            raise ProtocolCandidateError(
                f"eval #{self._id} got no response: subprocess exited "
                f"(stderr: {self._stderr_tail()})")
        if resp.get("type") == "result":
            pv = resp.get("pvalue")
            try:
                return float(pv)
            except (TypeError, ValueError):
                raise ProtocolCandidateError(
                    f"result frame pvalue is not a number: {pv!r}") from None
        if resp.get("type") == "error":
            raise ProtocolCandidateError(
                f"candidate error frame (#{resp.get('id')}): {resp.get('error')}")
        raise ProtocolCandidateError(f"protocol violation: unknown response frame {resp!r}")

    # ---- 帧传输 ----

    def _send(self, frame):
        import json

        if self._closed or self.proc.poll() is not None:
            raise ProtocolCandidateError(
                f"subprocess exited (rc={self.proc.poll()}), cannot send "
                f"(stderr: {self._stderr_tail()})")
        try:
            self.proc.stdin.write(json.dumps(frame, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ProtocolCandidateError(
                f"failed to write to subprocess: {exc} (stderr: {self._stderr_tail()})") from None

    def _read_line(self):
        """带超时读一行 JSON (POSIX select), 超时/EOF -> None 由调用方判语义."""
        import json

        fd = self.proc.stdout.fileno()
        r, _, _ = select.select([fd], [], [], self.timeout)
        if not r:
            raise ProtocolCandidateError(
                f"eval timed out (> {self.timeout}s), subprocess unresponsive")
        line = self.proc.stdout.readline()
        if line == "":
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            raise ProtocolCandidateError(
                f"protocol violation: subprocess output is not a JSON line: {line[:200]!r}") from None

    def _drain_stderr(self):
        for line in iter(self.proc.stderr.readline, ""):
            self._err_lines.append(line.rstrip("\n"))
            if len(self._err_lines) > 200:
                self._err_lines.pop(0)

    def _stderr_tail(self) -> str:
        return " | ".join(self._err_lines[-5:]) or "(none)"

    # ---- 生命周期 ----

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.proc.stdin.close()       # EOF = 终止信号 (协议第 7 行)
            self.proc.wait(timeout=5)
        except (BrokenPipeError, OSError):
            pass
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        finally:
            self._err_thread.join(timeout=1)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
