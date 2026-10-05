"""极大有限坐标下的近距离相遇 / 接触传播回归测试。

合法的极大有限坐标（如 1e155）曾触发距离平方溢出（dx² 与阈值² 都变成
inf，qc=NaN），使 POST /proximity 漏报本应计入的相遇，依赖相遇区间的
POST /contact-tracing 也随之把可达车次误判为不可达。

修复后：距离判定、区间端点与最小距离对任意合法有限输入均不得溢出；
仅距离严格小于阈值的正时长计入；恰等阈值或仅瞬时接触不计；
接点不满足条件时不得合并。
"""
import math

from tests.conftest import DAY_WINDOWS


def _register(client, zone_id, half):
    r = client.post(
        "/api/v1/zones",
        json={
            "zone_id": zone_id,
            "exterior": [[-half, -half], [half, -half], [half, half], [-half, half]],
            "holes": [],
            "windows": DAY_WINDOWS,
        },
    )
    assert r.status_code == 201, r.text


def _upload(client, trip, points):
    r = client.post(
        f"/api/v1/trips/{trip}/points",
        json={"points": [{"point_id": f"p{i}", "ts": ts, "x": x, "y": y}
                         for i, (ts, x, y) in enumerate(points)]},
    )
    assert r.status_code == 200, r.text


def _proximity(client, zone_id, trips, threshold, **extra):
    body = {
        "trip_ids": trips,
        "since": "2026-09-30T00:00:00Z",
        "until": "2026-09-30T00:00:10Z",
        "max_distance": threshold,
    }
    body.update(extra)
    return client.post(f"/api/v1/zones/{zone_id}/proximity", json=body)


def _tracing(client, zone_id, trips, source, threshold, **extra):
    body = {
        "trip_ids": trips,
        "source_trip_id": source,
        "since": "2026-09-30T00:00:00Z",
        "until": "2026-09-30T00:00:10Z",
        "max_distance": threshold,
    }
    body.update(extra)
    return client.post(f"/api/v1/zones/{zone_id}/contact-tracing", json=body)


# ---------------------------------------------------------------------
# 静止场景：距离平方溢出曾直接漏报
# ---------------------------------------------------------------------

def test_huge_stationary_pair_is_reported(client):
    """两车在区内连续静止 10s、平面距离 1e155、阈值 2e155：整段相遇。"""
    big = 1e155
    _register(client, "oh1", 2 * big)
    _upload(client, "A", [
        ("2026-09-30T00:00:00Z", 0.0, 0.0),
        ("2026-09-30T00:00:10Z", 0.0, 0.0),
    ])
    _upload(client, "B", [
        ("2026-09-30T00:00:00Z", big, 0.0),
        ("2026-09-30T00:00:10Z", big, 0.0),
    ])

    r = _proximity(client, "oh1", ["A", "B"], 2 * big)
    assert r.status_code == 200, r.text
    pairs = r.json()["pairs"]
    assert len(pairs) == 1
    intervals = pairs[0]["intervals"]
    assert len(intervals) == 1
    seg = intervals[0]
    assert seg["start"] == "2026-09-30T00:00:00Z"
    assert seg["end"] == "2026-09-30T00:00:10Z"
    assert seg["duration_sec"] == 10.0
    assert seg["min_distance"] == big
    assert math.isfinite(seg["min_distance"])


def test_huge_stationary_exact_threshold_excluded(client):
    """静止距离恰等于极大阈值：严格小于不成立，合法无相遇返回空。"""
    big = 1e155
    _register(client, "oh2", 2 * big)
    _upload(client, "A", [
        ("2026-09-30T00:00:00Z", 0.0, 0.0),
        ("2026-09-30T00:00:10Z", 0.0, 0.0),
    ])
    _upload(client, "B", [
        ("2026-09-30T00:00:00Z", big, 0.0),
        ("2026-09-30T00:00:10Z", big, 0.0),
    ])
    r = _proximity(client, "oh2", ["A", "B"], big)
    assert r.status_code == 200, r.text
    assert r.json()["pairs"] == []


def test_huge_stationary_contact_tracing_reachable(client):
    """同样的大坐标静止相遇必须能经接触传播把下游判为可达。"""
    big = 1e155
    _register(client, "oh3", 2 * big)
    _upload(client, "A", [
        ("2026-09-30T00:00:00Z", 0.0, 0.0),
        ("2026-09-30T00:00:10Z", 0.0, 0.0),
    ])
    _upload(client, "B", [
        ("2026-09-30T00:00:00Z", big, 0.0),
        ("2026-09-30T00:00:10Z", big, 0.0),
    ])
    r = _tracing(client, "oh3", ["A", "B"], "A", 2 * big)
    assert r.status_code == 200, r.text
    results = {row["trip_id"]: row for row in r.json()["results"]}
    assert results["A"]["reachable"] is True and results["A"]["hops"] == 0
    assert results["B"]["reachable"] is True
    assert results["B"]["hops"] == 1
    assert results["B"]["chain"] == ["A", "B"]
    assert results["B"]["arrived_at"] == "2026-09-30T00:00:00Z"


def test_huge_stationary_exact_threshold_not_transmissible(client):
    """恰等阈值的大坐标静止相遇不产生可传递窗口：下游不可达。"""
    big = 1e155
    _register(client, "oh4", 2 * big)
    _upload(client, "A", [
        ("2026-09-30T00:00:00Z", 0.0, 0.0),
        ("2026-09-30T00:00:10Z", 0.0, 0.0),
    ])
    _upload(client, "B", [
        ("2026-09-30T00:00:00Z", big, 0.0),
        ("2026-09-30T00:00:10Z", big, 0.0),
    ])
    r = _tracing(client, "oh4", ["A", "B"], "A", big)
    assert r.status_code == 200, r.text
    results = {row["trip_id"]: row for row in r.json()["results"]}
    assert results["B"]["reachable"] is False
    assert results["B"]["arrived_at"] is None and results["B"]["chain"] is None


# ---------------------------------------------------------------------
# 运动原子段：平方/交叉积溢出与端点、最小距离
# ---------------------------------------------------------------------

def test_huge_moving_crossing_interval(client):
    """大坐标相对运动穿越阈值：解析根、端点与顶点最小距离均正确且有限。

    A 静止于 x=0；B 从 x=3e155 匀速到 x=0.5e155，阈值 2e155。
    距离在 t=4s 处降入阈值，t=10s 处最小距离 0.5e155。
    """
    big = 1e155
    _register(client, "oh5", 4 * big)
    _upload(client, "A", [
        ("2026-09-30T00:00:00Z", 0.0, 0.0),
        ("2026-09-30T00:00:10Z", 0.0, 0.0),
    ])
    _upload(client, "B", [
        ("2026-09-30T00:00:00Z", 3 * big, 0.0),
        ("2026-09-30T00:00:10Z", 0.5 * big, 0.0),
    ])
    r = _proximity(client, "oh5", ["A", "B"], 2 * big)
    assert r.status_code == 200, r.text
    seg = r.json()["pairs"][0]["intervals"][0]
    assert seg["start"] == "2026-09-30T00:00:04Z"
    assert seg["end"] == "2026-09-30T00:00:10Z"
    assert seg["duration_sec"] == 6.0
    assert math.isclose(seg["min_distance"], 0.5 * big, rel_tol=1e-12)
    assert math.isfinite(seg["min_distance"])


def test_huge_moving_instantaneous_touch_ignored(client):
    """大坐标相对运动仅瞬时触及阈值（始终距离 >= 阈值）：不产生区间。"""
    big = 1e155
    _register(client, "oh6", 4 * big)
    _upload(client, "A", [
        ("2026-09-30T00:00:00Z", 0.0, 0.0),
        ("2026-09-30T00:00:10Z", 0.0, 0.0),
    ])
    # 距离从 3D 降到恰为 2D（t=10）再升回 3D：t=10 单点恰等
    _upload(client, "B", [
        ("2026-09-30T00:00:00Z", 3 * big, 0.0),
        ("2026-09-30T00:00:05Z", 2 * big, 0.0),
        ("2026-09-30T00:00:10Z", 3 * big, 0.0),
    ])
    r = _proximity(client, "oh6", ["A", "B"], 2 * big)
    assert r.status_code == 200, r.text
    assert r.json()["pairs"] == []


def test_huge_equal_threshold_joint_not_merged(client):
    """两段大坐标正时长相遇在采样时刻接点处恰等阈值：接点不满足，不得合并。

    B 从 0.5D 到 1.0D（t=5 恰等）再回 0.5D，得到 [0,5)、[5,10) 两段。
    """
    d = 1e155
    _register(client, "oh7", 2 * d)
    _upload(client, "A", [
        ("2026-09-30T00:00:00Z", 0.0, 0.0),
        ("2026-09-30T00:00:10Z", 0.0, 0.0),
    ])
    _upload(client, "B", [
        ("2026-09-30T00:00:00Z", 0.5 * d, 0.0),
        ("2026-09-30T00:00:05Z", 1.0 * d, 0.0),
        ("2026-09-30T00:00:10Z", 0.5 * d, 0.0),
    ])
    r = _proximity(client, "oh7", ["A", "B"], d)
    assert r.status_code == 200, r.text
    intervals = r.json()["pairs"][0]["intervals"]
    assert len(intervals) == 2
    assert intervals[0]["start"] == "2026-09-30T00:00:00Z"
    assert intervals[0]["end"] == "2026-09-30T00:00:05Z"
    assert intervals[1]["start"] == "2026-09-30T00:00:05Z"
    assert intervals[1]["end"] == "2026-09-30T00:00:10Z"
    for seg in intervals:
        assert math.isclose(seg["min_distance"], 0.5 * d, rel_tol=1e-12)


def test_huge_strictly_inside_joint_is_merged(client):
    """接点本身严格小于阈值的两段相接：合并为一段（大坐标）。"""
    d = 1e155
    _register(client, "oh8", 2 * d)
    _upload(client, "A", [
        ("2026-09-30T00:00:00Z", 0.0, 0.0),
        ("2026-09-30T00:00:10Z", 0.0, 0.0),
    ])
    # B 在 0.8D 附近折向（接点距离 0.8D < D），全程严格小于阈值
    _upload(client, "B", [
        ("2026-09-30T00:00:00Z", 0.5 * d, 0.0),
        ("2026-09-30T00:00:05Z", 0.8 * d, 0.0),
        ("2026-09-30T00:00:10Z", 0.5 * d, 0.0),
    ])
    r = _proximity(client, "oh8", ["A", "B"], d)
    assert r.status_code == 200, r.text
    intervals = r.json()["pairs"][0]["intervals"]
    assert len(intervals) == 1
    assert intervals[0]["start"] == "2026-09-30T00:00:00Z"
    assert intervals[0]["end"] == "2026-09-30T00:00:10Z"


# ---------------------------------------------------------------------
# 非法输入与既有状态码语义保持
# ---------------------------------------------------------------------

def test_huge_invalid_threshold_still_rejected(client):
    """非法阈值仍按原状态报错（服务校验 400 / 缺字段 422），不因溢出修复而改变。"""
    _register(client, "oh9", 1e200)
    _upload(client, "A", [
        ("2026-09-30T00:00:00Z", 0.0, 0.0),
        ("2026-09-30T00:00:10Z", 0.0, 0.0),
    ])
    _upload(client, "B", [
        ("2026-09-30T00:00:00Z", 1.0, 0.0),
        ("2026-09-30T00:00:10Z", 1.0, 0.0),
    ])
    for bad in (0, -1e155, "abc", "NaN", "Infinity", "-Infinity"):
        r = _proximity(client, "oh9", ["A", "B"], bad)
        assert r.status_code in (400, 422), (bad, r.status_code, r.text)
    r = client.post(
        "/api/v1/zones/oh9/proximity",
        json={
            "trip_ids": ["A", "B"],
            "since": "2026-09-30T00:00:00Z",
            "until": "2026-09-30T00:00:10Z",
        },
    )
    assert r.status_code == 422


def test_huge_unknown_trip_still_404(client):
    """未知车次仍按原状态报错（404）。"""
    big = 1e155
    _register(client, "oh10", 2 * big)
    _upload(client, "A", [
        ("2026-09-30T00:00:00Z", 0.0, 0.0),
        ("2026-09-30T00:00:10Z", 0.0, 0.0),
    ])
    r = _proximity(client, "oh10", ["A", "NOPE"], 2 * big)
    assert r.status_code == 404, r.text


def test_huge_legal_no_encounter_empty(client):
    """合法但无相遇：返回空 pairs（不报错）。"""
    big = 1e155
    _register(client, "oh11", 4 * big)
    _upload(client, "A", [
        ("2026-09-30T00:00:00Z", 0.0, 0.0),
        ("2026-09-30T00:00:10Z", 0.0, 0.0),
    ])
    _upload(client, "B", [
        ("2026-09-30T00:00:00Z", 3 * big, 0.0),
        ("2026-09-30T00:00:10Z", 3 * big, 0.0),
    ])
    r = _proximity(client, "oh11", ["A", "B"], 2 * big)
    assert r.status_code == 200
    assert r.json()["pairs"] == []
