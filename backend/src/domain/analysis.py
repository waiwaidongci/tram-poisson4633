"""Fit a ridge Poisson rate model and review chronological holdout windows."""

from collections import defaultdict
from datetime import date
from math import exp, fsum, inf, isclose, isfinite, lgamma, log, log1p, ulp

from .spec import SPEC


_MAX_ROWS = 5000
_RIDGE = 1.0
_FEATURES = ((0, 0), (0, 1), (1, 0), (1, 1))


def _validate_field(field, value):
    kind = field["type"]
    label = field["label"]
    if kind in {"int", "number"}:
        types = (int,) if kind == "int" else (int, float)
        if type(value) not in types or not field["min"] <= value <= field["max"]:
            unit = "整数" if kind == "int" else "数值"
            raise ValueError(f"{label}须为 {field['min']} 至 {field['max']} 之间的{unit}")
    elif kind == "date":
        try:
            if not isinstance(value, str) or len(value) != 10 or date.fromisoformat(value).isoformat() != value:
                raise ValueError
        except ValueError:
            raise ValueError(f"{label}须为有效的 YYYY-MM-DD 日期") from None
    elif kind == "select":
        if value not in field["options"]:
            raise ValueError(f"{label}须为{'、'.join(field['options'])}之一")
    elif kind == "bool":
        if value is None:
            return False
        if not isinstance(value, bool):
            raise ValueError(f"{label}须为是或否")
        return value
    elif not isinstance(value, str) or len(value) > field["max_length"] or not value.strip():
        raise ValueError(f"{label}须为非空文本，最多 {field['max_length']} 字符")
    return value.strip() if kind == "string" else value


def validate_rows(rows):
    if not isinstance(rows, (list, tuple)) or not 1 <= len(rows) <= _MAX_ROWS:
        raise ValueError(f"观测台账须包含 1 至 {_MAX_ROWS} 行")
    readings = {}
    for index, row in enumerate(rows, start=1):
        try:
            if not isinstance(row, dict):
                raise ValueError("须为一条观测记录")
            clean = {field["key"]: _validate_field(field, row.get(field["key"])) for field in SPEC["fields"]}
            window = (clean["date"], clean["road"], clean["intersection"], clean["shift"])
            if window in readings:
                raise ValueError("日期、道路、路口与时段重复")
            readings[window] = clean
        except ValueError as error:
            raise ValueError(f"第 {index} 条观测 · {error}") from None
    return [readings[key] for key in sorted(readings)]


def _censored(bound, lam):
    """Tail moments for a capped counter reading, Y ~ Poisson(lam).

    Returns (log P(Y >= bound), q, s) where q and s are the first and second
    derivatives of the log tail probability with respect to log(lam). The tail
    probability is the regularized lower incomplete gamma P(bound, lam).
    """
    if bound <= 0:
        return 0.0, 0.0, 0.0
    log_pmf = bound * log(lam) - lam - lgamma(bound + 1)
    log_bound_pmf = log_pmf + log(bound)
    if lam < bound + 1.0:
        # Lower-tail series: P(Y >= c) = pmf(c) * sum_k lam^k / (c+1)...(c+k).
        term = total = 1.0
        for k in range(1, 100000):
            term *= lam / (bound + k)
            total += term
            if term <= total * 1e-16:
                break
        log_tail = log_pmf + log(total)
    else:
        # Continued fraction for the upper regularized incomplete gamma, so the
        # tail stays accurate when P(Y >= c) is close to one.
        tiny = 1e-300
        b = lam + 1.0 - bound
        c_frac = 1.0 / tiny
        d = 1.0 / max(b, tiny)
        h = d
        for i in range(1, 100000):
            an = -i * (i - bound)
            b += 2.0
            d = an * d + b
            if abs(d) < tiny:
                d = tiny
            c_frac = b + an / c_frac
            if abs(c_frac) < tiny:
                c_frac = tiny
            d = 1.0 / d
            delta = d * c_frac
            h *= delta
            if abs(delta - 1.0) <= 1e-15:
                break
        upper = min(exp(log_bound_pmf + log(h)), 1.0 - 1e-16)
        log_tail = log1p(-upper)
    q = exp(min(log_bound_pmf - log_tail, log(bound)))
    return log_tail, q, q * (bound - lam - q)


def _solve3(matrix, vector):
    """Solve a 3x3 linear system by Gaussian elimination with partial pivoting."""
    rows = [list(coefficients) + [value] for coefficients, value in zip(matrix, vector)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(rows[row][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        if not abs(rows[col][col]) > 0.0:
            raise ValueError("阻断模型未收敛，请核对训练期的车次和阻断次数。")
        for row in range(col + 1, 3):
            factor = rows[row][col] / rows[col][col]
            for k in range(col, 4):
                rows[row][k] -= factor * rows[col][k]
    solution = [0.0, 0.0, 0.0]
    for col in (2, 1, 0):
        remainder = fsum(rows[col][k] * solution[k] for k in range(col + 1, 3))
        solution[col] = (rows[col][3] - remainder) / rows[col][col]
    if not all(isfinite(value) for value in solution):
        raise ValueError("阻断模型未收敛，请核对训练期的车次和阻断次数。")
    return solution


def _features(row):
    return (int(row["shift"] == "高峰"), int(row["track"] == "混行"))


def _profile(slopes, groups, total):
    """Profile out the intercept by matching the fitted training count total."""
    first, second = slopes
    log_weights = [log_exposure + first * peak + second * shared for peak, shared, log_exposure, _ in groups]
    center = max(log_weights)
    centered = [weight - center for weight in log_weights]
    weights = [exp(weight) for weight in centered]
    weight_sum = fsum(weights)
    dominant = log_weights.index(center)
    log_weight_sum = log1p(fsum(weight for index, weight in enumerate(weights) if index != dominant))
    probabilities = [weight / weight_sum for weight in weights]
    # This differs from the profile likelihood by a data-only constant. Center
    # before multiplying by counts to preserve precision under separation.
    objective = fsum(count * (log_weight_sum - weight)
                     for (_, _, _, count), weight in zip(groups, centered))
    objective += 0.5 * _RIDGE * (first * first + second * second)
    return objective, probabilities, center + log_weight_sum


def _fit(training):
    exposure = defaultdict(int)
    counts = defaultdict(int)
    for row in training:
        features = _features(row)
        exposure[features] += row["exposure"]
        counts[features] += row["count"]
    total = sum(counts.values())
    if not total:
        raise ValueError("训练期没有阻断事件，无法估计有限的截距。请补充观测或调整留出观测日数。")
    groups = [(peak, shared, log(exposure[peak, shared]), counts[peak, shared])
              for peak, shared in _FEATURES if (peak, shared) in exposure]
    observed_peak = sum(peak * count for peak, _, _, count in groups)
    observed_shared = sum(shared * count for _, shared, _, count in groups)
    slopes = (0.0, 0.0)
    for _ in range(100):
        objective, probabilities, normalizer = _profile(slopes, groups, total)
        mean_peak = fsum(p * group[0] for p, group in zip(probabilities, groups))
        mean_shared = fsum(p * group[1] for p, group in zip(probabilities, groups))
        mean_joint = fsum(p * group[0] * group[1] for p, group in zip(probabilities, groups))
        g_peak = total * mean_peak - observed_peak + _RIDGE * slopes[0]
        g_shared = total * mean_shared - observed_shared + _RIDGE * slopes[1]
        h_peak = total * mean_peak * (1 - mean_peak) + _RIDGE
        h_shared = total * mean_shared * (1 - mean_shared) + _RIDGE
        h_joint = total * (mean_joint - mean_peak * mean_shared)
        determinant = h_peak * h_shared - h_joint * h_joint
        step_peak = (h_shared * g_peak - h_joint * g_shared) / determinant
        step_shared = (h_peak * g_shared - h_joint * g_peak) / determinant
        step_size = max(abs(step_peak), abs(step_shared))
        if step_size <= 1e-8 * (1 + max(map(abs, slopes))):
            return (log(total) - normalizer, *slopes), total / sum(exposure.values()), set(exposure)

        # Small Newton steps can improve the score below objective rounding error.
        # For larger steps, backtrack on the profile objective.
        scale = 1.0
        descent = g_peak * step_peak + g_shared * step_shared
        for _ in range(40):
            candidate = (slopes[0] - scale * step_peak, slopes[1] - scale * step_shared)
            candidate_objective, _, _ = _profile(candidate, groups, total)
            rounding = 4 * ulp(objective)
            if step_size <= 1e-5 or candidate_objective <= objective - 1e-4 * scale * descent + rounding:
                slopes = candidate
                break
            scale *= 0.5
        else:
            break
    raise ValueError("阻断模型未在迭代上限内收敛，请核对训练期的车次和阻断次数。")


def _fit_censored(exact, capped):
    """Fit (intercept, peak, shared) when some training windows only report a lower bound.

    exact and capped hold cleaned training rows. The likelihood uses the exact
    count where reported and P(Y >= bound) where the counter capped, with the
    same unit L2 ridge on the two slopes as _fit. Solved by Newton iteration on
    the intercept and both slopes, since censoring breaks the profile identity.
    """
    exposure = defaultdict(int)
    counts = defaultdict(int)
    for row in exact:
        features = _features(row)
        exposure[features] += row["exposure"]
        counts[features] += row["count"]
    exact_groups = [(peak, shared, log(exposure[peak, shared]), counts[peak, shared])
                    for peak, shared in _FEATURES if (peak, shared) in exposure]
    capped_windows = [(*_features(row), log(row["exposure"]), row["count"]) for row in capped]

    def objective(beta):
        intercept, peak, shared = beta
        value = 0.5 * _RIDGE * (peak * peak + shared * shared)
        for is_peak, is_shared, log_exposure, count in exact_groups:
            mu = log_exposure + intercept + peak * is_peak + shared * is_shared
            if mu > 350.0:
                return inf
            value += exp(mu) - count * (intercept + peak * is_peak + shared * is_shared)
        for is_peak, is_shared, log_exposure, bound in capped_windows:
            mu = log_exposure + intercept + peak * is_peak + shared * is_shared
            if mu > 350.0:
                return inf
            log_tail, _, _ = _censored(bound, exp(mu))
            value -= log_tail
        return value

    def gradient_hessian(beta):
        intercept, peak, shared = beta
        gradient = [0.0, 0.0, 0.0]
        hessian = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        for is_peak, is_shared, log_exposure, count in exact_groups:
            basis = (1.0, float(is_peak), float(is_shared))
            lam = exp(log_exposure + intercept + peak * is_peak + shared * is_shared)
            for i in range(3):
                gradient[i] += (lam - count) * basis[i]
                for j in range(i, 3):
                    hessian[i][j] += lam * basis[i] * basis[j]
        for is_peak, is_shared, log_exposure, bound in capped_windows:
            basis = (1.0, float(is_peak), float(is_shared))
            _, q, s = _censored(bound, exp(log_exposure + intercept + peak * is_peak + shared * is_shared))
            for i in range(3):
                gradient[i] -= q * basis[i]
                for j in range(i, 3):
                    hessian[i][j] -= s * basis[i] * basis[j]
        gradient[1] += _RIDGE * peak
        gradient[2] += _RIDGE * shared
        hessian[1][1] += _RIDGE
        hessian[2][2] += _RIDGE
        hessian[1][0], hessian[2][0], hessian[2][1] = hessian[0][1], hessian[0][2], hessian[1][2]
        return gradient, hessian

    total_exposure = fsum(exp(log_exposure) for _, _, log_exposure, _ in exact_groups)
    total_count = sum(count for _, _, _, count in exact_groups)
    beta = (log((total_count + 0.5) / total_exposure), 0.0, 0.0)
    for _ in range(100):
        value = objective(beta)
        gradient, hessian = gradient_hessian(beta)
        step = _solve3(hessian, gradient)
        if max(map(abs, step)) <= 1e-8 * (1 + max(map(abs, beta))):
            return beta
        descent = fsum(derivative * move for derivative, move in zip(gradient, step))
        scale = 1.0
        for _ in range(40):
            candidate = tuple(current - scale * move for current, move in zip(beta, step))
            if objective(candidate) <= value - 1e-4 * scale * descent + 4 * ulp(value):
                beta = candidate
                break
            scale *= 0.5
        else:
            break
    raise ValueError("阻断模型未在迭代上限内收敛，请核对训练期的车次和阻断次数。")


def _rate_censored(exact, capped):
    """Single blocking rate per exposure from exact and lower-bound training rows."""
    exact_windows = [(log(row["exposure"]), row["count"]) for row in exact]
    capped_windows = [(log(row["exposure"]), row["count"]) for row in capped]
    total_exposure = fsum(exp(log_exposure) for log_exposure, _ in exact_windows)
    total_count = sum(count for _, count in exact_windows)

    def objective(beta):
        value = 0.0
        for log_exposure, count in exact_windows:
            if log_exposure + beta > 350.0:
                return inf
            value += exp(log_exposure + beta) - count * beta
        for log_exposure, bound in capped_windows:
            if log_exposure + beta > 350.0:
                return inf
            log_tail, _, _ = _censored(bound, exp(log_exposure + beta))
            value -= log_tail
        return value

    beta = log((total_count + 0.5) / total_exposure)
    for _ in range(100):
        gradient = hessian = 0.0
        for log_exposure, count in exact_windows:
            lam = exp(log_exposure + beta)
            gradient += lam - count
            hessian += lam
        for log_exposure, bound in capped_windows:
            _, q, s = _censored(bound, exp(log_exposure + beta))
            gradient -= q
            hessian -= s
        step = gradient / hessian
        if abs(step) <= 1e-10 * (1 + abs(beta)):
            return exp(beta)
        value = objective(beta)
        scale = 1.0
        for _ in range(40):
            candidate = beta - scale * step
            if objective(candidate) <= value - 1e-4 * scale * gradient * step + 4 * ulp(value):
                beta = candidate
                break
            scale *= 0.5
        else:
            break
    raise ValueError("基线阻断率未在迭代上限内收敛，请核对训练期的车次和阻断次数。")


def _deviance(count, expected):
    if count == 0:
        return 2 * expected
    return max(0.0, 2 * (count * (log(count) - log(expected)) - count + expected))


def _deviance_bound(bound, expected):
    """Deviance of a capped window, scored by the probability of reaching the lower bound."""
    log_tail, _, _ = _censored(bound, expected)
    return -2.0 * log_tail


def _planning(windows, keys, threshold):
    groups = defaultdict(list)
    for row in windows:
        groups[tuple(row[key] for key in keys)].append(row)
    plan = []
    for group, records in groups.items():
        exposure = sum(row["exposure"] for row in records)
        expected = fsum(row["predicted"] for row in records)
        rate = 100 * expected / exposure
        severity = sum(rate >= limit or isclose(rate, limit, rel_tol=1e-12)
                       for limit in (threshold, 2 * threshold))
        plan.append({
            **dict(zip(keys, group)), "exposure": exposure, "predicted": expected,
            "rate": rate, "severity": ("常规", "重点", "优先")[severity],
            "action": ("按常规频次抽查", "增排路口巡查", "优先复核放行相位与混行冲突")[severity],
        })
    return sorted(plan, key=lambda row: (-row["predicted"], -row["rate"], *(row[key] for key in keys)))


def _table(title, columns, rows):
    return {"title": title, "columns": [{"key": key, "label": label} for key, label in columns], "rows": rows}


def analyze(rows, normalized_params):
    readings = validate_rows(rows)
    if set(normalized_params) - {field["key"] for field in SPEC["params"]}:
        raise ValueError("存在未知运行参数")
    params = {field["key"]: _validate_field(field, normalized_params.get(field["key"], field["default"]))
              for field in SPEC["params"]}
    # Reconcile capped counters against the signal station: matching records are
    # kept (exact count or lower bound), mismatched windows are set aside.
    reconciled = [row for row in readings if row["capped"] == row["signal_capped"]]
    aside = [row for row in readings if row["capped"] != row["signal_capped"]]
    if not reconciled:
        raise ValueError("所有窗口的计数器封顶记录都未与信号台对上，没有可计算的窗口。")
    days = sorted({row["date"] for row in reconciled})
    holdout_days = params["holdout_days"]
    if len(days) <= holdout_days:
        raise ValueError("留出观测日数须少于台账中的不同日期数，以保留训练数据。")
    cutoff = days[-holdout_days]
    training = [row for row in reconciled if row["date"] < cutoff]
    exact = [row for row in training if not row["capped"]]
    capped = [row for row in training if row["capped"]]
    if capped:
        if not exact:
            raise ValueError("训练窗口全部封顶，无法估计有限的阻断率，本次停排。"
                             "请等待信号台更正记录或补充确数窗口后重算。")
        if not any(row["count"] for row in training):
            raise ValueError("训练期没有阻断事件，无法估计有限的截距。请补充观测或调整留出观测日数。")
        coefficients = _fit_censored(exact, capped)
        baseline_rate = _rate_censored(exact, capped)
        seen = {_features(row) for row in training}
    else:
        coefficients, baseline_rate, seen = _fit(training)
    intercept, peak_coefficient, shared_coefficient = coefficients
    windows = []
    for row in reconciled[len(training):]:
        peak, shared = _features(row)
        log_mean = log(row["exposure"]) + intercept + peak_coefficient * peak + shared_coefficient * shared
        expected = exp(log_mean)
        baseline = row["exposure"] * baseline_rate
        if row["capped"]:
            model_deviance = _deviance_bound(row["count"], expected)
            baseline_deviance = _deviance_bound(row["count"], baseline)
        else:
            model_deviance = _deviance(row["count"], expected)
            baseline_deviance = _deviance(row["count"], baseline)
        windows.append({
            **row, "predicted": expected, "baseline": baseline, "unseen": (peak, shared) not in seen,
            "censored": bool(row["capped"]),
            "model_deviance": model_deviance,
            "baseline_deviance": baseline_deviance,
        })
    mean_model = fsum(row["model_deviance"] for row in windows) / len(windows)
    mean_baseline = fsum(row["baseline_deviance"] for row in windows) / len(windows)
    better = mean_model < mean_baseline and not isclose(mean_model, mean_baseline, rel_tol=1e-12, abs_tol=1e-12)
    unseen_count = sum(row["unseen"] for row in windows)
    bounded_training = len(capped)
    bounded_holdout = sum(row["censored"] for row in windows)
    threshold = params["severity_rate"]
    roads = _planning(windows, ("road",), threshold)
    intersections = _planning(windows, ("road", "intersection"), threshold)
    focus = intersections[0]
    plan_columns = (("exposure", "情景车次"), ("predicted", "预计阻断次数"), ("rate", "预计次／百车次"),
                    ("severity", "巡查级别"), ("action", "排班动作"))
    coefficient_rows = [
        {"term": term, "coefficient": coefficient, "multiplier": exp(coefficient)}
        for term, coefficient in zip(("截距（平峰、独立通行）", "高峰（对比平峰）", "混行（对比独立通行）"), coefficients)
    ]
    note = (f"训练 {days[0]} 至 {training[-1]['date']}，留出 {cutoff} 至 {days[-1]}。"
            f"重点阈值 {threshold:g} 次／百车次，优先阈值 {2 * threshold:g} 次／百车次。")
    if aside:
        note += f"另有 {len(aside)} 个窗口的封顶记录未与信号台对上，已搁置不计。"
    comparison = "GLM 留出偏差低于基线" if better else "GLM 留出偏差未低于基线"
    field_columns = tuple((field["key"], field["label"]) for field in SPEC["fields"])
    tables = [
        _table("道路巡查计划", (("road", "道路"),) + plan_columns, roads),
        _table("路口巡查计划", (("road", "道路"), ("intersection", "路口")) + plan_columns, intersections),
        _table("模型系数与发生率倍数", (("term", "模型项"), ("coefficient", "系数"),
                 ("multiplier", "每车次基准率或发生率倍数")), coefficient_rows),
        _table("留出窗口核对", field_columns + (
            ("predicted", "GLM 预计次数"), ("baseline", "基线预计次数"), ("model_deviance", "GLM 偏差"),
            ("baseline_deviance", "基线偏差"), ("censored", "按下限计分"), ("unseen", "未见条件组合"),
        ), windows),
    ]
    if aside:
        tables.append(_table("搁置窗口（封顶记录未与信号台对上）", field_columns, aside))
    return {
        "state": "ok" if better and not unseen_count and not aside else "warning",
        "headline": f"{comparison}，{focus['road']} · {focus['intersection']}预计工作量最高。",
        "note": note,
        "metrics": [
            {"label": "训练窗口", "value": len(training),
             "detail": f"{len(days) - holdout_days} 个观测日 · {bounded_training} 个按下限计"},
            {"label": "留出窗口", "value": len(windows),
             "detail": f"{holdout_days} 个观测日 · {unseen_count} 个窗口的条件组合未见于训练 · {bounded_holdout} 个按下限计分"},
            {"label": "搁置窗口", "value": len(aside), "detail": "计数器封顶记录与信号台未对上，未参与训练与评分"},
            {"label": "GLM 平均偏差", "value": mean_model, "detail": "留出期每窗口偏差，下限窗口按至少达到下限的概率折算"},
            {"label": "基线平均偏差", "value": mean_baseline, "detail": f"训练基线 {100 * baseline_rate:.4g} 次／百车次"},
            {"label": "偏差减少量", "value": mean_baseline - mean_model, "detail": "基线平均偏差减去 GLM 平均偏差"},
        ],
        "charts": [
            {"title": "留出窗口平均 Poisson 偏差", "kind": "bar", "series": [
                {"label": "GLM", "value": mean_model}, {"label": "单一阻断率", "value": mean_baseline},
            ]},
            {"title": "预计阻断次数最高的路口（前 10）", "kind": "bar", "series": [
                {"label": f"{row['road']} · {row['intersection']}", "value": row["predicted"]}
                for row in intersections[:10]
            ]},
        ],
        "tables": tables,
        "sections": [
            {"title": "系数口径", "text": "截距的指数为平峰、独立通行条件下的每车次发生率。"
             "其余系数取指数后表示另一项条件相同时的发生率倍数。过车车次翻倍时，预计次数也翻倍。"},
            {"title": "排班口径", "text": "巡查表复用留出期的过车车次和条件，按预计阻断次数排序。"
             "路段和路口各自按每百车次的预计发生率定级。复核时请对照留出偏差和未见条件组合。"},
            {"title": "封顶与更正口径", "text": "计数器一封顶，报上的阻断数就是下限。与信号台封顶记录对上的窗口按下限参与训练和留出评分："
             "确数窗口用 Poisson 偏差，下限窗口按至少达到下限的概率折算，两种模型始终共用同一批窗口。"
             "未对上的窗口先搁置，不计入任何分组。训练窗口全部封顶、估不出有限阻断率时停排。"
             "更正批次入库后，旧运行的比较即撤下，须用更正后的批次重算全部留出窗再排班；没有封顶字段的旧批次照用确数。"},
        ],
    }
