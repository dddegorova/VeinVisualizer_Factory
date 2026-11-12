"""
Выбор наиболее подходящего места для венепункции по фото руки.

Выделяет вены красным и ставит ЧЁРНУЮ ТОЧКУ на лучшую точку.
Это вспомогательный инструмент (прототип): он не видит глубину вены,
пульсацию артерий, рубцы и гематомы. Окончательный выбор места всегда
делает обученный медицинский работник (осмотр, пальпация, жгут).

"""
import argparse
import heapq
import os

import cv2
import numpy as np
from skimage.morphology import skeletonize

from vein_detect import REF_WIDTH, auto_mask, detect_veins, hand_mask_ir, overlay_red, read_image

SQ2 = 2 ** 0.5


# масштаб
def estimate_px_per_cm(mask, arm_width_cm):
    """Грубо: самая широкая часть маски ~ ширина предплечья."""
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    return float(2 * np.percentile(dt[dt > 0], 98) / arm_width_cm)


# граф
def branch_count(sk):
    """Для каждого пикселя скелета - число отходящих от него ветвей:
    1 = конец вены, 2 = обычный участок, >=3 = развилка/пересечение."""
    H, W = sk.shape
    p = np.pad(sk.astype(np.uint8), 1)

    def sh(dy, dx):
        return p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]

    ring = [sh(-1, 0), sh(-1, 1), sh(0, 1), sh(1, 1), sh(1, 0), sh(1, -1), sh(0, -1), sh(-1, -1)]
    trans = sum(((ring[i] == 0) & (ring[(i + 1) % 8] == 1)).astype(np.uint8) for i in range(8))
    return trans * sk


def prune_spurs(sk, n):
    """Убираем короткие "хвостики" скелета (шум), отщипывая концы n раз."""
    sk = sk.copy()
    for _ in range(n):
        ends = sk & (branch_count(sk) == 1)
        if not ends.any():
            break
        sk[ends] = False
    return sk


def build_graph(sk):
    ys, xs = np.nonzero(sk)
    ids = -np.ones(sk.shape, np.int32)
    ids[ys, xs] = np.arange(len(ys))
    adj = [[] for _ in range(len(ys))]
    H, W = sk.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            w = SQ2 if dy and dx else 1.0
            ny, nx = ys + dy, xs + dx
            ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            nid = np.full(len(ys), -1, np.int32)
            nid[ok] = ids[ny[ok], nx[ok]]
            for i in np.nonzero(nid >= 0)[0]:
                adj[i].append((int(nid[i]), w))
    return ys, xs, adj


def dijkstra(sources, adj, n, cutoff=np.inf):
    dist = np.full(n, np.inf)
    h = []
    for s in sources:
        dist[s] = 0.0
        heapq.heappush(h, (0.0, s))
    while h:
        d, u = heapq.heappop(h)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd <= cutoff and nd < dist[v]:
                dist[v] = nd
                heapq.heappush(h, (nd, v))
    return dist


def local_window(c, adj, radius):
    """Индексы пикселей скелета в пределах radius (вдоль вены) от точки c."""
    dist = {c: 0.0}
    h = [(0.0, c)]
    while h:
        d, u = heapq.heappop(h)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd <= radius and nd < dist.get(v, 1e18):
                dist[v] = nd
                heapq.heappush(h, (nd, v))
    return list(dist)


def rms_deviation(pts):
    """Среднеквадратичное отклонение точек от прямой (px). 0 = идеально прямо."""
    if len(pts) < 5:
        return np.inf
    q = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    perp = q @ vt[1]
    return float(np.sqrt(np.mean(perp ** 2)))


# выбор точки
def axis_offset(mask, hand_side, ys, xs):
    """
    Насколько точки смещены от продольной оси руки: 0 = ровно по центру,
    1 = на самом краю. Для каждой строки (или столбца) снимка берём середину
    и полуширину руки.
    """
    m = mask > 0
    vertical = hand_side in ("top", "bottom") or (hand_side == "none" and m.shape[0] >= m.shape[1])
    if not vertical:
        m, ys, xs = m.T, xs, ys
    has = m.any(1)
    first = m.argmax(1).astype(np.float32)
    last = (m.shape[1] - 1 - m[:, ::-1].argmax(1)).astype(np.float32)
    center = (first + last) / 2
    half = np.maximum((last - first) / 2, 1.0)
    # лёгкое сглаживание вдоль руки, чтобы ось не дёргалась
    k = max(5, m.shape[0] // 40) | 1
    center = cv2.GaussianBlur(center.reshape(-1, 1), (1, k), 0).ravel()
    half = cv2.GaussianBlur(half.reshape(-1, 1), (1, k), 0).ravel()
    off = np.abs(xs - center[ys]) / half[ys]
    off[~has[ys]] = 1.0
    return np.clip(off, 0, 1)


def wrist_distance_cm(shape, mask, hand_side, px_per_cm):
    """Расстояние (см) от каждого пикселя до края маски со стороны кисти.
    Принимаем, что именно там лучезапястный сустав / кисть."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ys, xs = np.nonzero(mask)
    if hand_side == "top":
        d = yy - ys.min()
    elif hand_side == "bottom":
        d = ys.max() - yy
    elif hand_side == "left":
        d = xx - xs.min()
    elif hand_side == "right":
        d = xs.max() - xx
    else:
        return np.full(shape, np.inf, np.float32)
    return d / px_per_cm


def find_sites(vein_mask, ridges, hand_mask, px_per_cm, hand_side="none",
               margin_cm=1.0, wrist_min_cm=2.0, wrist_ok_cm=5.0, top=1, center_weight=0.50):
    sk = skeletonize(vein_mask > 0)
    sk = prune_spurs(sk, max(3, int(0.12 * px_per_cm)))
    if sk.sum() < 20:
        return [], sk

    bc = branch_count(sk)
    ys, xs, adj = build_graph(sk)
    n = len(ys)
    bcv = bc[ys, xs]
    stops = np.nonzero((bcv == 1) | (bcv >= 3))[0]
    d_stop = dijkstra(stops, adj, n) / px_per_cm if len(stops) else np.full(n, np.inf)

    dt = cv2.distanceTransform((vein_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    wrist = wrist_distance_cm(vein_mask.shape, hand_mask > 0, hand_side, px_per_cm)

    # --- жёсткие критерии отбора кандидатов
    ok = (d_stop >= margin_cm) & (wrist[ys, xs] >= wrist_min_cm)
    cand = np.nonzero(ok)[0]
    if len(cand) == 0:
        return [], sk
    if len(cand) > 3000:
        cand = cand[::len(cand) // 3000 + 1]

    radius_px = margin_cm * px_per_cm
    rows = []
    for c in cand:
        win = local_window(c, adj, radius_px)
        wy, wx = ys[win], xs[win]
        dev_mm = rms_deviation(np.column_stack([wy, wx]).astype(np.float32)) / px_per_cm * 10
        rows.append((c,
                     2 * dt[wy, wx].mean() / px_per_cm * 10,   # ширина, мм
                     float(ridges[wy, wx].mean()),             # контрастность
                     dev_mm))
    if not rows:
        return [], sk

    idx = np.array([r[0] for r in rows])
    width = np.array([r[1] for r in rows])
    contrast = np.array([r[2] for r in rows])
    dev = np.array([r[3] for r in rows])

    # мягкие критерии -> оценка 0..1
    s_width = np.clip(width / (np.percentile(width, 95) + 1e-9), 0, 1)       # толще лучше
    s_contrast = np.clip(contrast / (np.percentile(contrast, 95) + 1e-9), 0, 1)  # виднее лучше
    s_straight = np.clip(1 - dev / 3.0, 0, 1)                                # прямее лучше
    s_run = np.clip((d_stop[idx] - margin_cm) / 2.0, 0, 1)                   # дальше от развилок
    w_cm = wrist[ys[idx], xs[idx]]
    s_wrist = np.clip((w_cm - wrist_min_cm) / (wrist_ok_cm - wrist_min_cm), 0, 1)

    off = axis_offset(hand_mask, hand_side, ys[idx], xs[idx])
    s_center = np.clip(1 - off, 0, 1)                                        # ближе к оси руки лучше

    # центральность - главный критерий; остальные делят оставшийся вес
    # в прежних пропорциях (толщина 35, видимость 25, прямота 25, развилки 10, запястье 5)
    rest = 1 - center_weight
    score = center_weight * s_center + rest * (
        0.35 * s_width + 0.25 * s_contrast + 0.25 * s_straight + 0.10 * s_run + 0.05 * s_wrist)
    # слишком извилистые участки не рассматриваем вовсе
    score[dev > 2.5] = -1

    #  берём top лучших, не ближе 1.5 см друг к другу
    order = np.argsort(-score)
    chosen = []
    for k in order:
        if score[k] <= 0:
            break
        y, x = ys[idx[k]], xs[idx[k]]
        if all(np.hypot(y - c["y"], x - c["x"]) > 1.5 * px_per_cm for c in chosen):
            chosen.append(dict(y=int(y), x=int(x), score=float(score[k]),
                               width_mm=float(width[k]), straight_dev_mm=float(dev[k]),
                               wrist_cm=float(w_cm[k]), run_cm=float(d_stop[idx[k]]),
                               axis_offset=float(off[k])))
        if len(chosen) >= top:
            break
    return chosen, sk


# рисунок
def draw_sites(img, sites, px_per_cm, label=False):
    out = img.copy()
    for i, s in enumerate(sites):
        r = max(5, int((0.15 if i == 0 else 0.10) * px_per_cm))
        color = (0, 0, 0) if i == 0 else (60, 60, 60)
        cv2.circle(out, (s["x"], s["y"]), r + 3, (255, 255, 255), -1, cv2.LINE_AA)  # белая обводка
        cv2.circle(out, (s["x"], s["y"]), r, color, -1, cv2.LINE_AA)
        if label:
            txt = f"({s['x']}, {s['y']})"
            org = (min(s["x"] + r + 10, out.shape[1] - 200), max(s["y"] - r - 6, 20))
            cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 5, cv2.LINE_AA)
            cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        elif i > 0:
            cv2.putText(out, str(i + 1), (s["x"] + r + 6, s["y"] - r),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.6, r / 14), (0, 0, 0), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default="Vein_Test2.jpg",
                    help="исходное фото (по умолчанию Vein_Test2.jpg)")
    ap.add_argument("output", nargs="?", default="result_site.jpg",
                    help="файл результата (по умолчанию result_site.jpg)")
    ap.add_argument("--hand-side", choices=["top", "bottom", "left", "right", "none"], default="none",
                    help="с какой стороны кадра кисть/запястье (там укол не ставим)")
    ap.add_argument("--px-per-cm", type=float, default=None, help="масштаб снимка")
    ap.add_argument("--arm-width-cm", type=float, default=6.0,
                    help="ширина предплечья в кадре, если масштаб не задан")
    ap.add_argument("--thresh", type=float, default=0.15)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--ir", action="store_true", help="снимок с ИК-камеры")
    ap.add_argument("--top", type=int, default=1, help="сколько вариантов показать")
    ap.add_argument("--center-weight", type=float, default=0.50,
                    help="вес близости к оси руки в оценке, 0..1 (больше -> точка ближе к центру)")
    ap.add_argument("--label", action="store_true", help="подписать координаты на картинке")
    ap.add_argument("--coords-file", default=None,
                    help="файл для координат x,y (по умолчанию рядом с картинкой, .txt)")
    ap.add_argument("--margin-cm", type=float, default=1.0,
                    help="минимальный свободный прямой участок вены в обе стороны от точки")
    args = ap.parse_args()

    bgr, (ox, oy) = read_image(args.input, return_offset=True)

    scale = args.scale or bgr.shape[1] / REF_WIDTH
    hand = hand_mask_ir(bgr, scale) if args.ir else auto_mask(bgr, scale)
    veins, ridges = detect_veins(bgr, args.thresh, mask=hand, scale=scale, return_response=True)

    px_per_cm = float(args.px_per_cm or estimate_px_per_cm(hand, args.arm_width_cm))
    print(f"Масштаб: {px_per_cm:.1f} px/см")

    sites, _ = find_sites(veins, ridges, hand, px_per_cm, args.hand_side,
                          margin_cm=args.margin_cm, top=args.top,
                          center_weight=args.center_weight)
    H, W = bgr.shape[:2]
    for s_ in sites:  # дополняем координатами в разных системах
        s_.update(
            x_px_original=s_["x"] + ox, y_px_original=s_["y"] + oy,
            x_rel=round(s_["x"] / W, 4), y_rel=round(s_["y"] / H, 4),
            x_cm=round(s_["x"] / px_per_cm, 2), y_cm=round(s_["y"] / px_per_cm, 2))

    result = overlay_red(bgr, veins)
    if sites:
        result = draw_sites(result, sites, px_per_cm, label=args.label)
        print("\nКоординаты точек (начало отсчёта - левый верхний угол, x вправо, y вниз):")
        for i, s_ in enumerate(sites, 1):
            tag = "ЛУЧШАЯ" if i == 1 else f"вариант {i}"
            print(f"  #{i} ({tag})")
            print(f"     пиксели на результате : x={s_['x']}, y={s_['y']}")
            if (ox, oy) != (0, 0):
                print(f"     пиксели на исходнике  : x={s_['x_px_original']}, y={s_['y_px_original']}"
                      f"   (поля срезаны: смещение {ox},{oy})")
            print(f"     доля кадра            : x={s_['x_rel']:.3f}, y={s_['y_rel']:.3f}  (0..1)")
            print(f"     сантиметры от угла    : x={s_['x_cm']} см, y={s_['y_cm']} см")
            print(f"     до запястья           : {s_['wrist_cm']:.1f} см;  от оси руки: {s_['axis_offset']*100:.0f}% полуширины")
            print(f"     оценка={s_['score']:.2f}, ширина вены~{s_['width_mm']:.1f} мм, "
                  f"отклонение от прямой={s_['straight_dev_mm']:.1f} мм")
    else:
        print("Подходящее место не найдено: нет достаточно длинного прямого участка вены "
              "вдали от развилок и запястья. Попробуйте другой ракурс или освещение.")

    # файл с координатами: только "x,y", по строке на точку (первая - лучшая)
    coords_path = args.coords_file or os.path.splitext(args.output)[0] + ".txt"
    with open(coords_path, "w", encoding="utf-8") as f:
        for s_ in sites:
            f.write(f"{s_['x']},{s_['y']}\n")
    print(f"Координаты сохранены: {coords_path}")
    cv2.imwrite(args.output, result)
    print(f"Готово: {args.output}")


if __name__ == "__main__":
    main()