"""
Поиск вен на фотографии руки и выделение их красным цветом.

Работает и с обычными цветными фото, и с ИК-снимками (чёрно-белыми):
вены темнее окружающей кожи, а на цветных фото ещё и менее "красные".

"""
import argparse
import os

import cv2
import numpy as np
from skimage.filters import sato

def find_file(path):
    """Ищет файл сначала по указанному пути (относительно текущей папки
    терминала), затем рядом со скриптом."""
    if os.path.isfile(path):
        return path
    alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    return alt if os.path.isfile(alt) else path


def read_image(path, return_offset=False):
    """Читает картинку; работает и с кириллицей в пути (cv2.imread на Windows - нет)."""
    real = find_file(path)
    if not os.path.isfile(real):
        raise SystemExit(
            f"Файл не найден: {path}\n"
            f"Текущая папка терминала: {os.getcwd()}\n"
            f"Папка скрипта: {os.path.dirname(os.path.abspath(__file__))}\n"
            f"Укажите полный путь к фото или перейдите в папку с фото (cd).\n"
            f"Проверьте, что имя не 'xxx.jpg.jpg' (Windows может скрывать расширение).")
    img = cv2.imdecode(np.fromfile(real, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Файл найден, но это не изображение: {real}")
    cropped, offset = crop_black_borders(img, return_offset=True)
    return (cropped, offset) if return_offset else cropped


def crop_black_borders(img, mean_thr=35, std_thr=3.0, return_offset=False):
    """Убирает ровные тёмные полосы по краям скриншота (иначе их границы дают
    ложные линии). Тёмный фон реальной ИК-съёмки не трогаем: у него есть шум
    сенсора, а у полосы скриншота колонки идеально ровные."""
    lum = img.mean(axis=2)
    keep_c = ~((lum.mean(axis=0) < mean_thr) & (lum.std(axis=0) < std_thr))
    keep_r = ~((lum.mean(axis=1) < mean_thr) & (lum.std(axis=1) < std_thr))
    cols, rows = np.where(keep_c)[0], np.where(keep_r)[0]
    if len(cols) == 0 or len(rows) == 0:
        return (img, (0, 0)) if return_offset else img
    out = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return (out, (int(cols[0]), int(rows[0]))) if return_offset else out


REF_WIDTH = 1756  # ширина, под которую подобраны размеры фильтров


def odd(x):
    x = int(round(x))
    return x if x % 2 else x + 1


def white_bg_mask(bgr, scale=1.0):
    """Маска руки для снимка на белом фоне (как Vein_Test2.jpg)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray < 235).astype(np.uint8) * 255
    k = odd(15 * scale)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return mask


def hand_mask_ir(bgr, scale=1.0):
    """
    Маска руки для снимка с камеры с ИК-подсветкой: рука рядом с камерой,
    поэтому подсвечена сильнее фона. Берём Otsu + самый большой объект.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 5 * scale)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.full_like(gray, 255)
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    m = (labels == biggest).astype(np.uint8) * 255
    # заливаем дыры внутри руки
    k = odd(31 * scale)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return m


def hand_mask_skin(bgr, scale=1.0):
    """
    Маска руки для обычного цветного фото на любом нейтральном (серый, белый,
    чёрный) фоне: кожа заметно "краснее" фона - берём канал Cr и порог Otsu.
    Дыры (синеватые вены тоже попадают в них) заливаем.
    """
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr = cv2.GaussianBlur(ycc[..., 1], (0, 0), 3 * scale)
    _, m = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.full(m.shape, 255, np.uint8)
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    m = (labels == biggest).astype(np.uint8) * 255
    k = odd(15 * scale)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # заливаем внутренние дыры
    h, w = m.shape
    pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = pad.copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    holes = cv2.bitwise_not(ff)[1:-1, 1:-1]
    m = cv2.bitwise_or(m, holes)
    return remove_shadows(bgr, m, scale)  # щели между пальцами - не вены


def auto_mask(bgr, scale=1.0):
    """Белый фон -> простая маска, иначе -> по цвету кожи."""
    h, w = bgr.shape[:2]
    corners = np.concatenate([bgr[:h // 10, :w // 10].reshape(-1, 3),
                              bgr[:h // 10, -w // 10:].reshape(-1, 3)])
    if corners.min(axis=1).mean() > 240:
        return white_bg_mask(bgr, scale)
    return hand_mask_skin(bgr, scale)


def remove_shadows(bgr, mask, scale=1.0, ratio=0.6):
    """Глубокие тени (щели между пальцами, складки) намного темнее кожи, а вена
    лишь слегка темнее. Всё, что темнее ratio*медианы, - не вена: выкидываем."""
    lum = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), 1.5 * scale)
    med = np.median(lum[mask > 0]) if (mask > 0).any() else 0
    shadow = (lum < ratio * med).astype(np.uint8) * 255
    k = odd(9 * scale)
    shadow = cv2.dilate(shadow, np.ones((k, k), np.uint8))
    m = cv2.bitwise_and(mask, cv2.bitwise_not(shadow))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((odd(5 * scale),) * 2, np.uint8))


def shrink(mask, scale=1.0):
    """Отступ от края руки: контур даёт ложные срабатывания."""
    k = odd(41 * scale)
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def vein_feature(bgr, mask, scale=1.0):
    """
    Вены на коже сине-серые и темнее окружающей кожи. Берём "недостаток красного"
    относительно локального фона (убирает влияние освещения) плюс потемнение.
    На ч/б (ИК) снимке остаётся только потемнение.
    """
    img = cv2.GaussianBlur(bgr, (0, 0), 3 * scale).astype(np.float32)
    b, g, r = cv2.split(img)

    lum = (b + g + r) / 3.0 + 1e-6
    redness = r / lum  # у цветной вены ниже; на ч/б снимке ~константа
    background = cv2.GaussianBlur(redness, (0, 0), 40 * scale)
    diff = background - redness

    lum_bg = cv2.GaussianBlur(lum, (0, 0), 40 * scale)
    dark = (lum_bg - lum) / (lum_bg + 1e-6)  # вена темнее кожи

    feat = diff * 4.0 + dark
    feat[mask == 0] = 0
    return np.clip(feat, 0, None)


def detect_veins(bgr, thresh=0.15, min_area=600, mask=None, scale=None, return_response=False):
    """Возвращает бинарную маску вен. mask - маска руки (если None - белый фон)."""
    if scale is None:
        scale = bgr.shape[1] / REF_WIDTH
    if mask is None:
        mask = auto_mask(bgr, scale)
    mask = shrink(mask, scale)

    feat = vein_feature(bgr, mask, scale)

    sigmas = [max(s * scale, 1.0) for s in (4, 6, 8, 10)]
    ridges = sato(feat, sigmas=sigmas, black_ridges=False)
    ridges[mask == 0] = 0
    ridges = ridges / (ridges.max() + 1e-9)

    binary = (ridges > thresh * ridges[mask > 0].max()).astype(np.uint8) * 255
    k = odd(7 * scale)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    clean = np.zeros_like(binary)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area * scale**2 and max(w, h) >= 80 * scale:
            clean[labels == i] = 255
    if return_response:
        return clean, ridges  # ridges: отклик фильтра линий, 0..1
    return clean


def overlay_red(bgr, vein_mask, alpha=0.85):
    scale = bgr.shape[1] / REF_WIDTH
    red = np.zeros_like(bgr)
    red[:] = (0, 0, 255)  # BGR
    m = (cv2.GaussianBlur(vein_mask, (0, 0), 1.5 * scale).astype(np.float32) / 255.0 * alpha)[..., None]
    return (bgr * (1 - m) + red * m).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default="Vein_Test2.jpg",
                    help="исходное фото (по умолчанию Vein_Test2.jpg)")
    ap.add_argument("output", nargs="?", default="result.jpg",
                    help="файл результата (по умолчанию result.jpg)")
    ap.add_argument("--thresh", type=float, default=0.15,
                    help="порог чувствительности (меньше -> больше вен), 0..1")
    ap.add_argument("--min-area", type=int, default=600,
                    help="минимальная площадь объекта в пикселях")
    ap.add_argument("--ir", action="store_true",
                    help="снимок с ИК-камеры (рука светлее тёмного фона)")
    ap.add_argument("--scale", type=float, default=None,
                    help="масштаб фильтров (по умолчанию ширина/1756); "
                         "увеличьте, если вены на фото толстые")
    ap.add_argument("--debug", action="store_true", help="сохранить маску вен отдельно")
    args = ap.parse_args()

    bgr = read_image(args.input)

    scale = args.scale or bgr.shape[1] / REF_WIDTH
    mask = hand_mask_ir(bgr, scale) if args.ir else None
    veins = detect_veins(bgr, args.thresh, args.min_area, mask, scale)
    cv2.imwrite(args.output, overlay_red(bgr, veins))
    if args.debug:
        cv2.imwrite(args.output.rsplit(".", 1)[0] + "_mask.png", veins)
    print(f"Готово: {args.output}")


if __name__ == "__main__":
    main()