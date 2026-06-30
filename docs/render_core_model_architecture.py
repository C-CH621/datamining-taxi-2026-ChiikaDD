from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).with_name("core_model_architecture.png")
W, H = 1920, 1080


def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    path = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
        if weight == "bold"
        else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    )
    if Path(path).exists():
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def centered(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, fnt, fill="#0f172a") -> None:
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2), text, font=fnt, fill=fill)


def node(draw, xy, fill, outline, title, lines, title_size=28):
    draw.rounded_rectangle(xy, radius=22, fill=fill, outline=outline, width=4)
    x1, y1, x2, _ = xy
    cx = (x1 + x2) // 2
    centered(draw, cx, y1 + 60, title, font(title_size, "bold"))
    for i, line in enumerate(lines):
        centered(draw, cx, y1 + 108 + i * 40, line, font(20), "#334155")


def arrow(draw, points, color="#2563eb", width=5):
    draw.line(points, fill=color, width=width, joint="curve")
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    if abs(x2 - x1) >= abs(y2 - y1):
        sign = 1 if x2 >= x1 else -1
        tri = [(x2, y2), (x2 - 20 * sign, y2 - 12), (x2 - 20 * sign, y2 + 12)]
    else:
        sign = 1 if y2 >= y1 else -1
        tri = [(x2, y2), (x2 - 12, y2 - 20 * sign), (x2 + 12, y2 - 20 * sign)]
    draw.polygon(tri, fill=color)


def main():
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    centered(draw, 960, 65, "最终模型方法架构图", font(46, "bold"))
    centered(draw, 960, 118, "核心是模型方法：特征表示、LGB 调参、RF 互补与加权融合", font(24), "#475569")

    node(draw, (90, 300, 420, 600), "#f0fdfa", "#0f766e", "统一特征表示", [
        "时间拆解：hour / day / month",
        "等待时长：request→pickup / scene",
        "8 类业务类别 One-Hot",
        "数值列转 float 并填补",
    ])
    node(draw, (515, 215, 845, 455), "#fff7ed", "#ea580c", "LightGBM 参数搜索", [
        "学习率 0.04",
        "叶子数 63 / 深度 7",
        "采样 + L1/L2 正则",
    ])
    node(draw, (940, 215, 1270, 455), "#ffedd5", "#c2410c", "优化 LGB 单模型", [
        "160 轮全量训练",
        "控制树容量与过拟合",
        "RMSE = 2.0229",
    ])
    node(draw, (515, 570, 845, 790), "#fef2f2", "#dc2626", "Random Forest 互补", [
        "Bagging 降低方差",
        "100 棵树",
        "max_depth = 20",
    ])
    node(draw, (1365, 335, 1715, 595), "#dcfce7", "#16a34a", "预测融合策略", [
        "LGB：梯度提升信号",
        "RF：方差互补",
        "0.5 / 0.5 加权平均",
    ])

    draw.rounded_rectangle((1365, 690, 1715, 850), radius=22, fill="#e0f2fe", outline="#0284c7", width=5)
    centered(draw, 1540, 745, "最终方法输出", font(28, "bold"))
    centered(draw, 1540, 792, "RMSE = 1.9625", font(26, "bold"), "#166534")
    centered(draw, 1540, 830, "较最佳 baseline 提升 5.6%", font(20), "#334155")

    draw.rounded_rectangle((940, 520, 1270, 670), radius=18, fill="#f8fafc", outline="#94a3b8", width=3)
    centered(draw, 1105, 570, "调参目标", font(23, "bold"))
    centered(draw, 1105, 612, "提升单模型精度", font(19), "#475569")
    centered(draw, 1105, 646, "避免过拟合", font(19), "#475569")

    draw.rounded_rectangle((90, 720, 420, 850), radius=18, fill="#f8fafc", outline="#94a3b8", width=3)
    centered(draw, 255, 765, "方法约束", font(23, "bold"))
    centered(draw, 255, 806, "同一特征表示下比较模型", font(19), "#475569")

    arrow(draw, [(420, 405), (515, 335)])
    arrow(draw, [(420, 520), (515, 680)])
    arrow(draw, [(845, 335), (940, 335)], "#ea580c")
    arrow(draw, [(1270, 335), (1315, 335), (1315, 465), (1365, 465)], "#16a34a")
    arrow(draw, [(845, 680), (1050, 680), (1190, 522), (1365, 522)], "#16a34a")
    arrow(draw, [(1540, 595), (1540, 690)], "#16a34a")
    arrow(draw, [(1105, 455), (1105, 520)], "#ea580c")
    arrow(draw, [(255, 600), (255, 720)])

    draw.rounded_rectangle((90, 925, 1715, 1003), radius=16, fill="#f8fafc", outline="#e2e8f0", width=2)
    centered(draw, 902, 974, "方法主线：显式特征表示保持可比性，LGB 调参提升单模型，RF 提供互补，最终通过简单融合获得稳定增益。", font(20), "#475569")

    img.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
