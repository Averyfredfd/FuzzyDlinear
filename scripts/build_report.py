"""Build the one-page A4 interview handout (docs/项目速览.docx).

Numbers come from results/multi_summary.md (RTX 4090, seed 0, clean protocol,
official per-dataset hyper-parameters). ETTh1 / ETTm1 / ETTm2 multivariate ETT;
ETTh2 excluded because the official lr=5e-2 made the fuzzy input filter train
unstably on clean data.

Run with the bundled runtime python that has python-docx:
    <bundled python> scripts/build_report.py
"""
import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

CN = '微软雅黑'
EN = 'Segoe UI'
NAVY = RGBColor(0x17, 0x36, 0x5D)
ACCENT = RGBColor(0x1F, 0x5F, 0xA9)
GREY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0x80, 0x80, 0x80)
INK = RGBColor(0x22, 0x22, 0x22)

FIG_MULTI = 'docs/figs/fig_multi.png'
FIG_STRUCT = 'docs/figs/fig_structure.png'
OUT = 'docs/项目速览.docx'


def run_fmt(run, size=9.0, bold=False, color=None, cn=CN, en=EN):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = en
    run._element.rPr.rFonts.set(qn('w:eastAsia'), cn)
    if color is not None:
        run.font.color.rgb = color


def par(doc_or_cell, text='', size=9.0, bold=False, color=None, before=0, after=0,
        align=None, line=1.12, cn=CN, en=EN):
    p = doc_or_cell.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    if align is not None:
        pf.alignment = align
    if text:
        run_fmt(p.add_run(text), size=size, bold=bold, color=color, cn=cn, en=en)
    return p


def shade(pr, fill):
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill)
    pr.append(shd)


def borders(table, color='D0D0D0', sz=4, inside=True, val='single'):
    tblPr = table._tbl.tblPr
    el = OxmlElement('w:tblBorders')
    edges = ['top', 'left', 'bottom', 'right'] + (['insideH', 'insideV'] if inside else [])
    for edge in edges:
        e = OxmlElement('w:' + edge)
        e.set(qn('w:val'), val)
        e.set(qn('w:sz'), str(sz))
        e.set(qn('w:space'), '0')
        e.set(qn('w:color'), color)
        el.append(e)
    tblPr.append(el)


def heading(doc, text, before=6):
    p = par(doc, before=before, after=1.5)
    run_fmt(p.add_run(text), size=10, bold=True, color=NAVY)
    pPr = p._p.get_or_add_pPr()
    pbdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), 'C9D6E8')
    pbdr.append(bottom)
    pPr.append(pbdr)
    return p


def cell_text(cell, text, size=8.5, bold=False, color=None, align=WD_ALIGN_PARAGRAPH.CENTER):
    cell.text = ''
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0.5)
    p.paragraph_format.space_after = Pt(0.5)
    p.paragraph_format.line_spacing = 1.0
    p.alignment = align
    run_fmt(p.add_run(text), size=size, bold=bold, color=color)


def bullet(doc, title, body, after=1.2):
    p = par(doc, after=after, line=1.1)
    run_fmt(p.add_run('▪ ' + title + '：'), size=8.8, bold=True, color=INK)
    run_fmt(p.add_run(body), size=8.8, color=INK)
    return p


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(1.5)
    sec.top_margin = Cm(0.9)
    sec.bottom_margin = Cm(0.8)

    normal = doc.styles['Normal']
    normal.font.size = Pt(8.8)
    normal.font.name = EN
    normal.element.rPr.rFonts.set(qn('w:eastAsia'), CN)

    # ---------- header ----------
    p = par(doc, after=0)
    run_fmt(p.add_run('变压器状态抗扰预测'), size=17, bold=True, color=NAVY)
    p = par(doc, after=4)
    run_fmt(p.add_run('模糊逻辑软截断 · 轻量化边缘部署 · ETT 多数据集  ·  项目说明'), size=8, color=GREY)

    # ---------- conclusion box ----------
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders(t, color='1F5FA9', sz=6, inside=False)
    c = t.cell(0, 0)
    shade(c._tc.get_or_add_tcPr(), 'EAF2FB')
    c.text = ''
    cp = c.paragraphs[0]
    cp.paragraph_format.space_before = Pt(4)
    cp.paragraph_format.space_after = Pt(4)
    cp.paragraph_format.line_spacing = 1.12
    run_fmt(cp.add_run('在 0.68M 参数的线性预测器上，用一个'), size=9.2, color=INK)
    run_fmt(cp.add_run('零参数'), size=9.2, bold=True, color=ACCENT)
    run_fmt(cp.add_run('模糊隶属度软截断拦截遥测飞点：3 个 ETT 数据集上，5σ 脉冲下 MSE / MAE 平均'), size=9.2, color=INK)
    run_fmt(cp.add_run('降低 24% / 14%'), size=9.2, bold=True, color=ACCENT)
    run_fmt(cp.add_run('，10σ 下'), size=9.2, color=INK)
    run_fmt(cp.add_run('50% / 30%'), size=9.2, bold=True, color=ACCENT)
    run_fmt(cp.add_run('，三个数据集全部一致提升。'), size=9.2, color=INK)

    # ---------- KPI ----------
    kpi = doc.add_table(rows=2, cols=4)
    kpi.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders(kpi, val='nil', inside=False)
    kpi.autofit = False
    values = [('0.68M', '参数量（与基线持平）'), ('+24%', 'MSE 平均降幅 @ 5σ'),
              ('+50%', 'MSE 平均降幅 @ 10σ'), ('3 个', '数据集（ETT 多变量）')]
    for i, (v, label) in enumerate(values):
        cell = kpi.cell(0, i)
        cell.width = Cm(4.5)
        cell_text(cell, v, size=13.5, bold=True, color=ACCENT)
        cell2 = kpi.cell(1, i)
        cell2.width = Cm(4.5)
        cell_text(cell2, label, size=7, color=GREY)

    # ---------- 1 problem ----------
    heading(doc, '1  问题：污染是怎么被放大的', before=4)
    p = par(doc, after=1, line=1.12)
    run_fmt(p.add_run('强电磁干扰下遥测偶发飞点（单点偏离数 σ、下一点恢复）。移动平均分解会把单点飞点摊进整个窗口、'
                      '变成一段虚假趋势，线性层再把它放大到未来 96 步；而边缘设备算力有限，不能靠增加容量解决。'
                      '问题因此定义为：在 ≤1M 参数预算下，抑制输入端脉冲对多步预测的放大。'), size=8.8, color=INK)

    # ---------- 2 structure ----------
    heading(doc, '2  做法：把拦截放在污染源头')
    doc.add_picture(FIG_STRUCT, width=Cm(16.6))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.space_after = Pt(2)
    doc.paragraphs[-1].paragraph_format.space_before = Pt(1)

    bullet(doc, '软截断前置（核心）',
           '局部中位数 + MAD 求归一化偏差 d，隶属度 μ = 1/(1+(d/3)⁴)，输出 μx + (1−μ)m；连续加权无台阶，零可学习参数。')
    bullet(doc, '为什么不用均值 / 方差',
           '二者本身会被飞点污染（一个 10σ 点就能把方差拉高一个量级）；中位数与 MAD 的崩溃点是 50% 污染比例。')
    bullet(doc, '空间混合后置',
           '7×7 可学习邻接 + 门控 α（初值 0.01）做残差融合，仅 50 参数，避免图卷积的过度平滑。')

    # ---------- 3 results ----------
    heading(doc, '3  结果：干扰越强，差距越大')
    rows = [
        ('数据集', 'ΔMSE @0σ', 'ΔMSE @5σ', 'ΔMSE @10σ', 'ΔMAE @10σ'),
        ('ETTh1', '−1.0%', '+6.7%', '+22.7%', '+16.9%'),
        ('ETTm1', '−0.8%', '+20.9%', '+50.7%', '+30.3%'),
        ('ETTm2', '−2.3%', '+44.9%', '+75.3%', '+42.9%'),
        ('平均', '−1.4%', '+24.2%', '+49.6%', '+30.0%'),
    ]
    tb = doc.add_table(rows=len(rows), cols=5)
    tb.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders(tb, color='D0D0D0', sz=4)
    tb.autofit = False
    widths = [2.6, 3.0, 3.0, 3.0, 3.0]
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = tb.cell(ri, ci)
            cell.width = Cm(widths[ci])
            if ri == 0:
                shade(cell._tc.get_or_add_tcPr(), '17365D')
                cell_text(cell, val, size=7.8, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
            else:
                is_avg = (ri == len(rows) - 1)
                shade(cell._tc.get_or_add_tcPr(), 'EAF2FB' if is_avg else ('FAFAFA' if ri % 2 else 'FFFFFF'))
                cell_text(cell, val, size=8.2,
                          bold=(ci >= 1 and (val.startswith('+') or ri == len(rows) - 1)),
                          color=(ACCENT if ci >= 1 and (val.startswith('+') or ri == len(rows) - 1) else INK))

    doc.add_picture(FIG_MULTI, width=Cm(16.2))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.space_before = Pt(3)
    doc.paragraphs[-1].paragraph_format.space_after = Pt(0)

    p = par(doc, after=1, align=WD_ALIGN_PARAGRAPH.CENTER)
    run_fmt(p.add_run('三数据集全部一致：干扰越强，软截断优势越大（正值 = 相对 DLinear 的 MSE/MAE 降幅）。'),
            size=7.2, color=LIGHT)

    # ---------- 4 ablation + limits ----------
    heading(doc, '4  消融与局限', before=4)
    bullet(doc, '反直觉消融',
           '最初用可学习置信度网络（+0.114M 参数），5σ 下反而比基线差 6.4%——它只见过干净数据，学到的只是对季节项重新加权。'
           '抗噪来自机制，不来自参数。')
    bullet(doc, '如实说明',
           '干净数据平均差 1.4%（精度换鲁棒）；噪声为合成脉冲、仅验证 3 个 ETT 多变量数据集；ETTh2 因官方 lr 下软截断训练不稳而剔除；'
           '"软截断放分解前 vs 输出端"的位置消融与空间模块消融尚未完成。')

    # ---------- footer ----------
    p = par(doc, before=4, after=0, line=1.05)
    run_fmt(p.add_run('复现：'), size=7.0, bold=True, color=GREY)
    run_fmt(p.add_run('python scripts/noise_robustness.py --epochs 10 --seeds 0 --amps 0,1,3,5,10 '
                      '--protocols clean,noisy --models "DLinear,FuzzyDLinear:membership" '
                      '--datasets ETTh1,ETTm1,ETTm2'),
            size=7.0, color=GREY, cn='Consolas', en='Consolas')
    p = par(doc, before=1, after=0, line=1.05)
    run_fmt(p.add_run('口径：标准化尺度误差；单种子；基线与本文共用同一噪声种子；输入 336 步 / 预测 96 步 / 7 变量；'
                      '每数据集使用官方脚本超参；RTX 4090 上约 30 分钟。'), size=7.0, color=LIGHT)

    doc.save(OUT)
    print('saved', OUT)


if __name__ == '__main__':
    main()