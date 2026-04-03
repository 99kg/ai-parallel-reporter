"""
多AI并行问答工具 v3.0
同时向多个国产AI平台提问,并行获取所有回答

支持:DeepSeek / 豆包 / 腾讯元宝(混元) / 通义千问 / 文心一言 / Kimi / 智谱 / 纳米AI
"""

import asyncio
import json
import os
import sys
import io
import re
import html
import hashlib
import time
import emoji
from datetime import datetime
from openai import AsyncOpenAI
from pathlib import Path
from config import MODELS, CACHE, PDF_TEMPLATE, API_CONFIG

# PDF相关导入
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak, Image
from reportlab.lib import colors

# 解决Windows GBK编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 创建输出目录
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# 缓存目录
CACHE_DIR = Path(CACHE["cache_dir"])
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ========== 工具函数 ==========

def get_api_key(config: dict) -> str:
    """获取API Key"""
    return config.get("api_key", "")


def get_cache_key(question: str, keyword: str, model_name: str) -> str:
    """生成缓存键"""
    raw = f"{question}|{keyword}|{model_name}"
    return hashlib.md5(raw.encode('utf-8')).hexdigest()


def get_cached_result(question: str, keyword: str, model_name: str) -> tuple:
    """获取缓存结果,返回 (answer, info) 或 (None, None)"""
    if not CACHE["enabled"]:
        return None, None
    
    # allow_duplicate=False 时不使用缓存，直接返回 None
    if not CACHE.get("allow_duplicate", True):
        return None, None
    
    cache_key = get_cache_key(question, keyword, model_name)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    if not cache_file.exists():
        return None, None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查缓存是否过期
        cached_time = data.get("cached_at", 0)
        if time.time() - cached_time > CACHE["cache_ttl"]:
            return None, None
        
        return data.get("answer"), data.get("info")
    except Exception:
        return None, None


def save_to_cache(question: str, keyword: str, model_name: str, answer: str, info: dict):
    """保存结果到缓存"""
    if not CACHE["enabled"]:
        return
    
    cache_key = get_cache_key(question, keyword, model_name)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    try:
        data = {
            "question": question,
            "keyword": keyword,
            "model_name": model_name,
            "answer": answer,
            "info": info,
            "cached_at": time.time(),
        }
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"缓存保存失败: {e}")


def check_keyword_in_answer(answer: str, keyword: str) -> str:
    """检查关键词是否在回答中"""
    if not keyword or not answer:
        return "未检测"
    if keyword.lower() in answer.lower():
        return "已收录"
    return "未收录"


def print_progress(current: int, total: int, name: str, status: str = ""):
    """打印进度条"""
    bar_length = 20
    filled = int(bar_length * current / total)
    bar = "#" * filled + "-" * (bar_length - filled)
    percent = int(100 * current / total)
    
    status_text = f" [{status}]" if status else ""
    print(f"\r[{current}/{total}] {percent}% {name}{status_text:<8}{bar}", end='', flush=True)
    
    if current >= total:
        print()  # 完成后换行


# ========== 核心代码 ==========

async def ask_one_ai(name: str, config: dict, question: str, keyword: str = ""):
    """向单个AI提问"""
    # 检查缓存
    cached_answer, cached_info = get_cached_result(question, keyword, name)
    if cached_answer is not None:
        return name, cached_answer, None, cached_info, True  # 最后一个参数表示来自缓存
    
    try:
        api_key = get_api_key(config)
        if not api_key:
            return name, None, "API Key未配置", None, False
        
        # 豆包不支持timeout参数
        if name == "豆包":
            client = AsyncOpenAI(
                base_url=config["base_url"],
                api_key=api_key,
            )
        else:
            client = AsyncOpenAI(
                base_url=config["base_url"],
                api_key=api_key,
                timeout=API_CONFIG["timeout"],
            )
        response = await client.chat.completions.create(
            model=config["model"],
            messages=[{"role": "user", "content": question}],
            temperature=API_CONFIG["temperature"],
            max_tokens=API_CONFIG["max_tokens"],
        )
        answer = response.choices[0].message.content
        usage = response.usage

        status = check_keyword_in_answer(answer, keyword)
        asked_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "keyword_status": status,
            "asked_at": asked_time,  # 保存原始提问时间
        }
        
        # 保存到缓存
        save_to_cache(question, keyword, name, answer, info)
        
        return name, answer, None, info, False
    except Exception as e:
        return name, None, str(e), None, False


def get_enabled_models():
    """获取启用的模型"""
    enabled = {}
    for name, config in MODELS.items():
        api_key = get_api_key(config)
        if config.get("enabled", False) and api_key:
            enabled[name] = config
    return enabled


async def ask_all(question: str, keyword: str = ""):
    """并行向所有AI提问"""
    ready = get_enabled_models()

    if not ready:
        print("警告:没有启用任何AI！请检查MODELS配置中的enabled字段和api_key")
        print("\n可用的AI:")
        for name, config in MODELS.items():
            enabled = "启用" if config.get("enabled", False) else "禁用"
            api_key = get_api_key(config)
            status = "✓" if api_key else "✗ (未配置)"
            print(f"  [{enabled}] {name} {status}")
        return []

    total = len(ready)
    print(f"\n正在向 {total} 个AI提问中...")
    print(f"关键词监控:{keyword if keyword else '未设置'}")
    print(f"缓存功能:{'启用' if CACHE['enabled'] else '禁用'}\n")

    tasks = []
    for name, config in ready.items():
        tasks.append(ask_one_ai(name, config, question, keyword))
    
    # 逐个执行任务并显示进度
    results = []
    for i, task in enumerate(tasks, 1):
        result = await task
        results.append(result)
        if result[4]:  # 来自缓存
            status = "缓存"
        else:
            status = "完成"
        print_progress(i, total, result[0], status)

    # 格式化输出
    print("\n" + "=" * 70)
    print(f"  问题:{question}")
    if keyword:
        print(f"  关键词:{keyword}")
    print("=" * 70)

    # 统计缓存命中数量
    cached_count = sum(1 for _, _, _, _, c in results if c)
    if cached_count > 0 and CACHE["enabled"]:
        print(f"\n[缓存] 本次有 {cached_count} 个平台使用了缓存结果")
        print(f"       如需重新请求,请在 config.py 中设置 cache_ttl 为更小值或清空 cache 目录\n")

    for name, answer, error, info, from_cache in results:
        status = info.get("keyword_status", "未检测") if info else "未检测"
        token_info = f"(输入{info['prompt_tokens']}字 / 输出{info['completion_tokens']}字)" if info and info.get('prompt_tokens') else ""
        cache_tag = " [缓存]" if from_cache else ""

        print(f"\n{'─' * 60}")
        print(f"  {name} {token_info}{cache_tag}")
        print(f"  收录状态:{status}")
        print(f"{'─' * 60}")
        if error:
            print(f"  错误:{error}")
        else:
            print(f"  {answer}")

    print(f"\n{'=' * 70}")
    success = sum(1 for _, a, e, _, _ in results if a and not e)
    print(f"  共 {len(results)} 个AI响应,{success} 个成功")
    if cached_count > 0:
        print(f"  [缓存] {cached_count} 个来自缓存")
    print(f"{'=' * 70}")

    # 返回时去掉from_cache标记
    return [(n, a, e, i) for n, a, e, i, _ in results]


# ========== PDF报告生成 ==========
# 全局变量记录字体是否已注册
_font_registered = False
_chinese_font_name = None
_chinese_bold_font_name = None

def get_chinese_font():
    """注册并返回系统中可用的中文字体"""
    global _font_registered, _chinese_font_name, _chinese_bold_font_name
    
    if _font_registered:
        return _chinese_font_name, _chinese_bold_font_name
    
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import platform

    system = platform.system()

    if system == "Windows":
        # Windows中文字体
        font_options = [
            ("C:/Windows/Fonts/msyh.ttc", "msyh", "C:/Windows/Fonts/msyhbd.ttc", "msyh-Bold"),  # 微软雅黑
            ("C:/Windows/Fonts/simhei.ttf", "simhei", None, None),                              # 黑体
            ("C:/Windows/Fonts/simsun.ttc", "simsun", None, None),                              # 宋体
        ]
    elif system == "Darwin":
        font_options = [
            ("/System/Library/Fonts/STHeiti Light.ttc", "heiti", None, None),
        ]
    else:
        font_options = [
            ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", "wqy", None, None),
            ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", "droid", None, None),
        ]

    for font_path, font_name, bold_path, bold_name in font_options:
        if os.path.exists(font_path):
            try:
                font = TTFont(font_name, font_path)
                pdfmetrics.registerFont(font)
                
                # 注册加粗版本（如果提供了单独的粗体文件）
                if bold_path and os.path.exists(bold_path):
                    try:
                        bold_font = TTFont(bold_name, bold_path)
                        pdfmetrics.registerFont(bold_font)
                        _chinese_bold_font_name = bold_name
                    except:
                        _chinese_bold_font_name = font_name  # 回退到普通字体
                else:
                    try:
                        bold_font = TTFont(font_name + "-Bold", font_path)
                        pdfmetrics.registerFont(bold_font)
                        _chinese_bold_font_name = font_name + "-Bold"
                    except:
                        _chinese_bold_font_name = font_name  # 回退到普通字体
                
                _chinese_font_name = font_name
                _font_registered = True
                return font_name, _chinese_bold_font_name
            except Exception as e:
                continue

    return "Helvetica", "Helvetica-Bold"

def parse_markdown_tables(text: str) -> list:
    """解析Markdown表格,返回 [(table_data, start_pos, end_pos), ...]"""
    tables = []
    lines = text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 检测表格开始:检查是否有 | 分隔的行
        if '|' in line:
            # 收集表头
            header_cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            
            # 检查下一行是否为分隔符行
            if i + 1 < len(lines):
                separator = lines[i + 1].strip()
                # 分隔符行应该是类似 |---|---| 或 |:---|:---|:---|
                if re.match(r'^\|[\s\-:]+\|[\s\-:]+\|', separator) or separator.replace('|', '').replace('-', '').replace(':', '').strip() == '':
                    # 找到表头和分隔符,开始收集数据行
                    data_rows = []
                    j = i + 2
                    while j < len(lines):
                        row = lines[j].strip()
                        if '|' in row:
                            cells = [cell.strip() for cell in row.split('|') if cell.strip()]
                            if cells:  # 忽略空行
                                data_rows.append(cells)
                            j += 1
                        else:
                            break
                    
                    if data_rows:
                        full_table = [header_cells] + data_rows
                        tables.append((full_table, i, j - 1))
                    i = j - 1
        i += 1
    
    return tables


def clean_text_for_pdf(text: str, tables_info: list = None) -> list:
    """安全地将文本转换为ReportLab支持的格式,返回story元素列表"""
    if not text:
        return []
    
    elements = []
    lines = text.split('\n')
    
    # 表格位置集合
    table_ranges = [(start, end) for _, start, end in (tables_info or [])]
    
    for idx, line in enumerate(lines):
        # 检查这行是否在表格范围内
        in_table = any(start <= idx <= end for start, end in table_ranges)
        if in_table:
            continue
        
        line = line.strip()
        if not line:
            continue
        
        # 转义特殊字符
        line = html.escape(line)
        # 处理加粗
        line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
        # 清理Markdown标题
        line = re.sub(r'#+\s+', '', line)
        # 移除特殊符号
        line = re.sub(r'[\u2000-\u200A]', '', line)
        
        if line:
            elements.append(Paragraph(line, None))  # style会在后面统一设置
    
    return elements


def add_watermark(canvas, doc):
    """添加水印（支持文字或图片）"""
    from reportlab.platypus import Image as RLImage
    
    watermark_text = PDF_TEMPLATE.get("watermark_text")
    watermark_image = PDF_TEMPLATE.get("watermark_image")
    
    if not watermark_text and not watermark_image:
        return
    
    width, height = A4
    
    # 文字水印
    if watermark_text:
        canvas.saveState()
        canvas.setFont('Helvetica', 12)
        canvas.setFillColor(colors.HexColor('#E8E8E8'))  # 更淡的灰色
        
        for y in range(0, int(height), 60):
            for x in range(0, int(width), 180):
                canvas.saveState()
                canvas.translate(x + 60, y + 30)
                canvas.rotate(45)
                canvas.drawString(0, 0, watermark_text)
                canvas.restoreState()
        
        canvas.restoreState()
    
    # 图片水印
    if watermark_image and os.path.exists(watermark_image):
        try:
            canvas.saveState()
            
            # 加载图片并计算尺寸（水印图片尺寸的1/3）
            img = RLImage(watermark_image, width=3*cm, height=3*cm)
            img_w, img_h = 3*cm, 3*cm
            
            for y in range(0, int(height), 100):
                for x in range(0, int(width), 200):
                    canvas.saveState()
                    canvas.translate(x + 80, y + 40)
                    canvas.rotate(30)
                    # 设置透明度
                    canvas.setFillAlpha(0.15)
                    canvas.drawImage(watermark_image, -img_w/2, -img_h/2, width=img_w, height=img_h)
                    canvas.restoreState()
            
            canvas.restoreState()
        except Exception as e:
            pass


def on_page_number(canvas, doc):
    """添加页码"""
    if not PDF_TEMPLATE.get("include_page_number", True):
        return
    
    page_num = canvas.getPageNumber()
    total_pages = doc.page
    text = f"- {page_num} -"
    
    canvas.saveState()
    canvas.setFont('Helvetica', 10)
    canvas.setFillColor(colors.HexColor('#888888'))
    # 居中显示
    canvas.drawCentredString(A4[0] / 2, 1.2*cm, text)
    canvas.restoreState()


def render_markdown_content(text: str, chinese_font: str, bold_font: str = None) -> list:
    """使用 mistune 将 Markdown 转换为 ReportLab 元素列表"""
    import mistune
    import re
    
    if bold_font is None:
        bold_font = chinese_font
    
    # 移除无法渲染的 Emoji
    text = emoji.replace_emoji(text, replace="")
    
    elements = []
    
    # 创建各种样式（标题使用加粗字体）
    md_h1_style = ParagraphStyle('MDH1', fontName=bold_font, fontSize=14,
                                  spaceBefore=15, spaceAfter=8, textColor=colors.HexColor('#2D3436'))
    md_h2_style = ParagraphStyle('MDH2', fontName=bold_font, fontSize=12,
                                  spaceBefore=12, spaceAfter=6, textColor=colors.HexColor('#333333'))
    # h3-h6 使用加粗字体
    md_h3_style = ParagraphStyle('MDH3', fontName=bold_font, fontSize=11,
                                  spaceBefore=10, spaceAfter=6, textColor=colors.black, leading=14)
    md_h4_style = ParagraphStyle('MDH4', fontName=bold_font, fontSize=10,
                                  spaceBefore=8, spaceAfter=4, textColor=colors.black, leading=13)
    md_h5_style = ParagraphStyle('MDH5', fontName=bold_font, fontSize=9,
                                  spaceBefore=6, spaceAfter=3, textColor=colors.black, leading=12)
    md_h6_style = ParagraphStyle('MDH6', fontName=bold_font, fontSize=9,
                                  spaceBefore=6, spaceAfter=3, textColor=colors.black, leading=12)
    md_content_style = ParagraphStyle('MDContent', fontName=chinese_font, fontSize=9, leading=14)
    
    # 多层级列表样式（无序列表）
    ul_styles = [
        ParagraphStyle('MDList0', fontName=chinese_font, fontSize=9, leading=13,
                       leftIndent=15, firstLineIndent=-10),  # 第1层
        ParagraphStyle('MDList1', fontName=chinese_font, fontSize=9, leading=13,
                       leftIndent=30, firstLineIndent=-12),  # 第2层
        ParagraphStyle('MDList2', fontName=chinese_font, fontSize=9, leading=13,
                       leftIndent=45, firstLineIndent=-12),  # 第3层
        ParagraphStyle('MDList3', fontName=chinese_font, fontSize=9, leading=13,
                       leftIndent=60, firstLineIndent=-12),  # 第4层
    ]
    
    # 多层级列表样式（有序列表）
    ol_styles = [
        ParagraphStyle('MDOList0', fontName=chinese_font, fontSize=9, leading=13,
                       leftIndent=15, firstLineIndent=-12),  # 第1层
        ParagraphStyle('MDOList1', fontName=chinese_font, fontSize=9, leading=13,
                       leftIndent=30, firstLineIndent=-14),  # 第2层
        ParagraphStyle('MDOList2', fontName=chinese_font, fontSize=9, leading=13,
                       leftIndent=45, firstLineIndent=-14),  # 第3层
        ParagraphStyle('MDOList3', fontName=chinese_font, fontSize=9, leading=13,
                       leftIndent=60, firstLineIndent=-14),  # 第4层
    ]
    
    md_quote_style = ParagraphStyle('MDQuote', fontName=chinese_font, fontSize=9, leading=13,
                                     leftIndent=20, textColor=colors.HexColor('#666666'))
    
    # 无序列表符号
    ul_bullets = ['•', '◦', '▪', '▸']

    # 创建 Markdown 解析器
    md = mistune.create_markdown(plugins=['table', 'strikethrough'])
    
    # 将 Markdown 转换为 HTML
    html = md(text)
    
    # 辅助函数：从文本中提取纯文本（移除所有标签）
    def strip_tags(text):
        return re.sub(r'<[^>]+>', '', text).strip()
    
    # 辅助函数：提取列表项（处理嵌套结构）
    def extract_list_items(list_content: str) -> list:
        """提取列表项，处理嵌套的 ul/ol"""
        items = []
        i = 0
        while i < len(list_content):
            # 跳过空白
            if list_content[i].isspace():
                i += 1
                continue
            # 检查是否是 <li>
            if list_content[i:i+4] == '<li>':
                # 找到 </li> 位置（处理嵌套）
                depth = 1
                j = i + 4
                while j < len(list_content) and depth > 0:
                    if list_content[j:j+4] == '<li>':
                        depth += 1
                    elif list_content[j:j+5] == '</li>':
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                if depth == 0:
                    items.append(list_content[i+4:j])
                    i = j + 5
                else:
                    i += 4
            else:
                i += 1
        return items
    
    # 辅助函数：处理列表项内容
    def process_list_item(li_content: str, indent_level: int, list_type: str, item_num: int) -> list:
        """处理单个列表项，返回元素列表"""
        result = []
        li_content = li_content.strip()
        
        # 检测是否包含嵌套列表
        has_ul = '<ul>' in li_content
        has_ol = '<ol>' in li_content
        
        if has_ul:
            # 提取列表项的主文本（嵌套 ul 之前的内容）
            ul_start = li_content.find('<ul>')
            main_text = strip_tags(li_content[:ul_start]).strip()
            
            if main_text:
                # 根据父级列表类型决定前缀
                if list_type == 'ol':
                    prefix = f"{item_num}."
                else:
                    prefix = ul_bullets[min(indent_level, len(ul_bullets) - 1)]
                style = ul_styles[min(indent_level, len(ul_styles) - 1)]
                result.append(Paragraph(f"{prefix} {main_text}", style))
            
            # 提取嵌套 ul 内的所有 li 内容（使用深度追踪）
            nested_content = li_content[ul_start:]
            # 跳过第一个 <ul>，处理内部的 <li>
            inner_content = nested_content[4:]  # 跳过 '<ul>'
            
            # 使用 extract_list_items 提取嵌套的 li
            nested_lis = extract_list_items(inner_content)
            for nested_li in nested_lis:
                # 递归处理嵌套的列表项
                sub_elements = process_list_item(nested_li, indent_level + 1, 'ul', 0)
                result.extend(sub_elements)
            
        elif has_ol:
            # 提取列表项的主文本（嵌套 ol 之前的内容）
            ol_start = li_content.find('<ol>')
            main_text = strip_tags(li_content[:ol_start]).strip()
            
            if main_text:
                style = ol_styles[min(indent_level, len(ol_styles) - 1)]
                result.append(Paragraph(f"{item_num}. {main_text}", style))
            
            # 提取嵌套 ol 内的所有 li 内容
            nested_content = li_content[ol_start:]
            inner_content = nested_content[4:]  # 跳过 '<ol>'
            
            nested_lis = extract_list_items(inner_content)
            for idx, nested_li in enumerate(nested_lis, 1):
                sub_elements = process_list_item(nested_li, indent_level + 1, 'ol', idx)
                result.extend(sub_elements)
            
        else:
            # 没有嵌套的纯文本列表项
            li_text = strip_tags(li_content)
            li_text = _html_to_reportlab(li_text)
            
            if list_type == 'ol':
                style = ol_styles[min(indent_level, len(ol_styles) - 1)]
                result.append(Paragraph(f"{item_num}. {li_text}", style))
            else:
                bullet = ul_bullets[min(indent_level, len(ul_bullets) - 1)]
                style = ul_styles[min(indent_level, len(ul_styles) - 1)]
                result.append(Paragraph(f"{bullet} {li_text}", style))
        
        return result
    
    # 递归处理 HTML 内容
    def parse_html_content(html_text: str, indent_level: int = 0, list_type: str = 'ul', item_num: int = 0) -> list:
        """递归解析 HTML 内容 - 不按行分割，直接匹配"""
        result = []
        html_text = html_text.strip()
        
        while html_text:
            # 跳过前导空白
            leading_space = len(html_text) - len(html_text.lstrip())
            if leading_space > 0:
                html_text = html_text.lstrip()
                continue
            
            # 提取并处理标题
            h_match = re.match(r'<h([1-6])>(.*?)</h\1>', html_text, re.DOTALL)
            if h_match:
                h_level = int(h_match.group(1))
                content = h_match.group(2)
                content = _html_to_reportlab(content)
                # h3-h6 使用加粗效果（先清理可能存在的嵌套 b 标签）
                if h_level >= 3:
                    content = re.sub(r'<b>\s*', '', content)  # 清理开头的 b
                    content = re.sub(r'\s*</b>', '', content)  # 清理结尾的 b
                    # 移除内部的 <b></b> 对
                    while '<b></b>' in content:
                        content = content.replace('<b></b>', '')
                    # 保留嵌套的加粗内容但清理外层
                    if content.startswith('<b>') and content.endswith('</b>'):
                        # 内容已经是完整的 b 标签，不需要再包装
                        pass
                    else:
                        content = f"<b>{content}</b>"
                if h_level == 1:
                    result.append(Paragraph(content, md_h1_style))
                elif h_level == 2:
                    result.append(Paragraph(content, md_h2_style))
                elif h_level == 3:
                    result.append(Paragraph(content, md_h3_style))
                elif h_level == 4:
                    result.append(Paragraph(content, md_h4_style))
                elif h_level == 5:
                    result.append(Paragraph(content, md_h5_style))
                else:  # h6
                    result.append(Paragraph(content, md_h6_style))
                html_text = html_text[h_match.end():]
                continue
            
            # 提取并处理段落
            p_match = re.match(r'<p>(.*?)</p>', html_text, re.DOTALL)
            if p_match:
                content = p_match.group(1)
                content = _html_to_reportlab(content)
                result.append(Paragraph(content, md_content_style))
                html_text = html_text[p_match.end():]
                continue
            
            # 辅助函数：匹配嵌套的 ul/ol 标签
            def match_nested_list(html_text: str) -> tuple:
                """匹配嵌套的列表，返回 (tag, content, end_pos) 或 None"""
                if html_text.startswith('<ul>'):
                    tag = 'ul'
                    depth = 1
                    j = 4  # 跳过 '<ul>'
                    while j < len(html_text) and depth > 0:
                        if html_text[j:j+4] == '<ul>' or html_text[j:j+4] == '<ol>':
                            depth += 1
                        elif html_text[j:j+5] in ('</ul>', '</ol>'):
                            depth -= 1
                            if depth == 0:
                                return (tag, html_text[4:j], j + 5)
                        j += 1
                elif html_text.startswith('<ol>'):
                    tag = 'ol'
                    depth = 1
                    j = 4  # 跳过 '<ol>'
                    while j < len(html_text) and depth > 0:
                        if html_text[j:j+4] == '<ul>' or html_text[j:j+4] == '<ol>':
                            depth += 1
                        elif html_text[j:j+5] in ('</ul>', '</ol>'):
                            depth -= 1
                            if depth == 0:
                                return (tag, html_text[4:j], j + 5)
                        j += 1
                return None
            
            # 提取并处理无序列表
            if html_text.startswith('<ul>'):
                match = match_nested_list(html_text)
                if match:
                    tag, list_content, end_pos = match
                    li_matches = extract_list_items(list_content)
                    for li_content in li_matches:
                        sub_elements = process_list_item(li_content, indent_level, tag, 0)
                        result.extend(sub_elements)
                    html_text = html_text[end_pos:]
                    continue
            
            # 提取并处理有序列表
            if html_text.startswith('<ol>'):
                match = match_nested_list(html_text)
                if match:
                    tag, list_content, end_pos = match
                    li_matches = extract_list_items(list_content)
                    for idx, li_content in enumerate(li_matches, start=1):
                        sub_elements = process_list_item(li_content, indent_level, tag, idx)
                        result.extend(sub_elements)
                    html_text = html_text[end_pos:]
                    continue
            
            # 提取并处理引用
            blockquote_match = re.match(r'<blockquote>(.*?)</blockquote>', html_text, re.DOTALL)
            if blockquote_match:
                content = blockquote_match.group(1)
                content = _html_to_reportlab(content)
                result.append(Paragraph(content, md_quote_style))
                html_text = html_text[blockquote_match.end():]
                continue
            
            # 提取并处理代码块
            pre_match = re.match(r'<pre><code>(.*?)</code></pre>', html_text, re.DOTALL)
            if pre_match:
                content = pre_match.group(1)
                content = _html_to_reportlab(content)
                code_style = ParagraphStyle('Code', fontName='Courier', fontSize=8, leading=10)
                result.append(Paragraph(content, code_style))
                html_text = html_text[pre_match.end():]
                continue
            
            # 处理水平线
            hr_match = re.match(r'<hr\s*/?>', html_text)
            if hr_match:
                result.append(Spacer(1, 8))
                result.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#DDDDDD')))
                result.append(Spacer(1, 8))
                html_text = html_text[hr_match.end():]
                continue
            
            # 清理空白和残留标签
            cleaned = html_text.strip()
            if cleaned:
                # 清理残留标签
                cleaned = _html_to_reportlab(cleaned)
                result.append(Paragraph(cleaned, md_content_style))
            break
        
        return result
    
    # 不按行分割，直接解析整个 HTML
    html = html.strip()
    if html:
        elements = parse_html_content(html)
    
    return elements
    

def _html_to_reportlab(text: str) -> str:
    """将 HTML 标签转换为 ReportLab 支持的格式"""
    import re
    
    # 加粗: <strong> 或 <b>
    text = re.sub(r'<strong>(.*?)</strong>', r'<b>\1</b>', text)
    text = re.sub(r'<b>(.*?)</b>', r'<b>\1</b>', text)
    
    # 斜体: <em> 或 <i>
    text = re.sub(r'<em>(.*?)</em>', r'<i>\1</i>', text)
    text = re.sub(r'<i>(.*?)</i>', r'<i>\1</i>', text)
    
    # 删除线: <del> <s> <strike>
    text = re.sub(r'</?(del|s|strike)[^>]*>', '', text)
    
    # 代码: <code>
    text = re.sub(r'<code>(.*?)</code>', r'<font face="Courier">\1</font>', text)
    
    # 链接: <a href="...">text</a> -> text (url)
    text = re.sub(r'<a href="([^"]+)">([^<]+)</a>', r'\2 (\1)', text)
    
    # 清理残留标签
    text = re.sub(r'</?span[^>]*>', '', text)
    text = re.sub(r'<br\s*/?>', '<br/>', text)
    text = re.sub(r'</?div[^>]*>', '', text)
    
    return text


def add_table_to_story(text: str, content_style: ParagraphStyle, cell_style: ParagraphStyle) -> list:
    """将Markdown表格转换为PDF表格"""
    from reportlab.pdfbase import pdfmetrics
    
    # 获取当前注册的字体名
    chinese_font, bold_font = get_chinese_font()
    
    elements = []
    tables_info = parse_markdown_tables(text)
    
    if not tables_info:
        # 没有表格,直接返回文本
        clean = clean_text_for_pdf(text)
        for elem in clean:
            if elem:
                elements.append(elem)
        return elements
    
    lines = text.split('\n')
    table_ranges = [(start, end) for _, start, end in tables_info]
    
    line_idx = 0
    for table_data, start, end in tables_info:
        # 添加表格前的文本
        while line_idx < start:
            line = lines[line_idx].strip()
            if line:
                line = html.escape(line)
                line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                line = re.sub(r'#+\s+', '', line)
                if line:
                    elements.append(Paragraph(line, content_style))
            line_idx += 1
        
        # 添加表格
        if table_data:
            max_cols = max(len(row) for row in table_data)
            col_width = (17*cm) / max_cols  # A4宽度减去边距
            
            table_rows = []
            for row in table_data:
                row_cells = [Paragraph(cell, cell_style) for cell in row]
                # 填充空列
                while len(row_cells) < max_cols:
                    row_cells.append(Paragraph("", cell_style))
                table_rows.append(row_cells[:max_cols])
            
            tbl = Table(table_rows, colWidths=[col_width] * max_cols)
            
            # 样式 - 使用中文字体
            style_cmds = [
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2D3436')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), bold_font),
                ('FONTNAME', (0, 1), (-1, -1), chinese_font),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DDDDDD')),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')]),
            ]
            
            tbl.setStyle(TableStyle(style_cmds))
            elements.append(tbl)
            elements.append(Spacer(1, 0.5*cm))
        
        line_idx = end + 1
    
    # 添加表格后的文本
    while line_idx < len(lines):
        line = lines[line_idx].strip()
        if line:
            line = html.escape(line)
            line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            line = re.sub(r'#+\s+', '', line)
            if line:
                elements.append(Paragraph(line, content_style))
        line_idx += 1
    
    return elements


def generate_pdf_report(question: str, keyword: str, results: list, output_base: Path):
    """生成PDF报告 - 包含汇总表与详情页"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
        from reportlab.lib import colors
    except ImportError:
        return None
    
    chinese_font, bold_font = get_chinese_font()

    # 生成PDF文件名: AI_Analysis_Report_问题_关键词_时间戳
    safe_question = re.sub(r'[\\/:*?"<>|]', '', question)[:20]  # 移除非法字符，截取前20字
    safe_keyword = re.sub(r'[\\/:*?"<>|]', '', keyword)[:10]   # 移除非法字符，截取前10字
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_filename = f"AI_Analysis_Report_[{safe_question} - {safe_keyword}]_{timestamp}.pdf"
    pdf_path = output_base / pdf_filename

    # 使用自定义页面回调
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=2*cm, bottomMargin=2.5*cm)

    story = []

    # --- 样式定义 ---
    title_style = ParagraphStyle('Title', fontName=bold_font, fontSize=24, spaceAfter=20, 
                                 textColor=colors.HexColor('#2D3436'))
    meta_style = ParagraphStyle('Meta', fontName=chinese_font, fontSize=10, textColor=colors.HexColor('#666666'))
    # 详情页标题栏样式(用于标题背景Table)
    h2_style = ParagraphStyle('H2', fontName=bold_font, fontSize=14, 
                              textColor=colors.white, leading=18)
    # Markdown标题样式
    md_h1_style = ParagraphStyle('MDH1', fontName=bold_font, fontSize=14,
                                  spaceBefore=15, spaceAfter=8, textColor=colors.HexColor('#2D3436'))
    md_h2_style = ParagraphStyle('MDH2', fontName=bold_font, fontSize=12, 
                                  spaceBefore=12, spaceAfter=6, textColor=colors.HexColor('#333333'))
    md_h3_style = ParagraphStyle('MDH3', fontName=bold_font, fontSize=10, 
                                  spaceBefore=8, spaceAfter=4, textColor=colors.HexColor('#555555'))
    md_content_style = ParagraphStyle('MDContent', fontName=chinese_font, fontSize=9, leading=14)
    md_list_style = ParagraphStyle('MDList', fontName=chinese_font, fontSize=9, leading=13, 
                                    leftIndent=15, firstLineIndent=-10)
    content_style = ParagraphStyle('Content', fontName=chinese_font, fontSize=9, leading=16)
    cell_style = ParagraphStyle('Cell', fontName=chinese_font, fontSize=9, alignment=1, leading=12)

    # --- 第一页:汇总看板 ---
    
    # 报告标题和Logo放在同一行
    report_title = PDF_TEMPLATE.get("report_title", "AI 智能检索汇总报告")
    logo_path = PDF_TEMPLATE.get("logo_path", "")
    
    if logo_path and os.path.exists(logo_path):
        try:
            # Logo 保持正方形，不拉伸
            logo = Image(logo_path, width=2.0*cm, height=2.0*cm)
            # 使用Table将标题和Logo放在同一行
            header_table = Table(
                [[Paragraph(report_title, title_style), logo]],
                colWidths=[15*cm, 2*cm]
            )
            header_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
            ]))
            story.append(header_table)
        except:
            story.append(Paragraph(report_title, title_style))
    else:
        story.append(Paragraph(report_title, title_style))
    
    story.append(Spacer(1, 1.5*cm))
    # 报告生成日期和公司名称颜色保持一致
    meta_color = PDF_TEMPLATE.get("meta_color", "#666666")
    story.append(Paragraph(f"<font color='{meta_color}'>报告生成日期:{datetime.now().strftime('%Y-%m-%d')}</font>", meta_style))
    
    # 公司名称(如果配置了)
    company_name = PDF_TEMPLATE.get("company_name", "")
    if company_name:
        story.append(Paragraph(f"<font color='{meta_color}'>{company_name}</font>", meta_style))
    
    story.append(Spacer(1, 1.5*cm))

    # --- 汇总表格 ---
    table_data = [["序号", "关键词", "检索问题", "收录状态", "平台名称", "提问时间"]]
    col_widths = [1.2*cm, 2.5*cm, 5.5*cm, 2.5*cm, 2.8*cm, 3.0*cm]

    for idx, (name, answer, error, info) in enumerate(results, 1):
        status = info.get("keyword_status", "未检测") if info else "未检测"
        s_color = "#28a745" if status == "已收录" else ("#dc3545" if status == "未收录" else "#888888")
        status_para = Paragraph(f"<font color='{s_color}'>● {status}</font>", cell_style)
        
        # 截断问题显示
        display_question = question[:30] + "..." if len(question) > 30 else question
        
        # 使用原始提问时间（从缓存或当前时间）
        asked_time = info.get("asked_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")) if info else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        row = [
            Paragraph(str(idx), cell_style),
            Paragraph(keyword, cell_style),
            Paragraph(display_question, cell_style),
            status_para,
            Paragraph(name, cell_style),
            Paragraph(asked_time, cell_style)
        ]
        table_data.append(row)

    ts = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2D3436')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), chinese_font),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#EEEEEE')),
        ('TOPPADDING', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ])

    summary_table = Table(table_data, colWidths=col_widths)
    summary_table.setStyle(ts)
    story.append(summary_table)
    
    # 首页表格下方添加声明
    story.append(Spacer(1, 0.5*cm))
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    story.append(Paragraph(
        f"<font color='#CCCCCC'>报告编号：AI-{timestamp} | 声明：本报告由 AI PARALLEL REPORTER 系统生成，内容仅供参考。</font>", 
        cell_style
    ))

    # --- 详情页:详细回答结果 ---
    # 只处理已收录或未收录的(跳过未检测的)
    detail_results = []
    for name, answer, error, info in results:
        status = info.get("keyword_status", "未检测") if info else "未检测"
        if status == "未检测":
            continue  # 跳过未检测的
        detail_results.append((name, answer, error, info, status))
    
    for name, answer, error, info, status in detail_results:
        story.append(PageBreak())
        
        s_color = "#28a745" if status == "已收录" else "#dc3545"
        
        # 使用带背景的Table作为标题（宽度=页面宽度-左右边距）
        header_para = Paragraph(f"{name} 响应分析 <font size='11' color='{s_color}'>[{status}]</font>", h2_style)
        header_table = Table([[header_para]], colWidths=[18*cm])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#2D3436')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 0.5*cm))  # 标题和内容之间的间距
        
        if error:
            story.append(Paragraph(f"<font color='red'>错误提示:{error}</font>", content_style))
        else:
            # 使用 Markdown 渲染器处理格式
            elements = render_markdown_content(answer, chinese_font, bold_font)
            for elem in elements:
                story.append(elem)
        
        story.append(Spacer(1, 1*cm))

    # 尾页最下方添加声明(仅当有详情页时)
    if detail_results:
        story.append(Spacer(1, 1*cm))
        story.append(Paragraph(
            f"<font color='#CCCCCC'>报告编号：AI-{timestamp} | 声明：本报告由 AI PARALLEL REPORTER 系统生成，内容仅供参考。</font>", 
            cell_style
        ))

    # 构建PDF,添加水印和页码
    def on_first_page(canvas, doc):
        add_watermark(canvas, doc)
        on_page_number(canvas, doc)

    def on_later_pages(canvas, doc):
        add_watermark(canvas, doc)
        on_page_number(canvas, doc)

    doc.build(story, onFirstPage=on_first_page, onLaterPages=on_later_pages)
    
    return pdf_path


async def run_with_pdf(question: str, keyword: str = "", output_base: Path = None):
    """主控逻辑"""
    results = await ask_all(question, keyword)
    if not results:
        return None, None

    # 如果没有指定输出目录，则创建新的
    if output_base is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = OUTPUT_DIR / "report" / timestamp
        output_base.mkdir(parents=True, exist_ok=True)

    pdf_path = generate_pdf_report(question, keyword, results, output_base)
    
    return results, pdf_path


# ========== 批量处理 ==========

def load_questions_from_file(file_path: str) -> list:
    """从文件加载问题列表"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    questions = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # 跳过空行和注释行
            if not line or line.startswith('#'):
                continue
            # 解析格式:问题|关键词
            if '|' in line:
                parts = line.split('|', 1)
                question = parts[0].strip()
                keyword = parts[1].strip()
                if question and keyword:
                    questions.append({"question": question, "keyword": keyword})
                else:
                    print(f"警告:第{line_num}行格式不正确,已跳过")
            else:
                print(f"警告:第{line_num}行缺少分隔符|,已跳过")
    
    return questions


async def run_batch_questions(file_path: str):
    """批量处理问题文件"""
    print("=" * 50)
    print("批量问答模式")
    print("=" * 50)
    
    try:
        questions = load_questions_from_file(file_path)
    except FileNotFoundError as e:
        print(f"错误:{e}")
        return
    
    if not questions:
        print("错误:文件中没有找到有效的问题")
        print("\n文件格式说明:")
        print("  每行一个问题,格式:问题|关键词")
        print("  以#开头的行为注释,会被忽略")
        print("\n示例:")
        print("  # 这是注释")
        print("  有哪些4K显示器品牌推荐|DELL")
        print("  哪个中介口碑比较好|链家")
        return
    
    print(f"已加载 {len(questions)} 个问题\n")
    
    # 创建log目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = OUTPUT_DIR / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建PDF输出目录
    pdf_output = OUTPUT_DIR / "report" / timestamp
    pdf_output.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    for idx, item in enumerate(questions, 1):
        question = item["question"]
        keyword = item["keyword"]
        
        print(f"\n{'=' * 50}")
        print(f"进度:{idx}/{len(questions)}")
        print(f"{'=' * 50}")
        
        # 依次执行每个问题
        results, pdf_path = await run_with_pdf(question, keyword, pdf_output)
        
        if results:
            all_results[question] = {
                "keyword": keyword,
                "results": results,
                "pdf_path": str(pdf_path) if pdf_path else None
            }
        
    # 保存汇总JSON到log目录
    summary_path = log_dir / f"{timestamp}_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'=' * 50}")
    print("批量处理完成！")
    print(f"{'=' * 50}")
    print(f"PDF输出目录:{pdf_output}")
    print(f"汇总文件:{summary_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    # 默认读取 questions.txt
    default_file = "questions.txt"
    
    if os.path.exists(default_file):
        asyncio.run(run_batch_questions(default_file))
    else:
        print("=" * 50)
        print("错误:未找到 questions.txt 文件")
        print("=" * 50)
        print("请在同一目录下创建 questions.txt 文件")
        print()
        print("文件格式:")
        print("  问题|关键词")
        print()
        print("示例:")
        print("  有哪些4K显示器品牌推荐|DELL")