"""
RAG 模型测试评价脚本
用于调试和评价消防问答 RAG 系统的效果

功能：
1. 加载 questions_dataset.md 中的测试用例
2. 批量测试并计算评价指标
3. 单个问题调试模式
4. 参数调整支持
"""

import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

# 导入 RAG 引擎
from rag_inference import FireSafetyRAG, RAGConfig


@dataclass
class TestCase:
    """测试用例"""
    question: str
    expected_standard: str      # 规范名称
    expected_standard_id: str   # 规范编号
    expected_articles: List[str]  # 条文编号列表


@dataclass
class TestResult:
    """测试结果"""
    question: str
    expected: str
    actual: str
    retrieved_articles: List[str]  # 检索到的条款
    matched: bool
    partial_match: bool
    retrieval_hit: bool


def load_test_cases(dataset_path: str) -> List[TestCase]:
    """从 questions_dataset.md 加载测试用例"""
    test_cases = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 解析 test_questions 和 expected_outputs
    test_match = re.search(r'test_questions = \[(.*?)\]', content, re.DOTALL)
    expected_match = re.search(r'expected_outputs = \[(.*?)\]', content, re.DOTALL)

    if not test_match or not expected_match:
        print("无法解析测试用例")
        return []

    # 提取问题列表
    questions = []
    for line in test_match.group(1).split('\n'):
        if '"' in line and 'test_questions' not in line:
            q = line.strip().replace('"', '').replace(',', '').strip()
            if q:
                questions.append(q)

    # 提取期望输出列表
    expected_outputs = []
    for line in expected_match.group(1).split('\n'):
        if '"""违反' in line:
            # 解析期望输出格式：违反：《xxx》GBxxx 第xxx条、第xxx条
            output = line.replace('"""', '').strip().rstrip(',').strip()
            expected_outputs.append(output)

    # 构建 TestCase
    for i, (q, exp) in enumerate(zip(questions, expected_outputs)):
        # 解析期望输出
        case = parse_expected_output(q, exp)
        if case:
            test_cases.append(case)

    print(f"加载了 {len(test_cases)} 个测试用例")
    return test_cases


def parse_expected_output(question: str, expected: str) -> TestCase:
    """解析期望输出字符串"""
    # 格式：违反：《xxx》GBxxx 第xxx条、第xxx条
    # 或多规范：违反：《xxx》GBxxx 第xxx条\n违反：《yyy》GByyy 第yyy条

    standards = []
    standard_ids = []
    articles = []

    # 处理可能的换行（多规范情况）
    lines = expected.replace('\\n', '\n').split('\n')

    for line in lines:
        if '违反：《' not in line:
            continue

        # 提取规范名称
        std_match = re.search(r'《([^》]+)》', line)
        if std_match:
            standards.append(std_match.group(1))

        # 提取规范编号
        gb_match = re.search(r'(GB[\d\-]+(?:[（\(][^）\)]+[）\)])?)', line)
        if gb_match:
            standard_ids.append(gb_match.group(1))

        # 提取条文编号
        article_matches = re.findall(r'第([\d\.]+)条', line)
        articles.extend(article_matches)

    if not standards:
        return None

    # 合合规范信息（简化处理，取第一个规范）
    return TestCase(
        question=question,
        expected_standard=standards[0] if standards else "",
        expected_standard_id=standard_ids[0] if standard_ids else "",
        expected_articles=articles
    )


def extract_answer_info(answer: str) -> Dict:
    """从模型回答中提取规范和条文信息"""
    result = {
        'standards': [],
        'standard_ids': [],
        'articles': [],
        'has_answer': bool(answer and answer.strip())
    }

    # 提取规范名称
    std_matches = re.findall(r'《([^》]+)》', answer)
    result['standards'] = std_matches

    # 提取规范编号（GB开头的编号，可能带括号版本）
    gb_matches = re.findall(r'GB[\d\-]+(?:[（(][^）)]+[）)])?', answer)
    result['standard_ids'] = gb_matches

    # 提取条文编号（多种格式）
    # 格式1: 第X.X条
    articles = re.findall(r'第([\d\.]+)条', answer)
    # 格式2: X.X.X 条款
    articles2 = re.findall(r'([\d]+\.[\d]+(?:\.[\d]+)?)\s*(?:条|条款|规定)', answer)
    # 格式3: 直接数字（如 12.0.2）
    articles3 = re.findall(r'[\d]+\.[\d]+(?:\.[\d]+)?', answer)

    result['articles'] = articles or articles2 or articles3

    return result


def evaluate_single(rag: FireSafetyRAG, test_case: TestCase, verbose: bool = False) -> TestResult:
    """评估单个测试用例"""
    # 运行 RAG
    result = rag.answer(test_case.question, stream=False)
    actual = result['answer']

    # 提取回答中的信息
    answer_info = extract_answer_info(actual)

    # 提取检索到的条款
    retrieved_articles = []
    for src in result.get('sources', []):
        if src.get('article_id'):
            retrieved_articles.append(src['article_id'])

    # 判断匹配情况
    # 1. 完全匹配：期望的条文都在回答中出现
    matched = False
    partial_match = False

    expected_set = set(test_case.expected_articles)
    actual_set = set(answer_info['articles'])

    if expected_set and actual_set:
        if expected_set <= actual_set:
            matched = True
        elif expected_set & actual_set:
            partial_match = True

    # 判断检索命中：期望条款是否在检索结果中
    retrieval_hit = bool(expected_set & set(retrieved_articles))

    if verbose:
        print(f"\n问题: {test_case.question}")
        print(f"期望: {test_case.expected_standard} {test_case.expected_standard_id} 第{', '.join(test_case.expected_articles)}条")
        print(f"实际回答: {actual[:200]}...")
        print(f"提取条文: {answer_info['articles']}")
        print(f"检索条款: {retrieved_articles[:5]}")
        print(f"匹配: {matched}, 部分匹配: {partial_match}, 检索命中: {retrieval_hit}")

    return TestResult(
        question=test_case.question,
        expected=f"{test_case.expected_standard} {test_case.expected_standard_id} 第{', '.join(test_case.expected_articles)}条",
        actual=actual,
        retrieved_articles=retrieved_articles,
        matched=matched,
        partial_match=partial_match,
        retrieval_hit=retrieval_hit
    )


def evaluate_batch(rag: FireSafetyRAG, test_cases: List[TestCase],
                   start: int = 0, end: int = None, verbose: bool = False) -> Dict:
    """批量评估"""
    if end is None:
        end = len(test_cases)

    results = []
    for i in range(start, min(end, len(test_cases))):
        print(f"\n测试 {i+1}/{min(end, len(test_cases))}")
        result = evaluate_single(rag, test_cases[i], verbose)
        results.append(result)

    # 计算指标
    total = len(results)
    matched_count = sum(1 for r in results if r.matched)
    partial_count = sum(1 for r in results if r.partial_match)
    retrieval_hit_count = sum(1 for r in results if r.retrieval_hit)
    has_answer_count = sum(1 for r in results if r.actual and r.actual.strip())

    metrics = {
        'total': total,
        'exact_match': matched_count,
        'exact_match_rate': matched_count / total if total > 0 else 0,
        'partial_match': partial_count,
        'partial_match_rate': partial_count / total if total > 0 else 0,
        'retrieval_hit': retrieval_hit_count,
        'retrieval_hit_rate': retrieval_hit_count / total if total > 0 else 0,
        'has_answer': has_answer_count,
        'has_answer_rate': has_answer_count / total if total > 0 else 0,
    }

    return metrics, results


def debug_single(rag: FireSafetyRAG, question: str):
    """单个问题调试模式"""
    print(f"\n{'='*60}")
    print(f"调试问题: {question}")
    print('='*60)

    # 运行 RAG
    result = rag.answer(question, stream=False)

    print(f"\n【检索结果】")
    for i, src in enumerate(result.get('sources', []), 1):
        article = src.get('article_id', '')
        score = src.get('score', 0)
        print(f"  [{i}] {src['source']} 第{article}条 (相似度: {score:.3f})")
        print(f"      {src['text'][:100]}...")

    print(f"\n【模型回答】")
    print(result['answer'])

    print(f"\n【回答分析】")
    answer_info = extract_answer_info(result['answer'])
    print(f"  规范: {answer_info['standards']}")
    print(f"  编号: {answer_info['standard_ids']}")
    print(f"  条文: {answer_info['articles']}")

    if result.get('validation'):
        print(f"\n【条款验证】")
        print(f"  有效条款: {result['validation'].get('valid_ids', [])}")
        print(f"  无效条款: {result['validation'].get('invalid_ids', [])}")
        if result['validation'].get('warning'):
            print(f"  {result['validation']['warning']}")


def interactive_mode(rag: FireSafetyRAG, test_cases: List[TestCase]):
    """交互式调试模式"""
    print("\n进入交互调试模式")
    print("命令:")
    print("  <数字> - 测试第N个用例")
    print("  <问题文本> - 直接输入问题测试")
    print("  batch <start> <end> - 批量测试")
    print("  stats - 查看统计")
    print("  quit - 退出")

    while True:
        cmd = input("\n> ").strip()

        if cmd.lower() in ('quit', 'q', 'exit'):
            break

        if cmd.lower() == 'stats':
            print(f"总用例数: {len(test_cases)}")
            continue

        if cmd.startswith('batch '):
            parts = cmd.split()
            try:
                start = int(parts[1]) if len(parts) > 1 else 0
                end = int(parts[2]) if len(parts) > 2 else len(test_cases)
                metrics, results = evaluate_batch(rag, test_cases, start, end, verbose=True)
                print_metrics(metrics)
            except:
                print("用法: batch <start> <end>")
            continue

        # 尝试解析为数字（测试用例索引）
        try:
            idx = int(cmd) - 1
            if 0 <= idx < len(test_cases):
                debug_single(rag, test_cases[idx].question)
            else:
                print(f"索引超出范围 (1-{len(test_cases)})")
        except ValueError:
            # 直接作为问题输入
            if cmd:
                debug_single(rag, cmd)


def print_metrics(metrics: Dict):
    """打印评估指标"""
    print(f"\n{'='*50}")
    print("评估结果")
    print('='*50)
    print(f"测试数量: {metrics['total']}")
    print(f"完全匹配: {metrics['exact_match']} ({metrics['exact_match_rate']*100:.1f}%)")
    print(f"部分匹配: {metrics['partial_match']} ({metrics['partial_match_rate']*100:.1f}%)")
    print(f"检索命中: {metrics['retrieval_hit']} ({metrics['retrieval_hit_rate']*100:.1f}%)")
    print(f"有回答: {metrics['has_answer']} ({metrics['has_answer_rate']*100:.1f}%)")


def save_results(results: List[TestResult], output_path: str):
    """保存测试结果"""
    data = []
    for r in results:
        data.append({
            'question': r.question,
            'expected': r.expected,
            'actual': r.actual,
            'retrieved_articles': r.retrieved_articles,
            'matched': r.matched,
            'partial_match': r.partial_match,
            'retrieval_hit': r.retrieval_hit
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='RAG 模型测试评价')
    parser.add_argument('--mode', choices=['batch', 'interactive', 'single'],
                        default='batch', help='运行模式')
    parser.add_argument('--dataset', default='rag/questions_dataset.md',
                        help='测试数据集路径')
    parser.add_argument('--start', type=int, default=0, help='批量测试起始索引')
    parser.add_argument('--end', type=int, default=None, help='批量测试结束索引')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    parser.add_argument('--output', default='rag/test_results/test_results_qwen3.5_0.8b_1.json',
                        help='结果输出路径')
    parser.add_argument('--question', type=str, help='单问题测试')

    # RAG 参数调整
    parser.add_argument('--top_k', type=int, default=10, help='检索数量')
    parser.add_argument('--min_score', type=float, default=0.3, help='最小相似度')
    parser.add_argument('--temperature', type=float, default=0.3, help='生成温度')
    parser.add_argument('--max_tokens', type=int, default=512, help='最大生成长度')

    args = parser.parse_args()

    # 加载测试用例
    test_cases = load_test_cases(args.dataset)
    if not test_cases:
        return

    # 配置 RAG
    config = RAGConfig(
        top_k=args.top_k,
        min_score=args.min_score,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens
    )

    print("\n加载 RAG 模型...")
    rag = FireSafetyRAG(config)

    # 根据模式运行
    if args.mode == 'single' and args.question:
        debug_single(rag, args.question)

    elif args.mode == 'interactive':
        interactive_mode(rag, test_cases)

    elif args.mode == 'batch':
        metrics, results = evaluate_batch(rag, test_cases, args.start, args.end, args.verbose)
        print_metrics(metrics)
        save_results(results, args.output)


if __name__ == '__main__':
    main()