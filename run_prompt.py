import os
import sys
import subprocess
import re
import time
import google.generativeai as genai
from google.api_core import exceptions
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_fixed

# --- 全局变量和配置 ---
gemini_model = None

# --- 配置参数 (请根据您的环境修改) ---
PROMPT_FILE_PATH = r"prompt.txt"
PROGRAM_A_FILENAME_TEMPLATE = "program_A_attempt_{}.py"
MAX_ATTEMPTS = 20  # 最大尝试次数

# 安全警告：强烈建议使用环境变量
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    # 警告：请替换为您的真实API密钥或设置为环境变量
    API_KEY = "(YOUR_API_KEY)"
    print("警告: 正在使用硬编码的API密钥，存在安全风险。", file=sys.stderr)

PROXY_URL = "http://127.0.0.1:7897"
REQUEST_INTERVAL_SECONDS = 5


# --- 核心功能函数 ---

def init_gemini_client():
    """初始化Gemini客户端"""
    global gemini_model
    print("\n[循环优化流程] 初始化Gemini客户端...")
    os.environ["HTTP_PROXY"] = PROMPT_FILE_PATH
    os.environ["HTTPS_PROXY"] = PROMPT_FILE_PATH
    print(f"已设置代理: {PROXY_URL}")
    try:
        genai.configure(api_key=API_KEY, transport="rest")
        model_name = "gemini-2.0-flash" 
        gemini_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"temperature": 0.5, "max_output_tokens": 8192}
        )
        print(f"Gemini客户端初始化成功，准备使用模型: {model_name}")
        return True
    except Exception as e:
        print(f"错误: 初始化客户端失败: {e}", file=sys.stderr)
        return False


def parse_prompt_file(filepath):
    """解析初始的提示文件"""
    sections = {'requirements': [], 'goals': [], 'tips': []}
    current_section = None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or (line.startswith('#') and not line.startswith(('# requirements', '# goals', '# tips'))):
                    continue
                if line.startswith('# requirements'):
                    current_section = 'requirements'
                elif line.startswith('# goals'):
                    current_section = 'goals'
                elif line.startswith('# tips'):
                    current_section = 'tips'
                elif current_section:
                    prompt_text = re.sub(r'^\d+[.\)\-]\s*', '', line).strip()
                    sections[current_section].append(prompt_text)
        if not sections['requirements']: raise ValueError("提示文件必须包含'# requirements'部分")
        return sections
    except Exception as e:
        print(f"错误: 解析提示文件失败: {e}", file=sys.stderr)
        sys.exit(1)


def extract_content(response_text):
    """从响应中提取代码和评估"""
    code_match = re.search(r"```python\s*(.*?)\s*```", response_text, re.DOTALL)
    code = code_match.group(1).strip() if code_match else ""
    conclusion_match = re.search(r'##\s*最终结论\s*:\s*(YES|NO)', response_text, re.IGNORECASE)
    conclusion = conclusion_match.group(1).upper() if conclusion_match else "NO"
    summary_match = re.search(r'##\s*评估摘要\s*:\s*(.*)', response_text, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else "未能提取评估摘要。"
    if not code:
        print("警告: 未能从模型响应中提取出Python代码块。", file=sys.stderr)
        return response_text, "NO", "未能提取代码，无法评估。"
    return code, conclusion, summary


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(15),
    retry=retry_if_exception_type(
        (exceptions.DeadlineExceeded, exceptions.ServiceUnavailable, exceptions.ResourceExhausted))
)
def call_gemini_api(prompt_text):
    """封装的API调用函数"""
    global gemini_model
    try:
        response = gemini_model.generate_content(prompt_text)
        try:
            total_used = response.usage_metadata.total_token_count
            print(f"API调用成功！实际Token使用: {total_used}")
        except AttributeError:
            print("API调用成功，但无法获取Token使用数据。")
        print(f"等待 {REQUEST_INTERVAL_SECONDS} 秒以控制请求频率...")
        time.sleep(REQUEST_INTERVAL_SECONDS)
        return extract_content(response.text)
    except exceptions.ResourceExhausted as e:
        print(f"错误: 资源配额已用尽 (429)。{e}", file=sys.stderr)
        if "requests_per_day" in str(e):
            print("每日请求配额已用尽，程序终止。", file=sys.stderr)
            sys.exit(1)
        raise
    except Exception as e:
        print(f"错误: 调用Gemini API时发生未知错误: {e}", file=sys.stderr)
        raise


def save_and_run_code(code, filename):
    """保存并运行生成的代码"""
    # 保存代码
    header = f"#!/usr/bin/env python3\n# Model: {gemini_model.model_name}\n"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(header + "\n" + code)
        print(f"程序已保存到: {os.path.abspath(filename)}")
    except Exception as e:
        return "", f"保存文件失败: {e}", False

    # 运行代码
    print(f"准备运行程序: {filename}...")
    stdout_path = f"stdout_{os.path.splitext(filename)[0]}.txt"
    stderr_path = f"stderr_{os.path.splitext(filename)[0]}.txt"
    try:
        result = subprocess.run(
            [sys.executable, filename], capture_output=True, text=True, timeout=300,
            encoding='utf-8', errors='ignore'
        )
        with open(stdout_path, 'w', encoding='utf-8') as f:
            f.write(result.stdout)
        with open(stderr_path, 'w', encoding='utf-8') as f:
            f.write(result.stderr)

        if result.stderr:
            print(f"警告: 程序运行产生错误/警告! 日志: {os.path.abspath(stderr_path)}")
        else:
            print(f"成功: 程序运行完毕。输出: {os.path.abspath(stdout_path)}")

        # 返回标准输出、标准错误和布尔型的运行成功标志
        return result.stdout, result.stderr, not bool(result.stderr.strip())

    except subprocess.TimeoutExpired:
        err = "错误: 程序运行超时（300秒）。"
        print(err, file=sys.stderr)
        return "", err, False
    except Exception as e:
        err = f"错误: 运行程序时发生意外错误: {e}"
        print(err, file=sys.stderr)
        return "", err, False


def evaluate_output(stdout_text, goals):
    """
    解析生成脚本的stdout，检查是否满足训练集和测试集的数值目标。
    """
    goal_correlation = 0.75
    train_goal_met = False
    test_goal_met = False

    # --- 改进点: 提供更明确的反馈信息 ---
    train_msg = "在程序输出中未找到'Train Correlation'。请确保代码打印的格式严格为 'Train Correlation: [数值]'。"
    test_msg = "在程序输出中未找到'Test Correlation'。请确保代码打印的格式严格为 'Test Correlation: [数值]'。"
    
    # 1. 检查训练集相关系数
    train_match = re.search(r"Train.*?Correlation.*?:?\s*(-?[\d\.]+)", stdout_text, re.IGNORECASE)
    if train_match:
        try:
            actual_correlation = float(train_match.group(1))
            print(f"评估检查: 从输出中提取的训练集相关系数为 {actual_correlation:.4f}")
            if actual_correlation >= goal_correlation:
                train_goal_met = True
                train_msg = f"训练集数值目标达成: 相关系数 {actual_correlation:.4f} >= {goal_correlation}"
            else:
                train_msg = f"训练集数值目标未达成: 相关系数为 {actual_correlation:.4f}，但目标是 >= {goal_correlation}"
        except (ValueError, IndexError):
            train_msg = "找到'Train Correlation'但无法解析其数值。"
    print(f"评估(训练集): {train_msg}")

    # 2. 检查测试集相关系数
    test_match = re.search(r"Test.*?Correlation.*?:?\s*(-?[\d\.]+)", stdout_text, re.IGNORECASE)
    if test_match:
        try:
            actual_correlation = float(test_match.group(1))
            print(f"评估检查: 从输出中提取的测试集相关系数为 {actual_correlation:.4f}")
            if actual_correlation >= goal_correlation:
                test_goal_met = True
                test_msg = f"测试集数值目标达成: 相关系数 {actual_correlation:.4f} >= {goal_correlation}"
            else:
                test_msg = f"测试集数值目标未达成: 相关系数为 {actual_correlation:.4f}，但目标是 >= {goal_correlation}"
        except (ValueError, IndexError):
            test_msg = "找到'Test Correlation'但无法解析其数值。"
    print(f"评估(测试集): {test_msg}")
    
    # 3. 综合判断并返回最终结果
    all_goals_met = train_goal_met and test_goal_met
    combined_feedback = f"训练集评估: {train_msg}\n测试集评估: {test_msg}"
    
    if all_goals_met:
        print("评估成功: 所有数值目标均已达成。")
    else:
        print(f"评估失败: {combined_feedback}")
        
    return all_goals_met, combined_feedback


# --- 主函数 ---
def main():
    """主函数，包含循环优化逻辑"""
    print("=" * 60)
    print("      Gemini 代码生成循环优化脚本      ")
    print(f"最大尝试次数: {MAX_ATTEMPTS}")
    print("=" * 60)

    if not init_gemini_client():
        sys.exit(1)

    prompt_data = parse_prompt_file(PROMPT_FILE_PATH)

    # 初始化迭代上下文
    last_code = ""
    last_feedback = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print("\n" + "-" * 25 + f" 第 {attempt} 次尝试 " + "-" * 25)

        # 1. 构建Prompt
        if attempt == 1:
            prompt_parts = [
                "You are an expert Python developer specializing in advanced signal processing...",
                "\n--- REQUIREMENTS ---\n" + "\n".join(prompt_data['requirements']),
                "\n--- GOALS ---\n" + "\n".join(prompt_data['goals']),
                "\n--- TIPS ---\n" + "\n".join(prompt_data['tips']),
                "\n--- OUTPUT FORMAT ---\nFollow this format strictly:",
                "```python\n# [Your Python code here]\n```",
                "\n## 评估摘要:\n# [Your analysis...]",
                "\n## 最终结论: YES/NO"
            ]
        else:
            prompt_parts = [
                "You are an expert Python developer tasked with fixing a script that failed to meet its goals. Analyze the previous attempt's code and the feedback from its execution, then provide a new, improved version.",
                "\n--- ORIGINAL GOALS ---\n" + "\n".join(prompt_data['goals']),
                "\n--- PREVIOUS CODE (ATTEMPT " + str(attempt - 1) + ") ---\n```python\n" + last_code + "\n```",
                "\n--- FEEDBACK FROM PREVIOUS ATTEMPT ---\n" + last_feedback,
                "\n--- INSTRUCTION ---",
                "Please provide a new, complete Python script that fixes the issues and is more likely to meet the goals. Your primary task is to ensure the script runs without errors and produces the required output in the correct format. Then, focus on improving the signal processing algorithm to meet the numerical targets.",
                "\n--- NEW OUTPUT ---",
                "```python\n# [Your new and improved Python code here]\n```",
                "\n## 评估摘要:\n# [Explain what you changed and why the new code is better...]",
                "\n## 最终结论: YES/NO"
            ]
        prompt_text = "\n".join(prompt_parts)

        # 2. 调用API
        generated_code, conclusion, summary = call_gemini_api(prompt_text)
        if not generated_code:
            print("错误: API未能返回有效代码，终止尝试。", file=sys.stderr)
            break

        print("\n--- 模型自我评估结果 ---")
        print(f"摘要: {summary}")
        print(f"结论: {conclusion}")
        print("--------------------------")

        # 3. 保存并运行代码
        filename = PROGRAM_A_FILENAME_TEMPLATE.format(attempt)
        stdout, stderr, run_success = save_and_run_code(generated_code, filename)

        # 4. 检查成功条件
        numerical_goals_met = False
        if run_success:
            numerical_goals_met, last_feedback = evaluate_output(stdout, prompt_data['goals'])
        else:
            last_feedback = f"The script failed to run. Stderr:\n{stderr}"

        if numerical_goals_met:
            print("\n" + "=" * 60)
            print("✅ 成功！代码运行无误且数值目标达成。")
            print(f"最终代码保存在: {os.path.abspath(filename)}")
            print(f"评估详情: {last_feedback}")
            print("=" * 60)
            return

        print("❌ 本次尝试失败，准备下一次迭代...")
        last_code = generated_code

    print("\n" + "=" * 60)
    print(f"❌ 失败: 经过 {MAX_ATTEMPTS} 次尝试后，仍未达成目标。")
    print("请检查最后一次尝试生成的代码和日志。")
    print(f"最后一次尝试的代码: {os.path.abspath(PROGRAM_A_FILENAME_TEMPLATE.format(attempt))}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断。")
    except Exception as e:
        print(f"\n程序因未捕获的异常而终止: {e}")

