"""
使用 Gemini API 為 score1112.json 中的樣本進行評分
使用最新的 Gemini API SDK (Client-based)

安裝套件: pip install google-genai tqdm
"""
import os
import json
import csv
import time
from tqdm import tqdm
from google import genai

# --- Configuration ---
STAGE_DIR = "./CONCH/who4e/for_demo"
INPUT_FILE = f"{os.path.expandvars(STAGE_DIR)}/score_1112.json" 
OUTPUT_FILE = f"{os.path.expandvars(STAGE_DIR)}/gemini_scored_results.csv"

API_KEY = "AIzaSyD0eO9Mn5o0nBKCluYcc8Tf3TuxJ2lkm6g" # 雅安的apikey


# --- Instructions for Gemini ---
INSTRUCTION_TEMPLATE = """Role: You are an expert ophthalmologist reviewing a generated pathology eye image caption.
Task: Compare the Model Output Caption against the Ground Truth Caption. Evaluate the Model Output based on its Accuracy (no medical errors or hallucinations) and Completeness (captures all major pathological features).

Input:
1. Ground Truth Caption: {ground_truth}
2. Model Output Caption: {model_output}

Instructions:
1. Score: Provide a single integer score from 1 to 5, where 5 is perfect (medically indistinguishable from the ground truth) and 1 is critically flawed or dangerously inaccurate.
2. Critique: Summarize the main reasons for the score. State any critical omissions or hallucinations found.
3. Word Limit: Limit the entire response (including the score and critique) to approximately 100 words.
4. Output Format: Provide your response strictly in the following JSON format:

{{
   "overall_score": [1-5 integer],
   "critique": "Concise summary of accuracy/completeness issues and the justification for the score."
}}"""

def call_gemini_api(client, ground_truth, model_output, max_retries=3):
    """調用 Gemini API 進行評分（使用新的 Client API）"""
    prompt = INSTRUCTION_TEMPLATE.format(
        ground_truth=ground_truth,
        model_output=model_output
    )
    
    for attempt in range(max_retries):
        try:
            # 使用新的 Client API
            # 模型選擇：
            # - gemini-2.5-flash: 較快、較便宜，適合大量請求
            # - gemini-2.5-pro: 較慢、較貴，但更準確，適合複雜評分任務
            response = client.models.generate_content(
                model="gemini-2.5-flash",  # 已切換為 Pro 模型（更準確但較慢較貴）
                contents=prompt
            )
            
            # 提取 JSON 部分（可能包含 markdown 代碼塊）
            response_text = response.text.strip()
            
            # 移除可能的 markdown 代碼塊標記
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # 解析 JSON
            result = json.loads(response_text)
            
            # 驗證格式
            if "overall_score" not in result or "critique" not in result:
                raise ValueError("Invalid response format")
            
            return result
            
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"  JSON decode error, retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(1)
            else:
                print(f"  Failed to parse JSON after {max_retries} attempts: {e}")
                print(f"  Response text: {response_text[:200]}")
                return {"overall_score": 0, "critique": f"Error parsing response: {str(e)}"}
        
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Error, retrying... (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return {"overall_score": 0, "critique": f"Error: {str(e)}"}
    
    return {"overall_score": 0, "critique": "Failed to get response"}

def main():
    # 初始化 Gemini Client
    try:
        # 直接使用代碼中定義的 API_KEY 變數
        if not API_KEY:
            print("Error: API_KEY not set in the code.")
            print("Please set API_KEY variable at the top of the file.")
            return
        
        # 使用新的 Client API
        client = genai.Client(api_key=API_KEY)
        print("✅ Gemini Client initialized successfully.\n")
        
    except Exception as e:
        print(f"Error initializing Gemini Client: {e}")
        print("Make sure you have installed: pip install google-genai")
        print("Note: The package name is 'google-genai', not 'google-generativeai'")
        return
    
    # 讀取輸入文件
    print(f"Loading scores from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} samples to score.\n")
    
    # 處理每個樣本
    results = []
    
    for item in tqdm(data, desc="Scoring with Gemini"):
        idx = item.get("idx", "")
        number = item.get("number", "")
        ground_truth = item.get("gts", "")
        model_output = item.get("res", "")
        
        # 獲取現有的分數
        scores = item.get("score", {})
        bleu1 = scores.get("BLEU1", 0.0)
        bleu4 = scores.get("BLEU4", 0.0)
        rouge = scores.get("ROUGE", 0.0)
        cider_r = scores.get("CIDEr-R", 0.0)
        meteor = scores.get("METEOR", 0.0)
        
        # 調用 Gemini API
        print(f"\nProcessing idx={idx}, number={number}...")
        gemini_result = call_gemini_api(client, ground_truth, model_output)
        
        # 構建結果
        result = {
            "id": idx,
            "number": number,
            "ground_truth": ground_truth,
            "res": model_output,
            "gemini_scoring": gemini_result.get("overall_score", 0),
            "gemini_critique": gemini_result.get("critique", ""),
            "BLEU1": bleu1,
            "BLEU4": bleu4,
            "ROUGE": rouge,
            "CIDEr-R": cider_r,
            "METEOR": meteor
        }
        
        results.append(result)
        
        # 添加延遲以避免 API 限制
        time.sleep(1)
    
    # 保存結果為 CSV
    print(f"\nSaving results to {OUTPUT_FILE}...")
    
    # 定義 CSV 欄位順序（gemini_scoring 和 gemini_critique 放在 METEOR 後面）
    fieldnames = ["id", "number", "ground_truth", "res", "BLEU1", "BLEU4", "ROUGE", "CIDEr-R", "METEOR", "gemini_scoring", "gemini_critique"]
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        for result in results:
            # 準備 CSV 行數據
            row = {
                "id": result.get("id", ""),
                "number": result.get("number", ""),
                "ground_truth": result.get("ground_truth", ""),
                "res": result.get("res", ""),
                "BLEU1": result.get("BLEU1", 0.0),
                "BLEU4": result.get("BLEU4", 0.0),
                "ROUGE": result.get("ROUGE", 0.0),
                "CIDEr-R": result.get("CIDEr-R", 0.0),
                "METEOR": result.get("METEOR", 0.0),
                "gemini_scoring": result.get("gemini_scoring", 0),
                "gemini_critique": result.get("gemini_critique", "")
            }
            writer.writerow(row)
    
    print(f"✅ Successfully scored {len(results)} samples!")
    print(f"Results saved to: {OUTPUT_FILE}")
    print("Note: CSV file uses UTF-8-BOM encoding for Excel compatibility.")
    
    # 打印統計信息
    if results:
        gemini_scores = [r["gemini_scoring"] for r in results if r["gemini_scoring"] > 0]
        if gemini_scores:
            print(f"\n=== Gemini Score Summary ===")
            print(f"Mean: {sum(gemini_scores)/len(gemini_scores):.2f}")
            print(f"Min: {min(gemini_scores)}")
            print(f"Max: {max(gemini_scores)}")
            print(f"Valid scores: {len(gemini_scores)}/{len(results)}")

if __name__ == "__main__":
    main()
