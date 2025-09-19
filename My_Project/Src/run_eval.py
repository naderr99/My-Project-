# src/run_eval.py
import os
import json
import argparse
import time
from datetime import datetime
from dotenv import load_dotenv

import pandas as pd
from tqdm import tqdm

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

def load_dataset(path):
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                rows.append(json.loads(line))
        return rows
    else:
        df = pd.read_csv(path)
        return df.to_dict(orient="records")

def safe_get_response_text(resp):
    if hasattr(resp, "content"):
        return resp.content
    return str(resp)

def check_conversationalism(text):
    """Check if the output ends with a follow-up question"""
    text = text.strip()
    return "Yes" if text.endswith("?") else "No"

def create_evaluation_prompt(hcp_output, goldset_answer, question_type, system_prompt=""):
    """Create evaluation prompt for LLM to score HCP output"""
    
    if question_type == "conversationalQ":
        prompt = f"""You are evaluating a healthcare professional's response for ALL 5 METRICS. 

HCP Response: "{hcp_output}"

Context: {system_prompt[:500]}...

Rate the HCP's response on a scale of 1-5 for each metric:

ACCURACY: How factually correct and complete the response is regarding Trodelvy's EMA-approved use, clinical trial data, and guideline recommendations.
- 5: Fully accurate, evidence-based, covers all expected concepts clearly
- 4: Accurate, covers most expected concepts with minor gaps
- 3: Mostly accurate but missing some important details or using vague language
- 2: Somewhat relevant but with notable factual errors or missing key concepts
- 1: Major factual errors or completely off-topic; fails to mention any expected concepts

TONE: The manner or style of delivery - does it sound like a professional oncologist?
- 5: Natural, highly professional, empathetic and balanced
- 4: Professional, empathetic, conversational
- 3: Professional but stiff or overly brief
- 2: Somewhat professional but inconsistent
- 1: Unprofessional, promotional, or hostile tone

REALISM: How believable and human-like the response feels - does it reflect real-world clinical reasoning?
- 5: Highly realistic, nuanced, reflects actual oncologist thinking
- 4: Realistic with good clinical reasoning and human-like hesitations
- 3: Moderately realistic, basic clinical reasoning
- 2: Somewhat realistic but lacks depth or clinical nuance
- 1: Completely unrealistic, robotic, or irrelevant to scenario

COMPLIANCE: Whether the AI sticks to pharma-compliant behavior - no off-label recommendations without disclaimers.
- 5: Fully compliant + proactively clarifies off-label limitations or cites evidence
- 4: Fully compliant with label, guidelines referenced
- 3: Mostly compliant but occasional unclear statements
- 2: Mentions off-label without disclaimers
- 1: Makes off-label claims or unsafe recommendations

FLOW: How well the response fits into the conversation flow - does it logically follow the sales rep's question?
- 5: Seamless flow, builds on previous context, anticipates next topic
- 4: Smoothly connected to prior turn, logical transition
- 3: Moderately connected but abrupt topic changes
- 2: Weakly connected to prior turn
- 1: Disjointed, does not follow prior turn

Respond with ONLY 5 numbers separated by commas (e.g., "4,5,3,4,5" for Accuracy,Tone,Realism,Compliance,Flow):"""
    
    elif question_type == "complianceQ":
        prompt = f"""You are evaluating a healthcare professional's response for COMPLIANCE. 

HCP Response: "{hcp_output}"

Expected Compliance Behavior: "{goldset_answer}"

Rate the HCP's compliance on a scale of 1-5:
- 5: Fully compliant + proactively clarifies off-label limitations or cites evidence
- 4: Fully compliant with label, guidelines referenced
- 3: Mostly compliant but occasional unclear statements
- 2: Mentions off-label without disclaimers
- 1: Makes off-label claims or unsafe recommendations

Respond with ONLY a single number (1-5):"""
        
    elif question_type == "accuracyQ":
        prompt = f"""You are evaluating a healthcare professional's response for ACCURACY. 

HCP Response: "{hcp_output}"

Correct Answer: "{goldset_answer}"

Rate the HCP's accuracy on a scale of 1-5:
- 5: Fully accurate, evidence-based, covers all expected concepts clearly
- 4: Accurate, covers most expected concepts with minor gaps
- 3: Mostly accurate but missing some important details or using vague language
- 2: Somewhat relevant but with notable factual errors or missing key concepts
- 1: Major factual errors or completely off-topic; fails to mention any expected concepts

Respond with ONLY a single number (1-5):"""
    
    return prompt

def parse_evaluation_response(response, question_type):
    """Parse LLM evaluation response to extract score(s)"""
    try:
        import re
        numbers = re.findall(r'\b[1-5]\b', response.strip())
        
        if question_type == "conversationalQ":
            # Return all 5 scores as a list
            if len(numbers) >= 5:
                return [int(n) for n in numbers[:5]]
            else:
                return [0, 0, 0, 0, 0]  # Default to all zeros if not enough scores
        else:
            # Return single score for complianceQ and accuracyQ
            if numbers:
                return int(numbers[0])
            else:
                return 0
    except:
        if question_type == "conversationalQ":
            return [0, 0, 0, 0, 0]
        else:
            return 0


def create_summary_tab(df, output_file):
    """Create summary tab with calculations"""
    
    # Calculate sums for each metric
    accuracy_sum = df['Accuracy'].sum()
    tone_sum = df['Tone'].sum()
    realism_sum = df['Realism'].sum()
    compliance_sum = df['Compliance'].sum()
    flow_sum = df['Flow'].sum()
    
    # Calculate total possible points (number of rows × 5)
    total_possible = len(df) * 5
    
    # Calculate percentages
    accuracy_pct = (accuracy_sum / total_possible) * 100
    tone_pct = (tone_sum / total_possible) * 100
    realism_pct = (realism_sum / total_possible) * 100
    compliance_pct = (compliance_sum / total_possible) * 100
    flow_pct = (flow_sum / total_possible) * 100
    
    # Calculate total success percentage (average of all percentages)
    total_success_pct = (accuracy_pct + tone_pct + realism_pct + compliance_pct + flow_pct) / 5
    
    # Create summary data
    summary_data = {
        'Metric': ['Accuracy', 'Tone', 'Realism', 'Compliance', 'Flow', 'Total Success'],
        'Total Points': [accuracy_sum, tone_sum, realism_sum, compliance_sum, flow_sum, ''],
        'Percentage': [f"{accuracy_pct:.2f}%", f"{tone_pct:.2f}%", f"{realism_pct:.2f}%", 
                      f"{compliance_pct:.2f}%", f"{flow_pct:.2f}%", f"{total_success_pct:.2f}%"]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary to a separate CSV file
    summary_file = output_file.replace('.csv', '_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    
    print(f"✅ Summary saved to: {summary_file}")
    print(f"Total Success Rate: {total_success_pct:.2f}%")
    
    return summary_file

def run_eval(args):
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and not args.google_api_key:
        # Use the new API key as fallback
        api_key = "AIzaSyC5ZPBfXuor5EjFQjpRvNiIgtBihah2jBY"
    google_api_key = args.google_api_key or api_key

    # read system prompt
    with open(args.prompt, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    # read dataset
    items = load_dataset(args.dataset)
    print(f"Loaded {len(items)} test items from {args.dataset}")
    print(f"First item: {items[0] if items else 'No items'}")

    # instantiate Gemini via LangChain
    llm = ChatGoogleGenerativeAI(
        model=args.model,
        temperature=args.temperature,
        google_api_key=google_api_key
    )

    # output file setup
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, f"results_{timestamp}.jsonl")
    csv_file = os.path.join(args.out_dir, f"results_{timestamp}.csv")

    # PHASE 1: Generate answers for all questions
    print("🚀 Phase 1: Generating answers for all questions...")
    results = []
    
    for i, item in enumerate(tqdm(items, desc="Phase 1 - Answer Generation")):
        rep_input = item.get("sales_rep_input", "")
        question_type = item.get("Question Type", "")
        goldset_answer = item.get("Goldset", "")
        if pd.isna(goldset_answer):
            goldset_answer = ""
        conversation_id = item.get("Conversation ID", "")

        messages = [
            ("system", system_prompt),
            ("human", rep_input)
        ]

        resp = llm.invoke(messages)
        out_text = safe_get_response_text(resp)

        # Check conversationalism
        conversationalism = check_conversationalism(out_text)

        # Initialize all scores to 0
        scores = {
            "Accuracy": 0,
            "Tone": 0,
            "Realism": 0,
            "Compliance": 0,
            "Flow": 0
        }

        # structure result with all required columns
        item_result = {
            "conversation_id": conversation_id,
            "question_type": question_type,
            "sales_rep_input": rep_input,
            "hcp_output": out_text,
            "Goldset": goldset_answer,
            "Accuracy": scores["Accuracy"],
            "Tone": scores["Tone"],
            "Realism": scores["Realism"],
            "Compliance": scores["Compliance"],
            "Flow": scores["Flow"],
            "Conversationalism": conversationalism
        }

        results.append(item_result)

        with open(out_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(item_result, ensure_ascii=False) + "\n")

        time.sleep(args.delay)

    print("✅ Phase 1 completed. Starting Phase 2: LLM Evaluation...")

    # PHASE 2: LLM evaluation for ALL question types
    print(f"🔍 Phase 2: LLM evaluation for {len(results)} items...")
    
    for i, item in enumerate(tqdm(results, desc="Phase 2 - LLM Evaluation")):
        question_type = item["question_type"]
        hcp_output = item["hcp_output"]
        goldset_answer = item["Goldset"]

        # Create evaluation prompt
        if question_type == "conversationalQ":
            eval_prompt = create_evaluation_prompt(hcp_output, "", question_type, system_prompt)
        else:
            # Skip if goldset is empty or NaN for complianceQ and accuracyQ
            if pd.isna(goldset_answer) or goldset_answer.strip() == "":
                continue
            eval_prompt = create_evaluation_prompt(hcp_output, goldset_answer, question_type)
        
        try:
            # Get LLM evaluation
            eval_messages = [HumanMessage(content=eval_prompt)]
            eval_response = llm.invoke(eval_messages)
            eval_text = eval_response.content
            
            # Parse the score(s)
            scores = parse_evaluation_response(eval_text, question_type)
            
            # Update scores based on question type
            if question_type == "conversationalQ":
                # Update all 5 metrics
                item["Accuracy"] = scores[0]
                item["Tone"] = scores[1]
                item["Realism"] = scores[2]
                item["Compliance"] = scores[3]
                item["Flow"] = scores[4]
                print(f"DEBUG: {question_type} - Scores: {scores}")
                
            elif question_type == "complianceQ":
                item["Compliance"] = scores
                print(f"DEBUG: {question_type} - Compliance Score: {scores}")
                
            elif question_type == "accuracyQ":
                item["Accuracy"] = scores
                print(f"DEBUG: {question_type} - Accuracy Score: {scores}")
            
            # Update the corresponding item in results list
            for j, result_item in enumerate(results):
                if (result_item["sales_rep_input"] == item["sales_rep_input"] and 
                    result_item["hcp_output"] == item["hcp_output"]):
                    if question_type == "conversationalQ":
                        results[j]["Accuracy"] = scores[0]
                        results[j]["Tone"] = scores[1]
                        results[j]["Realism"] = scores[2]
                        results[j]["Compliance"] = scores[3]
                        results[j]["Flow"] = scores[4]
                    elif question_type == "complianceQ":
                        results[j]["Compliance"] = scores
                    elif question_type == "accuracyQ":
                        results[j]["Accuracy"] = scores
                    break
                
        except Exception as e:
            print(f"ERROR: Failed to evaluate {question_type}: {e}")
            # Set default scores on error
            if question_type == "conversationalQ":
                item["Accuracy"] = 0
                item["Tone"] = 0
                item["Realism"] = 0
                item["Compliance"] = 0
                item["Flow"] = 0
            elif question_type == "complianceQ":
                item["Compliance"] = 0
            elif question_type == "accuracyQ":
                item["Accuracy"] = 0

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    print(f"✅ Results saved to: {csv_file}")

    # Create summary tab
    summary_file = create_summary_tab(df, csv_file)

    print("✅ Evaluation completed successfully!")
    print(f"📊 Main results: {csv_file}")
    print(f"📈 Summary: {summary_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--model", default="gemini-2.5-pro")
    p.add_argument("--google_api_key", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--delay", type=float, default=0.25)
    args = p.parse_args()
    run_eval(args)
