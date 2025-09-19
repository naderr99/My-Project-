# Project README

This is the official README for defining this pipeline and instructions for running it smoothly.

---

# Model

**Gemini 2.5 Pro** was chosen due to:
- Efficiency  
- Massive training data  
- Ease of access through API calls  
- Ease of steering via System Prompt  

---

# Prompt Architecture Overview  

This prompt is structured into **five major sections**, each with subsections that define the persona, rules of interaction, and medical context.  

## 1. Persona  

- **Overview** – Defines the AI’s role as Dr. Sarah Tam, a breast cancer oncologist.  
- **Characteristics** – Clinical focus, common challenges, tone, and regulatory compliance.  

## 2. Instructions  

- **Do’s** – Behavioral rules: realism, empathy, evidence-based reasoning, guideline references, compliance, concise replies, follow-up questioning.  
- **Conversational Style** – Maintain professional tone while mimicking natural HCP hesitations, skepticism, and concerns.  

## 3. Boundaries  

- **Don’ts** – Restrictions: no personal data, no off-label promotion without disclaimers, no irrelevant or exam-like questions, max 4 sentences per reply.  

## 4. Example Dialogue  

- Illustrative exchange between a sales representative and the oncologist persona.  
- Demonstrates tone, compliance, questioning style, and flow of conversation.  

## 5. Context Summary  

- **Purpose** – Why the persona exists (HCP training simulation for Trodelvy discussions).  
- **Guidelines Reference** – NCCN Breast Cancer Guidelines (v4.2025) covering diagnosis, pathology, treatment types.  
- **Drug-Specific Knowledge** – Trodelvy mechanism, approved indications, placement in therapy, clinical trial outcomes.  
- **Administration & Safety** – Dosing, infusion protocols, prophylaxis, monitoring, and adverse reaction management.  
- **Compliance & Keywords** – Reinforces regulatory standards and natural incorporation of key medical terms.  


---

# Evaluation Categories and Scoring

The evaluation categories were chosen manually and crafted according to the needs of this task in the following matrix:

## Manual Evaluation Scoring System (5-Point Scale + Conversationalism)

### Accuracy
- **Measures**: factual correctness & completeness (Trodelvy EMA label, clinical trial data, ESMO/NCCN guidelines).  
- **Scale**:  
  - 1 = Major factual errors, off-topic  
  - 3 = Mostly accurate, but missing some important details or vague  
  - 5 = Fully accurate, evidence-based, covers all key points  
- **Success requirement**: >95%  

### Tone
- **Measures**: professionalism, empathy, neutrality (like an oncologist, not promotional).  
- **Scale**:  
  - 1 = Unprofessional, promotional, or hostile  
  - 3 = Professional but stiff or inconsistent  
  - 5 = Natural, highly professional, empathetic, balanced  
- **Success requirement**: >90%  

### Realism
- **Measures**: believability & clinical nuance (real-world reasoning, hesitations, workflow constraints).  
- **Scale**:  
  - 1 = Completely unrealistic or robotic  
  - 3 = Some realism, basic reasoning  
  - 5 = Highly realistic, nuanced, reflects true oncologist thinking  
- **Success requirement**: >90%  

### Compliance
- **Measures**: adherence to pharma compliance (no off-label without disclaimers, no unsafe claims, correct references).  
- **Scale**:  
  - 1 = Unsafe/off-label claims  
  - 3 = Mostly compliant, some unclear statements  
  - 5 = Fully compliant + clarifies off-label limitations or cites evidence  
- **Success requirement**: >95%  

### Flow
- **Measures**: conversational smoothness & logical progression (follows rep’s question, avoids abrupt changes).  
- **Scale**:  
  - 1 = Disjointed, doesn’t follow prior turn  
  - 3 = Moderately connected but abrupt transitions  
  - 5 = Seamless flow, anticipates next topic, smooth continuity  
- **Success requirement**: >90%  

### Conversationalism
- **Measures**: whether the model asks follow-up questions.  
- **Boolean**: Yes / No  
- **Success requirement**: >50% Yes  

**Note:**  
The metrics chosen are all subjective due to the nature of the task, as one cannot measure tone, flow, and realism in an automated fashion. Accuracy and compliance are also difficult to measure in free text conversations, although attempts were made (see Limitations section).

 # How to Run Locally


1. **Obtain Gemini API** through Google AI Studio  

2. **Create & activate a virtual environment**  
   ```
   # Create venv
   python -m venv .venv

   # Activate (Windows PowerShell)
   .venv\Scripts\Activate
3. **Install dependencies**



pip install -r dependencies.txt
Set up environment variables
Create a file called .env in the project root and add your Gemini API key:

4. **Create a .env file with your API**

GEMINI_API_KEY=your_api_key_here

5. **Run the evaluation script**



python src/run_eval.py \
  --dataset eval/eval_set.jsonl \
  --prompt prompt/hcp_system_prompt.md \
  --out_dir results

5. **Arguments**
Already added in the main script:

python

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
Post-Processing
After results are generated, run src/convert_to_csv to convert outputs to a more usable form for human evaluation.

6. **After manual evaluation is complete, paste the evaluation matrix from:**

Evaluation Result/Human Evaluation Results/Evaluation Matrix tab
This will automatically calculate the counts and percentages.



# Limitations
## Model Selection
Gemini is not medically refined and this may cause accuracy issues. This was partly mitigated by adding knowledge base snippets in the system prompt and forcing the model selection to Gemini 2.5 Pro, which has more favorable reasoning.
Limitations included lack of compute power to host open-source models and a free API tier restricting usage to 50 requests per day.

## Evaluation Metrics
The evaluation metrics chosen were subjective and scored manually. Attempts to automate accuracy and compliance scoring included adding question type and conversation ID. LLM auto scoring failed due to limited API requests, and fuzzy-cosine matching yielded irregular results requiring more refinement than was feasible.

## Dataset Creation
The dataset was created semi-synthetically.

A skeleton of 15 questions was generated using GPT, then refined to resemble real-world interactions.

The remaining questions were manually created.

This means the dataset may not fully capture real-world data.

## Special Question Types
accuracyQ and complianceQ were added solely to score these two metrics respectively, which explains the shape of the scoring results.

