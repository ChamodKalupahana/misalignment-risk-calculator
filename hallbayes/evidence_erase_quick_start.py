import os

from hallucination_toolkit import OpenAIItem, OpenAIPlanner, OpenAIBackend, generate_answer_if_allowed

backend = OpenAIBackend(model="gpt-4o-mini")
planner = OpenAIPlanner(backend, temperature=0.3)

prompt = """Task: Answer strictly based on the evidence below.
Question: Who won the Nobel Prize in Physics in 2019?
Evidence:
- Nobel Prize press release (2019): James Peebles (1/2); Michel Mayor & Didier Queloz (1/2).
Constraints: If evidence is insufficient or conflicting, refuse.
"""

item = OpenAIItem(
    prompt=prompt, 
    n_samples=1, 
    m=2, 
    fields_to_erase=["Evidence"], 
    skeleton_policy="auto"
)

planner = OpenAIPlanner(backend, temperature=0.3)
metrics = planner.run([item], h_star=0.05, isr_threshold=1.0)
metric = metrics[0]
    
print(f"Decision: {'ANSWER' if metric.decision_answer else 'REFUSE'}")
print(f"ISR: {metric.isr:.3f}")
print(f"Delta_bar: {metric.delta_bar:.4f} nats")
print(f"B2T: {metric.b2t:.4f} nats")
print(f"RoH bound: {metric.roh_bound:.3f}")
print(f"Rationale: {metric.rationale}")

is_debug_mode = os.environ.get("HALLUCINATION_TOOLKIT_DEBUG", "0") == "1"
if is_debug_mode:
    print("\n DEBUG: Generating answer...")
    answer = generate_answer_if_allowed(backend, item, metric)
    if answer:
        print(f"Answer: {answer}")