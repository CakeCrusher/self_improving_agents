from self_improving_agents.evaluator_handler.evaluator_saver import EvaluatorSaver

evaluator_saver = EvaluatorSaver()
evaluator_data = evaluator_saver.load_evaluator("formatting_classify")

if not evaluator_data:
    raise ValueError("Evaluator data not found")

print(evaluator_data.model_dump_json(indent=4))
