# Objective
## Self-improving agents
something we've been wanting to build for a while. Essentially this would be creating an agent that improves automatically using prompt optimization techniques. You'd combine initial ground truth data, experiments, then production evals and prompt optimization through something like DSPy to achieve this.

## Core implemntation
- This is intended to be used in conjunction with the Arize set of tools, including:
    - Telemetry = they offer instrumentation such that all calls performed using the OpenAI SDK are logged as telemetry including all of the parameters and the entire output object
    - Functions for evaluating an LLM call (input and output exchange)
- Configuration requires telemetry to be configured by the user
- Must provide instrumentor (using wrapt) to track the evals being used
- Must download the telemetry logs from Arize/Pheonix (they utilize the same SDK interface) to get base data including:
    - llm input parameters
    - llm outputs
    - eval scores
    - (optional) eval explanations
- Must provide an interface for the developer to dowload the logs where they specify the data time from which to start sampling logs
    - This interface should probably be a class that could have further downstream uses for the developer to see progress, results, etc.
- Think of updating as an reinforcement learning (ML) problem
- Should deliver everything as a single python package.
# Plan
For the purposes of this project we will discard the agent nomenclature (since the scope of this project does not allude to anything commonly interpreted as agentic) and redefine the problem as "Self-improving LLM calls".

LLM calls will imply industry standard chat completion format.

Self-improving LLM calls suggests a system for autonimous improving of the LLM call to maximize outcomes against a user defined evaluation.

Originally I wanted to use Reinforcement Learning as a guide for how to implement a system of self improving agents. The scope of this project is focused on a single LLM call and optimizing the associated parameters to that single call there is no state treansitions. Consequently there is a major distinction in the sense that there is not transitions in state in our current form. In the event where this is expanded to multiple LLM calls (agents) then the this problem converges with RL. Because of the natural furure state of the project I will design this v0 such that it is extensible to accomodate a reinforcement learning system with multiple agents.

It necessarily needs to be offline because it depends on visibility of evals, since you cannot access the evals / tasks configured on the platform programatically, I have to use the SDK like `llm_classify` or functions within https://docs.arize.com/phoenix/api/evals#returns-1 . These functions are designed to be ran offline. Additionally offline evals simplifies things.

- Will only work with offline evals otherwise there is no access to eval params.

## TODO

- Function Wrapping (instrumentation)
- V2: Markov Decision Process
- Problems to Address
    - Convergence on local optima
    - Systematically prevent degeneration or regression
    - If the same model is used for generation and evaluation then you may see feedback collapse
- Incorporate chain of thought for update
    - Is there benefits to chain of thought + reasoning
- Ensure prompting is crystal clear use definitions and explicitly describe everything
- Introduce variability to the offline dataset
    - V1: Could also sample them through distinct groupings in clustered embeddings for maximum variability
- Run multiple trials of the update steps and merge them together imporiving generalization
- Add a threshold minimum improvement to determine if update is good enough to consider things as completed
- Update template should focus on having the LLM serve as gradient descent with small incremental steps
- Use train test split
- Make sure outputs are structured at least for the update step
- Maintain code modular, here are some key modules: Prompt update, orchestrator, data retrieval, data preprocessing
- Add some failure checks for generated actions
- V1: Treat each prompt version as a new build
- V2: Sometimes encourage exploration
    - V2: can use pseudo gamma scalar by using more maginitude words like high and very high
- Adding examples in prompt can introduce a sub problem of in-context optimization
- Generating feedback is proven to improve updates
- V1: When dealing with multiple evals we must deal with pareto prompt optimization
- Run update several times on different samples to reduce variance
- Make sure to use correct typings for chat completion messages
- For the primitive Actions object: should this be the entire model call object or should it be destructured???
    - for specificity it should be flat
- Need to be able to quantify evals as otherwise I am limited on evaluating comparisons using LLM. Need to also increase post update evaluation quantity.
- Use arize internally for the RL system
- Explore nested spans
- Exploration ~= Temperature
- Currentely dependent on chat completions llm calls and llm_categorize eval

## Research
- Offline inverse RL = learn the eval

## Completed
- Function Wrapping
    - function wrapping can be used to intercept calls and explore args
    - May be some value to finding function paths of the telemetry function wrapper to intercept that instead of the OpenAI function
    - This would enable a seamless online Reinforcement learning system
- Markov Decision Process
    - It hinges on having several timesteps to examine traces and select the most valuable trace and therefore has no purpose in the v0 of this project

# Design
PLease create the following workflow inside runner, first the user will establish the date from which  data will be fetched and onward,
once that data is provided, the runner will export (get) data from arize (telemetry provider) as shown here  inside @run_and_upsert_evals.py and will return data as a df as shown in context in jsonofied form ,
then you will get the evaluator data using the .get method ,
Using those two data sources you will compose a model that will consist of a list of samples (chat_history (dict), output_generation (dict), eval_scores (Any), eval_reasoning (str)), actions (system_prompt (dict), model (str)), eval_constants (eval_template(str), eval_rails()).
That will be our checkpoint,  then we will demo that inside @update_policy.py
