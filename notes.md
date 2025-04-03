# Objective
## Self-improving agents
something we've been wanting to build for a while. Essentially this would be creating an agent that improves automatically using prompt optimization techniques. You'd combine initial ground truth data, experiments, then production evals and prompt optimization through something like DSPy to achieve this.

# Plan
For the purposes of this project we will discard the agent nomenclature (since the scope of this project does not allude to anything commonly interpreted as agentic) and redefine the problem as "Self-improving LLM calls".

LLM calls will imply industry standard chat completion format.

Self-improving LLM calls suggests a system for autonimous improving of the LLM call to maximize outcomes against a user defined evaluation.

Originally I wanted to use Reinforcement Learning as a guide for how to implement a system of self improving agents. The scope of this project is focused on a single LLM call and optimizing the associated parameters to that single call there is no state treansitions. Consequently there is a major distinction in the sense that there is not transitions in state in our current form. In the event where this is expanded to multiple LLM calls (agents) then the this problem converges with RL. Because of the natural furure state of the project I will design this v0 such that it is extensible to accomodate a reinforcement learning system with multiple agents. 

It necessarily needs to be offline because it depends on visibility of evals, since you cannot access the evals / tasks configured on the platform programatically, I have to use the SDK like `llm_classify` or functions within https://docs.arize.com/phoenix/api/evals#returns-1 . These functions are designed to be ran offline. Additionally offline evals simplifies things.

# Research
## TODO

- Function Wrapping
- Markov Decision Process
## Completed
- Function Wrapping
    - function wrapping can be used to intercept calls and explore args
    - May be some value to finding function paths of the telemetry function wrapper to intercept that instead of the OpenAI function
    - This would enable a seamless online Reinforcement learning system
- Markov Decision Process
    - It hinges on having several timesteps to examine traces and select the most valuable trace and therefore has no purpose in the v0 of this project

# Design
