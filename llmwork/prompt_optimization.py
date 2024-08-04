import json

from langchain_core.utils.json import _custom_parser

from llmwork.llm_job import LLMProcessor


class PromptOptimizationAgent:
    def __init__(self, llm_processor, initial_prompt, task_type, iterations=10):
        self.llm_processor = llm_processor
        self.initial_prompt = initial_prompt
        self.task_type = task_type
        self.iterations = iterations
        self.prompt_library = PromptLibrary()
        self.performance_history = []

    def optimize(self):
        try:
            best_prompt = self.initial_prompt
            best_score = 0

            for i in range(self.iterations):
                print(f"\nIteration {i + 1}:")
                current_prompt = self.initial_prompt if i == 0 else self.dynamic_optimize(best_prompt, i)
                criticism = self.criticize(current_prompt)
                improved_prompt = self.modify(current_prompt, criticism)
                performance = self.evaluate_prompt(improved_prompt)

                self.performance_history.append(performance)
                print(f"Performance: {performance}")

                if performance > best_score:
                    best_prompt = improved_prompt
                    best_score = performance

                print(f"Best score so far: {best_score}")

            self.prompt_library.add_prompt(self.task_type, best_prompt, best_score)
            return best_prompt
        except Exception as e:
            print(f"An error occurred during optimization: {e}")
            return self.initial_prompt

    def dynamic_optimize(self, prompt, iteration):
        try:
            optimization_prompt = f"""
            We are on iteration {iteration} of optimizing the following prompt:

            Current prompt: {prompt}

            Task type: {self.task_type}
            Performance history: {self.performance_history}

            Based on this information, please suggest an improved version of the prompt. 
            Consider the following aspects:
            1. Clarity and specificity
            2. Relevance to the task type
            3. Encouraging more detailed or creative responses
            4. Addressing any weaknesses indicated by the performance history

            Provide your optimized prompt:
            """

            optimized_prompt = self.llm_processor.process_llm_request(optimization_prompt, f"optimize_{iteration}",
                                                                      self.llm_processor.model)
            print(f"Optimized prompt: {optimized_prompt}")
            return optimized_prompt
        except Exception as e:
            print(f"Error in dynamic optimization: {e}")
            return prompt

    def criticize(self, prompt):
        try:
            criticism_prompt = f"""
            Please critically analyze the following prompt for the task type: {self.task_type}

            Prompt to criticize: {prompt}

            Provide a detailed criticism, focusing on:
            1. Potential ambiguities or lack of clarity
            2. Missing important aspects related to the task
            3. Possible improvements in structure or wording
            4. Any other weaknesses you can identify

            Your criticism:
            """

            criticism = self.llm_processor.process_llm_request(criticism_prompt, "criticize", self.llm_processor.model)
            print(f"Criticism: {criticism}")
            return criticism
        except Exception as e:
            print(f"Error in criticism: {e}")
            return "No specific criticism available."

    def modify(self, prompt, criticism):
        try:
            modification_prompt = f"""
            Based on the following criticism, please modify and improve the given prompt:

            Original prompt: {prompt}

            Criticism: {criticism}

            Task type: {self.task_type}

            Please provide an improved version of the prompt that addresses the criticism:
            """

            modified_prompt = self.llm_processor.process_llm_request(modification_prompt, "modify",
                                                                     self.llm_processor.model)
            print(f"Modified prompt: {modified_prompt}")
            return modified_prompt
        except Exception as e:
            print(f"Error in modification: {e}")
            return prompt

    def evaluate_prompt(self, prompt):
        try:
            evaluation_prompt = f"""
            Evaluate the following prompt on a scale of 0 to 10, considering its effectiveness for the task type: {self.task_type}

            Prompt to evaluate: {prompt}

            Provide your evaluation in JSON format:
            {{
                "score": <score from 0 to 10>,
                "explanation": "<brief explanation of the score>",
                "strengths": [<list of prompt strengths>],
                "weaknesses": [<list of prompt weaknesses>]
            }}
            """
            evaluation_json = self.llm_processor.process_llm_request(evaluation_prompt, "evaluate",
                                                                     self.llm_processor.model)
            print(f"Raw evaluation response: {evaluation_json}")

            try:
                evaluation = json.loads(evaluation_json)
            except json.JSONDecodeError:
                print("Failed to parse JSON, attempting to use custom parser")
                evaluation = _custom_parser(evaluation_json)

            print(f"Parsed evaluation: {evaluation}")

            if isinstance(evaluation, dict) and 'score' in evaluation:
                return evaluation['score']
            else:
                print("Evaluation does not contain a 'score' key")
                return 0
        except Exception as e:
            print(f"Error in prompt evaluation: {e}")
            return 0


class PromptLibrary:
    def __init__(self):
        self.prompts = {}

    def add_prompt(self, task_type, prompt, performance_score):
        if task_type not in self.prompts:
            self.prompts[task_type] = []
        self.prompts[task_type].append((prompt, performance_score))

    def get_best_prompt(self, task_type):
        if task_type in self.prompts:
            return max(self.prompts[task_type], key=lambda x: x[1])[0]
        return None


if __name__ == "__main__":
    db_path = "llm_results.db"
    openai_base_url = "http://localhost:5000/v1"
    openai_api_key = "810001a0a02948d5bf640a98cb69f653"

    llm_processor = LLMProcessor(db_path, openai_base_url, openai_api_key)
    initial_prompt = "帮我写一个搞笑小说"
    agent = PromptOptimizationAgent(llm_processor, initial_prompt, "AI_explanation", iterations=10)
    optimized_prompt = agent.optimize()
    print(f"Optimized prompt: {optimized_prompt}")
