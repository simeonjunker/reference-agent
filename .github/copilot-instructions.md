# Copilot Instructions

## Project Overview

This project implements a vision-language AI system that generates descriptions distinguishing a target image from two distractor images.

The system receives three images:
- `target`: the image to describe.
- `distractor1`: the first image that the description must distinguish from the target.
- `distractor2`: the second image that the description must distinguish from the target.

The system produces a description of the target image that highlights relevant visual features and distinguishes it from both distractors.

The implementation must be based on LangGraph and a vision-language model (VLM) that is capable of processing images and is hosted on a custom server via llama.cpp (IP address: 129.70.40.28, port: 8345)

Use uv for project management.

## Core Design Principles

- Keep the implementation as simple as possible.
- Prioritize concise, readable, and transparent code over abstractions, extensibility, or clever optimizations.
- Use LangGraph for the core execution flow; do not introduce additional orchestration frameworks.
- Make data flow and LLM invocation explicit and easy to trace.
- Prefer a small number of functions, modules, and graph nodes.
- Avoid unnecessary classes, interfaces, factories, wrappers, and design patterns.
- Do not introduce abstractions until a concrete requirement justifies them.
- Reuse existing dependencies and project conventions.
- Do not add features that have not been requested.
- Make the smallest change that correctly implements the requested behavior.

## System Behavior

The system must:

- Accept exactly three logically distinct image inputs: `target`, `distractor1`, and `distractor2`.
- Pass the images to a vision-language model using the project's established model integration.
- Generate a description of the target image that distinguishes it from both distractors.
- Return the generated description in a clearly defined output structure.
- Make failures in input processing, graph execution, or model invocation visible to the caller.

The generated description should:

- Be short and concise, but informative
- Focus on visually observable characteristics of the target.
- Emphasize features that distinguish the target from both distractors.
- Avoid describing features that are equally applicable to all three images when more discriminative features are available.
- Avoid unsupported assumptions about image contents.
- Remain concise while preserving enough detail to identify the target.

## LangGraph Architecture

- Use LangGraph as the primary orchestration mechanism.
- Represent the execution flow with the smallest graph that satisfies the requirements.
- Define graph state explicitly using a lightweight, typed structure where appropriate.
- Keep node inputs, outputs, and state updates easy to understand.
- Use explicit graph edges and compile the graph through the standard LangGraph API.
- Keep model invocation visible in the relevant node or function.
- Avoid unnecessary nodes, conditional routing, cycles, agents, tools, memory, and checkpointing.
- Do not use LangGraph features unless they serve a concrete requirement.

Prefer a straightforward flow such as:
- Input validation → VLM invocation → Output

Only add separate graph nodes when they improve clarity or are required by the system's behavior.

## Vision-Language Model Integration

- Use the VLM integration and provider already established in the repository.
- Keep model configuration separate from the core description-generation logic where practical.
- Make the model invocation explicit and easy to test.
- Pass all three images to the model in a clearly defined order, with their roles identified unambiguously.
- Use a prompt that explicitly asks for a description of the target that distinguishes it from both distractors.
- Keep prompts concise and focused on the task.
- Avoid unnecessary prompt templates, chains, or intermediate LLM calls.
- Do not hardcode API keys, credentials, or environment-specific configuration.
- Do not silently substitute a mock, fallback model, or different provider when invocation fails.

The core system must remain testable without requiring live model API calls.

## Data I/O and Interfaces

- Define a clear input and output contract.
- Use consistent names for `target`, `distractor1`, and `distractor2` throughout the codebase.
- Make the accepted image representation explicit, such as file paths, URLs, bytes, or supported image objects.
- Follow the existing repository's image-loading and serialization conventions.
- Validate required inputs at the system boundary.
- Handle missing, malformed, or unsupported image inputs explicitly.
- Keep input conversion, model invocation, and output handling straightforward.
- Return the generated description in a predictable format.
- Avoid unnecessary data transformations, serialization layers, and wrapper objects.
- Keep the public interface small and stable.

Do not introduce multiple input formats unless required by the project.

## Code Style and Implementation

- Follow the repository's existing formatting, linting, and typing conventions.
- Use type hints for public functions, graph state, and important data structures.
- Prefer descriptive names and short functions with clear responsibilities.
- Keep control flow explicit; avoid deeply nested logic and unnecessary indirection.
- Avoid Any and broad exception handling unless justified.
- Raise meaningful errors rather than silently ignoring failures.
- Add comments only when they clarify non-obvious decisions.
- Do not add docstrings that merely restate a function's name or implementation.
- Avoid premature generalization and speculative future-proofing.
- Do not modify unrelated code or refactor working components without a clear reason.

## Testing Requirements

Tests must verify the core properties of the system, not merely that individual functions execute without errors.

### Core behavior
- Verify that the graph can be invoked with the required three inputs.
- Verify that the output contains a description in the expected format.
- Verify that the target and both distractors are represented correctly in the model input.
- Verify that the generated description is returned to the caller without unintended changes.
### LLM integration
- Verify that the configured LLM/VLM backbone is invoked during graph execution.
- Verify that the model receives all three images and the task instructions.
- Verify that image roles are preserved and not accidentally swapped.
- Verify that the model response is correctly extracted and propagated to the graph output.
- Verify that model invocation failures are surfaced appropriately.

Use a deterministic fake or mock model for unit tests. Tests must not require API credentials or incur model API costs.

### Data I/O
- Verify that valid inputs are accepted and passed through the expected processing path.
- Verify that missing or invalid inputs are rejected with useful errors.
- Verify that supported image representations are loaded or converted correctly.
- Verify that the output structure and field names match the public contract.

Test only the image formats and input representations the project actually supports.

### LangGraph execution
- Verify that the compiled graph can be invoked through its public interface.
- Verify that the expected execution path reaches the model invocation.
- Verify that state updates and final outputs are propagated correctly.
- Avoid tests that depend unnecessarily on LangGraph's internal implementation details.

### Testing principles
- Prefer small, deterministic tests with clear assertions.
- Test observable behavior rather than implementation details.
- Mock external services at the model integration boundary.
- Keep unit tests independent of network access and external infrastructure.
- Add integration tests only where they verify behavior that unit tests cannot adequately cover.
- Do not weaken or remove tests simply to make an implementation pass.

## Error Handling and Reliability

- Fail clearly when required inputs are missing or invalid.
- Preserve useful error context when image processing or model invocation fails.
- Do not silently return empty descriptions or fabricated results after failures.
- Avoid retries unless explicitly required and configured.
- Do not add fallback mechanisms without a clear requirement.
- Keep error handling close to the operation that can fail.
- Ensure errors remain observable through the public invocation interface.

## Dependencies and Configuration
- Use the existing package manager and dependency management files.
- Add only dependencies that are necessary for the requested functionality.
- Use supported LangGraph APIs and the repository's established VLM integration.
- Avoid introducing another graph framework, orchestration library, or validation framework without a concrete need.
- Keep provider configuration, model identifiers, and credentials outside the core business logic where practical.
- Update dependency files only when the implementation requires it.

## Documentation
- Keep the README focused on the system's purpose, setup, configuration, and usage.
- Document the expected image input format and output contract.
- Provide a minimal example of invoking the graph.
- Document how to run the tests.
- Explain any required environment variables without including secret values.
- Keep documentation synchronized with the actual implementation.

## Implementation Workflow

For each task:

- Inspect the existing repository, dependencies, tests, and conventions.
- Identify the smallest change that satisfies the requirement.
- Trace the relevant data flow from input through LangGraph and the VLM to output.
- Implement the change using the existing architecture wherever possible.
- Add or update tests for the affected behavior.
- Run the relevant tests and available quality checks.
- Review the diff for unnecessary abstractions, unrelated changes, and untested behavior.
- Summarize what changed and report which checks actually passed.

For architectural changes, explain the proposed approach before introducing additional complexity.

## Definition of Done

A change is complete when:

- The requested functionality is implemented.
- The LangGraph execution flow remains simple and explicit.
- The three image roles are preserved throughout the data flow.
- The VLM is invoked correctly and its output is handled as specified.
- Relevant input validation and failure behavior are covered.
- Tests verify core graph execution, model invocation, data I/O, and output handling as applicable.
- Tests do not require live LLM calls unless an explicitly separate integration test requires them.
- Relevant tests and quality checks pass, or failures are reported honestly.
- Documentation and dependency files are updated when necessary.
- No unnecessary abstractions, dependencies, or unrelated changes have been introduced.