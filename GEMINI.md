
## Gemini Agent Guide

This document outlines the conventions and guidelines for interacting with the Gemini agent in this repository.

### Primary Goal
The primary objective is to build a reusable and repeatable framework and application for:
1. Based on property information, property geocoding information, and macro economic data, train and build a model to predict property sale prices. The model will predict a `normalized_price` (the sale price relative to the median price of its local market) which can then be used to estimate a specific property's value.
2. Integrate data to visualise data for:
   - Data Profiling
   - Housing Market Pricing based on Year, Property Type, Street and Similar Property Groupings

### Interaction Style
- **Clarity and Conciseness:** Communicate instructions and requests clearly and concisely.
- **Provide Context:** When asking for changes, provide enough context for the agent to understand the task.
- **Use File Paths:** Refer to files and directories using their full paths.

### Development Workflow
The agent should follow the existing development workflow:
1.  **Understand the Task:** Analyze the request and the relevant codebase.
2.  **Propose a Plan:** Outline the steps to complete the task.
3.  **Implement the Changes:** Use the available tools to modify the code.
4.  **Verify the Changes:** Run tests and linting to ensure the changes are correct and follow the coding standards.

### Module-Specific Guidelines
Each module has its own `GEMINI.md` file with specific guidelines for interacting with the agent in that module. Please refer to those files for more detailed information.
