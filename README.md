# Serverless Function: LLM connector
SF to post prompts to LLM and storing the response back among annotation fields.

## Supported LLMs
Currently, the following LLMs are supported:
* OpenAI
* AWS Bedrock

## Installation guide
* Copy the code of the SF for your selected provider
* In Rossum UI, create a new extension
* Select **Custom function** extension type and **Python** as programming language
* Paste the code and save the extension config
* Edit the extension and set configuration & secrets as described below


## Configuration

The following configuration fields are required:

### Input
Here you can configure annotation fields as variables for the prompt in the form of key-value pairs (supports both header fields and line items):
```json
"input": {
  "variable": "{fieldId}"
}
```
for example:
```json
"input": {
  "input_address": "sender_address"
}
```
You can then use the variable `input_address` in the prompt definition.

### Output
Defines `field_id` where the function stores its result:
```json
"output": "sender_address_parsed_json"
```

### Prompt
Defines the prompt that will be submitted to the LLM:
```json
"prompt": "This is an address extracted from a document: \"input_address\" Parse the address into the following fields: name, street, house number, city, state, country. Return valid JSON. Return no extra text."
```

### Example configuration
```json
{
  "keep_history": true,
  "configurations": [
    {
      "input": {
        "input_address": "sender_address"
      },
      "output": "sender_address_parsed_json",
      "prompt": "This is an address extracted from a document: \"input_address\" Parse the address into the following fields: name, street, house number, city, state, country. Return valid JSON. Return no extra text."
    },
    ...
  ]
}
```

### Secrets

```json
{
  "key": "<access_key_id>",
  "secret": "<access_key_secret>"
}
```

## Advanced configuration
Every LLM comes with its own parameters such as model, limit of tokens, used region, etc. Configuration of the following fields is optional.

### AWS Bedrock

- `model`  
Specifies the Claude model to use (default: `anthropic.claude-3-haiku-20240307-v1:0`)
- `region`  
AWS Bedrock region to use (default: `us-east-1`)
- `max_tokens`  
Maximum tokens for output (default: `64`)
- `temperature`  
Controls output randomness (range: 0.0 to 1.0, default: `0.0`)
- `batch_size`  
For line items; defines how many items are processed per request. Default is `1` (processes
  individually). Set to 0 to process all items in one request, or specify any other number to indicate how many line
  items will be sent in each request. Note: Processing multiple items in one request increases the risk of AI
  hallucinations
- `keep_history`  
Determines if the model retains session history for context-aware responses (default: `false`)

#### Model information

- Model availability and pricing based on region - https://aws.amazon.com/bedrock/pricing/#Anthropic
- Model IDs - https://docs.anthropic.com/en/docs/about-claude/models#model-names (AWS Bedrock column)

### OpenAI

- `model`  
Specifies the model to use (default: `gpt-3.5-turbo`)
- `max_tokens`  
Maximum tokens for output (default: `64`)
- `temperature`  
Controls output randomness (range: 0.0 to 2.0, default: `0.0`)
- `batch_size`  
For line items; defines how many items are processed per request. Default is `1` (processes
  individually). Set to 0 to process all items in one request, or specify any other number to indicate how many line
  items will be sent in each request. Note: Processing multiple items in one request increases the risk of AI
  hallucinations
