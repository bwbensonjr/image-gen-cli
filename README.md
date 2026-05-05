# Image Generation CLI 

A command line interface for generating images. The intial focus will
be on using the OpenAI API and `gpt-image-2`.

## Behavior 

- Accepts an overall prompt describing what is wanted 
- Optionally it can accept other inputs including
  - Markdown or text files 
  - Example images
- An output file name specification (defaults to "output-image")
- The output is one or more images with names like
  "output-image-1.png" using the output file name spec.

## Technical Guidelines 

- Use OpenAI Python API which will use environment variables for
  OpenAI key and base URL.
- Use Python `argparse`.
- Use clig.dev command line guidelines cached locally as
  [`docs/cli_guidelines.md`](docs/cli_guidelines.md).
  
## OpenAI `gpt-image-2` Use

The recommendation for using `gpt-image-2` from the OpenAI API is
shown in this example:

```python
prompt = "Generate an image of Otter hugging a Penguin"
output_file = "image_output.png"
client = OpenAI() 
response = client.responses.create(
    model="gpt-5.5",
    input=prompt,
    tools=[{"type": "image_generation"}],
)
# Save the image to a file
image_data = [
    output.result
    for output in response.output
    if output.type == "image_generation_call"
]
if image_data:
    image_base64 = image_data[0]
    with open(output_file, "wb") as f:
        f.write(base64.b64decode(image_base64))
```

