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

