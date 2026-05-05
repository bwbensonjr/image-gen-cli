import base64
from openai import OpenAI
import sys

def main():
    prompt = sys.argv[1]
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

if __name__ == "__main__":
    main()
    
