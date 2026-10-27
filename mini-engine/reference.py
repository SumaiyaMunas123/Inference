"""
Load the GPT-2 tokenizer and the GPT-2 model (the small one is called "gpt2").
Turn the prompt "The weather in Colombo is" into token IDs.
Generate about 20 new tokens with greedy decoding.
Turn the result back into text and print it.
"""

# imports AutoTokenizer and AutoModelForCasualLM from the transformers library from Hugging Face
from transformers import AutoTokenizer, AutoModelForCausalLM

# downloads and uses the small GPT-2 model and tokenizer from Hugging Face's model hub
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# The tokenizer turns text into numbers and back. 
# The model is the 124M weights plus the maths.

prompt = "The weather in Colombo is"

"""
Prompt to token IDs. Call tokenizer(...) with your prompt string and add return_tensors="pt". That option means “give me PyTorch tensors”, which the model needs. It returns a dictionary, and the token IDs are under the key "input_ids". Print it and look at what’s inside.
Generate. Call model.generate(...) and pass it the inputs. Two options matter: max_new_tokens=20 (how many new tokens to add) and do_sample=False (this is greedy decoding: always pick the single most likely next token, no randomness). Tip: **inputs unpacks the dictionary into the function’s arguments.
Back to text. generate returns token IDs, not words. Use tokenizer.decode(...) on the first row of the result. The result has one row per prompt, and you only gave one prompt, so use index [0].
Print it.
"""

# Tokenize the prompt and convert it into token IDs
inputs =tokenizer(prompt,return_tensors="pt") #turns the sentences into a dic
ids = inputs["input_ids"][0].tolist()#converts pytorch tensor into a normal py list
tokens= tokenizer.convert_ids_to_tokens(ids)

# Generate new tokens using greedy decoding (no randomness)
# greedy means that the model always picks the single most likely next token, rather than sampling from a distribution of possible next tokens. 
# This is done by setting do_sample=False.
output = model.generate(
    **inputs, #hands everything in the dictionary to generate
    max_new_tokens=20, # Generate 20 new tokens
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id, # use the end-of-sequence token of the tokenizer as the padding token ID, 
    # what is padding? It’s a special token that the model uses to fill in empty space when it needs to make all the inputs the same length. 
    # The model doesn’t need it here, but it’s a good habit to include it.
)

# Convert the generated token IDs back into text
text = tokenizer.decode(output[0]) # Decode the first (and only) generated sequence

# Print the generated text
print("Token IDs:", ids)
print("Token count:", len(ids))
print("Tokens:", tokens)
print("Generated:", text)

# save the answer key
with open("refernece_output.txt", "w",encoding="utf-8") as f:
    f.write(f"Prompt:{prompt}\n")
    f.write(f"Token IDs:{ids}\n")
    f.write(f"Generated:{text}\n")





