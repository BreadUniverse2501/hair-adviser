import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model
model_path = '.BreadUniverse/dialoGPT-hair-recommender'  # Update this path to your model's location
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Streamlit app title
st.title("💆‍♂️ Hair Advisor")

# Description
description = """
Welcome to your personal Hair Product Matchmaker Bot! 🤖✨🧴 Are you ready to find your perfect hair care soulmate? Our bot specializes in pairing your unique tresses with the potions and lotions they've been dreaming of. Whether you're battling frizz, dreaming of volume, or longing for shine, we've got the scoop on the perfect products for you. 

Simply tell us about your hair type and concerns, and we'll concoct a personalized recommendation list just for you. 📝 From shampoos to serums, conditioners to masks, and everything in between—consider us your go-to guru for all things hair care. Say hello to happier hair days ahead! 💆‍♀️💆‍♂️💫
"""
st.markdown(description)

# Conversation history
conversation = []
user_input = st.text_input("You:", value="", max_chars=500)

if user_input:
    conversation.append(f"You: {user_input}")

    # Generate response
    input_ids = tokenizer.encode(user_input, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    conversation.append(f"Bot: {response}")
    st.write(f"Bot: {response}")

# Display conversation history
st.text_area("Conversation History", value="\n".join(conversation), height=200)
