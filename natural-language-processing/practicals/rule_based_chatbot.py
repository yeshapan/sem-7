# -*- coding: utf-8 -*-
"""
Module: Rule-Based Chatbot
Utilizes NLTK Chat and reflections for conversational intent-matching.
"""

from nltk.chat.util import Chat, reflections


def create_chatbot() -> Chat:
    """Instantiates a regex pattern-matching chatbot."""
    chat_pairs = [
        [r"hi|hello|hey", ["Hello! How can I assist you with your queries today?"]],
        [r"(.*)price(.*)", ["Our pricing varies by category, starting from 999 INR."]],
        [r"(.*)delivery(.*)", ["Orders are delivered within 3 to 5 business days."]],
        [r"(.*)return(.*)", ["Unused products can be returned within 7 days of delivery."]],
        [r"thanks|thank you", ["You are very welcome!"]],
        [r"bye", ["Goodbye! Have a great day ahead."]],
        [r"(.*)", ["I am sorry, could you rephrase that? Ask about pricing, returns, or delivery."]]
    ]
    return Chat(chat_pairs, reflections)


def interact_with_bot(bot: Chat, queries: list):
    """Simulates responses for a list of queries."""
    for query in queries:
        response = bot.respond(query)
        print(f"User: {query}")
        print(f"Bot : {response}\n")


if __name__ == "__main__":
    bot_instance = create_chatbot()
    test_queries = [
        "Hello there",
        "What is the delivery time?",
        "Can you tell me about the price?",
        "bye"
    ]
    interact_with_bot(bot_instance, test_queries)