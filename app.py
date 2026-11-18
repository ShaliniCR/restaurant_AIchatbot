"""Restaurant AI Agent — LangChain + Claude Haiku 5.5 + Streamlit."""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic

# Reads ANTHROPIC_API_KEY from .env locally; on Render it comes from the Environment tab.
load_dotenv()


# ============================================================
# DATA (same in-memory data as the session notebook)
# ============================================================

RESTAURANTS_BY_CUISINE = {
    "indian": [
        {"name": "Spice Palace", "rating": 4.5, "price": "₹₹"},
        {"name": "Curry House", "rating": 4.3, "price": "₹₹"},
        {"name": "Taj Flavors", "rating": 4.7, "price": "₹₹₹"},
    ],
    "italian": [
        {"name": "Pizza Bella", "rating": 4.4, "price": "₹₹"},
        {"name": "Pasta Dreams", "rating": 4.6, "price": "₹₹₹"},
        {"name": "Roma Kitchen", "rating": 4.2, "price": "₹₹"},
    ],
    "chinese": [
        {"name": "Dragon House", "rating": 4.5, "price": "₹₹"},
        {"name": "Ming's Kitchen", "rating": 4.3, "price": "₹₹"},
        {"name": "Golden Wok", "rating": 4.8, "price": "₹₹₹"},
    ],
    "mexican": [
        {"name": "Taco Fiesta", "rating": 4.4, "price": "₹₹"},
        {"name": "Salsa Nights", "rating": 4.6, "price": "₹₹₹"},
        {"name": "Burrito House", "rating": 4.2, "price": "₹₹"},
    ],
}

RESTAURANT_DETAILS = {
    "Spice Palace": {
        "cuisine": "Indian",
        "rating": 4.5,
        "address": "123 Food Street, City Center",
        "phone": "9876543210",
        "hours": "11 AM - 11 PM",
        "speciality": "Butter Chicken, Biryani",
    },
    "Pizza Bella": {
        "cuisine": "Italian",
        "rating": 4.4,
        "address": "456 Dine Avenue, Downtown",
        "phone": "9123456789",
        "hours": "12 PM - 10:30 PM",
        "speciality": "Margherita Pizza, Tiramisu",
    },
    "Dragon House": {
        "cuisine": "Chinese",
        "rating": 4.5,
        "address": "789 Noodle Lane, Market Area",
        "phone": "9111222333",
        "hours": "11:30 AM - 11 PM",
        "speciality": "Hakka Noodles, Fried Rice",
    },
    "Taco Fiesta": {
        "cuisine": "Mexican",
        "rating": 4.4,
        "address": "321 Spice Row, Hip District",
        "phone": "9222333444",
        "hours": "1 PM - 12 AM",
        "speciality": "Fish Tacos, Guacamole",
    },
    "Curry House": {
        "cuisine": "Indian",
        "rating": 4.3,
        "address": "555 Flavor Street, Old Town",
        "phone": "9333444555",
        "hours": "11 AM - 10 PM",
        "speciality": "Paneer Tikka, Dal Makhani",
    },
    "Taj Flavors": {
        "cuisine": "Indian",
        "rating": 4.7,
        "address": "999 Taste Lane, Premium Zone",
        "phone": "9444555666",
        "hours": "12 PM - 11 PM",
        "speciality": "Biryani, Tandoori Chicken",
    },
    "Golden Wok": {
        "cuisine": "Chinese",
        "rating": 4.8,
        "address": "777 Wok Street, Central Market",
        "phone": "9555666777",
        "hours": "11 AM - 11 PM",
        "speciality": "Szechuan Noodles, Dim Sum",
    },
}

AVAILABLE_SLOTS = {
    "Spice Palace": {"2024-11-15": True, "2024-11-16": False, "2024-11-17": True},
    "Pizza Bella": {"2024-11-15": False, "2024-11-16": True, "2024-11-17": True},
    "Dragon House": {"2024-11-15": True, "2024-11-16": True, "2024-11-17": False},
    "Taco Fiesta": {"2024-11-15": True, "2024-11-16": True, "2024-11-17": True},
    "Curry House": {"2024-11-15": True, "2024-11-16": False, "2024-11-17": True},
    "Taj Flavors": {"2024-11-15": False, "2024-11-16": True, "2024-11-17": True},
}


def _match_name(name: str, names) -> str | None:
    """Return the canonical restaurant name, matching case-insensitively."""
    wanted = name.strip().lower()
    for candidate in names:
        if candidate.lower() == wanted:
            return candidate
    return None


# ============================================================
# TOOLS
# ============================================================

@tool
def search_restaurants_by_cuisine(cuisine_type: str) -> list:
    """Search for restaurants by cuisine type.

    Supported cuisines: Indian, Italian, Chinese, Mexican (case-insensitive).

    Args:
        cuisine_type: The cuisine to search for, e.g. "Indian".

    Returns:
        A list of restaurants, each with "name", "rating" and "price".
        An empty list if the cuisine is not supported.
    """
    return RESTAURANTS_BY_CUISINE.get(cuisine_type.strip().lower(), [])


@tool
def get_restaurant_details(restaurant_name: str) -> dict:
    """Get detailed information about a restaurant.

    Args:
        restaurant_name: The restaurant's name, e.g. "Golden Wok".

    Returns:
        A dict with cuisine, rating, address, phone, hours and speciality,
        or {"error": "..."} if the restaurant is not found.
    """
    name = _match_name(restaurant_name, RESTAURANT_DETAILS)
    if name is None:
        return {"error": f"Restaurant '{restaurant_name}' not found"}
    return {"name": name, **RESTAURANT_DETAILS[name]}


@tool
def check_reservation_availability(restaurant_name: str, date: str, party_size: int) -> str:
    """Check if a restaurant has reservation availability on a date for a party size.

    Args:
        restaurant_name: The restaurant's name, e.g. "Spice Palace".
        date: The date in YYYY-MM-DD format, e.g. "2024-11-15".
        party_size: The number of people in the party.

    Returns:
        A message saying the restaurant is available (✓), fully booked (✗),
        or not found.
    """
    name = _match_name(restaurant_name, AVAILABLE_SLOTS)
    if name is None:
        return f"Restaurant '{restaurant_name}' not found."

    if AVAILABLE_SLOTS[name].get(date.strip(), False):
        return f"✓ {name} has availability for {party_size} people on {date}"
    return f"✗ {name} is fully booked for {date}"


TOOLS = [search_restaurants_by_cuisine, get_restaurant_details, check_reservation_availability]


# ============================================================
# AGENT
# ============================================================

SYSTEM_PROMPT = """
You are a helpful restaurant recommendation assistant.

You have access to these tools:

1. Search restaurants by cuisine
2. Get restaurant details
3. Check reservation availability

Use the appropriate tool whenever necessary.

Do not invent restaurant information.

If the available tools cannot answer the question,
clearly say that the requested information is outside
your available capabilities.

You only answer questions that your tools can answer: finding
restaurants by cuisine, restaurant details, and reservation
availability. For anything else, including general knowledge
(geography, history, trivia, etc.), do not answer it, even if
you know the answer. Politely say it is outside your
capabilities and offer to help with restaurants instead.

Be friendly, conversational, and concise.
"""


@st.cache_resource
def build_agent():
    model = ChatAnthropic(model="claude-haiku-5-5", max_tokens=4096)
    return create_agent(model=model, tools=TOOLS, system_prompt=SYSTEM_PROMPT)


def message_text(message) -> str:
    """Pull the plain text out of a model message (content may be a list of blocks)."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


# ============================================================
# STREAMLIT CHAT UI
# ============================================================

st.set_page_config(page_title="Restaurant AI Agent", page_icon="🍽️")
st.title("🍽️ Restaurant AI Agent")
st.caption(
    "Find restaurants by cuisine (Indian, Italian, Chinese, Mexican), "
    "get restaurant details, and check reservation availability."
)

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("ANTHROPIC_API_KEY is not set. Add it to .env locally or to Render's Environment tab.")
    st.stop()

# Full agent history (user, assistant, tool calls and tool results) so follow-ups keep context.
if "history" not in st.session_state:
    st.session_state.history = []
# What the user sees: (role, text) pairs.
if "display" not in st.session_state:
    st.session_state.display = []

with st.sidebar:
    st.subheader("Try asking")
    st.markdown(
        "- I want to eat Indian food, what are some good restaurants?\n"
        "- Can I make a reservation for 4 people at Spice Palace on 2024-11-15?\n"
        "- What Chinese restaurants do you recommend?\n"
        "- How about Golden Wok? Tell me more details."
    )
    if st.button("Clear chat"):
        st.session_state.history = []
        st.session_state.display = []
        st.rerun()

for role, text in st.session_state.display:
    with st.chat_message(role):
        st.markdown(text)

if prompt := st.chat_input("Ask about restaurants..."):
    st.session_state.display.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = build_agent().invoke(
                    {"messages": st.session_state.history + [{"role": "user", "content": prompt}]}
                )
                st.session_state.history = result["messages"]
                reply = message_text(result["messages"][-1]) or "Sorry, I couldn't come up with an answer."
            except Exception as exc:  # show the error in the chat instead of crashing the app
                reply = f"Sorry, something went wrong: {exc}"
        st.markdown(reply)
    st.session_state.display.append(("assistant", reply))
