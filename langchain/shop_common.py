from __future__ import annotations
from functools import lru_cache
import numpy as np

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model

def embed_text(text: str) -> np.ndarray:
    model = get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embedding, dtype=np.float32)

FAQ_DOCUMENTS = [
    {
        "id": "faq-001",
        "category": "returns_refunds",
        "text": "You can return any item within 30 days of delivery for a full refund, as long as you have your order confirmation.",
    },
    {
        "id": "faq-002",
        "category": "returns_refunds",
        "text": "Items returned between 31 and 45 days after delivery are only eligible for store credit, and must be unopened and in original packaging.",
    },
    {
        "id": "faq-003",
        "category": "returns_refunds",
        "text": "After 45 days we cannot accept returns, except for items that arrive damaged or defective, which can be reported within 90 days of delivery.",
    },
    {
        "id": "faq-004",
        "category": "returns_refunds",
        "text": "Opened electronics like headphones, blenders, and kitchen appliances can only be returned within 30 days, and only if the item is actually defective.",
    },
    {
        "id": "faq-005",
        "category": "returns_refunds",
        "text": "Clearance and final sale items cannot be returned or refunded under any circumstances.",
    },
    {
        "id": "faq-006",
        "category": "shipping_delivery",
        "text": "Standard shipping typically takes 3 to 5 business days after your order is confirmed.",
    },
    {
        "id": "faq-007",
        "category": "shipping_delivery",
        "text": "You can track your package in real time using the tracking link emailed to you once your order ships.",
    },
    {
        "id": "faq-008",
        "category": "shipping_delivery",
        "text": "If your package shows as delivered but you haven't received it, contact support within 48 hours so we can investigate with the carrier.",
    },
    {
        "id": "faq-009",
        "category": "shipping_delivery",
        "text": "Expedited shipping is available at checkout for an extra fee and typically arrives within 1 to 2 business days.",
    },
    {
        "id": "faq-010",
        "category": "shipping_delivery",
        "text": "We currently ship to over 40 countries; international orders may be subject to customs fees charged by your local authority.",
    },
    {
        "id": "faq-011",
        "category": "product_support",
        "text": "If a product arrives damaged or stops working shortly after purchase, contact support with your order number and a photo of the issue.",
    },
    {
        "id": "faq-012",
        "category": "product_support",
        "text": "Most electronics come with a 1-year manufacturer's warranty covering defects in materials and workmanship.",
    },
    {
        "id": "faq-013",
        "category": "product_support",
        "text": "Assembly instructions for furniture items are included in the box and also available as a PDF download on the product page.",
    },
    {
        "id": "faq-014",
        "category": "product_support",
        "text": "If a product is missing parts, we will ship the missing components at no extra cost within 5 to 7 business days.",
    },
    {
        "id": "faq-015",
        "category": "account_billing",
        "text": "You can update your saved payment method any time from the Account > Payment Methods page.",
    },
    {
        "id": "faq-016",
        "category": "account_billing",
        "text": "Refunds are issued to your original payment method and typically take 5 to 10 business days to appear on your statement.",
    },
    {
        "id": "faq-017",
        "category": "account_billing",
        "text": "If you were charged twice for the same order, contact support with both transaction IDs so we can reverse the duplicate charge.",
    },
    {
        "id": "faq-018",
        "category": "account_billing",
        "text": "You can change the email address on your account from Account > Profile Settings, but this does not change past order confirmations.",
    },
]
