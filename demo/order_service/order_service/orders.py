"""Order creation logic."""

from order_service.users_client import get_confirmation_recipient, get_order_owner_id, get_order_owner_name


def create_order(user_id: int, items: list[dict[str, float]]) -> dict[str, object]:
    return {
        "owner_id": get_order_owner_id(user_id),
        "owner_name": get_order_owner_name(user_id),
        "confirmation_email": get_confirmation_recipient(user_id),
        "items": items,
        "total": sum(item["price"] for item in items),
    }
