"""A Model Context Protocol (MCP) server for mathematical operations.

This module provides a debuggable MCP server with tools for calculating 
averages and applying discounts, complete with execution logging.
"""

import logging
import sys
from typing import Sequence

from mcp.server import MCPServer

# Configure logging at the module level.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)

logger = logging.getLogger(__name__)

# The server instance.
mcp = MCPServer("Debuggable Server")


@mcp.tool()
def calculate_average(numbers: Sequence[float]) -> float:
    """Calculates the average of a sequence of numbers.

    Args:
        numbers: A sequence of numerical values (e.g., list of floats).

    Returns:
        The calculated average of the provided numbers.

    Raises:
        ValueError: If the sequence of numbers is empty.
    """
    # Use lazy string formatting for logging.
    logger.info("Calculating average for numbers: %s", numbers)
    
    if not numbers:
        logger.warning("Empty list provided to calculate_average.")
        raise ValueError("The sequence of numbers cannot be empty.")

    average = sum(numbers) / len(numbers)
    
    logger.info("Calculated average: %f", average)
    return average


@mcp.tool()
def apply_discount(price: float, discount_percentage: float) -> float:
    """Applies a percentage discount to a price.

    Args:
        price: The original price. Must be non-negative.
        discount_percentage: The discount to apply, expressed as a percentage
            (e.g., 20.0 for 20%). Must be non-negative.

    Returns:
        The final price after the discount is applied.

    Raises:
        ValueError: If either the price or the discount percentage is negative.
    """
    logger.info(
        "Applying discount: %f%% to price: %f", discount_percentage, price
    )
    
    if price < 0 or discount_percentage < 0:
        logger.error("Price and discount percentage must be non-negative.")
        raise ValueError("Price and discount percentage must be non-negative.")

    discounted_price = price * (1 - discount_percentage / 100.0)
    
    logger.info("Discounted price: %f", discounted_price)
    return discounted_price


if __name__ == "__main__":
    mcp.run()