import re
import asyncio
from mcp.base import MCPServer, MCPTool
from playwright.async_api import async_playwright


class AmazonMCPServer(MCPServer):

    @property
    def server_name(self) -> str:
        return "amazon"

    def list_tools(self) -> list[MCPTool]:
        return [
            MCPTool(
                name        = "search_products",
                description = "Search Amazon India for products with filters",
                parameters  = {
                    "query":      "string — product search query",
                    "max_price":  "number — maximum price in INR (optional)",
                    "min_rating": "number — minimum star rating 1-5 (optional)",
                    "max_results": "number — how many results to return (default 5)"
                }
            )
        ]

    async def call_tool(self, tool_name: str, args: dict) -> dict:
        if tool_name == "search_products":
            return await self._search(args)
        return {"error": f"unknown tool: {tool_name}"}

    async def _search(self, args: dict) -> dict:
        query       = args.get("query", "")
        max_price   = args.get("max_price")
        min_rating  = args.get("min_rating", 0)
        max_results = int(args.get("max_results", 5))

        print(f"[AMAZON MCP] searching: '{query}' "
              f"max_price={max_price} min_rating={min_rating}")

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page    = await browser.new_page()

                # set realistic headers
                await page.set_extra_http_headers({
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                })

                # build search URL
                url = f"https://www.amazon.in/s?k={query.replace(' ', '+')}"
                if max_price:
                    url += f"&rh=p_36%3A-{int(max_price * 100)}"

                await page.goto(url, timeout=20000,
                                wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)

                # extract product cards
                products = await page.evaluate("""
                () => {
                    const cards = document.querySelectorAll(
                        '[data-component-type="s-search-result"]'
                    );
                    const results = [];
                    cards.forEach(card => {
                        try {
                            const title = card.querySelector(
                                'h2 a span'
                            )?.innerText?.trim() || '';

                            const priceWhole = card.querySelector(
                                '.a-price-whole'
                            )?.innerText?.trim() || '';

                            const rating = card.querySelector(
                                '.a-icon-alt'
                            )?.innerText?.trim() || '';

                            const reviews = card.querySelector(
                                '.a-size-base.s-underline-text'
                            )?.innerText?.trim() || '0';

                            const img = card.querySelector(
                                '.s-image'
                            )?.src || '';

                            const link = card.querySelector(
                                'h2 a'
                            )?.href || '';

                            if (title && priceWhole) {
                                results.push({
                                    title, priceWhole,
                                    rating, reviews, img, link
                                });
                            }
                        } catch(e) {}
                    });
                    return results;
                }
                """)

                await browser.close()

            # clean and parse
            parsed = []
            for p in products:
                price = self._parse_price(p["priceWhole"])
                stars = self._parse_rating(p["rating"])
                count = self._parse_reviews(p["reviews"])

                if max_price and price and price > max_price:
                    continue
                if min_rating and stars < min_rating:
                    continue

                parsed.append({
                    "title":       p["title"][:100],
                    "price_inr":   price,
                    "price_str":   f"₹{price:,}" if price else "Price N/A",
                    "rating":      stars,
                    "review_count": count,
                    "image_url":   p["img"],
                    "link":        p["link"],
                    "score":       self._score(stars, count)
                })

            # sort by score — more reviews weighted higher
            parsed.sort(key=lambda x: x["score"], reverse=True)
            parsed = parsed[:max_results]

            print(f"[AMAZON MCP] found {len(parsed)} products")
            return {
                "status":   "ok",
                "query":    query,
                "count":    len(parsed),
                "products": parsed
            }

        except Exception as e:
            print(f"[AMAZON MCP] error: {e}")
            return {
                "status":   "error",
                "error":    str(e),
                "products": []
            }

    def _parse_price(self, price_str: str) -> int | None:
        try:
            return int(re.sub(r'[^\d]', '', price_str))
        except Exception:
            return None

    def _parse_rating(self, rating_str: str) -> float:
        try:
            return float(rating_str.split()[0])
        except Exception:
            return 0.0

    def _parse_reviews(self, review_str: str) -> int:
        try:
            clean = re.sub(r'[^\d]', '', review_str)
            return int(clean) if clean else 0
        except Exception:
            return 0

    def _score(self, rating: float, review_count: int) -> float:
        """
        Weighted score — high review count matters more than raw rating.
        A 4.3 star product with 5000 reviews beats
        a 4.8 star product with 10 reviews.
        """
        if review_count == 0:
            return 0.0
        import math
        # Wilson score lower bound approximation
        # gives confidence-adjusted rating
        n = review_count
        p = rating / 5.0
        z = 1.96  # 95% confidence
        score = (
            (p + z*z/(2*n) - z * math.sqrt((p*(1-p)+z*z/(4*n))/n))
            / (1 + z*z/n)
        )
        return score