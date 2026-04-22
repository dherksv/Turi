import re
import math
import asyncio
from mcp.base import MCPServer, MCPTool
from debug_logger import log_event, log_error, log_tool_call


class AmazonMCPServer(MCPServer):

    @property
    def server_name(self) -> str:
        return "amazon"

    def list_tools(self) -> list[MCPTool]:
        return [
            MCPTool(
                name        = "search_products",
                description = "Search Amazon India",
                parameters  = {
                    "query":       "string",
                    "max_price":   "number",
                    "min_rating":  "number",
                    "max_results": "number"
                }
            )
        ]

    async def call_tool(
        self, tool_name: str, args: dict
    ) -> dict:
        if tool_name == "search_products":
            return await self._search(args)
        return {"error": f"unknown tool: {tool_name}"}

    async def _search(self, args: dict) -> dict:
        query       = args.get("query", "")
        max_price   = args.get("max_price")
        min_rating  = float(args.get("min_rating", 0))
        max_results = int(args.get("max_results", 5))

        print(f"[AMAZON MCP] searching: '{query}' "
              f"max_price={max_price}")

        log_event("amazon_search_start", "amazon_mcp", {
            "query": query,
            "max_price": max_price,
            "min_rating": min_rating
        })

        try:
            import time
            start  = time.time()
            result = await asyncio.to_thread(
                self._search_sync,
                query, max_price, min_rating, max_results
            )
            elapsed = int((time.time() - start) * 1000)

            log_tool_call(
                tool        = "amazon_search",
                args        = args,
                result      = result,
                agent       = "amazon_mcp",
                duration_ms = elapsed
            )
            return result

        except Exception as e:
            log_error("amazon_mcp", e, {"query": query})
            print(f"[AMAZON MCP] outer error: "
                  f"{type(e).__name__}: {e}")
            return {
                "status":   "error",
                "error":    f"{type(e).__name__}: {str(e)}",
                "products": []
            }

    def _search_sync(
        self,
        query:       str,
        max_price,
        min_rating:  float,
        max_results: int
    ) -> dict:
        import traceback

        print(f"[AMAZON SYNC] starting playwright for '{query}'")
        log_event("amazon_playwright_start", "amazon_mcp",
                  {"query": query})

        try:
            from playwright.sync_api import sync_playwright

            url = (f"https://www.amazon.in/s"
                   f"?k={query.replace(' ', '+')}")

            print(f"[AMAZON SYNC] url: {url}")

            with sync_playwright() as p:
                print("[AMAZON SYNC] launching browser...")
                browser = p.chromium.launch(
                    headless = True,
                    args     = ["--no-sandbox",
                                "--disable-dev-shm-usage"]
                )
                page = browser.new_page()
                page.set_extra_http_headers({
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                        " AppleWebKit/537.36 Chrome/120.0.0.0"
                        " Safari/537.36"
                    )
                })
                print(f"[AMAZON SYNC] navigating...")
                page.goto(
                    url,
                    timeout    = 30000,
                    wait_until = "domcontentloaded"
                )
                page.wait_for_timeout(2500)

                print("[AMAZON SYNC] extracting products...")
                products = page.evaluate("""
                () => {
                  const cards = document.querySelectorAll(
                    '[data-component-type="s-search-result"]'
                  );
                  const out = [];
                  cards.forEach(card => {
                    try {
                      const title = card.querySelector(
                        'h2 a span'
                      )?.innerText?.trim() || '';
                      const price = card.querySelector(
                        '.a-price-whole'
                      )?.innerText?.trim() || '';
                      const rating = card.querySelector(
                        '.a-icon-alt'
                      )?.innerText?.trim() || '';
                      const reviews = card.querySelector(
                        '[aria-label*="ratings"],' +
                        '.a-size-base.s-underline-text'
                      )?.innerText?.trim() || '0';
                      const img = card.querySelector(
                        '.s-image'
                      )?.src || '';
                      const link = 'https://amazon.in' +
                        (card.querySelector('h2 a')
                         ?.getAttribute('href') || '');
                      if (title && price) {
                        out.push({
                          title, price, rating,
                          reviews, img, link
                        });
                      }
                    } catch(e) {}
                  });
                  return out;
                }
                """)

                browser.close()

            print(f"[AMAZON SYNC] raw products: "
                  f"{len(products)}")

            if not products:
                log_event("amazon_no_products", "amazon_mcp", {
                    "query": query,
                    "url":   url
                })

            parsed = []
            for p in products:
                price  = self._parse_price(p["price"])
                stars  = self._parse_rating(p["rating"])
                count  = self._parse_reviews(p["reviews"])

                if max_price and price and price > max_price:
                    continue
                if min_rating and stars and stars < min_rating:
                    continue

                parsed.append({
                    "title":        p["title"][:100],
                    "price_inr":    price,
                    "price_str":    (f"₹{price:,}"
                                     if price else "N/A"),
                    "rating":       stars,
                    "review_count": count,
                    "image_url":    p["img"],
                    "link":         p["link"],
                    "score":        self._score(stars, count)
                })

            parsed.sort(
                key     = lambda x: x["score"],
                reverse = True
            )
            parsed = parsed[:max_results]

            print(f"[AMAZON SYNC] returning "
                  f"{len(parsed)} products")

            log_event("amazon_search_done", "amazon_mcp", {
                "query":   query,
                "found":   len(parsed),
                "sample":  [p["title"][:40]
                            for p in parsed[:3]]
            })

            return {
                "status":   "ok",
                "query":    query,
                "count":    len(parsed),
                "products": parsed
            }

        except Exception as e:
            tb = traceback.format_exc()
            print(f"[AMAZON SYNC] error: "
                  f"{type(e).__name__}: {e}")
            print(f"[AMAZON SYNC] traceback:\n{tb}")
            log_error("amazon_mcp", e, {
                "query": query,
                "traceback": tb
            })
            return {
                "status":   "error",
                "error":    f"{type(e).__name__}: {str(e)}",
                "products": [],
                "traceback": tb[:500]
            }

    def _parse_price(self, s: str):
        try:
            return int(re.sub(r'[^\d]', '', s))
        except Exception:
            return None

    def _parse_rating(self, s: str) -> float:
        try:
            return float(s.split()[0])
        except Exception:
            return 0.0

    def _parse_reviews(self, s: str) -> int:
        try:
            clean = re.sub(r'[^\d]', '', s)
            return int(clean) if clean else 0
        except Exception:
            return 0

    def _score(self, rating: float, reviews: int) -> float:
        if not rating or reviews == 0:
            return 0.0
        n = reviews
        p = rating / 5.0
        z = 1.96
        return (
            (p + z*z/(2*n)
             - z * math.sqrt(
                 (p*(1-p) + z*z/(4*n)) / n
             ))
            / (1 + z*z/n)
        )